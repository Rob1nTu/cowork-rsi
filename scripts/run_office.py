"""Pinned mini-swe-agent runner with explicit split guards and isolated Office workspaces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import litellm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/mini-swe-agent/src'))
os.environ['MSWEA_SILENT_STARTUP'] = '1'
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'true'

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.extra.bubblewrap import BubblewrapEnvironment
from minisweagent.models.litellm_model import LitellmModel

from context_manager import ContextWindowManager


class ManagedLitellmModel(LitellmModel):
    def __init__(self, *, context_manager: ContextWindowManager, **kwargs):
        self.context_manager = context_manager
        super().__init__(**kwargs)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        managed = self.context_manager.prepare(messages)
        try:
            return super().query(managed, **kwargs)
        except litellm.exceptions.ContextWindowExceededError:
            return super().query(self.context_manager.prepare(messages, force=True), **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        for message in prepared:
            if message.get("reasoning_content") and not message.get("reasoning"):
                message["reasoning"] = message.pop("reasoning_content")
        return prepared

    def serialize(self) -> dict:
        data = super().serialize()
        data["info"]["context_management"] = self.context_manager.serialize()
        return data

BASE_PYTHON = '/data/L202500306/L202500306/tlb/conda-envs/vllm025'
AGENT_PYTHON = '/tmp/cowork-agent-env'
CLEAN_ENV = {'PATH': '/usr/bin:/bin', 'HOME': '/root', 'LANG': 'C.UTF-8'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sandbox_args(workspace: Path, *, tests: Path | None = None, logs: Path | None = None) -> list[str]:
    args = ['--unshare-user', '--unshare-pid', '--unshare-net', '--die-with-parent',
            '--new-session', '--tmpfs', '/tmp']
    for source in ('/usr', '/bin', '/lib', '/lib64', BASE_PYTHON, AGENT_PYTHON):
        args += ['--ro-bind', source, source]
    args += ['--dir', '/proc', '--dev', '/dev', '--dir', '/root',
             '--setenv', 'HOME', '/root', '--setenv', 'LANG', 'C.UTF-8',
             '--setenv', 'PATH', AGENT_PYTHON + '/bin:/usr/bin:/bin',
             '--bind' if tests is None else '--ro-bind', str(workspace), '/workspace',
             '--chdir', '/workspace']
    if tests is not None:
        args += ['--ro-bind', str(tests), '/tests', '--bind', str(logs), '/logs/verifier',
                 '--setenv', 'WB_BENCH_GOLD_PATH', '/tests/gold/gold_answer.json',
                 '--setenv', 'WB_BENCH_FIXTURES_DIR', '/tests/gold/fixtures',
                 '--setenv', 'PYTHONPATH', '/workspace']
    return args


class OfficeEnvironment(BubblewrapEnvironment):
    def __init__(self, workspace: Path, command_log: Path):
        self.command_log = command_log
        self.commands = 0
        super().__init__(cwd='/workspace', executable=str(ROOT / 'tools/usr/bin/bwrap'),
                         wrapper_args=sandbox_args(workspace), timeout=120)

    def execute(self, action: dict, cwd: str = '', *, timeout: int | None = None) -> dict:
        # The upstream environment normally adds a host cwd bind; our wrapper already maps /workspace.
        command = action.get('command', '')
        started = time.monotonic()
        result = subprocess.run([self.config.executable, *self.config.wrapper_args,
                                 'bash', '-c', command], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', env=CLEAN_ENV, timeout=timeout or self.config.timeout)
        output = {'output': result.stdout + result.stderr, 'returncode': result.returncode,
                  'exception_info': ''}
        self.commands += 1
        with self.command_log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'index': self.commands, 'command': command,
                                     'returncode': result.returncode,
                                     'seconds': time.monotonic() - started}, ensure_ascii=False) + '\n')
        print(json.dumps({'command': self.commands, 'returncode': result.returncode,
                          'seconds': round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
        self._check_finished(output)
        return output


OBSERVATION = '''<returncode>{{output.returncode}}</returncode>
<output>{{output.output[:14000]}}{% if output.output|length > 14000 %}\n[truncated; inspect selected ranges]{% endif %}</output>'''
SYSTEM = '''You are an assistant that completes professional file and data workflows in /workspace.
Use the bash tool to inspect the provided files, compute results, edit artifacts and verify them.
Inputs are available in /workspace. Use the installed Python libraries when helpful.
Follow the user's requested output locations. Keep all persistent work inside /workspace.
Do not search for hidden grading files or answers. Verify using source inputs and the user request.
To finish, make a bash call with only: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Every response must contain at least one bash tool call. Each command starts in /workspace in a fresh shell.'''


def grade(task_dir: Path, workspace: Path, result_dir: Path) -> dict:
    logs = result_dir / 'verifier'
    logs.mkdir(exist_ok=True)
    args = [str(ROOT / 'tools/usr/bin/bwrap'), *sandbox_args(workspace, tests=task_dir / 'tests', logs=logs),
            AGENT_PYTHON + '/bin/python', '-m', 'pytest', '/tests/grading', '-c', '/dev/null',
            '--confcutdir=/tests', '-p', 'no:cacheprovider', '-v', '--tb=short',
            '--junitxml=/logs/verifier/results.xml']
    completed = subprocess.run(args, capture_output=True, text=True, env=CLEAN_ENV, timeout=1800)
    (logs / 'test_output.txt').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    # Execute the pinned task's official score reducer, outside agent visibility.
    reducer = [str(ROOT / 'tools/usr/bin/bwrap'), *sandbox_args(workspace, tests=task_dir / 'tests', logs=logs),
               AGENT_PYTHON + '/bin/python', '/tests/grading/score_core.py',
               '--pytest-exit', str(completed.returncode)]
    reduced = subprocess.run(reducer, capture_output=True, text=True, env=CLEAN_ENV, timeout=120)
    (logs / 'score_reducer.txt').write_text(reduced.stdout + reduced.stderr, encoding='utf-8')
    if reduced.returncode != 0 or not (logs / 'score.json').exists():
        raise RuntimeError('Official rule score reduction failed; see verifier logs.')
    score = json.loads((logs / 'score.json').read_text())
    return {'metric': 'official_rule_component_only', 'score': score,
            'pytest_exit': completed.returncode,
            'limitation': 'No LLM component; not official composite score or full benchmark result.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--memory', type=Path)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--context-limit', type=int, default=24576)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--grade-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'splits/office-v1.json').read_text(encoding='utf-8'))
    allowed = {r['id']: r for r in manifest['tasks']['evolution']}
    if args.task not in allowed:
        raise SystemExit('Pilot runner only permits evolution tasks. Selection and sealed tasks are locked.')
    if Path(args.run).name != args.run or not args.run:
        raise SystemExit('Run must be a single directory name.')
    task_dir = ROOT / 'datasets/wb-bench-office-v1.0/tasks' / args.task
    assert digest(task_dir / 'environment/workspace.tar.gz') == allowed[args.task]['workspace_sha256']
    result_dir = ROOT / 'runs' / args.run / args.task
    workspace = result_dir / 'workspace'
    if not args.grade_only:
        if result_dir.exists():
            raise SystemExit('Run already exists; use a fresh run label or --grade-only.')
        workspace.mkdir(parents=True)
        with tarfile.open(task_dir / 'environment/workspace.tar.gz') as archive:
            archive.extractall(workspace, filter='data')
        env = OfficeEnvironment(workspace, result_dir / 'commands.jsonl')
        # Fail closed: test and project paths must be invisible before the model sees any task.
        probe = env.execute({'command': f'test ! -e /tests && test ! -e "{ROOT}" && python -c "import openpyxl,pandas,docx; print(\'ISOLATION_OK\')"'})
        dump(result_dir / 'isolation.json', probe)
        if probe['returncode'] != 0 or 'ISOLATION_OK' not in probe['output']:
            raise SystemExit('Sandbox preflight failed.')
        context_manager = ContextWindowManager(
            context_limit=args.context_limit,
            max_output_tokens=args.max_tokens,
            workspace=workspace,
            audit_path=result_dir / 'context-compactions.jsonl',
        )
        model = ManagedLitellmModel(
            context_manager=context_manager,
            model_name='openai/qwen27b', cost_tracking='ignore_errors',
            model_kwargs={'api_base': 'http://127.0.0.1:8000/v1', 'api_key': 'local',
                          'temperature': 0.0, 'seed': 20261112, 'max_tokens': args.max_tokens,
                          'timeout': 180, 'num_retries': 2,
                          'extra_body': {'chat_template_kwargs': {'enable_thinking': not args.no_thinking}}},
            observation_template=OBSERVATION,
        )
        task = (task_dir / 'instruction.md').read_text(encoding='utf-8')
        if args.memory:
            memory = json.loads(args.memory.read_text(encoding='utf-8'))
            assert memory['source_split'] == 'evolution'
            task += '\n\nPreviously learned procedural guidance (check applicability):\n' + memory['guidance']
        agent = DefaultAgent(model, env, system_template=SYSTEM, instance_template='{{task}}',
                             step_limit=args.steps, cost_limit=0, wall_time_limit_seconds=1800,
                             output_path=result_dir / 'trajectory.json')
        started = time.monotonic()
        try:
            exit_info = agent.run(task)
        finally:
            agent.save(result_dir / 'trajectory.json')
            dump(result_dir / 'execution.json', {'task': args.task, 'split': 'evolution',
                 'run': args.run, 'seconds': time.monotonic() - started,
                 'calls': agent.n_calls, 'commands': env.commands,
                 'model': 'qwen27b', 'seed': 20261112, 'runner_sha256': digest(Path(__file__)), 'steps': args.steps,
                                   'memory_sha256': digest(args.memory) if args.memory else None,
                  'context_limit': args.context_limit, 'max_tokens': args.max_tokens,
                  'thinking': not args.no_thinking,

                 'split_manifest_sha256': digest(ROOT / 'splits/office-v1.json'),
                 'upstream_commits': manifest['upstream_commits']})
        print('EXIT', exit_info, flush=True)
    result = grade(task_dir, workspace, result_dir)
    dump(result_dir / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
