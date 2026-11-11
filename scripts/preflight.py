"""Real namespace and native tool-call integration checks, without benchmark execution."""
from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

from run_office import ROOT, OfficeEnvironment, LitellmModel, dump


def main() -> None:
    result_dir = ROOT / 'diagnostics'
    result_dir.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='cowork-preflight-'))
    env = OfficeEnvironment(workspace, result_dir / 'preflight-commands.jsonl')
    code = '''import errno, pathlib, socket
for path in ['/tests', '/proc/1/root', '/root/.ssh', 'PROJECT_ROOT']:
    assert not pathlib.Path(path).exists(), path
try:
    socket.create_connection(('127.0.0.1', 8000), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('Host API reachable from task sandbox')
try:
    pathlib.Path('/tmp/cowork-agent-env/SHOULD_NOT_WRITE').write_text('bad')
except OSError as error:
    assert error.errno in (errno.EROFS, errno.EACCES), repr(error)
else:
    raise AssertionError('Runtime mount is writable')
import openpyxl,pandas,docx,pypdf
pathlib.Path('/workspace/checkpoint.txt').write_text('persisted')
print('ISOLATION_RUNTIME_OK')'''.replace('PROJECT_ROOT', str(ROOT))
    first = env.execute({'command': 'python -c ' + shlex.quote(code)})
    assert first['returncode'] == 0 and 'ISOLATION_RUNTIME_OK' in first['output'], first
    second = env.execute({'command': 'test "$(cat checkpoint.txt)" = persisted && echo WORKSPACE_PERSISTENCE_OK'})
    assert second['returncode'] == 0 and 'WORKSPACE_PERSISTENCE_OK' in second['output'], second
    model = LitellmModel(model_name='openai/qwen27b', cost_tracking='ignore_errors',
                         model_kwargs={'api_base': 'http://127.0.0.1:8000/v1', 'api_key': 'local',
                                       'temperature': 0.0, 'max_tokens': 128, 'timeout': 120})
    response = model.query([{'role': 'user', 'content':
                            'Call the bash tool with the exact command echo MSWEA_MODEL_OK. No prose.'}])
    actions = response['extra']['actions']
    assert len(actions) == 1 and actions[0]['command'] == 'echo MSWEA_MODEL_OK', actions
    third = env.execute(actions[0])
    assert third['returncode'] == 0 and third['output'].strip() == 'MSWEA_MODEL_OK', third
    dump(result_dir / 'preflight.json', {'namespace': first, 'persistence': second,
                                        'native_toolcall': response, 'tool_execution': third})
    print('PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
