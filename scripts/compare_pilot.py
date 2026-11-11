"""Compare matched pilot executions and gate a versioned procedural-memory candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def usage(path: Path) -> dict:
    messages = load(path)['messages']
    values = [(m.get('extra') or {}).get('response', {}).get('usage') or {} for m in messages
              if isinstance((m.get('extra') or {}).get('response', {}), dict)]
    return {k: sum(v.get(k, 0) or 0 for v in values)
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate-run', required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = load(ROOT / 'splits/office-v1.json')
    allowed = {r['id'] for r in manifest['tasks']['evolution']}
    if not set(args.tasks) <= allowed or len(set(args.tasks)) != len(args.tasks):
        raise SystemExit('Pilot comparison permits distinct evolution tasks only.')
    memory = load(args.memory)
    memory_hash = hashlib.sha256(args.memory.read_bytes()).hexdigest()
    rows = []
    for task in args.tasks:
        row = {'task': task, 'is_source_task': task in memory['source_tasks']}
        runs = {}
        for label, run in [('baseline', args.baseline), ('candidate', args.candidate_run)]:
            directory = ROOT / 'runs' / run / task
            execution = load(directory / 'execution.json')
            evaluation = load(directory / 'evaluation.json')
            if evaluation['metric'] != 'official_rule_component_only':
                raise SystemExit('Metric mismatch.')
            if evaluation['score']['test_status'] == 'build_error':
                raise SystemExit('Infrastructure failure cannot be treated as a task score.')
            runs[label] = execution
            row[label] = {'rule_pass_rate': evaluation['score']['test_pass_rate'],
                          'passed': evaluation['score']['tests_passed'],
                          'total': evaluation['score']['tests_total'],
                          'seconds': execution['seconds'], 'calls': execution['calls'],
                          'usage': usage(directory / 'trajectory.json')}
        for key in ('model', 'seed', 'steps', 'split_manifest_sha256', 'upstream_commits', 'runner_sha256'):
            if runs['baseline'][key] != runs['candidate'][key]:
                raise SystemExit('Unmatched run configuration: ' + key)
        if runs['baseline']['memory_sha256'] is not None or runs['candidate']['memory_sha256'] != memory_hash:
            raise SystemExit('Memory identity mismatch.')
        row['delta'] = row['candidate']['rule_pass_rate'] - row['baseline']['rule_pass_rate']
        rows.append(row)
    transfer = [r for r in rows if not r['is_source_task']]
    positive_gain = sum(r['delta'] for r in rows) > 1e-8
    no_regression = all(r['delta'] >= -1e-8 for r in rows)
    transfer_positive = bool(transfer) and sum(r['delta'] for r in transfer) > 1e-8
    promoted = positive_gain and no_regression and transfer_positive
    result = {'metric': 'official_rule_component_only', 'tasks': rows,
              'mean_baseline': sum(r['baseline']['rule_pass_rate'] for r in rows) / len(rows),
              'mean_candidate': sum(r['candidate']['rule_pass_rate'] for r in rows) / len(rows),
              'candidate_memory_sha256': memory_hash,
              'generation_usage': memory['generation_usage'],
              'generation_seconds': memory['generation_seconds'],
              'gate': {'positive_gain': positive_gain, 'no_per_task_regression': no_regression,
                       'positive_gain_on_other_evolution_tasks': transfer_positive,
                       'decision': 'provisionally_accept_on_evolution' if promoted else 'reject_or_inconclusive'},
              'limitations': ['Small evolution pilot; no selection or sealed results.',
                              'Rule component excludes any official LLM judging.',
                              'Temperature-zero paired runs do not estimate sampling uncertainty.',
                              'Generation overhead is reported separately; this is not yet a matched total-budget A5 search comparison.']}
    if args.output.exists():
        raise SystemExit('Comparison version exists; refusing overwrite.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
