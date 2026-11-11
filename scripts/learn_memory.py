"""Generate source-grounded procedural candidates from evolution trajectories only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'splits/office-v1.json').read_text(encoding='utf-8'))
    evolution = {r['id'] for r in manifest['tasks']['evolution']}
    if args.task not in evolution:
        raise SystemExit('Memory generation may consume evolution trajectories only.')
    if Path(args.source_run).name != args.source_run or Path(args.output).name != args.output:
        raise SystemExit('Labels must be single path components.')
    source = ROOT / 'runs' / args.source_run / args.task
    evaluation = json.loads((source / 'evaluation.json').read_text(encoding='utf-8'))
    if evaluation['score']['test_status'] == 'build_error':
        raise SystemExit('Infrastructure failures cannot become procedural memory.')
    trajectory_path = source / 'trajectory.json'
    trajectory = json.loads(trajectory_path.read_text(encoding='utf-8'))
    # No hidden grader, gold, failure values or rubric text are included in the prompt.
    visible = [{'role': m['role'], 'content': m.get('content'),
                'tool_calls': m.get('tool_calls')} for m in trajectory['messages'] if m['role'] != 'exit']
    evidence = json.dumps(visible, ensure_ascii=False)
    if len(evidence) > 18000:
        evidence = evidence[:7000] + '\n[Middle omitted]\n' + evidence[-11000:]
    target = ROOT / 'memory' / (args.output + '.json')
    if target.exists():
        raise SystemExit('Memory version already exists; refusing overwrite.')
    target.parent.mkdir(exist_ok=True)
    schema = {'type': 'object', 'properties': {
        'guidance': {'type': 'string', 'maxLength': 1800},
        'applicability': {'type': 'string', 'maxLength': 400},
        'evidence': {'type': 'string', 'maxLength': 700}},
        'required': ['guidance', 'applicability', 'evidence'], 'additionalProperties': False}
    payload = {'model': 'qwen27b', 'temperature': 0.0, 'seed': 20261112, 'max_tokens': 2400,
               'response_format': {'type': 'json_schema', 'json_schema': {
                   'name': 'procedural_memory', 'strict': True, 'schema': schema}},
               'messages': [{'role': 'system', 'content':
                 'Analyze this agent execution and propose reusable procedural guidance for future professional file tasks. '
                 'Use only the user-visible request and execution evidence supplied. Do not infer hidden answers. '
                 'Return JSON with guidance (a string, at most 1800 characters), applicability (a string), '
                 'and evidence (a short explanation of which observed operations support it). '
                 'Guidance must cover actionable procedure and validation, not source task entities, numeric answers, '
                 'specific record identifiers, fixed output counts or hidden grading tricks. Do not claim a failure '
                 'unless it is directly visible. This is an unvalidated candidate, not an established successful lesson.'},
                {'role': 'user', 'content': evidence}],
               'chat_template_kwargs': {'enable_thinking': False}}
    started = time.monotonic()
    request = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = json.load(response)
    response_path = target.parent / f'{args.output}.{time.time_ns()}.response.json'
    response_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding='utf-8')
    content = raw['choices'][0]['message']['content'] or ''
    if raw['choices'][0].get('finish_reason') == 'length':
        raise SystemExit('Memory output truncated; raw response retained: ' + str(response_path))
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
    candidate = json.loads(text)
    if not isinstance(candidate.get('guidance'), str) or len(candidate['guidance']) > 1800:
        raise SystemExit('Invalid or over-budget procedural candidate.')
    result = {'status': 'candidate_unvalidated', 'source_split': 'evolution',
              'source_tasks': [args.task], 'source_run': args.source_run,
              'source_trajectory_sha256': hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
              'guidance': candidate['guidance'], 'applicability': candidate.get('applicability', ''),
              'evidence': candidate.get('evidence', ''), 'model': 'qwen27b',
              'generation_usage': raw.get('usage', {}), 'generation_seconds': time.monotonic() - started,
              'protocol': 'Generated from request and visible trace only. Requires measured replay and transfer tests before promotion.'}
    result['generation_response_file'] = response_path.name
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
