"""Freeze source-grouped splits before any task execution or grader inspection."""
import hashlib
import json
import random
import subprocess
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets/wb-bench-office-v1.0/tasks'
SEED = 20261112
REVISION = '3e8ad839685c589d83a1dd60fe2d15eabb75f27a'
ARCHIVE_HASH = '89d4480b483ca2e0555ba96cc7d73c16c78802b17cd822044d5f7f27ab0558bf'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / 'splits/office-v1.json'
    if output.exists():
        raise SystemExit('Split already frozen; refusing to overwrite.')
    assert sha(ROOT / 'wb-bench-office-v1.0.tar.gz') == ARCHIVE_HASH
    rows = []
    parent = {}
    fingerprints = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for path in sorted(DATASET.glob('*/task.toml')):
        meta = tomllib.loads(path.read_text(encoding='utf-8'))['metadata']
        task = path.parent.name
        find(task)
        row = {'id': task, **meta,
               'workspace_sha256': sha(path.parent / 'environment/workspace.tar.gz'),
               'instruction_sha256': sha(path.parent / 'instruction.md')}
        rows.append(row)
        for key in ('source_case', 'workspace_sha256', 'instruction_sha256'):
            fingerprint = (key, row[key])
            if fingerprint in fingerprints:
                parent[find(task)] = find(fingerprints[fingerprint])
            fingerprints[fingerprint] = task
    groups = defaultdict(list)
    for row in rows:
        row['group'] = find(row['id'])
        groups[row['group']].append(row)
    assert len(rows) == 50
    # Do not split grouped variants. Current pinned Office release has 50 distinct groups.
    assert len(groups) == 50, 'Grouped variants found; adapt capacity allocation without splitting groups.'
    strata = Counter((r['category'], r['difficulty']) for r in rows)
    targets = {'evolution': 30, 'selection': 10, 'sealed': 10}
    rng = random.Random(SEED)
    best = None
    for _ in range(5000):
        candidate = rows.copy()
        rng.shuffle(candidate)
        partitions, start, loss = {}, 0, 0.0
        for name, count in targets.items():
            subset = candidate[start:start + count]
            partitions[name] = subset
            actual = Counter((r['category'], r['difficulty']) for r in subset)
            loss += sum((actual[s] - n * count / 50) ** 2 for s, n in strata.items())
            start += count
        if best is None or loss < best[0]:
            best = (loss, partitions)
    partitions = best[1]
    all_ids = [r['id'] for rs in partitions.values() for r in rs]
    assert len(all_ids) == len(set(all_ids)) == 50
    source_revisions = {}
    for name in ('workbuddy-bench', 'mini-swe-agent'):
        source_revisions[name] = subprocess.check_output(
            ['git', '-C', str(ROOT / 'upstream' / name), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'dataset': 'Tencent WorkBuddy Bench Office v1.0', 'hf_revision': REVISION,
              'archive_sha256': ARCHIVE_HASH, 'seed': SEED,
              'grouping': ['source_case', 'exact workspace archive SHA256', 'exact instruction SHA256'],
              'limitations': 'No semantic-equivalence guarantee. Public benchmark may be in model pretraining. All future variants inherit parent split.',
              'upstream_commits': source_revisions, 'counts': targets,
              'protocol': {'evolution': 'Feedback, memory writes and candidate development allowed.',
                           'selection': 'Aggregate candidate selection only; no test-specific memory or task feedback to optimizer.',
                           'sealed': 'Never executed during development. Freeze model, harness and memory first; evaluate once with memory read-only and per-task state reset.'},
              'tasks': {name: sorted(rs, key=lambda r: r['id']) for name, rs in partitions.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, rs in partitions.items():
        print(name, len(rs), dict(Counter((r['category'], r['difficulty']) for r in rs)))
    print('manifest_sha256', sha(output))


if __name__ == '__main__':
    main()
