"""Fetch pinned public inputs; preserve the committed split rather than regenerating it."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'mini-swe-agent': 'https://github.com/SWE-agent/mini-swe-agent.git',
           'workbuddy-bench': 'https://github.com/Tencent/workbuddy-bench.git'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'splits/office-v1.json').read_text(encoding='utf-8'))
    archive = ROOT / 'wb-bench-office-v1.0.tar.gz'
    if not archive.exists():
        curl = shutil.which('curl.exe') or shutil.which('curl')
        if not curl:
            raise SystemExit('curl is required to fetch the pinned dataset archive.')
        with tempfile.NamedTemporaryFile(prefix='office-', suffix='.partial', dir=ROOT, delete=False) as file:
            partial = Path(file.name)
        url = ('https://huggingface.co/datasets/tencent/workbuddy-bench/resolve/'
               + manifest['hf_revision'] + '/wb-bench-office-v1.0.tar.gz')
        subprocess.run([curl, '--fail', '--location', '--retry', '3', '--connect-timeout', '15',
                        '--max-time', '180', '--output', str(partial), url], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != manifest['archive_sha256']:
            raise SystemExit('Downloaded archive checksum mismatch; partial file retained for diagnosis.')
        assert archive.resolve().parent == ROOT.resolve() and not archive.exists()
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise SystemExit('Existing archive checksum mismatch; refusing extraction.')
    dataset_dir = ROOT / 'datasets'
    dataset_dir.mkdir(exist_ok=True)
    expected = {row['id']: row for rows in manifest['tasks'].values() for row in rows}
    dataset = dataset_dir / 'wb-bench-office-v1.0/tasks'
    if set(p.parent.name for p in dataset.glob('*/task.toml')) != set(expected):
        with tarfile.open(archive) as source:
            source.extractall(dataset_dir, filter='data')
    for task, row in expected.items():
        for path, key in [(dataset / task / 'environment/workspace.tar.gz', 'workspace_sha256'),
                          (dataset / task / 'instruction.md', 'instruction_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                raise SystemExit('Pinned dataset task fingerprint mismatch: ' + task)
    print('VERIFIED_DATASET_TASKS', len(expected), flush=True)
    if args.dataset_only:
        return
    for name, url in SOURCES.items():
        destination = ROOT / 'upstream' / name
        pin = manifest['upstream_commits'][name]
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            subprocess.run(['git', 'init', str(destination)], check=True)
            subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', url], check=True)
        if not (destination / '.git').exists():
            raise SystemExit('Existing source snapshot has no Git metadata: ' + name)
        current = subprocess.run(['git', '-C', str(destination), 'rev-parse', '--verify', 'HEAD'],
                                  capture_output=True, text=True)
        if current.returncode == 0:
            if current.stdout.strip() != pin:
                raise SystemExit('Existing upstream is at another commit; refusing to change it: ' + name)
            print('VERIFIED_UPSTREAM_COMMIT', name, pin, flush=True)
            continue
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', pin], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != pin:
            raise SystemExit('Fetched source commit mismatch: ' + name)
        print('VERIFIED_UPSTREAM_COMMIT', name, pin, flush=True)


if __name__ == '__main__':
    main()
