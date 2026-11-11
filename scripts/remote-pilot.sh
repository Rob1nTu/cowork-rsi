#!/usr/bin/env bash
set -euo pipefail
PERSIST=/data/L202500306/L202500306/tlb/cowork-rsi
ROOT=/tmp/cowork-pilot-project
PYTHON=/tmp/cowork-agent-env/bin/python
mkdir -p "$ROOT" "$PERSIST/pilot-checkpoints"
# Source snapshots are immutable inputs; transient files use local storage for speed.
if [ ! -f "$ROOT/splits/office-v1.json" ]; then
  tar --no-same-owner -xzf /tmp/cowork-baseline-bundle.tar.gz -C "$ROOT"
fi
mkdir -p "$ROOT/tools/usr/bin"
cp "$PERSIST/tools/usr/bin/bwrap" "$ROOT/tools/usr/bin/bwrap"
cp "$PERSIST/scripts/"{run_office.py,learn_memory.py,compare_pilot.py,preflight.py} "$ROOT/scripts/"
"$PYTHON" -m pip install --disable-pip-version-check --timeout 30 --retries 5 --proxy http://127.0.0.1:17897 'tenacity==9.1.2'
"$PYTHON" -m pip freeze > "$PERSIST/agent-requirements.lock.txt"
cd "$ROOT"
checkpoint() {
  local stage="$1"
  local parts=()
  for part in runs memory results diagnostics; do
    if [ -d "$part" ]; then parts+=("$part"); fi
  done
  if [ "${#parts[@]}" -gt 0 ]; then
    tar -czf "$PERSIST/pilot-checkpoints/$stage.tar.gz" "${parts[@]}"
  fi
}
trap 'checkpoint interrupted' EXIT
"$PYTHON" -u scripts/preflight.py
for task in contract-extract-L3-014 procurement-reconcile-L3-015; do
  "$PYTHON" -u scripts/run_office.py --task "$task" --run baseline-v1
  checkpoint "baseline-$task"
done
"$PYTHON" -u scripts/learn_memory.py --task contract-extract-L3-014 --source-run baseline-v1 --output procedural-v1
checkpoint learned-v1
for task in contract-extract-L3-014 procurement-reconcile-L3-015; do
  "$PYTHON" -u scripts/run_office.py --task "$task" --run memory-v1 --memory memory/procedural-v1.json
  checkpoint "memory-$task"
done
"$PYTHON" -u scripts/compare_pilot.py --baseline baseline-v1 --candidate-run memory-v1 --memory memory/procedural-v1.json --tasks contract-extract-L3-014 procurement-reconcile-L3-015 --output results/pilot-v1.json
checkpoint pilot-v1-complete
trap - EXIT
printf 'PILOT_COMPLETE\n'
