#!/usr/bin/env bash
set -euo pipefail
TLB=/data/L202500306/L202500306/tlb
VENV="$TLB/conda-envs/vllm025"
RUN_DIR="$TLB/cowork-rsi/deployment"
mkdir -p "$RUN_DIR"
if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models > "$RUN_DIR/existing-models.json"; then
  echo EXISTING_API_REUSED
  cat "$RUN_DIR/existing-models.json"
  exit 0
fi
if ss -ltn | awk '{print $4}' | grep -Eq ':8000$'; then
  echo 'Port 8000 is occupied without a healthy models endpoint; refusing to replace it.' >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -q .; then
  echo 'Compute processes detected. Inspect before launching; no existing process was stopped.' >&2
  exit 3
fi
VLLM_LIB="$VENV/lib/python3.12/site-packages/torch/lib"
for lib in cublas cuda_runtime cudnn cuda_cupti cufft curand cusolver cusparse nvjitlink nvtx; do
  VLLM_LIB="$VLLM_LIB:$VENV/lib/python3.12/site-packages/nvidia/$lib/lib"
done
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_LIBRARY_PATH="$VLLM_LIB" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$TLB/models/Qwen3.8-27B" \
  --served-model-name qwen27b \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 24576 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
  --enforce-eager --language-model-only --skip-mm-profiling \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  > "$RUN_DIR/qwen27b-tp4.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/qwen27b-tp4.pid"
printf 'STARTED_PID=%s\nLOG=%s\n' "$PID" "$RUN_DIR/qwen27b-tp4.log"
