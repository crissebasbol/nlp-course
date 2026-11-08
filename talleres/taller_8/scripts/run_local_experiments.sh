#!/usr/bin/env bash
# Full experiment run on the Mac (no Docker, uses the GPU through MPS/Metal).
# BERT and the LLM do not fit together in GPU memory, so they run one after the other:
#   1-2 data  ->  3a BERT train + scores  ->  start llama-server  ->  3b DSPy  ->  3c combine
#   ->  4-5 correct  ->  6 evaluate  ->  stop llama-server
# Every step skips work already saved in outputs/, so the script can be re-run after a failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GGUF=outputs/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLM_PORT=8001

step() { echo; echo "=== $* ==="; }

step "1-2. Datos"
[ -f outputs/splits/test.parquet ] || $PY scripts/run_prepare_data.py
$PY scripts/run_baselines.py

step "3a. BiomedBERT"
[ -d outputs/models/bert/final ] || $PY scripts/run_train_bert.py
[ -f outputs/scores/bert_test.jsonl ] || $PY scripts/run_predict_bert.py

step "LLM: llama-server con Qwen2.5-7B (Metal)"
if [ ! -f "$GGUF" ]; then
  .venv/bin/hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir outputs/models/gguf
fi
STARTED_LLM=0
if ! curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null; then
  llama-server -m "$GGUF" --alias qwen2.5-7b-instruct --host 127.0.0.1 --port $LLM_PORT \
    -c 24576 --parallel 4 -ngl 99 > outputs/llama_server_local.log 2>&1 &
  LLM_PID=$!
  STARTED_LLM=1
  trap '[ $STARTED_LLM = 1 ] && kill $LLM_PID 2>/dev/null || true' EXIT
fi

step "3b. Qwen + DSPy (compilado y sin compilar)"
$PY scripts/run_dspy_detector.py
$PY scripts/run_dspy_detector.py --no-compile

step "3c. Combinación"
$PY scripts/run_combine.py

step "4-5. Corrección y control de calidad"
$PY scripts/run_correct.py

step "6. Evaluación"
$PY scripts/run_evaluate.py
