#!/bin/bash
# =====================================================================
# CONDITIONAL FALLBACK — qwen3-32b generation endpoint on the H200.
#
# Status: PARKED. All generation (gpt-oss-120b / mistral-3.2 / qwen3-32b /
# deepseek-v3.1 / o4-mini) runs on OpenRouter. This job is submitted ONLY
# if the qwen3-32b defective-cell re-run still hits the provider gateway
# refusal after OpenRouter provider-pinning (team-lead decision, 2026-05-19).
#
# Why this fixes it: the ~12 degenerate qwen3-32b cells are an identical
# Chinese canned deflection from a provider-side moderation wrapper, not
# the model. Vanilla Qwen3-32B served here has no such gateway.
#
# Consistency: the existing ~28k good qwen3-32b cells are non-thinking
# (0 <think> tags, ~350-token median, no budget clamping). So this endpoint
# MUST serve thinking-OFF — build the no-think chat template first and pass
# it via QWEN3_CHAT_TEMPLATE (see below). max_tokens stays 4096, temp 0.
#
# SUBMIT (qwen-only, 1 GPU):
#   cd /path/to/identity-opinion-sycophancy
#   QWEN3_CHAT_TEMPLATE=scripts/qwen3_nothink.jinja \
#     sbatch --partition=<CONFIRM> --account=at8231 scripts/serve_gen.slurm
#
# eng (runs the pipeline locally + tunnels — Option A) then sets:
#   GEN_BASE_URL=http://localhost:18003/v1   GEN_API_KEY=vllm-dummy
# and re-runs ONLY the defective qwen3-32b cells against it.
# The job log (logs/hpc/serve_gen_<jobid>.out) prints the exact tunnel line.
# =====================================================================
#SBATCH --job-name=idb-gen-qwen
#SBATCH --account=at8231
#SBATCH --partition=PARTITION_TODO        # <-- CONFIRM the H200 partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                       # qwen3-32b only, single H200
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/hpc/serve_gen_%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
LOGDIR="logs/hpc"; mkdir -p "$LOGDIR"

# --- cluster environment -------------------------------------------------
# TODO(confirm on cluster): make `vllm` importable (module load or venv).
module load cuda 2>/dev/null || true
[ -n "${VLLM_VENV:-}" ] && source "${VLLM_VENV}/bin/activate"
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/.hf_cache}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

NODE="$(hostname -s)"
echo "[$(date)] serve_gen (qwen3-32b fallback) job ${SLURM_JOB_ID:-?} on node ${NODE}"

# qwen3-32b: served thinking-OFF to match the existing non-thinking good
# cells. Mechanism: a chat template whose enable_thinking default is false,
# passed via QWEN3_CHAT_TEMPLATE (build it pre-submit).
QWEN3_ARGS=()
if [ -n "${QWEN3_CHAT_TEMPLATE:-}" ]; then
  QWEN3_ARGS=(--chat-template "${QWEN3_CHAT_TEMPLATE}")
  echo "[$(date)] qwen3-32b: no-think chat template = ${QWEN3_CHAT_TEMPLATE}"
else
  echo "[$(date)] WARN qwen3-32b: QWEN3_CHAT_TEMPLATE unset -> thinking ON;" \
       "do NOT hand this endpoint to eng until it is set."
fi

# served-model-name registers BOTH ids (per eng): the bare name (used by
# generate_responses.py for the response-file prefix) and the OpenRouter id
# (used by opinion_only.py). vLLM maps both to the one loaded model.
echo "[$(date)] launch qwen3-32b: load=Qwen/Qwen3-32B port=18003 tp=1 gpu=0"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b qwen/qwen3-32b \
  --port 18003 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --trust-remote-code \
  --disable-log-requests \
  "${QWEN3_ARGS[@]}" \
  > "${LOGDIR}/vllm_qwen3-32b_${SLURM_JOB_ID}.log" 2>&1 &
echo "$!" > "${LOGDIR}/vllm_qwen3-32b.pid"

# health check (first run also downloads ~65GB of weights -> allow 40 min)
t=0
until curl -sf "http://localhost:18003/health" >/dev/null 2>&1; do
  t=$((t+1))
  if [ "$t" -gt 480 ]; then echo "[ERROR] qwen3-32b not healthy after 40min"; break; fi
  sleep 5
done
echo "[$(date)] qwen3-32b health-check done (t=${t})"

cat <<EOF

============================ GENERATION ENDPOINT =============================
Compute node: ${NODE}
  qwen3-32b   port 18003   --model qwen3-32b   (thinking OFF)
API key: vllm-dummy   (server ignores it; the OpenAI client still needs one set)

eng runs locally + tunnels (Option A). Open the tunnel (reuses the SSH master):
  ssh -N -L 18003:${NODE}:18003 nchc
then:  GEN_BASE_URL=http://localhost:18003/v1
==============================================================================
EOF

# keep the job alive while the server runs (until walltime or scancel)
wait
