#!/bin/bash
#SBATCH -J idb-judge-qwen36
#SBATCH -o idb_judge_qwen36.%j.out
#SBATCH -e idb_judge_qwen36.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH -p h200q
#SBATCH --account=at8231
# =====================================================================
# Panel judge 3 — Qwen3.6-27B, served as a PURE long-lived endpoint
# (data-parallel x2). Served NON-THINKING: --chat-template qwen36_nothink
# .jinja flips the template's enable_thinking default to false, so a judge
# call emits a short structured score, not ~3k tokens of reasoning (that
# was the 0.3-call/s throughput killer). Validated panel role; gemma is
# likewise non-reasoning, so this is consistent.
# Qwen3.6's Gated-DeltaNet prefill is forced to triton (no nvcc here).
# eng's run_judge_panel.sh drives judging against this endpoint.
# Set JUDGE_WORKDIR to the directory containing local judge assets.
# =====================================================================
set -u
MODEL="Qwen/Qwen3.6-27B"
PORT=$((8000 + SLURM_JOB_ID % 1000))
NODE=$(hostname -s)
NOTHINK="${JUDGE_WORKDIR:?Set JUDGE_WORKDIR to the local judge-assets directory}/qwen36_nothink.jinja"

module purge
module load slurm/slurm/23.02.4
module load anaconda/2024.02
source activate "$HOME/envs/medrag"            # vLLM 0.21
export HF_HOME="$HOME/hf_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

cd "${JUDGE_WORKDIR:?Set JUDGE_WORKDIR to the local judge-assets directory}"
[ -f "$NOTHINK" ] || { echo "[ERROR] missing $NOTHINK"; exit 1; }
echo "=== job $SLURM_JOB_ID node=$NODE port=$PORT model=$MODEL DP=1 (non-thinking) ==="

vllm serve "$MODEL" \
    --port "$PORT" --host 0.0.0.0 \
    --served-model-name "$MODEL" qwen/qwen3.6-27b \
    --reasoning-parser qwen3 \
    --chat-template "$NOTHINK" \
    --additional-config '{"gdn_prefill_backend":"triton"}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    > "vllm_qwen36.${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

READY=0
for i in $(seq 1 220); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER DIED"; tail -n 70 "vllm_qwen36.${SLURM_JOB_ID}.log"; exit 1
    fi
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "server ready after ~$((i * 10))s"; READY=1; break
    fi
    sleep 10
done
[ "$READY" -ne 1 ] && { echo "server failed"; tail -n 70 "vllm_qwen36.${SLURM_JOB_ID}.log"; exit 1; }

echo "$NODE $PORT" > qwen36_endpoint.txt
echo "=== qwen3.6 judge endpoint live: $NODE:$PORT — staying up for walltime ==="
wait $SERVER_PID
