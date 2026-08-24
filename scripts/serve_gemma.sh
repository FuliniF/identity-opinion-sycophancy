#!/bin/bash
#SBATCH -J idb-judge-gemma
#SBATCH -o idb_judge_gemma.%j.out
#SBATCH -e idb_judge_gemma.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:gpu:1
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH -p h200q
#SBATCH --account=at8231
# =====================================================================
# Panel judge 2 — gemma-4-31b-it, served as a PURE long-lived endpoint
# (single H200; defq was booked ~6 days out, h200q has immediate 1-GPU
# slots). Non-reasoning model: vLLM defaults, no reasoning config.
# eng's run_judge_panel.sh drives judging against this endpoint.
# Set JUDGE_WORKDIR to the directory containing local judge assets.
# =====================================================================
set -u
MODEL="google/gemma-4-31B-it"                 # exact cached HF repo id
PORT=$((8000 + SLURM_JOB_ID % 1000))          # per-job port (avoid collisions)
NODE=$(hostname -s)

module purge
module load slurm/slurm/23.02.4
module load anaconda/2024.02
source activate "$HOME/envs/medrag"            # vLLM 0.21
export HF_HOME="$HOME/hf_cache"
# No CUDA toolkit / nvcc on this cluster — keep vLLM off the JIT-only kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

cd "${JUDGE_WORKDIR:?Set JUDGE_WORKDIR to the local judge-assets directory}"
echo "=== job $SLURM_JOB_ID node=$NODE port=$PORT model=$MODEL DP=1 ==="

vllm serve "$MODEL" \
    --port "$PORT" --host 0.0.0.0 \
    --served-model-name "$MODEL" google/gemma-4-31b-it \
    --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    > "vllm_gemma.${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

READY=0
for i in $(seq 1 200); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "SERVER DIED"; tail -n 60 "vllm_gemma.${SLURM_JOB_ID}.log"; exit 1
    fi
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "server ready after ~$((i * 10))s"; READY=1; break
    fi
    sleep 10
done
[ "$READY" -ne 1 ] && { echo "server failed"; tail -n 60 "vllm_gemma.${SLURM_JOB_ID}.log"; exit 1; }

echo "$NODE $PORT" > gemma_endpoint.txt
echo "=== gemma judge endpoint live: $NODE:$PORT — staying up for walltime ==="
wait $SERVER_PID
