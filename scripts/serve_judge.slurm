#!/bin/bash
# =====================================================================
# Phase 2 — 3-JUDGE PANEL on the NCHC H200, SHARDED (team-lead option (b)).
# Serves one or more panel judges, ONE GPU each, via the vLLM Singularity
# container. Small jobs backfill into the (full) cluster far faster than
# waiting for a contiguous 4-GPU block.
#
# Judges (each fits a single H200; gpt-oss-120b ~63GB at TP=1):
#   gpt-oss   -> openai/gpt-oss-120b      port 18001
#   gemma     -> google/gemma-4-31b-it    port 18004
#   qwen36    -> Qwen/Qwen3.6-27B         port 18005
#
# Set BASE below to your project directory on the cluster.
# Container + weights are pre-staged by stage.sh.
#
# SUBMIT (from your project directory; --gres N == number of judge args):
#   1-GPU job:  sbatch --account=<ACC> scripts/serve_judge.slurm gpt-oss
#   2-GPU job:  sbatch --account=<ACC> --gres=gpu:H200:2 scripts/serve_judge.slurm gemma qwen36
# ACCOUNT: one of gov112004 | gov112009 | gov113008 | mst113296  (NOT at8231).
# MaxJobsPU=2 -> recommended layout is 2 jobs (1-GPU gpt-oss + 2-GPU gemma+qwen36)
# so all 3 judges are online at once rather than 2-then-1.
#
# eng (judges locally + tunnels — Option A) per online endpoint:
#   ssh -N -L <port>:<node>:<port> nchc
#   JUDGE_BASE_URL=http://localhost:<port>/v1  JUDGE_API_KEY=vllm-dummy
# Each job's log (logs/serve_judge_<jobid>.out) prints its node + tunnel line.
# =====================================================================
#SBATCH --job-name=idb-judge
#SBATCH --partition=normal2
#SBATCH --account=ACCOUNT_TODO              # <-- set via: sbatch --account=<gov…|mst…>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H200:1                   # override on CLI for multi-judge jobs
#SBATCH --cpus-per-task=12               # cluster limit is 12 CPUs per GPU
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve_judge_%j.out

set -u
BASE=${BASE:-$PWD}
SIF="$BASE/sif/vllm-openai.sif"
HFC="$BASE/hf_cache"
LOGDIR="$BASE/logs"
mkdir -p "$LOGDIR"

if [ "$#" -lt 1 ]; then echo "[ERROR] usage: serve_judge.slurm <judge> [<judge>...]"; exit 1; fi

# This job may backfill in before pre-staging (stage.sh) finishes. Wait for the
# container SIF + the stage.sh DONE marker rather than failing (timeout 2h).
STAGELOG="$LOGDIR/stage.log"
t=0
while [ ! -f "$SIF" ] || ! grep -q '\[stage\] DONE' "$STAGELOG" 2>/dev/null; do
  t=$((t+1))
  if [ "$t" -gt 1440 ]; then echo "[ERROR] staging incomplete after 2h (SIF=$SIF)"; exit 1; fi
  [ $((t % 24)) -eq 1 ] && echo "[$(date)] waiting for staging (container SIF + DONE marker)..."
  sleep 5
done
echo "[$(date)] staging complete — container + weights ready"

NODE="$(hostname -s)"
echo "[$(date)] serve_judge job ${SLURM_JOB_ID:-?} on node ${NODE}  judges: $*"

# short name -> "hf_model|served-names|port"
spec() {
  case "$1" in
    gpt-oss) echo "openai/gpt-oss-120b|gpt-oss-120b openai/gpt-oss-120b|18001" ;;
    gemma)   echo "google/gemma-4-31b-it|gemma-4-31b-it google/gemma-4-31b-it|18004" ;;
    qwen36)  echo "Qwen/Qwen3.6-27B|qwen3.6-27b qwen/qwen3.6-27b|18005" ;;
    *)       echo "ERR" ;;
  esac
}

launch() {  # <name> <hf_model> <served-names> <port> <cuda_device>
  local name="$1" model="$2" served="$3" port="$4" dev="$5"
  echo "[$(date)] launch ${name}: model=${model} port=${port} gpu=${dev}"
  singularity exec --nv --bind /work \
    --env CUDA_VISIBLE_DEVICES="${dev}" \
    --env HF_HOME="${HFC}" \
    "$SIF" \
    vllm serve "${model}" \
      --served-model-name ${served} \
      --port "${port}" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.92 \
      --max-model-len 32768 \
      --trust-remote-code \
      --disable-log-requests \
    > "${LOGDIR}/vllm_${name}_${SLURM_JOB_ID}.log" 2>&1 &
  echo "$!" > "${LOGDIR}/vllm_${name}_${SLURM_JOB_ID}.pid"
}

wait_healthy() {  # <port> <name>
  local port="$1" name="$2" t=0
  until curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
    t=$((t+1))
    if [ "$t" -gt 360 ]; then echo "[ERROR] ${name} not healthy after 30min"; return 1; fi
    sleep 5
  done
  echo "[$(date)] OK ${name} healthy on :${port}"
}

dev=0
PORTS=()
for j in "$@"; do
  s="$(spec "$j")"
  if [ "$s" = "ERR" ]; then echo "[ERROR] unknown judge '$j'"; exit 1; fi
  IFS='|' read -r model served port <<< "$s"
  launch "$j" "$model" "$served" "$port" "$dev"
  PORTS+=("$j:$port")
  dev=$((dev+1))
done

for p in "${PORTS[@]}"; do
  IFS=':' read -r j port <<< "$p"
  wait_healthy "$port" "$j"
done

echo
echo "========================= JUDGE ENDPOINT(S) on ${NODE} ========================="
for p in "${PORTS[@]}"; do
  IFS=':' read -r j port <<< "$p"
  echo "  ${j}  http://${NODE}:${port}/v1   (API key: vllm-dummy)"
done
echo "Tunnel from the laptop (reuses the SSH master):"
TUN=""; for p in "${PORTS[@]}"; do IFS=':' read -r j port <<< "$p"; TUN="$TUN -L ${port}:${NODE}:${port}"; done
echo "  ssh -N${TUN} nchc"
echo "==============================================================================="

# keep the job alive while the server(s) run (until walltime or scancel)
wait
