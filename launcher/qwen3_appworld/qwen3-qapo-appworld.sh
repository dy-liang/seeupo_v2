#!/bin/bash
# Single-node QAPO training on AppWorld.
# appworld.sh binds 127.0.0.1:8080 — set env_service.env_url in your YAML to http://localhost:8080 (or change the port in appworld.sh and YAML together).
# 1) Set CONDA_SH (and optionally SWANLAB_API_KEY, APPWORLD_CONDA_ENV, TRAIN_CONDA_ENV, APPWORLD_ROOT).
# 2) Run from anywhere:  bash /path/to/SeeUPO/launcher/qwen3_appworld/qwen3-qapo-appworld.sh

# nohup bash launcher/qwen3_appworld/qwen3-qapo-appworld.sh > ./log/seeupo_qapo_qwen3-14b.log 2>&1 &
set -e
# export HF_ENDPOINT=https://hf-mirror.com
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

APPWORLD_CONDA_ENV="${APPWORLD_CONDA_ENV:-appworld}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-seeupo}"
# Optional: export APPWORLD_ROOT="/path/to/appworld/assets" if not using the default under env_service/environments/appworld
LOG_FILE="${PROJECT_ROOT}/appworld_service.log"

echo "[1/2] Starting AppWorld env service (nohup) -> ${LOG_FILE}"

# nohup bash -c "
#     . \"${CONDA_SH}\"
#     conda init
#     conda activate \"${APPWORLD_CONDA_ENV}\"
#     cd \"${PROJECT_ROOT}/env_service/launch_script\"
#     exec bash appworld.sh
# " > "${LOG_FILE}" 2>&1 &
# echo "AppWorld env service PID: $!"
# sleep "${APPWORLD_STARTUP_SLEEP:-10}"

echo "[2/2] Starting training (launcher.py)..."
. "${CONDA_SH}"
conda activate "${TRAIN_CONDA_ENV}"
export SWANLAB_API_KEY=HJ7NQyKonax5VYOeVCl1P
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

exec python launcher.py --conf "${PROJECT_ROOT}/launcher/qwen3_appworld/qwen3-qapo-appworld.yaml"

sh /home/hly/projects/llm-agent/AgentEvolver/gpu.sh
echo "GPU held."
