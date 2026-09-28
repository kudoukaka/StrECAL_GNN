#!/usr/bin/env bash
#SBATCH --job-name=ecal_multiobj_eval
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --gres=shard:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SETUP="${1:-em_20to120MeV_10cmPlateGun_0To10deg}";STEP="${2:-1}";BASE=/home/comet/davidz/ReconDev/ReconData/runs/ecal_clustering/$SETUP
case "$STEP" in 1)DEF="$BASE/step1/checkpoints/best.pt";;2)DEF="$BASE/step2/checkpoints/best.pt";;legacy)DEF="$BASE/checkpoints/best.pt";;*)exit 2;;esac
CKPT="${3:-$DEF}";SPLIT="${4:-test}";EXTRA=("${@:5}")
mkdir -p logs;source /home/comet/davidz/PyTorch/miniforge3/etc/profile.d/conda.sh;conda activate gnn-pyg
export PYTHONPYCACHEPREFIX="/home/comet/davidz/ReconDev/ReconData/cache/pycache/ecal_clustering/${SETUP}/step${STEP}";mkdir -p "$PYTHONPYCACHEPREFIX"
cd /home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer
./run_eval_ecal_gnn.sh "$SETUP" "$STEP" "$CKPT" "$SPLIT" "${EXTRA[@]}"
