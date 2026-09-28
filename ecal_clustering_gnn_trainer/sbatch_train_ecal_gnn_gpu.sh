#!/usr/bin/env bash
#SBATCH --job-name=ecal_physaware_train
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=0
#SBATCH --gres=shard:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SETUP="${1:-em_20to120MeV_10cmPlateGun_0To10deg}"
STEP="${2:-1}"
EXTRA=("${@:3}")
[[ "$STEP" == legacy || "$STEP" == 1 || "$STEP" == 2 ]] || exit 2
mkdir -p logs
source /home/comet/davidz/PyTorch/miniforge3/etc/profile.d/conda.sh
conda activate gnn-pyg
export PYTHONPYCACHEPREFIX="/home/comet/davidz/ReconDev/ReconData/cache/pycache/ecal_clustering/${SETUP}/step${STEP}"
mkdir -p "$PYTHONPYCACHEPREFIX"
cd /home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer
DEFAULT=(
 --epochs 50 --batch-size 256 --lr 1e-3 --weight-decay 1e-5
 --node-input-dim 8 --edge-input-dim 6
 --node-hidden-dim 64 --edge-hidden-dim 64
 --node-encoder-mid-dim 32 --edge-encoder-mid-dim 32
 --message-hidden-dim 128 --num-gnn-layers 3
 --window-input-dim 13 --window-hidden-dim 64
 --event-edge-hidden-dim 32 --event-message-hidden-dim 128 --num-event-gnn-layers 2
 --association-edge-hidden-dim 32 --association-hidden-dim 128
 --association-loss-weight 0.20 --association-pos-weight auto
 --graph-pool add_mean_max --event-pool add_mean_max
 --event-edge-mode hybrid --event-hybrid-bidirectional-ns 30
 --step1-prompt-time-gate-ns 30
 --regression-hidden-dim 128
 --track-hidden-dim 64 --track-embedding-dim 64 --pdg-embedding-dim 8
 --step1-time-weight 2.0 --step1-position-weight 2.0
 --step1-direction-weight 1.0 --step1-cluster-energy-weight 1.0 --step1-max-crystal-energy-weight 1.0
 --window-role-loss-weight 0.10 --window-role-class-weighting auto
 --stats-max-events 100000 --stats-seed 12345
 --mlp-depth 2
)
echo "[job] host=$(hostname) date=$(date) setup=$SETUP step=$STEP extra=${EXTRA[*]:-none}"
python -c "import torch; print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),'nGPU',torch.cuda.device_count())"
nvidia-smi || true
./run_train_ecal_gnn.sh "$SETUP" "$STEP" "${DEFAULT[@]}" "${EXTRA[@]}"
