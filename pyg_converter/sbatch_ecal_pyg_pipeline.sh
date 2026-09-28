#!/usr/bin/env bash
#SBATCH --job-name=ecal_pyg_hier
#SBATCH --partition=geant
#SBATCH --account=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
SETUP="${1:-em_20to120MeV_10cmPlateGun_0To10deg}"; SOURCE="${2:-truth}"; STEP="${3:-1}"; EXTRA=("${@:4}")
[[ "$SOURCE" == reco || "$SOURCE" == truth ]] || { echo "[error] second arg must be reco/truth" >&2; exit 2; }; [[ "$STEP" == legacy || "$STEP" == 1 || "$STEP" == 2 ]] || { echo "[error] third arg must be legacy/1/2" >&2; exit 2; }
mkdir -p logs; source /home/comet/davidz/PyTorch/miniforge3/etc/profile.d/conda.sh; conda activate gnn-pyg
export PYTHONPYCACHEPREFIX="/home/comet/davidz/ReconDev/ReconData/cache/pycache/ecal_clustering/${SETUP}/pyg_step${STEP}"; mkdir -p "$PYTHONPYCACHEPREFIX"
echo "[job] host=$(hostname) date=$(date) setup=$SETUP source=$SOURCE step=$STEP"; python -c "import torch; print('[env] torch',torch.__version__)"
cd /home/comet/davidz/ReconDev/pyg_converter
DEFAULT=(--graph-source "$SOURCE" --reconstruction-step "$STEP" --y-scale-mm 500 --z-scale-mm 500 --truth-node-energy-threshold-mev 1.2 --edge-radius-mm 100 --edge-time-ns 30 --edge-weight-scale-mm 10 --seed-threshold-mv 8 --seed-window-half-mm 70 --seed-min-total-amplitude-mv 60 --amp-to-energy 17.96 --n-train-files 16 --n-val-files 2 --n-test-files 2)
# EXTRA is appended last so command-line choices override defaults where argparse uses the last occurrence.
./run_ecal_pyg_pipeline.sh "$SETUP" "${DEFAULT[@]}" "${EXTRA[@]}"
echo "[done] date=$(date)"
