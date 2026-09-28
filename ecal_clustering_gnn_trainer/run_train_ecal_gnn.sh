#!/usr/bin/env bash
set -euo pipefail
usage(){ cat <<'EOF'
Usage:
  run_train_ecal_gnn.sh <setup> <legacy|1|2> [extra train args...]

Physics-aware hierarchy defaults/options:
  --event-edge-mode hybrid|time_directed|bidirectional_complete
  --event-hybrid-bidirectional-ns 30
  --step1-prompt-time-gate-ns 30
  --association-loss-weight 0.20
  --association-pos-weight auto

Step-1 residual targets:
  dt = tmin(prompt window) - t0
  dy = yc(prompt window) - y_inc
  dz = zc(prompt window) - z_inc

Step-1 separate heads:
  --step1-timing-head-depth N
  --step1-position-head-depth N
  --step1-direction-head-depth N
  --step1-energy-head-depth N

Default Step-1 loss weights:
  time=2, position=2, direction=1, Ecluster=1, Emax=1

Statistics:
  --stats-max-events 100000
  --stats-seed N

Global/per-block MLP depth:
  --mlp-depth N plus individual --*-depth overrides.
EOF
}
[[ $# -ge 2 ]] || { usage; exit 1; }
SETUP="$1"; STEP="$2"; shift 2
[[ "$STEP" == legacy || "$STEP" == 1 || "$STEP" == 2 ]] || { echo "[error] step must be legacy/1/2" >&2; exit 2; }
BASE=/home/comet/davidz/ReconDev/ReconData
case "$STEP" in
  legacy) DATA="$BASE/datasets/ecal_clustering/processed/pyg/$SETUP"; RUN="$BASE/runs/ecal_clustering/$SETUP";;
  1) DATA="$BASE/datasets/ecal_clustering/processed/pyg_step1/$SETUP"; RUN="$BASE/runs/ecal_clustering/$SETUP/step1";;
  2) DATA="$BASE/datasets/ecal_clustering/processed/pyg_step2/$SETUP"; RUN="$BASE/runs/ecal_clustering/$SETUP/step2";;
esac
mkdir -p "$RUN"
python train_ecal_gnn.py --dataset-dir "$DATA" --run-dir "$RUN" --reconstruction-step "$STEP" "$@"
