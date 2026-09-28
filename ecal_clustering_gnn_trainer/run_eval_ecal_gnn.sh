#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 3 ]] || { echo "Usage: $0 <setup> <legacy|1|2> <checkpoint> [split] [extra args...]"; exit 1; }
SETUP="$1";STEP="$2";CKPT="$3";SPLIT="${4:-test}"; if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
BASE=/home/comet/davidz/ReconDev/ReconData
case "$STEP" in legacy) DATA="$BASE/datasets/ecal_clustering/processed/pyg/$SETUP";RUN="$BASE/runs/ecal_clustering/$SETUP";;1) DATA="$BASE/datasets/ecal_clustering/processed/pyg_step1/$SETUP";RUN="$BASE/runs/ecal_clustering/$SETUP/step1";;2) DATA="$BASE/datasets/ecal_clustering/processed/pyg_step2/$SETUP";RUN="$BASE/runs/ecal_clustering/$SETUP/step2";;*)exit 2;;esac
python evaluate_ecal_gnn.py --dataset-dir "$DATA" --run-dir "$RUN" --checkpoint "$CKPT" --split "$SPLIT" --batch-size 256 --root-output auto "$@"
