#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'EOF'
Usage:
  run_ecal_pyg_pipeline.sh <setup> [options]

Important options:
  --graph-source reco|truth            default: reco
  --reconstruction-step legacy|1|2     default: 1
  --processed-dir DIR                  override automatic output directory
  --edge-time-ns T                     pulse-edge cut and, for Step1/2, seed-window time half-width
  --shuffle-files                      randomize ROOT-file order before train/val/test split
  --split-seed N                       default: 12345

Automatic processed directories:
  legacy : .../processed/pyg/<setup>
  step 1 : .../processed/pyg_step1/<setup>
  step 2 : .../processed/pyg_step2/<setup>

Hierarchical Step1/2 policy:
  one PyG Data per ROOT event; data.y has K rows (one target object per represented primary).
  Regression node features are 8D absolute+window-relative features.
  Window features are 13D and include Esum/shower moments/Emax fraction/core position.
  Truth and reco feature builders are separate code paths.
  Pulse edges remain inside seed windows. Window/event edges are built at runtime.
  The trainer default is hybrid: |dt|<=30 ns bidirectional, otherwise earlier -> later.
  association_target stores one same-primary label per unordered window pair.
EOF
}
[[ $# -ge 1 ]] || { usage; exit 1; }
SETUP="$1"; shift
BASE_DIR="/home/comet/davidz/ReconDev/ReconData/datasets/ecal_clustering"; TRUTH_DIR=""; SPLIT_DIR=""; PROCESSED_DIR=""; SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TOOLS_DIR="$SCRIPT_DIR"
PATTERN="*.root"; NTRAIN=16; NVAL=2; NTEST=2; SHUFFLE=0; SPLIT_SEED=12345
GRAPH_SOURCE="reco"; STEP="1"; RECO_TREE="ecalRecoPulses"; TRUTH_DIGITS="ecalTruthDigits"; TRUTH_CLUSTERS="ecalTruthClusters"; TRUTH_ENERGY="deposited"; TRUTH_FALLBACK=0
Y_SCALE=500; Z_SCALE=500; Z_OFFSET=7650; NODE_THRESHOLD=0.0; EDGE_RADIUS=100; EDGE_TIME=50; EDGE_WEIGHT_SCALE=10; SEED_THRESHOLD=10; SEED_WINDOW=35; SEED_MIN_TOTAL=80; AMP_TO_ENERGY=17.96; DYNAMIC=1; STORE_RAW=1; NORMALIZE=1; NORMALIZATION_JSON=""; SKIP_LISTS=0; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir) BASE_DIR="$2"; shift 2;; --truth-dir) TRUTH_DIR="$2"; shift 2;; --split-dir) SPLIT_DIR="$2"; shift 2;; --processed-dir) PROCESSED_DIR="$2"; shift 2;; --tools-dir) TOOLS_DIR="$2"; shift 2;; --pattern) PATTERN="$2"; shift 2;;
    --n-train-files) NTRAIN="$2"; shift 2;; --n-val-files) NVAL="$2"; shift 2;; --n-test-files) NTEST="$2"; shift 2;; --shuffle-files) SHUFFLE=1; shift;; --split-seed) SPLIT_SEED="$2"; shift 2;;
    --graph-source) GRAPH_SOURCE="$2"; shift 2;; --truth-mode) GRAPH_SOURCE="truth"; shift;; --reconstruction-step) STEP="$2"; shift 2;; --tree) RECO_TREE="$2"; shift 2;; --truth-digits-tree) TRUTH_DIGITS="$2"; shift 2;; --truth-clusters-tree) TRUTH_CLUSTERS="$2"; shift 2;; --truth-energy) TRUTH_ENERGY="$2"; shift 2;; --truth-label-fallback-primary) TRUTH_FALLBACK=1; shift;;
    --normalization-json) NORMALIZATION_JSON="$2"; shift 2;; --y-scale-mm) Y_SCALE="$2"; shift 2;; --z-scale-mm) Z_SCALE="$2"; shift 2;; --z-offset-mm) Z_OFFSET="$2"; shift 2;; --truth-node-energy-threshold-mev) NODE_THRESHOLD="$2"; shift 2;; --edge-radius-mm) EDGE_RADIUS="$2"; shift 2;; --edge-time-ns) EDGE_TIME="$2"; shift 2;; --edge-weight-scale-mm) EDGE_WEIGHT_SCALE="$2"; shift 2;; --seed-threshold-mv) SEED_THRESHOLD="$2"; shift 2;; --seed-window-half-mm) SEED_WINDOW="$2"; shift 2;; --seed-min-total-amplitude-mv) SEED_MIN_TOTAL="$2"; shift 2;; --amp-to-energy) AMP_TO_ENERGY="$2"; shift 2;;
    --disable-dynamic-seeding) DYNAMIC=0; shift;; --no-time-cut) EDGE_TIME=-1; shift;; --skip-make-lists) SKIP_LISTS=1; shift;; --no-store-raw) STORE_RAW=0; shift;; --no-normalize) NORMALIZE=0; shift;; --dry-run) DRY=1; shift;; -h|--help) usage; exit 0;; *) echo "[error] unknown option $1" >&2; usage; exit 2;;
  esac
done
[[ "$GRAPH_SOURCE" == reco || "$GRAPH_SOURCE" == truth ]] || { echo "[error] graph-source must be reco/truth" >&2; exit 2; }
[[ "$STEP" == legacy || "$STEP" == 1 || "$STEP" == 2 ]] || { echo "[error] reconstruction-step must be legacy/1/2" >&2; exit 2; }
[[ -n "$TRUTH_DIR" ]] || TRUTH_DIR="$BASE_DIR/truthmaker/$SETUP"; [[ -n "$SPLIT_DIR" ]] || SPLIT_DIR="$BASE_DIR/splits/$SETUP"
if [[ -z "$PROCESSED_DIR" ]]; then case "$STEP" in legacy) PROCESSED_DIR="$BASE_DIR/processed/pyg/$SETUP";; 1) PROCESSED_DIR="$BASE_DIR/processed/pyg_step1/$SETUP";; 2) PROCESSED_DIR="$BASE_DIR/processed/pyg_step2/$SETUP";; esac; fi
MAKE="$TOOLS_DIR/make_filelists.py"; CONV="$TOOLS_DIR/convert_ecal_reco_pulses_filelists_to_pyg.py"; CHECK="$TOOLS_DIR/check_ecal_pyg_dataset.py"
run(){ echo; echo "[cmd] $*"; [[ "$DRY" -eq 1 ]] || "$@"; }
echo "[config] setup=$SETUP source=$GRAPH_SOURCE step=$STEP"; echo "  truth=$TRUTH_DIR"; echo "  split=$SPLIT_DIR"; echo "  output=$PROCESSED_DIR"; echo "  seed spatial half=$SEED_WINDOW mm time half=$EDGE_TIME ns"; echo "  file shuffle=$SHUFFLE seed=$SPLIT_SEED"
[[ -d "$TRUTH_DIR" ]] || { echo "[error] missing $TRUTH_DIR" >&2; exit 2; }; mkdir -p "$SPLIT_DIR" "$PROCESSED_DIR"
if [[ "$SKIP_LISTS" -eq 0 ]]; then
  LISTARGS=(python "$MAKE" "$TRUTH_DIR/$PATTERN" -o "$SPLIT_DIR" --n-train-files "$NTRAIN" --n-val-files "$NVAL" --n-test-files "$NTEST")
  [[ "$SHUFFLE" -eq 0 ]] || LISTARGS+=(--shuffle --seed "$SPLIT_SEED"); run "${LISTARGS[@]}"
fi
for f in train val test; do [[ -f "$SPLIT_DIR/$f.list" ]] || { echo "[error] missing $SPLIT_DIR/$f.list" >&2; exit 2; }; done
CMD=(python "$CONV" --train-list "$SPLIT_DIR/train.list" --val-list "$SPLIT_DIR/val.list" --test-list "$SPLIT_DIR/test.list" -o "$PROCESSED_DIR" --graph-source "$GRAPH_SOURCE" --reconstruction-step "$STEP" --tree "$RECO_TREE" --truth-digits-tree "$TRUTH_DIGITS" --truth-clusters-tree "$TRUTH_CLUSTERS" --truth-energy "$TRUTH_ENERGY" --y-scale-mm "$Y_SCALE" --z-scale-mm "$Z_SCALE" --z-offset-mm "$Z_OFFSET" --truth-node-energy-threshold-mev "$NODE_THRESHOLD" --edge-radius-mm "$EDGE_RADIUS" --edge-time-ns "$EDGE_TIME" --edge-weight-scale-mm "$EDGE_WEIGHT_SCALE" --seed-threshold-mv "$SEED_THRESHOLD" --seed-window-half-mm "$SEED_WINDOW" --seed-min-total-amplitude-mv "$SEED_MIN_TOTAL" --amp-to-energy "$AMP_TO_ENERGY")
[[ -z "$NORMALIZATION_JSON" ]] || CMD+=(--normalization-json "$NORMALIZATION_JSON"); [[ "$TRUTH_FALLBACK" -eq 0 ]] || CMD+=(--truth-label-fallback-primary); [[ "$STORE_RAW" -eq 0 ]] || CMD+=(--store-raw); [[ "$DYNAMIC" -eq 1 ]] || CMD+=(--disable-dynamic-seeding); [[ "$NORMALIZE" -eq 1 ]] || CMD+=(--no-normalize)
run "${CMD[@]}"
echo; echo "[check]"; run python "$CHECK" "$PROCESSED_DIR" --split train --max-files 1 --strict
echo "[done] $PROCESSED_DIR"
