#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ECAL raw production pipeline helper
#
# Steps:
#   sim        : submit Geant4/ICEDUST simulation array using submit.sbatch
#   noise      : submit AddStrEcalRandomNoise array for all ROOT files in input dir
#   detresp    : submit RunSimDetectorResponse array for all ROOT files in input dir
#   truthmaker : submit RunECALTruthMaker array for all ROOT files in input dir
#   all        : submit sim, then after sim finishes submit noise/detresp/truthmaker
#
# Output layout under BASE_DIR:
#   g4/<SIMSETUP>/
#   g4_with_noise/<SIMSETUP>_<NOISESETUP>/
#   detresp/<SIMSETUP>[_<NOISESETUP>]/
#   truthmaker/<SIMSETUP>[_<NOISESETUP>]/
# ============================================================

BASE_DIR="/home/comet/davidz/ReconDev/ReconData/datasets"
SUBMIT_SBATCH="submit.sbatch"
CFG_FILE="particle_gun.cfg"
SIMSETUP=""
NOISESETUP=""
RUN_NUMBER="1"
SIM_ARRAY="0-9%5"
MAX_PARALLEL="5"
STEP="all"
INPUT_DIR=""
ROOT_GLOB="*.root"

PARTITION="geant"
ACCOUNT="research"
CPUS_PER_TASK="1"
MEM="8G"
TIME="08:00:00"

STRAW_RATE="0"
STRAW_SIGMA="15"
ECAL_OCC="0"

ICEDUST_SETUP="/home/comet/davidz/icedust/ICEDUST_install/setup.sh"
FIELDMAP_ENABLE="/home/comet/davidz/icedust/ICEDUST_fieldmaps/240700_Phase-I_Opera_10mmGrid_4Regions_PCS4p4T_TSso2p7T_TSdp44mT/enable.sh"
RECON_BIN="/home/comet/davidz/ReconDev/build/bin"

DRY_RUN="0"
SELF_PATH=""

usage() {
  cat <<USAGE
Usage:
  $0 [options]

Main options:
  --step all|sim|noise|detresp|truthmaker|continue-after-sim|run-stage
  --simsetup NAME                 simulation setup name, e.g. gnn_baseline_001
  --noisesetup NAME               noise setup suffix, e.g. occ01
  --base-dir DIR                  default: ${BASE_DIR}
  --input-dir DIR                 required for standalone noise/detresp/truthmaker
  --root-glob GLOB                default: "*.root"

Simulation options:
  --submit-sbatch FILE            default: ${SUBMIT_SBATCH}
  --cfg FILE                      default: ${CFG_FILE}
  --run-number N                  default: ${RUN_NUMBER}
  --sim-array ARRAY               default: ${SIM_ARRAY}, e.g. 0-9%5

Noise options:
  --straw-rate VALUE              StrawCenterRateMHzPerChannel, default: ${STRAW_RATE}
  --straw-sigma VALUE             StrawSigmaWire, default: ${STRAW_SIGMA}
  --ecal-occ VALUE               EcalOccupancy, default: ${ECAL_OCC}

Slurm options:
  --partition NAME                default: ${PARTITION}
  --account NAME                  default: ${ACCOUNT}
  --max-parallel N                default: ${MAX_PARALLEL}
  --mem MEM                       default: ${MEM}
  --time TIME                     default: ${TIME}
  --cpus-per-task N               default: ${CPUS_PER_TASK}

Environment options:
  --icedust-setup FILE            default: ${ICEDUST_SETUP}
  --fieldmap-enable FILE          default: ${FIELDMAP_ENABLE}
  --recon-bin DIR                 default: ${RECON_BIN}

Other:
  --dry-run                       print sbatch commands but do not submit
  -h, --help

Examples:
  # Full chain: sim -> noise -> detresp -> truthmaker
  $0 --step all --simsetup gnn_baseline_001 --noisesetup noise_occ01 \\
     --run-number 1 --sim-array 0-9%5 \\
     --straw-rate 0.05 --straw-sigma 1.0 --ecal-occ 0.02

  # Only submit simulation
  $0 --step sim --simsetup gnn_baseline_001 --run-number 1 --sim-array 0-9%5

  # Standalone noise from an existing G4 directory
  $0 --step noise --simsetup gnn_baseline_001 --noisesetup noise_occ01 \\
     --input-dir /home/comet/davidz/ReconDev/ReconData/datasets/ecal_raw/raw/g4/gnn_baseline_001 \\
     --straw-rate 0.05 --straw-sigma 1.0 --ecal-occ 0.02

  # Standalone detector response from an existing directory
  $0 --step detresp --simsetup gnn_baseline_001 --noisesetup noise_occ01 \\
     --input-dir /home/comet/davidz/ReconDev/ReconData/datasets/ecal_raw/raw/g4_with_noise/gnn_baseline_001_noise_occ01

  # Standalone truthmaker from an existing detresp directory
  $0 --step truthmaker --simsetup gnn_baseline_001 --noisesetup noise_occ01 \\
     --input-dir /home/comet/davidz/ReconDev/ReconData/datasets/ecal_raw/raw/detresp/gnn_baseline_001_noise_occ01
USAGE
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$(pwd)" "$p"
  fi
}

is_zero() {
  awk -v x="$1" 'BEGIN { exit !(x == 0) }'
}

noise_enabled() {
  if is_zero "$STRAW_RATE" && is_zero "$ECAL_OCC"; then
    return 1
  fi
  return 0
}

run_number_padded() {
  if [[ ! "$RUN_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RUN_NUMBER must be numeric: $RUN_NUMBER" >&2
    exit 1
  fi

  # raw file name uses 8 digits, e.g. run 1 -> 00000001
  printf '%08d\n' "$((10#$RUN_NUMBER))"
}

should_use_noise_label() {
  # In the full chain, the label includes NOISESETUP only when the noise
  # stage is actually enabled. For standalone detresp/truthmaker runs, also
  # use the noise label when the provided input directory already looks like
  # a noise output directory.
  if noise_enabled; then
    return 0
  fi
  if [[ -n "$NOISESETUP" && -n "$INPUT_DIR" ]]; then
    local b
    b="$(basename "$INPUT_DIR")"
    if [[ "$b" == "${SIMSETUP}_${NOISESETUP}" || "$INPUT_DIR" == *"/g4_with_noise/"* ]]; then
      return 0
    fi
  fi
  return 1
}

label_name() {
  if should_use_noise_label; then
    printf '%s_%s\n' "$SIMSETUP" "$NOISESETUP"
  else
    printf '%s\n' "$SIMSETUP"
  fi
}

pipeline_dir() {
  printf '%s/.pipeline/%s\n' "$BASE_DIR" "$(label_name)"
}

sim_dir() {
  printf '%s/raw/g4/%s\n' "$BASE_DIR" "$SIMSETUP"
}

noise_dir() {
  printf '%s/raw/g4_with_noise/%s_%s\n' "$BASE_DIR" "$SIMSETUP" "$NOISESETUP"
}

detresp_dir() {
  printf '%s/raw/detresp/%s\n' "$BASE_DIR" "$(label_name)"
}

truthmaker_dir() {
  printf '%s/ecal_clustering/truthmaker/%s\n' "$BASE_DIR" "$(label_name)"
}

log_dir_for_stage() {
  local stage="$1"
  case "$stage" in
    sim) printf '%s/logs\n' "$(sim_dir)" ;;
    noise) printf '%s/logs\n' "$(noise_dir)" ;;
    detresp) printf '%s/logs\n' "$(detresp_dir)" ;;
    truthmaker) printf '%s/logs\n' "$(truthmaker_dir)" ;;
    continue-after-sim) printf '%s/logs\n' "$(pipeline_dir)" ;;
    *) printf '%s/logs\n' "$(pipeline_dir)" ;;
  esac
}

mkdirs_common() {
  mkdir -p "$BASE_DIR" "$(pipeline_dir)"
}

run_sbatch() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN:' >&2
    printf ' %q' sbatch "$@" >&2
    printf '\n' >&2
    printf 'DRYRUN_JOBID\n'
  else
    sbatch --parsable "$@" | cut -d ';' -f 1
  fi
}

count_manifest() {
  local manifest="$1"
  if [[ ! -s "$manifest" ]]; then
    printf '0\n'
    return
  fi
  wc -l < "$manifest" | tr -d ' '
}

make_manifest_from_dir() {
  local input_dir="$1"
  local manifest="$2"
  local run8
  run8="$(run_number_padded)"

  mkdir -p "$(dirname "$manifest")"

  if [[ ! -d "$input_dir" ]]; then
    echo "ERROR: input dir does not exist: $input_dir" >&2
    exit 1
  fi

  # Match file names like:
  #   oa_g4_em_00000001-0001_xxx_SG4_000.root
  #   oa_g4_em_00000001-0001_xxx_SG4_000_noise.root
  #   oa_g4_em_00000001-0001_xxx_SG4_000_detresp.root
  #
  # The first 8-digit field is RUN_NUMBER.
  find "$input_dir" -maxdepth 1 -type f -name "$ROOT_GLOB" \
    | awk -v run8="$run8" '
        {
          n=$0
          sub(/^.*\//, "", n)
          if (n ~ "_" run8 "-[0-9][0-9][0-9][0-9]_") {
            print $0
          }
        }
      ' \
    | sort > "$manifest"

  local n
  n="$(count_manifest "$manifest")"
  if [[ "$n" == "0" ]]; then
    echo "ERROR: no ROOT files found in $input_dir with glob $ROOT_GLOB and RUN_NUMBER=$RUN_NUMBER, padded=$run8" >&2
    exit 1
  fi
}

make_noise_output_manifest() {
  local input_manifest="$1"
  local output_manifest="$2"
  local out_dir="$3"
  mkdir -p "$(dirname "$output_manifest")" "$out_dir"
  awk -v out="$out_dir" '
    {
      n=$0
      sub(/^.*\//, "", n)
      sub(/\.root$/, "", n)
      print out "/" n "_noise.root"
    }
  ' "$input_manifest" > "$output_manifest"
}

make_detresp_output_manifest() {
  local input_manifest="$1"
  local output_manifest="$2"
  local out_dir="$3"
  mkdir -p "$(dirname "$output_manifest")" "$out_dir"
  awk -v out="$out_dir" '
    {
      n=$0
      sub(/^.*\//, "", n)
      sub(/\.root$/, "", n)
      sub(/_noise$/, "", n)
      print out "/" n "_detresp.root"
    }
  ' "$input_manifest" > "$output_manifest"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --step) STEP="$2"; shift 2 ;;
      --simsetup) SIMSETUP="$2"; shift 2 ;;
      --noisesetup) NOISESETUP="$2"; shift 2 ;;
      --base-dir) BASE_DIR="$2"; shift 2 ;;
      --input-dir) INPUT_DIR="$2"; shift 2 ;;
      --root-glob) ROOT_GLOB="$2"; shift 2 ;;
      --submit-sbatch) SUBMIT_SBATCH="$2"; shift 2 ;;
      --cfg) CFG_FILE="$2"; shift 2 ;;
      --run-number) RUN_NUMBER="$2"; shift 2 ;;
      --sim-array) SIM_ARRAY="$2"; shift 2 ;;
      --straw-rate) STRAW_RATE="$2"; shift 2 ;;
      --straw-sigma) STRAW_SIGMA="$2"; shift 2 ;;
      --ecal-occ) ECAL_OCC="$2"; shift 2 ;;
      --partition) PARTITION="$2"; shift 2 ;;
      --account) ACCOUNT="$2"; shift 2 ;;
      --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
      --mem) MEM="$2"; shift 2 ;;
      --time) TIME="$2"; shift 2 ;;
      --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
      --icedust-setup) ICEDUST_SETUP="$2"; shift 2 ;;
      --fieldmap-enable) FIELDMAP_ENABLE="$2"; shift 2 ;;
      --recon-bin) RECON_BIN="$2"; shift 2 ;;
      --self-path) SELF_PATH="$2"; shift 2 ;;
      --dry-run) DRY_RUN="1"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "ERROR: unknown option: $1" >&2; usage; exit 1 ;;
    esac
  done
}

common_arg_list() {
  printf '%q ' \
    --base-dir "$BASE_DIR" \
    --simsetup "$SIMSETUP" \
    --noisesetup "$NOISESETUP" \
    --root-glob "$ROOT_GLOB" \
    --submit-sbatch "$SUBMIT_SBATCH" \
    --cfg "$CFG_FILE" \
    --run-number "$RUN_NUMBER" \
    --sim-array "$SIM_ARRAY" \
    --straw-rate "$STRAW_RATE" \
    --straw-sigma "$STRAW_SIGMA" \
    --ecal-occ "$ECAL_OCC" \
    --partition "$PARTITION" \
    --account "$ACCOUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --mem "$MEM" \
    --time "$TIME" \
    --cpus-per-task "$CPUS_PER_TASK" \
    --icedust-setup "$ICEDUST_SETUP" \
    --fieldmap-enable "$FIELDMAP_ENABLE" \
    --recon-bin "$RECON_BIN" \
    --self-path "$SELF_PATH"
}

require_self_path() {
  if [[ -z "$SELF_PATH" ]]; then
    SELF_PATH="$(abs_path "$0")"
  fi
  if [[ ! -f "$SELF_PATH" ]]; then
    echo "ERROR: cannot find pipeline script: $SELF_PATH" >&2
    exit 1
  fi
}

submit_sim() {
  mkdirs_common
  local sdir ldir submit_abs cfg_abs work_dir jobid
  sdir="$(sim_dir)"
  ldir="$(log_dir_for_stage sim)"
  mkdir -p "$sdir" "$ldir"

  submit_abs="$(abs_path "$SUBMIT_SBATCH")"
  cfg_abs="$(abs_path "$CFG_FILE")"
  work_dir="$(pwd)"

  if [[ ! -f "$submit_abs" ]]; then
    echo "ERROR: submit sbatch not found: $submit_abs" >&2
    exit 1
  fi
  if [[ ! -f "$cfg_abs" ]]; then
    echo "ERROR: cfg file not found: $cfg_abs" >&2
    exit 1
  fi

  jobid="$(run_sbatch \
    --array="$SIM_ARRAY" \
    --job-name="g4_${SIMSETUP}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --chdir="$work_dir" \
    --output="$ldir/%x_%A_%a.out" \
    --error="$ldir/%x_%A_%a.err" \
    --export="ALL,RUN_NUMBER=${RUN_NUMBER},JOB_OUT_DIR=${sdir},BASE_CFG=${cfg_abs},WORK_DIR=${work_dir},ICEDUST_ENABLE=${ICEDUST_SETUP},ICEDUST_FIELDMAP_ENABLE=${FIELDMAP_ENABLE},RECON_BIN=${RECON_BIN}" \
    "$submit_abs")"

  echo "$jobid"
}

submit_stage_array() {
  local stage="$1"
  local input_manifest="$2"
  local out_dir="$3"
  local dep="${4:-}"
  require_self_path

  local n array_spec ldir jobid
  n="$(count_manifest "$input_manifest")"
  if [[ "$n" == "0" ]]; then
    echo "ERROR: empty manifest for stage $stage: $input_manifest" >&2
    exit 1
  fi

  mkdir -p "$out_dir"
  ldir="$(log_dir_for_stage "$stage")"
  mkdir -p "$ldir"
  array_spec="0-$((n - 1))%${MAX_PARALLEL}"

  local extra_args=()
  if [[ -n "$dep" ]]; then
    extra_args+=(--dependency="afterok:${dep}")
  fi

  jobid="$(run_sbatch \
    "${extra_args[@]}" \
    --array="$array_spec" \
    --job-name="ecal_${stage}_$(label_name)" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$ldir/%x_%A_%a.out" \
    --error="$ldir/%x_%A_%a.err" \
    "$SELF_PATH" \
    --step run-stage \
    --input-dir "$input_manifest" \
    --base-dir "$BASE_DIR" \
    --simsetup "$SIMSETUP" \
    --noisesetup "$NOISESETUP" \
    --straw-rate "$STRAW_RATE" \
    --straw-sigma "$STRAW_SIGMA" \
    --ecal-occ "$ECAL_OCC" \
    --icedust-setup "$ICEDUST_SETUP" \
    --fieldmap-enable "$FIELDMAP_ENABLE" \
    --recon-bin "$RECON_BIN" \
    --root-glob "$ROOT_GLOB" \
    --self-path "$SELF_PATH" \
    --partition "$PARTITION" \
    --account "$ACCOUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --mem "$MEM" \
    --time "$TIME" \
    --cpus-per-task "$CPUS_PER_TASK" \
    --run-number "$RUN_NUMBER" \
    --submit-sbatch "$SUBMIT_SBATCH" \
    --cfg "$CFG_FILE" \
    --sim-array "$SIM_ARRAY" \
    --stage-name "$stage" \
    --out-dir "$out_dir")"

  echo "$jobid"
}

# Extra parser only used by --step run-stage.
STAGE_NAME=""
OUT_DIR=""
parse_run_stage_extra_args() {
  local rest=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stage-name) STAGE_NAME="$2"; shift 2 ;;
      --out-dir) OUT_DIR="$2"; shift 2 ;;
      *) rest+=("$1"); shift ;;
    esac
  done
  parse_args "${rest[@]}"
}

setup_runtime_env() {
  if [[ ! -f "$ICEDUST_SETUP" ]]; then
    echo "ERROR: ICEDUST setup script not found: $ICEDUST_SETUP" >&2
    exit 1
  fi
  if [[ ! -f "$FIELDMAP_ENABLE" ]]; then
    echo "ERROR: fieldmap enable script not found: $FIELDMAP_ENABLE" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$ICEDUST_SETUP"
  # shellcheck disable=SC1090
  source "$FIELDMAP_ENABLE"
  export PATH="$RECON_BIN:$PATH"
}

line_from_manifest() {
  local manifest="$1"
  local line_no=$((SLURM_ARRAY_TASK_ID + 1))
  sed -n "${line_no}p" "$manifest"
}

run_stage() {
  local manifest="$INPUT_DIR"
  local stage="$STAGE_NAME"
  local out_dir="$OUT_DIR"

  if [[ -z "$stage" || -z "$out_dir" || -z "$manifest" ]]; then
    echo "ERROR: run-stage requires --stage-name, --out-dir, and --input-dir MANIFEST" >&2
    exit 1
  fi
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: run-stage must be executed as a Slurm array job" >&2
    exit 1
  fi
  if [[ ! -s "$manifest" ]]; then
    echo "ERROR: manifest not found or empty: $manifest" >&2
    exit 1
  fi

  setup_runtime_env
  mkdir -p "$out_dir"

  local input_file base clean output_file
  input_file="$(line_from_manifest "$manifest")"
  if [[ -z "$input_file" ]]; then
    echo "ERROR: empty input file at task ${SLURM_ARRAY_TASK_ID} in $manifest" >&2
    exit 1
  fi

  base="$(basename "$input_file" .root)"
  echo "========================================"
  echo "Stage:          $stage"
  echo "Job ID:         ${SLURM_JOB_ID:-none}"
  echo "Array Task ID:  ${SLURM_ARRAY_TASK_ID:-none}"
  echo "Host:           $(hostname)"
  echo "Input:          $input_file"
  echo "Output dir:     $out_dir"
  echo "Start time:     $(date)"
  echo "========================================"

  case "$stage" in
    noise)
      output_file="$out_dir/${base}_noise.root"
      AddStrEcalRandomNoise \
        -O StrawCenterRateMHzPerChannel="$STRAW_RATE" \
        -O StrawFloorRateMHzPerChannel=0 \
        -O StrawSigmaWire="$STRAW_SIGMA" \
        -O EcalOccupancy="$ECAL_OCC" \
        -o "$output_file" \
        "$input_file"
      ;;

    detresp)
      clean="$base"
      clean="${clean%_noise}"
      output_file="$out_dir/${clean}_detresp.root"
      RunSimDetectorResponse \
        -O strecal \
        -o "$output_file" \
        "$input_file"
      ;;

    truthmaker)
      clean="$base"
      clean="${clean%_detresp}"
      clean="${clean%_noise}"
      output_file="$out_dir/${clean}_truthmaker.root"
      RunECALTruthMaker \
        -O OutputFile="$output_file" \
        "$input_file"
      ;;

    *)
      echo "ERROR: unknown run stage: $stage" >&2
      exit 1
      ;;
  esac

  echo "========================================"
  echo "Finished:       $(date)"
  echo "Output:         $output_file"
  echo "========================================"
}

submit_noise_from_dir() {
  local input="$1"
  local manifest pdir jobid
  if ! noise_enabled; then
    echo "Noise step skipped because StrawCenterRateMHzPerChannel=0 and EcalOccupancy=0."
    echo "Use G4 input directly: $input"
    return 0
  fi
  pdir="$(pipeline_dir)"
  manifest="$pdir/noise_input.manifest"
  make_manifest_from_dir "$input" "$manifest"
  jobid="$(submit_stage_array noise "$manifest" "$(noise_dir)")"
  echo "Submitted noise array: $jobid"
}

submit_detresp_from_dir() {
  local input="$1"
  local manifest pdir jobid
  pdir="$(pipeline_dir)"
  manifest="$pdir/detresp_input.manifest"
  make_manifest_from_dir "$input" "$manifest"
  jobid="$(submit_stage_array detresp "$manifest" "$(detresp_dir)")"
  echo "Submitted detresp array: $jobid"
}

submit_truthmaker_from_dir() {
  local input="$1"
  local manifest pdir jobid
  pdir="$(pipeline_dir)"
  manifest="$pdir/truthmaker_input.manifest"
  make_manifest_from_dir "$input" "$manifest"
  jobid="$(submit_stage_array truthmaker "$manifest" "$(truthmaker_dir)")"
  echo "Submitted truthmaker array: $jobid"
}

continue_after_sim() {
  local pdir g4_manifest noise_manifest detresp_manifest noise_job detresp_job truth_job next_input_for_detresp
  mkdirs_common
  pdir="$(pipeline_dir)"
  g4_manifest="$pdir/g4_files.manifest"
  make_manifest_from_dir "$(sim_dir)" "$g4_manifest"

  if noise_enabled; then
    noise_manifest="$pdir/noise_files.manifest"
    make_noise_output_manifest "$g4_manifest" "$noise_manifest" "$(noise_dir)"
    noise_job="$(submit_stage_array noise "$g4_manifest" "$(noise_dir)")"
    next_input_for_detresp="$noise_manifest"
    detresp_manifest="$pdir/detresp_files.manifest"
    make_detresp_output_manifest "$noise_manifest" "$detresp_manifest" "$(detresp_dir)"
    detresp_job="$(submit_stage_array detresp "$next_input_for_detresp" "$(detresp_dir)" "$noise_job")"
  else
    echo "Noise step skipped because StrawCenterRateMHzPerChannel=0 and EcalOccupancy=0."
    detresp_manifest="$pdir/detresp_files.manifest"
    make_detresp_output_manifest "$g4_manifest" "$detresp_manifest" "$(detresp_dir)"
    detresp_job="$(submit_stage_array detresp "$g4_manifest" "$(detresp_dir)")"
  fi

  truth_job="$(submit_stage_array truthmaker "$detresp_manifest" "$(truthmaker_dir)" "$detresp_job")"

  echo "Submitted downstream chain:"
  if [[ -n "${noise_job:-}" ]]; then
    echo "  noise:      $noise_job"
  fi
  echo "  detresp:    $detresp_job"
  echo "  truthmaker: $truth_job"
}

submit_all() {
  require_self_path
  local sim_job cont_job ldir
  sim_job="$(submit_sim)"
  ldir="$(log_dir_for_stage continue-after-sim)"
  mkdir -p "$ldir"
  cont_job="$(run_sbatch \
    --dependency="afterok:${sim_job}" \
    --job-name="ecal_continue_$(label_name)" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --cpus-per-task="1" \
    --mem="1G" \
    --time="01:00:00" \
    --output="$ldir/%x_%j.out" \
    --error="$ldir/%x_%j.err" \
    "$SELF_PATH" \
    --step continue-after-sim \
    --base-dir "$BASE_DIR" \
    --simsetup "$SIMSETUP" \
    --noisesetup "$NOISESETUP" \
    --root-glob "$ROOT_GLOB" \
    --submit-sbatch "$SUBMIT_SBATCH" \
    --cfg "$CFG_FILE" \
    --run-number "$RUN_NUMBER" \
    --sim-array "$SIM_ARRAY" \
    --straw-rate "$STRAW_RATE" \
    --straw-sigma "$STRAW_SIGMA" \
    --ecal-occ "$ECAL_OCC" \
    --partition "$PARTITION" \
    --account "$ACCOUNT" \
    --max-parallel "$MAX_PARALLEL" \
    --mem "$MEM" \
    --time "$TIME" \
    --cpus-per-task "$CPUS_PER_TASK" \
    --icedust-setup "$ICEDUST_SETUP" \
    --fieldmap-enable "$FIELDMAP_ENABLE" \
    --recon-bin "$RECON_BIN" \
    --self-path "$SELF_PATH")"

  echo "Submitted simulation array:     $sim_job"
  echo "Submitted continuation job:     $cont_job  afterok:$sim_job"
  echo "G4 output dir:                  $(sim_dir)"
  if noise_enabled; then
    echo "Noise output dir:               $(noise_dir)"
  else
    echo "Noise step:                     skipped"
  fi
  echo "Detector response output dir:   $(detresp_dir)"
  echo "Truthmaker output dir:          $(truthmaker_dir)"
}

main() {
  # run-stage has two extra options not accepted by normal parse_args.
  if [[ " ${*:-} " == *" --step run-stage "* ]]; then
    parse_run_stage_extra_args "$@"
  else
    parse_args "$@"
  fi

  case "$STEP" in
    all)
      submit_all
      ;;
    sim)
      echo "Submitted simulation array: $(submit_sim)"
      echo "G4 output dir: $(sim_dir)"
      ;;
    noise)
      if [[ -z "$INPUT_DIR" ]]; then
        INPUT_DIR="$(sim_dir)"
      fi
      submit_noise_from_dir "$INPUT_DIR"
      ;;
    detresp)
      if [[ -z "$INPUT_DIR" ]]; then
        if noise_enabled; then
          INPUT_DIR="$(noise_dir)"
        else
          INPUT_DIR="$(sim_dir)"
        fi
      fi
      submit_detresp_from_dir "$INPUT_DIR"
      ;;
    truthmaker)
      if [[ -z "$INPUT_DIR" ]]; then
        INPUT_DIR="$(detresp_dir)"
      fi
      submit_truthmaker_from_dir "$INPUT_DIR"
      ;;
    continue-after-sim)
      continue_after_sim
      ;;
    run-stage)
      run_stage
      ;;
    *)
      echo "ERROR: unknown --step: $STEP" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
