#!/bin/bash
#SBATCH --job-name=particle_gun
#SBATCH --partition=geant
#SBATCH --account=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ============================================================
# User settings, all can be overridden by sbatch --export=ALL,...
# ============================================================

BASE_CFG="${BASE_CFG:-${SLURM_SUBMIT_DIR}/particle_gun.cfg}"
JOB_OUT_DIR="${JOB_OUT_DIR:-${SLURM_SUBMIT_DIR}/ReconData/gun_raw}"
WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR}}"

RUN_NUMBER="${RUN_NUMBER:-1}"
SUBRUN="${SLURM_ARRAY_TASK_ID:-1}"
INPUT_FILE="${INPUT_FILE:-}"
RANDOM_SEED="${RANDOM_SEED:-$((RUN_NUMBER * 10000 + SUBRUN))}"

ICEDUST_ENABLE="${ICEDUST_ENABLE:-/home/comet/davidz/icedust/ICEDUST_install/setup.sh}"
ICEDUST_FIELDMAP_ENABLE="${ICEDUST_FIELDMAP_ENABLE:-/home/comet/davidz/icedust/ICEDUST_fieldmaps/240700_Phase-I_Opera_10mmGrid_4Regions_PCS4p4T_TSso2p7T_TSdp44mT/enable.sh}"
RECON_BIN="${RECON_BIN:-/home/comet/davidz/ReconDev/build/bin}"
ICEDUST_RUN_CMD="${ICEDUST_RUN_CMD:-RunIcedustControl}"

# ============================================================
# Environment
# ============================================================

if [[ ! -f "${ICEDUST_ENABLE}" ]]; then
  echo "ERROR: ICEDUST enable script not found: ${ICEDUST_ENABLE}" >&2
  exit 1
fi
if [[ ! -f "${ICEDUST_FIELDMAP_ENABLE}" ]]; then
  echo "ERROR: fieldmap enable script not found: ${ICEDUST_FIELDMAP_ENABLE}" >&2
  exit 1
fi

source "${ICEDUST_ENABLE}"
source "${ICEDUST_FIELDMAP_ENABLE}"
export PATH="${RECON_BIN}:${PATH}"

mkdir -p "${JOB_OUT_DIR}" "${JOB_OUT_DIR}/logs"

# ============================================================
# Variables expanded by particle_gun.cfg
# ============================================================

export SH_OUTPUT="${JOB_OUT_DIR}"
export SH_INPUT="${INPUT_FILE}"
export SH_RUN="${RUN_NUMBER}"
export SH_SUBRUN="${SUBRUN}"
export SH_RANDOM_SEED="${RANDOM_SEED}"

# ============================================================
# Print job info
# ============================================================

echo "========================================"
echo "Job ID:              ${SLURM_JOB_ID:-none}"
echo "Array Job ID:        ${SLURM_ARRAY_JOB_ID:-none}"
echo "Array Task ID:       ${SLURM_ARRAY_TASK_ID:-none}"
echo "Host:                $(hostname)"
echo "User:                $(whoami)"
echo "Submit dir:          ${SLURM_SUBMIT_DIR}"
echo "Work dir:            ${WORK_DIR}"
echo "Run number:          ${SH_RUN}"
echo "Subrun:              ${SH_SUBRUN}"
echo "Random seed:         ${SH_RANDOM_SEED}"
echo "Output dir:          ${SH_OUTPUT}"
echo "Input file:          ${SH_INPUT}"
echo "Config:              ${BASE_CFG}"
echo "ICEDUST_ENABLE:      ${ICEDUST_ENABLE}"
echo "FIELDMAP_ENABLE:     ${ICEDUST_FIELDMAP_ENABLE}"
echo "RECON_BIN:           ${RECON_BIN}"
echo "Run command:         ${ICEDUST_RUN_CMD}"
echo "Start time:          $(date)"
echo "========================================"

# ============================================================
# Run GEANT4 / ICEDUST
# ============================================================

cd "${WORK_DIR}"
"${ICEDUST_RUN_CMD}" -c "${BASE_CFG}"

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
