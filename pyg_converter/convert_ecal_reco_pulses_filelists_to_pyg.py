#!/usr/bin/env python3
"""
convert_ecal_reco_pulses_filelists_to_pyg.py

Convert RunEcalTruthMaker ROOT output into PyTorch Geometric datasets,
using explicit train / validation / test file lists.

For legacy classification, this version builds one PyG graph per dynamic ECAL seed window.
For regression steps 1/2 it builds one hierarchical PyG graph per ROOT event:
  pulse nodes -> seed-window embeddings -> event-level window GNN.

Dynamic seed-window rule:
  1. Sort all pulses in one event by fitAmplitude from high to low.
  2. Take the highest remaining pulse as a primary seed if amp > 10 mV.
  3. Add all remaining pulses inside |y-y_seed| < 35 mm and |z-z_seed| < 35 mm.
  4. Pulses already inside that first 7x7 window with amp > 10 mV become
     subsequent seeds.  Each subsequent seed expands the same graph once with
     another 7x7 window.  Newly added pulses are not recursively expanded, so the
     maximum footprint is roughly 13x13 crystals.
  5. Remove all assigned pulses from the global seed list.  Only windows with
     total amplitude > 80 mV are written as graphs.

Regression node input is source-specific and 8-dimensional:
  truth mode: [y_abs, z_abs, t_abs, dy_window, dz_window, dt_window, logE, E/Ewindow]
  reco mode : [y_abs, z_abs, t_abs, dy_window, dz_window, dt_window, logEproxy, Eproxy/Ewindow]
The truth/reco feature builders are deliberately separate so waveform-specific reco
features can be extended later without changing truth-mode code.

Per-node raw quantities are also saved for later reconstruction/validation:
  data.node_y, data.node_z, data.node_t, data.node_amp, data.node_area,
  data.cluster_id

Per-truth-cluster quantities are read from ecalTruthClusters and copied into
every graph produced from that event.  In particular, raw t0 and its position
belong to the exact (primaryId, timeSlice), not to the whole primary:
  data.truth_cluster_id, data.truth_cluster_primary_id,
  data.truth_cluster_time_slice, data.truth_cluster_energy,
  data.truth_cluster_t0, data.truth_cluster_t0_x,
  data.truth_cluster_t0_y, data.truth_cluster_t0_z

In truth mode, one reco pulse/digit may expand to several truth nodes, one for
each accepted cluster component in that digit.  This preserves overlap and
delayed components while dynamic seed windows are still formed once per pulse.

Edges are complete within the seed window by default.  The original 6 edge
features are retained, while data.edge_weight stores an inverse-distance weight
used by the GNN message passing.

Regression modes:
  --reconstruction-step 1
    Hierarchical ECAL event -> residual timing/position + direction + prompt energy:
      dt = tmin(prompt window) - t0
      dy = yc(prompt window) - y_inc
      dz = zc(prompt window) - z_inc
    followed by direction, E_cluster and E_most-significant-crystal.
    In reco graph mode t0 is first put on the waveform/reconstruction clock.

  --reconstruction-step 2
    ECAL graph + track entry position/direction + PDG -> E_incident.
    No track momentum magnitude/components are provided.  Until a reconstructed
    Straw-track tree is merged, the constructor stores truth ECAL-entry position
    and direction as an ideal-track training proxy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import uproot
    import awkward as ak
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'uproot' or 'awkward'. Install with: pip install uproot awkward"
    ) from exc

try:
    from torch_geometric.data import Data
except ImportError as exc:
    raise SystemExit(
        "Missing Python package 'torch_geometric'. Install PyTorch Geometric first."
    ) from exc


RAW_FEATURE_NAMES = [
    "y",
    "z",
    "fitPeakTimeNs",
    "fitAmplitude",
    "fitArea",
]

RECO_MODEL_FEATURE_NAMES = [
    "y_abs_scaled",
    "z_abs_scaled",
    "t_abs_norm",
    "dy_from_window_centroid_scaled",
    "dz_from_window_centroid_scaled",
    "dt_from_window_tmin_scaled",
    "log1p_reco_energy_proxy",
    "reco_energy_fraction_in_window",
]

TRUTH_MODEL_FEATURE_NAMES = [
    "y_abs_scaled",
    "z_abs_scaled",
    "truth_t_abs_norm",
    "dy_from_window_centroid_scaled",
    "dz_from_window_centroid_scaled",
    "dt_from_window_tmin_scaled",
    "log1p_truth_energy_mev",
    "truth_energy_fraction_in_window",
]
MODEL_FEATURE_NAMES = RECO_MODEL_FEATURE_NAMES

TRAIN_STD_FEATURE_NAMES = [
    "fitPeakTimeNs",
    "log1p_fitAmplitude",
    "log1p_fitArea",
]

EDGE_FEATURE_NAMES = [
    "dy_over_R",
    "dz_over_R",
    "dr_over_R",
    "dt_scaled",
    "dlogAmp",
    "dlogArea",
]

# Regression-only dataset modes. In multi-object mode data.y is [K,D], one
# target row per represented primary object in the event. Node/edge classifier
# labels are removed from saved graphs for step 1/2.
STEP1_TARGET_NAMES = [
    "dt_prompt_min_minus_t0_ns",
    "dy_prompt_centroid_minus_incident_y_mm",
    "dz_prompt_centroid_minus_incident_z_mm",
    "incident_dir_x",
    "incident_dir_y",
    "incident_dir_z",
    "cluster_energy_mev",
    "most_significant_crystal_energy_mev",
]

# Hierarchical regression event structure.  window_x contains only detector-
# observable summaries derived from the pulse-node model inputs; truth labels
# such as window_role are stored separately and are never model inputs.
WINDOW_FEATURE_NAMES = [
    "centroid_y_scaled",
    "centroid_z_scaled",
    "tmin_norm",
    "tmean_norm",
    "log1p_Esum",
    "log1p_Emax",
    "log1p_n_nodes",
    "sigmaY_scaled",
    "sigmaZ_scaled",
    "sigmaYZ_scaled",
    "Emax_over_Esum",
    "max_hit_dy_scaled",
    "max_hit_dz_scaled",
]
WINDOW_ROLE_NAMES = {0: "other_or_noise", 1: "prompt_any_object", 2: "nonprompt_truth_activity"}

STEP2_TRACK_FEATURE_NAMES = [
    "track_entry_x_mm",
    "track_entry_y_mm",
    "track_entry_z_mm",
    "track_dir_x",
    "track_dir_y",
    "track_dir_z",
]

STEP2_TARGET_NAMES = ["incident_energy_mev"]

REQUIRED_BRANCHES = [
    "event",
    "pulseIndex",
    "digitIndex",
    "channel",
    "sequenceId",
    "blockId",
    "crystalId",
    "y",
    "z",
    "fitPeakTimeNs",
    "fitAmplitude",
    "fitArea",
    "isSignal",
    "isAmbiguous",
    "clusterId",
]

# IECALGeom returns (1e10, 1e10, 1e10) for failed lookups. This generous
# sentinel bound is not a detector fiducial cut; IDs must also be valid.
GEOMETRY_COORD_LIMIT_MM = 1.0e9


def valid_geometry_mask(y, z, channel=None, sequence=None, block=None, crystal=None, x=None):
    valid = np.isfinite(y) & np.isfinite(z) & (abs(y) < GEOMETRY_COORD_LIMIT_MM) & (abs(z) < GEOMETRY_COORD_LIMIT_MM)
    if x is not None:
        valid = valid & np.isfinite(x) & (abs(x) < GEOMETRY_COORD_LIMIT_MM)
    if channel is not None:
        valid = valid & (channel > 0) & (((channel >> 25) & 63) == 4)
    for ids in (sequence, block, crystal):
        if ids is not None:
            valid = valid & (ids >= 0)
    return valid


def sanitize_reco_geometry(arrays, reconstruction_step="legacy", context="", digit_arrays=None, digit_branch_map=None):
    """Filter aligned pulse fields before seeding or normalization.

    Old ROOT cluster targets can contain contributions from invalid digits.
    For regression, quarantine the whole affected event instead of retaining
    those stale targets. Event entries and original pulse/digit IDs stay intact.
    Regenerated truthmaker output rebuilds targets after removing bad digits.
    """
    valid = valid_geometry_mask(arrays["y"], arrays["z"], arrays["channel"],
                                arrays["sequenceId"], arrays["blockId"], arrays["crystalId"])
    bad_events = ak.any(~valid, axis=1)
    n_bad = int(ak.sum(~valid))
    n_bad_digits = 0
    if digit_arrays is not None:
        def field(name):
            branch = digit_branch_map.get(name)
            return digit_arrays[branch][:len(bad_events)] if branch else None
        dv = valid_geometry_mask(field("y"), field("z"), field("channel"),
                                 field("sequence_id"), field("block_id"), field("crystal_id"), field("x"))
        n_bad_digits = int(ak.sum(~dv))
        bad_events = bad_events | ak.any(~dv, axis=1)
    n_events = int(ak.sum(bad_events))
    if n_events:
        print(f"[geometry] {context}: invalid_pulses={n_bad} invalid_digits={n_bad_digits} "
              f"affected_events={n_events} policy={'reject_event' if reconstruction_step != 'legacy' else 'drop_pulse'}", flush=True)
        for i in ak.to_list(ak.local_index(bad_events)[bad_events])[:10]:
            print(f"  [geometry] entry={i} event={ak.to_list(arrays['event'][i])}", flush=True)
    if reconstruction_step != "legacy":
        valid = valid & ~bad_events
    fields = arrays.keys() if isinstance(arrays, dict) else arrays.fields
    return {name: arrays[name] if name == "event" else arrays[name][valid] for name in fields}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ecalRecoPulses ROOT trees into dynamic seed-window PyG .pt "
            "files using explicit train/val/test file lists. One ROOT file -> one .pt file."
        )
    )
    parser.add_argument("--train-list", required=True, help="Training ROOT file list.")
    parser.add_argument("--val-list", required=True, help="Validation ROOT file list.")
    parser.add_argument("--test-list", required=True, help="Test ROOT file list.")
    parser.add_argument("-o", "--output-dir", default="ecal_pyg_dataset")
    parser.add_argument("--tree", default="ecalRecoPulses")
    parser.add_argument(
        "--graph-source",
        choices=["reco", "truth"],
        default="reco",
        help="reco: use ecalRecoPulses; truth: build nodes directly from ecalTruthDigits contributors.",
    )
    parser.add_argument(
        "--truth-mode",
        action="store_true",
        help="Shortcut for --graph-source truth.",
    )
    parser.add_argument("--truth-digits-tree", default="ecalTruthDigits")
    parser.add_argument("--truth-clusters-tree", default="ecalTruthClusters")
    parser.add_argument(
        "--reconstruction-step",
        choices=["legacy", "1", "2"],
        default="legacy",
        help=(
            "Dataset task: legacy keeps node/edge classification labels; "
            "1 stores hierarchical event regression targets "
            "[incident time, incident y/z, direction, E_cluster, E_max crystal]; "
            "2 stores E_incident as the target and only track entry position + "
            "direction plus PDG as auxiliary tracking/PID inputs. No track momentum "
            "is stored in the step-2 input tensor."
        ),
    )
    parser.add_argument(
        "--truth-energy",
        choices=["deposited", "saturated"],
        default="deposited",
        help="Raw-hit energy used as node energy in truth mode.",
    )
    parser.add_argument(
        "--truth-label-fallback-primary",
        action="store_true",
        help="If ecalTruthClusters has no rawHitKey vectors, label by accepted cluster primaryId as a fallback.",
    )
    parser.add_argument(
        "--truth-time-mode",
        choices=["energy-weighted", "earliest", "dominant"],
        default="earliest",
        help=(
            "How to collapse multiple truth contributors from the same cluster component "
            "in one digit/pulse to one node time. Default: earliest, so cluster t0 remains "
            "the earliest selected truth hit."
        ),
    )
    parser.add_argument(
        "--truth-missing-time-fallback",
        choices=["reco", "zero"],
        default="reco",
        help=(
            "Fallback node time for reco pulses whose digit has no truth contributor. "
            "Default uses fitPeakTimeNs so electronics/random-noise pulses keep their measured time."
        ),
    )

    parser.add_argument(
        "--truth-node-energy-threshold-mev",
        type=float,
        default=0.0,
        help=(
            "Only used in --graph-source truth. After reco-amplitude seed windows are built, "
            "keep only nodes with truth energy >= this threshold. Default: 1 MeV."
        ),
    )

    parser.add_argument("--y-scale-mm", type=float, default=500.0)
    parser.add_argument("--z-scale-mm", type=float, default=500.0)
    parser.add_argument(
        "--z-offset-mm",
        type=float,
        default=7650.0,
        help="z feature is (z - z_offset_mm) / z_scale_mm. Default: 7650 mm.",
    )

    parser.add_argument(
        "--edge-radius-mm",
        type=float,
        default=100.0,
        help="Normalization radius R used in edge features. In dynamic mode it is not a cut.",
    )
    parser.add_argument(
        "--edge-time-ns",
        type=float,
        default=-1.0,
        help="Optional time cut for legacy non-dynamic graphs. Dynamic graphs ignore this cut by default.",
    )
    parser.add_argument("--edge-weight-scale-mm", type=float, default=10.0)
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--max-events-per-file", type=int, default=-1)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--self-loops", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--store-raw", action="store_true", help="Kept for compatibility; raw node tensors are now always stored.")

    parser.add_argument(
        "--disable-dynamic-seeding",
        action="store_true",
        help="Use the old one-whole-event graph construction instead of dynamic seed windows.",
    )
    parser.add_argument("--seed-threshold-mv", type=float, default=10.0)
    parser.add_argument("--seed-window-half-mm", type=float, default=35.0)
    parser.add_argument("--seed-min-total-amplitude-mv", type=float, default=80.0)
    parser.add_argument(
        "--amp-to-energy",
        type=float,
        default=17.96,
        help="Amplitude/energy calibration factor. Cluster energy = total amplitude / amp_to_energy.",
    )
    parser.add_argument(
        "--normalization-json",
        default="",
        help=(
            "Load fixed normalization mean/std from a JSON file instead of "
            "scanning the complete training dataset."
        ),
    )
    return parser.parse_args()


def read_file_list(list_path: str) -> List[str]:
    path = Path(list_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File list does not exist: {path}")

    files: List[str] = []
    base_dir = path.parent
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue
            candidate = Path(line).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            matches = sorted(glob.glob(str(candidate)))
            if not matches:
                raise FileNotFoundError(
                    f"Path or glob from {path} does not match anything: {line}"
                )
            files.extend([str(Path(m).resolve()) for m in matches])

    seen = set()
    unique_files: List[str] = []
    for name in files:
        if name not in seen:
            unique_files.append(name)
            seen.add(name)
    if not unique_files:
        raise RuntimeError(f"No ROOT files found in list: {path}")
    return unique_files


def check_tree_branches(tree: uproot.TTree, required: Sequence[str]) -> None:
    available = set(tree.keys())
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(
            f"Tree {tree.name} is missing required branches:\n  " + "\n  ".join(missing)
        )

def load_normalization_stats(
    filename: str,
    graph_source: str,
) -> Dict[str, object]:
    path = Path(filename).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Normalization JSON does not exist: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    stats = payload.get("normalization", payload)

    if not isinstance(stats, dict):
        raise RuntimeError(
            f"Invalid normalization object in {path}"
        )

    if "mean" not in stats or "std" not in stats:
        raise RuntimeError(
            f"Normalization JSON must contain mean and std: {path}"
        )

    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)

    if mean.shape != (3,) or std.shape != (3,):
        raise RuntimeError(
            "Normalization mean/std must each contain exactly 3 values"
        )

    if not np.all(np.isfinite(mean)):
        raise RuntimeError("Normalization mean contains non-finite values")

    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise RuntimeError(
            "Normalization std must contain finite positive values"
        )

    expected_names = (
        ["truthTimeNs", "log1p_truthEnergy", "zeroPadding"]
        if graph_source == "truth"
        else ["fitPeakTimeNs", "log1p_fitAmplitude", "log1p_fitArea"]
    )

    feature_names = stats.get("feature_names", expected_names)
    if list(feature_names) != expected_names:
        raise RuntimeError(
            "Normalization feature mismatch:\n"
            f"  expected: {expected_names}\n"
            f"  found   : {feature_names}"
        )

    return {
        "method": stats.get(
            "method",
            f"loaded_fixed_normalization_from_{path.name}",
        ),
        "feature_names": expected_names,
        "n_train_nodes_for_stats": int(
            stats.get("n_train_nodes_for_stats", -1)
        ),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "source_file": str(path),
    }


def as_numpy_1d(x) -> np.ndarray:
    return np.asarray(x)


def safe_log1p_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def build_raw_node_features(arrays: Dict[str, object], entry: int) -> np.ndarray:
    """Return [y, z, t, raw amplitude, raw area] for all pulses in one event."""
    y = as_numpy_1d(arrays["y"][entry]).astype(np.float32)
    z = as_numpy_1d(arrays["z"][entry]).astype(np.float32)
    t = as_numpy_1d(arrays["fitPeakTimeNs"][entry]).astype(np.float32)
    amp = as_numpy_1d(arrays["fitAmplitude"][entry]).astype(np.float32)
    area = as_numpy_1d(arrays["fitArea"][entry]).astype(np.float32)
    return np.stack([y, z, t, amp, area], axis=1).astype(np.float32)


def train_std_features_from_raw(x_raw: np.ndarray) -> np.ndarray:
    """Return [t, log1p_amp, log1p_area] used for train-set normalization."""
    t = x_raw[:, 2].astype(np.float32)
    log_amp = safe_log1p_nonnegative(x_raw[:, 3]).astype(np.float32)
    log_area = safe_log1p_nonnegative(x_raw[:, 4]).astype(np.float32)
    return np.stack([t, log_amp, log_area], axis=1).astype(np.float32)


def make_model_node_features(
    x_raw: np.ndarray,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
) -> np.ndarray:
    if y_scale_mm <= 0.0:
        raise ValueError("y_scale_mm must be positive")
    if z_scale_mm <= 0.0:
        raise ValueError("z_scale_mm must be positive")

    x = np.empty_like(x_raw, dtype=np.float32)
    x[:, 0] = x_raw[:, 0] / np.float32(y_scale_mm)
    x[:, 1] = (x_raw[:, 1] - np.float32(z_offset_mm)) / np.float32(z_scale_mm)
    x[:, 2:] = train_std_features_from_raw(x_raw)

    if normalize:
        if stats is None:
            raise RuntimeError("normalize=True but stats is None")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        x[:, 2:5] = (x[:, 2:5] - mean) / std

    return x.astype(np.float32)



def _time_mean_std(stats: Optional[Dict[str, object]]) -> Tuple[float, float]:
    if stats is None:
        return 0.0, 1.0
    mean = np.asarray(stats.get("mean", [0.0]), dtype=np.float64)
    std = np.asarray(stats.get("std", [1.0]), dtype=np.float64)
    m = float(mean[0]) if mean.size else 0.0
    sd = float(std[0]) if std.size else 1.0
    if not np.isfinite(sd) or sd < 1.0e-12:
        sd = 1.0
    return m, sd


def _truth_window_physical_moments(
    y: np.ndarray, z: np.ndarray, t: np.ndarray, energy: np.ndarray
) -> Dict[str, float]:
    """TRUTH-ONLY physical window reference/moment calculation.

    The truth path intentionally owns its complete feature construction so it
    can evolve independently of waveform/reco features.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    e = np.where(np.isfinite(e), np.maximum(e, 0.0), 0.0)
    if float(e.sum()) <= 0.0:
        e = np.ones_like(y, dtype=np.float64)
    esum = float(e.sum())
    yc = float(np.sum(e * y) / esum)
    zc = float(np.sum(e * z) / esum)
    finite_t = t[np.isfinite(t)]
    tmin = float(np.min(finite_t)) if finite_t.size else 0.0
    tsafe = np.where(np.isfinite(t), t, tmin)
    tmean = float(np.sum(e * tsafe) / esum)
    dy = y - yc
    dz = z - zc
    sigma_y = math.sqrt(max(float(np.sum(e * dy * dy) / esum), 0.0))
    sigma_z = math.sqrt(max(float(np.sum(e * dz * dz) / esum), 0.0))
    sigma_yz = float(np.sum(e * dy * dz) / esum)
    imax = int(np.argmax(e)) if e.size else 0
    emax = float(e[imax]) if e.size else 0.0
    return {
        "yc_mm": yc,
        "zc_mm": zc,
        "tmin_ns": tmin,
        "tmean_ns": tmean,
        "esum_mev": esum,
        "emax_mev": emax,
        "sigma_y_mm": sigma_y,
        "sigma_z_mm": sigma_z,
        "sigma_yz_mm2": sigma_yz,
        "emax_over_esum": emax / esum if esum > 0.0 else 0.0,
        "max_hit_dy_mm": float(y[imax] - yc) if y.size else 0.0,
        "max_hit_dz_mm": float(z[imax] - zc) if z.size else 0.0,
    }


def _reco_window_physical_moments(
    y: np.ndarray, z: np.ndarray, t: np.ndarray, energy_proxy: np.ndarray
) -> Dict[str, float]:
    """RECO-ONLY physical window reference/moment calculation.

    `energy_proxy` currently comes from fitAmplitude/amp_to_energy.  This code
    is deliberately separate from truth mode so later waveform observables or
    a different reco energy estimator can be introduced locally.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(energy_proxy, dtype=np.float64)
    e = np.where(np.isfinite(e), np.maximum(e, 0.0), 0.0)
    if float(e.sum()) <= 0.0:
        e = np.ones_like(y, dtype=np.float64)
    esum = float(e.sum())
    yc = float(np.sum(e * y) / esum)
    zc = float(np.sum(e * z) / esum)
    finite_t = t[np.isfinite(t)]
    tmin = float(np.min(finite_t)) if finite_t.size else 0.0
    tsafe = np.where(np.isfinite(t), t, tmin)
    tmean = float(np.sum(e * tsafe) / esum)
    dy = y - yc
    dz = z - zc
    sigma_y = math.sqrt(max(float(np.sum(e * dy * dy) / esum), 0.0))
    sigma_z = math.sqrt(max(float(np.sum(e * dz * dz) / esum), 0.0))
    sigma_yz = float(np.sum(e * dy * dz) / esum)
    imax = int(np.argmax(e)) if e.size else 0
    emax = float(e[imax]) if e.size else 0.0
    return {
        "yc_mm": yc,
        "zc_mm": zc,
        "tmin_ns": tmin,
        "tmean_ns": tmean,
        "esum_mev": esum,
        "emax_mev": emax,
        "sigma_y_mm": sigma_y,
        "sigma_z_mm": sigma_z,
        "sigma_yz_mm2": sigma_yz,
        "emax_over_esum": emax / esum if esum > 0.0 else 0.0,
        "max_hit_dy_mm": float(y[imax] - yc) if y.size else 0.0,
        "max_hit_dz_mm": float(z[imax] - zc) if z.size else 0.0,
    }


def _legacy_window_physical_moments(
    y: np.ndarray, z: np.ndarray, t: np.ndarray, weight: np.ndarray
) -> Dict[str, float]:
    """Legacy-only compatibility helper; not used by Step 1/2 inputs."""
    # Keep legacy behavior independent of the new source-specific feature paths.
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    e = np.asarray(weight, dtype=np.float64)
    e = np.where(np.isfinite(e), np.maximum(e, 0.0), 0.0)
    if float(e.sum()) <= 0.0:
        e = np.ones_like(y, dtype=np.float64)
    esum = float(e.sum())
    yc = float(np.sum(e*y)/esum); zc = float(np.sum(e*z)/esum)
    finite_t=t[np.isfinite(t)]; tmin=float(np.min(finite_t)) if finite_t.size else 0.0
    tsafe=np.where(np.isfinite(t),t,tmin); tmean=float(np.sum(e*tsafe)/esum)
    dy=y-yc; dz=z-zc
    sy=math.sqrt(max(float(np.sum(e*dy*dy)/esum),0.0)); sz=math.sqrt(max(float(np.sum(e*dz*dz)/esum),0.0))
    syz=float(np.sum(e*dy*dz)/esum); imax=int(np.argmax(e)) if e.size else 0; emax=float(e[imax]) if e.size else 0.0
    return {"yc_mm":yc,"zc_mm":zc,"tmin_ns":tmin,"tmean_ns":tmean,"esum_mev":esum,"emax_mev":emax,
            "sigma_y_mm":sy,"sigma_z_mm":sz,"sigma_yz_mm2":syz,"emax_over_esum":emax/esum if esum>0 else 0.0,
            "max_hit_dy_mm":float(y[imax]-yc) if y.size else 0.0,"max_hit_dz_mm":float(z[imax]-zc) if z.size else 0.0}

def make_truth_model_node_features(
    x_raw_np: np.ndarray,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    relative_time_scale_ns: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """TRUTH-ONLY node feature path.

    x_raw_np = [y_mm, z_mm, truth_time_ns, truth_energy_mev, 0].
    No reco waveform quantity is referenced here.
    """
    y = x_raw_np[:, 0].astype(np.float64)
    z = x_raw_np[:, 1].astype(np.float64)
    t = x_raw_np[:, 2].astype(np.float64)
    energy = np.maximum(x_raw_np[:, 3].astype(np.float64), 0.0)
    ref = _truth_window_physical_moments(y, z, t, energy)
    tmean0, tstd0 = _time_mean_std(stats) if normalize else (0.0, 1.0)
    rts = max(float(relative_time_scale_ns), 1.0e-6)
    esum = max(float(np.sum(energy)), 1.0e-12)
    x = np.stack([
        y / float(y_scale_mm),
        (z - float(z_offset_mm)) / float(z_scale_mm),
        (t - tmean0) / tstd0,
        (y - ref["yc_mm"]) / float(y_scale_mm),
        (z - ref["zc_mm"]) / float(z_scale_mm),
        (t - ref["tmin_ns"]) / rts,
        np.log1p(energy),
        energy / esum,
    ], axis=1)
    return x.astype(np.float32), ref


def make_reco_model_node_features(
    x_raw_np: np.ndarray,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    relative_time_scale_ns: float,
    amp_to_energy: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """RECO-ONLY node feature path.

    Current reco model uses fitPeakTimeNs and an amplitude-derived energy proxy.
    This function is intentionally independent of the truth feature builder so
    fit width/chi2/template parameters can be appended later without touching
    truth mode.
    """
    y = x_raw_np[:, 0].astype(np.float64)
    z = x_raw_np[:, 1].astype(np.float64)
    t = x_raw_np[:, 2].astype(np.float64)
    amp = np.maximum(x_raw_np[:, 3].astype(np.float64), 0.0)
    energy = amp / max(float(amp_to_energy), 1.0e-12)
    ref = _reco_window_physical_moments(y, z, t, energy)
    tmean0, tstd0 = _time_mean_std(stats) if normalize else (0.0, 1.0)
    rts = max(float(relative_time_scale_ns), 1.0e-6)
    esum = max(float(np.sum(energy)), 1.0e-12)
    x = np.stack([
        y / float(y_scale_mm),
        (z - float(z_offset_mm)) / float(z_scale_mm),
        (t - tmean0) / tstd0,
        (y - ref["yc_mm"]) / float(y_scale_mm),
        (z - ref["zc_mm"]) / float(z_scale_mm),
        (t - ref["tmin_ns"]) / rts,
        np.log1p(energy),
        energy / esum,
    ], axis=1)
    return x.astype(np.float32), ref


def make_truth_window_features(ref: Dict[str, float], n_nodes: int, stats, y_scale_mm, z_scale_mm, z_offset_mm) -> np.ndarray:
    """TRUTH-ONLY 13D window feature builder."""
    tm, ts = _time_mean_std(stats)
    yz_scale = max(float(y_scale_mm) * float(z_scale_mm), 1.0e-12)
    return np.asarray([
        ref["yc_mm"] / float(y_scale_mm),
        (ref["zc_mm"] - float(z_offset_mm)) / float(z_scale_mm),
        (ref["tmin_ns"] - tm) / ts,
        (ref["tmean_ns"] - tm) / ts,
        math.log1p(max(ref["esum_mev"], 0.0)),
        math.log1p(max(ref["emax_mev"], 0.0)),
        math.log1p(max(int(n_nodes), 0)),
        ref["sigma_y_mm"] / float(y_scale_mm),
        ref["sigma_z_mm"] / float(z_scale_mm),
        ref["sigma_yz_mm2"] / yz_scale,
        ref["emax_over_esum"],
        ref["max_hit_dy_mm"] / float(y_scale_mm),
        ref["max_hit_dz_mm"] / float(z_scale_mm),
    ], dtype=np.float32)


def make_reco_window_features(ref: Dict[str, float], n_nodes: int, stats, y_scale_mm, z_scale_mm, z_offset_mm) -> np.ndarray:
    """RECO-ONLY 13D window feature builder; independent from truth mode."""
    tm, ts = _time_mean_std(stats)
    yz_scale = max(float(y_scale_mm) * float(z_scale_mm), 1.0e-12)
    return np.asarray([
        ref["yc_mm"] / float(y_scale_mm),
        (ref["zc_mm"] - float(z_offset_mm)) / float(z_scale_mm),
        (ref["tmin_ns"] - tm) / ts,
        (ref["tmean_ns"] - tm) / ts,
        math.log1p(max(ref["esum_mev"], 0.0)),
        math.log1p(max(ref["emax_mev"], 0.0)),
        math.log1p(max(int(n_nodes), 0)),
        ref["sigma_y_mm"] / float(y_scale_mm),
        ref["sigma_z_mm"] / float(z_scale_mm),
        ref["sigma_yz_mm2"] / yz_scale,
        ref["emax_over_esum"],
        ref["max_hit_dy_mm"] / float(y_scale_mm),
        ref["max_hit_dz_mm"] / float(z_scale_mm),
    ], dtype=np.float32)


def _attach_window_reference_fields(data: Data, ref: Dict[str, float], window_feature: np.ndarray) -> None:
    data.window_model_feature = torch.tensor(window_feature, dtype=torch.float32)
    data.window_centroid_y_mm = torch.tensor([ref["yc_mm"]], dtype=torch.float32)
    data.window_centroid_z_mm = torch.tensor([ref["zc_mm"]], dtype=torch.float32)
    data.window_tmin_ns = torch.tensor([ref["tmin_ns"]], dtype=torch.float32)
    data.window_tmean_ns = torch.tensor([ref["tmean_ns"]], dtype=torch.float32)
    data.window_esum_mev = torch.tensor([ref["esum_mev"]], dtype=torch.float32)
    data.window_emax_mev = torch.tensor([ref["emax_mev"]], dtype=torch.float32)
    data.window_sigma_y_mm = torch.tensor([ref["sigma_y_mm"]], dtype=torch.float32)
    data.window_sigma_z_mm = torch.tensor([ref["sigma_z_mm"]], dtype=torch.float32)
    data.window_sigma_yz_mm2 = torch.tensor([ref["sigma_yz_mm2"]], dtype=torch.float32)
    data.window_emax_over_esum = torch.tensor([ref["emax_over_esum"]], dtype=torch.float32)
    data.window_max_hit_dy_mm = torch.tensor([ref["max_hit_dy_mm"]], dtype=torch.float32)
    data.window_max_hit_dz_mm = torch.tensor([ref["max_hit_dz_mm"]], dtype=torch.float32)


def iter_root_events(files: Sequence[str], tree_name: str, branches: Sequence[str], reconstruction_step="legacy"):
    for file_index, filename in enumerate(files):
        print(f"[read] file {file_index + 1}/{len(files)}: {filename}")
        with uproot.open(filename) as f:
            if tree_name not in f:
                raise RuntimeError(f"Tree '{tree_name}' not found in {filename}")
            tree = f[tree_name]
            check_tree_branches(tree, branches)
            arrays = sanitize_reco_geometry(tree.arrays(branches, library="ak"), reconstruction_step, filename)
            for entry in range(tree.num_entries):
                yield file_index, filename, entry, arrays



TRUTH_DIGIT_BRANCH_ALIASES = {
    "event": ["event", "eventId", "eventNumber", "fEventNumber"],
    "digit_index": ["digitIndex", "index", "fIndex"],
    "channel": ["channel", "channelUInt", "fChannelUInt"],
    "sequence_id": ["sequenceId", "seqId", "fSequenceId"],
    "block_id": ["blockId", "fBlockId"],
    "crystal_id": ["crystalId", "fCrystalId"],
    "x": ["x", "fX"],
    "y": ["y", "fY"],
    "z": ["z", "fZ"],
    "contributor_begin": ["contributorBegin", "contributorOffset", "contributorStart", "fContributorBegin"],
    "contributor_count": ["contributorCount", "nContributors", "contributorSize", "fContributorCount"],
    "contributor_raw_hit_key": [
        "contributorRawHitKey", "contributorRawHitKeys", "rawHitKey",
        "fContributorRawHitKey", "fContributorRawHitKeys",
    ],
    "contributor_primary_id": [
        "contributorPrimaryId", "contributorPrimaryIds", "rawHitPrimaryId",
        "fContributorPrimaryId", "fContributorPrimaryIds",
    ],
    "contributor_crystal_id": [
        "contributorCrystalId", "contributorCrystalIds", "rawHitCrystalId",
        "fContributorCrystalId", "fContributorCrystalIds",
    ],
    "contributor_pid": [
        "contributorPID", "contributorPIDs", "contributorPid", "contributorPids",
        "rawHitPID", "rawHitPIDs", "fContributorPID", "fContributorPIDs",
    ],
    "contributor_edep": [
        "contributorEdepMeV", "contributorDepositedEnergyMeV", "rawHitEdepMeV",
        "fContributorEdepMeV",
    ],
    "contributor_saturated_edep": [
        "contributorSaturatedEdepMeV", "contributorSaturatedDepositedEnergyMeV",
        "rawHitSaturatedEdepMeV", "fContributorSaturatedEdepMeV",
    ],
    "contributor_time": [
        "contributorTimeNs", "rawHitTimeNs", "fContributorTimeNs",
    ],
    "contributor_x": ["contributorX", "rawHitX", "fContributorX"],
    "contributor_y": ["contributorY", "rawHitY", "fContributorY"],
    "contributor_z": ["contributorZ", "rawHitZ", "fContributorZ"],
    "contributor_mom": [
        "contributorIncidentMom", "contributorIncidentMomentum", "rawHitIncidentMom",
        "fContributorIncidentMom",
    ],
}

TRUTH_CLUSTER_BRANCH_ALIASES = {
    "event": ["event", "eventId", "eventNumber", "fEventNumber"],
    "cluster_id": ["clusterId", "fClusterId"],
    "primary_id": ["primaryId", "fPrimaryId"],
    "pid": ["pid", "PID", "pdg", "pdgCode", "fPID", "fPDGCode"],
    "time_slice": ["timeSlice", "fTimeSlice"],
    # Current RunECALTruthMaker writes "edepMeV".  Keep old aliases for
    # backward compatibility with earlier ROOT files.
    "total_edep": ["edepMeV", "totalEdepMeV", "fTotalEdepMeV"],
    "incident_mom": ["incidentMom", "fIncidentMom"],
    "incident_energy": ["incidentEnergyMeV", "fIncidentEnergyMeV"],
    "incident_kinetic_energy": ["incidentKineticEnergyMeV", "fIncidentKineticEnergyMeV"],
    "incident_time": ["incidentTimeNs", "fIncidentTimeNs"],
    "incident_x": ["incidentX", "fIncidentX"],
    "incident_y": ["incidentY", "fIncidentY"],
    "incident_z": ["incidentZ", "fIncidentZ"],
    "incident_px": ["incidentPx", "fIncidentPx"],
    "incident_py": ["incidentPy", "fIncidentPy"],
    "incident_pz": ["incidentPz", "fIncidentPz"],
    "incident_dir_x": ["incidentDirX", "fIncidentDirX"],
    "incident_dir_y": ["incidentDirY", "fIncidentDirY"],
    "incident_dir_z": ["incidentDirZ", "fIncidentDirZ"],
    "incident_state_valid": ["incidentStateValid", "fIncidentStateValid"],
    "most_significant_crystal_id": ["mostSignificantCrystalId", "fMostSignificantCrystalId"],
    "most_significant_crystal_edep": ["mostSignificantCrystalEdepMeV", "fMostSignificantCrystalEdepMeV"],
    "most_significant_crystal_raw_edep": ["mostSignificantCrystalRawEdepMeV", "fMostSignificantCrystalRawEdepMeV"],
    "most_significant_crystal_fraction": ["mostSignificantCrystalFraction", "fMostSignificantCrystalFraction"],
    "leakage": ["leakage", "fLeakage"],
    "raw_t0": ["rawT0Ns", "t0Ns", "fRawT0Ns"],
    "raw_t0_x": ["rawT0X", "t0X", "fRawT0X"],
    "raw_t0_y": ["rawT0Y", "t0Y", "fRawT0Y"],
    "raw_t0_z": ["rawT0Z", "t0Z", "fRawT0Z"],
    "raw_hit_begin": ["rawHitBegin", "rawHitOffset", "rawHitStart", "fRawHitBegin"],
    "raw_hit_count": ["rawHitCount", "nRawHits", "rawHitSize", "fRawHitCount"],
    "raw_hit_key": ["rawHitKey", "rawHitKeys", "fRawHitKey", "fRawHitKeys"],
    "raw_hit_primary_id": ["rawHitPrimaryId", "rawHitPrimaryIds", "fRawHitPrimaryId", "fRawHitPrimaryIds"],
    "raw_hit_crystal_id": ["rawHitCrystalId", "rawHitCrystalIds", "fRawHitCrystalId", "fRawHitCrystalIds"],
    "raw_hit_edep": ["rawHitEdepMeV", "fRawHitEdepMeV"],
    "raw_hit_saturated_edep": ["rawHitSaturatedEdepMeV", "fRawHitSaturatedEdepMeV"],
    "raw_hit_time": ["rawHitTimeNs", "fRawHitTimeNs"],
    "raw_hit_x": ["rawHitX", "fRawHitX"],
    "raw_hit_y": ["rawHitY", "fRawHitY"],
    "raw_hit_z": ["rawHitZ", "fRawHitZ"],
}


def regression_required_cluster_logicals(step: str) -> List[str]:
    if step == "1":
        return [
            "cluster_id", "primary_id", "pid", "time_slice", "raw_t0", "total_edep",
            "incident_time", "incident_x", "incident_y", "incident_z",
            "incident_dir_x", "incident_dir_y", "incident_dir_z",
            "incident_state_valid",
            "most_significant_crystal_id", "most_significant_crystal_edep",
            # Reco-mode Step-1 t0 is converted to the waveform/reconstruction
            # time scale using the most-significant crystal raw-hit timing.
            "raw_hit_crystal_id", "raw_hit_time", "raw_hit_saturated_edep",
        ]
    if step == "2":
        return [
            "cluster_id", "primary_id", "pid", "time_slice", "raw_t0", "incident_energy",
            "incident_x", "incident_y", "incident_z",
            "incident_dir_x", "incident_dir_y", "incident_dir_z",
            "incident_state_valid",
        ]
    return ["cluster_id"]


TRUTH_REQUIRED_DIGIT_LOGICAL = [
    "y", "z", "contributor_primary_id", "contributor_time", "contributor_edep",
]


def _available_tree_keys(tree: uproot.TTree) -> set:
    return set(str(k) for k in tree.keys())


def resolve_branch_aliases(
    tree: uproot.TTree,
    aliases: Dict[str, List[str]],
    required: Sequence[str],
    context: str,
) -> Dict[str, str]:
    available = _available_tree_keys(tree)
    resolved: Dict[str, str] = {}
    for logical, candidates in aliases.items():
        for name in candidates:
            if name in available:
                resolved[logical] = name
                break
        if logical in resolved:
            continue
        for name in candidates:
            suffix_matches = sorted(k for k in available if k.endswith("." + name))
            if suffix_matches:
                resolved[logical] = suffix_matches[0]
                break
    missing = [logical for logical in required if logical not in resolved]
    if missing:
        lines = [f"{context} is missing required logical branches:"]
        for logical in missing:
            lines.append(f"  {logical}: tried {aliases.get(logical, [])}")
        lines.append("Available branches include:")
        for name in sorted(available)[:80]:
            lines.append(f"  {name}")
        raise RuntimeError("\n".join(lines))
    return resolved


def _branches_from_map(branch_map: Dict[str, str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for name in branch_map.values():
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _ak_to_list(value):
    return ak.to_list(value)


def _get_event_value(arrays: Dict[str, object], branch_map: Dict[str, str], logical: str, entry: int, default=None):
    name = branch_map.get(logical)
    if name is None:
        return default
    return _ak_to_list(arrays[name][entry])


def _as_1d_list(value, n: Optional[int] = None, default=0):
    if value is None:
        return [default] * int(n or 0)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    if not isinstance(value, list):
        value = list(value)
    if n is not None and len(value) != n:
        if len(value) == 0:
            return [default] * int(n)
        raise RuntimeError(f"Expected vector length {n}, got {len(value)}")
    return value


def _as_nested_list(value, n_outer: int):
    if value is None:
        return [[] for _ in range(n_outer)]
    if not isinstance(value, list):
        value = list(value)
    if len(value) != n_outer:
        raise RuntimeError(f"Expected nested vector outer length {n_outer}, got {len(value)}")
    out = []
    for item in value:
        if item is None:
            out.append([])
        elif isinstance(item, list):
            out.append(item)
        else:
            out.append(list(item))
    return out


def _as_nested_from_flat(value, begin_value, count_value, n_outer: int):
    """Reconstruct per-object lists from flat ROOT vector + begin/count vectors."""
    if value is None:
        return [[] for _ in range(n_outer)]
    flat = _as_1d_list(value, default=0)
    begin = [int(v) for v in _as_1d_list(begin_value, n=n_outer, default=0)]
    count = [int(v) for v in _as_1d_list(count_value, n=n_outer, default=0)]
    out = []
    for i in range(n_outer):
        b = begin[i]
        c = count[i]
        if b < 0 or c < 0 or b + c > len(flat):
            raise RuntimeError(
                f"Invalid flat-vector range for object {i}: begin={b}, count={c}, flat_length={len(flat)}"
            )
        out.append(flat[b:b + c])
    return out


def _get_nested_event_value(
    arrays: Dict[str, object],
    branch_map: Dict[str, str],
    logical: str,
    entry: int,
    n_outer: int,
    begin_logical: str,
    count_logical: str,
    default=None,
):
    """
    Return a list-of-lists for either old nested-vector branches or new flat-vector
    branches with begin/count indexing.
    """
    value = _get_event_value(arrays, branch_map, logical, entry, default=default)
    if value is None:
        return [[] for _ in range(n_outer)]

    begin_name = branch_map.get(begin_logical)
    count_name = branch_map.get(count_logical)
    if begin_name is not None and count_name is not None:
        begin_value = _ak_to_list(arrays[begin_name][entry])
        count_value = _ak_to_list(arrays[count_name][entry])
        return _as_nested_from_flat(value, begin_value, count_value, n_outer)

    return _as_nested_list(value, n_outer)


def _get_digit_nested_event_value(arrays, branch_map, logical, entry, n_digits, default=None):
    return _get_nested_event_value(
        arrays=arrays,
        branch_map=branch_map,
        logical=logical,
        entry=entry,
        n_outer=n_digits,
        begin_logical="contributor_begin",
        count_logical="contributor_count",
        default=default,
    )


def _get_cluster_nested_event_value(arrays, branch_map, logical, entry, n_clusters, default=None):
    return _get_nested_event_value(
        arrays=arrays,
        branch_map=branch_map,
        logical=logical,
        entry=entry,
        n_outer=n_clusters,
        begin_logical="raw_hit_begin",
        count_logical="raw_hit_count",
        default=default,
    )


def _scalar_event_id(arrays: Dict[str, object], branch_map: Dict[str, str], entry: int) -> int:
    value = _get_event_value(arrays, branch_map, "event", entry, default=entry)
    if isinstance(value, list):
        if not value:
            return int(entry)
        return int(value[0])
    return int(value)


def _get_cluster_label_maps(
    cluster_arrays: Optional[Dict[str, object]],
    cluster_branch_map: Dict[str, str],
    entry: int,
    fallback_primary: bool,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    raw_key_to_cluster: Dict[int, int] = {}
    primary_to_cluster: Dict[int, int] = {}
    if cluster_arrays is None:
        return raw_key_to_cluster, primary_to_cluster

    cluster_ids = _as_1d_list(
        _get_event_value(cluster_arrays, cluster_branch_map, "cluster_id", entry, default=[]),
        default=-1,
    )
    n_clusters = len(cluster_ids)

    primary_ids = _as_1d_list(
        _get_event_value(cluster_arrays, cluster_branch_map, "primary_id", entry, default=None),
        n=n_clusters,
        default=-1,
    )
    clusters_by_primary: Dict[int, List[int]] = {}
    for cid, pid in zip(cluster_ids, primary_ids):
        if int(pid) >= 0:
            clusters_by_primary.setdefault(int(pid), []).append(int(cid))
    # A primary-id fallback is safe only when that primary has exactly one
    # accepted time slice.  Multiple slices must be resolved by rawHitKey.
    for pid, cids in clusters_by_primary.items():
        if len(cids) == 1:
            primary_to_cluster[pid] = cids[0]

    raw_keys_value = _get_event_value(cluster_arrays, cluster_branch_map, "raw_hit_key", entry, default=None)
    if raw_keys_value is not None:
        raw_keys_by_cluster = _get_cluster_nested_event_value(
            cluster_arrays, cluster_branch_map, "raw_hit_key", entry, n_clusters, default=None
        )
        for icl, keys in enumerate(raw_keys_by_cluster):
            cid = int(cluster_ids[icl]) if icl < len(cluster_ids) else int(icl)
            for key in keys:
                raw_key_to_cluster[int(key)] = cid

    if not raw_key_to_cluster and not fallback_primary:
        return raw_key_to_cluster, {}
    return raw_key_to_cluster, primary_to_cluster


def build_truth_hit_event(
    digit_arrays: Dict[str, object],
    digit_branch_map: Dict[str, str],
    cluster_arrays: Optional[Dict[str, object]],
    cluster_branch_map: Dict[str, str],
    entry: int,
    truth_energy: str,
    fallback_primary: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], int]:
    y_digits = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "y", entry))
    z_digits = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "z", entry), n=len(y_digits))
    n_digits = len(y_digits)

    digit_index = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "digit_index", entry, None), n=n_digits, default=-1)
    channel = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "channel", entry, None), n=n_digits, default=0)
    sequence_id = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "sequence_id", entry, None), n=n_digits, default=-1)
    block_id = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "block_id", entry, None), n=n_digits, default=-1)
    crystal_id = _as_1d_list(_get_event_value(digit_arrays, digit_branch_map, "crystal_id", entry, None), n=n_digits, default=-1)

    raw_keys_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_raw_hit_key", entry, n_digits, default=None)
    primary_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_primary_id", entry, n_digits)
    time_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_time", entry, n_digits)
    edep_logical = "contributor_edep" if truth_energy == "deposited" else "contributor_saturated_edep"
    if edep_logical not in digit_branch_map:
        edep_logical = "contributor_saturated_edep" if edep_logical == "contributor_edep" else "contributor_edep"
    edep_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, edep_logical, entry, n_digits)

    sat_edep_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_saturated_edep", entry, n_digits, default=None)
    contrib_crystal_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_crystal_id", entry, n_digits, default=None)
    pid_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_pid", entry, n_digits, default=None)
    mom_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_mom", entry, n_digits, default=None)
    raw_x_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_x", entry, n_digits, default=None)
    raw_y_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_y", entry, n_digits, default=None)
    raw_z_by_digit = _get_digit_nested_event_value(digit_arrays, digit_branch_map, "contributor_z", entry, n_digits, default=None)

    raw_key_to_cluster, primary_to_cluster = _get_cluster_label_maps(
        cluster_arrays=cluster_arrays,
        cluster_branch_map=cluster_branch_map,
        entry=entry,
        fallback_primary=fallback_primary,
    )

    rows: List[List[float]] = []
    meta = {
        "raw_hit_key": [],
        "raw_hit_primary_id": [],
        "raw_hit_crystal_id": [],
        "raw_hit_pid": [],
        "raw_hit_incident_mom": [],
        "raw_hit_x": [],
        "raw_hit_y": [],
        "raw_hit_z": [],
        "raw_hit_saturated_energy": [],
        "digit_index": [],
        "channel": [],
        "sequence_id": [],
        "block_id": [],
        "crystal_id": [],
    }
    cluster_ids: List[int] = []

    valid_digits = valid_geometry_mask(np.asarray(y_digits), np.asarray(z_digits),
        *[np.asarray(values) if name in digit_branch_map else None for name, values in (
            ("channel", channel), ("sequence_id", sequence_id),
            ("block_id", block_id), ("crystal_id", crystal_id))])
    for idig in range(n_digits):
        if not valid_digits[idig]:
            continue
        n_contrib = max(len(primary_by_digit[idig]), len(time_by_digit[idig]), len(edep_by_digit[idig]))
        for icontrib in range(n_contrib):
            energy = float(edep_by_digit[idig][icontrib]) if icontrib < len(edep_by_digit[idig]) else 0.0
            if not (energy > 0.0):
                continue
            time_ns = float(time_by_digit[idig][icontrib]) if icontrib < len(time_by_digit[idig]) else 0.0
            primary_id = int(primary_by_digit[idig][icontrib]) if icontrib < len(primary_by_digit[idig]) else -1
            raw_key = int(raw_keys_by_digit[idig][icontrib]) if icontrib < len(raw_keys_by_digit[idig]) else int((entry + 1) * 10**9 + idig * 10**5 + icontrib)
            cid = raw_key_to_cluster.get(raw_key, -1)
            if cid < 0 and fallback_primary:
                cid = primary_to_cluster.get(primary_id, -1)

            rows.append([float(y_digits[idig]), float(z_digits[idig]), time_ns, energy, energy])
            cluster_ids.append(int(cid))
            meta["raw_hit_key"].append(raw_key)
            meta["raw_hit_primary_id"].append(primary_id)
            meta["raw_hit_crystal_id"].append(int(contrib_crystal_by_digit[idig][icontrib]) if icontrib < len(contrib_crystal_by_digit[idig]) else int(crystal_id[idig]))
            meta["raw_hit_pid"].append(int(pid_by_digit[idig][icontrib]) if icontrib < len(pid_by_digit[idig]) else 0)
            meta["raw_hit_incident_mom"].append(float(mom_by_digit[idig][icontrib]) if icontrib < len(mom_by_digit[idig]) else 0.0)
            meta["raw_hit_x"].append(float(raw_x_by_digit[idig][icontrib]) if icontrib < len(raw_x_by_digit[idig]) else np.nan)
            meta["raw_hit_y"].append(float(raw_y_by_digit[idig][icontrib]) if icontrib < len(raw_y_by_digit[idig]) else np.nan)
            meta["raw_hit_z"].append(float(raw_z_by_digit[idig][icontrib]) if icontrib < len(raw_z_by_digit[idig]) else np.nan)
            meta["raw_hit_saturated_energy"].append(float(sat_edep_by_digit[idig][icontrib]) if icontrib < len(sat_edep_by_digit[idig]) else energy)
            meta["digit_index"].append(int(digit_index[idig]) if int(digit_index[idig]) >= 0 else idig)
            meta["channel"].append(int(channel[idig]))
            meta["sequence_id"].append(int(sequence_id[idig]))
            meta["block_id"].append(int(block_id[idig]))
            meta["crystal_id"].append(int(crystal_id[idig]))

    if rows:
        x_raw_event = np.asarray(rows, dtype=np.float32)
        cluster_id_event = np.asarray(cluster_ids, dtype=np.int64)
    else:
        x_raw_event = np.empty((0, 5), dtype=np.float32)
        cluster_id_event = np.empty((0,), dtype=np.int64)

    meta_np: Dict[str, np.ndarray] = {}
    for key, values in meta.items():
        if key in {"raw_hit_incident_mom", "raw_hit_x", "raw_hit_y", "raw_hit_z", "raw_hit_saturated_energy"}:
            meta_np[key] = np.asarray(values, dtype=np.float32)
        else:
            meta_np[key] = np.asarray(values, dtype=np.int64)

    truth_info = compute_truth_cluster_info(
        x_raw_event=x_raw_event,
        cluster_id_event=cluster_id_event,
        amp_to_energy=1.0,
    )
    event_id = _scalar_event_id(digit_arrays, digit_branch_map, entry)
    return x_raw_event, meta_np, cluster_id_event, truth_info, event_id


def make_truth_data_object(
    x_raw_event: np.ndarray,
    meta_event: Dict[str, np.ndarray],
    cluster_id_event: np.ndarray,
    node_indices: np.ndarray,
    event_id: int,
    file_index: int,
    local_entry: int,
    graph_index_in_event: int,
    seed_event_index: int,
    seed_energy: float,
    seed_total_energy: float,
    truth_info: Dict[str, np.ndarray],
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    edge_radius_mm: float,
    edge_time_ns: float,
    edge_weight_scale_mm: float,
    make_undirected: bool,
    self_loops: bool,
    complete_graph_edges: bool,
    regression_features: bool = True,
) -> Data:
    x_raw_np = x_raw_event[node_indices]
    if regression_features:
        x_np, window_ref = make_truth_model_node_features(
        x_raw_np=x_raw_np,
        stats=stats,
        normalize=normalize,
        y_scale_mm=y_scale_mm,
        z_scale_mm=z_scale_mm,
        z_offset_mm=z_offset_mm,
            relative_time_scale_ns=(edge_time_ns if edge_time_ns > 0.0 else 50.0),
        )
    else:
        x_np = make_model_node_features(
            x_raw=x_raw_np, stats=stats, normalize=normalize,
            y_scale_mm=y_scale_mm, z_scale_mm=z_scale_mm, z_offset_mm=z_offset_mm,
        )
        window_ref = _legacy_window_physical_moments(
            x_raw_np[:,0], x_raw_np[:,1], x_raw_np[:,2], np.maximum(x_raw_np[:,3],0.0)
        )

    x = torch.tensor(x_np, dtype=torch.float32)
    x_raw = torch.tensor(x_raw_np, dtype=torch.float32)
    cluster_id = torch.tensor(cluster_id_event[node_indices], dtype=torch.long)
    is_signal = (cluster_id >= 0).to(torch.float32)
    is_ambiguous = torch.zeros_like(is_signal, dtype=torch.bool)

    edge_index, edge_attr, edge_weight = build_edges(
        y=x_raw[:, 0],
        z=x_raw[:, 1],
        t=x_raw[:, 2],
        amp=x_raw[:, 3],
        area=x_raw[:, 4],
        edge_radius_mm=edge_radius_mm,
        edge_time_ns=edge_time_ns,
        edge_weight_scale_mm=edge_weight_scale_mm,
        make_undirected=make_undirected,
        self_loops=self_loops,
        complete_graph=complete_graph_edges,
    )
    y_edge = build_edge_labels(edge_index, cluster_id)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        y_node=is_signal,
        y_edge=y_edge,
        cluster_id=cluster_id,
        node_truth_cluster=cluster_id.clone(),
        is_ambiguous=is_ambiguous,
    )

    data.x_raw = x_raw
    data.node_y = x_raw[:, 0].clone()
    data.node_z = x_raw[:, 1].clone()
    # In hybrid truth mode, x_raw columns are [y, z, truth_time, truth_energy, 0].
    # The last column is an explicit zero padding column kept only so existing
    # 5-input GNN training scripts do not need to change.  It is not a duplicated
    # physics feature.  Reco pulse quantities are stored separately as node_reco_*.
    data.node_t = x_raw[:, 2].clone()
    data.node_amp = x_raw[:, 3].clone()
    data.node_area = x_raw[:, 4].clone()
    data.node_energy = x_raw[:, 3].clone()
    data.node_truth_time = x_raw[:, 2].clone()
    data.node_truth_energy = x_raw[:, 3].clone()

    _attach_window_reference_fields(
        data, window_ref,
        make_truth_window_features(window_ref, int(x.shape[0]), stats, y_scale_mm, z_scale_mm, z_offset_mm),
    )

    data.event = torch.tensor([int(event_id)], dtype=torch.long)
    data.file_index = torch.tensor([file_index], dtype=torch.long)
    data.local_entry = torch.tensor([local_entry], dtype=torch.long)
    data.graph_in_event = torch.tensor([graph_index_in_event], dtype=torch.long)
    data.seed_node = torch.tensor([seed_event_index], dtype=torch.long)
    data.seed_amp = torch.tensor([seed_energy], dtype=torch.float32)
    data.seed_total_amp = torch.tensor([seed_total_energy], dtype=torch.float32)
    data.seed_energy = torch.tensor([seed_energy], dtype=torch.float32)
    data.seed_total_energy = torch.tensor([seed_total_energy], dtype=torch.float32)
    data.graph_source = torch.tensor([1], dtype=torch.long)  # 1 = truth-hit mode

    data.original_node_index = torch.tensor(node_indices.astype(np.int64), dtype=torch.long)
    for name in [
        "pulse_index", "pulse_event_index", "digit_index", "channel",
        "sequence_id", "block_id", "crystal_id", "raw_hit_key",
        "raw_hit_primary_id", "raw_hit_crystal_id", "raw_hit_pid",
        "n_truth_contributors", "node_has_truth",
    ]:
        values = meta_event.get(name)
        if values is None:
            values = np.full((len(x_raw_event),), -1, dtype=np.int64)
        setattr(data, name, torch.tensor(values[node_indices], dtype=torch.long))

    for name in [
        "raw_hit_incident_mom", "raw_hit_x", "raw_hit_y", "raw_hit_z",
        "raw_hit_saturated_energy", "signal_truth_energy",
        "reco_t", "reco_amp", "reco_area",
    ]:
        values = meta_event.get(name)
        if values is None:
            values = np.full((len(x_raw_event),), np.nan, dtype=np.float32)
        tensor = torch.tensor(values[node_indices], dtype=torch.float32)
        setattr(data, name, tensor)

    # Explicit aliases make checks/readback less ambiguous.
    if hasattr(data, "reco_t"):
        data.node_reco_t = data.reco_t.clone()
    if hasattr(data, "reco_amp"):
        data.node_reco_amp = data.reco_amp.clone()
    if hasattr(data, "reco_area"):
        data.node_reco_area = data.reco_area.clone()

    data.truth_cluster_id = torch.tensor(truth_info["id"], dtype=torch.long)
    data.truth_cluster_primary_id = torch.tensor(truth_info["primary_id"], dtype=torch.long)
    data.truth_cluster_time_slice = torch.tensor(truth_info["time_slice"], dtype=torch.long)
    data.truth_cluster_size = torch.tensor(truth_info["size"], dtype=torch.long)
    data.truth_cluster_total_amp = torch.tensor(truth_info["total_amp"], dtype=torch.float32)
    data.truth_cluster_energy = torch.tensor(truth_info["energy"], dtype=torch.float32)
    data.truth_cluster_t0 = torch.tensor(truth_info["t0"], dtype=torch.float32)
    data.truth_cluster_t0_x = torch.tensor(truth_info["t0_x"], dtype=torch.float32)
    data.truth_cluster_t0_y = torch.tensor(truth_info["t0_y"], dtype=torch.float32)
    data.truth_cluster_t0_z = torch.tensor(truth_info["t0_z"], dtype=torch.float32)

    return data



def _open_truth_arrays(filename: str, args: argparse.Namespace):
    f = uproot.open(filename)
    if args.truth_digits_tree not in f:
        raise RuntimeError(f"Truth mode requested but tree '{args.truth_digits_tree}' not found in {filename}")
    if args.truth_clusters_tree not in f:
        raise RuntimeError(f"Truth mode requested but tree '{args.truth_clusters_tree}' not found in {filename}")
    digit_tree = f[args.truth_digits_tree]
    cluster_tree = f[args.truth_clusters_tree]

    # In truth mode the seed window is still built from reco pulse amplitude.
    # Truth arrays are only used to replace node time/energy and labels.
    required = ["y", "z", "contributor_primary_id", "contributor_time"]
    required.append("contributor_saturated_edep" if args.truth_energy == "saturated" else "contributor_edep")
    if not args.truth_label_fallback_primary:
        required.append("contributor_raw_hit_key")

    digit_branch_map = resolve_branch_aliases(
        digit_tree, TRUTH_DIGIT_BRANCH_ALIASES, required=required, context=args.truth_digits_tree,
    )
    cluster_required = regression_required_cluster_logicals(args.reconstruction_step)
    if not args.truth_label_fallback_primary and "raw_hit_key" not in cluster_required:
        cluster_required.append("raw_hit_key")
    cluster_branch_map = resolve_branch_aliases(
        cluster_tree, TRUTH_CLUSTER_BRANCH_ALIASES, required=cluster_required, context=args.truth_clusters_tree,
    )

    digit_arrays = digit_tree.arrays(_branches_from_map(digit_branch_map), library="ak")
    cluster_arrays = cluster_tree.arrays(_branches_from_map(cluster_branch_map), library="ak")
    return f, digit_tree, cluster_tree, digit_arrays, cluster_arrays, digit_branch_map, cluster_branch_map



def _truth_energy_logical(args: argparse.Namespace, digit_branch_map: Dict[str, str]) -> str:
    """Return the resolved logical contributor-energy branch for truth features."""
    preferred = "contributor_saturated_edep" if args.truth_energy == "saturated" else "contributor_edep"
    fallback = "contributor_edep" if preferred == "contributor_saturated_edep" else "contributor_saturated_edep"
    if preferred in digit_branch_map:
        return preferred
    if fallback in digit_branch_map:
        return fallback
    raise RuntimeError(
        "Truth mode needs contributor energy branches, but neither deposited nor saturated energy was resolved"
    )


def _pick_digit_index_list(
    digit_arrays: Dict[str, object],
    digit_branch_map: Dict[str, str],
    entry: int,
    n_digits: int,
) -> List[int]:
    value = _get_event_value(digit_arrays, digit_branch_map, "digit_index", entry, default=None)
    if value is None:
        return list(range(n_digits))
    return [int(v) for v in _as_1d_list(value, n=n_digits, default=-1)]



def build_truth_cluster_info_from_tree(
    cluster_arrays: Dict[str, object],
    cluster_branch_map: Dict[str, str],
    entry: int,
    truth_energy: str,
) -> Dict[str, np.ndarray]:
    """Read per-cluster truth, including the regression targets for steps 1/2."""
    cluster_ids = [
        int(v) for v in _as_1d_list(
            _get_event_value(cluster_arrays, cluster_branch_map, "cluster_id", entry, default=[]),
            default=-1,
        )
    ]
    n_clusters = len(cluster_ids)

    def int_vec(logical: str, default: int = -1) -> np.ndarray:
        return np.asarray(
            [int(v) for v in _as_1d_list(
                _get_event_value(cluster_arrays, cluster_branch_map, logical, entry, default=None),
                n=n_clusters,
                default=default,
            )],
            dtype=np.int64,
        )

    def float_vec(logical: str, default: float = float("nan")) -> np.ndarray:
        return np.asarray(
            [float(v) for v in _as_1d_list(
                _get_event_value(cluster_arrays, cluster_branch_map, logical, entry, default=None),
                n=n_clusters,
                default=default,
            )],
            dtype=np.float32,
        )

    primary_ids = int_vec("primary_id")
    pids = int_vec("pid", default=0)
    time_slices = int_vec("time_slice")
    incident_state_valid = int_vec("incident_state_valid", default=0)
    most_sig_crystal_id = int_vec("most_significant_crystal_id", default=-1)

    total_edep = float_vec("total_edep")
    incident_mom = float_vec("incident_mom")
    incident_energy = float_vec("incident_energy")
    incident_kinetic_energy = float_vec("incident_kinetic_energy")
    incident_time = float_vec("incident_time")
    incident_x = float_vec("incident_x")
    incident_y = float_vec("incident_y")
    incident_z = float_vec("incident_z")
    incident_px = float_vec("incident_px")
    incident_py = float_vec("incident_py")
    incident_pz = float_vec("incident_pz")
    incident_dir_x = float_vec("incident_dir_x")
    incident_dir_y = float_vec("incident_dir_y")
    incident_dir_z = float_vec("incident_dir_z")
    most_sig_edep = float_vec("most_significant_crystal_edep")
    most_sig_raw_edep = float_vec("most_significant_crystal_raw_edep")
    most_sig_fraction = float_vec("most_significant_crystal_fraction")
    leakage = float_vec("leakage")

    raw_t0 = float_vec("raw_t0")
    raw_t0_x = float_vec("raw_t0_x")
    raw_t0_y = float_vec("raw_t0_y")
    raw_t0_z = float_vec("raw_t0_z")

    raw_keys = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_key", entry, n_clusters, default=None
    )
    raw_crystal_ids = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_crystal_id", entry, n_clusters, default=None
    )
    raw_times = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_time", entry, n_clusters, default=None
    )
    raw_x = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_x", entry, n_clusters, default=None
    )
    raw_y = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_y", entry, n_clusters, default=None
    )
    raw_z = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_z", entry, n_clusters, default=None
    )

    # Raw-hit energy remains useful as a backward-compatible fallback only.
    energy_logical = (
        "raw_hit_saturated_edep" if truth_energy == "saturated" else "raw_hit_edep"
    )
    if energy_logical not in cluster_branch_map:
        energy_logical = (
            "raw_hit_edep" if energy_logical == "raw_hit_saturated_edep"
            else "raw_hit_saturated_edep"
        )
    raw_energy = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, energy_logical, entry, n_clusters, default=None
    )
    # For detector-time reconstruction use saturated energy as the timing
    # weight because the waveform pulse is generated by the visible/saturated
    # calorimeter response.  Updated TruthMaker files always contain this branch.
    raw_saturated_energy = _get_cluster_nested_event_value(
        cluster_arrays, cluster_branch_map, "raw_hit_saturated_edep", entry, n_clusters, default=None
    )

    sizes: List[int] = []
    energies: List[float] = []
    t0_values: List[float] = []
    t0_x_values: List[float] = []
    t0_y_values: List[float] = []
    t0_z_values: List[float] = []
    most_sig_truth_time_energy_weighted: List[float] = []
    most_sig_truth_time_dominant: List[float] = []
    most_sig_truth_time_earliest: List[float] = []

    for ic in range(n_clusters):
        sizes.append(len(raw_keys[ic]))

        raw_sum = float(sum(max(float(e), 0.0) for e in raw_energy[ic]))
        # E_cluster for regression is the TruthMaker cluster edepMeV.  Only use
        # the raw-hit sum for old files that do not contain that scalar branch.
        energies.append(float(total_edep[ic]) if np.isfinite(total_edep[ic]) else raw_sum)

        earliest_index = -1
        earliest_time = float("inf")
        for ih, value in enumerate(raw_times[ic]):
            t = float(value)
            if np.isfinite(t) and t < earliest_time:
                earliest_time = t
                earliest_index = ih

        t0 = float(raw_t0[ic]) if np.isfinite(raw_t0[ic]) else (
            earliest_time if earliest_index >= 0 else float("nan")
        )
        x0 = float(raw_t0_x[ic])
        y0 = float(raw_t0_y[ic])
        z0 = float(raw_t0_z[ic])
        if earliest_index >= 0:
            if not np.isfinite(x0) and earliest_index < len(raw_x[ic]):
                x0 = float(raw_x[ic][earliest_index])
            if not np.isfinite(y0) and earliest_index < len(raw_y[ic]):
                y0 = float(raw_y[ic][earliest_index])
            if not np.isfinite(z0) and earliest_index < len(raw_z[ic]):
                z0 = float(raw_z[ic][earliest_index])

        t0_values.append(t0)
        t0_x_values.append(x0)
        t0_y_values.append(y0)
        t0_z_values.append(z0)

        # Build the truth reference time of the target cluster in its
        # most-significant crystal.  This is later combined with the reco
        # fitPeakTimeNs to move the incident-time target onto the detector clock:
        #   t_entry(det) = t_pulse(reco) - [t_crystal(truth) - t_entry(truth)].
        ms_crystal = int(most_sig_crystal_id[ic])
        hit_times: List[float] = []
        hit_weights: List[float] = []
        n_raw = min(len(raw_crystal_ids[ic]), len(raw_times[ic]))
        for ih in range(n_raw):
            try:
                crystal = int(raw_crystal_ids[ic][ih])
                t_hit = float(raw_times[ic][ih])
            except (TypeError, ValueError):
                continue
            if crystal != ms_crystal or not np.isfinite(t_hit):
                continue
            weight = 0.0
            if ih < len(raw_saturated_energy[ic]):
                try:
                    weight = max(float(raw_saturated_energy[ic][ih]), 0.0)
                except (TypeError, ValueError):
                    weight = 0.0
            hit_times.append(t_hit)
            hit_weights.append(weight)

        if hit_times:
            earliest_ref = float(min(hit_times))
            dominant_ref = float(hit_times[int(np.argmax(np.asarray(hit_weights, dtype=np.float64)))])
            weight_sum = float(np.sum(np.asarray(hit_weights, dtype=np.float64)))
            if weight_sum > 0.0:
                weighted_ref = float(np.dot(
                    np.asarray(hit_times, dtype=np.float64),
                    np.asarray(hit_weights, dtype=np.float64),
                ) / weight_sum)
            else:
                weighted_ref = earliest_ref
        else:
            earliest_ref = float("nan")
            dominant_ref = float("nan")
            weighted_ref = float("nan")

        most_sig_truth_time_energy_weighted.append(weighted_ref)
        most_sig_truth_time_dominant.append(dominant_ref)
        most_sig_truth_time_earliest.append(earliest_ref)

    energy_np = np.asarray(energies, dtype=np.float32)
    return {
        "id": np.asarray(cluster_ids, dtype=np.int64),
        "primary_id": primary_ids,
        "pid": pids,
        "time_slice": time_slices,
        "size": np.asarray(sizes, dtype=np.int64),
        "total_amp": energy_np.copy(),
        "energy": energy_np,
        "t0": np.asarray(t0_values, dtype=np.float32),
        "t0_x": np.asarray(t0_x_values, dtype=np.float32),
        "t0_y": np.asarray(t0_y_values, dtype=np.float32),
        "t0_z": np.asarray(t0_z_values, dtype=np.float32),
        "incident_mom": incident_mom,
        "incident_energy": incident_energy,
        "incident_kinetic_energy": incident_kinetic_energy,
        "incident_time": incident_time,
        "incident_x": incident_x,
        "incident_y": incident_y,
        "incident_z": incident_z,
        "incident_px": incident_px,
        "incident_py": incident_py,
        "incident_pz": incident_pz,
        "incident_dir_x": incident_dir_x,
        "incident_dir_y": incident_dir_y,
        "incident_dir_z": incident_dir_z,
        "incident_state_valid": incident_state_valid,
        "most_significant_crystal_id": most_sig_crystal_id,
        "most_significant_crystal_edep": most_sig_edep,
        "most_significant_crystal_raw_edep": most_sig_raw_edep,
        "most_significant_crystal_fraction": most_sig_fraction,
        "most_significant_crystal_truth_time_energy_weighted": np.asarray(
            most_sig_truth_time_energy_weighted, dtype=np.float32
        ),
        "most_significant_crystal_truth_time_dominant": np.asarray(
            most_sig_truth_time_dominant, dtype=np.float32
        ),
        "most_significant_crystal_truth_time_earliest": np.asarray(
            most_sig_truth_time_earliest, dtype=np.float32
        ),
        "leakage": leakage,
    }


def _find_truth_cluster_index(truth_info: Dict[str, np.ndarray], cluster_id: int) -> int:
    ids = np.asarray(truth_info.get("id", []), dtype=np.int64)
    matches = np.nonzero(ids == int(cluster_id))[0]
    return int(matches[0]) if matches.size else -1


def _choose_regression_cluster_id(
    cluster_id_event: np.ndarray,
    node_indices: np.ndarray,
    seed_event_index: int,
    node_weight: np.ndarray,
) -> int:
    """Choose one physical cluster for a regression graph, preferring the seed cluster."""
    if 0 <= int(seed_event_index) < len(cluster_id_event):
        seed_cid = int(cluster_id_event[int(seed_event_index)])
        if seed_cid >= 0:
            return seed_cid

    # If the seed itself is noise/ambiguous, use the valid cluster carrying the
    # largest positive node weight inside this graph.  This is only target
    # association; it is not a node/edge classification task.
    best_cid = -1
    best_weight = -1.0
    for idx in np.asarray(node_indices, dtype=np.int64):
        cid = int(cluster_id_event[idx])
        if cid < 0:
            continue
        w = float(max(float(node_weight[idx]), 0.0))
        if w > best_weight:
            best_weight = w
            best_cid = cid
    return best_cid




def _truth_cluster_primary_id(truth_info: Dict[str, np.ndarray], cluster_id: int) -> int:
    ic = _find_truth_cluster_index(truth_info, cluster_id)
    if ic < 0:
        return -1
    return int(truth_info["primary_id"][ic])


def _truth_cluster_t0(truth_info: Dict[str, np.ndarray], cluster_id: int) -> float:
    ic = _find_truth_cluster_index(truth_info, cluster_id)
    if ic < 0:
        return float("nan")
    return float(truth_info["t0"][ic])


def _window_model_summary(data: Data) -> np.ndarray:
    """Return six detector-observable window features from pulse-node model inputs.

    The summary intentionally uses only data.x / detector-level node weights, not
    truth cluster information.  These six values are supplied to the event-level
    window encoder and are therefore safe model inputs in reco mode.
    """
    x = data.x.detach().cpu().numpy().astype(np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((6,), dtype=np.float32)

    weights = None
    for name in ("node_amp", "node_energy"):
        if hasattr(data, name):
            value = getattr(data, name)
            if torch.is_tensor(value) and int(value.numel()) == int(x.shape[0]):
                w = value.detach().cpu().numpy().astype(np.float64).reshape(-1)
                w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
                if float(w.sum()) > 0.0:
                    weights = w
                    break
    if weights is None:
        weights = np.ones((x.shape[0],), dtype=np.float64)
    wsum = float(weights.sum())

    centroid_y = float(np.sum(weights * x[:, 0]) / wsum)
    centroid_z = float(np.sum(weights * x[:, 1]) / wsum)
    finite_t = x[np.isfinite(x[:, 2]), 2]
    earliest_t = float(np.min(finite_t)) if finite_t.size else 0.0
    mean_t = float(np.sum(weights * np.where(np.isfinite(x[:, 2]), x[:, 2], earliest_t)) / wsum)
    finite_activity = x[np.isfinite(x[:, 3]), 3]
    max_activity = float(np.max(finite_activity)) if finite_activity.size else 0.0
    log_n = float(np.log1p(x.shape[0]))
    return np.asarray(
        [centroid_y, centroid_z, earliest_t, mean_t, max_activity, log_n],
        dtype=np.float32,
    )


def _window_cluster_ids(data: Data) -> np.ndarray:
    if not hasattr(data, "cluster_id"):
        return np.empty((0,), dtype=np.int64)
    value = data.cluster_id.detach().cpu().numpy().astype(np.int64).reshape(-1)
    return np.unique(value[value >= 0])


def _select_hierarchical_targets(
    records: List[Dict[str, object]],
    truth_info: Dict[str, np.ndarray],
) -> List[Tuple[int, int, int]]:
    """Return all represented reconstructable primaries in one event.

    Each returned tuple is
        (primary_id, prompt_cluster_id, prompt_window_local).

    A primary can have multiple time-sliced truth clusters (prompt + delayed).
    Only the earliest represented cluster becomes its prompt target.  Delayed
    windows remain in the event and are associated to the same primary through
    the window-window association target.

    The target list is sorted by prompt truth time, then primary id, so tensor
    ordering is deterministic.  We do NOT encode or ask the model to predict a
    truth clusterId; clusterId/primaryId are validation metadata only.
    """
    represented_by_primary: Dict[int, set] = {}
    for rec in records:
        data = rec["data"]
        assert isinstance(data, Data)
        for cid in _window_cluster_ids(data):
            pid = _truth_cluster_primary_id(truth_info, int(cid))
            if pid >= 0:
                represented_by_primary.setdefault(pid, set()).add(int(cid))

    targets: List[Tuple[int, int, int]] = []
    for primary_id, represented in represented_by_primary.items():
        finite_prompt = []
        fallback_prompt = []
        for cid in sorted(represented):
            ic = _find_truth_cluster_index(truth_info, cid)
            if ic < 0:
                continue
            if int(truth_info["incident_state_valid"][ic]) <= 0:
                continue
            t0 = float(truth_info["t0"][ic])
            tslice = int(truth_info["time_slice"][ic]) if "time_slice" in truth_info else 0
            fallback_prompt.append((tslice, cid))
            if np.isfinite(t0):
                finite_prompt.append((t0, tslice, cid))
        if finite_prompt:
            prompt_cluster = min(finite_prompt)[2]
            prompt_time = min(finite_prompt)[0]
        elif fallback_prompt:
            prompt_cluster = min(fallback_prompt)[1]
            prompt_time = float("inf")
        else:
            continue

        prompt_windows = []
        for iw, rec in enumerate(records):
            data = rec["data"]
            assert isinstance(data, Data)
            if prompt_cluster in set(int(v) for v in _window_cluster_ids(data)):
                prompt_windows.append((float(rec.get("total_activity", 0.0)), iw))
        if not prompt_windows:
            continue
        prompt_window = max(prompt_windows)[1]
        targets.append((int(primary_id), int(prompt_cluster), int(prompt_window)))

    def key(item):
        pid,cid,_=item
        t=_truth_cluster_t0(truth_info,cid)
        return (float(t) if np.isfinite(t) else float("inf"), pid, cid)
    return sorted(targets, key=key)


def _object_target_payload(
    truth_info: Dict[str, np.ndarray],
    reconstruction_step: str,
    target_cluster_id: int,
    prompt_reference: Optional[Dict[str, float]] = None,
    reco_incident_time_target: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, object]]:
    """Build one physical target object without attaching it to a Data yet."""
    ic = _find_truth_cluster_index(truth_info, target_cluster_id)
    if ic < 0 or int(truth_info["incident_state_valid"][ic]) <= 0:
        return None
    pid = int(truth_info["pid"][ic])
    primary_id = int(truth_info["primary_id"][ic])
    incident_time = float(truth_info["incident_time"][ic])
    entry = np.asarray([
        truth_info["incident_x"][ic], truth_info["incident_y"][ic], truth_info["incident_z"][ic]
    ], dtype=np.float32)
    direction = np.asarray([
        truth_info["incident_dir_x"][ic], truth_info["incident_dir_y"][ic], truth_info["incident_dir_z"][ic]
    ], dtype=np.float32)
    if not (np.isfinite(incident_time) and np.all(np.isfinite(entry)) and np.all(np.isfinite(direction))):
        return None

    common = {
        "cluster_id": int(target_cluster_id), "primary_id": primary_id, "pid": pid,
        "incident_time_truth_ns": incident_time,
        "incident_x_mm": float(entry[0]), "incident_y_mm": float(entry[1]), "incident_z_mm": float(entry[2]),
        "incident_direction": direction,
    }
    if reconstruction_step == "1":
        ecl = float(truth_info["energy"][ic])
        emax = float(truth_info["most_significant_crystal_edep"][ic])
        if not (np.isfinite(ecl) and np.isfinite(emax)):
            return None
        if prompt_reference is None:
            return None
        t_target = incident_time if reco_incident_time_target is None else float(reco_incident_time_target["target_ns"])
        yc = float(prompt_reference["yc_mm"])
        zc = float(prompt_reference["zc_mm"])
        tmin = float(prompt_reference["tmin_ns"])
        if not (np.isfinite(t_target) and np.isfinite(yc) and np.isfinite(zc) and np.isfinite(tmin)):
            return None
        dt = tmin - t_target
        dy = yc - float(entry[1])
        dz = zc - float(entry[2])
        y = np.asarray([
            dt, dy, dz,
            direction[0], direction[1], direction[2],
            ecl, emax,
        ], dtype=np.float32)
        common.update({
            "y": y,
            "incident_time_target_ns": t_target,
            "prompt_centroid_y_mm": yc,
            "prompt_centroid_z_mm": zc,
            "prompt_tmin_ns": tmin,
            "target_dt_ns": dt,
            "target_dy_mm": dy,
            "target_dz_mm": dz,
            "cluster_energy_mev": ecl,
            "max_crystal_energy_mev": emax,
            "max_crystal_id": int(truth_info["most_significant_crystal_id"][ic]),
            "time_diag": reco_incident_time_target,
        })
        return common

    if reconstruction_step == "2":
        einc = float(truth_info["incident_energy"][ic])
        if pid == 0 or not np.isfinite(einc):
            return None
        common.update({
            "y": np.asarray([einc], dtype=np.float32),
            "incident_energy_mev": einc,
            "track_features": np.concatenate([entry,direction]).astype(np.float32),
        })
        return common
    raise ValueError(f"Unknown reconstruction_step: {reconstruction_step}")


def _window_primary_sets(records: List[Dict[str, object]], truth_info: Dict[str, np.ndarray]) -> List[set]:
    out=[]
    for rec in records:
        data=rec["data"]
        assert isinstance(data,Data)
        pset=set()
        for cid in _window_cluster_ids(data):
            pid=_truth_cluster_primary_id(truth_info,int(cid))
            if pid>=0: pset.add(int(pid))
        out.append(pset)
    return out


def _merge_hierarchical_event(
    records: List[Dict[str, object]],
    truth_info: Dict[str, np.ndarray],
    reconstruction_step: str,
    graph_source: str,
    reco_arrays: Optional[Dict[str, object]] = None,
    reco_entry: Optional[int] = None,
) -> Optional[Data]:
    """Merge all seed windows into one multi-object hierarchical event.

    Unlike the previous single-primary version, data.y now has K rows where K
    is the number of represented valid prompt objects in this event.  The model
    therefore produces K training regression outputs while the event keeps all
    windows (including delayed and unrelated overlapping activity).
    """
    if not records:
        return None
    selected = _select_hierarchical_targets(records, truth_info)
    if not selected:
        return None

    xs=[]; edge_indices=[]; edge_attrs=[]; edge_weights=[]; node_window_local=[]; node_offset=0
    candidate_node_fields = [
        "x_raw","node_y","node_z","node_t","node_amp","node_area","node_energy",
        "node_truth_time","node_truth_energy","node_reco_t","node_reco_amp","node_reco_area",
        "cluster_id","node_truth_cluster","is_ambiguous","pulse_index","digit_index","channel",
        "sequence_id","block_id","crystal_id","raw_hit_key","raw_hit_primary_id","raw_hit_crystal_id",
        "n_truth_contributors","node_has_truth",
    ]
    node_field_parts={name:[] for name in candidate_node_fields}
    window_x=[]; window_target_cluster_id=[]; window_primary_id=[]; window_seed_amp=[]; window_total_amp=[]
    window_centroid_y=[]; window_centroid_z=[]; window_tmin=[]; window_tmean=[]; window_esum=[]; window_emax=[]
    window_sigma_y=[]; window_sigma_z=[]; window_sigma_yz=[]; window_emax_fraction=[]; window_max_hit_dy=[]; window_max_hit_dz=[]
    window_truth_t0=[]
    primary_sets=_window_primary_sets(records,truth_info)
    prompt_clusters={cid for _,cid,_ in selected}

    for iw,rec in enumerate(records):
        data=rec["data"]; assert isinstance(data,Data); n=int(data.num_nodes)
        xs.append(data.x); edge_indices.append(data.edge_index+node_offset); edge_attrs.append(data.edge_attr)
        edge_weights.append(data.edge_weight if hasattr(data,"edge_weight") else torch.ones((data.edge_index.shape[1],),dtype=torch.float32))
        node_window_local.append(torch.full((n,),iw,dtype=torch.long))
        for name in candidate_node_fields:
            if hasattr(data,name):
                value=getattr(data,name)
                if torch.is_tensor(value) and value.dim()>0 and int(value.shape[0])==n:
                    node_field_parts[name].append(value)
        if not hasattr(data, "window_model_feature"):
            raise RuntimeError("window_model_feature missing: source-specific feature builder was not used")
        window_x.append(data.window_model_feature.detach().cpu().numpy().astype(np.float32))
        window_centroid_y.append(float(data.window_centroid_y_mm.view(-1)[0]))
        window_centroid_z.append(float(data.window_centroid_z_mm.view(-1)[0]))
        window_tmin.append(float(data.window_tmin_ns.view(-1)[0]))
        window_tmean.append(float(data.window_tmean_ns.view(-1)[0]))
        window_esum.append(float(data.window_esum_mev.view(-1)[0]))
        window_emax.append(float(data.window_emax_mev.view(-1)[0]))
        window_sigma_y.append(float(data.window_sigma_y_mm.view(-1)[0]))
        window_sigma_z.append(float(data.window_sigma_z_mm.view(-1)[0]))
        window_sigma_yz.append(float(data.window_sigma_yz_mm2.view(-1)[0]))
        window_emax_fraction.append(float(data.window_emax_over_esum.view(-1)[0]))
        window_max_hit_dy.append(float(data.window_max_hit_dy_mm.view(-1)[0]))
        window_max_hit_dz.append(float(data.window_max_hit_dz_mm.view(-1)[0]))
        dominant_cid=int(rec.get("target_cid",-1)); window_target_cluster_id.append(dominant_cid)
        window_primary_id.append(_truth_cluster_primary_id(truth_info,dominant_cid))
        window_seed_amp.append(float(rec.get("seed_amp",float("nan")))); window_total_amp.append(float(rec.get("total_activity",float("nan"))))
        cids=set(int(v) for v in _window_cluster_ids(data))
        t0s=[_truth_cluster_t0(truth_info,cid) for cid in cids]
        t0s=[v for v in t0s if np.isfinite(v)]
        window_truth_t0.append(min(t0s) if t0s else float("nan"))
        node_offset+=n

    # Window role is now event-global rather than relative to one chosen primary:
    # 0 other/noise; 1 contains a prompt cluster of any target object;
    # 2 contains truth activity but no target prompt cluster (typically delayed).
    window_role=[]
    for rec,pset in zip(records,primary_sets):
        data=rec["data"]; cids=set(int(v) for v in _window_cluster_ids(data))
        if any(cid in prompt_clusters for cid in cids): role=1
        elif pset: role=2
        else: role=0
        window_role.append(role)

    # Association supervision is one label for each unordered i<j pair, in the
    # exact same nested-loop order used by ecal_gnn_model._build_association_pairs.
    association_target=[]
    for i in range(len(records)):
        for j in range(i+1,len(records)):
            association_target.append(1.0 if (primary_sets[i] & primary_sets[j]) else 0.0)

    event_data=Data(
        x=torch.cat(xs,0),
        edge_index=torch.cat(edge_indices,1) if edge_indices else torch.empty((2,0),dtype=torch.long),
        edge_attr=torch.cat(edge_attrs,0) if edge_attrs else torch.empty((0,6),dtype=torch.float32),
        edge_weight=torch.cat(edge_weights,0) if edge_weights else torch.empty((0,),dtype=torch.float32),
    )
    event_data.node_window_local=torch.cat(node_window_local,0)
    event_data.window_x=torch.tensor(np.asarray(window_x),dtype=torch.float32)
    event_data.window_centroid_y_mm=torch.tensor(window_centroid_y,dtype=torch.float32)
    event_data.window_centroid_z_mm=torch.tensor(window_centroid_z,dtype=torch.float32)
    event_data.window_tmin_ns=torch.tensor(window_tmin,dtype=torch.float32)
    event_data.window_tmean_ns=torch.tensor(window_tmean,dtype=torch.float32)
    event_data.window_esum_mev=torch.tensor(window_esum,dtype=torch.float32)
    event_data.window_emax_mev=torch.tensor(window_emax,dtype=torch.float32)
    event_data.window_sigma_y_mm=torch.tensor(window_sigma_y,dtype=torch.float32)
    event_data.window_sigma_z_mm=torch.tensor(window_sigma_z,dtype=torch.float32)
    event_data.window_sigma_yz_mm2=torch.tensor(window_sigma_yz,dtype=torch.float32)
    event_data.window_emax_over_esum=torch.tensor(window_emax_fraction,dtype=torch.float32)
    event_data.window_max_hit_dy_mm=torch.tensor(window_max_hit_dy,dtype=torch.float32)
    event_data.window_max_hit_dz_mm=torch.tensor(window_max_hit_dz,dtype=torch.float32)
    event_data.num_windows=torch.tensor([len(records)],dtype=torch.long)
    event_data.window_role_target=torch.tensor(window_role,dtype=torch.long)
    event_data.window_role=event_data.window_role_target.clone()
    event_data.window_is_prompt=(event_data.window_role_target==1)
    event_data.window_is_delayed=(event_data.window_role_target==2)
    event_data.window_target_cluster_id=torch.tensor(window_target_cluster_id,dtype=torch.long)
    event_data.window_primary_id=torch.tensor(window_primary_id,dtype=torch.long)
    event_data.window_seed_amp=torch.tensor(window_seed_amp,dtype=torch.float32)
    event_data.window_total_amp=torch.tensor(window_total_amp,dtype=torch.float32)
    event_data.window_truth_t0_ns=torch.tensor(window_truth_t0,dtype=torch.float32)
    event_data.association_target=torch.tensor(association_target,dtype=torch.float32)
    event_data.num_association_pairs=torch.tensor([len(association_target)],dtype=torch.long)
    event_data.hierarchical_event=torch.tensor([1],dtype=torch.long)

    first=records[0]["data"]; assert isinstance(first,Data)
    for name in ("event","file_index","local_entry","graph_source"):
        if hasattr(first,name): setattr(event_data,name,getattr(first,name).clone())
    for name,parts in node_field_parts.items():
        if len(parts)==len(records): setattr(event_data,name,torch.cat(parts,0))

    payloads=[]; prompt_windows=[]
    for primary_id,prompt_cluster,prompt_window in selected:
        reco_time_target=None
        if reconstruction_step=="1" and graph_source=="reco":
            if reco_arrays is None or reco_entry is None:
                continue
            prompt_nodes=np.asarray(records[prompt_window]["node_indices"],dtype=np.int64)
            reco_time_target=_build_reco_detector_incident_time_target(
                arrays=reco_arrays, entry=int(reco_entry), node_indices=prompt_nodes,
                truth_info=truth_info, target_cluster_id=prompt_cluster,
            )
            if reco_time_target is None:
                continue
        prompt_data = records[prompt_window]["data"]
        prompt_ref = {
            "yc_mm": float(prompt_data.window_centroid_y_mm.view(-1)[0]),
            "zc_mm": float(prompt_data.window_centroid_z_mm.view(-1)[0]),
            "tmin_ns": float(prompt_data.window_tmin_ns.view(-1)[0]),
        }
        payload=_object_target_payload(
            truth_info,reconstruction_step,prompt_cluster,
            prompt_reference=prompt_ref,
            reco_incident_time_target=reco_time_target,
        )
        if payload is None:
            continue
        payloads.append(payload); prompt_windows.append(prompt_window)

    if not payloads:
        return None
    K=len(payloads)
    event_data.num_targets=torch.tensor([K],dtype=torch.long)
    event_data.target_prompt_window_local=torch.tensor(prompt_windows,dtype=torch.long)
    event_data.target_prompt_cluster_id=torch.tensor([int(p["cluster_id"]) for p in payloads],dtype=torch.long)
    event_data.target_cluster_id=event_data.target_prompt_cluster_id.clone()
    event_data.target_primary_id=torch.tensor([int(p["primary_id"]) for p in payloads],dtype=torch.long)
    event_data.target_pid=torch.tensor([int(p["pid"]) for p in payloads],dtype=torch.long)
    event_data.target_incident_x_mm=torch.tensor([float(p["incident_x_mm"]) for p in payloads],dtype=torch.float32)
    event_data.target_incident_y_mm=torch.tensor([float(p["incident_y_mm"]) for p in payloads],dtype=torch.float32)
    event_data.target_incident_z_mm=torch.tensor([float(p["incident_z_mm"]) for p in payloads],dtype=torch.float32)
    event_data.target_incident_direction=torch.tensor(np.stack([p["incident_direction"] for p in payloads]),dtype=torch.float32)
    event_data.target_incident_time_truth_ns=torch.tensor([float(p["incident_time_truth_ns"]) for p in payloads],dtype=torch.float32)
    event_data.y=torch.tensor(np.stack([p["y"] for p in payloads]),dtype=torch.float32)

    if reconstruction_step=="1":
        event_data.step1_target=event_data.y.clone()
        event_data.target_dt_ns=event_data.y[:,0].clone()
        event_data.target_dy_mm=event_data.y[:,1].clone()
        event_data.target_dz_mm=event_data.y[:,2].clone()
        event_data.target_prompt_centroid_y_mm=torch.tensor([float(p["prompt_centroid_y_mm"]) for p in payloads],dtype=torch.float32)
        event_data.target_prompt_centroid_z_mm=torch.tensor([float(p["prompt_centroid_z_mm"]) for p in payloads],dtype=torch.float32)
        event_data.target_prompt_tmin_ns=torch.tensor([float(p["prompt_tmin_ns"]) for p in payloads],dtype=torch.float32)
        event_data.target_incident_time_ns=torch.tensor([float(p["incident_time_target_ns"]) for p in payloads],dtype=torch.float32)
        event_data.target_incident_position_yz=torch.stack([event_data.target_incident_y_mm,event_data.target_incident_z_mm],dim=1)
        event_data.target_incident_position=event_data.target_incident_position_yz.clone()
        event_data.target_cluster_energy_mev=event_data.y[:,6].clone()
        event_data.target_most_significant_crystal_energy_mev=event_data.y[:,7].clone()
        event_data.target_most_significant_crystal_id=torch.tensor([int(p["max_crystal_id"]) for p in payloads],dtype=torch.long)
        event_data.target_incident_time_scale=torch.tensor([1 if p.get("time_diag") is not None else 0 for p in payloads],dtype=torch.long)
        # Multi-object timing diagnostics. Missing truth-mode ingredients are NaN/-1.
        def diag_float(key):
            return torch.tensor([float(p["time_diag"].get(key,float("nan"))) if p.get("time_diag") is not None else float("nan") for p in payloads],dtype=torch.float32)
        def diag_int(key):
            return torch.tensor([int(p["time_diag"].get(key,-1)) if p.get("time_diag") is not None else -1 for p in payloads],dtype=torch.long)
        event_data.target_time_reference_reco_pulse_ns=diag_float("reco_pulse_time_ns")
        event_data.target_time_reference_truth_crystal_ns=diag_float("truth_crystal_ns")
        event_data.target_time_reference_truth_delta_ns=diag_float("truth_delta_ns")
        event_data.target_incident_time_dominant_ns=diag_float("target_dominant_ns")
        event_data.target_incident_time_earliest_ns=diag_float("target_earliest_ns")
        event_data.target_time_reference_crystal_id=diag_int("reference_crystal_id")
        event_data.target_time_reference_pulse_event_index=diag_int("reference_pulse_event_index")
        event_data.target_time_reference_pulse_index=diag_int("reference_pulse_index")
        event_data.target_time_reference_match_level=diag_int("reference_match_level")
        event_data.reconstruction_step=torch.tensor([1],dtype=torch.long)
    elif reconstruction_step=="2":
        event_data.step2_target_incident_energy_mev=event_data.y.clone()
        event_data.step2_track_features=torch.tensor(np.stack([p["track_features"] for p in payloads]),dtype=torch.float32)
        event_data.step2_pdg=torch.tensor([int(p["pid"]) for p in payloads],dtype=torch.long)
        event_data.reconstruction_step=torch.tensor([2],dtype=torch.long)
    else:
        raise ValueError(f"Unknown reconstruction_step: {reconstruction_step}")

    # Classification truth is auxiliary only; never expose it as model input.
    for name in ("y_node","y_edge","raw_hit_incident_mom","raw_hit_pid"):
        if hasattr(event_data,name): delattr(event_data,name)
    return event_data


def _build_reco_detector_incident_time_target(
    arrays: Dict[str, object],
    entry: int,
    node_indices: np.ndarray,
    truth_info: Dict[str, np.ndarray],
    target_cluster_id: int,
) -> Optional[Dict[str, float]]:
    """
    Build the Step-1 incident-time target on the *reco waveform clock*.

    Definition (default/reference used for training):

        t_entry_detector = t_peak_reco
                           - (t_crystal_truth - t_entry_truth)

    where
      * t_peak_reco is fitPeakTimeNs of the most-significant crystal pulse,
      * t_crystal_truth is the saturated-energy-weighted truth hit time of the
        selected truth cluster in that same crystal,
      * t_entry_truth is ecalTruthClusters.incidentTimeNs.

    The returned dictionary also contains dominant-hit and earliest-hit variants
    for diagnostics.  No detector-time quantity is written back to TruthMaker;
    it is a derived reconstruction target constructed here from reco + truth.
    """
    ic = _find_truth_cluster_index(truth_info, target_cluster_id)
    if ic < 0:
        return None

    ms_crystal = int(truth_info["most_significant_crystal_id"][ic])
    t_entry_truth = float(truth_info["incident_time"][ic])
    t_crystal_weighted = float(
        truth_info["most_significant_crystal_truth_time_energy_weighted"][ic]
    )
    t_crystal_dominant = float(
        truth_info["most_significant_crystal_truth_time_dominant"][ic]
    )
    t_crystal_earliest = float(
        truth_info["most_significant_crystal_truth_time_earliest"][ic]
    )
    if ms_crystal < 0 or not np.isfinite(t_entry_truth):
        return None

    pulse_crystal = as_numpy_1d(arrays["crystalId"][entry]).astype(np.int64)
    pulse_cluster = as_numpy_1d(arrays["clusterId"][entry]).astype(np.int64)
    pulse_time = as_numpy_1d(arrays["fitPeakTimeNs"][entry]).astype(np.float64)
    pulse_amp = as_numpy_1d(arrays["fitAmplitude"][entry]).astype(np.float64)
    n_pulses = len(pulse_crystal)

    graph_mask = np.zeros(n_pulses, dtype=bool)
    graph_indices = np.asarray(node_indices, dtype=np.int64)
    graph_indices = graph_indices[(graph_indices >= 0) & (graph_indices < n_pulses)]
    graph_mask[graph_indices] = True

    # Prefer the actual target-cluster pulse inside the graph.  The fallbacks
    # keep the constructor usable for edge cases where matching/graph-window
    # bookkeeping is incomplete, while still requiring the same physical crystal.
    candidate_masks = [
        graph_mask & (pulse_crystal == ms_crystal) & (pulse_cluster == target_cluster_id),
        (pulse_crystal == ms_crystal) & (pulse_cluster == target_cluster_id),
        graph_mask & (pulse_crystal == ms_crystal),
        (pulse_crystal == ms_crystal),
    ]

    chosen = -1
    match_level = -1
    for ilevel, mask in enumerate(candidate_masks):
        candidates = np.nonzero(mask & np.isfinite(pulse_time))[0]
        if candidates.size == 0:
            continue
        # Multiple pulses in one crystal are possible in principle.  Use the
        # largest reconstructed pulse because it is the timing measurement with
        # the strongest connection to the most-significant truth deposition.
        candidate_amp = np.where(np.isfinite(pulse_amp[candidates]), pulse_amp[candidates], -np.inf)
        chosen = int(candidates[int(np.argmax(candidate_amp))])
        match_level = ilevel
        break

    if chosen < 0:
        return None

    t_peak_reco = float(pulse_time[chosen])
    if not np.isfinite(t_peak_reco) or not np.isfinite(t_crystal_weighted):
        return None

    def convert(t_crystal_truth: float) -> float:
        if not np.isfinite(t_crystal_truth):
            return float("nan")
        return float(t_peak_reco - (t_crystal_truth - t_entry_truth))

    target_weighted = convert(t_crystal_weighted)
    target_dominant = convert(t_crystal_dominant)
    target_earliest = convert(t_crystal_earliest)
    if not np.isfinite(target_weighted):
        return None

    return {
        "target_ns": target_weighted,
        "target_dominant_ns": target_dominant,
        "target_earliest_ns": target_earliest,
        "truth_entry_ns": t_entry_truth,
        "truth_crystal_ns": t_crystal_weighted,
        "truth_crystal_dominant_ns": t_crystal_dominant,
        "truth_crystal_earliest_ns": t_crystal_earliest,
        "truth_delta_ns": float(t_crystal_weighted - t_entry_truth),
        "reco_pulse_time_ns": t_peak_reco,
        "reference_crystal_id": float(ms_crystal),
        "reference_pulse_event_index": float(chosen),
        "reference_pulse_index": float(as_numpy_1d(arrays["pulseIndex"][entry]).astype(np.int64)[chosen]),
        "reference_match_level": float(match_level),
    }

def attach_regression_task(
    data: Data,
    reconstruction_step: str,
    truth_info: Dict[str, np.ndarray],
    target_cluster_id: int,
    reco_incident_time_target: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Attach one graph-level regression target.

    Returns False when the selected cluster does not contain all truth needed by
    the requested step, so the caller can skip that graph instead of training on
    NaN labels.
    """
    if reconstruction_step == "legacy":
        data.reconstruction_step = torch.tensor([0], dtype=torch.long)
        return True

    ic = _find_truth_cluster_index(truth_info, target_cluster_id)
    if ic < 0:
        return False

    state_valid = int(truth_info["incident_state_valid"][ic]) > 0
    pid = int(truth_info["pid"][ic])
    cluster_energy = float(truth_info["energy"][ic])
    max_crystal_energy = float(truth_info["most_significant_crystal_edep"][ic])
    incident_time = float(truth_info["incident_time"][ic])
    entry = np.asarray([
        truth_info["incident_x"][ic],
        truth_info["incident_y"][ic],
        truth_info["incident_z"][ic],
    ], dtype=np.float32)
    direction = np.asarray([
        truth_info["incident_dir_x"][ic],
        truth_info["incident_dir_y"][ic],
        truth_info["incident_dir_z"][ic],
    ], dtype=np.float32)

    common_finite = (
        state_valid
        and np.isfinite(incident_time)
        and np.all(np.isfinite(entry))
        and np.all(np.isfinite(direction))
    )
    if not common_finite:
        return False

    data.target_cluster_id = torch.tensor([int(target_cluster_id)], dtype=torch.long)
    data.target_primary_id = torch.tensor([int(truth_info["primary_id"][ic])], dtype=torch.long)
    data.target_incident_state_valid = torch.tensor([1], dtype=torch.bool)

    if reconstruction_step == "1":
        if not (np.isfinite(cluster_energy) and np.isfinite(max_crystal_energy)):
            return False

        # Truth-mode graphs keep the pure G4 ECAL-entry time.  Reco-mode graphs
        # pass a detector-clock override derived from fitPeakTimeNs.
        incident_time_target = incident_time
        if reco_incident_time_target is not None:
            incident_time_target = float(reco_incident_time_target["target_ns"])
        if not np.isfinite(incident_time_target):
            return False

        # Step 1 intentionally does not regress the ECAL entrance x coordinate:
        # the entrance plane x is fixed by geometry.  The position target is y,z.
        target = np.asarray([
            incident_time_target,
            entry[1], entry[2],
            direction[0], direction[1], direction[2],
            cluster_energy,
            max_crystal_energy,
        ], dtype=np.float32)
        data.y = torch.tensor(target.reshape(1, -1), dtype=torch.float32)
        data.step1_target = data.y.clone()
        data.target_incident_time_ns = torch.tensor([incident_time_target], dtype=torch.float32)
        # Keep the original G4 time and the ingredients of the detector-clock
        # conversion for validation/debugging; only target_incident_time_ns is
        # part of the Step-1 regression target.
        data.target_incident_time_truth_ns = torch.tensor([incident_time], dtype=torch.float32)
        data.target_incident_time_scale = torch.tensor(
            [1 if reco_incident_time_target is not None else 0], dtype=torch.long
        )
        if reco_incident_time_target is not None:
            data.target_time_reference_reco_pulse_ns = torch.tensor(
                [reco_incident_time_target["reco_pulse_time_ns"]], dtype=torch.float32
            )
            data.target_time_reference_truth_crystal_ns = torch.tensor(
                [reco_incident_time_target["truth_crystal_ns"]], dtype=torch.float32
            )
            data.target_time_reference_truth_delta_ns = torch.tensor(
                [reco_incident_time_target["truth_delta_ns"]], dtype=torch.float32
            )
            data.target_incident_time_dominant_ns = torch.tensor(
                [reco_incident_time_target["target_dominant_ns"]], dtype=torch.float32
            )
            data.target_incident_time_earliest_ns = torch.tensor(
                [reco_incident_time_target["target_earliest_ns"]], dtype=torch.float32
            )
            data.target_time_reference_crystal_id = torch.tensor(
                [int(reco_incident_time_target["reference_crystal_id"])], dtype=torch.long
            )
            data.target_time_reference_pulse_event_index = torch.tensor(
                [int(reco_incident_time_target["reference_pulse_event_index"])], dtype=torch.long
            )
            data.target_time_reference_pulse_index = torch.tensor(
                [int(reco_incident_time_target["reference_pulse_index"])], dtype=torch.long
            )
            data.target_time_reference_match_level = torch.tensor(
                [int(reco_incident_time_target["reference_match_level"])], dtype=torch.long
            )
        data.target_incident_x_mm = torch.tensor([float(entry[0])], dtype=torch.float32)  # diagnostic only
        data.target_incident_position_yz = torch.tensor(entry[1:3].reshape(1, 2), dtype=torch.float32)
        # Backward-compatible named field now contains only the two regressed coordinates.
        data.target_incident_position = data.target_incident_position_yz.clone()
        data.target_incident_direction = torch.tensor(direction.reshape(1, 3), dtype=torch.float32)
        data.target_cluster_energy_mev = torch.tensor([cluster_energy], dtype=torch.float32)
        data.target_most_significant_crystal_energy_mev = torch.tensor([max_crystal_energy], dtype=torch.float32)
        data.target_most_significant_crystal_id = torch.tensor(
            [int(truth_info["most_significant_crystal_id"][ic])], dtype=torch.long
        )
        data.reconstruction_step = torch.tensor([1], dtype=torch.long)

    elif reconstruction_step == "2":
        incident_energy = float(truth_info["incident_energy"][ic])
        if pid == 0 or not np.isfinite(incident_energy):
            return False
        # IMPORTANT: no momentum magnitude or momentum components are included
        # in this auxiliary track tensor.  The only track inputs are entry point
        # and direction; PDG is stored separately for an embedding/one-hot path.
        track_features = np.concatenate([entry, direction]).astype(np.float32)
        data.step2_track_features = torch.tensor(track_features.reshape(1, 6), dtype=torch.float32)
        data.step2_pdg = torch.tensor([pid], dtype=torch.long)
        data.target_pid = data.step2_pdg.clone()
        data.y = torch.tensor([[incident_energy]], dtype=torch.float32)
        data.step2_target_incident_energy_mev = data.y.clone()
        data.reconstruction_step = torch.tensor([2], dtype=torch.long)
    else:
        raise ValueError(f"Unknown reconstruction_step: {reconstruction_step}")

    # Regression datasets intentionally do not expose classification labels.
    for name in ["y_node", "y_edge"]:
        if hasattr(data, name):
            delattr(data, name)

    # Avoid accidental truth leakage in later training scripts.  Step 1 is
    # ECAL-only, and step 2 gets PDG only at graph level after PID.  These raw
    # per-hit truth values are not model inputs and are removed in regression mode.
    for name in ["raw_hit_incident_mom", "raw_hit_pid"]:
        if hasattr(data, name):
            delattr(data, name)

    # Full-event truth-cluster vectors were useful for the old classification /
    # reconstruction checker but are unnecessary for a one-target regression
    # graph and can accidentally leak E_cluster/E_incident.  Keep only the
    # selected graph-level target fields created above.
    for name in [
        "truth_cluster_id", "truth_cluster_primary_id", "truth_cluster_time_slice",
        "truth_cluster_size", "truth_cluster_total_amp", "truth_cluster_energy",
        "truth_cluster_t0", "truth_cluster_t0_x", "truth_cluster_t0_y",
        "truth_cluster_t0_z",
    ]:
        if hasattr(data, name):
            delattr(data, name)

    return True

def _collapse_truth_component_time(
    items: List[Tuple[int, float, float]],
    mode: str,
    fallback: float,
) -> float:
    if not items:
        return float(fallback)
    total_energy = float(sum(e for _, e, _ in items))
    if mode == "earliest":
        return float(min(t for _, _, t in items))
    if mode == "dominant":
        return float(max(items, key=lambda item: item[1])[2])
    if total_energy > 0.0:
        return float(sum(e * t for _, e, t in items) / total_energy)
    return float(fallback)


def build_truth_feature_reco_event(
    reco_arrays: Dict[str, object],
    digit_arrays: Dict[str, object],
    digit_branch_map: Dict[str, str],
    cluster_arrays: Dict[str, object],
    cluster_branch_map: Dict[str, str],
    entry: int,
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    Dict[str, np.ndarray],
    int,
    np.ndarray,
]:
    """
    Build one truth node for each (reco pulse, accepted truth-cluster component).

    A digit still produces exactly one placeholder reco pulse.  When raw
    contributors from several accepted time slices share that digit, the pulse
    is expanded into several truth nodes at the same crystal position, each with
    its own cluster id, component energy, and component-only truth time.
    """
    x_reco_pulse_event = build_raw_node_features(reco_arrays, entry)
    n_pulses = int(x_reco_pulse_event.shape[0])

    reco_digit_indices = as_numpy_1d(reco_arrays["digitIndex"][entry]).astype(np.int64)
    reco_pulse_indices = as_numpy_1d(reco_arrays["pulseIndex"][entry]).astype(np.int64)

    y_digits = _as_1d_list(
        _get_event_value(digit_arrays, digit_branch_map, "y", entry, default=[])
    )
    n_digits = len(y_digits)
    truth_digit_indices = _pick_digit_index_list(
        digit_arrays, digit_branch_map, entry, n_digits
    )
    digit_vector_by_index = {
        int(digit_index): int(i)
        for i, digit_index in enumerate(truth_digit_indices)
    }

    energy_logical = _truth_energy_logical(args, digit_branch_map)
    energy_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, energy_logical, entry, n_digits
    )
    time_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_time", entry, n_digits
    )
    primary_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_primary_id", entry, n_digits
    )
    raw_keys_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_raw_hit_key", entry, n_digits, default=None
    )
    sat_energy_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_saturated_edep", entry, n_digits, default=None
    )
    contrib_crystal_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_crystal_id", entry, n_digits, default=None
    )
    pid_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_pid", entry, n_digits, default=None
    )
    mom_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_mom", entry, n_digits, default=None
    )
    raw_x_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_x", entry, n_digits, default=None
    )
    raw_y_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_y", entry, n_digits, default=None
    )
    raw_z_by_digit = _get_digit_nested_event_value(
        digit_arrays, digit_branch_map, "contributor_z", entry, n_digits, default=None
    )

    raw_key_to_cluster, primary_to_cluster = _get_cluster_label_maps(
        cluster_arrays=cluster_arrays,
        cluster_branch_map=cluster_branch_map,
        entry=entry,
        fallback_primary=args.truth_label_fallback_primary,
    )
    truth_info = build_truth_cluster_info_from_tree(
        cluster_arrays=cluster_arrays,
        cluster_branch_map=cluster_branch_map,
        entry=entry,
        truth_energy=args.truth_energy,
    )

    rows: List[List[float]] = []
    cluster_ids: List[int] = []
    node_to_pulse: List[int] = []

    meta_lists: Dict[str, List[float]] = {
        "pulse_index": [],
        "pulse_event_index": [],
        "digit_index": [],
        "channel": [],
        "sequence_id": [],
        "block_id": [],
        "crystal_id": [],
        "raw_hit_key": [],
        "raw_hit_primary_id": [],
        "raw_hit_crystal_id": [],
        "raw_hit_pid": [],
        "raw_hit_incident_mom": [],
        "raw_hit_x": [],
        "raw_hit_y": [],
        "raw_hit_z": [],
        "raw_hit_saturated_energy": [],
        "n_truth_contributors": [],
        "node_has_truth": [],
        "signal_truth_energy": [],
        "reco_t": [],
        "reco_amp": [],
        "reco_area": [],
    }

    reco_channels = as_numpy_1d(reco_arrays["channel"][entry]).astype(np.int64)
    reco_sequence = as_numpy_1d(reco_arrays["sequenceId"][entry]).astype(np.int64)
    reco_block = as_numpy_1d(reco_arrays["blockId"][entry]).astype(np.int64)
    reco_crystal = as_numpy_1d(reco_arrays["crystalId"][entry]).astype(np.int64)

    for ipulse in range(n_pulses):
        digit_index = int(reco_digit_indices[ipulse])
        idig = digit_vector_by_index.get(digit_index)
        reco_t = float(x_reco_pulse_event[ipulse, 2])
        reco_amp = float(x_reco_pulse_event[ipulse, 3])
        reco_area = float(x_reco_pulse_event[ipulse, 4])

        grouped: Dict[int, List[Tuple[int, float, float]]] = {}
        if idig is not None:
            n_contrib = max(
                len(energy_by_digit[idig]),
                len(time_by_digit[idig]),
                len(primary_by_digit[idig]),
                len(raw_keys_by_digit[idig]),
            )
            for icontrib in range(n_contrib):
                energy = (
                    float(energy_by_digit[idig][icontrib])
                    if icontrib < len(energy_by_digit[idig]) else 0.0
                )
                if not (energy > 0.0):
                    continue
                time_ns = (
                    float(time_by_digit[idig][icontrib])
                    if icontrib < len(time_by_digit[idig]) else reco_t
                )
                raw_key = (
                    int(raw_keys_by_digit[idig][icontrib])
                    if icontrib < len(raw_keys_by_digit[idig]) else None
                )
                primary_id = (
                    int(primary_by_digit[idig][icontrib])
                    if icontrib < len(primary_by_digit[idig]) else -1
                )
                cid = raw_key_to_cluster.get(raw_key, -1) if raw_key is not None else -1
                if cid < 0 and args.truth_label_fallback_primary:
                    cid = primary_to_cluster.get(primary_id, -1)
                grouped.setdefault(int(cid), []).append((icontrib, energy, time_ns))

        # Preserve a pure-noise/electronics pulse as one negative node.
        if not grouped:
            grouped[-1] = []

        for cid in sorted(grouped.keys(), key=lambda value: (value < 0, value)):
            items = grouped[cid]
            total_energy = float(sum(e for _, e, _ in items))
            truth_time = _collapse_truth_component_time(
                items, args.truth_time_mode,
                reco_t if args.truth_missing_time_fallback == "reco" else 0.0,
            )

            dominant_i = max(items, key=lambda item: item[1])[0] if items else -1
            primary_id = -1
            raw_key = -1
            raw_crystal = int(reco_crystal[ipulse])
            raw_pid = 0
            raw_mom = float("nan")
            raw_x = float("nan")
            raw_y = float("nan")
            raw_z = float("nan")
            raw_sat_e = total_energy

            if idig is not None and dominant_i >= 0:
                if dominant_i < len(primary_by_digit[idig]):
                    primary_id = int(primary_by_digit[idig][dominant_i])
                if dominant_i < len(raw_keys_by_digit[idig]):
                    raw_key = int(raw_keys_by_digit[idig][dominant_i])
                if dominant_i < len(contrib_crystal_by_digit[idig]):
                    raw_crystal = int(contrib_crystal_by_digit[idig][dominant_i])
                if dominant_i < len(pid_by_digit[idig]):
                    raw_pid = int(pid_by_digit[idig][dominant_i])
                if dominant_i < len(mom_by_digit[idig]):
                    raw_mom = float(mom_by_digit[idig][dominant_i])
                if dominant_i < len(raw_x_by_digit[idig]):
                    raw_x = float(raw_x_by_digit[idig][dominant_i])
                if dominant_i < len(raw_y_by_digit[idig]):
                    raw_y = float(raw_y_by_digit[idig][dominant_i])
                if dominant_i < len(raw_z_by_digit[idig]):
                    raw_z = float(raw_z_by_digit[idig][dominant_i])
                if dominant_i < len(sat_energy_by_digit[idig]):
                    raw_sat_e = float(sat_energy_by_digit[idig][dominant_i])

            rows.append([
                float(x_reco_pulse_event[ipulse, 0]),
                float(x_reco_pulse_event[ipulse, 1]),
                float(truth_time),
                float(total_energy),
                0.0,
            ])
            cluster_ids.append(int(cid))
            node_to_pulse.append(int(ipulse))

            meta_lists["pulse_index"].append(int(reco_pulse_indices[ipulse]))
            meta_lists["pulse_event_index"].append(int(ipulse))
            meta_lists["digit_index"].append(digit_index)
            meta_lists["channel"].append(int(reco_channels[ipulse]))
            meta_lists["sequence_id"].append(int(reco_sequence[ipulse]))
            meta_lists["block_id"].append(int(reco_block[ipulse]))
            meta_lists["crystal_id"].append(int(reco_crystal[ipulse]))
            meta_lists["raw_hit_key"].append(raw_key)
            meta_lists["raw_hit_primary_id"].append(primary_id)
            meta_lists["raw_hit_crystal_id"].append(raw_crystal)
            meta_lists["raw_hit_pid"].append(raw_pid)
            meta_lists["raw_hit_incident_mom"].append(raw_mom)
            meta_lists["raw_hit_x"].append(raw_x)
            meta_lists["raw_hit_y"].append(raw_y)
            meta_lists["raw_hit_z"].append(raw_z)
            meta_lists["raw_hit_saturated_energy"].append(raw_sat_e)
            meta_lists["n_truth_contributors"].append(len(items))
            meta_lists["node_has_truth"].append(1 if total_energy > 0.0 else 0)
            meta_lists["signal_truth_energy"].append(
                total_energy if cid >= 0 else 0.0
            )
            meta_lists["reco_t"].append(reco_t)
            meta_lists["reco_amp"].append(reco_amp)
            meta_lists["reco_area"].append(reco_area)

    x_truth_event = (
        np.asarray(rows, dtype=np.float32)
        if rows else np.empty((0, 5), dtype=np.float32)
    )
    cluster_id_event = np.asarray(cluster_ids, dtype=np.int64)
    node_to_pulse_np = np.asarray(node_to_pulse, dtype=np.int64)

    # truth_cluster_size is defined in graph-node units (one node per
    # digit/pulse-cluster component), while t0/position/energy remain the full
    # raw-slice observables read from ecalTruthClusters.
    if truth_info["id"].size > 0:
        truth_info["size"] = np.asarray(
            [
                int(np.count_nonzero(cluster_id_event == int(cid)))
                for cid in truth_info["id"]
            ],
            dtype=np.int64,
        )

    int_fields = {
        "pulse_index", "pulse_event_index", "digit_index", "channel",
        "sequence_id", "block_id", "crystal_id", "raw_hit_key",
        "raw_hit_primary_id", "raw_hit_crystal_id", "raw_hit_pid",
        "n_truth_contributors", "node_has_truth",
    }
    meta: Dict[str, np.ndarray] = {}
    for name, values in meta_lists.items():
        dtype = np.int64 if name in int_fields else np.float32
        meta[name] = np.asarray(values, dtype=dtype)

    event_value = np.asarray(reco_arrays["event"][entry])
    if event_value.ndim == 0:
        event_id = int(event_value)
    elif event_value.size > 0:
        event_id = int(event_value.reshape(-1)[0])
    else:
        event_id = int(entry)

    return (
        x_reco_pulse_event,
        x_truth_event,
        meta,
        cluster_id_event,
        truth_info,
        event_id,
        node_to_pulse_np,
    )


def compute_train_feature_stats_truth(
    train_files: Sequence[str],
    args: argparse.Namespace,
    min_nodes: int,
) -> Dict[str, object]:
    n_cols = 3
    total_nodes = 0
    feature_sum = np.zeros(n_cols, dtype=np.float64)
    feature_sumsq = np.zeros(n_cols, dtype=np.float64)

    for file_index, filename in enumerate(train_files):
        print(f"[read truth stats] file {file_index + 1}/{len(train_files)}: {filename}")
        f, digit_tree, _, digit_arrays, cluster_arrays, digit_branch_map, cluster_branch_map = _open_truth_arrays(filename, args)
        try:
            if args.tree not in f:
                raise RuntimeError(f"Truth mode requested but reco pulse tree '{args.tree}' not found in {filename}")
            reco_tree = f[args.tree]
            check_tree_branches(reco_tree, REQUIRED_BRANCHES)
            reco_arrays = sanitize_reco_geometry(
                reco_tree.arrays(REQUIRED_BRANCHES, library="ak"), args.reconstruction_step, filename,
                digit_arrays, digit_branch_map)
            if reco_tree.num_entries > digit_tree.num_entries:
                raise RuntimeError(
                    f"Reco tree has {reco_tree.num_entries} entries but truth digit tree has {digit_tree.num_entries}: {filename}"
                )
            for entry in range(reco_tree.num_entries):
                _, x_truth_event, _, _, _, _, _ = build_truth_feature_reco_event(
                    reco_arrays=reco_arrays,
                    digit_arrays=digit_arrays,
                    digit_branch_map=digit_branch_map,
                    cluster_arrays=cluster_arrays,
                    cluster_branch_map=cluster_branch_map,
                    entry=entry,
                    args=args,
                )
                if x_truth_event.shape[0] < min_nodes:
                    continue

                # Truth mode node-level threshold.
                keep = x_truth_event[:, 3] >= float(args.truth_node_energy_threshold_mev)
                x_truth_for_stats = x_truth_event[keep]
                if x_truth_for_stats.shape[0] < min_nodes:
                    continue

                values = train_std_features_from_raw(x_truth_event).astype(np.float64)
                total_nodes += values.shape[0]
                feature_sum += values.sum(axis=0)
                feature_sumsq += np.square(values).sum(axis=0)
        finally:
            f.close()

    if total_nodes == 0:
        raise RuntimeError("No training truth-feature nodes found; cannot compute normalization")

    mean = feature_sum / total_nodes
    variance = feature_sumsq / total_nodes - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std[std < 1.0e-12] = 1.0

    return {
        "method": "truth_mode_hybrid_reco_window_train_mean_std_for_t_logTruthEnergy_zeroPadding",
        "feature_names": ["truthTimeNs", "log1p_truthEnergy", "zeroPadding"],
        "n_train_nodes_for_stats": int(total_nodes),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def convert_one_truth_root_file(
    split_name: str,
    file_index: int,
    filename: str,
    out_dir: Path,
    args: argparse.Namespace,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    make_undirected: bool,
) -> Tuple[str, Dict[str, float]]:
    print(f"[convert truth] split={split_name} file={file_index}: {filename}")
    print("  truth mode policy: reco fitAmplitude builds seed windows; truth energy/time fill node features")
    if args.reconstruction_step != "legacy":
        print("  hierarchical regression: one PyG Data per ROOT event; seed windows become window nodes")

    graphs: List[Data] = []
    n_nodes = 0
    n_edges = 0
    n_signal_nodes = 0
    n_positive_edges = 0
    n_seed_windows_total = 0
    n_seed_windows_written = 0
    n_seed_windows_rejected_low_amp = 0
    n_regression_graphs_skipped = 0

    f, digit_tree, _, digit_arrays, cluster_arrays, digit_branch_map, cluster_branch_map = _open_truth_arrays(filename, args)
    try:
        if args.tree not in f:
            raise RuntimeError(f"Truth mode requested but reco pulse tree '{args.tree}' not found in {filename}")
        reco_tree = f[args.tree]
        check_tree_branches(reco_tree, REQUIRED_BRANCHES)
        reco_arrays = sanitize_reco_geometry(
            reco_tree.arrays(REQUIRED_BRANCHES, library="ak"), args.reconstruction_step, filename,
            digit_arrays, digit_branch_map)
        if reco_tree.num_entries > digit_tree.num_entries:
            raise RuntimeError(
                f"Reco tree has {reco_tree.num_entries} entries but truth digit tree has {digit_tree.num_entries}: {filename}"
            )

        for entry in range(reco_tree.num_entries):
            if args.max_events_per_file > 0 and len(graphs) >= args.max_events_per_file:
                break

            (
                x_reco_event,
                x_truth_event,
                meta_event,
                cluster_id_event,
                truth_info,
                event_id,
                node_to_pulse,
            ) = build_truth_feature_reco_event(
                reco_arrays=reco_arrays,
                digit_arrays=digit_arrays,
                digit_branch_map=digit_branch_map,
                cluster_arrays=cluster_arrays,
                cluster_branch_map=cluster_branch_map,
                entry=entry,
                args=args,
            )
            if x_reco_event.shape[0] == 0 or x_truth_event.shape[0] < args.min_nodes:
                continue

            if not args.disable_dynamic_seeding:
                pulse_windows = build_dynamic_seed_windows(
                    y=x_reco_event[:, 0],
                    z=x_reco_event[:, 1],
                    amp=x_reco_event[:, 3],
                    seed_threshold_mv=args.seed_threshold_mv,
                    seed_window_half_mm=args.seed_window_half_mm,
                    seed_min_total_amplitude_mv=args.seed_min_total_amplitude_mv,
                    total_scale=0.24,
                    t=x_reco_event[:, 2],
                    seed_window_time_ns=(args.edge_time_ns if args.reconstruction_step != "legacy" else -1.0),
                )
                complete_graph_edges = True
            else:
                pulse_windows = [np.arange(x_reco_event.shape[0], dtype=np.int64)]
                complete_graph_edges = False

            n_seed_windows_total += len(pulse_windows)
            window_records: List[Dict[str, object]] = []

            for graph_index_in_event, pulse_indices in enumerate(pulse_windows):
                pulse_indices = np.asarray(pulse_indices, dtype=np.int64)
                node_indices = np.nonzero(np.isin(node_to_pulse, pulse_indices))[0].astype(np.int64)

                if args.truth_node_energy_threshold_mev > 0.0:
                    node_indices = node_indices[
                        x_truth_event[node_indices, 3] >= float(args.truth_node_energy_threshold_mev)
                    ]
                if len(node_indices) < args.min_nodes:
                    continue

                positive_amp = np.maximum(
                    x_reco_event[pulse_indices, 3].astype(np.float64), 0.0
                )
                total_amp = float(np.sum(positive_amp))
                if not args.disable_dynamic_seeding and total_amp * 0.24 <= args.seed_min_total_amplitude_mv:
                    n_seed_windows_rejected_low_amp += 1
                    continue

                seed_pulse = int(pulse_indices[int(np.argmax(positive_amp))])
                seed_candidates = node_indices[node_to_pulse[node_indices] == seed_pulse]
                if seed_candidates.size == 0:
                    seed_local = int(node_indices[0])
                else:
                    seed_local = int(seed_candidates[int(np.argmax(x_truth_event[seed_candidates, 3]))])
                seed_amp = float(max(x_reco_event[seed_pulse, 3], 0.0))
                seed_energy = float(max(x_truth_event[seed_local, 3], 0.0))
                seed_total_energy = float(np.sum(np.maximum(x_truth_event[node_indices, 3], 0.0)))

                data = make_truth_data_object(
                    x_raw_event=x_truth_event,
                    meta_event=meta_event,
                    cluster_id_event=cluster_id_event,
                    node_indices=node_indices,
                    event_id=event_id,
                    file_index=file_index,
                    local_entry=entry,
                    graph_index_in_event=graph_index_in_event,
                    seed_event_index=seed_local,
                    seed_energy=seed_energy,
                    seed_total_energy=seed_total_energy,
                    truth_info=truth_info,
                    stats=stats,
                    normalize=normalize,
                    y_scale_mm=args.y_scale_mm,
                    z_scale_mm=args.z_scale_mm,
                    z_offset_mm=args.z_offset_mm,
                    edge_radius_mm=args.edge_radius_mm,
                    edge_time_ns=args.edge_time_ns,
                    edge_weight_scale_mm=args.edge_weight_scale_mm,
                    make_undirected=make_undirected,
                    self_loops=args.self_loops,
                    complete_graph_edges=complete_graph_edges,
                    regression_features=(args.reconstruction_step != "legacy"),
                )
                data.seed_amp = torch.tensor([seed_amp], dtype=torch.float32)
                data.seed_total_amp = torch.tensor([total_amp], dtype=torch.float32)
                data.seed_reco_amp = torch.tensor([seed_amp], dtype=torch.float32)
                data.seed_total_reco_amp = torch.tensor([total_amp], dtype=torch.float32)

                target_cid = _choose_regression_cluster_id(
                    cluster_id_event=cluster_id_event,
                    node_indices=node_indices,
                    seed_event_index=seed_local,
                    node_weight=x_truth_event[:, 3],
                )

                if args.reconstruction_step == "legacy":
                    if not attach_regression_task(
                        data=data,
                        reconstruction_step="legacy",
                        truth_info=truth_info,
                        target_cluster_id=target_cid,
                    ):
                        continue
                    graphs.append(data)
                    n_seed_windows_written += 1
                    n_nodes += int(data.num_nodes)
                    n_edges += int(data.edge_index.shape[1])
                    if hasattr(data, "y_node"):
                        n_signal_nodes += int(data.y_node.sum().item())
                    if hasattr(data, "y_edge"):
                        n_positive_edges += int(data.y_edge.sum().item())
                else:
                    window_records.append({
                        "data": data,
                        "node_indices": node_indices,
                        "target_cid": int(target_cid),
                        "seed_amp": seed_amp,
                        # Outer/event GNN should still be based on detector activity;
                        # use reco total amplitude rather than truth energy here.
                        "total_activity": total_amp,
                    })
                    n_seed_windows_written += 1

            if args.reconstruction_step != "legacy":
                event_data = _merge_hierarchical_event(
                    records=window_records,
                    truth_info=truth_info,
                    reconstruction_step=args.reconstruction_step,
                    graph_source="truth",
                )
                if event_data is None:
                    n_regression_graphs_skipped += 1
                    continue
                graphs.append(event_data)
                n_nodes += int(event_data.num_nodes)
                n_edges += int(event_data.edge_index.shape[1])
    finally:
        f.close()

    out_path = out_dir / output_name_for_root(split_name, file_index, filename)
    torch.save(graphs, out_path)
    print(f"[write] {out_path}  graphs={len(graphs)}")

    summary = {
        "root_file": filename,
        "pt_file": str(out_path),
        "n_graphs": int(len(graphs)),
        "n_seed_windows_total": int(n_seed_windows_total),
        "n_seed_windows_written": int(n_seed_windows_written),
        "n_seed_windows_rejected_low_amp": int(n_seed_windows_rejected_low_amp),
        "n_regression_graphs_skipped": int(n_regression_graphs_skipped),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_signal_nodes": int(n_signal_nodes),
        "n_positive_edges": int(n_positive_edges),
        "signal_node_fraction": float(n_signal_nodes / n_nodes) if n_nodes else 0.0,
        "positive_edge_fraction": float(n_positive_edges / n_edges) if n_edges else 0.0,
        "mean_nodes_per_graph": float(n_nodes / len(graphs)) if graphs else 0.0,
        "mean_edges_per_graph": float(n_edges / len(graphs)) if graphs else 0.0,
        "hierarchical_regression": bool(args.reconstruction_step != "legacy"),
    }
    return str(out_path), summary


def convert_truth_split(
    split_name: str,
    files: Sequence[str],
    out_dir: Path,
    args: argparse.Namespace,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    make_undirected: bool,
) -> Tuple[List[str], Dict[str, object]]:
    output_files: List[str] = []
    file_summaries: List[Dict[str, float]] = []
    for file_index, filename in enumerate(files):
        out_file, summary = convert_one_truth_root_file(
            split_name=split_name,
            file_index=file_index,
            filename=filename,
            out_dir=out_dir,
            args=args,
            stats=stats,
            normalize=normalize,
            make_undirected=make_undirected,
        )
        output_files.append(out_file)
        file_summaries.append(summary)

    n_graphs = sum(s["n_graphs"] for s in file_summaries)
    n_nodes = sum(s["n_nodes"] for s in file_summaries)
    n_edges = sum(s["n_edges"] for s in file_summaries)
    n_signal_nodes = sum(s["n_signal_nodes"] for s in file_summaries)
    n_positive_edges = sum(s["n_positive_edges"] for s in file_summaries)
    n_seed_windows_total = sum(s["n_seed_windows_total"] for s in file_summaries)
    n_seed_windows_written = sum(s["n_seed_windows_written"] for s in file_summaries)
    n_regression_graphs_skipped = sum(s.get("n_regression_graphs_skipped", 0) for s in file_summaries)

    split_summary = {
        "n_files": int(len(files)),
        "n_pt_files": int(len(output_files)),
        "n_graphs": int(n_graphs),
        "n_seed_windows_total": int(n_seed_windows_total),
        "n_seed_windows_written": int(n_seed_windows_written),
        "n_regression_graphs_skipped": int(n_regression_graphs_skipped),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_signal_nodes": int(n_signal_nodes),
        "n_positive_edges": int(n_positive_edges),
        "signal_node_fraction": float(n_signal_nodes / n_nodes) if n_nodes else 0.0,
        "positive_edge_fraction": float(n_positive_edges / n_edges) if n_edges else 0.0,
        "mean_nodes_per_graph": float(n_nodes / n_graphs) if n_graphs else 0.0,
        "mean_edges_per_graph": float(n_edges / n_graphs) if n_graphs else 0.0,
        "files": file_summaries,
    }
    return output_files, split_summary


def compute_train_feature_stats(
    train_files: Sequence[str],
    tree_name: str,
    min_nodes: int,
    reconstruction_step: str = "legacy",
) -> Dict[str, object]:
    n_cols = 3
    total_nodes = 0
    feature_sum = np.zeros(n_cols, dtype=np.float64)
    feature_sumsq = np.zeros(n_cols, dtype=np.float64)

    for _, _, entry, arrays in iter_root_events(train_files, tree_name, REQUIRED_BRANCHES, reconstruction_step):
        x_raw = build_raw_node_features(arrays, entry)
        if x_raw.shape[0] < min_nodes:
            continue
        values = train_std_features_from_raw(x_raw).astype(np.float64)
        total_nodes += values.shape[0]
        feature_sum += values.sum(axis=0)
        feature_sumsq += np.square(values).sum(axis=0)

    if total_nodes == 0:
        raise RuntimeError("No training nodes found; cannot compute normalization")

    mean = feature_sum / total_nodes
    variance = feature_sumsq / total_nodes - mean * mean
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std[std < 1.0e-12] = 1.0

    return {
        "method": "train_mean_std_for_t_logAmp_logArea_only",
        "feature_names": TRAIN_STD_FEATURE_NAMES,
        "n_train_nodes_for_stats": int(total_nodes),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def build_dynamic_seed_windows(
    y: np.ndarray,
    z: np.ndarray,
    amp: np.ndarray,
    seed_threshold_mv: float,
    seed_window_half_mm: float,
    seed_min_total_amplitude_mv: float,
    total_scale: float = 0.24,
    t: Optional[np.ndarray] = None,
    seed_window_time_ns: float = -1.0,
) -> List[np.ndarray]:
    n = int(len(amp))
    if n == 0:
        return []

    positive_amp = np.maximum(amp.astype(np.float64), 0.0)
    sorted_indices = sorted(range(n), key=lambda i: float(positive_amp[i]), reverse=True)
    remaining = set(range(n))
    windows: List[np.ndarray] = []

    def in_window(idx: int, center: int) -> bool:
        spatial = (
            abs(float(y[idx] - y[center])) < seed_window_half_mm
            and abs(float(z[idx] - z[center])) < seed_window_half_mm
        )
        if not spatial:
            return False
        if t is None or seed_window_time_ns <= 0.0:
            return True
        return abs(float(t[idx] - t[center])) < seed_window_time_ns

    for seed in sorted_indices:
        if seed not in remaining:
            continue
        if positive_amp[seed] <= seed_threshold_mv:
            break

        initial_members = {idx for idx in remaining if in_window(idx, seed)}
        subsequent_seeds = [
            idx
            for idx in sorted(initial_members, key=lambda i: float(positive_amp[i]), reverse=True)
            if idx != seed and positive_amp[idx] > seed_threshold_mv
        ]

        members = set(initial_members)
        # Only one expansion step: later newly added >threshold hits do not seed again.
        for sub_seed in subsequent_seeds:
            members.update(idx for idx in remaining if in_window(idx, sub_seed))

        remaining.difference_update(members)

        total_amp = float(np.sum(positive_amp[list(members)])) if members else 0.0
        if len(members) > 0 and total_amp * total_scale > seed_min_total_amplitude_mv:
            # Preserve original event order inside the graph.  This makes pulseIndex,
            # digitIndex and truth-cluster bookkeeping easier to inspect.
            windows.append(np.asarray(sorted(members), dtype=np.int64))

    return windows


def compute_truth_cluster_info(
    x_raw_event: np.ndarray,
    cluster_id_event: np.ndarray,
    amp_to_energy: float,
) -> Dict[str, np.ndarray]:
    if amp_to_energy <= 0.0:
        raise ValueError("amp_to_energy must be positive")

    valid_ids = sorted(int(cid) for cid in np.unique(cluster_id_event) if int(cid) >= 0)
    ids: List[int] = []
    sizes: List[int] = []
    total_amp: List[float] = []
    energy: List[float] = []
    t0: List[float] = []
    t0_y: List[float] = []
    t0_z: List[float] = []

    for cid in valid_ids:
        idx = np.nonzero(cluster_id_event == cid)[0]
        if idx.size == 0:
            continue
        amp_sum = float(np.sum(np.maximum(x_raw_event[idx, 3].astype(np.float64), 0.0)))
        earliest_local = int(idx[np.argmin(x_raw_event[idx, 2])])
        ids.append(cid)
        sizes.append(int(idx.size))
        total_amp.append(amp_sum)
        energy.append(amp_sum / amp_to_energy)
        t0.append(float(x_raw_event[earliest_local, 2]))
        t0_y.append(float(x_raw_event[earliest_local, 0]))
        t0_z.append(float(x_raw_event[earliest_local, 1]))

    return {
        "id": np.asarray(ids, dtype=np.int64),
        "size": np.asarray(sizes, dtype=np.int64),
        "total_amp": np.asarray(total_amp, dtype=np.float32),
        "energy": np.asarray(energy, dtype=np.float32),
        "t0": np.asarray(t0, dtype=np.float32),
        "t0_y": np.asarray(t0_y, dtype=np.float32),
        "t0_z": np.asarray(t0_z, dtype=np.float32),
    }


def build_edges(
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    amp: torch.Tensor,
    area: torch.Tensor,
    edge_radius_mm: float,
    edge_time_ns: float,
    edge_weight_scale_mm: float,
    make_undirected: bool,
    self_loops: bool,
    complete_graph: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(y.numel())
    src: List[int] = []
    dst: List[int] = []
    attr: List[List[float]] = []
    weight: List[float] = []

    if edge_radius_mm <= 0.0:
        raise ValueError("edge_radius_mm must be positive")
    if edge_weight_scale_mm <= 0.0:
        raise ValueError("edge_weight_scale_mm must be positive")

    log_amp = torch.log1p(torch.clamp(amp, min=0.0))
    log_area = torch.log1p(torch.clamp(area, min=0.0))

    for i in range(n):
        j_start = i if make_undirected else i + 1
        for j in range(j_start, n):
            if not self_loops and i == j:
                continue

            dy = float(y[j] - y[i])
            dz = float(z[j] - z[i])
            dt = float(t[j] - t[i])
            dr = math.sqrt(dy * dy + dz * dz)
            dlog_amp = float(log_amp[j] - log_amp[i])
            dlog_area = float(log_area[j] - log_area[i])

            if not complete_graph:
                if dr > edge_radius_mm:
                    continue
                if edge_time_ns > 0.0 and abs(dt) > edge_time_ns:
                    continue

            dt_scale = edge_time_ns if edge_time_ns > 0.0 else 100.0
            w = edge_weight_scale_mm / (dr + edge_weight_scale_mm)

            src.append(i)
            dst.append(j)
            attr.append([
                dy / edge_radius_mm,
                dz / edge_radius_mm,
                dr / edge_radius_mm,
                dt / dt_scale,
                dlog_amp,
                dlog_area,
            ])
            weight.append(w)

            if make_undirected and i != j:
                src.append(j)
                dst.append(i)
                attr.append([
                    -dy / edge_radius_mm,
                    -dz / edge_radius_mm,
                    dr / edge_radius_mm,
                    -dt / dt_scale,
                    -dlog_amp,
                    -dlog_area,
                ])
                weight.append(w)

    if len(src) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 6), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    return (
        torch.tensor([src, dst], dtype=torch.long),
        torch.tensor(attr, dtype=torch.float32),
        torch.tensor(weight, dtype=torch.float32),
    )


def build_edge_labels(edge_index: torch.Tensor, cluster_id: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    i = edge_index[0]
    j = edge_index[1]
    cid_i = cluster_id[i]
    cid_j = cluster_id[j]
    return ((cid_i >= 0) & (cid_i == cid_j)).to(torch.float32)


def make_data_object(
    x_raw_event: np.ndarray,
    node_indices: np.ndarray,
    arrays: Dict[str, object],
    entry: int,
    file_index: int,
    local_entry: int,
    graph_index_in_event: int,
    seed_event_index: int,
    seed_amp: float,
    seed_total_amp: float,
    truth_info: Dict[str, np.ndarray],
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    edge_radius_mm: float,
    edge_time_ns: float,
    edge_weight_scale_mm: float,
    make_undirected: bool,
    self_loops: bool,
    complete_graph_edges: bool,
    amp_to_energy: float,
    regression_features: bool = True,
) -> Data:
    x_raw_np = x_raw_event[node_indices]
    if regression_features:
        x_np, window_ref = make_reco_model_node_features(
        x_raw_np=x_raw_np,
        stats=stats,
        normalize=normalize,
        y_scale_mm=y_scale_mm,
        z_scale_mm=z_scale_mm,
        z_offset_mm=z_offset_mm,
        relative_time_scale_ns=(edge_time_ns if edge_time_ns > 0.0 else 50.0),
            amp_to_energy=amp_to_energy,
        )
    else:
        x_np = make_model_node_features(
            x_raw=x_raw_np, stats=stats, normalize=normalize,
            y_scale_mm=y_scale_mm, z_scale_mm=z_scale_mm, z_offset_mm=z_offset_mm,
        )
        reco_e = np.maximum(x_raw_np[:,3],0.0) / max(float(amp_to_energy),1.0e-12)
        window_ref = _legacy_window_physical_moments(x_raw_np[:,0],x_raw_np[:,1],x_raw_np[:,2],reco_e)

    x = torch.tensor(x_np, dtype=torch.float32)
    x_raw = torch.tensor(x_raw_np, dtype=torch.float32)

    is_signal_all = as_numpy_1d(arrays["isSignal"][entry]).astype(np.float32)
    cluster_id_all = as_numpy_1d(arrays["clusterId"][entry]).astype(np.int64)
    is_ambiguous_all = as_numpy_1d(arrays["isAmbiguous"][entry]).astype(np.int64)

    is_signal = torch.tensor(is_signal_all[node_indices], dtype=torch.float32)
    cluster_id = torch.tensor(cluster_id_all[node_indices], dtype=torch.long)
    is_ambiguous = torch.tensor(is_ambiguous_all[node_indices], dtype=torch.bool)

    if x.shape[0] != is_signal.shape[0] or x.shape[0] != cluster_id.shape[0]:
        raise RuntimeError(
            f"Vector size mismatch at entry {entry}: x={x.shape[0]}, "
            f"isSignal={is_signal.shape[0]}, clusterId={cluster_id.shape[0]}"
        )

    edge_index, edge_attr, edge_weight = build_edges(
        y=x_raw[:, 0],
        z=x_raw[:, 1],
        t=x_raw[:, 2],
        amp=x_raw[:, 3],
        area=x_raw[:, 4],
        edge_radius_mm=edge_radius_mm,
        edge_time_ns=edge_time_ns,
        edge_weight_scale_mm=edge_weight_scale_mm,
        make_undirected=make_undirected,
        self_loops=self_loops,
        complete_graph=complete_graph_edges,
    )
    y_edge = build_edge_labels(edge_index, cluster_id)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        y_node=is_signal,
        y_edge=y_edge,
        cluster_id=cluster_id,
        node_truth_cluster=cluster_id.clone(),
        is_ambiguous=is_ambiguous,
    )

    # Raw node quantities are deliberately not used by the model input, but are
    # carried through evaluation ROOT for downstream cluster reconstruction.
    data.x_raw = x_raw
    data.node_y = x_raw[:, 0].clone()
    data.node_z = x_raw[:, 1].clone()
    data.node_t = x_raw[:, 2].clone()
    data.node_amp = x_raw[:, 3].clone()
    data.node_area = x_raw[:, 4].clone()
    data.node_energy = torch.clamp(x_raw[:, 3], min=0.0) / float(amp_to_energy)

    _attach_window_reference_fields(
        data, window_ref,
        make_reco_window_features(window_ref, int(x.shape[0]), stats, y_scale_mm, z_scale_mm, z_offset_mm),
    )

    data.event = torch.tensor([int(arrays["event"][entry])], dtype=torch.long)
    data.file_index = torch.tensor([file_index], dtype=torch.long)
    data.local_entry = torch.tensor([local_entry], dtype=torch.long)
    # Avoid metadata names containing "index" for PyG batching; Data.__inc__
    # treats such names as index-like and would offset them by num_nodes.
    data.graph_in_event = torch.tensor([graph_index_in_event], dtype=torch.long)
    data.seed_node = torch.tensor([seed_event_index], dtype=torch.long)
    data.seed_amp = torch.tensor([seed_amp], dtype=torch.float32)
    data.seed_total_amp = torch.tensor([seed_total_amp], dtype=torch.float32)

    data.original_node_index = torch.tensor(node_indices.astype(np.int64), dtype=torch.long)
    data.pulse_index = torch.tensor(as_numpy_1d(arrays["pulseIndex"][entry]).astype(np.int64)[node_indices], dtype=torch.long)
    data.digit_index = torch.tensor(as_numpy_1d(arrays["digitIndex"][entry]).astype(np.int64)[node_indices], dtype=torch.long)
    data.channel = torch.tensor(as_numpy_1d(arrays["channel"][entry]).astype(np.int64)[node_indices], dtype=torch.long)
    data.sequence_id = torch.tensor(as_numpy_1d(arrays["sequenceId"][entry]).astype(np.int64)[node_indices], dtype=torch.long)
    data.block_id = torch.tensor(as_numpy_1d(arrays["blockId"][entry]).astype(np.int64)[node_indices], dtype=torch.long)
    data.crystal_id = torch.tensor(as_numpy_1d(arrays["crystalId"][entry]).astype(np.int64)[node_indices], dtype=torch.long)

    # Full-event truth-cluster observables.  These are graph-level vectors.
    data.truth_cluster_id = torch.tensor(truth_info["id"], dtype=torch.long)
    data.truth_cluster_size = torch.tensor(truth_info["size"], dtype=torch.long)
    data.truth_cluster_total_amp = torch.tensor(truth_info["total_amp"], dtype=torch.float32)
    data.truth_cluster_energy = torch.tensor(truth_info["energy"], dtype=torch.float32)
    data.truth_cluster_t0 = torch.tensor(truth_info["t0"], dtype=torch.float32)
    data.truth_cluster_t0_y = torch.tensor(truth_info["t0_y"], dtype=torch.float32)
    data.truth_cluster_t0_z = torch.tensor(truth_info["t0_z"], dtype=torch.float32)

    return data


def output_name_for_root(split_name: str, file_index: int, root_file: str) -> str:
    stem = Path(root_file).stem
    return f"{split_name}_{file_index:03d}_{stem}.pt"


def convert_one_root_file(
    split_name: str,
    file_index: int,
    filename: str,
    out_dir: Path,
    tree_name: str,
    truth_clusters_tree: str,
    truth_energy: str,
    reconstruction_step: str,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    edge_radius_mm: float,
    edge_time_ns: float,
    edge_weight_scale_mm: float,
    make_undirected: bool,
    self_loops: bool,
    dynamic_seeding: bool,
    seed_threshold_mv: float,
    seed_window_half_mm: float,
    seed_min_total_amplitude_mv: float,
    amp_to_energy: float,
    min_nodes: int,
    max_events_per_file: int,
) -> Tuple[str, Dict[str, float]]:
    print(f"[convert] split={split_name} file={file_index}: {filename}")
    if reconstruction_step != "legacy":
        print("  hierarchical regression: one PyG Data per ROOT event; seed windows become window nodes")

    graphs: List[Data] = []
    n_nodes = 0
    n_edges = 0
    n_signal_nodes = 0
    n_positive_edges = 0
    n_seed_windows_total = 0
    n_seed_windows_written = 0
    n_seed_windows_rejected_low_amp = 0
    n_regression_graphs_skipped = 0

    with uproot.open(filename) as f:
        if tree_name not in f:
            raise RuntimeError(f"Tree '{tree_name}' not found in {filename}")

        tree = f[tree_name]
        check_tree_branches(tree, REQUIRED_BRANCHES)
        arrays = sanitize_reco_geometry(tree.arrays(REQUIRED_BRANCHES, library="ak"), reconstruction_step, filename)

        regression_cluster_arrays = None
        regression_cluster_branch_map: Dict[str, str] = {}
        if reconstruction_step != "legacy":
            if truth_clusters_tree not in f:
                raise RuntimeError(
                    f"Regression step {reconstruction_step} requires tree "
                    f"'{truth_clusters_tree}' in {filename}"
                )
            cluster_tree = f[truth_clusters_tree]
            if cluster_tree.num_entries < tree.num_entries:
                raise RuntimeError(
                    f"Reco tree has {tree.num_entries} entries but {truth_clusters_tree} "
                    f"has only {cluster_tree.num_entries}: {filename}"
                )
            regression_cluster_branch_map = resolve_branch_aliases(
                cluster_tree,
                TRUTH_CLUSTER_BRANCH_ALIASES,
                required=regression_required_cluster_logicals(reconstruction_step),
                context=truth_clusters_tree,
            )
            regression_cluster_arrays = cluster_tree.arrays(
                _branches_from_map(regression_cluster_branch_map), library="ak"
            )

        for entry in range(tree.num_entries):
            if max_events_per_file > 0 and len(graphs) >= max_events_per_file:
                break

            x_raw_event = build_raw_node_features(arrays, entry)
            if x_raw_event.shape[0] < min_nodes:
                continue

            cluster_id_event = as_numpy_1d(arrays["clusterId"][entry]).astype(np.int64)
            if reconstruction_step != "legacy":
                assert regression_cluster_arrays is not None
                truth_info = build_truth_cluster_info_from_tree(
                    cluster_arrays=regression_cluster_arrays,
                    cluster_branch_map=regression_cluster_branch_map,
                    entry=entry,
                    truth_energy=truth_energy,
                )
            else:
                truth_info = compute_truth_cluster_info(
                    x_raw_event=x_raw_event,
                    cluster_id_event=cluster_id_event,
                    amp_to_energy=amp_to_energy,
                )

            if dynamic_seeding:
                windows = build_dynamic_seed_windows(
                    y=x_raw_event[:, 0],
                    z=x_raw_event[:, 1],
                    amp=x_raw_event[:, 3],
                    seed_threshold_mv=seed_threshold_mv,
                    seed_window_half_mm=seed_window_half_mm,
                    seed_min_total_amplitude_mv=seed_min_total_amplitude_mv,
                    t=x_raw_event[:, 2],
                    seed_window_time_ns=(edge_time_ns if reconstruction_step != "legacy" else -1.0),
                )
                complete_graph_edges = True
            else:
                windows = [np.arange(x_raw_event.shape[0], dtype=np.int64)]
                complete_graph_edges = False

            n_seed_windows_total += len(windows)
            window_records: List[Dict[str, object]] = []

            for graph_index_in_event, node_indices in enumerate(windows):
                node_indices = np.asarray(node_indices, dtype=np.int64)
                if len(node_indices) < min_nodes:
                    continue

                positive_amp = np.maximum(x_raw_event[node_indices, 3].astype(np.float64), 0.0)
                total_amp = float(np.sum(positive_amp))
                if dynamic_seeding and total_amp * 0.24 <= seed_min_total_amplitude_mv:
                    n_seed_windows_rejected_low_amp += 1
                    continue

                seed_local = int(node_indices[int(np.argmax(positive_amp))])
                seed_amp = float(max(x_raw_event[seed_local, 3], 0.0))

                data = make_data_object(
                    x_raw_event=x_raw_event,
                    node_indices=node_indices,
                    arrays=arrays,
                    entry=entry,
                    file_index=file_index,
                    local_entry=entry,
                    graph_index_in_event=graph_index_in_event,
                    seed_event_index=seed_local,
                    seed_amp=seed_amp,
                    seed_total_amp=total_amp,
                    truth_info=truth_info,
                    stats=stats,
                    normalize=normalize,
                    y_scale_mm=y_scale_mm,
                    z_scale_mm=z_scale_mm,
                    z_offset_mm=z_offset_mm,
                    edge_radius_mm=edge_radius_mm,
                    edge_time_ns=edge_time_ns,
                    edge_weight_scale_mm=edge_weight_scale_mm,
                    make_undirected=make_undirected,
                    self_loops=self_loops,
                    complete_graph_edges=complete_graph_edges,
                    amp_to_energy=amp_to_energy,
                    regression_features=(reconstruction_step != "legacy"),
                )

                target_cid = _choose_regression_cluster_id(
                    cluster_id_event=cluster_id_event,
                    node_indices=node_indices,
                    seed_event_index=seed_local,
                    node_weight=x_raw_event[:, 3],
                )

                if reconstruction_step == "legacy":
                    if not attach_regression_task(
                        data=data,
                        reconstruction_step="legacy",
                        truth_info=truth_info,
                        target_cluster_id=target_cid,
                    ):
                        continue
                    graphs.append(data)
                    n_seed_windows_written += 1
                    n_nodes += int(data.num_nodes)
                    n_edges += int(data.edge_index.shape[1])
                    if hasattr(data, "y_node"):
                        n_signal_nodes += int(data.y_node.sum().item())
                    if hasattr(data, "y_edge"):
                        n_positive_edges += int(data.y_edge.sum().item())
                else:
                    window_records.append({
                        "data": data,
                        "node_indices": node_indices,
                        "target_cid": int(target_cid),
                        "seed_amp": seed_amp,
                        "total_activity": total_amp,
                    })
                    n_seed_windows_written += 1

            if reconstruction_step != "legacy":
                event_data = _merge_hierarchical_event(
                    records=window_records,
                    truth_info=truth_info,
                    reconstruction_step=reconstruction_step,
                    graph_source="reco",
                    reco_arrays=arrays,
                    reco_entry=entry,
                )
                if event_data is None:
                    n_regression_graphs_skipped += 1
                    continue
                graphs.append(event_data)
                n_nodes += int(event_data.num_nodes)
                n_edges += int(event_data.edge_index.shape[1])

    out_path = out_dir / output_name_for_root(split_name, file_index, filename)
    torch.save(graphs, out_path)
    print(f"[write] {out_path}  graphs={len(graphs)}")

    summary = {
        "root_file": filename,
        "pt_file": str(out_path),
        "n_graphs": int(len(graphs)),
        "n_seed_windows_total": int(n_seed_windows_total),
        "n_seed_windows_written": int(n_seed_windows_written),
        "n_seed_windows_rejected_low_amp": int(n_seed_windows_rejected_low_amp),
        "n_regression_graphs_skipped": int(n_regression_graphs_skipped),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_signal_nodes": int(n_signal_nodes),
        "n_positive_edges": int(n_positive_edges),
        "signal_node_fraction": float(n_signal_nodes / n_nodes) if n_nodes else 0.0,
        "positive_edge_fraction": float(n_positive_edges / n_edges) if n_edges else 0.0,
        "mean_nodes_per_graph": float(n_nodes / len(graphs)) if graphs else 0.0,
        "mean_edges_per_graph": float(n_edges / len(graphs)) if graphs else 0.0,
        "hierarchical_regression": bool(reconstruction_step != "legacy"),
    }
    return str(out_path), summary


def convert_split(
    split_name: str,
    files: Sequence[str],
    out_dir: Path,
    tree_name: str,
    truth_clusters_tree: str,
    truth_energy: str,
    reconstruction_step: str,
    stats: Optional[Dict[str, object]],
    normalize: bool,
    y_scale_mm: float,
    z_scale_mm: float,
    z_offset_mm: float,
    edge_radius_mm: float,
    edge_time_ns: float,
    edge_weight_scale_mm: float,
    make_undirected: bool,
    self_loops: bool,
    dynamic_seeding: bool,
    seed_threshold_mv: float,
    seed_window_half_mm: float,
    seed_min_total_amplitude_mv: float,
    amp_to_energy: float,
    min_nodes: int,
    max_events_per_file: int,
) -> Tuple[List[str], Dict[str, object]]:
    output_files: List[str] = []
    file_summaries: List[Dict[str, float]] = []

    for file_index, filename in enumerate(files):
        out_file, summary = convert_one_root_file(
            split_name=split_name,
            file_index=file_index,
            filename=filename,
            out_dir=out_dir,
            tree_name=tree_name,
            truth_clusters_tree=truth_clusters_tree,
            truth_energy=truth_energy,
            reconstruction_step=reconstruction_step,
            stats=stats,
            normalize=normalize,
            y_scale_mm=y_scale_mm,
            z_scale_mm=z_scale_mm,
            z_offset_mm=z_offset_mm,
            edge_radius_mm=edge_radius_mm,
            edge_time_ns=edge_time_ns,
            edge_weight_scale_mm=edge_weight_scale_mm,
            make_undirected=make_undirected,
            self_loops=self_loops,
            dynamic_seeding=dynamic_seeding,
            seed_threshold_mv=seed_threshold_mv,
            seed_window_half_mm=seed_window_half_mm,
            seed_min_total_amplitude_mv=seed_min_total_amplitude_mv,
            amp_to_energy=amp_to_energy,
            min_nodes=min_nodes,
            max_events_per_file=max_events_per_file,
        )
        output_files.append(out_file)
        file_summaries.append(summary)

    n_graphs = sum(s["n_graphs"] for s in file_summaries)
    n_nodes = sum(s["n_nodes"] for s in file_summaries)
    n_edges = sum(s["n_edges"] for s in file_summaries)
    n_signal_nodes = sum(s["n_signal_nodes"] for s in file_summaries)
    n_positive_edges = sum(s["n_positive_edges"] for s in file_summaries)
    n_seed_windows_total = sum(s["n_seed_windows_total"] for s in file_summaries)
    n_seed_windows_written = sum(s["n_seed_windows_written"] for s in file_summaries)
    n_regression_graphs_skipped = sum(s.get("n_regression_graphs_skipped", 0) for s in file_summaries)

    split_summary = {
        "n_files": int(len(files)),
        "n_pt_files": int(len(output_files)),
        "n_graphs": int(n_graphs),
        "n_seed_windows_total": int(n_seed_windows_total),
        "n_seed_windows_written": int(n_seed_windows_written),
        "n_regression_graphs_skipped": int(n_regression_graphs_skipped),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_signal_nodes": int(n_signal_nodes),
        "n_positive_edges": int(n_positive_edges),
        "signal_node_fraction": float(n_signal_nodes / n_nodes) if n_nodes else 0.0,
        "positive_edge_fraction": float(n_positive_edges / n_edges) if n_edges else 0.0,
        "mean_nodes_per_graph": float(n_nodes / n_graphs) if n_graphs else 0.0,
        "mean_edges_per_graph": float(n_edges / n_graphs) if n_graphs else 0.0,
        "files": file_summaries,
    }

    return output_files, split_summary


def main() -> None:
    args = parse_args()
    if args.truth_mode:
        args.graph_source = "truth"

    train_files = read_file_list(args.train_list)
    val_files = read_file_list(args.val_list)
    test_files = read_file_list(args.test_list)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dynamic_seeding = not args.disable_dynamic_seeding

    print("[config]")
    print(f"  train files                  : {len(train_files)}")
    print(f"  val files                    : {len(val_files)}")
    print(f"  test files                   : {len(test_files)}")
    print(f"  output dir                   : {out_dir}")
    print(f"  graph source                 : {args.graph_source}")
    print(f"  reconstruction step          : {args.reconstruction_step}")
    if args.graph_source == "reco" and args.reconstruction_step == "1":
        print("  Step-1 incident time scale   : detector/reco waveform clock")
        print("  Step-1 incident time formula : fitPeakTimeNs(max crystal) - [truthCrystalTime - truthIncidentTime]")
        print("  truth crystal time reference : saturated-energy-weighted rawHitTimeNs")
    print(f"  reco tree                    : {args.tree}")
    if args.graph_source == "truth":
        print(f"  truth digits tree            : {args.truth_digits_tree}")
        print(f"  truth clusters tree          : {args.truth_clusters_tree}")
        print(f"  truth energy                 : {args.truth_energy}")
        print(f"  truth time mode              : {args.truth_time_mode}")
        print(f"  truth missing time fallback  : {args.truth_missing_time_fallback}")
        print("  truth window source          : reco fitAmplitude")
    print(f"  y scale [mm]                 : {args.y_scale_mm}")
    print(f"  z scale [mm]                 : {args.z_scale_mm}")
    print(f"  z offset [mm]                : {args.z_offset_mm}")
    print(f"  dynamic seeding              : {dynamic_seeding}")
    print(f"  seed threshold [mV]          : {args.seed_threshold_mv}")
    print(f"  seed window half [mm]        : {args.seed_window_half_mm}")
    print(f"  seed min total amp [mV]      : {args.seed_min_total_amplitude_mv}")
    print(f"  edge radius/norm [mm]        : {args.edge_radius_mm}")
    print(f"  edge weight scale [mm]       : {args.edge_weight_scale_mm}")
    print(f"  edge time [ns]               : {args.edge_time_ns}")
    print(f"  amp / energy                 : {args.amp_to_energy}")
    print(f"  normalize t/log              : {not args.no_normalize}")
    print("  output policy                : one ROOT file -> one .pt file")
    if args.reconstruction_step != "legacy":
        print("  regression graph unit        : one ROOT event with multiple seed windows")
        print("  Step-1 position target       : incident y,z only (x fixed by ECAL entrance plane)")

    stats = None
    
    if not args.no_normalize:
        if args.normalization_json:
            print(
                "[stats] loading fixed normalization from "
                f"{args.normalization_json}"
            )
    
            stats = load_normalization_stats(
                filename=args.normalization_json,
                graph_source=args.graph_source,
            )
    
        elif args.graph_source == "truth":
            print(
                "[stats] computing truth normalization "
                "from training data"
            )
    
            stats = compute_train_feature_stats_truth(
                train_files=train_files,
                args=args,
                min_nodes=args.min_nodes,
            )
    
        else:
            print(
                "[stats] computing reco normalization "
                "from training data"
            )
    
            stats = compute_train_feature_stats(
                train_files=train_files,
                tree_name=args.tree,
                min_nodes=args.min_nodes,
                reconstruction_step=args.reconstruction_step,
            )
    
        for name, mean, std in zip(
            stats["feature_names"],
            stats["mean"],
            stats["std"],
        ):
            print(
                f"  {name:20s} "
                f"mean={mean: .6g} std={std: .6g}"
            )

    make_undirected = not args.directed

    outputs: Dict[str, List[str]] = {}
    summaries: Dict[str, Dict[str, object]] = {}

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"[convert split] {split_name}: {len(files)} ROOT files")
        if args.graph_source == "truth":
            out_files, summary = convert_truth_split(
                split_name=split_name,
                files=files,
                out_dir=out_dir,
                args=args,
                stats=stats,
                normalize=not args.no_normalize,
                make_undirected=make_undirected,
            )
        else:
            out_files, summary = convert_split(
                split_name=split_name,
                files=files,
                out_dir=out_dir,
                tree_name=args.tree,
                truth_clusters_tree=args.truth_clusters_tree,
                truth_energy=args.truth_energy,
                reconstruction_step=args.reconstruction_step,
                stats=stats,
                normalize=not args.no_normalize,
                y_scale_mm=args.y_scale_mm,
                z_scale_mm=args.z_scale_mm,
                z_offset_mm=args.z_offset_mm,
                edge_radius_mm=args.edge_radius_mm,
                edge_time_ns=args.edge_time_ns,
                edge_weight_scale_mm=args.edge_weight_scale_mm,
                make_undirected=make_undirected,
                self_loops=args.self_loops,
                dynamic_seeding=dynamic_seeding,
                seed_threshold_mv=args.seed_threshold_mv,
                seed_window_half_mm=args.seed_window_half_mm,
                seed_min_total_amplitude_mv=args.seed_min_total_amplitude_mv,
                amp_to_energy=args.amp_to_energy,
                min_nodes=args.min_nodes,
                max_events_per_file=args.max_events_per_file,
            )
        outputs[split_name] = out_files
        summaries[split_name] = summary
        print(
            f"[summary] {split_name}: n_graphs={summary['n_graphs']} "
            f"n_pt_files={summary['n_pt_files']} mean_nodes={summary['mean_nodes_per_graph']:.2f}"
        )

    metadata = {
        "converter": "convert_ecal_reco_pulses_filelists_to_pyg.py",
        "invalid_geometry_policy": "reject affected regression events with cached truth targets; drop invalid legacy pulses before seeding and normalization",
        "geometry_coordinate_sentinel_limit_mm": GEOMETRY_COORD_LIMIT_MM,
        "graph_source": args.graph_source,
        "reconstruction_step": args.reconstruction_step,
        "task": (
            "node_edge_classification" if args.reconstruction_step == "legacy"
            else f"hierarchical_event_regression_step_{args.reconstruction_step}"
        ),
        "tree": args.tree,
        "truth_digits_tree": args.truth_digits_tree,
        "truth_clusters_tree": args.truth_clusters_tree,
        "truth_energy": args.truth_energy,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "model_feature_names": TRUTH_MODEL_FEATURE_NAMES if args.graph_source == "truth" else RECO_MODEL_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "edge_weight": f"edge_weight = {args.edge_weight_scale_mm} / (dr + {args.edge_weight_scale_mm})",
        "truth_mode_policy": "hybrid: reco ecalRecoPulses fitAmplitude builds seed windows; node features use ecalTruthDigits contributor truth time/energy matched by digitIndex",
        "truth_time_mode": args.truth_time_mode,
        "truth_missing_time_fallback": args.truth_missing_time_fallback,
        "node_label": (
            "reco: data.y_node = isSignal; truth: accepted truth-cluster membership"
            if args.reconstruction_step == "legacy" else None
        ),
        "edge_label": (
            "data.y_edge = 1 if same non-negative clusterId else 0"
            if args.reconstruction_step == "legacy" else None
        ),
        "cluster_label": "data.cluster_id is diagnostic bookkeeping in regression modes",
        "regression": {
            "step1_target_names": STEP1_TARGET_NAMES,
            "step1_target_definition": (
                "predict residuals dt=tmin(prompt window)-t0, dy=yc(prompt window)-y_inc, dz=zc(prompt window)-z_inc; "
                "reconstruct t0=tmin-dt, y_inc=yc-dy, z_inc=zc-dz. In reco mode t0 is detector-clock incident time."
            ),
            "step1_incident_time_reference_crystal": "ecalTruthClusters.mostSignificantCrystalId",
            "step1_incident_time_truth_weight": "rawHitSaturatedEdepMeV",
            "step1_incident_time_diagnostics": [
                "target_incident_time_truth_ns",
                "target_time_reference_reco_pulse_ns",
                "target_time_reference_truth_crystal_ns",
                "target_time_reference_truth_delta_ns",
                "target_incident_time_dominant_ns",
                "target_incident_time_earliest_ns",
                "target_time_reference_crystal_id",
                "target_time_reference_pulse_index",
                "target_time_reference_match_level",
            ],
            "step2_target_names": STEP2_TARGET_NAMES,
            "step2_track_feature_names": STEP2_TRACK_FEATURE_NAMES,
            "step2_pdg_storage": "data.step2_pdg (integer; use embedding/one-hot, not as a continuous coordinate)",
            "step2_track_momentum_provided": False,
            "step2_training_track_proxy": (
                "truth ecalTruthClusters incidentX/Y/Z + incidentDirX/Y/Z; replace with Straw-predicted entry/direction for realistic train/eval when available"
                if args.reconstruction_step == "2" else None
            ),
            "target_cluster_association": (
                "multi-object hierarchical: create one target object for every represented valid primary; "
                "for each primary the prompt target cluster is its earliest represented truth cluster t0"
            ),
            "step1_position_target": "residual dy,dz relative to energy-weighted prompt-window centroid; incident x remains diagnostic only",
        },
        "raw_node_quantities": ["node_y", "node_z", "node_t", "node_amp", "node_area", "cluster_id"],
        "truth_mode_node_quantities": ["node_truth_time", "node_truth_energy", "truth_padding_zero_in_node_area", "node_reco_t", "node_reco_amp", "node_reco_area", "node_has_truth", "n_truth_contributors"],
        "hierarchical_event_fields": (
            [
                "node_window_local", "num_windows", "window_x", "window_tmin_ns",
                "window_centroid_y_mm", "window_centroid_z_mm", "window_esum_mev",
                "window_role_target", "window_is_prompt", "window_is_delayed",
                "window_target_cluster_id", "window_primary_id", "window_seed_amp", "window_total_amp",
                "window_truth_t0_ns", "association_target", "num_association_pairs",
                "num_targets", "target_prompt_window_local",
                "target_prompt_cluster_id", "target_primary_id"
            ] if args.reconstruction_step != "legacy" else []
        ),
        "window_feature_names": WINDOW_FEATURE_NAMES if args.reconstruction_step != "legacy" else [],
        "window_role_names": WINDOW_ROLE_NAMES if args.reconstruction_step != "legacy" else {},
        "window_role_definition": (
            "0=other/noise, 1=window containing a prompt cluster of any target object, 2=truth-associated nonprompt activity (typically delayed)"
            if args.reconstruction_step != "legacy" else None
        ),
        "window_association_definition": (
            "association_target has one label for each unordered window pair i<j; 1 if the two windows share any truth primaryId, else 0. The model never receives this target as input."
            if args.reconstruction_step != "legacy" else None
        ),
        "multi_object_output": (
            "data.y has shape [K,D], data.num_targets=[K], and data.target_prompt_window_local maps each truth object to its prompt-window anchor; K can exceed one per event"
            if args.reconstruction_step != "legacy" else None
        ),
        "truth_cluster_quantities": [
            "regression targets are KxD per hierarchical event (one row per represented primary); truth IDs are validation metadata only and are not model inputs"
        ],
        "position_scaling": {
            "y_feature": "y / y_scale_mm",
            "z_feature": "(z - z_offset_mm) / z_scale_mm",
            "y_scale_mm": args.y_scale_mm,
            "z_scale_mm": args.z_scale_mm,
            "z_offset_mm": args.z_offset_mm,
        },
        "normalization": stats,
        "dynamic_seeding": {
            "enabled": dynamic_seeding,
            "seed_threshold_mv": args.seed_threshold_mv,
            "seed_window_half_mm": args.seed_window_half_mm,
            "seed_min_total_amplitude_mv": args.seed_min_total_amplitude_mv,
            "window_source": "fitAmplitude from ecalRecoPulses for both reco and truth graph_source",
            "one_step_subsequent_seed_expansion": True,
            "hierarchical_time_window_ns": (
                args.edge_time_ns if args.reconstruction_step != "legacy" else -1.0
            ),
            "hierarchical_time_window_policy": (
                "for regression steps 1/2, spatial seed windows also require |dt| < edge_time_ns so prompt and delayed pulses can form separate window nodes"
                if args.reconstruction_step != "legacy" else "legacy spatial-only seed windows"
            ),
        },
        "amp_to_energy": args.amp_to_energy,
        "edge_radius_mm": args.edge_radius_mm,
        "edge_time_ns": args.edge_time_ns,
        "edge_weight_scale_mm": args.edge_weight_scale_mm,
        "directed": args.directed,
        "self_loops": args.self_loops,
        "output_policy": (
            "one ROOT file -> one .pt; legacy stores seed-window graphs, regression steps 1/2 store one multi-object hierarchical Data per ROOT event"
        ),
        "input_lists": {
            "train_list": str(Path(args.train_list).resolve()),
            "val_list": str(Path(args.val_list).resolve()),
            "test_list": str(Path(args.test_list).resolve()),
        },
        "input_files": {"train": train_files, "val": val_files, "test": test_files},
        "outputs": outputs,
        "summaries": summaries,
    }

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[write] {metadata_path}")
    print("[done]")


if __name__ == "__main__":
    main()
