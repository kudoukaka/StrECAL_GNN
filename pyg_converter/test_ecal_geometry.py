"""Regression coverage for failed ECAL geometry lookups in old ROOT files."""
import unittest

import awkward as ak
import numpy as np

from convert_ecal_reco_pulses_filelists_to_pyg import (
    sanitize_reco_geometry, valid_geometry_mask, build_raw_node_features,
    build_dynamic_seed_windows,
)


def pulses():
    # Keep IDs non-contiguous to detect accidental relabeling after filtering.
    return {
        "event": ak.Array([16474, 16475]),
        "pulseIndex": ak.Array([[4, 7, 9], [12]]),
        "digitIndex": ak.Array([[2, 0, 5], [8]]),
        "channel": ak.Array([[2282324480, 0, 2282333184], [2282324480]]),
        "sequenceId": ak.Array([[1, -1, 2], [1]]),
        "blockId": ak.Array([[0, -1, 0], [0]]),
        "crystalId": ak.Array([[1, -1, 2], [1]]),
        "y": ak.Array([[30.7, 1e10, 51.2], [30.7]]),
        "z": ak.Array([[7701.2, 1e10, 7680.7], [7701.2]]),
        "fitPeakTimeNs": ak.Array([[1621., 1621., 1622.], [20.]]),
        "fitAmplitude": ak.Array([[100., 1000., 50.], [100.]]),
        "fitArea": ak.Array([[100., 1000., 50.], [100.]]),
        "clusterId": ak.Array([[3, 3, 3], [4]]),
    }


class GeometryTest(unittest.TestCase):
    def test_geometry_rejects_finite_sentinel_nan_and_invalid_ids(self):
        y = np.array([30., 1e10, np.nan, 30., 30., 30.])
        z = np.full(6, 7700.)
        channel = np.array([2282324480]*3 + [0, 2282324480, 1])
        ids = np.array([1, 1, 1, 1, -1, 1])
        self.assertEqual(valid_geometry_mask(y, z, channel, ids).tolist(), [True, False, False, False, False, False])

    def test_legacy_drops_hit_and_keeps_all_metadata_aligned(self):
        clean = sanitize_reco_geometry(pulses())
        self.assertEqual(ak.to_list(clean["pulseIndex"]), [[4, 9], [12]])
        self.assertEqual(ak.to_list(clean["digitIndex"]), [[2, 5], [8]])
        self.assertEqual(ak.to_list(clean["event"]), [16474, 16475])
        x = build_raw_node_features(clean, 0)
        windows = build_dynamic_seed_windows(x[:, 0], x[:, 1], x[:, 3], 10., 35., 0.)
        self.assertEqual(len(windows), 1)
        self.assertEqual(sorted(windows[0].tolist()), [0, 1])
        self.assertEqual(x[:, 3].tolist(), [100., 50.])

    def test_regression_rejects_contaminated_event_but_keeps_entry_alignment(self):
        for step in ("1", "2"):
            clean = sanitize_reco_geometry(pulses(), step)
            self.assertEqual(ak.to_list(clean["pulseIndex"]), [[], [12]])
            self.assertEqual(build_raw_node_features(clean, 0).shape, (0, 5))
            self.assertEqual(ak.to_list(clean["event"]), [16474, 16475])

    def test_invalid_truth_digit_rejects_event_even_without_invalid_reco_pulse(self):
        clean = sanitize_reco_geometry(pulses())
        truth = {"y": ak.Array([[1e10], [30.7]]), "z": ak.Array([[1e10], [7701.2]])}
        result = sanitize_reco_geometry(clean, "1", digit_arrays=truth, digit_branch_map={"y": "y", "z": "z"})
        self.assertEqual(ak.to_list(result["pulseIndex"]), [[], [12]])

    def test_valid_input_unchanged_for_truth_and_reco(self):
        clean = sanitize_reco_geometry(pulses())
        for step in ("legacy", "1", "2"):
            result = sanitize_reco_geometry(clean, step)
            for name in clean:
                self.assertEqual(ak.to_list(result[name]), ak.to_list(clean[name]))


if __name__ == "__main__":
    unittest.main()
