# Muon/electron/pion timing diagnosis — 2026-09-24

## Scope and conclusion

Compared current `mum`, `em`, and `pim` runs named `{species}_20to120MeV_10cmPlateGun_0To10deg`, under `ReconData/runs/ecal_clustering`, using their best test evaluations: epochs 048, 049 and 049 respectively.

Muon timing has learned a nonconstant correction, unlike the nearly inactive early electron position head. Its remaining error combines a large tail, inadequately constrained sub-nanosecond corrections under the current loss scaling, and a mixture of prompt timing and inference from late-only activity. Further training might improve some metrics, but the measurements do not support treating this as simply a late-starting head that only needs more epochs.

No production code, datasets or checkpoints were modified. All thresholds below define diagnostic subsets, not proposed automatic rejection cuts.

## Full test evaluation, same statistical definitions

`resolution` below is population standard deviation of residuals, using all matched objects regardless of histogram display range. Oracle metrics use the truth prompt-window anchor for every truth object, before reconstructed-object selection. They measure different populations.

| Species | Epoch | Matched objects | Matched bias (ns) | Matched std (ns) | Matched central 68% halfwidth (ns) | All-object oracle RMSE (ns) |
|---|---:|---:|---:|---:|---:|---:|
| electron | 049 | 90719 | 0.000142 | 0.008072 | 0.001237 | 0.060102 |
| muon | 048 | 65265 | 0.423286 | 8.898125 | 0.729637 | 272.151461 |
| pion | 049 | 60039 | 1.072162 | 3.235446 | 0.002346 | 250.360881 |

Central 68% halfwidth means `(q84 - q16)/2`, not a Gaussian fit and not a standard deviation. All three datasets use `graph_source=truth`, `truth_time_mode=earliest`. These numbers refer to that input configuration.

The default time matching gate is 100 ns in `evaluate_ecal_gnn.py`. The matched residual ranges are correspondingly approximately [-100,100] ns for muon and pion; their oracle distributions extend to thousands of ns. A small matched width cannot by itself demonstrate convergence of all-object timing.

### Muon tails

- 781/65265 matched objects (1.1967%) have `|reco_t0-truth_t0| > 50 ns`; these contribute 83.7892% of the uncentered residual square sum.
- 1703 objects (2.6094%) exceed 10 ns and contribute 98.1124% of that square sum.
- The diagnostic subset `|residual| <= 5 ns` has 62876 objects, mean 0.30678 ns and std 0.95940 ns. This is a truncated subset, not a replacement performance metric.
- Only 252 matched muon objects use a reconstructed anchor different from their truth anchor. They contribute about 0.50% of the total centered residual square sum; anchor switching alone does not explain the large width.

## Target scales and late reference windows

The model learns `dt = window_tmin - truth_t0`, then reconstructs `t0 = window_tmin - predicted_dt`.

| Species | Training dt mean (ns) | Training dt std (ns) |
|---|---:|---:|
| electron | 0.004706 | 0.267687 |
| muon | 49.154654 | 368.154757 |
| pion | 33.871189 | 578.444979 |

The checkpoints use standardized SmoothL1/Huber loss with beta 1, time weight 2 and total regression weight 11. For muon errors below 368 ns, the weighted per-object time contribution is `(2/11) * 0.5 * (error_ns/368.154757)^2`. A 1 ns error contributes only **6.7073e-7**. Improving ordinary sub-nanosecond timing therefore changes this objective very little relative to fitting large errors and other regression dimensions.

In the full muon test set:

- 65618/67836 objects (96.73%) have `|window_tmin-truth_t0| <= 1 ns`. Their baseline correction has mean 0.011415 ns and std 0.011423 ns.
- 2207 objects (3.2534%) have `window_tmin-truth_t0 > 30 ns`. Their mean required correction is 1520.30 ns.
- 2110 of these 2207 late-reference cases have only **one** window; the other 97 have two.
- None of these 2207 objects have an additional same-primary role-2 window. Consequently all are classified **NO delayed** by the current summary. That label does not mean the sole observed activity is prompt.
- Of the matched muon sample, 1081 objects have a truth reference window delayed by >30 ns. They contribute 70.36% of the centered matched residual square sum. The analogous pion sample has only 178 such matched objects.

An illustrative muon event is test file index 0, event 36795, graph index 24937: truth incident time 16.54144 ns; sole window tmin 11210.38672 ns; window truth-cluster t0 11210.38672 ns; oracle prediction 11143.66895 ns. The reference hit activity is thousands of ns later than the requested incident-time target.

The converter's `_select_hierarchical_targets` chooses the earliest **represented** valid cluster per primary and the largest-activity window containing it. `_object_target_payload` then takes `incident_time` as the truth-mode time target. This permits a late represented cluster to provide the anchor for a much earlier incident time. Existing event trees establish the mismatch in temporal scales; they do not establish whether each missing prompt observation resulted from physical activity, seeding thresholds, reconstruction, or a truth-assignment issue. That requires an upstream event audit, not blind rejection of late hits.

There is also the opposite failure: 1192 muon objects have a small required correction (`|dt| <= 1 ns`) but truth incident time >30 ns. Their oracle mean error is -691.49 ns. The model often subtracts a large delay even when the target is already near the observed time. This is evidence that the current mixed timing problem includes ambiguity beyond ordinary local time refinement.

## Epoch comparison and activation check

Whole-validation timing MAE falls from 49.35 ns (epoch 1) to 26.79 ns (20), then fluctuates: 25.42 (35), 25.71 (48), 25.78 (49), 27.21 (50). Thus there is no electron-Z-like onset of sustained rapid improvement near the end.

Re-evaluated epochs 005, 015, 025, 035, 048 and 050 on the same 4096 evenly spaced events from the current first muon test shard (33858 events). No matching or residual selection was applied. The truth-defined `|dt| <= 1 ns` subset contains the same 3954 objects for all epochs.

| Epoch | All-object sample MAE (ns) | Small-required-correction subset central 68% halfwidth (ns) | Timing-head all-zero hidden-window fraction |
|---|---:|---:|---:|
| 005 | 34.9652 | 1.2363 | 0% |
| 015 | 29.7503 | 0.3050 | 0.4964% |
| 025 | 24.3875 | 0.5537 | 0% |
| 035 | 24.1335 | 0.8552 | 0.0226% |
| 048 | 25.4948 | 0.7444 | 0.1579% |
| 050 | 22.0591 | 1.1274 | 0.0451% |

This small test sample is diagnostic, not a checkpoint-selection dataset. Its all-object tails differ from full validation, so the apparent epoch-050 sample improvement does not establish a better full-test checkpoint. It does demonstrate that coarse error improvement and ordinary timing precision are not improving together.

The current muon epoch-048 timing head has 27 of 128 hidden units active on at least one sampled window. Its full-test predicted correction std is 323.11 ns, with correction-target correlation 0.70393. It is not producing a single constant correction for almost every input.

## Why pion's narrower distribution is not sufficient evidence of better learning

Pion's matched bias is +1.072 ns. About 96.04% of **all** pion truth objects receive a correction within 0.001 ns of the same value, -1.065601 ns. A constant offset preserves a narrow core standard deviation while shifting its bias. Pion also has a 250.36 ns all-object RMSE and thousands-of-ns tails. Its smaller matched std does not establish a well-converged unbiased timing estimator.

## Recommended next work

1. Audit late-only reference events upstream and define whether the task is observable prompt timing or inference of an earlier incident time from delayed activity. Keep genuinely different cases separately labeled and evaluated; do not silently delete valid late hits to improve the summary.
2. Separate or balance these two timing tasks in supervision. For local prompt timing, use a meaningful physical time scale in the loss, rather than allowing rare thousand-ns corrections to set the scale for every object. A fixed-ns Huber term plus balanced subgroups is an experiment to validate; simply shrinking the scale for the existing mixed population can let the tail dominate gradients again.
3. Track time bias, central-68% width, full RMSE, tail fraction and efficiency on validation, and save a timing-specific checkpoint alongside the combined-regression best checkpoint. Prefer these checks over assuming longer training or a ReLU change alone will fix this case.
4. Existing checkpoints' normalization defines their output units; changing only their de-normalization is not a valid loss fix. A changed training objective must preserve the output mapping or be accompanied by a deliberate model/target conversion and training validation.

## Artifacts and reproduction

- `export_timing.C`: exports the raw event-tree fields for all three full test samples.
- `mum.csv`, `em.csv`, `pim.csv`: raw exported object records.
- `summarize_timing.py`, `raw_statistics.txt`: distribution and subgroup statistics.
- `check_muon_epochs.py`, `.json`, `.txt`: reproducible checkpoint forward passes and hidden-activation check. Uses `torch.load(..., mmap=True)` for the input shard; no cached sample is needed.
- `timing_comparison.png`: validation trajectory and matched absolute-error tail curves.

From `/home/comet/davidz/ReconDev`, run the ROOT macro with `root -l -b -q ecal_clustering_gnn_trainer/diagnostics/particle_timing_comparison/export_timing.C`. Python scripts use `/home/comet/davidz/PyTorch/miniforge3/envs/gnn-pyg/bin/python`.
