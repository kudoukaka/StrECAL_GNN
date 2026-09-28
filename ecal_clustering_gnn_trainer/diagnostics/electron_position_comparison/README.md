# Electron Y/Z resolution comparison (2026-09-23)

## Inputs and definitions

- Previous run: `ReconData/runs/ecal_clustering/em_regression_20to120MeV_10cmPlateGun_0To10deg/step1`, epoch 036.
- Current run: `ReconData/runs/ecal_clustering/em_20to120MeV_10cmPlateGun_0To10deg/step1`, epochs 025, 035, 046, 049.
- Resolution below means population standard deviation of the residual, with all raw TTree entries included, independent of histogram display limits.
- Matched statistics use each run's own matched objects. Oracle statistics use every truth object's truth prompt-window anchor, without reconstructed-object selection.
- Position targets are corrections: `dy = centroid_y - truth_y`, `dz = centroid_z - truth_z`; reconstructed position is `centroid - predicted_correction`.

## Full raw-tree statistics

| Run / epoch | Matched Y std (mm) | Matched Z std (mm) | Oracle Y correction std (mm) | Oracle Z correction std (mm) | Correlation of Z correction with its target |
|---|---:|---:|---:|---:|---:|
| Previous / 036 | 4.70747 | 4.75422 | 5.25143 | 5.26132 | 0.71760 |
| Current / 025 | 7.11165 | 7.10415 | 0.15636 | 0.22352 | 0.06396 |
| Current / 035 | 4.74370 | 7.08224 | 5.20830 | 0.34624 | 0.06607 |
| Current / 046 | 4.72081 | 7.06869 | 5.49443 | 0.35178 | 0.07710 |
| Current / 049 | 4.71036 | 6.98936 | 5.27818 | 1.14297 | 0.16563 |

Previous/current all-truth centroid baseline Z standard deviations are 7.28291 / 7.29465 mm. Their matched baseline Z standard deviations are 7.12398 / 7.14103 mm (current epoch 049).

The previous ROOT histogram omitted display-range overflows: its displayed Y/Z resolutions were 4.69371 / 4.69829 mm. Recomputing all entries raises them slightly to 4.70747 / 4.75422 mm. This explains only a small fraction of the difference, not the current approximately 7 mm Z result.

The current model's Z correction remains nearly constant through epoch 046. It therefore reconstructs Z close to the centroid baseline. Its Y correction has already acquired the approximately 5 mm spread of the previous model. These observations also hold for oracle predictions, so candidate selection alone cannot explain the effect.

## Training evidence

Source: each run's `step1/logs/metrics.jsonl`. These MAEs are oracle-object metrics and must not be equated numerically with matched-object residual standard deviations.

| Run / epoch | Validation Y MAE (mm) | Validation Z MAE (mm) |
|---|---:|---:|
| Previous / 036 | 3.7613 | 3.7794 |
| Current / 025 | 5.5851 | 5.5728 |
| Current / 027 | 4.4544 | 5.5737 |
| Current / 028 | 3.8626 | 5.5731 |
| Current / 046 | 3.7737 | 5.5677 |
| Current / 049 | 3.7704 | 5.5114 |
| Current / 050 | 3.8287 | 5.3652 |

The first validation MAE below 5 mm occurs at epochs 7 / 11 for previous Y / Z, and epoch 27 / not yet reached for current Y / Z. Training and validation show the same Z plateau. This is delayed learning / underfitting, rather than a model that first learned Z and subsequently lost test-set performance.

![Training and validation curves](training_yz.png)

## Configuration and preprocessing checks

- Checkpoint training arguments agree except for dataset and output directories: seed 12345, AdamW learning rate 0.001, 50 epochs, identical architecture and loss settings.
- Both Y and Z have weight 2 in `step1_loss_weights`; the standardized Huber loss treats them symmetrically.
- Previous target Y/Z standard deviations: 7.15539 / 7.31150 mm. Current: 7.16794 / 7.27227 mm. There is no large relative target scaling change.
- The position head outputs two unconstrained linear values; there is no final ReLU clipping negative corrections.
- Both datasets specify 500 mm Y/Z input scales and a 7650 mm Z offset. Feature names, target definitions, graph source and window definitions agree.
- Input time standard deviation did change: 2.565064 -> 3.605301 ns, with mean 7.219129 -> 7.234139 ns. Training samples and geometry policy also changed. Same seed and hyperparameters therefore do not constitute a controlled identical-input replay.
- These configuration checks do not identify what triggered the longer optimization plateau. The forward-activation checks below directly establish widespread inactive ReLUs at epoch 025, but do not identify which preprocessing or optimization change caused that state.

## Same-new-data cross-check and position-head activations

Selected 4096 evenly spaced events from the current dataset's first test shard (45362 events), giving 4108 truth objects and 4110 windows. Both checkpoints were evaluated on exactly this sample, without candidate matching or energy selection. The old checkpoint was tested both directly and after transforming the three normalized absolute-time inputs back to its original training normalization: node column 2 and window columns 2/3. No spatial inputs or targets were changed.

| Checkpoint | Input time normalization | Oracle Y std (mm) | Oracle Z std (mm) |
|---|---|---:|---:|
| Previous 036 | Current, unadapted | 4.96002 | 5.01793 |
| Previous 036 | Adapted to previous training | 4.78017 | 4.78732 |
| Current 025 | Current | 7.16964 | 7.04750 |
| Current 046 | Current | 4.79961 | 7.04013 |
| Current 049 | Current | 4.78864 | 6.96259 |
| Current 050 | Current | 4.92331 | 6.80235 |

The previous model still reconstructs both coordinates well from the new data. This strongly argues against the new dataset having lost the Z information needed to achieve the earlier resolution. The sample is bounded to one shard; these oracle numbers are not full-test matched summary values.

A forward hook inspected the 128-dimensional ReLU output of the position head:

| Checkpoint | Units positive on any sampled window | Units positive on >1% of windows | Windows with all 128 outputs zero |
|---|---:|---:|---:|
| Previous 036, adapted | 31 | 10 | 0% |
| Current 025 | 5 | 0 | **99.8297%** |
| Current 046 | 6 | 5 | 0% |
| Current 049 | 6 | 3 | 0.0730% |
| Current 050 | 7 | 5 | 0.0243% |

For a window whose entire hidden activation is zero, this head's prediction is just its final linear bias. The position loss can update that bias, but has zero gradient through the inactive ReLU into the preceding head layer and shared representation for that window. Other task heads can still change the shared representation, so this inactivity need not be permanent. Epoch 025 is strong direct evidence of an early inactive-head plateau. By epoch 046, the head is active but its Z correction still has little useful variation; that later Z underfitting should not be described as every ReLU still being inactive.

The same sample gives Z correction standard deviations of 5.24019 mm (old adapted), 0.20696 mm (current 046), 1.09589 mm (current 049), and 1.80151 mm (current 050). This is consistent with the full-tree diagnosis and confirms that Z learning has only begun near the end of the current run.

## Suggested next experiments

1. Continue from current epoch 050 on the existing train/validation split, monitoring Y and Z separately. Z starts improving near the end; recovery to the previous resolution is not guaranteed without measuring it.
2. Keep separate position validation metrics/checkpoints alongside the combined regression criterion. Epoch 049 regression loss is 0.1996764 versus epoch 050's 0.1997207, even though epoch 050 Z MAE is better. Aggregate selection can hide this improvement.
3. The observed inactive position head motivates a controlled new-training experiment with a nonzero-negative-slope activation, such as LeakyReLU, specifically in its hidden layer. Do not replace activations in an existing checkpoint and treat the resulting inference as an equivalent model. Compare convergence and final per-axis validation performance before accepting this as a fix.
4. If needed, compare seeds and a position-focused training control, then test a separate local-coordinate input scale or a direct local-shape path into the position head. Current local position differences are divided by the same 500 mm scale as global coordinates; changing that is an experiment, not a verified fix.

## Reproduction

From `/home/comet/davidz/ReconDev`:

```bash
root -l -b -q ecal_clustering_gnn_trainer/diagnostics/electron_position_comparison/compare_electron.C
```

`raw_tree_statistics.txt` records the complete numerical output. No training, converter or evaluation implementation was modified for this diagnosis.

The same-sample check is reproducible with:

```bash
/home/comet/davidz/PyTorch/miniforge3/envs/gnn-pyg/bin/python ecal_clustering_gnn_trainer/diagnostics/electron_position_comparison/crosscheck_electron.py
```

This caches the 4096-event sample at `/tmp/electron_new_test_sample.pt`; delete that cache before rerunning if the underlying dataset changes. It writes `crosscheck_electron.json`. `crosscheck_electron.txt` contains the recorded stdout from the diagnostic run.
