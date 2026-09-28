# ECAL physics-aware hierarchical GNN pipeline

This package is the updated Step-1 / Step-2 multi-object hierarchical ECAL GNN.
It supersedes the previous absolute-`t0/y/z` regression design.

## 1. Step-1 residual targets

Step 1 now trains on

```text
[dt, dy, dz, dir_x, dir_y, dir_z, E_cluster, E_max]
```

with

```text
dt = tmin(prompt seed window) - t0
dy = yc(prompt seed window) - y_inc
dz = zc(prompt seed window) - z_inc
```

`yc,zc` are energy/activity-weighted prompt-window centroids. Evaluation reconstructs

```text
t0_pred = tmin - dt_pred
y_pred  = yc - dy_pred
z_pred  = zc - dz_pred
```

so reported resolutions remain in physical ns/mm.

For reco mode, the absolute `t0` used to build `dt` is the detector-clock target:

```text
t_entry_detector = fitPeakTimeNs(reference pulse)
                   - (truth crystal hit time - truth ECAL-entry time)
```

## 2. Source-specific 8D pulse features

Truth and reco feature builders are separate functions.

Truth mode:

```text
[y_abs, z_abs, t_abs, dy_window, dz_window, dt_window,
 log(1+E_truth), E_truth/E_window]
```

Reco mode currently uses the same geometric layout but an amplitude-derived energy proxy:

```text
[y_abs, z_abs, t_abs, dy_window, dz_window, dt_window,
 log(1+E_reco_proxy), E_reco_proxy/E_window]
```

The reco function is intentionally independent so future waveform variables such as width,
fit chi2, template timing, CFD timing, etc. can be appended without modifying truth mode.

## 3. 13D window features

```text
1  yc
2  zc
3  tmin
4  tmean
5  log(1+Esum)
6  log(1+Emax)
7  log(1+Npulse)
8  sigmaY
9  sigmaZ
10 sigmaYZ = energy-weighted covariance(y,z)
11 Emax/Esum
12 y_max_hit - yc
13 z_max_hit - zc
```

Spatial quantities are scaled before entering the Window Encoder. `tmin/tmean` use the
training-time normalization already produced by the converter.

## 4. Hierarchical GNN

```text
Pulse nodes
  -> Pulse Encoder
  -> Pulse GNN x N
  -> ADD+MEAN+MAX pooling by seed window
  -> Window Encoder + Window Fusion
  -> Window embeddings
  -> Event GNN x M
  -> Window-role head + window-window association head
```

The Event GNN default is hybrid:

```text
|Delta t_window| <= 30 ns : both directions i <-> j
|Delta t_window| >  30 ns : earlier -> later only
```

Options:

```bash
--event-edge-mode hybrid
--event-hybrid-bidirectional-ns 30
```

## 5. Prompt-only Step-1 context

For Step 1, delayed same-primary windows are suppressed. For anchor window `i`, member
window `j` receives weight

```text
w_ij = P_same(i,j) * P_prompt(j) * exp(-|dt_ij|/tau)
```

and is hard-rejected outside the prompt timing gate. The anchor itself has weight 1.
Default:

```bash
--step1-prompt-time-gate-ns 30
```

The prompt object representation is parallel association-weighted
`ADD + MEAN + MAX` pooling.

Step 2 does **not** use this prompt restriction. It uses the full same-primary object
context, allowing delayed decay/stopping topology to contribute to `E_incident`.

## 6. Separate Step-1 heads

Step 1 now has four independent heads:

```text
Timing head    -> dt
Position head  -> dy,dz
Direction head -> dx,dy,dz direction components
Energy head    -> E_cluster,E_max
```

Default loss weights:

```text
time      2
position  2
direction 1
Ecluster  1
Emax      1
```

All regression targets are still standardized with training-set mean/std before Huber/MSE/L1.

## 7. Stats sampling

Trainer statistics use a random sample of 100k training events by default:

```bash
--stats-max-events 100000
--stats-seed 12345
```

Use `--stats-max-events -1` for a full scan.

## Convert

```bash
cd /home/comet/davidz/ReconDev/pyg_converter
sbatch sbatch_ecal_pyg_pipeline.sh <setup> truth 1
# or
sbatch sbatch_ecal_pyg_pipeline.sh <setup> reco 1
```

Check:

```bash
python check_ecal_pyg_dataset.py \
  /home/comet/davidz/ReconDev/ReconData/datasets/ecal_clustering/processed/pyg_step1/<setup> \
  --split train --max-files 1 --strict
```

For Step 1, the checker should report `node_dim=8`, `window_dim=13` and residual target statistics.

## Train

```bash
cd /home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer
sbatch sbatch_train_ecal_gnn_gpu.sh <setup> 1
```

Example depth overrides:

```bash
sbatch sbatch_train_ecal_gnn_gpu.sh <setup> 1 \
  --message-mlp-depth 3 \
  --event-message-mlp-depth 3 \
  --step1-timing-head-depth 3 \
  --step1-position-head-depth 3 \
  --step1-direction-head-depth 3 \
  --step1-energy-head-depth 3
```

Step 2:

```bash
sbatch sbatch_train_ecal_gnn_gpu.sh <setup> 2
```

Step 2 inputs remain

```text
full ECAL object context + full event context
+ track ECAL-entry (x,y,z,dx,dy,dz)
+ PDG embedding
```

No track momentum is used.

## Evaluate

```bash
sbatch sbatch_eval_ecal_gnn_gpu.sh <setup> 1 \
  /home/comet/davidz/ReconDev/ReconData/runs/ecal_clustering/<setup>/step1/checkpoints/best.pt \
  test
```

The evaluator converts residual predictions back to absolute `t0,y,z` before object matching
and ROOT output. Candidate `E_cluster < 20 MeV` is rejected by default.

## Important compatibility note

The Step-1 dataset and checkpoints from the previous 5D-node/6D-window absolute-target model
are **not compatible** with this version. Regenerate the PyG dataset and retrain.
