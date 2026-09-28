ROOT diagnostics now write each histogram as a 2×2 canvas: total, electron,
muon, pion. The original histogram name remains the total; explicit `_total`,
`_electron`, `_muon`, `_pion` histograms and `_by_primary` canvases are also saved.
PNGs go into `<eval-stem>_figures/` beside the input ROOT file.

Run the existing macro as before:

```bash
root -l -b -q 'reconstruct_ecal_clusters_from_eval_root.cxx("/path/to/eval.root")'
```

Object diagnostics use the matched truth object's PID (oracle/baseline plots
use their truth object's PID directly). Electron/muon include antiparticles;
pion includes charged and neutral pions. Total includes all species. Window
plots use the window's dominant truth primary; pair/edge plots use the source
window's dominant truth primary. Unknown/unrepresented primaries appear only
in total. Older files without `window_primary_id` require evaluation to be
rerun for these window/pair/edge species panels.

Truth shower validation is produced directly by ICEDUST RunECALTruthMaker,
in the separate `ecalTruthValidation` tree and `validation` plot directory.
No shower branches are added to PyG or evaluation ROOT files; no PyG regeneration
is needed for this validation.
