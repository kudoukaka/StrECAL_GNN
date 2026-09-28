# Follow-up: physical delayed-only activity and pion subset performance

## Physical interpretation

Prompt energy deposition can exist without a retained prompt pulse/window. Finite thresholds, inactive material and spatial/time selection can suppress the observable prompt activity while later decay or secondary interactions produce a retained signal. Charged particles normally ionize as they traverse active material; delayed-only graphs alone do not prove absence of prompt physical energy deposition.

Free muon mean lifetime is approximately 2.197 microseconds ([PDG](https://pdg.lbl.gov/2025/listings/rpp2025-list-muon.pdf)). Negative muons in matter also undergo nuclear capture, competing with decay and changing the effective lifetime ([MuCAP/MuLAN](https://cds.cern.ch/record/704768/files/0401005.pdf)). Charged pion free lifetime is approximately 26.03 ns ([PDG](https://pdg.lbl.gov/2025/listings/rpp2025-list-pi-plus-minus.pdf)); stopped negative pions undergo nuclear absorption ([Geant4](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/hadronic/Stopping/piabs.html)). Microsecond activity in a pion-gun sample cannot automatically be identified as the pion's own decay time; secondary muons and nuclear secondary transport are possibilities that require truth-track/process identification.

## The effective reconstruction-window threshold

Both metadata files specify seed threshold 8 mV, minimum total amplitude 60 mV, and a 30 ns time-window setting. Current converter code calls `build_dynamic_seed_windows(..., total_scale=0.24)` and retains a window only when `sum(positive_amp) * 0.24 > 60`.

Thus the effective unscaled amplitude-sum threshold is **250 mV**, in addition to a seed above 8 mV. Using the configured 17.96 amplitude/MeV calibration gives approximately **13.92 MeV of amplitude-derived energy proxy**, not an exact truth deposited-energy threshold. This is an offline graph/window requirement, not evidence of a hardware trigger setting. There are additional digitization and reconstruction stages upstream.

## Four upstream spot checks

`inspect_late_examples.C` reads original truthmaker trees for four late events. Results are in `late_examples.txt`.

| Gun dataset / test-file-0 event | Target PID | Target incident kinetic energy (MeV) | Target incident time (ns) | Truth-cluster first time (ns) | Cluster deposited energy (MeV) |
|---|---:|---:|---:|---:|---:|
| mum / 36795 | 13 | 10.35084 | 16.54144 | 11210.38672 | 44.00788 |
| mum / 33742 | 13 | 2.42975 | 26.11226 | 9795.49805 | 35.16079 |
| pim / 35050 | 13 | 8.62003 | 12.02052 | 11551.74219 | 40.98663 |
| pim / 21672 | 13 | 7.59691 | 26.08774 | 10059.43359 | 30.09381 |

All four have zero *recorded digit contributor* energy in the first 30 ns after target incidence. This is **not** a measurement of all original Geant4 prompt deposits: hits without retained digits need not be present in this tree. These spot checks establish that the missing early contribution predates GNN inference and is already absent from the inspected digit truth; determining whether it was physically absent, below a digitization threshold, or otherwise unrecorded requires original simulation hits/trajectories and detector-response audit.

The low recorded incident kinetic energies also show why the folder label `20to120MeV` must not be equated automatically with the kinetic energy deposited promptly in active crystals.

## Truth PID composition of the evaluated objects

`check_target_pid.C` reads `truth_pid` directly from the evaluation ROOT trees; full output is in `target_pid.txt`.

| Pion-gun target PID | Total objects | Objects requiring dt >30 ns |
|---|---:|---:|
| -211 (pi-) | 46264 | 45 |
| 13 (mu-) | 14977 | **1259** |
| 11 (electron) | 1193 | 3 |
| Other | 86 | 0 |

Of 1307 large-delay pion-gun targets, **96.33% have muon truth PID**. The dataset label describes the gun, not necessarily the particle represented by each reconstruction target. Exact parentage and creation processes were not reconstructed in this follow-up, so a specific decay chain should not be asserted solely from these PID counts.

## Does the pion model only learn the small corrections?

Computed from all 62520 oracle objects, with `dt = truth_anchor_tmin - truth_t0`. Constant correction means within 0.001 ns of -1.065601 ns.

| Truth-defined subset | Objects | Near-constant predictions | Matched fraction |
|---|---:|---:|---:|
| abs(dt) <=1 ns | 61048 | **98.08%** | **97.88%** |
| dt >30 ns | 1307 | **0.61%** | **13.62%** |

The remaining 165 objects lie between these diagnostic ranges. Within the small-correction group with incident time <=30 ns, 99.93% receive the constant prediction. Since reconstructed t0 equals tmin minus predicted dt, a correction of -1.065601 ns produces approximately a +1.07 ns time bias while preserving the narrow baseline core. This is closer to a constant-offset baseline than learned event-specific prompt corrections.

The model does not completely ignore large corrections: their predicted correction median is 579.30 ns and prediction/target correlation is 0.4631. Accuracy is poor (all-object MAE 787.06 ns for this subset), and most are not matched. Thus the matched summary predominantly displays easy small-correction objects, even though the model attempts a coarse large-delay prediction.

Reproduction: `pion_subgroups.py` writes the numbers shown in `pion_subgroups.json`. No production thresholds or event selection were modified.
