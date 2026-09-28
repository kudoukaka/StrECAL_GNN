# Why the pion core stays narrow while the muon core broadens

## First distinguish the core from the full small-correction population

These are all-object oracle predictions from the full test ROOT files, without matching or cuts on prediction error. Small correction means `abs(truth_anchor_tmin - truth_t0) <= 1 ns`.

| Gun model | Small-correction objects | Residual MAE (ns) | Residual std (ns) | Central 68% halfwidth (ns) | Fraction with error >50 ns |
|---|---:|---:|---:|---:|---:|
| muon 048 | 65618 | 13.6116 | 162.3647 | 0.741516 | 1.5819% |
| pion 049 | 61048 | 14.8182 | 132.9373 | 0.002385 | 1.8739% |

Pion's small-correction population is not free from large errors. It has a very narrow majority peak plus substantial tails. Matching excludes many of these tail objects. Its narrow central width should not be interpreted as uniformly accurate small-correction prediction.

For the subset additionally requiring truth incident time <=30 ns (a truth-based diagnostic, not an inference cut):

| Model | Objects | True correction central 68% halfwidth | Predicted correction central 68% halfwidth | Residual central 68% halfwidth |
|---|---:|---:|---:|---:|
| muon | 64426 | 0.003834 ns | 0.713478 ns | 0.713921 ns |
| pion | 59768 | 0.002329 ns | effectively zero | 0.002330 ns |

For pion, a nearly constant correction preserves the baseline's narrow core while shifting its bias. For muon, the predicted correction varies much more than the tiny true correction, and its variation dominates the reconstructed core width.

The full small-correction group also contains objects with late absolute incidence but a small local correction. For these objects, oracle MAEs are 691.49 ns (muon, 1192 objects) and 649.51 ns (pion, 1280 objects). Both models can misinterpret a late absolute time as a large delay requiring subtraction.

## The head's output mapping

The timing head is a Linear-ReLU-Linear MLP, followed by target de-normalization:

`predicted_dt = target_mean + target_std * (w * ReLU(W * features + b) + c)`.

When all hidden activations are zero, only `target_mean + target_std*c` remains. From the best checkpoints:

- pion: bias-only physical correction = **-1.065601 ns**;
- muon: bias-only physical correction = **+66.717606 ns**.

The pion constant measured in the raw ROOT predictions equals this bias-only value. An inactive hidden layer makes the correction locally insensitive to features and hidden/output weights; changing the final bias can still shift every such prediction. A constant shift changes the residual bias, not its width. This is not an explicitly designed prompt/delayed separation, and changing the encoder or activation boundaries can move objects out of that region.

Muon requires a nonzero hidden contribution to cancel its +66.72 ns bias-only value and produce an ordinary correction near zero. Event-dependent errors in that contribution can therefore broaden its ordinary timing peak. This parameterization explains sensitivity of the current checkpoint; it does not prove that the offset itself is an independently causal defect.

## Direct activation checks

Ran each best checkpoint on 4096 evenly spaced events from its own first test shard. These are separate diagnostic samples, not identical physics events across gun datasets. Activations were indexed at each truth object's anchor window.

| Truth-defined subset | Pion model: all 128 hidden outputs zero | Muon model: all 128 hidden outputs zero |
|---|---:|---:|
| Small correction, all incident times | 98.0592% (4019 objects) | 0.1012% (3954 objects) |
| Small correction, incident time <=30 ns | **99.9746%** (3936 objects) | **0%** (3882 objects) |
| Large correction >30 ns | 0% (68 objects) | 1.4085% (142 objects) |

For the ordinary early small-correction group, the median local derivative of the pion head with respect to its input representation is exactly zero; the muon derivative is nonzero. Among units active on more than 1% of a group, pion has 0 in that ordinary group and 10 in the large-correction group. Muon has 2 and 13 respectively, with **both ordinary-group units also active in the large-correction group**.

Thus the pion model's ordinary region is presently a bias-only plateau, while muon's ordinary and large-correction predictions share active paths. This supports different sensitivity to head/representation changes. It is not complete isolation: pion's final output bias is shared, its ReLU boundaries can move, and late-incidence small-correction objects often activate the wrong large-correction behavior.

The same phenomenon appears even for muon-PID targets inside the pion-gun sample: 99.7748% of 888 small-correction muon targets have a zero hidden layer. In the muon-gun model, none of 3868 small-correction muon-PID targets do. The contrast is therefore not explained just by saying that a muon intrinsically needs a broader local timing correction than a pion. The two models and their training distributions differ.

## Loss scaling and interpretation limits

The time target std is 368.15 ns for muon and 578.44 ns for pion. Both models assign a very small objective cost to ns-scale errors under the standardized Huber loss. The larger pion scale does not protect its prompt timing; it makes its approximately 1 ns constant bias especially cheap.

Both models use shared representations and a shared timing head across the timing populations. Activation patterns can make the current response differ sharply by input. Shared active units provide a route for parameter updates on large-delay objects to affect ordinary objects. A snapshot activation/Jacobian measurement establishes local sensitivity, not the historical cause of every training change. Establishing that cause requires a controlled training comparison with the large-delay time loss excluded or independently modeled, holding the data, initialization and other objectives fixed.

Different target species mixtures and feature normalizations were established in the preceding reports, but no controlled retraining here isolates which difference caused the two runs to settle into different activation regimes.

## Reproduction

`compare_timing_populations.py` generates `timing_populations.json` from the full exported ROOT records. `compare_timing_heads.py {pim,mum}` performs independent read-only forward passes on 4096 evenly spaced events from each dataset's first test shard, indexes hidden activations at the truth target windows, and measures the head's local derivative with respect to its input features. It does not train or alter checkpoints.
