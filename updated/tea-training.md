# Staged HandNet forest training and evaluation

HandNet now links a complete native Tea forest CPU/API path through its pinned
[DepthTrack submodule](depthtrack.md). Training new HandNet models and measuring
their quality are separate stages from implementing that inference backend.

## Current state — 2026-11-15 16:35 UTC

The repaired trainer and conversion/reference pipeline are available in
[Tea's `handnet-forest-training` branch](https://github.com/twerdster/tea/tree/handnet-forest-training),
starting with [reviewed commit `27ae728`](https://github.com/twerdster/tea/commit/27ae728).
All **eight pilots** have completed. Their **48 validation configurations** were
compared on the same frozen **300 validation frames**; final test data did not
select these settings. The [Tea pilot report](https://github.com/twerdster/tea/blob/handnet-forest-training/docs/handnet-pilots.md)
defines the experiment grid, metric, foreground policies and limitations.

| Pilot comparison | Probe scale | Mean fingertip F1@15mm |
| --- | ---: | ---: |
| Best learned-background (eight-class) pilot | 25000 | 0.236501 |
| Best seven-class pilot | 25000 | 0.227339 |

The best eight-class pilot used 20 split decisions and a 0.5 probability threshold.
These are pilot validation selection scores, not final forest results or a
classification-accuracy percentage. Scale 25000 and learned background were
selected for full-tree training preparation. **Final deployment depth and
probability threshold remain to be selected** from validation predictions after
all three full trees are available; the pilot's 20/.5 settings are not frozen as
the final configuration.

The full-training stage has begun with feature conversion for tree 0, started
**2026-11-15 16:27:33 UTC**. No complete full-tree models, final three-tree ensemble
metrics or final-test results are available at this dated snapshot. Training
remains sequential under the existing GPU queue, budget and recurring monitor.
Failures retain evidence and need a diagnosis before a replacement run; the
monitor does not expand the budget or control unrelated GPU jobs.

The original source checks passed:
[Tea tooling CI](https://github.com/twerdster/tea/actions/runs/43717599628) for
`27ae728`, and [DepthTrack CI](https://github.com/twerdster/depthtrack/actions/runs/43719231425)
for its pinned `6bbbe74` backend. Implementation CI and synthetic parity checks
are distinct from pilot validation and the pending full-model evaluation.

Official randomized frame splits do not establish participant-independent
generalization. The validation scores above do not reproduce the BMVC 2015 paper
metric, and they do not establish measured temporal tracking performance.

## Reproducible stages

1. Freeze official training/validation/test membership and source frame identities.
   Official randomized frame splits alone do not establish participant-independent
   generalization.
2. Prepare class-aware training samples with separate frame, pixel and coordinate
   seeds. Compare historical probe 50 and spatial-scale hypotheses 5000/25000/50000
   on validation data; no best scale is established by implementation tests.
3. Train each tree with its own recorded feature/sample diversity. Tea CLI depth 19
   means 20 split decisions; the output tree index is not an RNG seed. Preserve
   atomic depth checkpoints and each tree's exact `Coords.co`.
4. Freeze a native forest manifest with model/coordinate checksums, depth units,
   float32 feature rules, image geometry, class semantics and tree aggregation.
5. Compare actual validation predictions, individual trees and the ensemble.
   Select depth and postprocessing thresholds without using final test data.
6. Evaluate the selected configuration on untouched test data. Report natural-
   distribution segmentation metrics, foreground errors and fingertip localization
   recall/coverage, including missed present fingertips.

The [Tea deep-training guide](https://github.com/twerdster/tea/blob/27ae728/docs/deep-training.md)
contains bounded preparation/training commands, memory estimates, checkpoint
semantics and queue/monitoring requirements. The
[HandNet conversion and evaluation guide](https://github.com/twerdster/tea/blob/27ae728/docs/handnet.md)
defines class sampling, provenance and held-out metrics. The historical
10,000-frame modernization run reported majority-class self-success before the
trainer repairs; it is not evidence that the new forest is accurate.

## Inference integration

The pinned [native Tea backend guide](https://github.com/twerdster/depthtrack/blob/6bbbe74711b4f3dc77699291c84876fff405db52/docs/tea-forest.md)
provides the C++17 loader/CPU CLI/API for complete 20-decision, three-tree forests,
calibrated fingertip observations and timestamp-aware temporal tracking. It
checks the native model layout and exact coordinate banks, normalizes each leaf
posterior, and combines tree probabilities. Synthetic tests verify numerical
and deep-path correctness; recognition quality needs the staged evaluations above.

The pinned [parity and tracking harness](https://github.com/twerdster/depthtrack/blob/6bbbe74711b4f3dc77699291c84876fff405db52/tools/validate_tea_handnet.py)
can compare selected deployment artifacts against the matching Tea Python
reference on identical train/validation frames. The published harness check used
three real training frames, an **untrained synthetic fixture forest** and
**synthetic 0.1 s timestamps**. It passed implementation agreement checks only;
that original fixture check is separate from trained-model evidence. The
[pilot report](https://github.com/twerdster/tea/blob/handnet-forest-training/docs/handnet-pilots.md)
also records native/Python parity for the selected pilot on three training frames.
Full-model parity, final held-out recognition and temporal accuracy remain
separate from those implementation checks.

The HandNet label modes are explicit:

| Mode | Modeled classes | Foreground/invalid-centre contract |
| --- | --- | --- |
| Seven classes | thumb, index, middle, ring, pinky, palm, wrist (`0..6`) | Needs a foreground policy; invalid/excluded centres are outside modeled classes |
| Optional eight classes | the same seven plus valid-depth raw label 0 mapped to learned background 7 | Learned background 7 is distinct from invalid-centre 8 and unknown/no-vote 255 |

A centre mask restricts queried pixels without altering raw-depth neighbour
lookups. Metric XYZ observations also need the selected camera/crop's calibrated
intrinsics and real monotonic timestamps. The preserved HandNet `Parameters.mat`
is unchanged and must not be treated as a universal camera calibration.

Deep native Tea currently executes on Mac/Linux CPU. Metal, mobile and FPGA
support for the historical packed ten-decision model does not imply support for
this complete forest format. The browser demo and its recorded tracking results
remain the historical detector path, not results from new HandNet training.

## Historical tools and terms

The [original MATLAB workflow](../README.md#visualize-a-dataset-split), calibration,
indices, attribution and archive provenance remain unchanged. The separate
trainer/backend does not reproduce the HandNet neural network or BMVC 2015 paper
metric. [Code and dataset terms](../README.md#license) retain their original scope;
linking these repositories does not relicense historical/vendor code or recordings.
