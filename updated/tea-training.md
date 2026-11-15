# Staged HandNet forest training and evaluation

HandNet now links a complete native Tea forest CPU/API path through its pinned
[DepthTrack submodule](depthtrack.md). Training new HandNet models and measuring
their quality are separate stages from implementing that inference backend.

## Current state

The repaired trainer and conversion/reference pipeline are available in
[Tea's `handnet-forest-training` branch](https://github.com/twerdster/tea/tree/handnet-forest-training),
starting with [reviewed commit `27ae728`](https://github.com/twerdster/tea/commit/27ae728).
The bounded campaign is active but waiting for GPU resources. Its initial scope
is at most eight pilot configurations and three final trees with up to 20 split
decisions, trained sequentially under the project queue, budget and recurring
monitor. New trained model artifacts, final selections and held-out metrics are
not ready. Do not substitute synthetic backend agreement or training self-success
for those results.

[Tea tooling CI](https://github.com/twerdster/tea/actions/runs/43717599628) passed
for `27ae728`; [DepthTrack CI](https://github.com/twerdster/depthtrack/actions/runs/43717830988)
passed for its pinned `de06719` backend. These completed source checks do not
mean that the GPU training campaign has run or that models are accurate.

The training workflow preserves source archives and existing GPU jobs. Resource
waiting remains an explicit monitored phase. Failures retain evidence and need a
diagnosis before a replacement run; the monitor does not expand the budget.

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

The pinned [native Tea backend guide](https://github.com/twerdster/depthtrack/blob/de06719f1be7e44b6dffb3ea612edfba01050e0f/docs/tea-forest.md)
provides the C++17 loader/CPU CLI/API for complete 20-decision, three-tree forests,
calibrated fingertip observations and timestamp-aware temporal tracking. It
checks the native model layout and exact coordinate banks, normalizes each leaf
posterior, and combines tree probabilities. Synthetic tests verify numerical
and deep-path correctness; recognition quality needs the staged evaluations above.

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
