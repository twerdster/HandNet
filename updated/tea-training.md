# HandNet forest training and evaluation

HandNet links the complete native Tea forest CPU/API through its pinned
[DepthTrack submodule](depthtrack.md). The repaired training campaign and its
original final evaluation are complete as of **2026-11-16**.

## Audited final results

Three independently sampled, eight-class trees each completed **20 split
decisions** using **1,200 features** and **181,979 training frames**. Seeds
201/202/203 produced 84,759,330 / 84,758,588 / 84,759,516 sampled pixels. The
training implementation remained frozen at
[Tea `27ae728e9cb012d2255337d9b3583bbe7f66b3cc`](https://github.com/twerdster/tea/commit/27ae728e9cb012d2255337d9b3583bbe7f66b3cc).
Each tree retains its exact coordinate bank and complete checkpoints. The
ensemble averages the normalized per-tree leaf posteriors with equal weights.

| Frozen model | Official test: 10,000 frames | Full validation: 2,773 frames |
| --- | ---: | ---: |
| Three-tree ensemble, seeds 201/202/203 | **0.37494755678731473** | 0.3833437004586795 |
| Best single, seed 202 | **0.32798765233655836** | 0.3370599042841775 |

Values are **mean fingertip F1@15mm**, averaged over thumb/index/middle/ring/pinky,
not pixel-classification accuracy. For each finger the operational score is
`2 × hits_within_15mm / (sensor_present + sensor_detected + false_positive_on_absent)`;
present fingers with missing detections remain failures. Positions use valid
HandNet sensor references and the documented calibrated largest-component
localizer. The full-validation column includes the selection subset and is
**descriptive**, not another untouched confirmation.

The ensemble produced false-positive predictions on **3,231 of 3,232 absent-finger
occasions**; the single predicted a fingertip on all 3,232. Absent-finger handling
remains a substantial limitation despite the ensemble's higher frozen-metric F1.
These modest results improve on the best single under the frozen metric. They
do not reproduce the BMVC 2015 neural network/mean-shift paper metric, establish
subject-independent generalization, calibrated posterior probabilities, temporal
tracking accuracy or inference latency. Official randomized frame splits alone
do not establish participant/session independence. Any additional 200m/256
experiment after disclosure of these test outcomes would be exploratory;
preparation of that optional experiment is not an additional completed result.

The [Tea final-results report](https://github.com/twerdster/tea/blob/main/docs/handnet-forest-results.md)
and its [safe aggregate evidence](https://github.com/twerdster/tea/blob/main/docs/evidence/handnet-forest-2026-11-16.json)
contain full counts, foreground/classification metrics and provenance. Selected
artifacts are indexed by release tag
[`handnet-forest-2026-11-16`](https://github.com/twerdster/tea/releases/tag/handnet-forest-2026-11-16).
Use the release checksums and each package's manifest; do not substitute coordinate
banks or rely on unrelated historical packed-model assets. Attribution remains
attached; no blanket model license is inferred from the HandNet data license.

## Selection and protocol

Eight pilots and 48 settings on the same frozen **300 validation frames** selected
learned background and probe scale **25000 pixel·mm**. The
[earlier pilot report](https://github.com/twerdster/tea/blob/main/docs/handnet-pilots.md)
preserves their scores and foreground comparisons. After the three full trees
finished, **24 validation configurations** compared the ensemble and each single
at 12/16/20 decisions and probability thresholds 0.25/0.5. Test inputs did not
select class mode, scale, tree, depth or cutoff.

Both selected deployments freeze **20 decisions**, **0.25 probability cutoff**,
**3-pixel minimum four-connected component support**, foreground **`none`**, and
fingertip class IDs **0..4**. The best single is seed **202**. No ground-truth
foreground mask enters these selected predictions; learned background is a
modeled class. The largest surviving component uses a probability-weighted
centroid and calibrated depth-to-XYZ projection. Both original 10,000-frame
test evaluations and their metric/count arithmetic passed independent audit,
without test-driven retuning. The later optional experiment cannot be called an
independent untouched confirmation after these test results are known.

Training remained sequential within the original monitored eight-pilot/three-final
budget. All three finals exited successfully, retained 20 checkpoints and passed
finite-model/header/copied-asset and saved-prediction checks. Source CI and
numerical parity are implementation evidence; the audited held-out scores above
are separate evidence of model quality. The historical modernization run's
majority-class self-success is not evidence for the new forest's accuracy.

## Reproducible workflow

1. Freeze official training/validation/test membership and source frame identities.
2. Prepare class-aware samples with separate frame, pixel and coordinate seeds.
   Retain each tree's exact coordinate bank and feature profile.
3. Train independently sampled trees with atomic depth checkpoints. Tea CLI/header
   depth 19 means 20 split decisions; the output tree index is not an RNG seed.
4. Freeze a manifest containing model/coordinate checksums, millimetre depth units,
   float32 feature rules, image geometry, class order and posterior aggregation.
5. Select individual trees/ensemble, depth and postprocessing on validation only.
6. Evaluate untouched test frames using those frozen choices; report misses,
   absent-finger false positives, foreground errors and the evaluation domain.

The [Tea deep-training guide](https://github.com/twerdster/tea/blob/27ae728e9cb012d2255337d9b3583bbe7f66b3cc/docs/deep-training.md)
contains bounded commands, memory/checkpoint semantics and monitoring requirements.
The [HandNet conversion/evaluation guide](https://github.com/twerdster/tea/blob/27ae728e9cb012d2255337d9b3583bbe7f66b3cc/docs/handnet.md)
defines class sampling, provenance and metrics. Preserve failure evidence and
validate a diagnosis before a replacement run; monitoring never expands budget
or permits interference with unrelated GPU work.

## Inference integration and numerical checks

The pinned [native Tea backend guide](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/tea-forest.md)
provides C++17 loading, CPU CLI/API, calibrated fingertip observations and
timestamp-aware tracking for complete forests. Deep native Tea executes on
**Mac/Linux CPU**. The historical ten-decision Metal/mobile/FPGA paths do not
establish support for this complete forest format.

The [parity harness](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/tools/validate_tea_handnet.py)
and [selected-model evidence](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/tea-handnet-selected-model-validation.json)
record the actual selected ensemble and seed-202 single on **three training
frames**, both `none`/`near-component` foreground policies, cutoff 0.25 and support
3. Labels, probability bytes, reached heaps and ROI masks match exactly between
Python/native inference on Linux/Mac. Observation/tracker difference is at most
**3.411e-13 mm**, and Linux/Mac outputs match exactly. The timestamps are
**synthetic 0.1 s cadence**: this verifies numerical implementation agreement,
not held-out recognition, temporal tracking accuracy or latency. The earlier
untrained-fixture harness record remains separate.

| Mode | Modeled classes | Foreground/invalid-centre contract |
| --- | --- | --- |
| Seven classes | thumb, index, middle, ring, pinky, palm, wrist (`0..6`) | Needs a foreground policy; excluded centres lie outside modeled classes |
| Selected eight classes | those seven plus valid-depth raw label 0 mapped to learned background 7 | Learned background 7 is distinct from invalid-centre 8 and unknown/no-vote 255 |

A centre mask restricts queried pixels without changing raw-depth neighbour
lookups. Metric XYZ observations require the camera/crop's calibrated intrinsics;
real temporal deployment requires monotonic measured timestamps. The unchanged
HandNet `Parameters.mat` is not universal camera calibration. The browser viewer
and recorded HandMeshRecord tracking animation still use their original data/
detector paths, not these new HandNet model outputs.

## Historical tools and terms

The [original MATLAB workflow](../README.md#visualize-a-dataset-split), calibration,
indices, attribution and archive provenance remain unchanged. No MATLAB/Octave
runtime or original preprocessing/paper reproduction follows from this separate
trainer/backend. [Code and dataset terms](../README.md#license) retain their
scope: HandNet dataset CC BY 4.0 attribution is separate from the historical
MATLAB academic-use notice and unrelated recovered/vendor code or recordings.
