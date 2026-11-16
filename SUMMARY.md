# HandNet context

## Purpose

MATLAB visualization, rotation, HDF5 preprocessing and saved-result inspection for HandNet depth-camera data from the BMVC 2015 work.

## Implementation evidence

DisplaySession loads Parameters.mat and Indices.mat and defaults to ValidationData/session 0. ShowResults defaults to a 15 mm 3D threshold; the source distinguishes the paper’s 2D pixel metric.

Key entry points: `DisplaySession.m`, `ShowResults.m`, `CreateH5Data.m`, `rotateData.m`, `derotate.m`, `compute_maP.m`.

## Documentation state

The README explains the dataset tools, setup, metrics and limits. The image is the project figure preserved from the former Technion page, not a new runtime result. Source files and attribution are unchanged.

## Validation scope

Historical MATLAB: source inspection only; no current MATLAB/Octave run, training, dataset download or benchmark reproduction. No automated regression suite for those scripts is included. The separate DepthTrack submodule has its own executed tests and recorded validation, described below.

## Limitations

Older toolbox APIs, hard-coded preprocessing paths, separate result files, and derotate.m declaring derotate2. Dataset CC BY 4.0 does not change the original code academic-use notice.

## Modern viewer navigation — 2026-11-14

The root README adds a local New! badge and the exact
“Updated and ported for modern hardware” link to `updated/README.md`.
That GitHub guide contains the modern-viewer summary, usage, sample scope,
provenance and license links. It links the existing website explorer source and
assets directly, without depending on a separate update guide on the website.
The website viewer has 20 precomputed rows/eight pairs; the separate research
Space has 96 samples. The historical MATLAB implementation, reference README
body, original code terms and validation limits are unchanged. No MATLAB
runtime, preprocessing, training or learned-inference claim follows from the
browser visualization. The requested update navigation is scoped to GitHub
repositories; website navigation changes require a separate user request.

## DepthTrack submodule and tracking outputs —2026-11-15

Added`external/depthtrack`as a pinned public Git submodule from`twerdster/depthtrack`, with`.gitmodules`and clone/update instructions. Root README and updated/README.md link`updated/depthtrack.md`, which embeds the recorded output and links the base repo's exact Mac/Linux/iOS/Android/FPGA commands, result evidence and component notices.

DepthTrack contains portable C++17/C ABI, NumPy and JavaScript tracking, CPU/Metal/Mojo tree inference, a static mobile-friendly replay,650-state fixtures, lossless sample depth and archived output evidence. Mac/Linux native tracking and CPU checks execute; Metal executes onMac; iOS device/simulator libraries cross-compile; Android JNIarm64 cross-build passed in CI, with device/app runtime explicitly unverified. The FPGA performs the seven-class ten-split per-pixel classifier, with temporal tracking on the host; it does not contain the full hand detector/tracker.

The demo uses130 HandMeshRecord depth frames and recovered ZScanV02 model detections. Training dataset/HandNet relationship and train/test independence are unverified. No new HandNet network training or paper accuracy is claimed. Detector-relative tracking residuals improve, with original and improved states preserved on identical inputs. Historical MATLAB/calibration/indices bytes, original attribution, data/code terms and existing website remain unchanged.

## Native Tea forest integration — 2026-11-15 (historical stage)

The isolated `tea-handnet-forest` worktree starts from the clean historical
`master` commit `0a72574`; the original checkout remains unchanged.
`external/depthtrack` remains a true mode 160000 Git submodule, deliberately
initially advanced from `c9913d3f03089bb2f8568e2880c2f3d4d7bf40f9` to reviewed public
commit `de06719f1be7e44b6dffb3ea612edfba01050e0f`; the current pin is recorded below. `.gitmodules` tracks its public
`tea-handnet-forest` branch. Ordinary recursive submodule checkout uses the exact
recorded gitlink, not a copied implementation or build tree.

The root/update guides now link the complete native Tea CPU/API backend and
`updated/tea-training.md`. The backend supports three-tree forests and complete
20-decision trees, with separate exact coordinate banks, normalized tree
posteriors, calibrated fingertip observations and timestamp-aware tracking.
Seven hand-only classes and optional eight-class learned-background models have
explicit separate contracts. Header/Tea CLI 19 means 20 decisions. The deep Tea
backend currently executes on Mac/Linux CPU; packed ten-decision Metal/mobile/
FPGA examples retain their prior scope.

Training source is Tea branch `handnet-forest-training`, reviewed commit
`27ae728`, with repaired native training, class-aware conversion, manifests and
held-out evaluation. As of 2026-11-15 16:35 UTC, all eight pilots and 48 validation
configurations are complete on frozen 300 validation frames. Learned background
and probe 25000 were selected for full-tree training preparation; conversion for
tree 0 began at 16:27:33 UTC. Three final trees remain under the existing sequential
queue, authorized budget and recurring monitor. No complete full-tree model or
final ensemble/test metrics are ready; final depth/threshold selection is pending.
The historical HandMeshRecord animation still uses ZScanV02 detections; it is not
a new forest result. Official frame splits alone do not prove subject independence.

Integration checks: the exact upstream commit is reachable on its public branch;
`git submodule update --init --recursive` resolves the staged new gitlink. A fresh
native build from this submodule runs the small three-tree CPU CLI fixture with
labels and reached heaps matching the independent portable fixtures exactly;
probabilities are checked against the fixture tolerance. The six Mac/Linux native
CTests were already executed upstream and were not repeated for this docs/pin
integration. All 10 tracked MATLAB/MAT source, data/calibration and index files
match the original checkout byte-for-byte. No MATLAB runtime, source archive,
calibration, website navigation or historical output asset was changed.

Initial source CI was independently verified publicly: DepthTrack run
[43717830988](https://github.com/twerdster/depthtrack/actions/runs/43717830988)
reports Success at `de06719`; Tea tooling run
[43717599628](https://github.com/twerdster/tea/actions/runs/43717599628)
reports Success at `27ae728`. CI implementation/tooling checks are separate from
the separately monitored GPU campaign and held-out scientific evaluation.

## HandNet parity harness pin — 2026-11-15 (historical stage)

The current `external/depthtrack` gitlink is reviewed public commit
`6bbbe74711b4f3dc77699291c84876fff405db52` on `tea-handnet-forest`, advanced from
`de06719`. Native inference behavior is unchanged; upstream adds
`tools/validate_tea_handnet.py`, evidence/documentation and sanitized machine paths.
Pinned guide/tool links now use this commit. The completed current CI run
[43719231425](https://github.com/twerdster/depthtrack/actions/runs/43719231425)
reports Success for Mac/Linux/Android checks; Pages was skipped, so no new website
deployment is claimed. Earlier CI links remain historical evidence.

The recorded harness check processed three real HandNet training frames with an
explicitly untrained synthetic fixture forest and synthetic 0.1 s timestamps.
Native/Python queried masks, labels, heaps and probabilities agreed, with calibrated
observation/tracker/history agreement within the recorded tolerances. This is
harness validation only: it is not an actual trained-model parity result, held-out
recognition score or measured temporal accuracy. It remains separate from the
pilot validation and the then-pending full-tree campaign. The completed original
forest evaluation is recorded below.

Integration validation rechecks remote reachability, normal recursive checkout,
relative guide links and all 10 original MATLAB/MAT file hashes. Native builds/tests
are not repeated because this pin changes tools/docs/evidence only; the previous
fresh CLI fixture check and upstream native validation retain their stated scope.

## Completed original forest evaluation — 2026-11-16

All three full trees (seeds 201/202/203) completed 20 split decisions with 1200
features and 181,979 training frames per tree. Sample counts are 84,759,330 /
84,758,588 / 84,759,516. Frozen training source remains Tea
`27ae728e9cb012d2255337d9b3583bbe7f66b3cc`; documentation publication does not change
that implementation identity. All 20 checkpoints/model headers/finite posteriors,
retained copies and independent saved-prediction checks passed. Historical pilot
scores and the earlier stage records above remain dated evidence, not current
status.

The frozen 300-validation-frame, 24-case final grid selected both the ensemble and
best single (seed 202) at 20 decisions, probability cutoff 0.25, four-connected
minimum component support 3, foreground `none` and fingertip classes 0..4. All 2,773
validation frames report mean fingertip F1@15mm 0.3833437004586795 ensemble /
0.3370599042841775 single; this descriptive cohort includes the selection subset.
Both original 10,000-frame test evaluations independently audited PASS:
**0.37494755678731473 ensemble / 0.32798765233655836 single**. No test-driven
retuning was performed. Official random-frame splits do not establish subject/
session independence; these modest largest-component/sensor-reference results
are not the BMVC 2015 paper metric, calibrated probabilities or a temporal benchmark.
Any later optional 200m/256 comparison after disclosure is exploratory.

Actual selected ensemble/single native/Python parity passed on Linux/Mac using
three training frames, both foreground policies and synthetic 0.1 s cadence: exact
labels/probability bytes/heaps/ROI; observation/tracker differences <= 3.411e-13 mm,
Linux/Mac difference 0. This is numerical agreement, not measured temporal accuracy
or latency. Historical untrained-fixture parity and the HandMeshRecord animation
retain their separate scope.

The guides link the [Tea final report](https://github.com/twerdster/tea/blob/main/docs/handnet-forest-results.md),
[safe aggregate](https://github.com/twerdster/tea/blob/main/docs/evidence/handnet-forest-2026-11-16.json)
and selected-model release tag `handnet-forest-2026-11-16`. Dataset CC BY 4.0 attribution
and original historical MATLAB terms remain separate; no blanket model license
is declared. Source/docs links and the true submodule gitlink must be synchronized
to the reviewed DepthTrack documentation commit before publication. No website
navigation, MATLAB/MAT/calibration/indices bytes, original worktree or training
process is changed by this integration.
