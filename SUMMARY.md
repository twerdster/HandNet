# HandNet context

## Purpose

MATLAB visualization, rotation, HDF5 preprocessing and saved-result inspection for HandNet depth-camera data from the BMVC 2015 work.

## Current integration — 2026-11-16

The exact DepthTrack pin is `356060770b65e5c60ef5db702482da8a30f7dbb0`,
combining native Tea forest results/inference with the upstream live monocular
camera experiment. Original three-tree training and both final test audits are
complete; the dated historical stage records below are preserved for provenance.
See the current combined-pin section for checkout/validation scope.

The completed optional seed204 comparison uses 200,216,884 samples, 256 saved
features and the same 181,979 ordered frames as original seed201/tree0. Independent
raw-reference audits passed all four frozen full-cohort evaluations at 20
decisions/cutoff 0.25/support 3/foreground none. Test F1@15mm is 0.3302297543 versus
0.3402043018 for matched seed201; full validation is 0.3372283285 / 0.3477379235
and includes the 300 selection frames. The jointly changed recipe is
post-exposure exploratory and did not improve the matched test baseline; original
forest defaults and validation selection remain unchanged.

The guides link Tea's published comparison report at docs commit
`5c991f6e439402e95d4b03673aa77383a1dd447e` and the optional
`handnet-200m-256-seed204-2026-11-16` prerelease. Report/site availability and the
anonymous ZIP download were verified; the archive matches the documented
53b8fc95 checksum. The generic CPU CLI loads this model explicitly. Original
browser routes remain unchanged; the current reviewed DepthTrack pin is recorded
below. No new native build or MATLAB runtime is claimed here.

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

At this historical stage, the `external/depthtrack` gitlink was reviewed public commit
`6bbbe74711b4f3dc77699291c84876fff405db52` on `tea-handnet-forest`, advanced from
`de06719`. Native inference behavior is unchanged; upstream adds
`tools/validate_tea_handnet.py`, evidence/documentation and sanitized machine paths.
Guide/tool links used that commit. The completed stage CI run
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
is declared. Source/docs links and the true submodule gitlink are synchronized to the
reviewed combined DepthTrack commit recorded below. No website
navigation, MATLAB/MAT/calibration/indices bytes, original worktree or training
process is changed by this integration.

## Live monocular DepthTrack adapter —2026-11-15 (historical stage)

At this stage the DepthTrack gitlink advanced to de334f7e2a89bcb607c8142cb00756136b46b91b; the combined current pin is recorded below. Its separate public `/depthtrack/live/` page adds local browser camera/photo input, low-resolution Depth Anything V2 Small, approximate two-anchor inverse-depth calibration, range/centre masks, exact compact-tree inference and candidate component tracking.75 browser tests pass locally, plus actual Mac neural inference and synthetic-source browser camera lifecycle checks. Physical phone execution, calibrated anatomical accuracy and a useful live Mac hand scene remain unverified; the FaceTime probe was nearly black. The recorded replay, historical MATLAB/calibration/indices, dataset/code terms and original website navigation remain preserved. No training or FPGA changes occurred. Root README and updated/depthtrack.md link the new route and explicit scope; detailed commands/evidence stay in the submodule.

DepthTrack sourcecommit de334f7 passed clean-checkout Mac/Linux, AndroidJNI, iOS static cross-builds and Pages in https://github.com/twerdster/depthtrack/actions/runs/43735739340 . The deployed `/depthtrack/live/` returned HTTP200 and hosted code/model assets match the pinned source bytes. Submodule-local75 browser tests pass; original MATLAB/`.mat` files have no changes.

## Combined DepthTrack pin — 2026-11-16 (historical stage)

`external/depthtrack` is a true mode 160000 submodule at published `main` commit
`1b3f86dd315d23a495b387c607ba3be4cb51662e`, merged through
[DepthTrack PR1](https://github.com/twerdster/depthtrack/pull/1). This combines the
reviewed native Tea forest/docs work with upstream live monocular commit
`de334f7`, preserving camera/model/runtime assets and the recorded replay.
The selected 8-class, 20-decision HandNet forest remains a Mac/Linux CPU path;
the browser live experiment retains its recovered 7-class, 10-split model,
approximate monocular calibration and unverified anatomical/physical-phone scope.
No full native forest inference or HandNet-trained neural network is implied in
the browser experiment.

Backend/API/harness, selected trained-model evidence and live-camera guide links
use this exact pin. `.gitmodules` now names `main` for deliberate future updates;
ordinary `git submodule update --init --recursive` uses the exact recorded gitlink.
The combined change passed hosted Mac/Linux/Android checks in
[run 43786443312](https://github.com/twerdster/depthtrack/actions/runs/43786443312)
and [43786447391](https://github.com/twerdster/depthtrack/actions/runs/43786447391)
before merge. The current default-branch
[CI/Pages run 43786623292](https://github.com/twerdster/depthtrack/actions/runs/43786623292)
at exact pin `1b3f86d` also completed successfully. Both hosted HTML routes and
17 deployed assets were verified byte-identical to the reviewed source. This is
deployment/integration evidence, not measured physical-phone or anatomical accuracy.
Prior local native 6 CTests, 75 browser checks and selected real-model parity
retain their recorded scope; no new native runtime/phone/accuracy claim follows
from this parent documentation and pin update.

Final parent checks verify public `main` reachability, exact recursive checkout,
relative links and all 10 historical MATLAB/MAT files against the original
checkout. The original checkout and historical source/calibration/index bytes
remain unchanged. No training, source archive, website navigation or unrelated
workload is modified by the HandNet integration.

## Current seed204 DepthTrack integration — 2026-11-16

`external/depthtrack` deliberately advances to published `main` merge
`356060770b65e5c60ef5db702482da8a30f7dbb0` through
[DepthTrack PR2](https://github.com/twerdster/depthtrack/pull/2), from the historical
combined pin above. Its reviewed head
`fffa2156d5d3351095c409468451962ecec77375` passed hosted Mac/Linux/Android checks in
[run 43832512300](https://github.com/twerdster/depthtrack/actions/runs/43832512300)
and [run 43832414751](https://github.com/twerdster/depthtrack/actions/runs/43832414751).
Feature-branch Pages was skipped as expected; exact merge/default-branch CI
at `3560607` passed all four jobs in
[run 43832975971](https://github.com/twerdster/depthtrack/actions/runs/43832975971):
Mac desktop, Ubuntu desktop, Android build and Pages. The Mac iOS static
cross-build step and Linux FPGA simulation also passed in their existing scope;
they do not establish phone runtime or seed204 inference on historical hardware.

Only DepthTrack context, its Tea backend guide and the seed204 parity aggregate
changed upstream. Runtime/browser/camera/model/test/tool bytes are unchanged.
Current backend/API/harness/parity links use the exact new pin; historical stage
records retain their original commits. The optional seed204 model remains an
explicit generic CPU choice and the original forest stays the default.

The nested worktree was clean before and after anonymous public fetch/checkout.
Published `main` reachability, true mode 160000 gitlink and ordinary recursive
submodule checkout were verified, together with relative links and all 10
historical MATLAB/MAT bytes. No native rebuild, MATLAB run, training or unrelated
workload change accompanies this docs/evidence integration.
