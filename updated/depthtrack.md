# DepthTrack: depth inference and temporal tracking

[DepthTrack](https://github.com/twerdster/depthtrack) is included as the pinned Git submodule [`external/depthtrack`](../external/depthtrack). It packages recovered depth-tree inference, native Tea forest CPU inference, a timestamp-aware 3D tracker, recorded outputs, a browser replay and native integration examples.

**[Open recorded tracking](https://twerdster.github.io/depthtrack/)** · **[Try live monocular camera depth](https://twerdster.github.io/depthtrack/live/)** · [Base repository and all platform commands](https://github.com/twerdster/depthtrack#run-on-each-platform) · [Recorded outputs and limits](https://github.com/twerdster/depthtrack/blob/main/docs/results.md)

![Recorded depth-image tracking on Mac CPU](https://raw.githubusercontent.com/twerdster/depthtrack/b3b17e6219a93f0d915cb67565e710ee8278eda3/docs/media/improved_tracking_phone.gif)

The130-frame,17.6s depth recording is **HandMeshRecord**. Its fingertip observations come from the recovered ZScanV02 tree detector. The newer filter consumes the exact same observations using real timestamps and explicit missing/held/lost states. No magnetic sensor positions were supplied to detection or tracking. The [before/after comparison](https://github.com/twerdster/depthtrack/blob/b3b17e6219a93f0d915cb67565e710ee8278eda3/docs/media/comparison_frame_066.png) uses equal trail duration, projection and scale.

On373 comparable samples where the original tracker was freshly matched,95th-percentile separation from the current detections drops28.94→11.86mm. These are detector-relative measurements, not anatomical ground-truth accuracy. Detector mistakes and ambiguous modes remain possible. See the linked result record for all metrics, one retained held jump, synthetic checks and raw expected states.

## Complete native Tea forests

The submodule is deliberately pinned to
[`1b3f86d`](https://github.com/twerdster/depthtrack/commit/1b3f86dd315d23a495b387c607ba3be4cb51662e).
Its [native Tea backend guide](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/tea-forest.md)
and [C++ API](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/include/depthtrack/tea_forest.hpp)
cover complete 20-decision trees and three-tree forests with their exact
`Coords.co` banks. Each leaf posterior is normalized before the weighted tree
mean. The optional calibrated observation stage and existing tracker consume
these predictions with explicit missing measurements and real timestamps.

Seven-class models contain only hand classes and require an explicit foreground
policy. An optional eight-class model learns valid-depth background as class 7;
invalid centres remain outside modeled classes (for example label 8), and
unknown/no-vote 255 remains distinct. Caller masks restrict queried centres only;
feature neighbours read unmodified raw depth. Calibration and class mapping are
explicit manifest data. See the guide for the independent depth-only ROI, API,
CLI outputs and label/coordinate/aggregation rules.

Mac and Linux CPU execution passed all six native tests, including independent
portable-fixture comparisons and all 20 decisions of a three-tree forest. These
are synthetic correctness checks; they establish no HandNet accuracy result.
The reviewed change passed [hosted Mac/Linux/Android CI](https://github.com/twerdster/depthtrack/actions/runs/43786447391)
before its merge to `main`. The exact pinned merge also passed the current
[default-branch CI/Pages run](https://github.com/twerdster/depthtrack/actions/runs/43786623292).
Both hosted HTML routes and 17 deployed assets match the reviewed source bytes;
this does not establish physical-phone execution or anatomical accuracy. The
combined source retains the separate live-camera route described below; neither
route changes the other's model contract.
The new [native/Python HandNet parity harness](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/tools/validate_tea_handnet.py)
compares identical depth frames through inference, calibrated observations and
complete temporal tracker state/history. Its
[recorded harness check](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/tea-handnet-harness-validation.json)
passed on three real training frames using an **untrained synthetic fixture
forest** and **synthetic 0.1 s timestamps**. This verifies integration agreement;
it is not a trained-model result, recognition score or measured temporal accuracy.
The selected trained ensemble and seed-202 single now also passed their own
[actual-model native/Python check](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/tea-handnet-selected-model-validation.json)
on Linux and Mac: the same three **training** frames, both foreground policies,
cutoff 0.25/minimum support 3. Labels, posterior bytes, reached heaps and ROI
masks match exactly. Calibrated observation/tracker differences are at most
3.411e-13 mm; Linux/Mac outputs match exactly. The synthetic 0.1 s cadence makes
this numerical parity evidence, not a temporal benchmark or latency measurement.

The deep Tea backend has not been ported to Metal, mobile inference or FPGA.
The older packed seven-class ten-decision hardware examples remain separate.

[HandNet forest training and held-out evaluation](tea-training.md) now records
three completed full trees and independently audited 10,000-frame test results:
mean fingertip F1@15mm **0.37495 ensemble / 0.32799 best single (seed 202)**.
Full 2,773-frame validation is 0.38334 / 0.33706 and includes the selection
subset. Both deployments were fixed on 300 validation frames at 20 decisions,
cutoff 0.25, minimum support 3 pixels, foreground `none` and fingertip IDs 0..4.
The [Tea results report](https://github.com/twerdster/tea/blob/main/docs/handnet-forest-results.md)
and [release](https://github.com/twerdster/tea/releases/tag/handnet-forest-2026-11-16)
provide the metric, provenance, assets and limits. Official random-frame splits
do not prove subject independence; these modest results are not the paper metric.
The HandMeshRecord animation above still uses the historical ZScanV02 detector,
not the newly trained HandNet ensemble.

## Get the code and run

```sh
git clone --recurse-submodules https://github.com/twerdster/HandNet.git
cd HandNet/external/depthtrack
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 4
ctest --test-dir build/native --output-on-failure
build/native/depthtrack_csv samples/handmesh/tracking_fixture.csv build/native/tracks.csv
```

For an existing HandNet checkout:

```sh
git pull
git submodule update --init --recursive
```

For the browser replay, from the DepthTrack submodule:

```sh
python3 -m http.server 8000 --directory web
```

Open `http://localhost:8000`, or use the hosted HTTPS demo on a phone. Play/pause, scrub real timestamps and compare original/improved states. JavaScript computes the causal tracker from saved detections; it does not run a HandNet neural network or capture a camera. No account, Tailscale or full archive download is required for the bundled sample.

## Live webcam / phone-camera experiment

The pinned submodule also includes a separate [live monocular-depth page](https://twerdster.github.io/depthtrack/live/): camera RGB → Depth Anything V2 Small at294² → approximate two-anchor depth scale → range/centre masks → the exact128² ten-split seven-class tree → candidate-region tracking. Processing happens locally in the browser; frames are not uploaded. Load the model, start the camera, then capture two measured near/far anchors before enabling the tree. The recorded replay above remains independent of the neural runtime.

The model gives relative depth, so a distance cutoff is approximate after calibration. Low-resolution monocular depth can lose finger geometry; the old tree's anatomical accuracy on this input is unverified. The compact tree has no nonzero-depth background class, and largest-class components are experimental observations, not the restored full detector's fingertip modes. This is browser execution, not FPGA execution or HandNet training.

Actual Mac browser model inference and synthetic-camera start/stop/restart tests passed;75 numerical/lifecycle/app browser tests pass locally. No physical phone runtime or useful live webcam hand scene is claimed. [Setup, measurements and limits](https://github.com/twerdster/depthtrack/blob/1b3f86dd315d23a495b387c607ba3be4cb51662e/docs/monocular-live.md).

## Platforms and scope

The table describes the preserved tracking/packed-model paths. Complete deep
Tea forests currently execute on Mac/Linux CPU only; the new guide above records
that separate backend's scope.


| Platform | Implementation | Evidence boundary |
| --- | --- | --- |
| Mac | Native C++/Python tracking, C++ tree inference, Metal GPU, browser | Actual native, numerical and Metal execution |
| Linux | Same C++/Python tracking and CPU inference, RTL simulation | Actual native/full-state checks and CPU fixture validation |
| iOS | Browser viewer and C API/Swift integration | Device/simulator static cross-build and Swift type-check; no executed phone app |
| Android | Browser viewer and NDK/JNI/Kotlin integration | JNIarm64 cross-build passed in CI; Kotlin/app/device runtime is untested |
| FPGA | Mojo V3 seven-class ten-split depth-tree inference; temporal tracking on host | Preserved actual50MHz board evidence and repeated packaged RTL simulations |

The [base README](https://github.com/twerdster/depthtrack#run-on-each-platform) contains exact commands and links to the C API/mobile examples, Python replay, GPU/backend checks, explicit FPGA programming and ISE rebuild instructions. Device code must supply metric XYZ observations and monotonic timestamps; the examples are not complete camera apps. Full-depth historical detector models remain external, while the small10-split model and numerical fixtures are included.

## Relationship to HandNet

The recovered ZScanV02 classifier's training dataset and its relationship to **HandNet are unverified**. The recorded demo did not retrain its historical detector, and training/test independence of the recording is unknown. The separate [bounded HandNet campaign](tea-training.md) has completed three-tree training and audited held-out evaluation. That operational largest-component fingertip evaluation is distinct from the original paper metric, the historical recording and synthetic tracking parity. This integration makes related code and evidence accessible; it does not reproduce the HandNet neural network.

HandNet's historical MATLAB scripts, calibration/indices files, citation, academic-use code notice and dataset licensing remain unchanged. DepthTrack has its own [component notices](https://github.com/twerdster/depthtrack/blob/main/NOTICE.md); HandNet's dataset license is not assigned to unrelated recovered recordings or historical/vendor code.
