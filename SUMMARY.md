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

## Live monocular DepthTrack adapter —2026-11-15

The DepthTrack gitlink advances to de334f7e2a89bcb607c8142cb00756136b46b91b. Its separate public `/depthtrack/live/` page adds local browser camera/photo input, low-resolution Depth Anything V2 Small, approximate two-anchor inverse-depth calibration, range/centre masks, exact compact-tree inference and candidate component tracking.75 browser tests pass locally, plus actual Mac neural inference and synthetic-source browser camera lifecycle checks. Physical phone execution, calibrated anatomical accuracy and a useful live Mac hand scene remain unverified; the FaceTime probe was nearly black. The recorded replay, historical MATLAB/calibration/indices, dataset/code terms and original website navigation remain preserved. No training or FPGA changes occurred. Root README and updated/depthtrack.md link the new route and explicit scope; detailed commands/evidence stay in the submodule.

DepthTrack sourcecommit de334f7 passed clean-checkout Mac/Linux, AndroidJNI, iOS static cross-builds and Pages in https://github.com/twerdster/depthtrack/actions/runs/43735739340 . The deployed `/depthtrack/live/` returned HTTP200 and hosted code/model assets match the pinned source bytes. Submodule-local75 browser tests pass; original MATLAB/`.mat` files have no changes.
