# DepthTrack: depth inference and temporal tracking

[DepthTrack](https://github.com/twerdster/depthtrack) is included as the pinned Git submodule [`external/depthtrack`](../external/depthtrack). It packages recovered depth-tree inference, a new timestamp-aware3D tracker, recorded outputs, a browser replay, and native integration examples.

**[Open the phone-friendly demo](https://twerdster.github.io/depthtrack/)** · [Base repository and all platform commands](https://github.com/twerdster/depthtrack#run-on-each-platform) · [Recorded outputs and limits](https://github.com/twerdster/depthtrack/blob/main/docs/results.md)

![Recorded depth-image tracking on Mac CPU](https://raw.githubusercontent.com/twerdster/depthtrack/b3b17e6219a93f0d915cb67565e710ee8278eda3/docs/media/improved_tracking_phone.gif)

The130-frame,17.6s depth recording is **HandMeshRecord**. Its fingertip observations come from the recovered ZScanV02 tree detector. The newer filter consumes the exact same observations using real timestamps and explicit missing/held/lost states. No magnetic sensor positions were supplied to detection or tracking. The [before/after comparison](https://github.com/twerdster/depthtrack/blob/b3b17e6219a93f0d915cb67565e710ee8278eda3/docs/media/comparison_frame_066.png) uses equal trail duration, projection and scale.

On373 comparable samples where the original tracker was freshly matched,95th-percentile separation from the current detections drops28.94→11.86mm. These are detector-relative measurements, not anatomical ground-truth accuracy. Detector mistakes and ambiguous modes remain possible. See the linked result record for all metrics, one retained held jump, synthetic checks and raw expected states.

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

## Platforms and scope

| Platform | Implementation | Evidence boundary |
| --- | --- | --- |
| Mac | Native C++/Python tracking, C++ tree inference, Metal GPU, browser | Actual native, numerical and Metal execution |
| Linux | Same C++/Python tracking and CPU inference, RTL simulation | Actual native/full-state checks and CPU fixture validation |
| iOS | Browser viewer and C API/Swift integration | Device/simulator static cross-build and Swift type-check; no executed phone app |
| Android | Browser viewer and NDK/JNI/Kotlin integration | JNIarm64 cross-build passed in CI; Kotlin/app/device runtime is untested |
| FPGA | Mojo V3 seven-class ten-split depth-tree inference; temporal tracking on host | Preserved actual50MHz board evidence and repeated packaged RTL simulations |

The [base README](https://github.com/twerdster/depthtrack#run-on-each-platform) contains exact commands and links to the C API/mobile examples, Python replay, GPU/backend checks, explicit FPGA programming and ISE rebuild instructions. Device code must supply metric XYZ observations and monotonic timestamps; the examples are not complete camera apps. Full-depth historical detector models remain external, while the small10-split model and numerical fixtures are included.

## Relationship to HandNet

The recovered ZScanV02 classifier's training dataset and its relationship to **HandNet are unverified**. No training occurred during this work, and training/test independence of the recording is unknown. This integration makes related reusable code and evidence accessible; it does not reproduce the HandNet neural network, paper metric or held-out evaluation.

HandNet's historical MATLAB scripts, calibration/indices files, citation, academic-use code notice and dataset licensing remain unchanged. DepthTrack has its own [component notices](https://github.com/twerdster/depthtrack/blob/main/NOTICE.md); HandNet's dataset license is not assigned to unrelated recovered recordings or historical/vendor code.
