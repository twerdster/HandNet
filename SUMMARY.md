# HandNet context

## Purpose

MATLAB visualization, rotation, HDF5 preprocessing and saved-result inspection for HandNet depth-camera data from the BMVC 2015 work.

## Implementation evidence

DisplaySession loads Parameters.mat and Indices.mat and defaults to ValidationData/session 0. ShowResults defaults to a 15 mm 3D threshold; the source distinguishes the paper’s 2D pixel metric.

Key entry points: `DisplaySession.m`, `ShowResults.m`, `CreateH5Data.m`, `rotateData.m`, `derotate.m`, `compute_maP.m`.

## Documentation state

The README explains the dataset tools, setup, metrics and limits. The image is the project figure preserved from the former Technion page, not a new runtime result. Source files and attribution are unchanged.

## Validation scope

Source inspection only; no current MATLAB/Octave run, training, dataset download or benchmark reproduction. No automated regression suite is included.

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
