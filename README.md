# HandNet

[![New!](updated/assets/new-badge.svg)](updated/README.md#updated-and-ported-for-modern-hardware) **[Updated and ported for modern hardware](updated/README.md#updated-and-ported-for-modern-hardware)**

[**DepthTrack: runnable inference and improved tracking**](updated/depthtrack.md) — a pinned code submodule, complete native Tea forest CPU/API support, recorded tracking outputs, experimental live monocular camera depth and platform guides. [HandNet forest training and evaluation](updated/tea-training.md) records the completed three-tree campaign: audited mean fingertip F1@15mm on 10,000 test frames is **0.37495 for the ensemble** and **0.32799 for the best single tree**. These modest operational scores are separate from the original paper metric.

[![MATLAB](https://img.shields.io/badge/code-MATLAB-0076A8)](DisplaySession.m)
[![Dataset on Hugging Face](https://img.shields.io/badge/dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/twerdster/HandNet)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

<img src="docs/images/project-preview.png" alt="Original HandNet project figure showing fingertip and palm annotations" width="320">

HandNet contains MATLAB scripts for inspecting and preparing the depth-camera hand dataset used
in the BMVC 2015 paper *Rule of Thumb: Deep Derotation for Improved Fingertip Detection* by Aaron
Wetzler, Ron Slossberg and Ron Kimmel. The scripts read per-frame `.mat` files containing depth,
infrared images, segmentation, labels, heatmaps and tracked sensor positions/orientations.
They support visualization, geometric rotation, HDF5 preprocessing and inspection of saved
predictions; this repository does not contain a complete network training pipeline.

*The image above is the original project figure, not a newly generated result.*

## Data and browser tools

- [Project page](https://twerdster.github.io/HandNet/).
- [Dataset release on Hugging Face](https://huggingface.co/datasets/twerdster/HandNet): the current
  download location for seven archives corresponding to the original distribution, including training, test,
  validation, ground-truth-derotated and rotation-perturbed data.
- [Browser explorer](https://twerdster.github.io/HandNet/explorer/): precomputed examples that do
  not require downloading the dataset. This is separate from the MATLAB code in this repository.
- [HandNet Explorer on Hugging Face Spaces](https://huggingface.co/spaces/twerdster/HandNet-Explorer).

## Visualize a dataset split

Use MATLAB and extract an archive inside this checkout, so `HandNet/ValidationData/` is present
for the default example. Keep the extracted directory name, such
as `ValidationData`, consistent with `dirName` in [DisplaySession.m](DisplaySession.m). The script
also loads the included `Parameters.mat` camera calibration and `Indices.mat` ordering metadata.

```matlab
cd('/path/to/HandNet')
DisplaySession
```

`DisplaySession.m` plots depth and a bounding box, pixel labels, summed heatmaps and a 3D point
cloud with sensor poses. Edit `dirName`, `showOrdered`, `chooseSession` and `skipFrames` in the
script to select the split, ordering and session. The default is `ValidationData`, chronological
ordering and session `0`.

For Python data inspection, install SciPy and load an extracted MAT file directly:

```python
from scipy.io import loadmat
frame = loadmat('ValidationData/Data_0000000.mat')
depth = frame['depth']
```

This is a data-loading example, not a Python port of the visualizer. Use an actual filename from
the extracted archive. Octave was not tested. SciPy reads non-HDF5 MAT files; MAT v7.3 input needs an HDF5-aware reader.
The archive format was not rechecked in this documentation review.

## Other scripts

- [rotateData.m](rotateData.m) rotates image arrays around the camera principal point and adjusts
  3D positions and rotations about the camera's z-axis. It calls [rotateAround.m](rotateAround.m).
- [derotate.m](derotate.m) derives a canonical in-plane angle from a sensor rotation matrix.
- [CreateH5Data.m](CreateH5Data.m) crops and normalizes depth, generates multiple image scales and
  writes HDF5 data and label arrays plus file lists. Change its hard-coded input/output paths
  before use. It calls image-processing functions (`maketform`, `imtransform`, `imfilter`,
  `imresize`) and rotation-conversion functions (`dcm2angle`, `dcm2quat`), so the corresponding
  MATLAB toolbox functions are required.
- [ShowResults.m](ShowResults.m) loads `TestData_Perturbed_Results.mat` and plots precision/recall
  using [compute_maP.m](compute_maP.m). Its default comparison uses a 15 mm threshold on 3D
  positions; the source notes that the paper's fingertip score instead uses 2D pixel distance.
  Do not interpret that default plot as a direct reproduction of the paper's metric. The required
  saved predictions are separate from the seven dataset archives; their historical download
  links below could not be retrieved during this review.

The inherited dataset notes describe the derotated variants as using ground-truth orientation
as an oracle, and the perturbed variants as applying random in-plane rotations over −90° to 90°
with the corresponding 3D rotation. The archive generation was not rerun in this review. They
change the orientation distribution used for evaluation; that transformation alone does not
establish generalization for a new model.

## Validation and limitations

Review date: 2026-11-12.

The historical review covered the visualization, rotation, HDF5 and result-inspection scripts. It did
not execute MATLAB or Octave, download the full dataset, train a model or reproduce paper results.
The separate [completed Tea forest evaluation](updated/tea-training.md) now has audited final
results; it does not establish runtime compatibility of the preserved MATLAB scripts.
The historical MATLAB scripts have no automated regression suite or current runtime validation record.
The related [DepthTrack submodule](updated/depthtrack.md) provides its own native/browser/backend tests
and recorded evidence; those checks do not validate the preserved MATLAB preprocessing or paper results.
The original scripts and their historical terms are unchanged.

The preprocessing scripts use older MATLAB APIs and local paths. `derotate.m` declares its primary
function as `derotate2`, a historical naming inconsistency that needs checking in the chosen
MATLAB release. Result inspection requires separate saved prediction files; the historical links
below may be unavailable. The data license described below does not change the code's original
academic-use notice.

## Historical download links (2015)

These are the original Technion download addresses, kept for the record. During the 2026-11-12
review, the former project page refused the connection and the download addresses timed out.
They are not current download locations. Use the [Hugging Face dataset](https://huggingface.co/datasets/twerdster/HandNet)
for the data archives.

The former project page was <http://www.cs.technion.ac.il/~twerd/HandNet/>.

Data:
* [Train (12.5 GB)](http://gip.cs.technion.ac.il/files/HandNet/TrainData.rar)
* [Test (627 MB)](http://gip.cs.technion.ac.il/files/HandNet/TestData.zip)
* [Validation (174 MB)](http://gip.cs.technion.ac.il/files/HandNet/ValidationData.zip)
* [Train_DerotGT (10.5 GB)](http://gip.cs.technion.ac.il/files/HandNet/TrainData_DerotGT.zip)
* [Validation_DerotGT (174 MB)](http://gip.cs.technion.ac.il/files/HandNet/ValidationData_DerotGT.zip)
* [Test_Perturbed (521 MB)](http://gip.cs.technion.ac.il/files/HandNet/TestData_Perturbed.zip)
* [Validation_Perturbed (144 MB)](http://gip.cs.technion.ac.il/files/HandNet/ValidationData_Perturbed.zip)

BMVC'15 results on the perturbed data (used in the paper):
* [Test_Perturbed_Results (167 MB)](http://gip.cs.technion.ac.il/files/HandNet/TestData_Perturbed_Results.mat)
* [Validation_Perturbed_Results (23 MB)](http://gip.cs.technion.ac.il/files/HandNet/ValidationData_Perturbed_Results.mat)

Results on the non-perturbed data (not recommended; provided for completeness):
* [Test_Results (167 MB)](http://gip.cs.technion.ac.il/files/HandNet/TestData_Results.mat)
* [Validation_Results (23 MB)](http://gip.cs.technion.ac.il/files/HandNet/ValidationData_Results.mat)

## Citation
Please cite this paper ([LINK](http://arxiv.org/pdf/1507.05726.pdf)) if you use HandNet in your work.

    @article{wetzlerBMVC15,
      Author = {Wetzler, Aaron and Slossberg, Ron and Kimmel, Ron},
      Journal = {British Machine Vision Conference},
      Title = {Rule of Thumb: Deep Derotation for Improved Fingertip Detection,},
      Year = {2015}
    }

## License

- **Data:** HandNet data is available under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
  as stated in the [Hugging Face dataset release](https://huggingface.co/datasets/twerdster/HandNet).
  Credit Aaron Wetzler, Ron Slossberg and Ron Kimmel, link the license and indicate changes.
  Please also cite the BMVC 2015 paper.
- **Code:** [DisplaySession.m](DisplaySession.m) retains this original notice: “This database and
  code is made available for academic purposes only. For commercial applications please contact
  the authors.” The later data-release license above does not relicense this code.
- The image above is the HandNet figure preserved from the former Technion project page.
