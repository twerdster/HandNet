# HandNet updates

## Updated and ported for modern hardware

A new browser explorer displays precomputed HandNet images, calibrated depth
point clouds, labels, sensor measurements and matched orientation variants
without requiring MATLAB or a full dataset download. The academic website
bundles 20 sample rows and eight matched pairs; the separate Hugging Face Space
provides a larger 96-sample research viewer. This repository retains its
historical MATLAB scripts: the viewer does not execute their preprocessing
scripts, train a model or run learned fingertip inference.

### Native forests and staged HandNet training

The pinned [DepthTrack integration](depthtrack.md) now includes a native C++17
Tea forest loader and API for complete 20-decision, three-tree CPU inference,
calibrated fingertip observations and timestamp-aware tracking. Its
[HandNet parity harness](https://github.com/twerdster/depthtrack/blob/6bbbe74711b4f3dc77699291c84876fff405db52/tools/validate_tea_handnet.py)
checks the native/Python integration separately from recognition evaluation. Seven-class
hand-only and optional eight-class learned-background models have explicit,
separate label contracts. This deep Tea path is separate from the historical
packed ten-decision CPU/Metal/FPGA example and from the browser viewer.

The [HandNet forest workflow](tea-training.md) links the repaired Tea trainer,
reproducible class-aware conversion and held-out evaluation. As of **2026-11-15
16:35 UTC**, all eight pilots and 48 validation configurations are complete on a
frozen 300-frame validation set. Probe scale 25000 with learned background was
selected for full-tree training preparation, and the first tree's feature
conversion is underway. Final ensemble/test results are pending. The winning
pilot's depth/confidence setting is not yet a final deployment choice; those
settings will be selected after all three full trees are available.
The [Tea pilot report](https://github.com/twerdster/tea/blob/handnet-forest-training/docs/handnet-pilots.md)
provides the frozen validation protocol, comparison settings and metric limits.

### Open and use

For executable depth-tree inference and temporal fingertip tracking, see the new
[DepthTrack integration](depthtrack.md). Its pinned [`external/depthtrack`](../external/depthtrack)
submodule contains code, the recorded tracking outputs, a phone-friendly replay,
and run/build guides for Mac, Linux, iOS, Android and Mojo FPGA. This is a separate
related implementation: the recovered detector's HandNet training provenance is
unverified, and its demo does not reproduce the HandNet neural network or paper evaluation.

Open the [HandNet page](https://twerdster.github.io/HandNet/#explorer) for the
embedded viewer, or the [standalone viewer](https://twerdster.github.io/HandNet/explorer/).
Select an example, rotate the point cloud, inspect image and heatmap channels,
read recorded sensor measurements, or compare a matched pair. Viewing requires
a browser with JavaScript; it needs no account, backend or full-dataset download.

For a local copy, follow the [website repository setup](https://github.com/twerdster/twerdster.github.io/blob/main/README.md#run-locally).
Run `npm run dev` from that repository's root and open
`http://127.0.0.1:4173/HandNet/explorer/`; no package installation or build is
required. The [viewer guide](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/explorer/README.md)
describes coordinates, rendering conventions, controls and local assets.
The [implementation](https://github.com/twerdster/twerdster.github.io/tree/main/HandNet/explorer)
and [sample provenance](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/explorer/SAMPLE_PROVENANCE.json)
remain tracked with the website; this GitHub guide collects the update details.

### Scope and recorded checks

The local selection contains **20 examples and eight matched pairs**: five test
originals with their perturbed counterparts, three validation originals with
ground-truth-derotated counterparts, and four derotated training examples.
The provenance record identifies source frames and hashes, calibration,
paired-source consistency and point-array lengths. It records checks of the
precomputed assets; it is not a detector evaluation or a full-dataset benchmark.
The separate [Hugging Face research viewer](https://huggingface.co/spaces/twerdster/HandNet-Explorer)
contains 96 examples. Neither selection replaces the original distribution archives.

Depth previews are contrast-rendered, and point clouds are downsampled for
display. Ground-truth normalization uses measured palm orientation. Six tracked
sensor locations do not form a full anatomical hand skeleton. The viewer runs
no learned model inference and reports no new training or accuracy result.

### Data, code and attribution

Credit Aaron Wetzler, Ron Slossberg and Ron Kimmel and cite
*Rule of Thumb: Deep Derotation for Improved Fingertip Detection*, BMVC 2015.
The [HandNet dataset](https://huggingface.co/datasets/twerdster/HandNet) and
precomputed examples are [CC BY 4.0](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/explorer/DATA_LICENSE.txt).
New viewer code is [MIT](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/explorer/LICENSE);
bundled Three.js retains its [own MIT notice](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/explorer/vendor/THREE-LICENSE.txt).

### Historical code and limits

The [MATLAB usage and validation notes](../README.md#visualize-a-dataset-split)
remain the reference for [DisplaySession.m](../DisplaySession.m),
[CreateH5Data.m](../CreateH5Data.m) and the other preserved scripts.
No current MATLAB/Octave compatibility or benchmark reproduction is established
by the browser viewer. Original [code terms, dataset licensing and citation](../README.md#license)
remain separate; the dataset license does not relicense the historical scripts.
