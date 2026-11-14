# HandNet updates

## Updated and ported for modern hardware

A new browser explorer displays precomputed HandNet images, calibrated depth
point clouds, labels, sensor measurements and matched orientation variants
without requiring MATLAB or a full dataset download. The academic website
bundles 20 sample rows and eight matched pairs; the separate Hugging Face Space
provides a larger 96-sample research viewer. This repository retains its
historical MATLAB scripts: the viewer does not execute their preprocessing
scripts, train a model or run learned fingertip inference.

### Open and use

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
