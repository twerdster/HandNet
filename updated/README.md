# HandNet updates

## Updated and ported for modern hardware

A new browser explorer displays precomputed HandNet images, point clouds,
labels and matched orientation variants without requiring MATLAB or a full
dataset download. The academic website bundles 20 sample rows and eight
matched pairs; the separate Hugging Face Space provides a larger 96-sample
research viewer. This repository retains its historical MATLAB scripts:
the modern viewer does not execute their preprocessing scripts, train a model
or run learned fingertip inference.

### Modern viewer and documentation

- [Website viewer update guide](https://github.com/twerdster/twerdster.github.io/blob/main/HandNet/updated/README.md#updated-and-ported-for-modern-hardware):
  implementation, sample provenance, controls and validation boundaries.
- [Open the local-data browser explorer](https://twerdster.github.io/HandNet/explorer/)
  or the [HandNet project page](https://twerdster.github.io/HandNet/).
- [Website viewer source](https://github.com/twerdster/twerdster.github.io/tree/main/HandNet/explorer)
  and [separate research Space](https://huggingface.co/spaces/twerdster/HandNet-Explorer).

### Historical code and limits

The [MATLAB usage and validation notes](../README.md#visualize-a-dataset-split)
remain the reference for [DisplaySession.m](../DisplaySession.m),
[CreateH5Data.m](../CreateH5Data.m) and the other preserved scripts.
No current MATLAB/Octave compatibility or benchmark reproduction is established
by the browser viewer. Its precomputed subset is not the complete native depth
dataset; canonicalization uses ground-truth orientation rather than learned
model inference. Original [code terms, dataset licensing and citation](../README.md#license)
remain separate.
