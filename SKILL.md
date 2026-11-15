# HandNet maintenance workflow

Read `SUMMARY.md`, this file and `README.md` before editing.

## Working on the repository

Keep archive directory names aligned with DisplaySession.dirName. Read source comments before metric or compatibility claims. Preserve the original citation, data-license explanation, code terms and historical download provenance.

## Documentation checks

- Check claims against the named source files and recorded results. Separate source inspection,
  configuration generation, runtime execution and benchmark evidence.
- Keep README setup requirements available without following an advanced guide. Check Markdown
  headings, links and image references, including source files omitted by a sparse checkout.
- Inspect labels inside diagrams as well as captions. Never turn a schematic or generated input
  into an implied successful run.
- Keep original source, credits and license notices intact for documentation-only changes.
- Review and synchronize `SUMMARY.md` and `SKILL.md` before every commit. Record any unrun checks
  as limitations rather than implying that validation passed.

## Current evidence boundary

For historical MATLAB: source inspection only; no current MATLAB/Octave run, training, dataset download or benchmark reproduction. No automated regression suite for those scripts is included. The related DepthTrack submodule has separate executed native/browser/backend checks.

Older toolbox APIs, hard-coded preprocessing paths, separate result files, and derotate.m declaring derotate2. Dataset CC BY 4.0 does not change the original code academic-use notice.

## Modern viewer navigation

Keep the local New! badge and exact “Updated and ported for modern hardware”
link below the root README title, targeting `updated/README.md` and its matching
heading. Put the brief modernization summary and further links there.
Keep the update guide self-contained in this GitHub repository: briefly describe
the changes first, then provide usage, scope, provenance and license details
under the updated heading. Link existing website explorer documentation, source
and assets with absolute public URLs; do not depend on a duplicated website
update guide. Distinguish the website/Hugging Face viewers from the preserved
MATLAB code. Preserve the reference body and original script bytes, attribution
and code/data terms. This navigation request covers GitHub repositories only;
website navigation changes require a separate user request.

## DepthTrack integration

Keep`external/depthtrack`as a true Git submodule pinned to a reviewed upstream commit. Update the gitlink deliberately after validating the new commit; use`git submodule update --init --recursive`for ordinary checkout. Do not copy build trees or duplicate the full standalone code into this parent.

`updated/depthtrack.md`is the parent overview and output/platform entrypoint. Preserve pinned output image URLs when exact provenance matters; link the standalone repo for detailed commands, evidence and component notices. The130-frame HandMeshRecord demo and recovered ZScanV02 models have unverified HandNet training/split provenance. Distinguish detector-relative residuals from anatomical accuracy, cross-builds from device runtime, browser tracker execution from learned inference, and FPGA tree classification from host temporal tracking.

Before every commit, review/update this SKILL and SUMMARY for behavior, workflows, outputs and handoff context. Validate the submodule commit is reachable remotely, relative links resolve and recursive checkout works; verify original MATLAB/calibration/indices bytes remain unchanged. Native mobile examples are integration fragments, not complete camera apps. No website-navigation changes or training are required by this integration.
