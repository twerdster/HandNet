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

## Native Tea forests and staged training

Keep `external/depthtrack` pinned to a reviewed, remotely reachable commit; its
current source branch is `tea-handnet-forest`. The gitlink must remain mode 160000.
`updated/tea-training.md` connects the inference API to Tea's separate repaired
`handnet-forest-training` source. Record exact reviewed commits and verify ordinary
recursive checkout after pin changes. Preserve the original HandNet worktree.

Use the native Tea backend guide for complete 20-decision, three-tree CPU/API
claims. Header/Tea CLI 19 means 20 decisions. Seven hand classes have no learned
background; optional modeled class 7 in an eight-class forest is distinct from
invalid-centre 8 and reserved unknown 255. Coordinate banks, float32 complete-sum
features, posterior normalization, centre-mask policy, calibration and real
monotonic timestamps are explicit contracts. Keep historical packed ten-decision
CPU/Metal/FPGA and mobile-tracker evidence separate from deep Tea CPU execution.

Keep training status separate from implementation parity and the older recorded
tracking demo. The current bounded campaign is waiting for GPU resources; do not
claim newly trained models or held-out accuracy until artifacts and actual
predictions are verified. At most eight pilots and three final trees are in the
initial campaign, under its existing queue, budget and recurring monitor. A
HandNet documentation/submodule integration does not independently launch work,
change that budget or control unrelated jobs. Select configuration on validation
and reserve test data for final metrics; randomized frame splits do not prove
participant-independent generalization.

Before a commit, synchronize this SKILL and SUMMARY, verify every historical
MATLAB/MAT source/data/calibration/index byte remains unchanged, check guide links
and the exact submodule pin, and report synthetic/native checks separately from
training and scientific evaluation. Preserve all historical attribution and terms.

Link exact completed source CI runs when reporting implementation validation.
A passing Tea tooling/DepthTrack CI run does not mean GPU training completed or
that a new model has validated recognition accuracy.
