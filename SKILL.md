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
current published source branch is `main`, with exact reviewed pin
`356060770b65e5c60ef5db702482da8a30f7dbb0`. The gitlink must remain mode 160000.
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

Keep training quality separate from numerical parity and the older tracking demo.
The original three full trees and their 10,000-frame final evaluations completed
2026-11-16. Report mean fingertip F1@15mm 0.37494755678731473 ensemble /
0.32798765233655836 best single (seed 202); descriptive 2,773-frame validation is
0.3833437004586795 / 0.3370599042841775 and includes the 300-frame selection subset.
The 24-case final grid froze 20 decisions/cutoff 0.25/minimum four-connected support 3/
foreground `none`/fingertip IDs 0..4 before test inputs. Preserve frozen Tea training
revision 27ae728e9cb012d2255337d9b3583bbe7f66b3cc separately from newer docs commits.
Keep modest-result, operational metric, missed/absent-finger accounting and random-
frame/subject-independence limitations explicit. Additional post-disclosure
experiments are exploratory, not independent untouched confirmation. This HandNet
integration does not independently launch work, change authorized budgets or
control unrelated jobs.

Keep the optional seed204 package separate from the original default forest and
best seed202 single. Its matched tree0 bag has 181,979 frames, 200,216,884 balanced
samples and 256 exact saved coordinates; changing sampling/feature count and seed
together does not isolate one causal factor. Link Tea's `docs/handnet-comparison.md`
for the frozen six-setting validation protocol and audited reporting status.
The optional release tag is `handnet-200m-256-seed204-2026-11-16`; preserve the
reviewed ZIP checksum and verified public report/prerelease availability.
The independent full-cohort audit passed: test F1 0.3302297543 versus matched201
0.3402043018; full validation 0.3372283285 / 0.3477379235 includes the selection
subset. Preserve these exploratory outcomes and all absent-finger accounting;
do not promote or retune a model from test results. Raw-reference arithmetic uses
supplied predictions and is separate from selected-model native inference parity.
Use the generic Mac/Linux CPU loader explicitly; preserve the original default
downloads, browser models/routes and reviewed submodule pin until deliberately
advanced. No deep Metal/mobile/FPGA, temporal-accuracy or latency claim follows.
Current backend/API/harness and optional seed204 parity links use exact published
DepthTrack pin `356060770b65e5c60ef5db702482da8a30f7dbb0`. Its PR2 head passed
Mac/Linux/Android checks; exact-merge main run `43832975971` also passed Mac,
Ubuntu, Android and Pages. Preserve the distinction between source CI, hosted
asset verification, iOS cross-builds and actual device/model runtime. Preserve historical
stage pins when advancing current links. Refuse a dirty nested worktree; verify
public reachability and ordinary recursive checkout before committing the gitlink.

Before a commit, synchronize this SKILL and SUMMARY, verify every historical
MATLAB/MAT source/data/calibration/index byte remains unchanged, check guide links
and the exact submodule pin, and report synthetic/native checks separately from
training and scientific evaluation. Preserve all historical attribution and terms.

Link exact completed source CI runs when reporting implementation validation.
A passing Tea tooling/DepthTrack CI run does not mean GPU training completed or
that a new model has validated recognition accuracy.

## HandNet parity harness evidence

Keep current backend/tool/CI links aligned with the exact reviewed submodule pin.
The `validate_tea_handnet.py` tool compares supplied deployment artifacts and the
matching Python reference on identical train/validation frames; it is separate
from training, model selection and final-test evaluation. Declare model and data
roles and timestamp provenance. The historical fixture record uses an
untrained forest and synthetic 0.1 s timestamps on real training frames. The actual
selected ensemble and seed 202 single now have separate Linux/Mac parity on the
same 3 training frames, both foreground policies, cutoff 0.25/minimum support 3: exact pixel outputs,
observation/tracker differences <= 3.411e-13 mm and Linux/Mac difference 0. Both
records use synthetic cadence and establish numerical agreement only; do not
claim measured temporal accuracy, held-out recognition or latency from parity. A skipped Pages
job is not a website deployment. Tool/docs-only pin changes need reachability,
recursive checkout and protected-byte checks; repeat native builds/tests only
when source behavior changes or another unresolved concern justifies them.

## Publishing completed forest results

Keep the preserved eight-pilot/48-configuration report distinct from the 24-case
final selection grid, descriptive 2,773-validation-frame cohort and 10,000-frame final test.
The [Tea final report](https://github.com/twerdster/tea/blob/main/docs/handnet-forest-results.md)
and safe evidence JSON are the detailed result/metric references. Selected assets
use release tag `handnet-forest-2026-11-16`; verify public report/release availability
and checksums before publishing links. Keep both frozen selected choices and no
test-driven retuning explicit. Do not infer a blanket model license: embedded
HandNet dataset CC BY 4.0 attribution is separate from historical MATLAB academic terms.

Align backend/API/harness/selected-parity links with the exact reviewed published
DepthTrack gitlink; preserve mode 160000 and verify ordinary recursive checkout.
Recheck relative Markdown links, protected historical bytes and source revisions.
Native code/builds need not be repeated for documentation-only pin changes when
prior executed numerical/native checks retain their stated scope. Keep the
recurring monitor through original result reporting/publication/integration;
only pause after all watched work finishes and results are reported, or user
cancellation. A pending optional experiment-scope question must not indefinitely
block reporting the already authorized original campaign. No additional training
follows from documentation maintenance.

For the separate live monocular route, preserve the distinction between relative estimated depth and sensor mm, two-anchor approximate calibration, manual range/ROI masks and hand segmentation, and candidate component tracks versus verified fingertips. It runs neural inference and the compact tree locally in a browser; it is neither a HandNet-trained network nor FPGA execution.75 local browser tests and synthetic-camera lifecycle evidence do not establish physical phone runtime or anatomical accuracy. Link pinned DepthTrack docs when updating the overview; keep the existing recorded replay and original MATLAB sources separate.

When advancing the live adapter pin, use the source-commit CI evidence (initial live version:de334f7/run43735739340), verify hosted modules/model bytes, and keep physical-phone and anatomical-accuracy status separate from successful deployment/cross-builds.


## Combined upstream preservation

The current exact DepthTrack pin 356060770b65e5c60ef5db702482da8a30f7dbb0 combines
native Tea forests/selected HandNet results and the upstream live monocular
experiment. Preserve both during future merges: do not replace browser camera/
model/runtime files with the full forest assets, or turn compact-model browser
evidence into a native 20-decision/HandNet recognition claim. Keep backend/API/
harness/selected-evidence/live-guide links synchronized with the gitlink.
`.gitmodules` tracks published `main` for deliberate future changes; ordinary
recursive checkout remains pinned. Source documentation and pin integration
require protected 10-file hash/relative-link/reachability checks, not unrelated
training or destructive rebuilds. Recheck SUMMARY and this workflow before commit.
