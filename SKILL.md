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

Source inspection only; no current MATLAB/Octave run, training, dataset download or benchmark reproduction. No automated regression suite is included.

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
