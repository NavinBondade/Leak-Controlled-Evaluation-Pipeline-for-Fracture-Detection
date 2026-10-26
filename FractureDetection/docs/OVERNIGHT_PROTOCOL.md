# Analysis plan: reducing missed small fractures (written 2026-10-24 23:15, a self-timestamped local file, not an external registration)
Status note: the box 100/25/10 runs and the no-box aux control had ALREADY been seen when this was written (see 'Known before starting'); they are
exploratory. Only candidates 1 (fine100/fine0) and 2 (small3) were run after it. Rule A was tightened once, after the seed-43 baseline.

Target: sensitivity on the smallest-fracture quartile of FracAtlas positives (box area <= 25th percentile; 180 images), measured
out-of-fold at the fixed 90%-specificity operating point. Baseline 0.761 (0.693-0.822). Same folds, same threshold rule throughout.

## Known before starting (already reported)
Plain aux head with NO box information (negatives only) gave +0.044 (0.000 to +0.090) small-fracture sensitivity and +0.011 AUC.
Boxes at 100% / 25% / 10% gave +0.072 / +0.039 / +0.067. So box information is not demonstrated; the aux head itself may help.
Seed noise of the plain baseline is being measured by run s43.

## Candidates (fixed now; no others may be added to a "win" claim without being declared here first)
1. Fine localisation head (32x32, 16 px cells), boxes on 100% of fractured training images (`fine100`) with matching no-box control (`fine0`).
2. Oversampling of smallest-quartile fractured training images x3, no aux head (`small3`). Matching control: the plain baseline.
3. (Only if 1 and 2 both fail) training-data expansion with GRAZPEDWRI-DX patients. Declared here; design to be written before it is run.
4. Combination of whichever of 1 and 2 pass step A, with its own control.

## Decision rule for calling something a FIX (all required)
A. Paired group-bootstrap vs the plain baseline: small-fracture sensitivity difference 95% CI lower bound > 0; overall AUC not worse than -0.01 (CI lower bound); specificity not worse than -0.03.
B. Paired vs its matching control (candidate 1: `fine0`): difference CI lower bound > 0 if the claim is credited to boxes; otherwise the claim is only "the aux head helps".
C. Replication: the same candidate retrained with a second seed (43) must show a difference with the same sign, and the pooled estimate across both seeds must satisfy A.
D. External check: scored on GRAZPEDWRI-DX with the pre-specified threshold; reported whatever the outcome. A fix that makes GRAZ transfer clearly worse is reported as such.

## Multiplicity
Because several candidates are tried, a single passing run is NOT a fix; only step C replication counts. All candidates and all results, including nulls, go in the report.

## Stop
Stop at 09:00 on 2026-10-25 or when a candidate passes A-D, whichever is first. Report whatever exists. No decision rule above is changed after a result is seen.

## Addendum 2026-10-25 00:10 (written after run s43 finished, BEFORE any result of candidates 1 or 2 exists)
Result of s43 (plain baseline, no aux head, seed 43): AUC 0.932 (+0.008 vs seed 42), small-fracture sensitivity 0.794 (+0.033 vs seed 42).
So the run-to-run change of the unmodified model is about +0.03 on small-fracture sensitivity and +0.01 on AUC, the same size as the gains
seen earlier from the aux head and boxes. Those earlier "wins" are therefore NOT credited to the method; they are within seed noise.
New requirement added to rule A: the paired small-fracture sensitivity difference must have its 95% CI lower bound > 0 against BOTH baseline
seeds (42 and 43), not just seed 42. Rules B-D unchanged.

## Outcome (2026-10-25 ~04:30; stopped early by decision, rules unchanged)
Candidates 1 and 2 were run; no candidate passed rule A against both baseline seeds, so rules B-D and candidates 3-4 were not triggered.
Full table: results/compare_latest.txt, results/box_comparison.csv, results/box_vs_control.csv.
- fine100 vs seed 43: +0.033 (-0.013 to +0.080); vs fine0 control: +0.033 (-0.010 to +0.076). Not a fix.
- small3 (oversampling) vs seed 43: -0.033 (-0.086 to +0.015). No help.
- Every box setting vs its no-box control or seed 43: CI includes zero.
Conclusion: missed small fractures are NOT shown to be fixed, and the effect of the aux/box variants is INCONCLUSIVE (all point estimates positive, about +0.03 to +0.05; strict rule not met;
exploratory comparison against the mean of both baselines puts three box runs' intervals above zero). The earlier statement that differences under 0.08 cannot be established was wrong: the paired
95% half-width is about 0.045, so about 0.065 is detectable in principle, before run-to-run variance. The seed-noise yardstick is one pair of runs. A decisive test needs 3-5 seeds per arm (not run).
Candidate 3 (adding GRAZ patients to training) is left as proposed future work; its design is not written.
