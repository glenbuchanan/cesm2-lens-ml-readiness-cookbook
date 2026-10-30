# Glossary

The terms the chapters use, in the sense they use them.

| Term | Meaning here |
|---|---|
| ENSO, Niño 3.4 | El Niño–Southern Oscillation; the Niño 3.4 index is the mean surface temperature over 5°S–5°N, 170°–120°W, the input here |
| SON, DJF | September–October–November (the autumn input) and December–January–February (the winter target) |
| forced response | the warming path every ensemble member shares by calendar year; removed so a model reads ENSO, not the decade |
| precision | of the winters a model calls wet, the share that were wet |
| recall | of the winters that were wet, the share the model called wet |
| AP (average precision) | precision averaged over every threshold of the model's ranking; a ranking with no skill scores the base rate |
| base rate | the share of winters that are wet; a classifier with no skill has precision equal to it |
| wet-call share | the share of winters a classifier calls wet; a classifier with no skill has recall equal to it |
| ladder, rung | the candidate models in order of flexibility: rung 0 is three rules that fit nothing, rungs 1–3 are logistic regression, a decision tree and gradient boosting |
| trial | one recorded change, scored against the incumbent on the same folds |
| incumbent | the model currently ahead; a trial replaces it only if the promotion rule says so |
| register | the table of every trial, its paired-test results and its verdict |
| looks | the tests taken so far: two (recall, precision) per register row |
| Bonferroni line | 0.05 divided by the looks; a `p` below 0.05 but above it is a promotion under the uncorrected rule only |
