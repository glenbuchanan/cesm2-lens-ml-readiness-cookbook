# Pre-registered out-of-period test (written 2026-10-29, before any SSP3-7.0 data is read)

Glen chose "2": a fresh test. No ensemble member is unscored (80 training members drove every choice; the 20 test
members were scored on 2026-10-20), so the untouched data is the same members' future winters under SSP3-7.0.

Data: CESM2 LENS ssp370 stores (cmip6 and smbb; PRECC, PRECL, TS), the same box, Niño 3.4 box, season (DJF),
autumn (SON) and member list as the historical table. Winters 2016–2100 (DJF 2015 would need Dec 2014 from the
historical store; it is dropped). Derived with the same derivation code; committed as data/member_winters_ssp370.csv.

Frozen before reading: the final model is boosting@base-rate exactly as chapter 6 fits it (all 80 historical training
members, same seed, same hyperparameters, threshold 1 − WET_QUANTILE). Comparators: always-wet, always-dry, nino-sign,
logistic@base-rate, fitted the same way. No refitting on future data, no hyperparameter or threshold change.

Transforms for the future period: the forced response (degree FORCED_DEG polynomial in year) and the wet threshold
(pooled WET_QUANTILE) are fitted on the 80 TRAINING members' future winters only, then applied to the 20 TEST members'
future winters. Nothing is fitted on, plotted from, or summarised over the test members' future winters before the
single scored run.

Primary outcome (decided now): precision − base rate for boosting@base-rate on the test members' future winters, with
the same resampling-over-test-members 95% interval as chapter 6. "Edge replicates" if the interval's lower bound is
above 0; "does not replicate" otherwise. Secondary, reported whatever they show: AP − base rate with interval; recall
and wet-call share; false alarms per hit; the same numbers for the comparators; the decile wet rates on the training
edges with a chi-square test.

Reporting: whatever the run shows is reported as it comes, including a null. The scored run happens once; later builds
recompute the same numbers deterministically and change nothing. Limits to state: a different period under stronger
forcing (distribution shift), model world, the existing seven limits.
