# Machine Learning Handbook

*Content CC BY 4.0. Shape after NSF Unidata's "Machine Learning Handbook" (Supervised Machine Learning Readiness, Apache-2.0, NSF award #2319979): the learner records a response per exercise; the Rubric ([`course/rubric.md`](https://github.com/glenbuchanan/ncar-ml-readiness-course/blob/main/course/rubric.md) in the full course) scores it, first by the learner, then by the trainer.*

**How to use this handbook.** Work through `02_applications.ipynb`. Each exercise ends with "→ Handbook entry N"; the notebook's box is a pointer — the prompt here, with its evidence list, is the prompt of record and what the rubric scores. Copy the entry's evidence (a table, a printed line, a figure) into the box, then write the response. An entry with no evidence pasted is scored *Not yet* whatever the prose says.

| Learner | Date | Notebook run (`RNG_SEED`, `SEASON`, `WINDOW`, `N_MEMBERS_PER_HALF` as printed in 2a's cache header; `N_FOLDS` as printed in 2c) |
|---|---|---|
| | | |

---

## Part 1 · Problem framing

### Entry 1 — The expected sign, before fitting (Exercise 1) · O1 *framing*

**Prompt.** Write this entry after the `run started …` line has printed under Imports and before any data cell has run. NWS Boulder's own slides say that east of the Continental Divide the La Niña jet patterns bring "below average precipitation" with downslope winds, and that the southwest jet of moderate-to-strong El Niños brings above-average precipitation "particularly during the late winter and spring". (a) State the sign you expect between the autumn Niño 3.4 anomaly and your chosen season's Front Range precipitation — positive, negative, or none — in one sentence that names the slide it rests on. (b) Say why `SEASON` changes how much signal there is to learn: which months the office puts the El Niño wet signal in, and whether DJF or FMA contains more of them. (c) Record the `SEASON` you chose, the reason and the cost. DJF is the shipped path: the committed cache was derived for it and every cell after 2a runs from it in minutes. FMA is a real choice with a real cost: `REBUILD = True` in 2a, reading roughly 160 GB (`data/README.md`) — the one full derivation took 33 minutes on a fast link (2026-10-20); on home broadband expect an hour or more; the cache cell refuses to run if the cache was derived with a different season than you declare, so the choice cannot be made silently. If you keep DJF because of the cost, say so — a decision recorded with its cost is the exercise. A rebuild is a new access: 2a stamps its own access date into the cache header and into every figure's citation (the `ACCESS_DATE` in the parameter cell is the committed cache's, and 2a asserts the two agree).

**Evidence to paste.** The `run started …` line printed under Imports (date, time and kernel of your run). Write (a)–(c) after that line prints and *before* 2a runs: the pasted line is the record of when your run began, and Entry 3's cache header, pasted after it, is the record of what ran next. Nothing checks this order; it is yours to keep. The notebook cannot see when you typed this — the pasted line and this entry's place above Entry 3's pasted cache header are the record — and the trainer scores the entry on its content with the line present.

**Response.**

> | Part | Your line |
> |---|---|
> | `run started …` line, pasted | |
> | (a) the sign, and the slide it rests on | |
> | (b) why `SEASON` changes how much signal there is | |
> | (c) `SEASON` chosen · reason · cost | |

---

### Entry 2 — The three framing answers (Exercise 2) · O1 *framing*

**Prompt.** Answer Unidata's three framing questions yourself before you open the notebook's three answers — they sit folded under the two-boxes figure in Part 1 (*The worked answers*) and stay closed until you click them — then unfold them and note in one line where yours differ. *Does a simpler solution exist?* — name the three simple solutions every model must beat, the same three Entry 6 scores: the two base-rate classifiers, always-dry (the accuracy bar) and always-wet (the precision bar, at the base rate), and the Niño-sign rule (the physical prior — no bar on any metric, the words Entry 6's row uses for it); then answer: is any of the three a closed-form case? Say why or why not in one sentence. *Can machine learning requirements be met?* — the data side from the catalogue output in 2a (which stores, how many members and winters, which variable the index is built from and why) — this half is completed after 2a has printed, the rest of the entry after the `run started …` line and before any data cell runs — and the physics side from Stagge et al. 2023: the Front Range sits in the transition zone of the ENSO dipole, so the honest prior is a small signal. *Which scientific question should be answered?* — write the target as a decision, naming the person who would act on a "wet" call and what they would do differently; Unidata's own example is a one-line question in quotation marks, and yours should fit on one line too.

**Evidence to paste.** The `run started …` line (the line Entry 1 pastes); your own one-line question beside the italic one-line question from the notebook's Part 1 (the third framing answer); the three values `FRONT_RANGE`, `SEASON`, `WET_QUANTILE` copied from the parameter cell; for the data side, 2a's catalogue line (`catalogue last_updated … rows selected`) and its count line (`… members of … in the two stores`).

**Response.**

> | Framing question | Your answer |
> |---|---|
> | Simpler solution — the three simple solutions (always-dry = accuracy bar · always-wet = precision bar · Niño-sign = physical prior, no bar); is any of them closed-form? | |
> | Requirements met — the data side (2a's two lines) · the physics side | |
> | The question, as a decision — mine (one line, with the person who acts on it) | |
> | The notebook's, and where mine differs | |

---

## Part 2 · Data handling

### Entry 3 — The Metadata Document (Exercise 3) · O1 *data* · O1 *train*

**Prompt.** The Metadata Document in notebook 2a (general information · structure · quality · provenance) arrives filled. Check every cell the printed output can show against it — the `catalogue last_updated` line, the row counts, the store paths, the units and calendar, the member count, the cache header and the version lines under Imports — correct any cell the printed evidence contradicts, and mark it. **The two marks, defined — per value, not per cell:** *checked* = a value that appears in a line this run printed; *not printed this run* = a value no printed line shows, which you name. A row carrying both kinds of value takes both marks. Apply that rule row by row and value by value: a row is *checked* for every value a printed line shows and *not printed this run* for every value none shows, with that value named — a row whose text looks checkable can still carry an unprinted value, and there is no fixed number of such rows. For example, the Known-gap row's "ocean TEMP absent" is *not printed this run* (2a's search asked the catalogue for PRECC, PRECL and TS only, so no printout can show TEMP's absence); the Grid row is *checked* for the grid, calendar and chunking (the `dims`, `calendar` and `chunking` lines) and *not printed this run* for the "m/s → mm/day × 8.64e7" factor (it is in the derivation code, never printed); and the Environment row is *checked* for the versions (what prints under Imports is the versions this run resolved) and *not printed this run* for the pins (they live in `environment.yml`). The same per-value split applies wherever a row carries one value a line prints and another none does. Cells no printout can show (the initialisation groups, the smbb forcing description) are *not printed this run*; the six store paths and the chunk arithmetic are printed in full, so those rows are *checked*. A run whose 2a printed `OFFLINE` marks the catalogue and store rows *not printed this run* too. Copy the table here with the marks. For each *Quality* row, add one sentence on how that fact could bias the model. The evidence this entry carries — the seed, the version lines, the cache header — is what makes the fit rerunnable (O1's *train* row).

**Evidence to paste.** The completed table; the printed `catalogue last_updated` line; the first line of `data/member_winters.csv` — the derivation header, `# derived by 02_applications.ipynb: SEASON=… access=…`, which is the first stdout line of 2a's cache cell (the cell under *Deriving the member-winter table*, printed by `print(cache_header)`) — and the `cache header matches the parameter cell` line printed directly under it; the version lines printed under Imports.

**Response.**

> | Part | Field | Value | Evidence | Checked / not printed this run (name the value no printout shows) |
> |---|---|---|---|---|
> | | | | | |
>
> *Self-check — what a correct table looks like: every value no printed line shows is **named** in the last column as *not printed this run*; every value you can point to in a printed line is marked *checked*; a row with values of both kinds carries both marks — so a table in which no row carries both has applied the rule per cell, not per value: look again.*
>
> Quality → bias, one sentence per row:

---

### Entry 4 — The two halves (Exercise 4) · O1 *data* · O2

**Prompt.** The ensemble's two halves differ by design: 50 members ran on the original CMIP6 biomass-burning forcing and 50 on a smoothed version (11-year running means over 1990–2020) with two minor bug corrections, and the project page says the corrections did not change the solutions significantly. Before pooling them, read the per-half tables from 2b: do the two base rates differ by more than a few hundredths, and do the per-half input–target correlations agree in sign? Then decide whether pooling the halves is justified, and give the reason from the numbers you pasted; say also whether the sign you wrote in Entry 1 appeared — read the `per-member sign count` line (N positive · M negative, with the mean and SD) and the `per-half mean correlation` line printed under the 80 per-member lines in 2b. The alternative — keeping one half — is a costed next change for Entry 11(3), not a change made in this notebook: a filter on the table (`forcing_variant`), no rebuild, at the price of half the members and half the statistical power; `forcing_variant` is not offered as an input (the notebook's 3b says why: it labels a member's half, and the cache carries one input column). A difference that is small, or absent, is a finding to record, not a problem to fix.

**Evidence to paste.** The per-half base-rate table (training members), the `per-member sign count` line and the `per-half mean correlation` line from notebook 2b. The threshold is one for both halves, fit on training members, so the two base rates are free to differ; a difference, or its absence, is the finding.

**Response.**

> | training members | cmip6 | smbb |
> |---|---|---|
> | base rate (the per-half table) | | |
> | per-half mean correlation (the printed line) | | |
>
> Decision:
>
> Reason (from the two rows above):
>
> Did the sign from Entry 1 appear — sign count ___ positive / ___ negative of ___; per-half means ___ / ___:

---

### Entry 5 — The leak you cannot see in a split (Exercise 5) · O1 *split*

**Prompt.** A colleague proposes a tidier pipeline: fit the forced curve and the wet threshold (or a `StandardScaler`) on the whole table, test members included, before any split, "because they are only normalisation". (a) Say what the test score in 3g would then measure, and which step of the spine that fit belongs to. scikit-learn's rule is the one to apply: a normalisation step's average "should be the average of the train subset, not the average of all the data", and the general rule is "never call fit on the test data" — so name the leak and the step (*split*, then *train*: the split comes first, and every transform is fit inside it). (b) Read the contrast block from 2c — two lines per arm. The first line is the arm's scores: AP, precision ± its per-fold SD, recall ± its per-fold SD, accuracy, and the **wet-call count** (how many of the scored winters the model called wet, with its share). The second is the arm's gaps against A — AP, precision, recall — each as mean ± SD of the per-fold gaps, ending in a verdict: *gaps readable* or *NON-TEST*. The model is boosting at the base-rate decision threshold (3e's trade-off), so that it calls wet; the line printed above the block's header scores the same model at scikit-learn's default 0.5 and shows how rarely it calls wet there, and the precision of a model that almost never calls wet rests on a handful of calls — a gap in it is a non-test, which is what the verdict word records. The three arms, one change each against A:

| Arm | Split unit | Forced curve and wet threshold fit on | A held-out score measures |
|---|---|---|---|
| A — the notebook's pipeline | member | that fold's training members only (refit in every fold) | new members |
| B — the colleague's pipeline at fold level | member | all the training members, once, before the folds (the test members still outside) — so each fold's held-out members helped fit them | new members scored with transforms they helped fit |
| C — member-blind split | winter (members ignored) | that fold's training winters only | new years of members the model already knows |

**Prompt (continued).** (b1) For B and C: which way did AP, precision and recall move against A, and by how much? Read each gap in the direction it was measured, with its sign and size, however small — but only where the arm's wet-call share makes it readable: if the block prints *NON-TEST* for an arm (wet calls under one in a hundred winters), say that its precision and recall gaps are a non-test instead of reading them, and read its AP gap alone. (b2) Is each gap outside the SD of its per-fold values printed beside it? That SD is a rough scale here; 3f's paired test is the measurement. (b3) What does each arm measure — new members, new members scored with transforms they helped fit, or new years of members the model already knows? (c) Say what this design could and could not show. (c1) The one input carries no member identity, so what could a member-blind split memorise? (c2) What do the lag-1 lines in 2c say a member's winters share? (c3) Read the sibling line under them — `siblings − unrelated … → PASS: no sibling excess detectable (|excess| ≤ 2 SE, SE a lower bound) — the member is taken as the unit`, or its `FAIL` — as the measurement that licenses the member, not the start year, as the unit of the split: a non-rejection at the precision the line prints (its SE is a lower bound), not a proof that no memory remains. Entry 3's Window row records the design reason; this line records the check. Do not report an inflation your lines do not print.

**Evidence to paste.** The two assertion-passed lines from notebook 2c (`train: … members`, `… member-grouped folds verified`), the lag-1 `mean:` line, the two sibling lines (`… siblings − unrelated … → PASS` or `FAIL`, one per anomaly), and the contrast block: its header line (`boosting at threshold …`) and the two lines per arm (`A member split …`, `B member split …`, `C member-blind split …`, each followed by its `gap vs A:` line) with AP, precision ± SD, recall ± SD, accuracy, wet calls, and the three gaps with their SDs and the readable/NON-TEST verdict.

**Response.**

> (a) The leak, and the spine step the fit belongs to:
>
> | arm vs A | AP gap ± SD | precision gap ± SD | recall gap ± SD | wet calls (share) | gaps readable? | what the arm measures (b3) |
> |---|---|---|---|---|---|---|
> | B — transforms fit before the split | | | | | | |
> | C — member-blind split | | | | | | |
>
> (b1)–(b2) read off the table, in the direction measured:
>
> (c1) what a member-blind split could memorise here · (c2) the lag-1 line · (c3) the sibling line (PASS / FAIL) and what it licenses:

---

## Part 3 · Model development

### Entry 6 — The bar (Exercise 6) · O2

**Prompt.** Record precision and recall for the *wet* class, with accuracy and AP, for the three rung-0 rules (always-dry; always-wet; Niño-sign rule) on the member-grouped folds, and the base rate. Say which of the two base-rate rows is a bar on which metric, and which number in its own row makes it so — read off the rows. The Niño-sign row is the physical prior, not a bar: its *bar on* cell is pre-filled `physical prior — no bar` (the words Part 1 and Entry 2 use), and the question for that row is where its precision sits against the always-wet bar — above or below the base rate. These are the numbers "the simplest wins" refers to in Entry 11.

**Evidence to paste.** The `always-dry`, `always-wet` and `nino-sign` rows of the 3d summary table (`precision_pooled`, `recall_pooled`, `accuracy`, `ap`), with the `base_rate` column.

**Response.**

> | rung 0 | precision (wet) | recall (wet) | accuracy | AP | bar on |
> |---|---|---|---|---|---|
> | always-dry | | | | | |
> | always-wet | | | | | |
> | Niño-sign rule | | | | | physical prior — no bar; its precision sits ___ the always-wet bar |
>
> base rate = ___
>
> What the three rows mean for the ladder (one sentence):

---

### Entry 7 — Read the matrices (Exercise 7) · O2 *validate* · O1 *validate*

**Prompt.** For `nino-sign` — the rung-0 rule that calls wet in some winters but not all, so its precision is neither 0/0 nor the base rate by construction and both its numbers carry information — compute precision and recall for *wet* by hand from its summed confusion matrix (rows = true, columns = predicted) and check them against `precision_pooled` and `recall_pooled` in the 3d table; say why `precision_fold` can differ. Then relabel the same four counts in the vocabulary of MET, the Developmental Testbed Center's verification package (NSF NCAR, NOAA, CSU/CIRA and CU/CIRES) — whose 2×2 contingency table prints the same four counts with the axes swapped, rows = forecast, columns = observation, "Yes" first, so hits and correct rejections exchange corners (scikit-learn's bottom-right and top-left become MET's top-left and bottom-right) while false alarms stay top-right and misses bottom-left in both layouts — read every cell off the axis labels, not off scikit-learn's corners: wet·wet = *hits*, true-dry·predicted-wet = *false alarms*, true-wet·predicted-dry = *misses*, dry·dry = *correct rejections*; name POD = hits ÷ (hits + misses), which is recall, and FAR = false alarms ÷ (hits + false alarms), which is 1 − precision; and because precision and recall disagree for this rule, report the frequency bias, (hits + false alarms) ÷ (hits + misses) — wet calls per wet winter, above 1 when the rule calls wet too often. This is how NSF NCAR's own 2020 summer school reported a classifier: scikit-learn's confusion matrix read back as POD, success ratio and frequency bias. Both comparison questions that follow are read on the **pooled** columns (`precision_pooled` / `recall_pooled`, the ones your hand calculation reproduces), not the fold-mean columns — the two estimators can give different answers for a rung that calls wet rarely, and the table carries both. Which **fitted** rung (1–3), if any, beats `nino-sign` on *both* pooled precision and pooled recall? (Rung 0's always-wet beats it on recall by construction — 1.0 — and on precision whenever the sign rule's precision sits below the base rate; Entry 6 records which side yours is on. Either way it is read as a bar and not as a model, so it is not an answer here.) And which rung's `precision_pooled`, if any, clears the always-wet bar, the base rate? (The table's `called_wet` column — the pooled share of winters the rung called wet — shows how thin a rung's evidence is: a precision that clears the bar on a handful of wet calls rests on a handful of counts.) Then note whether `precision_fold` gives the same answer for that rung, and if not, why: a fold in which the rung calls wet in few winters, or none, scores a ratio of small counts (0 with `zero_division=0`) and the SD printed beside the fold mean is the tell; the pooled column divides once over all the folds' counts. No rung can beat `always-wet` on recall: its recall is 1.0 by construction, so say what that number means at base-rate precision and why it makes always-wet a bar rather than a model.

**Evidence to paste.** The summed `nino-sign` confusion matrix — the four counts the notebook prints as text above the 3d table (`nino-sign summed confusion matrix …`; the figure draws the same matrix), your two hand calculations, the four counts relabelled (hits · false alarms · misses · correct rejections) with POD, FAR and frequency bias, and the table's `precision_pooled` / `recall_pooled` / `precision_fold` for that row.

**Response.**

> | `nino-sign`, summed over the folds — scikit-learn's layout, as 3d prints it (rows = true, *dry* first) | predicted dry | predicted wet |
> |---|---|---|
> | **true dry** | correct rejections ___ | false alarms ___ |
> | **true wet** | misses ___ | hits ___ |
>
> (MET prints the same four counts with rows = forecast, "Yes" first; the MET names are written into the cells above so the counts are read off the labels, not the corners)
>
> Precision = ___ / ___ = ___ Recall = ___ / ___ = ___ (table: pooled ___ / ___; fold mean ___ — differs because:)
>
> POD = ___ FAR = ___ frequency bias = ___ / ___ = ___
>
> On the pooled columns — fitted rung (1–3) that beats nino-sign on both precision and recall: ___ Rung whose `precision_pooled` clears the always-wet bar (base rate ___): ___ Does `precision_fold` agree for that rung? ___ — because:
>
> Recall 1.0 at base-rate precision means (and why no rung can beat it on recall):

---

### Entry 8 — The trial register (Exercise 8) · O1 *refine* · O2

**Prompt.** The register arrives with four rows: the ladder's three rungs and the example trial. Choose and run your own refinement trials in the notebook's *Your trials* cell, one `record_trial` call at a time — keep every call in the cell, add the next below the last and re-run it (calls already in the register are skipped), so that — once you have set `FINAL = True` at the top of that cell after your last trial — Restart & Run All reproduces every trial before 3g and the register you paste here is the one the executed notebook shows. Up to two trials of your own: the cap is six tested rows *in all*, and four are already spent by the three rungs and the example trial. **Before each call**, write its three lines below: what changes (one thing), why, and which metric you expect to move. Then paste the register. For each row, one line: what changed, which metric (if any) moved outside the run-to-run spread — the corrected standard error the paired test divides the gap by (the `_t` and `_p` columns), with the per-fold `_sd` beside it for scale, not as the test — and whether the other metric was given back; the 3f rule promotes only a measured gain in one with no mean loss in the other. Each `p` is per test and uncorrected: the cap of six bounds how many looks are taken, it does not correct them. The register prints each `p` to three significant figures (`0.0312`, `0.00384`, `0.000013`), so it can be placed against the Bonferroni line, printed to four decimals. The Bonferroni line printed directly above the register table (0.05 ÷ the looks taken, two per tested row) is the corrected threshold: for a promoted row, say whether its `p` clears that line or sits between it and 0.05 — the latter is a promotion under the uncorrected rule only. Trials you considered but did not run are listed too, with the reason. Unidata's 3f is the shape: "it's now your turn … you'll create **new trials**".

**Evidence to paste.** Your trials' three lines (written first); the register DataFrame as the `FINAL = True` run prints it — the Restart & Run All that ran 3g, the register the executed notebook shows; not an earlier `FINAL = False` run's — (trial · candidate · incumbent · what_changed · recall_cand · recall_inc · recall_gap · recall_sd · recall_t · recall_p · the same six for precision · verdict) the `first incumbent:` / `incumbent after N trials:` lines and the `Bonferroni line after N tested row(s)` line, all printed above the register table.

**Response.**

> My trial(s), before running — change: ___ · why: ___ · expected to move: ___
>
> Register (paste the DataFrame; this scaffold is its shape):
>
> | trial | candidate | incumbent | what changed | recall gap · t · p | precision gap · t · p | verdict | my reading (which metric moved outside the corrected spread; the other given back?) |
> |---|---|---|---|---|---|---|---|
> | 1 | logistic | nino-sign | | | | | |
> | 2 | tree | (the incumbent at that row) | | | | | |
> | 3 | boosting | (the incumbent at that row) | | | | | |
> | 4 | logistic@base-rate | (the incumbent at that row) | | | | | |
> | 5 | (yours) | (the incumbent at that row) | | | | | |
> | 6 | (yours) | (the incumbent at that row) | | | | | |
>
> `incumbent after N trials:` line and the `Bonferroni line after N tested row(s)` line, pasted; promoted rows against that line:
>
> Considered but not run:

---

### Entry 9 — The paired test on a pair you choose (Exercise 9) · O1 *refine* · O2 *justify*

**Prompt.** Run the Exercise 9 scaffold cell in the notebook (below Exercise 9). It takes two names from `trial_scores` and one metric — by default the last promotion's pair, the final incumbent and the rung it replaced, on the metric that promotion was earned on (read from the verdict string; the cell's first printed line names the pair and the metric); if nothing was promoted (a rebuilt cache, or your own trials on another season), the scaffold takes the register's last row, the Niño-sign rule against the last candidate, on precision — so name the pair and metric you actually tested — sets `n_train` and `n_test` exactly as `compare` does, and makes one call to `compute_corrected_ttest`; set `HAND_METRIC` to the other metric to test the promotion's no-loss half, or `PAIR_A`, `PAIR_B` for another pair. The test is right-tailed (3f): a `PAIR_B` that *lost* prints `p` → 1 by design, so the no-loss half — and any pair whose gap is negative — is read on the gap's sign and the scaffold's LOSS line (`a measured LOSS for … corrected standard errors (left-tail p = …)`), never on `p` alone; the register's `verdict()` reads the sign the same way, and a loss of many corrected standard errors reported as "not a measured difference" is a false statement, not a reading. State whether the leader's lead is a measured gain (`p` < 0.05), a measured loss (the LOSS line), or not a measured difference (neither tail under 0.05), and for a gain whether that `p` clears the Bonferroni line the scaffold prints under its verdict or sits between it and 0.05 — the scaffold prints `p` to three significant figures (`0.0312`, `0.00384`, `1.25e-05`), so the comparison with the four-decimal line is a numeric one at any size — the latter is a measured difference under the uncorrected rule only, as Entry 8 read the promoted row, and is how this entry and Entry 10's interval lines can both be right. If the lead is not a measured difference, say which model you keep and on what grounds (cost, reproducibility, explanation).

**Evidence to paste.** The scaffold cell's five printed lines: the `default pair:` line (the register row, its verdict and the metric it chose); the pair and metric with `n_train`, `n_test`; the two means, the gap, the per-fold SD of the gap and the corrected standard error; `t`, `p` with the verdict; and the `Bonferroni line after N tested row(s)` line under it.

**Response.**

> | pair (candidate − incumbent) · metric | n_train · n_test | gap · corrected SE | t · p (as printed) | p < 0.05? | Bonferroni line (p <) | clears it / between it and 0.05 |
> |---|---|---|---|---|---|---|
> | | | | | | | |
>
> Model kept, and why:

---

### Entry 10 — The one test (Exercise 10) · O1 *test* · O2

**Prompt.** Paste the test block from 3g, starting at its `test run #` line (3g runs only after `FINAL = True` is set at the top of the *Your trials* cell; a cell that printed `test not run` has not tested anything): the final model (the register's last incumbent — the Niño-sign rule if nothing was promoted), the number of test members and winters, precision and recall for *wet*, accuracy and AP, the confusion matrix printed as text (the figure under the block draws the same matrix), the three rung-0 lines (the two bars and the physical prior) scored on the same test members, and the two lines that read precision and AP against the base rate of the same winters with an interval over the test members. Is the tested model's skill distinguishable from none? Did the validation numbers predict the test numbers? Name one reason they might not.

**Evidence to paste.** The printed test block from its `test run #` line, the three rung-0 lines (each carries precision, recall and accuracy only — 3g prints no AP for rung 0, so that cell of the scaffold is dashed) and the two `− base rate` interval lines. The rubric reads the `test run #` line: `#1` is the only passing value (the cell counts its own runs in a session — it cannot see a kernel restart; the evidence that the test came after the register is this handbook's order, Entries 8 and 9 written before 10, and `FINAL = True` in the executed notebook's *Your trials* cell). Set `FINAL = True` after your last 3f trial, then Restart & Run All: that is the one legitimate run of the test cell, and it leaves the counter at 1 — a run-all with `FINAL = False` scores nothing.

**Response.**

> | test: `___` on ___ members / ___ winters (base rate ___) | predicted dry | predicted wet |
> |---|---|---|
> | **true dry** | | |
> | **true wet** | | |
>
> | on the same test members | precision (wet) | recall (wet) | accuracy | AP |
> |---|---|---|---|---|
> | final model: ___ | | | | |
> | always-dry | | | | — |
> | always-wet | | | | — |
> | nino-sign | | | | — |
>
> | − base rate (same winters) | gap | 95% interval over test members | reading printed |
> |---|---|---|---|
> | precision | | | |
> | AP | | | |
>
> Skill distinguishable from none (precision; AP): yes / no — the interval lines say:
>
> Validation predicted test: yes / partly / no — because:

---

### Entry 11 — The final decision (Exercise 11) · O2 *justify*

**Prompt.** Write the recommendation to the person you named in Entry 2, in three labelled parts, in the voice Unidata's closing step asks for: "a final recommendation … on whether you believe this model provides sufficient skill for the needs of the situation". (1) *Does it deliver?* — the test numbers from Entry 10 against the base rate and the three rung-0 lines (the two bars and the physical prior) on the same members, read with the two interval lines: is the tested model's skill distinguishable from none? Did the validation rows in Entry 6 predict the test? (2) *If nothing showed skill, say so plainly.* When no model you tested beats the base rate measurably, the simplest answer is the base rate itself, and the outlook stays at "equal chances". Then keep what the run measured apart from what it cannot attribute. Three explanations fit a null: a signal absent from this box and season in CESM2 LENS; a signal your models, input or threshold did not reach (the AP column of Entry 6 and Entry 7's table against the base rate); or a sample too small to see it. Say which of your numbers bears on each, and what would be needed to tell them apart. The transition zone of the ENSO dipole and the late-winter El Niño signal are hypotheses for a weak signal, not results this run tested. If a model did show skill, say which, on what tested gain, and at what cost. (3) *The one further change most likely to move the result* — season, box, an input from 3b (a new box mean in `derive_member_winters` plus the cache rebuild), a threshold trial on another rung (minutes, no rebuild), keeping one half of the ensemble (the alternative Entry 4 deferred: a filter on the table, no rebuild, half the members and half the statistical power), or the snowfall variant of the target — and what it would cost to test: which parameter or derivation changes, whether the cache must be rebuilt, and how long that takes. Every claim points at a number already in Entries 6–10; "rarely is there a single correct answer", so the justification is what is scored.

**Evidence to paste.** The lines each claim points at, quoted from your own entries: the register's final `incumbent after N trials:` line and the verdict of its last row (Entry 8/9); the 3g `test run #1` block's two `− base rate` interval lines and its three rung-0 lines (Entry 10); the 3d rung-0 rows (Entry 6). A claim is scored on the line quoted under it.

**Response.**

> (1) Does it deliver — quoting Entry 10's two interval lines (`<model>: precision − base rate … → …` and `<model>: AP − base rate … → …`) and its rung-0 lines; skill is claimed only when BOTH lines read `above no skill` — one `above` and the other `NOT distinguishable` is "equal chances", said so:
>
> (2) Skill or no skill, said plainly — quoting the register's `incumbent after N trials:` line and the last row's verdict (Entry 8/9), and the 3d rung-0 rows (Entry 6): what the run measured, and what it cannot attribute:
>
> (3) The next change and its cost — the Entry 4 or Entry 7 line it rests on, quoted:

---

## Self-score

Before handing the handbook to the trainer, score each entry with the Rubric ([`course/rubric.md`](https://github.com/glenbuchanan/ncar-ml-readiness-course/blob/main/course/rubric.md)) and record it here. The trainer's column is filled by the trainer.

| Entry | Self (Not yet / Developing / Proficient) | Trainer | Note |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |

---
---
