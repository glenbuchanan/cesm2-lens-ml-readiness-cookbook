# CESM2 LENS ML Readiness Cookbook

<img src="thumbnails/thumbnail.png" alt="Wet-winter share by tenth of autumn Niño 3.4, training and test members, against the base rate: both lower at the extremes; the training peak in the third tenth does not repeat" width="300"/>

[![Launch this cookbook on the Project Pythia Binder](https://binder.projectpythia.org/badge_logo.svg)](https://binder.projectpythia.org/v2/gh/glenbuchanan/cesm2-lens-ml-readiness-cookbook/main?labpath=notebooks)

This Project Pythia Cookbook covers _supervised machine learning readiness on a climate-model large ensemble_: one classification
question — **does the autumn ENSO state predict a wet Front Range winter?** — answered on NSF NCAR's own
[CESM2 Large Ensemble](https://registry.opendata.aws/ncar-cesm2-lens/) (the AWS Open Data copy; no account, free egress),
walked along the analysis spine of NSF Unidata's
[*Supervised Machine Learning Readiness*](https://github.com/Unidata/supervised-ml-readiness), in seven short chapters.

This cookbook is for Earth system scientists, engineers and software staff who are new to machine learning, or who use it
without training. It assumes Python at the level of the [Xarray tutorial](https://tutorial.xarray.dev/) and no ML.
Time to learn: about 2 hours 35 minutes for the seven chapters (the sum of the times each chapter's Prerequisites states). Each chapter runs on its own in under a minute from the committed cache; a
re-derivation of the table from S3 is optional (the one full run took 33 minutes on a fast link; on home broadband expect an hour or more).

## Motivation

The existing [CESM LENS on AWS Cookbook](https://projectpythia.org/cesm-lens-aws-cookbook/) covers reading and analysing the
40-member CESM1 Large Ensemble with Xarray and Dask; the [scikit-learn Argo](https://projectpythia.org/sklearn-argo-cookbook/),
[Landsat ML](https://projectpythia.org/landsat-ml-cookbook/) and [xbatcher ML](https://projectpythia.org/xbatcher-ML-1-cookbook/)
cookbooks apply machine learning to ocean profiles, satellite imagery and CNN training batches, and the
[EO Data Science](https://projectpythia.org/eo-datascience-cookbook/) cookbook includes a supervised classifier on Sentinel-2 imagery. None yet walks a newcomer through
a supervised classifier on a **large ensemble**, where the members themselves are the unit the split has to respect. This cookbook teaches the whole spine on that data and changes
exactly one of Unidata's steps on purpose: the split is **by ensemble member**. Two reasons hold it there. The question is about
members the model has never seen, so the score has to come from unseen members. And the forced-response curve and the wet
threshold are fit on training members only, refit inside every validation fold, so no held-out member helps fit the transforms
that label it. What a member-blind split, or a transform fit before the split, would do to the score is not assumed: chapter 3
scores the same model all three ways and prints the gaps against their per-fold spread. A refinement trial is promoted on a paired test
with a corrected standard error (uncorrected `p < 0.05`), with the Bonferroni line printed beside it. The held-out test sits behind a `FINAL` switch that ships `True`, so the published page runs
it on every build and shows it; once you change anything after reading it, keeping it a one-time test is your discipline, helped by a run
counter. It is read against the base rate of the same winters. `FINAL = True` means every build reruns the test.

By the end you will be able to:

- explain the steps of a supervised analysis (framing → data → split → train → validate → refine → test → justify) and name the failure each step guards against;
- build and validate a classification model on a climate-model ensemble, reporting precision, recall and a confusion matrix;
- compare refinement trials as a paired test across runs, so a gap inside the run-to-run spread is not read as a difference;
- justify the chosen model honestly on measured performance and cost, including when nothing beats the base rate.

## Authors

[Glen Buchanan](https://github.com/glenbuchanan)

### Contributors

See the [contributors graph](https://github.com/glenbuchanan/cesm2-lens-ml-readiness-cookbook/graphs/contributors).

The analysis spine follows NSF Unidata's *Supervised Machine Learning Readiness* (Apache-2.0, NSF award #2319979),
authored by NSF Unidata with Metropolitan State University of Denver. The paired-test functions in `notebooks/lensml.py` adapt
scikit-learn's "Statistical comparison of models using grid search" example (BSD-3-Clause), credited in place.

## Structure

Seven chapters, each a Project Pythia notebook that runs on its own in a fresh kernel. Shared code — the parameters, the transforms, the
ladder, the scoring and the paired test — lives in one file, `notebooks/lensml.py`, which every chapter imports and which prints its own
source where a chapter introduces a function.

| Chapter | What it does |
|---|---|
| [01 Finding the data](notebooks/01-find-the-data.ipynb) | the question as a decision; Unidata's three framing questions; the `intake-esm` catalogue on AWS (offline-tolerant); the Metadata Document; the committed cache, asserted, and how a rebuild works |
| [02 The target and the forced response](notebooks/02-target-and-forced-response.ipynb) | the test members named first; the forced response removed on training members only, with a check that can fail; one pooled "wet" threshold; base rates per forcing half |
| [03 Splitting by ensemble member](notebooks/03-split-by-member.ipynb) | member-grouped folds; within- and between-member dependence, measured; one model scored three ways to measure the two leaks |
| [04 Baselines and the ladder](notebooks/04-baselines-and-the-ladder.ipynb) | three rules that fit nothing; logistic regression, a decision tree, gradient boosting; confusion matrices, precision, recall and AP against the base rate; the decision threshold |
| [05 Refinement trials](notebooks/05-refinement-trials.ipynb) | the register; the corrected paired t-test; the Bonferroni line; the trial budget; two threshold trials and one of your own |
| [06 The held-out test, and the justification](notebooks/06-test-and-justify.ipynb) | the test of the register's final incumbent (the published page runs it), read against no skill and the cost of a false alarm; the justification; the result's shape; the limits |
| [07 An out-of-period test, under a protocol fixed before the run](notebooks/07-out-of-period-test.ipynb) | the same members' SSP3-7.0 winters (2016–2100), read only after a protocol (self-attested; no external timestamp) fixed the model, comparators, outcomes and decision rule; transforms refitted on training members only; one scored run |

Chapters 1–6 carry short *Try it* prompts in place; none needs anything outside this repository. Two support files sit beside the
chapters: `cookbook-render.mjs`, a MyST plugin that copies each figure cell's `metadata.alt` into the rendered image's alt text (without
it the page would read "Figure size 990x418"), and `thumbnails/make_thumbnail.py`, which redraws the gallery thumbnail from the committed
cache with the same transforms the chapters use.

### Glossary

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

**What to expect** (terms in the glossary above). The signal is weak. The straight-line relation between autumn Niño 3.4 and a wet
winter is indistinguishable from zero. Over the training members the wet rate varies between bins of the input (the extremes drier than the
middle); on the test members the bins alone are not distinguishable from chance, and the training peak does not repeat. The final model, gradient boosting at the base-rate threshold, was chosen after its validation score had been
seen; on the test members its precision edge over the base rate is small and its false alarms outnumber its hits. **The test is not
blind**: during development (2026-10-20) the same 20 test members were scored with two candidate final models — logistic at the
base-rate threshold showed no detectable edge, and boosting at the base-rate threshold gave the result chapter 6 shows — so the edge is
provisional, and a fresh test would need data no choice was made on; chapter 7 uses the same members' SSP3-7.0 winters. **Chapter 7 runs that test** on the same members' SSP3-7.0 winters (2016–2100), under a protocol fixed before the scored run (self-attested; no external timestamp, shipped as `notebooks/data/ssp370-test-protocol.md`): **the edge does not replicate under the rule fixed before the run**. The final model's precision interval includes zero and also chapter 6's estimate, so the test cannot separate no edge from an edge of that size; it gives no support for leaning an outlook wet. Chapter 6 prints each
of these, and its limits: a model-world study, on a ~1° grid, of total precipitation, in DJF, for a box that crosses the Divide.

The cookbook stands on its own. As optional further material, a separate repository,
[ncar-ml-readiness-course](https://github.com/glenbuchanan/ncar-ml-readiness-course), keeps the same data and split and goes on to a small
neural network; nothing in it is needed here.

### Data

`notebooks/data/member_winters.csv` is the **full** member-winter table (50 members per forcing half, every member of the
ensemble), derived by the cookbook's own code and committed, so the book builds without touching S3 for the heavy step.
Every chapter asserts the cache header against the parameters in `notebooks/lensml.py` before reading it; chapter 1 re-derives the table
from S3 when `REBUILD = True` (the one full derivation took 33 minutes on a fast link; on home broadband expect an hour or more). See
`notebooks/data/README.md`.

`notebooks/data/member_winters_ssp370.csv` holds the same members' SSP3-7.0 winters, 2016–2100 (accessed 2026-10-29), read only by
chapter 7; `notebooks/data/ssp370-test-protocol.md` is the protocol chapter 7 follows, verbatim.

## Data citation

Community Earth System Model v2 Large Ensemble (CESM2 LENS) was accessed on 2026-10-20 from https://registry.opendata.aws/ncar-cesm2-lens.
Two tables are read: `member_winters.csv` (historical experiment, 1900–2014, accessed 2026-10-20) and `member_winters_ssp370.csv`
(SSP3-7.0 experiment, the ssp370 stores `s3://ncar-cesm2-lens/atm/monthly/cesm2LE-ssp370-{cmip6,smbb}-{PRECC,PRECL,TS}.zarr`, winters
2016–2100, accessed 2026-10-29). Both come from the same cloud-optimized dataset, cited below; the SSP3-7.0 access line is:
Community Earth System Model v2 Large Ensemble (CESM2 LENS) was accessed on 2026-10-29 from https://registry.opendata.aws/ncar-cesm2-lens.
The citation NSF NCAR's Data Citation section requires, and which the UCAR Data Terms of Use (CC BY 4.0) bind users to, is both
the AWS subset's DOI and the paper:

- Grover, M., & Banihirwe, A. (2021). *NCAR CESM2 Large Ensemble Cloud-Optimized Dataset*. NSF NCAR CISL. https://doi.org/10.26024/y48t-q717
- Rodgers, K. B. et al. (2021). Ubiquity of human-induced changes in climate variability. *Earth System Dynamics* 12, 1393–1411. https://doi.org/10.5194/esd-12-1393-2021

Results based on the CESM2 Large Ensemble acknowledge the CESM2 Large Ensemble Community Project and supercomputing resources
provided by the IBS Center for Climate Physics in South Korea. The same line sits under every figure the chapters draw.

## Running the Notebooks

You can either run the notebooks in the Cookbook using [Binder](https://binder.projectpythia.org/) or on your local machine.

### Running on Binder

The simplest way to interact with a Jupyter Notebook is through
[Binder](https://binder.projectpythia.org/), which enables "one click"
execution in the cloud. Simply navigate your mouse to
the top right corner of the book chapter you are viewing and click
on the rocket ship icon (see screenshots [here](https://foundations.projectpythia.org/preamble/how-to-use/#running-pythia-foundations-examples)),
and a text box will appear. Type or paste the Pythia Binder link
(`https://binder.projectpythia.org`) and click "Launch".
After a few moments you should be presented with a
notebook that you can interact with. You’ll be able to execute code
and even change the example programs. At first the code cells
have no output, until you execute them by pressing
{kbd}`Shift`\+{kbd}`Enter`. Complete details on how to interact with
a live Jupyter notebook are described in the Pythia Foundations chapter [Getting Started with
Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter).

Note, not all Cookbook chapters are executable. If you do not see
the rocket ship icon, such as on this page, you are not viewing an
executable book chapter.

### Running on Your Own Machine

If you are interested in running this material locally on your computer, you will need to follow this workflow:

1. Clone the `https://github.com/glenbuchanan/cesm2-lens-ml-readiness-cookbook` repository:

   ```bash
    git clone https://github.com/glenbuchanan/cesm2-lens-ml-readiness-cookbook.git
   ```

1. Move into the `cesm2-lens-ml-readiness-cookbook` directory
   ```bash
   cd cesm2-lens-ml-readiness-cookbook
   ```
1. Create and activate your conda environment from the `environment.yml` file
   ```bash
   conda env create -f environment.yml
   conda activate cesm2-lens-ml-readiness-cookbook-dev
   ```
1. Move into the `notebooks` directory and start up Jupyterlab
   ```bash
   cd notebooks/
   jupyter lab
   ```

### Running on the NSF NCAR JupyterHub

On the NSF NCAR JupyterHub (Casper/Derecho) create the same environment from `environment.yml` and pick the kernel it registers;
the chapters read the AWS copy of the ensemble, so no GLADE path is needed.

## Licence, provenance and review status

- Code is Apache-2.0; non-code content is CC BY 4.0 (the Project Pythia split). Data are under the UCAR Data Terms of Use (CC BY 4.0).
- Nothing in this cookbook is export-controlled: open data, open code.
- Funding: self-funded. This cookbook has not been reviewed or sponsored by NSF NCAR or NSF Unidata; it is offered to the
  Project Pythia community for its own review.
- Drafting: Drafted with AI assistance (Anthropic's Claude) from cited sources, under the author's direction, and reviewed by the author in several full read-throughs. Text and code were checked against those sources and the notebooks' own outputs by automated tests and independent AI reviewers; no human subject-matter expert has reviewed it yet.
- How it was checked: each chapter's *Resources and references* table lists its sources (13 distinct entries across the seven
  chapters; the scikit-learn User Guide entries name the sections used); automated tests execute every chapter from the cache and check its printed numbers; independent AI
  reviewers read the cookbook, and a finding was acted on only after it was reproduced.
