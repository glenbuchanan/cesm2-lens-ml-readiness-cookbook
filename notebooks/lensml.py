"""lensml: the shared code of the CESM2 LENS ML Readiness Cookbook.

Every chapter imports this file, so each chapter runs on its own, from the
committed cache, in a fresh kernel. One home for every value: the
parameters (the design decisions), the figure citation, the cache reader,
the test draw, the two transforms, the folds, the scoring function, the
ladder, the paired test and the register.

Nothing here holds state between calls. ``setup`` returns a ``State``;
the functions take what they need as arguments, and a ``Register`` belongs
to the chapter that creates it. A chapter that introduces a function shows
its source with ``show``.

Code: Apache-2.0. The paired-test functions are adapted from scikit-learn
(BSD-3-Clause), credited in place.
"""

import inspect
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
from scipy.stats import t as t_dist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Figure surfaces: warm canvas, muted axes, one gold ramp. Dry and wet are
# teal circles and magenta triangles: colour AND shape.
plt.rcParams['figure.dpi'] = 110
plt.rcParams.update(
    {
        'figure.facecolor': '#FBFAF6',
        'axes.facecolor': '#F5F3ED',
        'axes.edgecolor': '#E4E0D6',
        'axes.labelcolor': '#13161A',
        'text.color': '#13161A',
        'xtick.color': '#6A7077',
        'ytick.color': '#6A7077',
        'grid.color': '#E4E0D6',
        'legend.facecolor': '#F6F1E5',
        'legend.edgecolor': '#E4E0D6',
        'savefig.facecolor': '#FBFAF6',
    }
)
GOLD_CMAP = LinearSegmentedColormap.from_list(
    'gold', ['#FBFAF6', '#EAD27E', '#C9A23A', '#8F7022']
)

# ==========================================================================
# The parameters. Every value is a design decision; change one only as a
# refinement trial you record, never silently.
# ==========================================================================
RNG_SEED = 0  # declared once; every random_state derives from it

# The day the COMMITTED data/member_winters.csv was read from S3. The cache
# reader asserts it equals the header's access= field; after a rebuild, set
# it to the new header's date.
ACCESS_DATE = '2026-10-20'

# Under every LENS figure (the access line from the registry page's "How to
# cite"; the two DOIs from the dataset's Data Citation page): the credit,
# the two DOIs the UCAR Data Terms bind users to, then the access line.
LENS_TEMPLATE = (
    'Figure © 2026 Glen Buchanan, CC BY 4.0 · Data: CESM2 Large Ensemble '
    '(Rodgers et al. 2021, doi:10.5194/esd-12-1393-2021; Grover & '
    'Banihirwe 2021, doi:10.26024/y48t-q717)\n'
    'Community Earth System Model v2 Large Ensemble (CESM2 LENS) was '
    'accessed on ACCESS_DATE from https://registry.opendata.aws/'
    'ncar-cesm2-lens.'
)

# --- the target -----------------------------------------------------------
# The Front Range around Denver on the 0-360 grid: two rows x three 1.25°
# columns; the west column lies mostly west of the Divide.
FRONT_RANGE = dict(lat=slice(38.5, 40.5), lon=slice(253.75, 256.25))
# "DJF" (Dec of the prior year, Jan, Feb) or "FMA" (Feb-Apr). DJF is the
# committed cache; FMA needs a rebuild (chapter 1).
SEASON = 'DJF'
# "wet" = detrended precipitation above this quantile of the TRAINING
# members, pooled: one threshold for every member (chapter 2).
WET_QUANTILE = 2 / 3

# --- the input ------------------------------------------------------------
# 5°S-5°N, 170°-120°W: the Rodgers et al. 2021 box, on atmospheric TS.
NINO34 = dict(lat=slice(-5, 5), lon=slice(190, 240))
AUTUMN_MONTHS = [9, 10, 11]  # SON of the year before the winter

# --- the window and the forced response -----------------------------------
# Rodgers et al. 2021 §2.2: members "should not be considered ...
# independent" soon after 1850, and Rodgers analyse internal variability
# "after 1960". The 1900 start is this cookbook's choice; chapter 3's
# sibling check is what supports it, not Rodgers.
WINDOW = (1900, 2014)
FORCED_DEG = 3  # degree of the forced-response polynomial in year

# --- the split ------------------------------------------------------------
TEST_FRACTION = 0.2  # of EACH half's members, held out for the one test
N_FOLDS_MAX = 5  # member-grouped folds: min(N_FOLDS_MAX, training members)
N_REPEATS = 10  # fold re-draws per trial; 1 when every re-draw is the same

# --- the ladder's settings ------------------------------------------------
TREE_SETTINGS = dict(max_depth=3, min_samples_leaf=20)
BOOSTING_SETTINGS = dict(n_estimators=100, max_depth=2, learning_rate=0.05)

FEATURES = ['nino34_anom']  # the one input
ANOMALIES = {
    'nino34_anom': 'nino34_son_K',
    'prect_anom': 'prect_winter_mm_day',
}
YEARS = np.arange(WINDOW[0], WINDOW[1] + 1)


# ==========================================================================
# Showing code and crediting figures
# ==========================================================================
def show(*fns):
    """Display the source of functions from this file."""
    from IPython.display import Code, display

    src = '\n\n'.join(inspect.getsource(f) for f in fns)
    display(Code(src, language='python'))


def cite(fig, access_date=ACCESS_DATE):
    """Put the required CESM2 LENS citation line under a figure."""
    text = LENS_TEMPLATE.replace('ACCESS_DATE', access_date)
    fig.text(0.01, -0.02, text, fontsize=7, ha='left', va='top')


def credit_own(fig, what):
    """Under a diagram that draws no CESM2 LENS data: its own credit."""
    fig.text(
        0.01,
        -0.02,
        f"© 2026 Glen Buchanan · figure CC BY 4.0 · {what}",
        fontsize=7,
        ha='left',
        va='top',
    )


def show_fig(fig, alt):
    """Display fig as the inline backend would, with `alt` as its text/plain:
    the book theme renders an image output's text/plain as the <img alt>, and
    a bare matplotlib output would carry "<Figure size …>" there instead."""
    from IPython.core.pylabtools import print_figure
    from IPython.display import display

    png = print_figure(fig, 'png')
    plt.close(fig)
    display({'image/png': png, 'text/plain': alt}, raw=True)


# ==========================================================================
# The member-winter table: derivation (one-time, from S3) and the reader
# ==========================================================================
CACHE = Path('data') / 'member_winters.csv'  # relative to the notebooks
BUCKET = 's3://ncar-cesm2-lens/atm/monthly'


def open_var(var, variant, experiment='historical'):
    """One variable of one forcing half, lazily, from anonymous S3."""
    return xr.open_zarr(
        f"{BUCKET}/cesm2LE-{experiment}-{variant}-{var}.zarr",
        consolidated=True,
        storage_options={'anon': True},
    )[var]


def box_mean(da, box):
    """Area-weighted mean over a lat/lon box."""
    sub = da.sel(**box)
    return sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(('lat', 'lon'))


def derive_member_winters(
    variant, members, experiment='historical', window=WINDOW
):
    """One row per member-winter: raw SON Niño 3.4 (K) and the season-mean
    Front Range precipitation (mm/day), for winters in `window`."""
    ts = open_var('TS', variant, experiment).sel(member_id=members)
    pr = open_var('PRECC', variant, experiment) + open_var(
        'PRECL', variant, experiment
    )
    pr = pr.sel(member_id=members) * 8.64e7
    nino = box_mean(ts, NINO34).load()
    prfr = box_mean(pr, FRONT_RANGE).load()
    yr, mo = nino['time'].dt.year.values, nino['time'].dt.month.values
    rows = []
    for m in members:
        n, p = nino.sel(member_id=m).values, prfr.sel(member_id=m).values
        for y in range(window[0], window[1] + 1):
            son = n[(yr == y - 1) & np.isin(mo, AUTUMN_MONTHS)]
            if SEASON == 'DJF':
                prior_dec = (yr == y - 1) & (mo == 12)
                win = p[prior_dec | ((yr == y) & np.isin(mo, [1, 2]))]
            else:  # "FMA"
                win = p[(yr == y) & np.isin(mo, [2, 3, 4])]
            if len(son) == 3 and len(win) == 3:
                rows.append(
                    dict(
                        member_id=m,
                        forcing_variant=variant,
                        winter=y,
                        nino34_son_K=float(son.mean()),
                        prect_winter_mm_day=float(win.mean()),
                    )
                )
    return pd.DataFrame(rows)


# The header records the parameters the table was derived with; the reader
# checks the same string.
PARAMS_RECORD = (
    f"SEASON={SEASON} WINDOW={WINDOW} FRONT_RANGE={FRONT_RANGE} "
    f"NINO34={NINO34} AUTUMN_MONTHS={AUTUMN_MONTHS}"
)
# A header written before AUTUMN_MONTHS was recorded (the committed cache)
# was derived with SON, the only autumn the code had then.
LEGACY_AUTUMN = 'AUTUMN_MONTHS=[9, 10, 11]'


def cache_matches_parameters(header_line):
    """Raise a readable error if the header's parameters differ."""
    m = re.search(r'(SEASON=.*?)\s+n_members_per_half=', header_line)
    recorded = m.group(1) if m else '(no parameter record in the header)'
    if m and 'AUTUMN_MONTHS=' not in recorded:
        # The committed header predates the field: the file's autumn is
        # assumed (SON), not read from the header.
        recorded += ' ' + LEGACY_AUTUMN
        print(
            'cache header has no AUTUMN_MONTHS field (it predates the '
            f'field): taken as {LEGACY_AUTUMN}, the only autumn the '
            "derivation code had: the file's autumn is assumed, not read "
            'from the header. A rebuild records it.'
        )
    if recorded != PARAMS_RECORD:
        raise AssertionError(
            'data/member_winters.csv was derived with different parameters '
            'than lensml.py now declares.\n'
            f"  cache header : {recorded}\n  lensml.py    : {PARAMS_RECORD}\n"
            '  Re-derive it with load_cache(rebuild=True).'
        )


def cache_access_date(header_line):
    """The header's access= date."""
    m = re.search(r'\baccess=(\d{4}-\d{2}-\d{2})', header_line)
    return m.group(1) if m else '(no access= field in the header)'


def load_cache(rebuild=False, n_members_per_half=50):
    """Read the member-winter table, asserting its header against the
    parameters. With rebuild=True (or no cache) re-derive it from S3
    first; 50 members per half is the full ensemble. A rebuild is a new
    access: it stamps today's date into the header.
    Returns (table, cache_header, access_date)."""
    access_date = ACCESS_DATE
    if rebuild or not CACHE.exists():
        access_date = date.today().isoformat()
        print(f"deriving the table: access date {access_date}")
        CACHE.parent.mkdir(exist_ok=True)
        parts = []
        for variant in ['cmip6', 'smbb']:
            ids = open_var('TS', variant).member_id.values
            members = list(ids[:n_members_per_half])
            parts.append(derive_member_winters(variant, members))
        table = pd.concat(parts, ignore_index=True)
        header = (
            f"# derived by lensml.py: {PARAMS_RECORD} "
            f"n_members_per_half={n_members_per_half} "
            f"RNG_SEED={RNG_SEED} access={access_date}\n"
        )
        with open(CACHE, 'w') as f:
            f.write(header)
            table.to_csv(f, index=False)

    with open(CACHE) as f:
        cache_header = f.readline().strip()
    cache_matches_parameters(cache_header)
    assert cache_access_date(cache_header) == access_date, (
        f"the cache header says access={cache_access_date(cache_header)} "
        f"but lensml.py says ACCESS_DATE = {access_date!r}: set ACCESS_DATE "
        'to the header date, or rebuild for a new access'
    )
    table = pd.read_csv(CACHE, comment='#')
    return table, cache_header, access_date


# ==========================================================================
# The test draw and the two transforms
# ==========================================================================
# ==========================================================================
# The out-of-period table (chapter 7): the same members, boxes, season and
# autumn under SSP3-7.0. DJF 2015 would need Dec 2014 from the historical
# store, so the window starts at 2016.
# ==========================================================================
FUTURE_EXPERIMENT = 'ssp370'
FUTURE_WINDOW = (2016, 2100)
FUTURE_YEARS = np.arange(FUTURE_WINDOW[0], FUTURE_WINDOW[1] + 1)
FUTURE_CACHE = Path('data') / 'member_winters_ssp370.csv'
FUTURE_ACCESS_DATE = (
    '2026-10-29'  # the day the committed ssp370 table was read
)
FUTURE_RECORD = (
    f'experiment={FUTURE_EXPERIMENT} SEASON={SEASON} WINDOW={FUTURE_WINDOW} '
    f'FRONT_RANGE={FRONT_RANGE} NINO34={NINO34} '
    f'AUTUMN_MONTHS={AUTUMN_MONTHS}'
)


def derive_future_cache(historical_table):
    """Derive the SSP3-7.0 table for exactly the members of the historical
    table, per forcing half, with derive_member_winters, and write it with a
    header recording the experiment, window, parameters and access date."""
    access_date = date.today().isoformat()
    parts = []
    for variant, half in historical_table.groupby('forcing_variant'):
        members = list(dict.fromkeys(half['member_id']))
        parts.append(
            derive_member_winters(
                variant, members, FUTURE_EXPERIMENT, FUTURE_WINDOW
            )
        )
    table = pd.concat(parts, ignore_index=True)
    with open(FUTURE_CACHE, 'w') as f:
        f.write(
            f'# derived by lensml.py: {FUTURE_RECORD} '
            f'n_members={table.member_id.nunique()} RNG_SEED={RNG_SEED} '
            f'access={access_date}\n'
        )
        table.to_csv(f, index=False)
    return access_date


def load_future_cache():
    """Read the SSP3-7.0 table, asserting that its header records
    FUTURE_RECORD and an access= date equal to FUTURE_ACCESS_DATE.
    Returns (table, header)."""
    with open(FUTURE_CACHE) as f:
        header = f.readline().strip()
    assert FUTURE_RECORD in header, (
        f'{FUTURE_CACHE} was derived with other parameters:\n  {header}\n'
        f'  expected: {FUTURE_RECORD}'
    )
    assert cache_access_date(header) == FUTURE_ACCESS_DATE, (
        f'the header says access={cache_access_date(header)} but lensml.py '
        f'says FUTURE_ACCESS_DATE = {FUTURE_ACCESS_DATE!r}'
    )
    return pd.read_csv(FUTURE_CACHE, comment='#'), header


def draw_test_members(table):
    """Name the test members: TEST_FRACTION of each half, by member, from a
    RandomState created here from RNG_SEED (a re-run draws the same)."""
    rng = np.random.RandomState(RNG_SEED)
    test_members = []
    for _, half in table.groupby('forcing_variant'):
        hold = GroupShuffleSplit(
            n_splits=1, test_size=TEST_FRACTION, random_state=rng
        )
        _, held = next(hold.split(half, groups=half['member_id']))
        test_members += sorted(half.iloc[held]['member_id'].unique())
    return test_members


def fit_transforms(rows, years=YEARS):
    """Fit on `rows` ONLY: per series, a degree-FORCED_DEG polynomial in
    year through the ensemble mean of `rows`; and the wet threshold, the
    WET_QUANTILE of the pooled detrended precipitation of `rows`."""
    curves = {}
    for anom, raw in ANOMALIES.items():
        ens_mean = rows.groupby('winter')[raw].mean().reindex(years)
        coef = np.polyfit(years, ens_mean.values, FORCED_DEG)
        curves[anom] = pd.Series(np.polyval(coef, years), index=years)
    curve = curves['prect_anom'].reindex(rows['winter']).values
    prect_anom = pd.Series(rows['prect_winter_mm_day'].values - curve)
    return dict(
        curves=curves, wet_threshold=float(prect_anom.quantile(WET_QUANTILE))
    )


def apply_transforms(rows, tf):
    """Subtract the fitted curves and label `wet` against the fitted
    threshold, on any rows (held-out ones included). Fits nothing."""
    out = rows.copy()
    for anom, raw in ANOMALIES.items():
        curve = tf['curves'][anom].reindex(out['winter']).values
        out[anom] = out[raw].values - curve
    out['wet'] = (out['prect_anom'] > tf['wet_threshold']).astype(int)
    return out


@dataclass
class State:
    """What a chapter needs, from the cache; returned by setup()."""

    table: pd.DataFrame
    cache_header: str
    access_date: str
    test_members: list
    is_train: pd.Series
    transforms: dict
    train: pd.DataFrame
    test: pd.DataFrame
    n_train_members: int
    n_folds: int
    n_repeats_effective: int


def setup(rebuild=False, n_members_per_half=50):
    """Read the cache, name the test members, fit the transforms on all
    training members, split, and set the fold count from the data."""
    table, cache_header, access_date = load_cache(rebuild, n_members_per_half)
    test_members = draw_test_members(table)
    is_test = table['member_id'].isin(test_members)
    table['split'] = np.where(is_test, 'test', 'train')
    is_train = table['split'].eq('train')
    transforms = fit_transforms(table[is_train])
    table = apply_transforms(table, transforms)
    train = table[is_train].reset_index(drop=True)
    test = table[~is_train].reset_index(drop=True)
    shared = set(train.member_id) & set(test.member_id)
    assert not shared, f"split leaks members: {shared}"
    n_train_members = train.member_id.nunique()
    assert n_train_members >= 2, 'rebuild with more members per half'
    n_folds = min(N_FOLDS_MAX, n_train_members)
    # Leave-one-member-out re-draws are identical: repeating them would
    # count the same differences N_REPEATS times over.
    n_repeats = N_REPEATS if n_train_members > n_folds else 1
    return State(
        table,
        cache_header,
        access_date,
        test_members,
        is_train,
        transforms,
        train,
        test,
        n_train_members,
        n_folds,
        n_repeats,
    )


# ==========================================================================
# Folds and scoring
# ==========================================================================
def member_folds(seed, n_folds):
    """Member-grouped, class-balanced folds over the training members. The
    balancing reads the all-training labels; the labels scored in a fold
    are refit by score_splits, so this decides fold membership only."""
    return StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )


def score_splits(model, rows, splits, refit_transforms=True):
    """Fit on each split's training rows, score its held-out rows.

    With refit_transforms=True the forced curves and the wet threshold are
    refit on the split's training rows only and applied to both sides. With
    False the columns already in `rows` are used: transforms fit once on
    ALL training members, before the folds. Returns per-fold precision,
    recall, F1, accuracy, AP and wet-call share for class 1 (wet), and the
    summed confusion matrix (rows = true, columns = predicted)."""
    keys = ['precision', 'recall', 'f1', 'accuracy', 'ap', 'called_wet']
    out = {k: [] for k in keys}
    cms = []
    for a, b in splits:
        fit_rows, held_rows = rows.iloc[a], rows.iloc[b]
        if refit_transforms:
            tf = fit_transforms(fit_rows)
            fit_rows = apply_transforms(fit_rows, tf)
            held_rows = apply_transforms(held_rows, tf)
        Xa, ya = fit_rows[FEATURES].values, fit_rows['wet'].values
        Xb, yb = held_rows[FEATURES].values, held_rows['wet'].values
        model.fit(Xa, ya)
        yhat = model.predict(Xb)
        if hasattr(model, 'predict_proba'):
            score = model.predict_proba(Xb)[:, 1]
        else:
            score = model.decision_function(Xb)
        out['precision'].append(precision_score(yb, yhat, zero_division=0))
        out['recall'].append(recall_score(yb, yhat, zero_division=0))
        out['f1'].append(f1_score(yb, yhat, zero_division=0))
        out['accuracy'].append(accuracy_score(yb, yhat))
        out['ap'].append(average_precision_score(yb, score))
        out['called_wet'].append(float(np.mean(yhat)))
        cms.append(confusion_matrix(yb, yhat, labels=[0, 1]))
    return {k: np.array(v) for k, v in out.items()}, np.sum(cms, axis=0)


def run_folds(model, rows, seed, n_folds):
    """Member-grouped folds, scored with the transforms refit per fold."""
    X, y = rows[FEATURES].values, rows['wet'].values
    folds = member_folds(seed, n_folds).split(X, y, rows['member_id'].values)
    return score_splits(model, rows, folds)


def decile_edges(rows):
    """The edges of ten equal-count bins of the Niño 3.4 anomaly."""
    _, edges = pd.qcut(rows['nino34_anom'], 10, retbins=True)
    return edges


def wet_by_bin(rows, edges):
    """Wet rate per Niño 3.4 bin (the outer edges opened to ±inf, so
    edges fitted on one set of rows apply to any other), with a binomial
    standard error, the distance from these rows' own base rate in those
    errors, and a chi-square test of wet against bin.
    Returns (table, chi2, p, dof)."""
    e = np.array(edges, dtype=float)
    e[0], e[-1] = -np.inf, np.inf
    b = pd.cut(rows['nino34_anom'], e)
    base = rows['wet'].mean()
    t = rows.groupby(b, observed=True)['wet'].agg(['mean', 'size'])
    t.columns = ['wet_rate', 'n']
    t['binomial_se'] = np.sqrt(t['wet_rate'] * (1 - t['wet_rate']) / t['n'])
    t['(rate - base) / se'] = (t['wet_rate'] - base) / t['binomial_se']
    t.index.name = 'nino34_anom bin (K)'
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(b, rows['wet']))
    return t, float(chi2), float(p), int(dof)


def pooled(cm):
    """Precision and recall for wet from a summed confusion matrix."""
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    return dict(
        precision_pooled=tp / (tp + fp) if tp + fp else 0.0,
        recall_pooled=tp / (tp + fn) if tp + fn else 0.0,
    )


# ==========================================================================
# The ladder
# ==========================================================================
class SignRule(BaseEstimator, ClassifierMixin):
    """Rung 0c: wet if the autumn Niño 3.4 anomaly is positive. Fits
    nothing; its only score is its 0/1 decision."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > 0).astype(int)

    def predict_proba(self, X):
        p = self.predict(X)
        return np.c_[1 - p, p]


def ladder(seed):
    """The candidates, preprocessing inside a Pipeline. Rung 0 is three
    rules: always-dry (a bar on accuracy only), always-wet (recall 1 at
    precision = the base rate) and the Niño-sign rule."""
    return {
        'always-dry': DummyClassifier(strategy='most_frequent'),
        'always-wet': DummyClassifier(strategy='constant', constant=1),
        'nino-sign': Pipeline([('rule', SignRule())]),
        'logistic': Pipeline(
            [
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(random_state=seed)),
            ]
        ),
        'tree': DecisionTreeClassifier(**TREE_SETTINGS, random_state=seed),
        'boosting': GradientBoostingClassifier(
            **BOOSTING_SETTINGS, random_state=seed
        ),
    }


RUNG0 = ['always-dry', 'always-wet', 'nino-sign']


def at_base_rate(name):
    """A trial factory: the rung with its decision threshold lowered from
    0.5 to the designed base rate, 1 - WET_QUANTILE."""

    def make(seed):
        return FixedThresholdClassifier(
            ladder(seed)[name],
            threshold=1 - WET_QUANTILE,
            response_method='predict_proba',
        )

    return make


def rung(name):
    """A trial factory for one ladder rung at its default threshold."""
    return lambda seed: ladder(seed)[name]


# ==========================================================================
# The paired test
# ==========================================================================
# Copyright (c) 2007-2026 The scikit-learn developers. BSD-3-Clause; full
# text in LICENSES/scikit-learn-BSD-3-Clause.txt. From
# examples/model_selection/plot_grid_search_stats.py
def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach."""
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    return np.sqrt(corrected_var)


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance."""
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    if std == 0:  # added: identical scores in every fold (guards 0/0)
        return 0.0, 1.0
    t_stat = mean / std
    # changed from the example: t_stat, not np.abs(t_stat), so the test is
    # right-tailed as the docstring says and a LOSS prints p near 1
    p_val = t_dist.sf(t_stat, df)
    return t_stat, p_val


METRICS = ['recall', 'precision']  # the two numbers the decision uses


def trial(make_model, rows, state):
    """Repeat the member-grouped folds n_repeats_effective times with
    different fold draws; per-fold precision, recall and wet-call share."""
    keys = METRICS + ['called_wet']
    scores = {m: [] for m in keys}
    for rep in range(state.n_repeats_effective):
        seed = RNG_SEED + rep
        s, _ = run_folds(make_model(seed), rows, seed, state.n_folds)
        for m in keys:
            scores[m].extend(s[m])
    return {m: np.array(v) for m, v in scores.items()}


def compare(incumbent_scores, candidate_scores, n_rows, n_folds, metric):
    """Corrected paired t-test on the per-fold differences of one metric
    (same fold draws for both models). Unrounded: verdict() reads these."""
    n_test = n_rows // n_folds
    n_train = n_rows - n_test
    diff = candidate_scores[metric] - incumbent_scores[metric]
    t_stat, p_val = compute_corrected_ttest(
        diff, df=len(diff) - 1, n_train=n_train, n_test=n_test
    )
    return {
        f"{metric}_cand": float(candidate_scores[metric].mean()),
        f"{metric}_inc": float(incumbent_scores[metric].mean()),
        f"{metric}_gap": float(diff.mean()),
        f"{metric}_sd": float(diff.std(ddof=1)),
        f"{metric}_t": float(t_stat),
        f"{metric}_p": float(p_val),
    }


def show_p(p):
    """A p-value to three significant figures."""
    return f"{p:.3g}"


def display_row(row):
    """The register's display precision: 3 dp; `_t` 2 dp; `_p` three
    significant figures. The verdict was taken on unrounded values."""

    def shown(k, v):
        if not isinstance(v, float):
            return v
        if k.endswith('_t'):
            return round(v, 2)
        if k.endswith('_p'):
            return float(show_p(v))
        return round(v, 3)

    return {k: shown(k, v) for k, v in row.items()}


def verdict(row):
    """Promoted only if one metric improves measurably (p < 0.05, gap > 0)
    and the other does not fall on average."""
    for up, keep in [('recall', 'precision'), ('precision', 'recall')]:
        if (
            row[f"{up}_p"] < 0.05
            and row[f"{up}_gap"] > 0
            and row[f"{keep}_gap"] >= 0
        ):
            return f"promoted: {up} up, no {keep} loss"
    return 'kept: no measured gain without a loss'


# ==========================================================================
# The register
# ==========================================================================
MAX_TESTED_TRIALS = 6


class Register:
    """One row per trial, against the current incumbent. The first
    incumbent is the Niño-sign rule, the physical prior."""

    def __init__(self, state):
        self.state, self.rows, self.scores = state, [], {}
        self.incumbent_name = 'nino-sign'
        self.incumbent_make = rung('nino-sign')
        self.incumbent_scores = trial(self.incumbent_make, state.train, state)
        self.scores['nino-sign'] = self.incumbent_scores
        n = len(self.incumbent_scores['recall'])
        print(
            f"first incumbent: nino-sign   fold draws per trial: "
            f"{state.n_repeats_effective} × {state.n_folds} folds = "
            f"{n} paired scores per metric"
        )

    def record(self, name, make_model, what_changed):
        """Run a candidate against the incumbent on identical fold draws;
        promote only by verdict(). A name already recorded is skipped."""
        if any(r['candidate'] == name for r in self.rows):
            print(f"{name}: already in the register — skipped")
            return self.table()
        assert len(self.rows) < MAX_TESTED_TRIALS, 'trial budget spent'
        train = self.state.train
        cand = trial(make_model, train, self.state)
        self.scores[name] = cand
        row = dict(
            trial=len(self.rows) + 1,
            candidate=name,
            incumbent=self.incumbent_name,
            what_changed=what_changed,
        )
        inc = self.incumbent_scores
        for m in METRICS:
            row.update(compare(inc, cand, len(train), self.state.n_folds, m))
            if m == 'recall':  # the wet-call share, beside recall
                row['called_wet_cand'] = float(cand['called_wet'].mean())
                row['called_wet_inc'] = float(inc['called_wet'].mean())
        row['verdict'] = verdict(row)
        self.rows.append(display_row(row))
        if row['verdict'].startswith('promoted'):
            self.incumbent_name = name
            self.incumbent_make = make_model
            self.incumbent_scores = cand
        return self.table()

    def table(self):
        return pd.DataFrame(self.rows)

    def bonferroni_line(self):
        """0.05 divided by the looks taken so far (two per tested row), the
        multiple-comparisons line the register does NOT apply."""
        looks = 2 * len(self.rows)
        cap = 0.05 / (2 * MAX_TESTED_TRIALS)
        print(
            f"Bonferroni line after {len(self.rows)} tested row(s) = "
            f"{looks} looks: p < {0.05 / looks:.4f}   (at the cap of "
            f"{MAX_TESTED_TRIALS} rows: p < {cap:.4f}). A row promoted on a "
            'p above this line is promoted under the uncorrected rule only.'
        )


def recall_vs_calls(reg, name):
    """How much of a trial's recall gain calling wet more often accounts
    for. Random extra wet calls catch wet winters at the base rate, so a
    no-skill classifier's recall equals its wet-call share: a change in
    share of d moves recall by about d with no skill added."""
    row = next(r for r in reg.rows if r['candidate'] == name)
    cand = reg.scores[name]
    inc = reg.scores[row['incumbent']]
    d_share = cand['called_wet'].mean() - inc['called_wet'].mean()
    d_recall = cand['recall'].mean() - inc['recall'].mean()
    print(
        f'{name} vs {row["incumbent"]}: wet-call share '
        f'{cand["called_wet"].mean():.3f} vs {inc["called_wet"].mean():.3f}'
        f' (change {d_share:+.3f}); recall {cand["recall"].mean():.3f} vs '
        f'{inc["recall"].mean():.3f} (gain {d_recall:+.3f})'
    )
    print(
        f"calling wet more often, with no skill, accounts for "
        f"{d_share:+.3f} of the {d_recall:+.3f} recall gain "
        f"({d_share / d_recall:.0%}); the rest, {d_recall - d_share:+.3f}, "
        'is the ranking'
    )


# The registered trials, in order: chapter 5 runs them one by one; chapter
# 6 rebuilds the same register before its one test.
TRIALS = [
    (
        'logistic',
        rung('logistic'),
        'rung 1: logistic on the scaled anomaly, default threshold 0.5',
    ),
    ('tree', rung('tree'), f"rung 2: tree {TREE_SETTINGS}"),
    ('boosting', rung('boosting'), f"rung 3: boosting {BOOSTING_SETTINGS}"),
    (
        'logistic@base-rate',
        at_base_rate('logistic'),
        'example trial: logistic, decision threshold lowered from 0.5 '
        'to the designed base rate',
    ),
    (
        'boosting@base-rate',
        at_base_rate('boosting'),
        'boosting, decision threshold lowered from 0.5 to the designed '
        'base rate',
    ),
]


def rebuild_register(state, extra=()):
    """A new Register with every trial in TRIALS (then `extra`) recorded,
    in order."""
    reg = Register(state)
    for t in list(TRIALS) + list(extra):
        reg.record(*t)
    return reg


# ==========================================================================
# Scoring frozen models once (chapter 7; checked against chapter 6)
# ==========================================================================
def score_frozen(models, rows, edges, n_boot=1000):
    """Score already-fitted models on `rows`, once, fitting nothing.

    Per model: precision, recall, wet-call share, false alarms per hit, and
    precision and AP minus the base rate of these rows, each with a 95%
    interval from resampling whole members with replacement (chapter 6's
    method, a fresh RandomState(RNG_SEED) per model, bounds read at 3 dp).
    Then the wet rate by bin on `edges` with a chi-square test. Returns a
    dict of the point values and bounds per model."""
    X, y = rows[FEATURES].values, rows['wet'].values
    base = y.mean()
    ids = rows['member_id'].values
    member_rows = {m: np.flatnonzero(ids == m) for m in pd.unique(ids)}
    names = list(member_rows)
    print(f'{len(names)} members · {len(y)} winters · base rate {base:.3f}')
    results = {}
    for name, model in models.items():
        yhat = model.predict(X)
        score = model.predict_proba(X)[:, 1]
        tn, fp, fn, tp = (
            int(v) for v in confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        )
        point = {
            'precision − base rate': precision_score(y, yhat, zero_division=0)
            - base,
            'AP − base rate': average_precision_score(y, score) - base,
        }
        rng = np.random.RandomState(RNG_SEED)
        draws = {k: [] for k in point}
        for _ in range(n_boot):
            picked = rng.choice(names, size=len(names), replace=True)
            pick = np.concatenate([member_rows[m] for m in picked])
            yb, hb, sb = y[pick], yhat[pick], score[pick]
            draws['precision − base rate'].append(
                precision_score(yb, hb, zero_division=0) - yb.mean()
            )
            draws['AP − base rate'].append(
                average_precision_score(yb, sb) - yb.mean()
            )
        results[name] = {}
        for k, v in draws.items():
            lo, hi = (
                round(float(b), 3) for b in np.percentile(v, [2.5, 97.5])
            )
            if lo > 0:
                reading = 'above no skill'
            elif hi < 0:
                reading = 'below no skill'
            else:
                reading = 'NOT distinguishable from no skill'
            print(
                f'{name}: {k:22s} {point[k]:+.3f}   95% interval over '
                f'members [{lo:+.3f}, {hi:+.3f}]  →  {reading}'
            )
            results[name][k] = (float(point[k]), lo, hi)
        fa_per_hit = f'{fp / tp:.2f}' if tp else 'undefined (no hits)'
        print(
            f'{name}: recall {recall_score(y, yhat, zero_division=0):.3f}  '
            f'wet-call share {yhat.mean():.3f}  false alarms {fp} · hits '
            f'{tp} · false alarms per hit {fa_per_hit}'
        )
    bins, chi2, p, dof = wet_by_bin(rows, edges)
    print('wet rate by bin of the autumn Niño 3.4 anomaly (edges given):')
    print(bins.round(3).to_string())
    print(
        f'chi-square test of wet against bin: chi2 = {chi2:.1f}, '
        f'dof = {dof}, p = {p:.2g}'
    )
    return results
