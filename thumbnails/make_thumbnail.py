#!/usr/bin/env python3
"""make_thumbnail.py: draw thumbnails/thumbnail.png, the gallery thumbnail.

The picture is the cookbook's own target: the share of wet winters in each
tenth of the autumn Niño 3.4 anomaly (training-member decile edges), for the
training members and the held-out test members, with the forced response
removed and the wet threshold fitted on training members, as chapters 2 and
6 compute it (lensml.setup, lensml.wet_by_bin). The test line carries its
±2 binomial standard errors and its chi-square p (chapter 6's test line), so
the card does not show a replicated shape that the test does not support.
Minimal text, sized to be
read on a gallery card; the data citation lives in the README and
notebooks/how-to-cite.md, not on the image.

Run from the repository root:  python thumbnails/make_thumbnail.py
"""

import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.join(os.path.dirname(HERE), 'notebooks')
OUT = os.path.join(HERE, 'thumbnail.png')


def main():
    os.chdir(NOTEBOOKS)  # lensml reads data/member_winters.csv from here
    sys.path.insert(0, NOTEBOOKS)
    import lensml

    state = lensml.setup()
    edges = lensml.decile_edges(state.train)  # training edges for both
    lines = {
        'training members': ('#8F7022', '#C9A23A', 'o', state.train),
        'test members': ('#5B6168', '#FBFAF6', 's', state.test),
    }
    base = state.train['wet'].mean()

    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FBFAF6')
    ax.set_facecolor('#F5F3ED')  # the house canvas inside the axes
    for label, (line, face, marker, rows) in lines.items():
        table, _, p, _ = lensml.wet_by_bin(rows, edges)
        rate = table['wet_rate']
        if rows is state.test:
            test_p = p
            ax.errorbar(  # ±2 SE: the cookbook's own "beyond 2 SE" line
                range(1, 11),
                rate.values,
                yerr=2 * table['binomial_se'].values,
                fmt='none',
                ecolor=line,
                elinewidth=2,
                capsize=5,
                zorder=1,
            )
        ax.plot(
            range(1, 11),
            rate.values,
            color=line,
            lw=3,
            marker=marker,
            markersize=14,
            markerfacecolor=face,
            label=label,
        )
    train_rate = lensml.wet_by_bin(state.train, edges)[0]['wet_rate']
    ax.fill_between(  # the training curve's departure from no skill
        range(1, 11), train_rate.values, base, color='#EAD27E', lw=0, zorder=0
    )
    ax.axhline(base, color='#13161A', ls='--', lw=2)
    ax.text(10.4, base + 0.005, 'base rate', ha='right', va='bottom')
    ax.text(  # computed above from the test rows, never typed
        0.6,
        0.205,
        f'test bins ±2 SE; chi-square p = {test_p:.2f}',
        ha='left',
        va='bottom',
        fontsize=15,
        color='#5B6168',
    )
    ax.set_ylim(0.20, 0.48)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks([1, 10], ['La Niña', 'El Niño'])
    ax.set_ylabel('share of winters wet')
    ax.set_title('Wet Front Range winters,\nby autumn ENSO state', loc='left')
    ax.legend(frameon=False, fontsize=18, loc='upper right')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT, dpi=100)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
