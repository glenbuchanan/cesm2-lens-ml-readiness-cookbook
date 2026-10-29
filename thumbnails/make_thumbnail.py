#!/usr/bin/env python3
"""make_thumbnail.py — draw thumbnails/thumbnail.png, the cookbook's gallery thumbnail and README hero image.

The picture: autumn Niño 3.4 (minus the mean of the table) against DJF Front Range precipitation for every member-winter in the
committed cache, the wetter half of member-winters marked — the question the cookbook asks a classifier.

The credit under the axes is LENS_CITATION exactly as the notebook defines it (one home: notebooks/02_applications.ipynb,
Imports cell, with ACCESS_DATE filled from the settings cell), so the thumbnail carries what every notebook figure carries:
the author's © element, both DOIs, "figure CC BY 4.0" and the dated access line.

Run from the repo root:  python thumbnails/make_thumbnail.py   (or: uv run --with pandas --with matplotlib thumbnails/make_thumbnail.py)
"""
import ast
import json
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOTEBOOK = os.path.join(REPO, "notebooks", "02_applications.ipynb")
DATA = os.path.join(REPO, "notebooks", "data", "member_winters.csv")
OUT = os.path.join(HERE, "thumbnail.png")
WANTED = ("ACCESS_LINE", "LENS_CITATION", "ACCESS_DATE")


def citation_from_notebook(path):
    """LENS_CITATION with ACCESS_DATE filled, from the notebook's own assignments — nothing is retyped here."""
    nb = json.load(open(path, encoding="utf-8"))
    ns = {}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANTED for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    cit = ns.get("LENS_CITATION", "")
    if not cit or "ACCESS_DATE" in cit:
        sys.exit("STOP: LENS_CITATION / ACCESS_DATE not resolvable from %s (got %r)" % (path, cit[:80]))
    return cit


def main():
    citation = citation_from_notebook(NOTEBOOK)
    df = pd.read_csv(DATA, comment="#")
    x = df["nino34_son_K"] - df["nino34_son_K"].mean()
    y = df["prect_winter_mm_day"]
    wet = y > y.median()

    plt.rcParams["figure.dpi"] = 100
    plt.rcParams.update({"figure.facecolor": "#FBFAF6", "axes.facecolor": "#F5F3ED", "axes.edgecolor": "#E4E0D6", "axes.labelcolor": "#13161A",
                         "text.color": "#13161A", "xtick.color": "#6A7077", "ytick.color": "#6A7077", "savefig.facecolor": "#FBFAF6"})
    fig, ax = plt.subplots(figsize=(12, 9.4))
    ax.scatter(x[~wet], y[~wet], s=38, alpha=.8, c="#6A7077", marker="o", label="drier half of member-winters")
    ax.scatter(x[wet], y[wet], s=50, alpha=.95, c="#C9A23A", marker="^", label="wetter half of member-winters")
    ax.axvline(0, color="#7c828a", ls="--", lw=1.2)
    ax.set_xlabel("autumn (SON) Niño 3.4, K, minus the mean of the table", fontsize=15)
    ax.set_ylabel("DJF Front Range precipitation, mm/day", fontsize=15)
    ax.set_title("Does the autumn ENSO state predict a wet Front Range winter?\nA classifier on CESM2 LENS, split by ensemble member",
                 fontsize=18, loc="left")
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    n_members = df["member_id"].nunique()
    fig.text(0.01, -0.02, citation + "\nCommitted cache, %d members (notebooks/data/member_winters.csv)." % n_members,
             fontsize=9, ha="left", va="top")
    plt.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=100)
    print("wrote %s" % OUT)
    print("credit on image:\n" + citation)


if __name__ == "__main__":
    main()
