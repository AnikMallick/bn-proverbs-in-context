"""From per-instance results to report numbers: order collapsing, accuracy with CIs, paired comparison."""

import numpy as np
import pandas as pd
from scipy import stats

from probad.constants import FORMAT_CHANCE

SETTING = ["model", "condition", "lang", "render", "format", "control"]
META = ["proverb_id", "figurative", "has_false_friends", "region"]


def item_level(df, strict=False):
    """One row per item and setting, collapsing the option orders.

    Averaged (default): mean probability of each option over the orders, then the highest wins.
    Strict: correct only if correct in every order.
    """
    g = df.groupby(SETTING + ["item_id"], sort=False)
    p_cols = [c for c in df.columns if c.startswith("p_")]
    items = g[p_cols].mean()
    items["pred_key"] = items[p_cols].idxmax(axis=1).str.removeprefix("p_")
    items["gold_key"] = g["gold_key"].first()
    items["correct"] = g["correct"].all() if strict else items["pred_key"] == items["gold_key"]
    items["margin"] = g["margin"].mean()
    meta = [c for c in META if c in df.columns]
    items[meta] = g[meta].first()
    return items.reset_index()


def _ci(x, n_boot=2000, seed=0):
    """95% percentile bootstrap CI of the mean."""
    if len(x) < 2:
        return np.nan, np.nan
    ci = stats.bootstrap((x,), np.mean, n_resamples=n_boot, method="percentile", rng=seed).confidence_interval
    return ci.low, ci.high


def accuracy(items, by=("model", "condition", "format")):
    """Accuracy per group with a bootstrap CI over items, mean margin and chance level."""
    by = list(by)
    rows = []
    for key, g in items.groupby(by, sort=False):
        x = g["correct"].to_numpy(float)
        low, high = _ci(x)
        row = {**dict(zip(by, key)), "n": len(x), "acc": x.mean(), "ci_low": low, "ci_high": high, "margin": g["margin"].mean()}
        if "format" in by:
            row["chance"] = FORMAT_CHANCE[row["format"]]
        rows.append(row)
    return pd.DataFrame(rows)


def compare(items, a, b, by=("model", "format")):
    """Condition `a` vs `b` on the same items: accuracy difference (bootstrap CI) and McNemar's exact test."""
    keys = [c for c in SETTING if c != "condition"] + ["item_id"]
    pair = pd.concat(
        {name: items[items["condition"] == name].set_index(keys)["correct"] for name in (a, b)}, axis=1, join="inner"
    ).reset_index()
    by = list(by)
    rows = []
    for key, g in pair.groupby(by, sort=False):
        diff = g[a].to_numpy(float) - g[b].to_numpy(float)
        a_only, b_only = int((g[a] & ~g[b]).sum()), int((~g[a] & g[b]).sum())  # discordant pairs
        p = stats.binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0
        low, high = _ci(diff)
        rows.append(
            {
                "a": a,
                "b": b,
                **dict(zip(by, key)),
                "n": len(g),
                "acc_a": g[a].mean(),
                "acc_b": g[b].mean(),
                "diff": diff.mean(),
                "ci_low": low,
                "ci_high": high,
                "only_a": a_only,  # items only `a` gets right
                "only_b": b_only,
                "p_mcnemar": p,
            }
        )
    return pd.DataFrame(rows)
