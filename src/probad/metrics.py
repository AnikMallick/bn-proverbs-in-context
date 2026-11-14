import math

import numpy as np
import pandas as pd

KEYS = ["model", "label", "variant", "item_id", "proverb_id", "figurative"]

def p_gold(df):
    """Probability on the gold letter (does not depend on option order, unlike p_a / p_b)."""
    return df.apply(lambda r: r[f"p_{r['gold'].lower()}"], axis=1)


def chance(df):
    """1 / number of options shown."""
    return 1 / df.filter(regex=r"^p_[a-z]$").notna().sum(axis=1)


# collapse option orders
def item_level(df, mode="averaged"):
    """Rows for the same item scored with several option orders (seed=0, 1, ...) -> one row per item.
    averaged: share of orders correct (soft); strict: 1 only if every order is correct."""
    df = df.assign(p_gold=p_gold(df), chance=chance(df))
    keys = [k for k in KEYS if k in df]
    g = df.groupby(keys, dropna=False)
    out = g.agg(correct=("correct", "mean" if mode == "averaged" else "all"), p_gold=("p_gold", "mean"),
                margin=("margin", "mean"), letter_mass=("letter_mass", "mean"), chance=("chance", "first"),
                n_orders=("correct", "size")).reset_index()
    out["correct"] = out["correct"].astype(float)
    return out


# ---------------------------------------------------------------- accuracy with CIs
def _boot(values, clusters, n=2000, seed=0):
    """Bootstrap means, resampling whole proverbs (items of one proverb are not independent)."""
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums, counts = np.bincount(inv, np.asarray(values, float)), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(counts), (n, len(counts)))
    return sums[pick].sum(1) / counts[pick].sum(1)


def bootstrap(values, clusters, n=2000, seed=0):
    """Mean and 95% CI."""
    return float(np.mean(values)), *np.percentile(_boot(values, clusters, n, seed), [2.5, 97.5])


def accuracy(items, by=None):
    """Accuracy, 95% CI and chance per group (by = column name or list, e.g. "figurative", ["model", "label"])."""
    by = [by] if isinstance(by, str) else list(by or [])
    rows = []
    for key, g in (items.groupby(by) if by else [((), items)]):
        acc, lo, hi = bootstrap(g["correct"], g["proverb_id"])
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "acc": acc, "ci_lo": lo,
                     "ci_hi": hi, "chance": g["chance"].mean(), "n": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- comparing two conditions
def mcnemar(a, b):
    """Exact two-sided McNemar p on paired 0/1 outcomes."""
    a, b = np.asarray(a) >= 0.5, np.asarray(b) >= 0.5
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n, k = n01 + n10, min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p_mcnemar": p}


def compare(a, b, on="item_id", n=2000, seed=0):
    """Paired difference b - a on the items both have (e.g. C0 vs C1b, bn vs en, base vs adapter).
    CI by resampling proverbs; p = share of bootstrap differences on the other side of 0, x2."""
    m = a.merge(b, on=[on, "proverb_id"], suffixes=("_a", "_b"))
    d = m["correct_b"] - m["correct_a"]
    boot = _boot(d, m["proverb_id"], n, seed)
    diff, (lo, hi) = d.mean(), np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return {"acc_a": m["correct_a"].mean(), "acc_b": m["correct_b"].mean(), "diff": diff,
            "ci_lo": lo, "ci_hi": hi, "p_boot": p, **mcnemar(m["correct_a"], m["correct_b"]), "n": len(m)}


def kappa(a, b):
    """Cohen's kappa between two label lists (e.g. predictions of two conditions, or two annotators)."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- knowing (E1) vs using (E2/x1)
def memorised(mem):
    """MAPS rule: a proverb is memorised if any template completes it. One row per proverb x render."""
    return mem.groupby(["model", "proverb_id", "render"])["correct"].any().reset_index(name="memorised")


def memorisation_rate(mem):
    k = memorised(mem)
    return accuracy(k.rename(columns={"memorised": "correct"}).assign(chance=0.0), by=["model", "render"])


def knowing_vs_using(items, know, render="bengali"):
    """Accuracy for memorised vs not memorised proverbs, their difference, and the four quadrants."""
    k = know[know["render"] == render][["model", "proverb_id", "memorised"]]
    df = items.merge(k, on=["model", "proverb_id"], how="left").fillna({"memorised": False})
    df["memorised"] = df["memorised"].astype(bool)
    acc = accuracy(df, by="memorised")
    q = df.assign(quadrant=[("know" if m else "dont_know") + "+" + ("use" if c >= 0.5 else "dont_use")
                            for m, c in zip(df["memorised"], df["correct"])])["quadrant"].value_counts()
    return {"by_memorised": acc, "quadrants": q.to_dict(), "n_memorised": int(df["memorised"].sum())}