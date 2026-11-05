"""G1 retrospective information-revelation comparison.

For each checkpoint row we compare a reference state with all candidate labs
hidden ("no acquisition") against:

* ``none_hidden``: every lab visible, a full-revelation reference for an
  acquisition policy under the oracle-availability assumption (only labs that
  were actually recorded by the checkpoint can be revealed);
* ``reveal_<lab>``: exactly one lab revealed.

Small or uncertain differences motivate further analysis; they do not prove
that no selective acquisition policy could help.

All uncertainty is from patient-cluster bootstrap (patients resampled, all
their checkpoint rows kept together).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.g1_masking import CANDIDATE_LABS, apply_hidden_set, triplet_columns
from src.uncertainty.platt import apply_platt

CHECKPOINTS = (6, 12, 24)
NB_THRESHOLDS = (0.02, 0.05, 0.10)


# ----------------------------------------------------------------- data prep
def checkpoint_rows(df: pd.DataFrame, checkpoints: Sequence[int] = CHECKPOINTS) -> pd.DataFrame:
    rows = df[df["ICULOS"].isin(list(checkpoints))].copy()
    if rows.duplicated(["patient_id", "ICULOS"]).any():
        raise ValueError("Duplicate (patient, hour) rows")
    return rows.reset_index(drop=True)


def label_timing_summary(df: pd.DataFrame, checkpoints: Sequence[int] = CHECKPOINTS) -> dict:
    """How the checkpoint labels relate to the first positive hour."""
    ordered = df.sort_values(["patient_id", "ICULOS"])
    first_pos = (
        ordered[ordered["SepsisLabel"] == 1].groupby("patient_id")["ICULOS"].min().rename("first_pos")
    )
    cummax = ordered.groupby("patient_id")["SepsisLabel"].cummax()
    reverting = int(ordered.loc[cummax != ordered["SepsisLabel"], "patient_id"].nunique())

    post_rows = (
        ordered.join(first_pos, on="patient_id")
        .query("first_pos == first_pos and ICULOS > first_pos")
        .groupby("patient_id")
        .size()
    )
    summary = {
        "patients_where_label_returns_to_zero": reverting,
        "sepsis_patients": int(len(first_pos)),
        "median_rows_after_first_positive": float(post_rows.median()) if len(post_rows) else 0.0,
        "per_checkpoint": {},
    }
    for c in checkpoints:
        rows = ordered[ordered["ICULOS"] == c].join(first_pos, on="patient_id")
        positive = rows["SepsisLabel"] == 1
        summary["per_checkpoint"][int(c)] = {
            "rows": int(len(rows)),
            "positive": int(positive.sum()),
            "newly_positive_at_checkpoint": int((positive & (rows["first_pos"] == c)).sum()),
            "already_positive_before_checkpoint": int((positive & (rows["first_pos"] < c)).sum()),
            "negative_now_positive_later": int((~positive & (rows["first_pos"] > c)).sum()),
        }
    return summary


def staleness_summary(rows: pd.DataFrame, labs: Sequence[str] = CANDIDATE_LABS) -> dict:
    out: dict = {}
    for c, part in rows.groupby("ICULOS"):
        out[int(c)] = {}
        for lab in labs:
            _, observed, age = triplet_columns(lab)
            seen = part[observed] == 1
            ages = part.loc[seen, age]
            out[int(c)][lab] = {
                "fraction_ever_observed": float(seen.mean()),
                "median_age_hours": float(ages.median()) if len(ages) else None,
                "p90_age_hours": float(ages.quantile(0.9)) if len(ages) else None,
            }
    return out


# ------------------------------------------------------------------ metrics
def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Decision-curve net benefit at an alert threshold."""
    y = np.asarray(y)
    alert = np.asarray(p) >= threshold
    n = len(y)
    tp = float(np.sum(alert & (y == 1)))
    fp = float(np.sum(alert & (y == 0)))
    return tp / n - fp / n * threshold / (1.0 - threshold)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metric_vector(y: np.ndarray, p: np.ndarray, thresholds=NB_THRESHOLDS) -> dict[str, float]:
    """Metrics oriented so that HIGHER IS BETTER (Brier is negated)."""
    out = {"neg_brier": -float(np.mean((p - y) ** 2))}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = out["auprc"] = float("nan")
    for t in thresholds:
        out[f"nb@{t:g}"] = net_benefit(y, p, t)
    return out


# ---------------------------------------------------------------- bootstrap
def cluster_bootstrap_differences(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_alts: Mapping[str, np.ndarray],
    patient_ids: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired patient-cluster bootstrap of (alt - ref) for every metric."""
    y = np.asarray(y)
    codes, uniques = pd.factorize(np.asarray(patient_ids))
    order = np.argsort(codes, kind="stable")
    boundaries = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    members = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(uniques))]

    observed_ref = metric_vector(y, p_ref)
    observed = {
        name: {m: metric_vector(y, p)[m] - observed_ref[m] for m in observed_ref}
        for name, p in p_alts.items()
    }

    rng = np.random.default_rng(seed)
    draws = {name: {m: [] for m in observed_ref} for name in p_alts}
    skipped = 0
    for _ in range(n_resamples):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[i] for i in pick])
        yb = y[idx]
        if not 0 < yb.sum() < len(yb):
            skipped += 1
            continue
        ref_b = metric_vector(yb, p_ref[idx])
        for name, p in p_alts.items():
            alt_b = metric_vector(yb, p[idx])
            for m in ref_b:
                draws[name][m].append(alt_b[m] - ref_b[m])

    result = {"n_resamples": n_resamples, "skipped_single_class": skipped, "differences": {}}
    for name in p_alts:
        result["differences"][name] = {}
        for m, values in draws[name].items():
            lo, hi = (np.percentile(values, [2.5, 97.5]) if values else (float("nan"),) * 2)
            result["differences"][name][m] = {
                "observed": observed[name][m],
                "ci95": [float(lo), float(hi)],
            }
    return result


# --------------------------------------------------------------- prediction
def predict_calibrated(model, calibrator, X: pd.DataFrame) -> np.ndarray:
    features = list(model.feature_names_in_)
    raw = model.predict_proba(X[features])[:, 1]
    return apply_platt(calibrator, raw) if calibrator is not None else raw


def state_definitions(labs: Sequence[str] = CANDIDATE_LABS) -> dict[str, tuple[str, ...]]:
    states = {"all_hidden": tuple(labs), "none_hidden": ()}
    for lab in labs:
        states[f"reveal_{lab}"] = tuple(x for x in labs if x != lab)
    return states


def state_probabilities(model, calibrator, X: pd.DataFrame, labs=CANDIDATE_LABS) -> dict[str, np.ndarray]:
    return {
        name: predict_calibrated(model, calibrator, apply_hidden_set(X, hidden))
        for name, hidden in state_definitions(labs).items()
    }


# ------------------------------------------------------------------- driver
def run_headroom(
    hourly: pd.DataFrame,
    model,
    calibrator,
    checkpoints: Sequence[int] = CHECKPOINTS,
    labs: Sequence[str] = CANDIDATE_LABS,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    rows = checkpoint_rows(hourly, checkpoints)
    if rows.empty:
        raise ValueError("No checkpoint-eligible rows")
    probs = state_probabilities(model, calibrator, rows, labs)
    y = rows["SepsisLabel"].astype(int).to_numpy()
    patient_ids = rows["patient_id"].to_numpy()

    scopes: dict[str, np.ndarray] = {"pooled_all": np.ones(len(rows), dtype=bool)}
    for hospital in sorted(rows["hospital"].unique()):
        scopes[f"pooled_hospital_{hospital}"] = (rows["hospital"] == hospital).to_numpy()
    for c in checkpoints:
        scopes[f"hour_{c}_all"] = (rows["ICULOS"] == c).to_numpy()

    results: dict = {"scopes": {}}
    for scope, mask in scopes.items():
        ys = y[mask]
        entry = {
            "rows": int(mask.sum()),
            "patients": int(len(np.unique(patient_ids[mask]))),
            "positives": int(ys.sum()),
            "treat_all_nb": {f"{t:g}": net_benefit(ys, np.ones_like(ys, dtype=float), t) for t in NB_THRESHOLDS},
            "states": {
                name: {
                    **metric_vector(ys, p[mask]),
                    "ece": expected_calibration_error(ys, p[mask]),
                    "mean_probability": float(p[mask].mean()),
                }
                for name, p in probs.items()
            },
        }
        if entry["positives"] >= 2:
            alts = {n: p[mask] for n, p in probs.items() if n != "all_hidden"}
            entry["differences_vs_all_hidden"] = cluster_bootstrap_differences(
                ys, probs["all_hidden"][mask], alts, patient_ids[mask], n_resamples, seed
            )
        results["scopes"][scope] = entry

    results["label_timing"] = label_timing_summary(hourly, checkpoints)
    results["staleness"] = staleness_summary(rows, labs)
    results["config"] = {
        "checkpoints": list(checkpoints),
        "labs": list(labs),
        "nb_thresholds": list(NB_THRESHOLDS),
        "bootstrap_resamples": n_resamples,
        "seed": seed,
    }
    return results


# ------------------------------------------------------------------- report
def _fmt(x, digits=5):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def render_report(results: dict, tag: str) -> str:
    cfg = results["config"]
    lines = [
        f"# Headroom analysis ({tag})",
        "",
        "- Reference state: all candidate labs hidden (no acquisition).",
        "- `none_hidden` is the full-revelation reference, not a mathematical upper bound.",
        f"- Patient-cluster bootstrap, {cfg['bootstrap_resamples']} resamples, seed {cfg['seed']}; 95% percentile intervals.",
        "- Differences are alternative minus all_hidden; every metric is oriented so higher is better (Brier is negated).",
        "- `nb@t` is decision-curve net benefit at exploratory alert threshold t; thresholds are not clinically validated.",
        "- Multiple comparisons are exploratory; no multiplicity correction is applied.",
        "",
    ]
    for scope, entry in results["scopes"].items():
        lines += [
            f"## {scope}",
            "",
            f"{entry['patients']:,} patients, {entry['rows']:,} rows, {entry['positives']:,} positive labels.",
            "",
            "| State | AUROC | AUPRC | Brier | ECE | "
            + " | ".join(f"NB@{t:g}" for t in cfg["nb_thresholds"]) + " |",
            "|---|---:|---:|---:|---:|" + "---:|" * len(cfg["nb_thresholds"]),
        ]
        for name, m in entry["states"].items():
            lines.append(
                f"| {name} | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | {_fmt(-m['neg_brier'], 6)} | {_fmt(m['ece'])} | "
                + " | ".join(_fmt(m[f'nb@{t:g}'], 6) for t in cfg["nb_thresholds"]) + " |"
            )
        diffs = entry.get("differences_vs_all_hidden")
        if diffs:
            metrics = ["auroc", "auprc", "neg_brier"] + [f"nb@{t:g}" for t in cfg["nb_thresholds"]]
            lines += [
                "",
                "Difference vs all_hidden (95% CI):",
                "",
                "| State | " + " | ".join(metrics) + " |",
                "|---|" + "---:|" * len(metrics),
            ]
            for name, d in diffs["differences"].items():
                cells = [
                    f"{_fmt(d[m]['observed'], 5)} ({_fmt(d[m]['ci95'][0], 5)}, {_fmt(d[m]['ci95'][1], 5)})"
                    for m in metrics
                ]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Label timing at checkpoints", ""]
    lt = results["label_timing"]
    lines += [
        f"- Sepsis patients: {lt['sepsis_patients']:,}; patients whose label returns to 0 after turning 1: {lt['patients_where_label_returns_to_zero']:,}.",
        f"- Median rows after the first positive hour (sepsis patients): {lt['median_rows_after_first_positive']:.1f}.",
        "",
        "| Hour | Rows | Positive | Newly positive | Already positive | Negative now, positive later |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c, v in lt["per_checkpoint"].items():
        lines.append(
            f"| {c} | {v['rows']:,} | {v['positive']:,} | {v['newly_positive_at_checkpoint']:,} | "
            f"{v['already_positive_before_checkpoint']:,} | {v['negative_now_positive_later']:,} |"
        )
    lines += ["", "## Lab staleness at checkpoints", "", "| Hour | Lab | Ever observed | Median age (h) | P90 age (h) |", "|---:|---|---:|---:|---:|"]
    for c, labs_ in results["staleness"].items():
        for lab, v in labs_.items():
            lines.append(
                f"| {c} | {lab} | {_fmt(v['fraction_ever_observed'], 3)} | {_fmt(v['median_age_hours'], 1)} | {_fmt(v['p90_age_hours'], 1)} |"
            )
    return "\n".join(lines) + "\n"
