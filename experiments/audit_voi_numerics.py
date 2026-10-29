"""Training-only numerical audit of the existing VOI policy-fitting predictor.

Run from repository root: python -m experiments.audit_voi_numerics --policy-size 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_voi_acquisition import (
    TRAIN_SPLIT_FILE, TRAIN_DATA_FILE, CANDIDATE_LABS, CHECKPOINTS,
    _fit_independent_predictor, diagnostic_training_cohorts,
    initial_hidden_features, mask_checkpoint_feature_vector,
)

QUANTILES = [0, .25, .5, .75, .9, .95, .99, 1]


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    if not np.isfinite(a).all():
        raise ValueError("Non-finite diagnostic values")
    return {"count": int(len(a)), "mean": float(a.mean()),
            "quantiles": {str(q): float(np.quantile(a, q)) for q in QUANTILES},
            "exact_zero": int(np.count_nonzero(a == 0)),
            "abs_le_1e_12": int(np.count_nonzero(np.abs(a) <= 1e-12)),
            "abs_gt_1e_6": int(np.count_nonzero(np.abs(a) > 1e-6))}


def patient_concentration(records):
    """Share of absolute benefit attributable to top patients, not signed benefit."""
    by_patient = {}
    for r in records:
        key = r["patient_id"]
        by_patient[key] = by_patient.get(key, 0.) + abs(r["benefit"])
    weights = np.sort(np.asarray(list(by_patient.values()), dtype=float))[::-1]
    total = float(weights.sum())
    if total == 0:
        return {"patient_count": len(weights), "total_absolute_benefit": 0.,
                "top_1pct_share": None, "top_5pct_share": None,
                "top_10pct_share": None}
    result = {"patient_count": len(weights), "total_absolute_benefit": total}
    for pct in (1, 5, 10):
        n = max(1, int(np.ceil(len(weights) * pct / 100)))
        result[f"top_{pct}pct_share"] = float(weights[:n].sum() / total)
    return result


def audit(policy_size=3000, seed=42, n_estimators=20, batch_size=20000):
    manifest = pd.read_csv(TRAIN_SPLIT_FILE)
    fit, cohorts = diagnostic_training_cohorts(
        manifest, policy_sizes=(policy_size,), seed=seed)
    policy_patients = cohorts[policy_size]
    if not set(fit.patient_id).isdisjoint(set(policy_patients.patient_id)):
        raise AssertionError("Predictor-fit/policy patient overlap")
    model, features = _fit_independent_predictor(
        fit, n_estimators=n_estimators, seed=seed)
    ids = set(policy_patients.patient_id.astype(str))
    cols = ["patient_id", "ICULOS", "SepsisLabel"] + list(features)
    data = pd.read_parquet(TRAIN_DATA_FILE, columns=list(dict.fromkeys(cols)))
    data = data[data.patient_id.astype(str).isin(ids) & data.ICULOS.isin(CHECKPOINTS)]
    if data.duplicated(["patient_id", "ICULOS"]).any():
        raise ValueError("Duplicate patient/checkpoint rows")
    hidden = set(initial_hidden_features())
    vectors, metadata = [], []
    for row in data.itertuples(index=False, name="Row"):
        record = row._asdict()
        label = int(record["SepsisLabel"])
        if label not in (0, 1):
            raise ValueError("Nonbinary checkpoint label")
        original = np.asarray([record[f] for f in features], dtype=float)
        before = mask_checkpoint_feature_vector(original, features, hidden)
        for feature in CANDIDATE_LABS:
            if float(record[f"{feature}_observed"]) <= 0:
                continue
            after = mask_checkpoint_feature_vector(original, features, hidden - {feature})
            different = ~np.isclose(before, after, rtol=0, atol=0, equal_nan=True)
            expected = {
                feature,
                f"{feature}_observed",
                f"{feature}_age",
                f"{feature}_last",
            }                           
            changed = {features[i] for i in np.flatnonzero(different)}
            if not changed.issubset(expected):
                raise AssertionError(f"Unexpected feature changes: {changed - expected}")
            if not changed:
                raise AssertionError(f"Reveal had no effect for {feature}")
            vectors.extend((before, after))
            metadata.append((str(record["patient_id"]), int(record["ICULOS"]),
                             feature, label, sorted(changed)))
    if not vectors:
        raise ValueError("No eligible acquisition states")
    predictions = []
    for start in range(0, len(vectors), batch_size):
        batch = pd.DataFrame(np.asarray(vectors[start:start+batch_size]), columns=features)
        p = np.asarray(model.predict_proba(batch)[:, 1], dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid predicted probabilities")
        predictions.extend(p.tolist())
    records = []
    for i, (pid, hour, feature, label, changed) in enumerate(metadata):
        before, after = predictions[2*i:2*i+2]
        benefit = (label-before)**2 - (label-after)**2
        records.append({"patient_id": pid, "hour": hour, "feature": feature,
                        "label": label, "before": before, "after": after,
                        "delta": after-before, "benefit": benefit,
                        "changed_columns": changed})
    groups = {}
    for feature in CANDIDATE_LABS:
        for hour in CHECKPOINTS:
            for label in (0, 1):
                group = [r for r in records if r["feature"] == feature and
                         r["hour"] == hour and r["label"] == label]
                groups[f"{feature}|{hour}|{label}"] = {
                    "benefit": summarize([r["benefit"] for r in group]),
                    "absolute_probability_change": summarize([abs(r["delta"]) for r in group]),
                    "probability_before": summarize([r["before"] for r in group]),
                    "probability_after": summarize([r["after"] for r in group]),
                    "influence": patient_concentration(group)}
    return {"configuration": {"policy_size": policy_size, "seed": seed,
                "n_estimators": n_estimators, "fit_patients": len(fit),
                "policy_patients": len(policy_patients), "records": len(records),
                "predictor": "independent uncalibrated policy-fitting predictor",
                "test_or_validation_used": False},
            "checkpoint_positive_counts": {
                str(h): {"positive": int((data.loc[data.ICULOS == h, "SepsisLabel"] == 1).sum()),
                         "total": int((data.ICULOS == h).sum())} for h in CHECKPOINTS},
            "overall": {"benefit": summarize([r["benefit"] for r in records]),
                        "absolute_probability_change": summarize([abs(r["delta"]) for r in records]),
                        "influence": patient_concentration(records)},
            "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=20)
    args = parser.parse_args()
    result = audit(args.policy_size, args.seed, args.n_estimators)
    directory = Path("experiments/results/acquisition/voi")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"numerical_audit_{args.policy_size}"
    (directory / (stem + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = [f"# VOI numerical audit: {args.policy_size} policy patients", "",
             "Training-derived, patient-disjoint predictor; no validation/test data.", "",
             f"- Predictor-fit patients: {result['configuration']['fit_patients']}",
             f"- Acquisition records: {result['configuration']['records']}",
             "- The independent policy-fitting predictor is NOT sigmoid calibrated.",
             "- Absolute-benefit influence measures concentration, not net benefit.", "",
             "## Overall", "",
             "```json", json.dumps(result["overall"], indent=2), "```", "",
             "## By feature, checkpoint, and outcome", "",
             "| Feature | Hour | Label | N | Mean benefit | Mean absolute probability change |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, value in result["groups"].items():
        feature, hour, label = key.split("|")
        benefit = value["benefit"]
        delta = value["absolute_probability_change"]
        lines.append(f"| {feature} | {hour} | {label} | {benefit['count']} | "
                     f"{benefit.get('mean', 0):.10e} | {delta.get('mean', 0):.10e} |")
    (directory / (stem + ".md")).write_text("\n".join(lines) + "\n")
    print(f"Wrote {directory / (stem + '.md')}")
    print(json.dumps(result["configuration"], indent=2))


if __name__ == "__main__":
    main()
