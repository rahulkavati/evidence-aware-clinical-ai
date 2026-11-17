"""Single frozen final evaluation for the evidence-aware sepsis system.

Stages (run in order; ``--stage all`` runs every stage in one process):

1. ``freeze``  - validation only. Selects the early-warning alert thresholds
   on the validation *selection* half, fits the patient-specific acquisition
   selector on the training ``policy_fit`` cohort, and writes
   ``configs/final_protocol.json`` with SHA-256 hashes of every artifact.
2. ``predict`` - caches probabilities for every system on one split.
3. ``evaluate`` - discrimination, selective prediction with escalation,
   early-warning utility with patient-bootstrap intervals, wait/re-check and
   horizon analyses.
4. ``bootstrap`` - paired patient-cluster bootstrap of AUROC/AUPRC/Brier for
   each acquisition policy versus the labs-hidden baseline.
5. ``report`` - merges everything into
   ``experiments/results/final/final_<split>_evaluation.{json,md}``.

Held-out access is opt-in (``--split test --locked-test``), refused unless the
frozen protocol hashes match the files on disk, and is appended to
``experiments/results/final/test_access_ledger.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from experiments.run_complete_next_phase import (
    ALL_LABS,
    GROUP_ORDER,
    GROUPS,
    group_probabilities,
    observed_groups,
    policy_probabilities,
    predict,
)
from src.acquisition.patient_specific import PatientSpecificPolicy, choose_patient_specific
from src.data.g1_masking import apply_hidden_set
from src.data.multihorizon_labels import add_horizon_targets
from src.evaluation.abstention_acquisition import decision_metrics, evaluate_trivial_baselines
from src.evaluation.early_warning import (
    patient_level_warning,
    select_alert_threshold,
    utility_with_interval,
)
from src.evaluation.paired_bootstrap import paired_cluster_bootstrap_rows
from src.evaluation.selective_prediction import confidence
from src.evaluation.wait_recheck import wait_recheck_candidates
from src.uncertainty.calibrate_models import split_patient_ids

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / os.environ.get("FINAL_MODEL_DIR", "outputs/next_phase_all_labs")
MODEL_PATH = MODEL_DIR / "all_lab_mask_aware_xgboost.joblib"
CALIBRATOR_PATH = MODEL_DIR / "all_lab_sigmoid_calibrator.joblib"
TRAINING_SUMMARY_PATH = MODEL_DIR / "all_lab_validation.json"
SELECTOR_PATH = MODEL_DIR / "patient_specific_policy.joblib"
PROTOCOL_PATH = ROOT / "configs/final_protocol.json"
CACHE_DIR = ROOT / os.environ.get("FINAL_CACHE_DIR", "outputs/final_evaluation")
RESULTS_DIR = ROOT / "experiments/results/final"
LEDGER_PATH = RESULTS_DIR / "test_access_ledger.json"
DATA = {
    "train": ROOT / "data/processed/train.parquet",
    "validation": ROOT / "data/processed/val.parquet",
    "test": ROOT / "data/processed/test.parquet",
}
POLICY_COHORT = ROOT / "outputs/g1_cohorts/policy_fit.csv"
POLICIES = ("fixed_order", "random", "patient_specific", "oracle_upper_bound")
SYSTEMS = ("labs_hidden", "as_recorded") + POLICIES
BOOTSTRAP_RESAMPLES = int(os.environ.get("FINAL_BOOTSTRAP_RESAMPLES", 200))
UTILITY_RESAMPLES = int(os.environ.get("FINAL_UTILITY_RESAMPLES", 1000))


# ---------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - git may be unavailable
        return None


def working_tree_clean() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return out.strip() == ""
    except Exception:  # pragma: no cover
        return None


def load_model():
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "set_params"):
        model.set_params(n_jobs=int(os.environ.get("FINAL_PREDICT_JOBS", 1)))
    return model, joblib.load(CALIBRATOR_PATH)


def validation_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same patient-disjoint halves used when the abstention threshold was chosen."""
    selection_ids, evaluation_ids = split_patient_ids(
        frame.groupby("patient_id")["SepsisLabel"].max(), 42
    )
    selection = frame[frame["patient_id"].isin(selection_ids)].reset_index(drop=True)
    evaluation = frame[frame["patient_id"].isin(evaluation_ids)].reset_index(drop=True)
    return selection, evaluation


def evaluation_frame(split: str) -> pd.DataFrame:
    frame = pd.read_parquet(DATA[split])
    if split == "validation":
        # Thresholds were selected on the other half; report the untouched half.
        return validation_halves(frame)[1]
    return frame.reset_index(drop=True)


def read_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("Run --stage freeze first (configs/final_protocol.json missing)")
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def verify_protocol(protocol: dict[str, Any]) -> None:
    for name, path in (("model", MODEL_PATH), ("calibrator", CALIBRATOR_PATH), ("selector", SELECTOR_PATH)):
        if sha256(path) != protocol["artifact_sha256"][name]:
            raise RuntimeError(
                f"{name} file changed after the protocol was frozen; refusing to evaluate"
            )


def protocol_hash() -> str:
    return sha256(PROTOCOL_PATH)


def ledger() -> dict[str, Any]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def record_test_access(protocol: dict[str, Any]) -> None:
    data = ledger()
    digest = protocol_hash()
    if any(entry.get("protocol_sha256") == digest for entry in data["accesses"]):
        return  # later stages of the same locked access
    data["accesses"].append({
        "index": len(data["accesses"]) + 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "experiments/run_final_evaluation.py",
        "git_commit": git_commit(),
        "working_tree_clean": working_tree_clean(),
        "protocol_sha256": digest,
        "abstention_threshold": protocol["abstention_threshold"],
        "alert_thresholds": protocol["alert_thresholds"],
        "purpose": "Single final evaluation under the frozen final protocol.",
    })
    LEDGER_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ stages
def stage_freeze(force: bool) -> None:
    if PROTOCOL_PATH.exists() and not force:
        raise RuntimeError("Protocol already frozen; pass --force to refreeze (validation only)")
    if LEDGER_PATH.exists() and force:
        frozen = read_protocol() if PROTOCOL_PATH.exists() else None
        if frozen and any(
            entry.get("protocol_sha256") == protocol_hash() for entry in ledger()["accesses"]
        ):
            raise RuntimeError("This protocol was already used on the test split; it cannot be refrozen")
    model, calibrator = load_model()
    training_summary = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
    abstention_threshold = float(training_summary["threshold"])

    selection, _ = validation_halves(pd.read_parquet(DATA["validation"]))
    as_recorded = predict(model, calibrator, selection)
    labs_hidden = predict(model, calibrator, apply_hidden_set(selection, ALL_LABS))
    alert_as_recorded = select_alert_threshold(selection, as_recorded)
    alert_hidden = select_alert_threshold(selection, labs_hidden)

    train = pd.read_parquet(DATA["train"])
    cohort = set(pd.read_csv(POLICY_COHORT)["patient_id"].astype(str))
    policy_frame = train[train["patient_id"].astype(str).isin(cohort)].reset_index(drop=True)
    del train
    base, groups = group_probabilities(model, calibrator, policy_frame)
    # Rescue means moving below the rule-out boundary, so the selector learns
    # which group is likely to cross 1 - threshold (not the 0.5 boundary).
    selector = PatientSpecificPolicy.fit(
        policy_frame, base, groups, GROUPS, ALL_LABS,
        decision_threshold=1.0 - abstention_threshold,
    )
    joblib.dump(selector, SELECTOR_PATH)

    protocol = {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit_at_freeze": git_commit(),
        "model": "All-26-lab mask-aware XGBoost with state-specific Platt calibration",
        "artifact_sha256": {
            "model": sha256(MODEL_PATH),
            "calibrator": sha256(CALIBRATOR_PATH),
            "selector": sha256(SELECTOR_PATH),
            "train_parquet": sha256(DATA["train"]),
            "validation_parquet": sha256(DATA["validation"]),
        },
        "abstention_threshold": abstention_threshold,
        "abstention_rule": "retain a prediction when max(p, 1-p) >= threshold; otherwise escalate",
        "abstention_threshold_source": "validation selection half, selective risk <= 1%",
        "alert_thresholds": {
            "as_recorded": alert_as_recorded["threshold"],
            "labs_hidden_and_policies": alert_hidden["threshold"],
        },
        "alert_threshold_source": "validation selection half, maximum normalized PhysioNet utility",
        "alert_threshold_curves": {
            "as_recorded": alert_as_recorded["curve"],
            "labs_hidden_and_policies": alert_hidden["curve"],
        },
        "policies": {
            "acquisition_scope": "one clinical lab group, only for rows the labs-hidden baseline abstains on, only if historically observed",
            "fixed_order": list(GROUP_ORDER),
            "random_seed": 42,
            "patient_specific": "logistic selectors fit on training policy_fit cohort; target = crossing the rule-out boundary 1 - threshold",
            "oracle_upper_bound": "uses labels; evaluation ceiling only",
        },
        "validation_report_scope": "validation evaluation half (patient-disjoint from the selection half)",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "utility_resamples": UTILITY_RESAMPLES, "seed": 42},
    }
    write_json(PROTOCOL_PATH, protocol)
    print(json.dumps({k: protocol[k] for k in ("abstention_threshold", "alert_thresholds")}, indent=2))


def stage_predict(split: str) -> None:
    protocol = read_protocol()
    verify_protocol(protocol)
    if split == "test":
        record_test_access(protocol)
    model, calibrator = load_model()
    selector = joblib.load(SELECTOR_PATH)
    threshold = protocol["abstention_threshold"]
    frame = evaluation_frame(split)
    labs_hidden, groups = group_probabilities(model, calibrator, frame)
    as_recorded = predict(model, calibrator, frame)
    abstained = confidence(labs_hidden) < threshold
    arrays = {"labs_hidden": labs_hidden, "as_recorded": as_recorded}
    eligible_rows = {}
    for policy in ("fixed_order", "random", "oracle_upper_bound"):
        arrays[policy], eligible_rows[policy] = policy_probabilities(
            frame, labs_hidden, groups, policy, abstained, threshold=threshold
        )
    selected = choose_patient_specific(
        selector.score(frame, GROUPS, ALL_LABS), observed_groups(frame), abstained, GROUP_ORDER
    )
    patient_specific = labs_hidden.copy()
    for index, name in enumerate(GROUP_ORDER):
        patient_specific[selected == index] = groups[name][selected == index]
    arrays["patient_specific"] = patient_specific
    eligible_rows["patient_specific"] = int((selected >= 0).sum())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / f"{split}_predictions.npz",
        **arrays,
        **{f"group_{name}": values for name, values in groups.items()},
        patient_specific_selected=selected,
    )
    write_json(CACHE_DIR / f"{split}_eligible_rows.json", eligible_rows)
    print(f"Cached predictions for {split}: {len(frame)} rows")


def load_cache(split: str) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"{split}_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run --stage predict --split {split} first")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def discrimination(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def stage_evaluate(split: str) -> None:
    protocol = read_protocol()
    verify_protocol(protocol)
    frame = evaluation_frame(split)
    cache = load_cache(split)
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    threshold = protocol["abstention_threshold"]
    alert_hidden = protocol["alert_thresholds"]["labs_hidden_and_policies"]
    alert_recorded = protocol["alert_thresholds"]["as_recorded"]
    reference_alerts = (cache["labs_hidden"] >= alert_hidden).astype(int)
    eligible = json.loads((CACHE_DIR / f"{split}_eligible_rows.json").read_text())

    systems: dict[str, Any] = {}
    for name in SYSTEMS:
        p = cache[name]
        retained = confidence(p) >= threshold
        alert_threshold = alert_recorded if name == "as_recorded" else alert_hidden
        alerts = (p >= alert_threshold).astype(int)
        systems[name] = {
            "discrimination": discrimination(labels, p),
            "selective": decision_metrics(labels, p, retained),
            "escalation_patient_level": patient_level_warning(frame, ~retained),
            "alert_threshold": alert_threshold,
            "early_warning_utility": utility_with_interval(
                frame, alerts,
                reference_alerts=None if name == "labs_hidden" else reference_alerts,
                n_resamples=UTILITY_RESAMPLES,
            ),
            "alert_patient_level": patient_level_warning(frame, alerts.astype(bool)),
            "eligible_rows": eligible.get(name),
        }
        print(f"evaluated {name}")

    hidden = cache["labs_hidden"]
    references = {
        "never_alert_utility": 0.0,
        "always_alert_utility": utility_with_interval(
            frame, np.ones(len(frame), dtype=int), n_resamples=200
        )["normalized_utility"],
        "selective_trivial_baselines": evaluate_trivial_baselines(labels, hidden),
    }

    abstained = confidence(hidden) < threshold
    current, future = wait_recheck_candidates(frame, abstained)
    after = hidden[future]
    retained_after = confidence(after) >= threshold
    wait = {
        "baseline_abstained_rows": int(abstained.sum()),
        "rows_with_future_observation": int(len(current)),
        "median_delay_hours": float(np.median(
            frame["ICULOS"].to_numpy()[future] - frame["ICULOS"].to_numpy()[current]
        )) if len(current) else None,
        "resolved_after_wait_rate": float(retained_after.mean()) if len(current) else None,
        "next_observation_metrics": decision_metrics(labels[future], after, retained_after) if len(current) else None,
    }

    targets = add_horizon_targets(frame[["patient_id", "ICULOS", "SepsisLabel"]], (6, 12, 24))
    negative = targets["SepsisLabel"].to_numpy() == 0
    horizons = {}
    for horizon in (6, 12, 24):
        y = targets[f"SepsisWithin{horizon}h"].to_numpy()[negative]
        horizons[str(horizon)] = {
            "rows": int(negative.sum()),
            "positive_rate": float(y.mean()),
            **discrimination(y, hidden[negative]),
            "note": "current-label model on currently negative rows; descriptive, not horizon-trained",
        }

    write_json(CACHE_DIR / f"{split}_evaluation.json", {
        "split": split,
        "patients": int(frame["patient_id"].nunique()),
        "rows": int(len(frame)),
        "positive_rows": int(labels.sum()),
        "systems": systems,
        "references": references,
        "wait_recheck": wait,
        "horizons": horizons,
    })


def stage_bootstrap(split: str, policies: list[str]) -> None:
    protocol = read_protocol()
    verify_protocol(protocol)
    frame = evaluation_frame(split)
    cache = load_cache(split)
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    ids = frame["patient_id"].astype(str).to_numpy()

    def records(values):
        return [
            {"patient_id": patient, "label": int(label), "probability": float(probability)}
            for patient, label, probability in zip(ids, labels, values)
        ]

    baseline = records(cache["labs_hidden"])
    for policy in policies:
        result = paired_cluster_bootstrap_rows(
            baseline, records(cache[policy]), n_resamples=BOOTSTRAP_RESAMPLES
        )
        write_json(CACHE_DIR / f"{split}_bootstrap_{policy}.json", result)
        print(f"bootstrapped {policy}")


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def ci(values: list[float] | None, digits: int = 3) -> str:
    return "n/a" if not values else f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def stage_report(split: str) -> None:
    protocol = read_protocol()
    evaluation = json.loads((CACHE_DIR / f"{split}_evaluation.json").read_text())
    bootstraps = {}
    for policy in ("as_recorded",) + POLICIES:
        path = CACHE_DIR / f"{split}_bootstrap_{policy}.json"
        if path.exists():
            bootstraps[policy] = json.loads(path.read_text())
    result = {"protocol": protocol, "evaluation": evaluation, "paired_bootstrap_vs_labs_hidden": bootstraps}
    if split == "test":
        result["test_access_ledger"] = ledger()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / f"final_{split}_evaluation.json", result)
    (RESULTS_DIR / f"final_{split}_evaluation.md").write_text(render_markdown(result, split), encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / f'final_{split}_evaluation.md'}")


LABELS = {
    "labs_hidden": "All 26 labs hidden (baseline)",
    "as_recorded": "Labs as recorded (full information reference)",
    "fixed_order": "Acquire: fixed clinical-group order",
    "random": "Acquire: random eligible group",
    "patient_specific": "Acquire: patient-specific selector",
    "oracle_upper_bound": "Acquire: oracle ceiling (uses labels)",
}


def render_markdown(result: dict[str, Any], split: str) -> str:
    protocol, ev = result["protocol"], result["evaluation"]
    systems = ev["systems"]
    title = "held-out test" if split == "test" else "validation (evaluation half)"
    lines = [
        f"# Final evaluation — {title}",
        "",
        "Generated by `experiments/run_final_evaluation.py` under the frozen protocol in "
        "`configs/final_protocol.json`. Thresholds were selected on validation data only.",
        "",
        f"- Patients: **{ev['patients']:,}**, hourly rows: **{ev['rows']:,}**, positive rows: **{ev['positive_rows']:,}**",
        f"- Abstention threshold (confidence): `{protocol['abstention_threshold']:.6f}`",
        f"- Alert threshold, labs hidden / policies: `{protocol['alert_thresholds']['labs_hidden_and_policies']:.6f}`; "
        f"labs as recorded: `{protocol['alert_thresholds']['as_recorded']:.6f}`",
        "",
        "## 1. Early-warning performance (official PhysioNet 2019 utility)",
        "",
        "Normalized utility: 0 = never alert, 1 = perfect timing. 95% intervals resample patients.",
        "",
        "| System | AUROC | AUPRC | Utility [95% CI] | Δ vs baseline [95% CI] | Septic patients warned before onset | Median lead (h) | Non-septic patients alerted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SYSTEMS:
        s = systems[name]
        u = s["early_warning_utility"]
        a = s["alert_patient_level"]
        delta = "—" if name == "labs_hidden" else f"{u['difference_vs_reference']:+.3f} {ci(u['difference_ci95'])}"
        lines.append(
            f"| {LABELS[name]} | {num(s['discrimination']['auroc'])} | {num(s['discrimination']['auprc'])} | "
            f"{num(u['normalized_utility'])} {ci(u['ci95'])} | {delta} | {pct(a['septic_warning_rate'])} | "
            f"{num(a['median_lead_time_hours'], 1)} | {pct(a['non_septic_flag_rate'])} |"
        )
    refs = ev["references"]
    lines += [
        "",
        f"Reference systems: never alert = 0.000; alert every hour = {num(refs['always_alert_utility'])}.",
        "",
        "## 2. Selective prediction with escalation",
        "",
        "When the model is not confident it does not predict; the row is escalated (acquire a lab group, wait, or refer to a clinician). "
        "Retained predictions at this operating point are rule-out (negative) decisions, so escalation is how positives are surfaced.",
        "",
        "| System | Coverage | Risk on retained | Escalated rows | Positive rows escalated | Septic patients escalated before onset | Non-septic patients escalated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in SYSTEMS:
        s = systems[name]
        sel, esc = s["selective"], s["escalation_patient_level"]
        lines.append(
            f"| {LABELS[name]} | {pct(sel['coverage'])} | {pct(sel['risk'], 2)} | {pct(1 - sel['coverage'])} | "
            f"{pct(sel['abstained_positive_capture'])} | {pct(esc['septic_warning_rate'])} | {pct(esc['non_septic_flag_rate'])} |"
        )
    trivial = refs["selective_trivial_baselines"]["predict_negative_everywhere"]
    lines += [
        "",
        f"Trivial reference: predicting negative everywhere has 100% coverage and {pct(trivial['risk'], 2)} risk.",
        "",
        "## 3. Acquisition policies versus the labs-hidden baseline",
        "",
        "Paired patient-cluster bootstrap (all rows).",
        "",
        "| Policy | Eligible rows | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | Brier improvement [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, boot in result["paired_bootstrap_vs_labs_hidden"].items():
        o, c = boot["observed"], boot["confidence_intervals_95"]
        lines.append(
            f"| {LABELS[name]} | {systems[name]['eligible_rows'] or '—'} | {o['auroc']:+.4f} {ci(c['auroc'], 4)} | "
            f"{o['auprc']:+.4f} {ci(c['auprc'], 4)} | {o['brier_improvement']:+.6f} {ci(c['brier_improvement'], 6)} |"
        )
    wait = ev["wait_recheck"]
    lines += [
        "",
        "## 4. Wait and re-check",
        "",
        f"Of {wait['baseline_abstained_rows']:,} escalated rows, {wait['rows_with_future_observation']:,} had a next observation "
        f"(median delay {num(wait['median_delay_hours'], 0)} h); {pct(wait['resolved_after_wait_rate'])} became confident after waiting.",
        "",
        "## 5. Future-onset horizons (descriptive)",
        "",
        "| Horizon | Rows | Positive rate | AUROC | AUPRC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon, h in ev["horizons"].items():
        lines.append(f"| {horizon} h | {h['rows']:,} | {pct(h['positive_rate'], 2)} | {num(h['auroc'])} | {num(h['auprc'])} |")
    if split == "test":
        lines += [
            "",
            "## 6. Held-out access disclosure",
            "",
            "Every recorded access to the held-out split is listed in "
            "`experiments/results/final/test_access_ledger.json`, including earlier development-phase accesses.",
            "",
            "| # | Date | Source | Note |",
            "|---:|---|---|---|",
        ]
        for entry in result["test_access_ledger"]["accesses"]:
            lines.append(
                f"| {entry['index']} | {entry['timestamp'][:10]} | `{entry['source']}` | {entry.get('purpose', '')} |"
            )
    lines += ["", "Retrospective research results; not evidence of prospective clinical benefit.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("freeze", "predict", "evaluate", "bootstrap", "report", "all"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--locked-test", action="store_true", help="Required to touch the held-out split.")
    parser.add_argument("--policy", action="append", help="Limit --stage bootstrap to these systems.")
    parser.add_argument("--force", action="store_true", help="Refreeze a protocol never used on test.")
    args = parser.parse_args()
    if args.split == "test" and args.stage != "freeze" and not args.locked_test:
        parser.error("The held-out split requires --locked-test")
    policies = args.policy or ["as_recorded", *POLICIES]
    if args.stage == "freeze":
        stage_freeze(args.force)
    elif args.stage == "predict":
        stage_predict(args.split)
    elif args.stage == "evaluate":
        stage_evaluate(args.split)
    elif args.stage == "bootstrap":
        stage_bootstrap(args.split, policies)
    elif args.stage == "report":
        stage_report(args.split)
    else:
        if not PROTOCOL_PATH.exists():
            stage_freeze(False)
        stage_predict(args.split)
        stage_evaluate(args.split)
        stage_bootstrap(args.split, policies)
        stage_report(args.split)


if __name__ == "__main__":
    main()
