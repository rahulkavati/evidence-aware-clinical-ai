"""Regenerate every figure and table used in the project report.

Usage (from the repository root):
    PYTHONPATH=. python report/make_figures.py

Inputs are the committed final results and the cached held-out predictions
(run `make final-test` first if outputs/final_evaluation is empty).
Outputs: report/figures/*.png (300 dpi) and report/tables/*.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
TAB = ROOT / "report/tables"
FINAL_TEST = json.loads((ROOT / "experiments/results/final/final_test_evaluation.json").read_text())
FINAL_VAL = json.loads((ROOT / "experiments/results/final/final_validation_evaluation.json").read_text())
PROTOCOL = FINAL_TEST["protocol"]
SYSTEMS = FINAL_TEST["evaluation"]["systems"]
BLUE, ORANGE, INK, GREY, LIGHT = "#2a78d6", "#d9541e", "#1a1a1a", "#6f6d68", "#d9d8d1"
RED, AMBER = "#b42318", "#e8a317"
LABELS = {
    "labs_hidden": "All labs hidden (baseline)",
    "as_recorded": "Labs as recorded",
    "fixed_order": "Fixed clinical order",
    "random": "Random group",
    "patient_specific": "Patient-specific selector",
    "oracle_upper_bound": "Oracle (uses labels)",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 10.5,
    "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a8a",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.dpi": 100,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


def load_predictions():
    with np.load(ROOT / "outputs/final_evaluation/test_predictions.npz") as data:
        cache = {k: data[k] for k in data.files}
    frame = pd.read_parquet(ROOT / "data/processed/test.parquet").reset_index(drop=True)
    return frame, cache


def box(ax, x, y, w, h, title, sub="", fc="#f4f3ef", ec="#8a8a8a", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h * (0.64 if sub else 0.5), title, ha="center", va="center", fontsize=8.2, weight="bold" if bold else "normal")
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.4, color="#444")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    top = [("PhysioNet 2019", "40,336 ICU stays"), ("Patient split", "70 / 15 / 15"),
           ("Features", "108 per hour"), ("Model", "masked XGBoost")]
    bottom = [("Calibration", "Platt per state"), ("Frozen protocol", "validation only"),
              ("Acquisition", "4 policies"), ("Held-out test", "one locked run")]
    w, h, xs = 2.2, 1.0, [0.05, 2.6, 5.15, 7.7]
    for x, (t, sub) in zip(xs, top):
        box(ax, x, 2.9, w, h, t, sub)
    for x, (t, sub) in zip(xs, bottom):
        box(ax, x, 0.3, w, h, t, sub)
    for i in range(3):
        arrow(ax, xs[i] + w + 0.03, 3.4, xs[i + 1] - 0.03, 3.4)
        arrow(ax, xs[i] + w + 0.03, 0.8, xs[i + 1] - 0.03, 0.8)
    mx, cx = xs[3] + w / 2, xs[0] + w / 2
    ax.plot([mx, mx], [2.88, 2.1], color="#555", lw=0.9)
    ax.plot([mx, cx], [2.1, 2.1], color="#555", lw=0.9)
    arrow(ax, cx, 2.1, cx, 1.33)
    save(fig, "fig01_pipeline")


def fig_decision():
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
    box(ax, 0.05, 1.65, 1.8, 1.0, "Patient hour", "labs hidden")
    box(ax, 2.25, 1.65, 2.35, 1.0, "Calibrated risk p", "and its confidence")
    box(ax, 5.05, 3.0, 2.6, 1.0, "Decide", "confidence ≥ 0.968", fc="#e7f0fb", ec=BLUE)
    box(ax, 5.05, 0.3, 2.6, 1.0, "Order one lab group", "only if recorded", fc="#fdf1dc", ec=AMBER)
    box(ax, 8.1, 3.0, 1.85, 1.0, "Early alarm", "p ≥ 0.0217", fc="#fbe4e1", ec=RED)
    box(ax, 8.1, 0.3, 1.85, 1.0, "Escalate", "still unsure", fc="#fbe4e1", ec=RED)
    arrow(ax, 1.88, 2.15, 2.22, 2.15)
    arrow(ax, 4.63, 2.45, 5.02, 3.2)
    arrow(ax, 4.63, 1.85, 5.02, 1.1)
    arrow(ax, 7.68, 0.8, 8.07, 0.8)
    ax.annotate("", xy=(6.35, 2.97), xytext=(6.35, 1.33), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9, ls="--"))
    ax.text(6.45, 2.1, "re-score", fontsize=7.4, color="#444")
    ax.annotate("", xy=(8.07, 3.5), xytext=(7.68, 3.5), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9, ls=":"))
    ax.text(7.25, 4.12, "separate early-warning rule", fontsize=7.4, color="#444")
    save(fig, "fig02_decision")


def fig_missingness():
    miss = pd.read_csv(ROOT / "outputs/dataset_audit/feature_missingness.csv", index_col=0)["missing_percentage"]
    vitals = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
    keep = [f for f in miss.index if f not in {"Unit1", "Unit2", "HospAdmTime", "Gender", "Age", "ICULOS", "SepsisLabel"}]
    m = miss[keep].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    colors = [GREY if f in vitals else BLUE for f in m.index]
    ax.barh(range(len(m)), m.values, color=colors, height=0.7)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("Hours with no recorded value (%)")
    ax.grid(axis="y", visible=False)
    for i, (f, v) in enumerate(m.items()):
        if f in {"HR", "Lactate", "Temp", "Glucose", "Bilirubin_direct"}:
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="Vital sign"), Patch(color=BLUE, label="Laboratory test")], loc="lower right", frameon=False)
    save(fig, "fig03_missingness")
    m.rename("missing_percent").round(2).to_csv(TAB / "missingness.csv")


def fig_roc_calibration(frame, cache):
    y = frame["SepsisLabel"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    for key, color, ls in (("as_recorded", BLUE, "-"), ("labs_hidden", ORANGE, "--")):
        fpr, tpr, _ = roc_curve(y, cache[key])
        ax.plot(fpr, tpr, color=color, ls=ls, lw=1.6, label=f"{LABELS[key].replace(' (baseline)', '')}, AUROC {SYSTEMS[key]['discrimination']['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#aaa", lw=0.8, ls=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower right", frameon=False, fontsize=7.8)
    save(fig, "fig04_roc")

    edges = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0001])
    rows = []
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    for key, color, marker, ls in (("as_recorded", BLUE, "o", "-"), ("labs_hidden", ORANGE, "s", "--")):
        p = cache[key]
        which = np.searchsorted(edges, p, side="right") - 1
        pts = []
        for b in range(len(edges) - 1):
            mask = which == b
            if mask.sum() >= 50:
                pts.append((p[mask].mean(), y[mask].mean(), int(mask.sum())))
                rows.append({"system": key, "bin_low": edges[b], "bin_high": min(edges[b + 1], 1.0), "rows": int(mask.sum()),
                             "mean_predicted": round(float(p[mask].mean()), 5), "observed_rate": round(float(y[mask].mean()), 5)})
        pts = np.array(pts)
        ax.plot(pts[:, 0], np.maximum(pts[:, 1], 1e-3), color=color, ls=ls, marker=marker, ms=4, lw=1.3, label=LABELS[key])
    ax.plot([0.002, 1], [0.002, 1], color="#aaa", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.002, 1); ax.set_ylim(0.001, 1)
    ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed positive rate")
    pf = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:g}%")
    ax.xaxis.set_major_formatter(pf); ax.yaxis.set_major_formatter(pf)
    ax.legend(loc="upper left", frameon=False, fontsize=7.8)
    save(fig, "fig05_calibration")
    pd.DataFrame(rows).to_csv(TAB / "calibration_bins.csv", index=False)


def fig_utility_curve():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for key, color, ls, label in (("as_recorded", BLUE, "-", "Labs as recorded"), ("labs_hidden_and_policies", ORANGE, "--", "All labs hidden")):
        curve = PROTOCOL["alert_threshold_curves"][key]
        x = [c["threshold"] for c in curve]; u = [c["normalized_utility"] for c in curve]
        ax.plot(x, u, color=color, ls=ls, lw=1.6, label=label)
        t = PROTOCOL["alert_thresholds"][key]
        best = min(curve, key=lambda c: abs(c["threshold"] - t))
        ax.plot([best["threshold"]], [best["normalized_utility"]], marker="o", color=color, ms=6, mec="white")
        ax.annotate(f"frozen {t:.4f}", (best["threshold"], best["normalized_utility"]), textcoords="offset points",
                    xytext=(8, 6) if key == "as_recorded" else (30, -70), fontsize=8, arrowprops=None if key == "as_recorded" else dict(arrowstyle="-", color="#777", lw=0.7))
    ax.set_xscale("log"); ax.set_xlabel("Alarm threshold on predicted risk (log scale)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:g}%"))
    ax.set_ylabel("Normalized utility (validation)"); ax.set_ylim(-0.05, 0.5)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig06_utility_threshold")


def fig_risk_coverage(frame, cache):
    y = frame["SepsisLabel"].astype(int).to_numpy(); p = cache["labs_hidden"]
    conf = np.maximum(p, 1 - p); order = np.argsort(-conf, kind="stable")
    err = ((p[order] >= 0.5) != y[order].astype(bool)).astype(float)
    n = np.arange(1, len(p) + 1); risk = np.cumsum(err) / n; cov = n / len(p)
    keep = cov >= 0.5
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(cov[keep], 100 * risk[keep], color=ORANGE, lw=1.6, label="Selective model (labs hidden)")
    base = y.mean() * 100
    ax.axhline(base, color=GREY, ls="--", lw=1, label=f"Always predict no sepsis ({base:.2f}%)")
    s = SYSTEMS["labs_hidden"]["selective"]
    ax.plot([s["coverage"]], [100 * s["risk"]], "o", color=INK, ms=6)
    ax.annotate(f"frozen: {100*s['coverage']:.1f}% coverage, {100*s['risk']:.2f}% risk", (s["coverage"], 100 * s["risk"]),
                textcoords="offset points", xytext=(-235, 42), fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#777", lw=0.7))
    ax.set_xlabel("Coverage (share of hours on which the model decides)"); ax.set_ylabel("Selective risk (%)")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 2.6); ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "fig07_risk_coverage")


def fig_forest():
    boot = FINAL_TEST["paired_bootstrap_vs_labs_hidden"]
    order = ["as_recorded", "fixed_order", "random", "patient_specific", "oracle_upper_bound"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, title, getter in (
        (axes[0], "Δ normalized utility", lambda k: (SYSTEMS[k]["early_warning_utility"]["difference_vs_reference"], *SYSTEMS[k]["early_warning_utility"]["difference_ci95"])),
        (axes[1], "Δ AUROC", lambda k: (boot[k]["observed"]["auroc"], *boot[k]["confidence_intervals_95"]["auroc"])),
    ):
        for i, k in enumerate(order):
            est, lo, hi = getter(k); yv = len(order) - 1 - i
            color = BLUE if k == "as_recorded" else (GREY if k == "oracle_upper_bound" else INK)
            ax.plot([lo, hi], [yv, yv], color=color, lw=1.6)
            ax.plot([est], [yv], "o", color=color, mfc="white" if k == "oracle_upper_bound" else color, ms=6)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(title); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([LABELS[k] for k in reversed(order)], fontsize=8.5)
    save(fig, "fig08_acquisition_forest")


def fig_patient(frame, cache):
    pid = "A_p018345"
    rows = frame.index[frame.patient_id == pid]
    pt = frame.loc[rows].sort_values("ICULOS"); idx = pt.index.to_numpy()
    hours = pt["ICULOS"].to_numpy(); lab = pt["SepsisLabel"].to_numpy()
    onset = hours[np.argmax(lab)] + 6
    tau = PROTOCOL["abstention_threshold"]; alert = PROTOCOL["alert_thresholds"]["labs_hidden_and_policies"]
    ph, pr = cache["labs_hidden"][idx], cache["as_recorded"][idx]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    unsure = np.maximum(ph, 1 - ph) < tau
    for h, u in zip(hours, unsure):
        if u:
            ax.axvspan(h - 0.5, h + 0.5, color="#fdecc8", lw=0)
    ax.plot(hours, ph, color=ORANGE, lw=1.6, label="Risk, labs hidden")
    ax.plot(hours, pr, color=BLUE, lw=1.3, ls="--", label="Risk, labs as recorded")
    ax.axhline(1 - tau, color=INK, lw=0.8, ls=":", label="Confidence boundary (3.2%)")
    ax.axhline(alert, color=RED, lw=0.8, ls="-.", label="Alarm threshold (2.17%)")
    ax.axvline(onset, color=RED, lw=1.2)
    ax.text(onset - 0.6, 0.45, "clinical onset", color=RED, ha="right", fontsize=8.5)
    first_alarm = hours[ph >= alert][0]
    ax.annotate(f"first alarm, hour {first_alarm}", (first_alarm, ph[hours == first_alarm][0]), textcoords="offset points", xytext=(-110, 30),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#777", lw=0.7))
    ax.set_yscale("log"); ax.set_ylim(0.002, 1); ax.set_xlim(hours[0] - 0.5, hours[-1] + 0.5)
    ax.set_xlabel("ICU hour"); ax.set_ylabel("Calibrated sepsis risk (log scale)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.legend(frameon=False, loc="upper left", fontsize=7.8, ncol=2)
    save(fig, "fig09_patient_example")
    return {"patient": pid, "hours": int(len(hours)), "onset_hour": int(onset), "first_alarm_hour": int(first_alarm)}


def tables():
    ev = FINAL_TEST["evaluation"]; boot = FINAL_TEST["paired_bootstrap_vs_labs_hidden"]
    rows = []
    for k in ("labs_hidden", "as_recorded"):
        s = SYSTEMS[k]
        rows.append({"system": LABELS[k], "auroc": s["discrimination"]["auroc"], "auprc": s["discrimination"]["auprc"],
                     "brier": s["discrimination"]["brier"], "utility": s["early_warning_utility"]["normalized_utility"],
                     "utility_ci_low": s["early_warning_utility"]["ci95"][0], "utility_ci_high": s["early_warning_utility"]["ci95"][1],
                     "septic_warned": s["alert_patient_level"]["septic_warning_rate"],
                     "median_lead_h": s["alert_patient_level"]["median_lead_time_hours"],
                     "nonseptic_alarmed": s["alert_patient_level"]["non_septic_flag_rate"]})
    pd.DataFrame(rows).round(4).to_csv(TAB / "test_main.csv", index=False)
    rows = []
    for k in SYSTEMS:
        s = SYSTEMS[k]["selective"]; e = SYSTEMS[k]["escalation_patient_level"]
        rows.append({"system": LABELS[k], "coverage": s["coverage"], "risk": s["risk"], "escalated_positive_capture": s["abstained_positive_capture"],
                     "septic_escalated_before_onset": e["septic_warning_rate"], "nonseptic_escalated": e["non_septic_flag_rate"],
                     "eligible_rows": SYSTEMS[k].get("eligible_rows")})
    pd.DataFrame(rows).round(4).to_csv(TAB / "test_selective.csv", index=False)
    rows = []
    for k in ("as_recorded", "fixed_order", "random", "patient_specific", "oracle_upper_bound"):
        u = SYSTEMS[k]["early_warning_utility"]; b = boot[k]
        rows.append({"system": LABELS[k], "d_utility": u["difference_vs_reference"], "d_utility_low": u["difference_ci95"][0], "d_utility_high": u["difference_ci95"][1],
                     "d_auroc": b["observed"]["auroc"], "d_auroc_low": b["confidence_intervals_95"]["auroc"][0], "d_auroc_high": b["confidence_intervals_95"]["auroc"][1],
                     "d_auprc": b["observed"]["auprc"], "d_auprc_low": b["confidence_intervals_95"]["auprc"][0], "d_auprc_high": b["confidence_intervals_95"]["auprc"][1]})
    pd.DataFrame(rows).round(4).to_csv(TAB / "test_acquisition.csv", index=False)
    pd.DataFrame([{"horizon_h": int(h), **{k: v for k, v in d.items() if k != "note"}} for h, d in ev["horizons"].items()]).round(4).to_csv(TAB / "test_horizons.csv", index=False)
    vs = FINAL_VAL["evaluation"]["systems"]
    pd.DataFrame([{"system": LABELS[k], "val_auroc": vs[k]["discrimination"]["auroc"], "val_utility": vs[k]["early_warning_utility"]["normalized_utility"],
                   "test_auroc": SYSTEMS[k]["discrimination"]["auroc"], "test_utility": SYSTEMS[k]["early_warning_utility"]["normalized_utility"]} for k in SYSTEMS]).round(4).to_csv(TAB / "val_vs_test.csv", index=False)
    print("tables written")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    fig_pipeline(); fig_decision(); fig_missingness(); fig_utility_curve(); fig_forest()
    frame, cache = load_predictions()
    fig_roc_calibration(frame, cache); fig_risk_coverage(frame, cache)
    info = fig_patient(frame, cache)
    (TAB / "patient_example.json").write_text(json.dumps(info, indent=2))
    tables()


if __name__ == "__main__":
    main()
