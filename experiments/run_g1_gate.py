"""G1.5: retrospective information-revelation gate on development validation data.

Run from repository root. Never reads held-out test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.g1_headroom import render_report, run_headroom

VALIDATION = Path('data/processed/val.parquet')
MODEL = Path('outputs/g1_models/mask_aware_xgboost.joblib')
CALIBRATOR = Path('outputs/g1_models/mask_aware_sigmoid_calibrator.joblib')
OUTPUT = Path('experiments/results/g1_headroom')


def select_development_patients(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0:
        raise ValueError('sample size must be positive')
    ids = df['patient_id'].drop_duplicates()
    if size >= len(ids):
        return df.copy()
    selected = set(ids.sample(n=size, random_state=seed).astype(str))
    return df[df['patient_id'].astype(str).isin(selected)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--sample-size', type=int, default=200)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')

    # Deliberately fixed validation path: do not accept arbitrary --data inputs.
    hourly = pd.read_parquet(VALIDATION)
    required = {'patient_id', 'hospital', 'ICULOS', 'SepsisLabel'}
    if not required.issubset(hourly.columns):
        raise ValueError(f'Missing required columns: {sorted(required - set(hourly.columns))}')
    if args.development:
        hourly = select_development_patients(hourly, args.sample_size, args.seed)

    model = joblib.load(MODEL)
    calibrator = joblib.load(CALIBRATOR)
    if {'patient_id', 'hospital', 'SepsisLabel'} & set(model.feature_names_in_):
        raise ValueError('Model contains forbidden predictors')
    result = run_headroom(hourly, model, calibrator, n_resamples=args.bootstrap, seed=args.seed)
    result['evaluation'] = {
        'split': 'development validation',
        'development_sample': bool(args.development),
        'distinct_patients': int(hourly['patient_id'].nunique()),
        'model': str(MODEL),
        'calibrator': str(CALIBRATOR),
        'held_out_test_accessed': False,
        'primary_comparison': 'pooled_all: none_hidden vs all_hidden',
        'primary_metric': 'auprc',
        'note': 'Previously inspected validation split; results are exploratory.',
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = 'development' if args.development else 'validation'
    json_path = OUTPUT / f'headroom_{tag}.json'
    md_path = OUTPUT / f'headroom_{tag}.md'
    json_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding='utf-8')
    md_path.write_text(render_report(result, tag), encoding='utf-8')
    print(json.dumps({'report': str(md_path), 'json': str(json_path),
                      'patients': result['evaluation']['distinct_patients']}, indent=2))


if __name__ == '__main__':
    main()
