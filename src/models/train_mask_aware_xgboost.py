"""G1.3: train a mask-aware XGBoost model on predictor-fit patients only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


CANDIDATE_LABS = ('Lactate', 'WBC', 'Creatinine', 'Platelets', 'BUN', 'Glucose')
EXCLUDE = {'patient_id', 'hospital', 'SepsisLabel'}


def select_predictor_rows(data: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest['patient_id'].duplicated().any():
        raise ValueError('Duplicate patient IDs in predictor-fit manifest')
    if data['patient_id'].isna().any() or manifest['patient_id'].isna().any():
        raise ValueError('Missing patient IDs')
    ids = set(manifest['patient_id'].astype(str))
    selected = data.loc[data['patient_id'].astype(str).isin(ids)].copy()
    if set(selected['patient_id'].astype(str)) != ids:
        raise ValueError('Predictor-fit patients missing from processed training data')
    return selected


def mask_training_matrix(
    X: pd.DataFrame,
    patient_ids: pd.Series,
    *,
    seed: int = 42,
    masked_probability: float = 0.5,
) -> pd.DataFrame:
    """Deterministically mask candidate feature families, independently by patient and lab.

    All hourly rows for a patient share a mask to avoid inconsistent temporal states.
    The original unmasked rows are retained separately during training.
    """
    if not 0 <= masked_probability <= 1:
        raise ValueError('masked_probability must be between 0 and 1')
    if len(X) != len(patient_ids):
        raise ValueError('Patient IDs and feature matrix are misaligned')
    features = list(X.columns)
    required = {f'{lab}_{suffix}' for lab in CANDIDATE_LABS for suffix in ('last', 'observed', 'age')}
    if not required.issubset(features):
        raise ValueError(f'Missing lab feature columns: {sorted(required - set(features))}')
    rng = np.random.default_rng(seed)
    unique_ids = sorted(set(patient_ids.astype(str)))
    patient_masks = {
        pid: {lab for lab, selected in zip(CANDIDATE_LABS, rng.random(len(CANDIDATE_LABS)) < masked_probability) if selected}
        for pid in unique_ids
    }
    masked = X.copy()
    # Apply each patient mask to every hourly row, preserving original row order.
    for pid, positions in patient_ids.astype(str).groupby(patient_ids.astype(str), sort=False).indices.items():
        hidden = patient_masks[pid]
        if not hidden:
            continue
        for lab in hidden:
            masked.iloc[positions, masked.columns.get_loc(f'{lab}_last')] = np.nan
            masked.iloc[positions, masked.columns.get_loc(f'{lab}_observed')] = 0.0
            masked.iloc[positions, masked.columns.get_loc(f'{lab}_age')] = np.nan
    return masked


def build_training_examples(data: pd.DataFrame, seed: int = 42, mask_probability: float = 0.5):
    features = [c for c in data.columns if c not in EXCLUDE]
    if 'ICULOS' not in features:
        raise ValueError('ICULOS missing from predictor features')
    X = data.loc[:, features].reset_index(drop=True)
    y = data['SepsisLabel'].astype(int).reset_index(drop=True)
    ids = data['patient_id'].reset_index(drop=True)
    masked = mask_training_matrix(X, ids, seed=seed, masked_probability=mask_probability)
    return pd.concat([X, masked], ignore_index=True), pd.concat([y, y], ignore_index=True), features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask-probability', type=float, default=0.5)
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()
    train_path = Path('data/processed/train.parquet')
    cohort_path = Path('outputs/g1_cohorts/predictor_fit.csv')
    output_dir = Path('outputs/g1_models')
    data = pd.read_parquet(train_path)
    manifest = pd.read_csv(cohort_path)
    fit_data = select_predictor_rows(data, manifest)
    X, y, features = build_training_examples(fit_data, seed=args.seed, mask_probability=args.mask_probability)
    model = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
        eval_metric='logloss', tree_method='hist', random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(X, y, verbose=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / 'mask_aware_xgboost.joblib')
    metadata = {
        'seed': args.seed, 'mask_probability': args.mask_probability,
        'n_estimators': args.n_estimators, 'n_jobs': args.n_jobs,
        'predictor_fit_patients': int(fit_data['patient_id'].nunique()),
        'original_hourly_rows': int(len(fit_data)),
        'training_examples_with_augmentation': int(len(X)),
        'predictor_features': features,
        'calibration_fitted': False,
        'validation_or_test_loaded': False,
        'masking_protocol': 'patient-consistent random subset of six candidate lab triplets; original rows also retained',
    }
    (output_dir / 'training_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in metadata.items() if k != 'predictor_features'}, indent=2))


if __name__ == '__main__':
    main()
