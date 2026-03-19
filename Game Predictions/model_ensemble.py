from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data_processing as dp

META_FEATURE_COLUMNS = ['winner_model_proba', 'spread_model_pred']

_MODEL_CACHE = None


def _resolve_model_dir(model_dir: Path | str | None) -> Path:
    if model_dir is None:
        return Path(__file__).resolve().parent / "models"
    return Path(model_dir)


def _get_feature_names(model) -> list[str]:
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("Unable to infer feature names from model.")


def load_models(model_dir: Path | str | None = None) -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = _resolve_model_dir(model_dir)
    winner_model = joblib.load(model_path / "lgbm_winner_model.joblib")
    spread_model = joblib.load(model_path / "lgbm_spread_model.joblib")
    meta_model = joblib.load(model_path / "meta_model.joblib")

    _MODEL_CACHE = {
        "winner": {
            "model": winner_model,
            "features": _get_feature_names(winner_model),
        },
        "spread": {
            "model": spread_model,
            "features": _get_feature_names(spread_model),
        },
        "meta": {
            "model": meta_model,
            "features": META_FEATURE_COLUMNS,
        },
    }
    return _MODEL_CACHE


def apply_prediction_adjustments(feature_df: pd.DataFrame) -> pd.DataFrame:
    adjusted = feature_df.copy()
    if 'team_home_away' not in adjusted.columns:
        return adjusted

    home_mask = adjusted['team_home_away'] == 1
    away_mask = adjusted['team_home_away'] == 0

    if 'margin_estimate' in adjusted.columns:
        adjusted.loc[home_mask, 'margin_estimate'] = adjusted.loc[home_mask, 'margin_estimate'] + 7.00
        adjusted.loc[away_mask, 'margin_estimate'] = adjusted.loc[away_mask, 'margin_estimate'] - 7.00

    if 'point_differential_avg_diff' in adjusted.columns and 'margin_estimate' in adjusted.columns:
        adjusted.loc[home_mask, 'point_differential_avg_diff'] = adjusted.loc[home_mask, 'margin_estimate'] + 6.90
        adjusted.loc[away_mask, 'point_differential_avg_diff'] = adjusted.loc[away_mask, 'margin_estimate'] - 6.90

    return adjusted


def predict_base_models(feature_df: pd.DataFrame, model_bundle: dict | None = None) -> tuple[float, float]:
    models = model_bundle or load_models()
    adjusted = apply_prediction_adjustments(feature_df)

    winner_models = models["winner"]
    winner_X = dp.align_features_for_model(adjusted, winner_models["features"])
    winner_prob = float(winner_models["model"].predict_proba(winner_X)[:, 1][0])

    spread_models = models["spread"]
    spread_X = dp.align_features_for_model(adjusted, spread_models["features"])
    spread_pred = float(np.asarray(spread_models["model"].predict(spread_X), dtype=float)[0])

    return winner_prob, spread_pred


def predict_meta_ensemble(feature_df: pd.DataFrame, model_bundle: dict | None = None) -> tuple[float, float]:
    models = model_bundle or load_models()
    winner_prob, spread_pred = predict_base_models(feature_df, models)

    meta_input = pd.DataFrame(
        [
            {
                'winner_model_proba': winner_prob,
                'spread_model_pred': spread_pred,
            }
        ]
    )
    meta_X = dp.align_features_for_model(meta_input, models["meta"]["features"])
    meta_prob = float(models["meta"]["model"].predict_proba(meta_X)[:, 1][0])
    return meta_prob, spread_pred


def predict_ensemble(feature_df, model_bundle: dict | None = None) -> tuple[float, float]:
    return predict_meta_ensemble(feature_df, model_bundle)


def round_to_half(value: float) -> float:
    return round(value * 2) / 2


def format_spread_range(spread_pred: float) -> str:
    spread_abs = round_to_half(abs(float(spread_pred)))
    if spread_abs.is_integer():
        return f"+/-{int(spread_abs)}"
    return f"+/-{spread_abs:.1f}"
