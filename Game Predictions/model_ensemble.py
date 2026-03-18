from __future__ import annotations

from pathlib import Path
import numpy as np
import joblib

import data_processing as dp

CLASS_WEIGHTS = {
    "lgbm": 0.35,
    "xgb": 0.25,
    "logreg": 0.40,
}

REG_WEIGHTS = {
    "lgbm": 0.05,
    "logreg": 0.95,
}

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
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if booster and booster.feature_names:
            return list(booster.feature_names)
    raise ValueError("Unable to infer feature names from model.")


def load_models(model_dir: Path | str | None = None) -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = _resolve_model_dir(model_dir)
    winner_lgbm = joblib.load(model_path / "lgbm_winner_model.joblib")
    winner_xgb = joblib.load(model_path / "xgb_winner_model.joblib")
    winner_logreg = joblib.load(model_path / "logreg_winner_model.joblib")

    spread_lgbm = joblib.load(model_path / "lgbm_spread_model.joblib")
    spread_xgb = joblib.load(model_path / "xgb_spread_model.joblib")
    spread_logreg = joblib.load(model_path / "logreg_spread_model.joblib")

    winner_features = _get_feature_names(winner_lgbm)
    spread_features = _get_feature_names(spread_lgbm)

    _MODEL_CACHE = {
        "winner": {
            "lgbm": winner_lgbm,
            "xgb": winner_xgb,
            "logreg": winner_logreg,
            "features": winner_features,
        },
        "spread": {
            "lgbm": spread_lgbm,
            "xgb": spread_xgb,
            "logreg": spread_logreg,
            "features": spread_features,
        },
    }
    return _MODEL_CACHE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _predict_proba_like(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return _sigmoid(model.decision_function(X))
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def _weighted_sum(preds: dict, weights: dict) -> np.ndarray:
    total = np.zeros_like(next(iter(preds.values())))
    for key, weight in weights.items():
        total += weight * preds[key]
    return total


def predict_classification_ensemble(
    feature_df,
    model_bundle: dict,
    weights: dict | None = None
) -> float:
    weights = weights or CLASS_WEIGHTS
    winner_models = model_bundle["winner"]
    features = winner_models["features"]
    X = dp.align_features_for_model(feature_df, features)

    preds = {
        "lgbm": _predict_proba_like(winner_models["lgbm"], X),
        "xgb": _predict_proba_like(winner_models["xgb"], X),
        "logreg": _predict_proba_like(winner_models["logreg"], X),
    }
    return float(_weighted_sum(preds, weights)[0])


def predict_regression_ensemble(
    feature_df,
    model_bundle: dict,
    weights: dict | None = None
) -> float:
    weights = weights or REG_WEIGHTS
    spread_models = model_bundle["spread"]
    features = spread_models["features"]
    X = dp.align_features_for_model(feature_df, features)

    preds = {
        "lgbm": np.asarray(spread_models["lgbm"].predict(X), dtype=float),
        "logreg": np.asarray(spread_models["logreg"].predict(X), dtype=float),
    }
    return float(_weighted_sum(preds, weights)[0])


def predict_ensemble(feature_df, model_bundle: dict | None = None) -> tuple[float, float]:
    models = model_bundle or load_models()
    win_prob = predict_classification_ensemble(feature_df, models)
    spread_pred = predict_regression_ensemble(feature_df, models)
    return win_prob, spread_pred


def round_to_half(value: float) -> float:
    return round(value * 2) / 2


def format_spread_range(spread_pred: float) -> str:
    spread_abs = round_to_half(abs(float(spread_pred)))
    if spread_abs.is_integer():
        return f"±{int(spread_abs)}"
    return f"±{spread_abs:.1f}"
