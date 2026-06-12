"""Entropy weight method and TOPSIS scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_directional_standardize(
    matrix: pd.DataFrame,
    positive_indicators: list[str],
    negative_indicators: list[str],
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Direction-adjust and min-max standardize indicators to [0, 1]."""
    id_columns = id_columns or ["period", "year"]
    values = matrix.copy()
    indicator_cols = [col for col in values.columns if col not in id_columns]
    standardized = values[id_columns].copy() if id_columns else pd.DataFrame(index=values.index)

    for col in indicator_cols:
        series = pd.to_numeric(values[col], errors="coerce")
        min_val, max_val = series.min(), series.max()
        if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
            standardized[col] = 1.0
            continue
        if col in negative_indicators and col not in positive_indicators:
            standardized[col] = (max_val - series) / (max_val - min_val)
        else:
            standardized[col] = (series - min_val) / (max_val - min_val)
    return standardized


def entropy_weights(standardized: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
    """Calculate entropy weights from a standardized decision matrix."""
    id_columns = id_columns or ["period", "year"]
    indicator_cols = [col for col in standardized.columns if col not in id_columns]
    x = standardized[indicator_cols].fillna(0).clip(lower=0).to_numpy(dtype=float)
    eps = 1e-12
    proportions = x / (x.sum(axis=0, keepdims=True) + eps)
    n = x.shape[0]
    entropy = -(proportions * np.log(proportions + eps)).sum(axis=0) / np.log(max(n, 2))
    diversity = 1 - entropy
    if np.isclose(diversity.sum(), 0):
        weights = np.repeat(1 / len(indicator_cols), len(indicator_cols))
    else:
        weights = diversity / diversity.sum()
    return pd.DataFrame(
        {
            "indicator": indicator_cols,
            "entropy": entropy,
            "diversity": diversity,
            "weight": weights,
        }
    ).sort_values("weight", ascending=False)


def topsis_score(
    standardized: pd.DataFrame,
    weights: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Score alternatives using weighted TOPSIS closeness."""
    id_columns = id_columns or ["period", "year"]
    indicator_cols = [col for col in standardized.columns if col not in id_columns]
    weight_series = weights.set_index("indicator")["weight"].reindex(indicator_cols).fillna(0)
    weighted = standardized[indicator_cols].fillna(0).to_numpy(dtype=float) * weight_series.to_numpy()

    best = weighted.max(axis=0)
    worst = weighted.min(axis=0)
    d_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))
    score = d_worst / (d_best + d_worst + 1e-12)

    result = standardized[id_columns].copy()
    result["topsis_score"] = score
    result["rank"] = result["topsis_score"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("topsis_score", ascending=False).reset_index(drop=True)


def evaluate_performance(
    matrix: pd.DataFrame,
    positive_indicators: list[str],
    negative_indicators: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run standardization, entropy weighting and TOPSIS in one call."""
    id_columns = [col for col in ["period", "year"] if col in matrix.columns]
    standardized = minmax_directional_standardize(
        matrix,
        positive_indicators=positive_indicators,
        negative_indicators=negative_indicators,
        id_columns=id_columns,
    )
    weights = entropy_weights(standardized, id_columns=id_columns)
    scores = topsis_score(standardized, weights, id_columns=id_columns)
    return standardized, weights, scores
