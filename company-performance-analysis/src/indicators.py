"""Financial indicator construction for performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_INDICATORS = [
    "revenue",
    "net_profit",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "asset_liability_ratio",
    "current_ratio",
    "operating_cash_flow",
]

POSITIVE_INDICATORS = [
    "revenue",
    "net_profit",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "current_ratio",
    "operating_cash_flow",
    "revenue_growth",
    "net_profit_growth",
    "cash_flow_to_profit",
]

NEGATIVE_INDICATORS = ["asset_liability_ratio"]

INDICATOR_LABELS = {
    "revenue": "营业收入",
    "net_profit": "净利润",
    "roe": "ROE",
    "roa": "ROA",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "asset_liability_ratio": "资产负债率",
    "current_ratio": "流动比率",
    "operating_cash_flow": "经营现金流",
    "revenue_growth": "营收增长率",
    "net_profit_growth": "净利润增长率",
    "cash_flow_to_profit": "现金流/净利润",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_derived_indicators(financials: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing ratios and growth metrics from normalized financials."""
    df = financials.copy()
    df = df.sort_values(["year", "period"]).reset_index(drop=True)

    if "roe" not in df and {"net_profit", "equity"}.issubset(df.columns):
        df["roe"] = _safe_divide(df["net_profit"], df["equity"]) * 100

    if "roa" not in df and {"net_profit", "total_assets"}.issubset(df.columns):
        df["roa"] = _safe_divide(df["net_profit"], df["total_assets"]) * 100

    if "gross_margin" not in df and {"gross_profit", "revenue"}.issubset(df.columns):
        df["gross_margin"] = _safe_divide(df["gross_profit"], df["revenue"]) * 100

    if "net_margin" not in df and {"net_profit", "revenue"}.issubset(df.columns):
        df["net_margin"] = _safe_divide(df["net_profit"], df["revenue"]) * 100

    if "asset_liability_ratio" not in df and {"total_liabilities", "total_assets"}.issubset(df.columns):
        df["asset_liability_ratio"] = _safe_divide(df["total_liabilities"], df["total_assets"]) * 100

    if "current_ratio" not in df and {"current_assets", "current_liabilities"}.issubset(df.columns):
        df["current_ratio"] = _safe_divide(df["current_assets"], df["current_liabilities"])

    if "cash_flow_to_profit" not in df and {"operating_cash_flow", "net_profit"}.issubset(df.columns):
        df["cash_flow_to_profit"] = _safe_divide(df["operating_cash_flow"], df["net_profit"])

    if "revenue" in df:
        df["revenue_growth"] = df["revenue"].pct_change() * 100
    if "net_profit" in df:
        df["net_profit_growth"] = df["net_profit"].pct_change() * 100

    return df


def available_indicator_columns(df: pd.DataFrame, min_non_null: int = 2) -> list[str]:
    """Return indicators that have enough observations for entropy/TOPSIS."""
    candidates = CORE_INDICATORS + ["revenue_growth", "net_profit_growth", "cash_flow_to_profit"]
    return [
        col
        for col in candidates
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= min_non_null
    ]


def build_indicator_matrix(df: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    """Build a clean period-by-indicator matrix."""
    indicators = indicators or available_indicator_columns(df)
    id_cols = [col for col in ["period", "year"] if col in df.columns]
    matrix = df[id_cols + indicators].copy()
    for col in indicators:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return matrix.dropna(subset=indicators, how="all").reset_index(drop=True)
