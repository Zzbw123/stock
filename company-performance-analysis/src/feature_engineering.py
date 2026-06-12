"""Feature engineering for stock-trend prediction.

The functions in this module only use information available at or before the
current trading day when building features. The future return and direction
columns are labels and must be excluded from model inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PRICE_COLUMN_ALIASES = {
    "date": ["date", "日期", "交易日期"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["volume", "成交量"],
    "amount": ["amount", "成交额"],
    "turnover": ["turnover", "换手率", "换手率%"],
    "pct_change": ["pct_change", "涨跌幅", "涨跌幅%"],
}

LABEL_COLUMNS = ["future_5d_return", "future_5d_direction"]


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    exact = {str(col).strip(): col for col in columns}
    lower = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Chinese/AkShare price fields to canonical English names."""
    out = pd.DataFrame()
    used_columns: set[str] = set()
    for canonical, aliases in PRICE_COLUMN_ALIASES.items():
        col = _find_column(list(df.columns), aliases)
        if col is not None:
            out[canonical] = df[col]
            used_columns.add(col)

    if "date" not in out or "close" not in out:
        raise ValueError("Price data must contain at least date and close columns.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in [c for c in out.columns if c != "date"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Preserve supplemental numeric features such as index returns, valuation
    # ratios and market-cap fields from data/processed/stock_market_features.csv.
    for col in df.columns:
        if col in used_columns or col in {"date", "source", "adjust", "symbol"}:
            continue
        if str(col).lower() in set(PRICE_COLUMN_ALIASES):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            out[str(col)] = numeric
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def load_best_price_data(
    processed_path: str | Path = PROCESSED_DIR / "stock_market_features.csv",
    raw_path: str | Path = RAW_DIR / "000661_akshare_prices.csv",
) -> pd.DataFrame:
    """Load the richest available daily price table from processed/raw files."""
    candidates: list[pd.DataFrame] = []
    fallback_processed = PROCESSED_DIR / "stock_prices.csv"
    for path in [processed_path, fallback_processed, raw_path]:
        path = Path(path)
        if path.exists():
            candidates.append(normalize_price_columns(pd.read_csv(path)))

    if not candidates:
        raise FileNotFoundError("No price file found in data/processed or data/raw.")

    return max(candidates, key=lambda frame: frame.shape[1]).copy()


def _relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_features(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Create market, technical and target columns for LSTM modeling."""
    df = normalize_price_columns(prices).copy()
    close = df["close"]

    df["daily_return"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    for window in [5, 10, 20, 60]:
        df[f"ma{window}"] = close.rolling(window, min_periods=window).mean()

    if "volume" in df:
        df["volume_ma5"] = df["volume"].rolling(5, min_periods=5).mean()
        df["volume_ma20"] = df["volume"].rolling(20, min_periods=20).mean()

    df["volatility_5d"] = df["daily_return"].rolling(5, min_periods=5).std()
    df["volatility_20d"] = df["daily_return"].rolling(20, min_periods=20).std()
    df["rsi14"] = _relative_strength_index(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["boll_mid"] = close.rolling(20, min_periods=20).mean()
    boll_std = close.rolling(20, min_periods=20).std()
    df["boll_upper"] = df["boll_mid"] + 2 * boll_std
    df["boll_lower"] = df["boll_mid"] - 2 * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"].replace(0, np.nan)

    df[f"future_{horizon}d_return"] = close.shift(-horizon) / close - 1
    df[f"future_{horizon}d_direction"] = (df[f"future_{horizon}d_return"] > 0).astype(float)
    return df


def build_stock_features(
    processed_path: str | Path = PROCESSED_DIR / "stock_prices.csv",
    raw_path: str | Path = RAW_DIR / "000661_akshare_prices.csv",
    horizon: int = 5,
) -> pd.DataFrame:
    """Load price data and return an engineered feature table."""
    prices = load_best_price_data(processed_path, raw_path)
    return add_technical_features(prices, horizon=horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stock technical features and prediction labels.")
    parser.add_argument("--processed-price", default=str(PROCESSED_DIR / "stock_prices.csv"))
    parser.add_argument("--raw-price", default=str(RAW_DIR / "000661_akshare_prices.csv"))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output", default=str(PROCESSED_DIR / "stock_features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_stock_features(args.processed_price, args.raw_price, horizon=args.horizon)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved stock features: {output_path} ({features.shape[0]} rows, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
