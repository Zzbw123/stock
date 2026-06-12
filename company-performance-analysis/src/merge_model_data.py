"""Merge market features, financial indicators and TOPSIS scores for LSTM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_engineering import build_stock_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

DEFAULT_FINANCIAL_COLUMNS = [
    "revenue",
    "net_profit",
    "roe",
    "gross_margin",
    "net_margin",
    "asset_liability_ratio",
    "current_ratio",
    "revenue_growth",
    "net_profit_growth",
]


def _read_required_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def _prepare_financials(financials: pd.DataFrame) -> pd.DataFrame:
    df = financials.copy()
    if "period" not in df and "period_date" in df:
        df["period"] = pd.to_datetime(df["period_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "year" not in df:
        if "period_date" in df:
            df["year"] = pd.to_datetime(df["period_date"], errors="coerce").dt.year
        elif "period" in df:
            df["year"] = pd.to_datetime(df["period"], errors="coerce").dt.year
    if "year" not in df:
        raise ValueError("Financial indicators must contain year or a parsable period column.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    keep = [col for col in ["period", "year"] if col in df.columns] + [
        col for col in DEFAULT_FINANCIAL_COLUMNS if col in df.columns
    ]
    df = df[keep].dropna(subset=["year"]).copy()
    return df.groupby(["period", "year"], as_index=False).last()


def _prepare_topsis(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    if "period" not in df and "year" in df:
        df["period"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64").astype(str) + "-12-31"
    if "year" not in df:
        if "period" in df:
            df["year"] = pd.to_datetime(df["period"], errors="coerce").dt.year
        else:
            raise ValueError("TOPSIS scores must contain year or period.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    keep = [col for col in ["period", "year"] if col in df.columns] + [
        col for col in ["topsis_score", "rank"] if col in df.columns
    ]
    return df[keep].dropna(subset=["year"]).groupby(["period", "year"], as_index=False).last()


def _prepare_disclosure(disclosure: pd.DataFrame) -> pd.DataFrame:
    df = disclosure.copy()
    if "period" not in df or "disclosure_date" not in df:
        raise ValueError("Disclosure table must contain period and disclosure_date columns.")
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")
    keep = [col for col in ["period", "disclosure_date", "disclosure_source", "report_type"] if col in df.columns]
    return df[keep].dropna(subset=["period", "disclosure_date"]).sort_values("disclosure_date")


def build_lstm_model_data(
    stock_prices_path: str | Path = PROCESSED_DIR / "stock_market_features.csv",
    raw_prices_path: str | Path = PROJECT_ROOT / "data" / "raw" / "000661_akshare_prices.csv",
    financials_path: str | Path = PROCESSED_DIR / "financial_indicators.csv",
    topsis_path: str | Path = TABLE_DIR / "topsis_scores.csv",
    disclosure_path: str | Path = PROCESSED_DIR / "financial_disclosure_dates.csv",
    output_path: str | Path = PROCESSED_DIR / "lstm_model_data.csv",
    horizon: int = 5,
) -> pd.DataFrame:
    """Build and save the daily model table.

    Financial and TOPSIS fields are mapped by fiscal year because exact report
    disclosure dates are available. If not, the function falls back to a
    year-based approximation and marks this limitation in the console output.
    """
    stock_features = build_stock_features(stock_prices_path, raw_prices_path, horizon=horizon)
    stock_features["date"] = pd.to_datetime(stock_features["date"], errors="coerce")
    stock_features["trade_year"] = stock_features["date"].dt.year.astype("Int64")

    financials = _prepare_financials(_read_required_csv(financials_path, "Financial indicators"))
    topsis = _prepare_topsis(_read_required_csv(topsis_path, "TOPSIS scores"))
    fundamentals = financials.merge(topsis, on=["period", "year"], how="left")

    disclosure_path = Path(disclosure_path)
    if disclosure_path.exists():
        disclosure = _prepare_disclosure(pd.read_csv(disclosure_path))
        fundamentals = fundamentals.merge(disclosure, on="period", how="left")
        fundamentals = fundamentals.dropna(subset=["disclosure_date"]).sort_values("disclosure_date")
        merged = pd.merge_asof(
            stock_features.sort_values("date"),
            fundamentals.sort_values("disclosure_date"),
            left_on="date",
            right_on="disclosure_date",
            direction="backward",
        )
        merged["fundamental_mapping_method"] = "disclosure_date_asof"
    else:
        merged = stock_features.merge(financials, left_on="trade_year", right_on="year", how="left").merge(
            topsis,
            on=["period", "year"],
            how="left",
        )
        merged["fundamental_mapping_method"] = "year_approximation"
    merged = merged.sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LSTM modeling dataset.")
    parser.add_argument("--stock-prices", default=str(PROCESSED_DIR / "stock_market_features.csv"))
    parser.add_argument("--raw-prices", default=str(PROJECT_ROOT / "data" / "raw" / "000661_akshare_prices.csv"))
    parser.add_argument("--financials", default=str(PROCESSED_DIR / "financial_indicators.csv"))
    parser.add_argument("--topsis", default=str(TABLE_DIR / "topsis_scores.csv"))
    parser.add_argument("--disclosure", default=str(PROCESSED_DIR / "financial_disclosure_dates.csv"))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_lstm_model_data(
        stock_prices_path=args.stock_prices,
        raw_prices_path=args.raw_prices,
        financials_path=args.financials,
        topsis_path=args.topsis,
        disclosure_path=args.disclosure,
        output_path=args.output,
        horizon=args.horizon,
    )
    print(f"Saved LSTM model data: {args.output} ({data.shape[0]} rows, {data.shape[1]} columns)")
    method = data["fundamental_mapping_method"].dropna().iloc[-1] if "fundamental_mapping_method" in data else "unknown"
    print(f"Financial/TOPSIS mapping method: {method}")


if __name__ == "__main__":
    main()
