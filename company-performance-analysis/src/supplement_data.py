"""Supplement daily market, index, valuation and disclosure-date data.

The script prefers AkShare public interfaces and degrades gracefully when an
optional endpoint is unavailable. Stock prices are fetched as forward-adjusted
(`qfq`) OHLCV data by default, which is more suitable for return modeling than
raw unadjusted prices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_SYMBOL = "000661"
DEFAULT_START = "20190101"
DEFAULT_END = "20261231"

INDEX_SPECS = {
    "hs300": {"symbol": "000300", "label": "沪深300"},
    "csi_pharma": {"symbol": "000933", "label": "中证医药"},
    "chinext": {"symbol": "399006", "label": "创业板指"},
}


def _require_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AkShare is required for data supplementation. Install it with: pip install akshare") from exc
    return ak


def _to_numeric(frame: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    out = frame.copy()
    for col in out.columns:
        if col not in exclude:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_ohlcv(raw: pd.DataFrame, prefix: str | None = None) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "股票代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = raw.rename(columns=rename_map).copy()
    if "date" not in df:
        raise ValueError("OHLCV data must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _to_numeric(df, exclude={"date", "symbol"})
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if prefix:
        rename = {col: f"{prefix}_{col}" for col in df.columns if col not in {"date", "symbol"}}
        df = df.rename(columns=rename)
    return df


def fetch_stock_ohlcv(symbol: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    ak = _require_akshare()
    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    stock = normalize_ohlcv(raw)
    stock["adjust"] = adjust or "none"
    stock["source"] = "stock_zh_a_hist"
    return stock


def fetch_index_ohlcv(symbol: str, prefix: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _require_akshare()
    raw = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
    index_df = normalize_ohlcv(raw, prefix=prefix)
    index_df[f"{prefix}_source"] = "index_zh_a_hist"
    return index_df


def normalize_valuation(raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "数据日期": "date",
        "当日收盘价": "valuation_close",
        "当日涨跌幅": "valuation_pct_change",
        "总市值": "total_market_cap",
        "流通市值": "float_market_cap",
        "总股本": "total_shares",
        "流通股本": "float_shares",
        "PE(TTM)": "pe_ttm",
        "PE(静)": "pe_static",
        "市净率": "pb",
        "PEG值": "peg",
        "市现率": "pcf",
        "市销率": "ps",
    }
    df = raw.rename(columns=rename_map).copy()
    if "date" not in df:
        raise ValueError("Valuation data must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _to_numeric(df, exclude={"date"})
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_valuation(symbol: str) -> pd.DataFrame:
    ak = _require_akshare()
    raw = ak.stock_value_em(symbol=symbol)
    valuation = normalize_valuation(raw)
    valuation["source"] = "stock_value_em"
    return valuation


def _estimated_disclosure_date(period_date: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(period_date):
        return pd.NaT
    year = int(period_date.year)
    month = int(period_date.month)
    if month == 3:
        return pd.Timestamp(year=year, month=4, day=30)
    if month == 6:
        return pd.Timestamp(year=year, month=8, day=31)
    if month == 9:
        return pd.Timestamp(year=year, month=10, day=31)
    return pd.Timestamp(year=year + 1, month=4, day=30)


def _period_label(period_date: pd.Timestamp) -> str:
    year = int(period_date.year)
    month = int(period_date.month)
    if month == 3:
        return f"{year}一季"
    if month == 6:
        return f"{year}半年报"
    if month == 9:
        return f"{year}三季"
    return f"{year}年报"


def fetch_disclosure_dates(symbol: str, financials_path: Path) -> pd.DataFrame:
    """Fetch or estimate report disclosure dates for current financial periods."""
    if not financials_path.exists():
        return pd.DataFrame()

    financials = pd.read_csv(financials_path)
    period_col = "period_date" if "period_date" in financials.columns else "period"
    periods = pd.to_datetime(financials[period_col], errors="coerce").dropna().drop_duplicates()
    rows: list[dict[str, object]] = []
    ak = _require_akshare()

    for period_date in periods:
        label = _period_label(period_date)
        actual = pd.NaT
        source = "estimated_by_report_type"
        try:
            disclosure = ak.stock_report_disclosure(market="沪深京", period=label)
            hit = disclosure[disclosure["股票代码"].astype(str).str.zfill(6) == symbol]
            if not hit.empty and "实际披露" in hit:
                actual = pd.to_datetime(hit.iloc[0]["实际披露"], errors="coerce")
                if pd.notna(actual):
                    source = "stock_report_disclosure"
        except Exception:
            actual = pd.NaT

        if pd.isna(actual):
            actual = _estimated_disclosure_date(period_date)

        rows.append(
            {
                "period": period_date.strftime("%Y-%m-%d"),
                "year": int(period_date.year),
                "report_type": label,
                "disclosure_date": actual.strftime("%Y-%m-%d") if pd.notna(actual) else np.nan,
                "disclosure_source": source,
            }
        )
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def build_market_feature_table(
    stock: pd.DataFrame,
    index_tables: dict[str, pd.DataFrame],
    valuation: pd.DataFrame | None,
) -> pd.DataFrame:
    market = stock.copy()
    for prefix, index_df in index_tables.items():
        market = market.merge(index_df, on="date", how="left")
        close_col = f"{prefix}_close"
        if close_col in market:
            market[f"{prefix}_daily_return"] = market[close_col].pct_change()
            market[f"{prefix}_return_5d"] = market[close_col].pct_change(5)
            market[f"{prefix}_return_20d"] = market[close_col].pct_change(20)

    if valuation is not None and not valuation.empty:
        market = market.merge(valuation.drop(columns=["source"], errors="ignore"), on="date", how="left")
        valuation_cols = [
            "pe_ttm",
            "pe_static",
            "pb",
            "peg",
            "pcf",
            "ps",
            "total_market_cap",
            "float_market_cap",
        ]
        existing = [col for col in valuation_cols if col in market.columns]
        market[existing] = market[existing].ffill()
    return market.sort_values("date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplement stock, index, valuation and disclosure-date data.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--skip-valuation", action="store_true")
    parser.add_argument("--skip-disclosure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    stock = fetch_stock_ohlcv(args.symbol, args.start_date, args.end_date, args.adjust)
    stock_raw_path = RAW_DIR / f"{args.symbol}_akshare_prices_full_{args.adjust or 'none'}.csv"
    stock.to_csv(stock_raw_path, index=False, encoding="utf-8-sig")
    stock.to_csv(PROCESSED_DIR / "stock_prices.csv", index=False, encoding="utf-8-sig")
    print(f"Saved full stock OHLCV: {stock_raw_path} and data/processed/stock_prices.csv")

    index_tables: dict[str, pd.DataFrame] = {}
    for prefix, spec in INDEX_SPECS.items():
        try:
            index_df = fetch_index_ohlcv(spec["symbol"], prefix, args.start_date, args.end_date)
            index_tables[prefix] = index_df
            index_df.to_csv(RAW_DIR / f"index_{prefix}_{spec['symbol']}.csv", index=False, encoding="utf-8-sig")
            print(f"Saved index {spec['label']}: {len(index_df)} rows")
        except Exception as exc:
            print(f"[WARN] Index fetch failed for {spec['label']} ({spec['symbol']}): {exc}")

    valuation = pd.DataFrame()
    if not args.skip_valuation:
        try:
            valuation = fetch_valuation(args.symbol)
            valuation.to_csv(RAW_DIR / f"{args.symbol}_valuation.csv", index=False, encoding="utf-8-sig")
            print(f"Saved valuation data: {len(valuation)} rows")
        except Exception as exc:
            print(f"[WARN] Valuation fetch failed: {exc}")

    market = build_market_feature_table(stock, index_tables, valuation)
    market_path = PROCESSED_DIR / "stock_market_features.csv"
    market.to_csv(market_path, index=False, encoding="utf-8-sig")
    print(f"Saved merged market feature table: {market_path} ({market.shape[0]} rows, {market.shape[1]} columns)")

    if not args.skip_disclosure:
        disclosure = fetch_disclosure_dates(args.symbol, PROCESSED_DIR / "financial_indicators.csv")
        if not disclosure.empty:
            disclosure_path = PROCESSED_DIR / "financial_disclosure_dates.csv"
            disclosure.to_csv(disclosure_path, index=False, encoding="utf-8-sig")
            print(f"Saved disclosure date table: {disclosure_path} ({len(disclosure)} rows)")


if __name__ == "__main__":
    main()
