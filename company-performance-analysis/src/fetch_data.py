"""Data loading utilities for Changchun High-Tech performance analysis.

The project is intentionally source-flexible:
- manual CSV/XLSX files are treated as the primary auditable input;
- AkShare is used when available to fetch market prices and common financial
  indicator tables for A-share ticker 000661.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SYMBOL = "000661"
DEFAULT_NAME = "长春高新"


COLUMN_ALIASES = {
    "period": ["period", "date", "报告期", "日期", "截止日期", "统计周期", "年份", "年度"],
    "revenue": ["revenue", "营业收入", "营业总收入", "营业总收入(元)", "营业收入(元)"],
    "net_profit": ["net_profit", "归母净利润", "净利润", "净利润(元)", "归属于母公司股东的净利润"],
    "gross_profit": ["gross_profit", "毛利润", "营业毛利"],
    "total_assets": ["total_assets", "资产总计", "总资产", "资产总额"],
    "total_liabilities": ["total_liabilities", "负债合计", "总负债", "负债总额"],
    "equity": ["equity", "股东权益", "所有者权益", "归属于母公司股东权益合计"],
    "current_assets": ["current_assets", "流动资产", "流动资产合计"],
    "current_liabilities": ["current_liabilities", "流动负债", "流动负债合计"],
    "operating_cash_flow": [
        "operating_cash_flow",
        "经营现金流",
        "经营活动现金流量净额",
        "经营活动产生的现金流量净额",
    ],
    "roe": ["roe", "ROE", "净资产收益率", "净资产收益率(%)"],
    "roa": ["roa", "ROA", "总资产收益率", "总资产报酬率", "总资产报酬率(%)"],
    "gross_margin": ["gross_margin", "毛利率", "销售毛利率", "销售毛利率(%)"],
    "net_margin": ["net_margin", "净利率", "销售净利率", "销售净利率(%)"],
    "asset_liability_ratio": ["asset_liability_ratio", "资产负债率", "资产负债率(%)"],
    "current_ratio": ["current_ratio", "流动比率"],
}


def _parse_number(value: object) -> float:
    """Parse AkShare numeric strings such as 45.32亿, 12.08%, or False."""
    if pd.isna(value) or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"false", "none", "nan", "--", "-"}:
        return np.nan

    multiplier = 1.0
    if "亿" in text:
        multiplier = 100_000_000.0
    elif "万" in text:
        multiplier = 10_000.0

    text = text.replace(",", "").replace("%", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return np.nan
    return float(match.group()) * multiplier


def _read_table(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Only CSV, XLSX and XLS files are supported.")


def _find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized = {str(col).strip(): col for col in columns}
    lower_lookup = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
        if alias.lower() in lower_lookup:
            return lower_lookup[alias.lower()]
    return None


def normalize_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Chinese/English financial column names to canonical names."""
    result = pd.DataFrame()
    for canonical, aliases in COLUMN_ALIASES.items():
        col = _find_column(df.columns, aliases)
        if col is not None:
            result[canonical] = df[col]

    if "period" not in result:
        raise ValueError(
            "Financial table must contain a period/date column, e.g. period, 年份, 报告期 or 日期."
        )

    result["period"] = result["period"].astype(str).str.strip()
    result["period_date"] = pd.to_datetime(result["period"], errors="coerce")
    result["year"] = result["period_date"].dt.year
    result.loc[result["year"].isna(), "year"] = (
        result.loc[result["year"].isna(), "period"].str.extract(r"(\d{4})")[0].astype(float)
    )
    result["year"] = result["year"].astype("Int64")

    for col in result.columns:
        if col not in {"period", "period_date"}:
            result[col] = result[col].map(_parse_number)

    return result.sort_values(["year", "period"]).reset_index(drop=True)


def load_financial_file(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load manually prepared financial data from CSV or Excel."""
    return normalize_financial_columns(_read_table(path, sheet_name=sheet_name))


def load_price_file(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Load manually prepared price data with date/close/volume columns."""
    df = _read_table(path, sheet_name=sheet_name)
    aliases = {
        "date": ["date", "日期", "交易日期"],
        "close": ["close", "收盘", "收盘价"],
        "volume": ["volume", "成交量"],
    }
    out = pd.DataFrame()
    for canonical, names in aliases.items():
        col = _find_column(df.columns, names)
        if col is not None:
            out[canonical] = df[col]
    if "date" not in out:
        raise ValueError("Price table must contain date/日期 column.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["close", "volume"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_financial_indicators_akshare(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    """Fetch common financial indicators from AkShare if it is installed.

    AkShare APIs evolve, so this function tries several public interfaces and
    then normalizes whichever table is available.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AkShare is not installed. Run: pip install akshare") from exc

    attempts = [
        ("stock_financial_analysis_indicator", lambda: ak.stock_financial_analysis_indicator(symbol=symbol)),
        ("stock_financial_abstract_ths", lambda: ak.stock_financial_abstract_ths(symbol=symbol, indicator="按报告期")),
    ]
    errors: list[str] = []
    for source_name, get_data in attempts:
        try:
            df = get_data()
            if isinstance(df, pd.DataFrame) and not df.empty:
                normalized = normalize_financial_columns(df)
                normalized["source"] = source_name
                return normalized
        except Exception as exc:  # noqa: BLE001 - API fallback is intentional.
            errors.append(f"{source_name}: {exc}")
    raise RuntimeError("AkShare financial indicator fetch failed: " + " | ".join(errors))


def _normalize_price_table(raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "成交量": "volume",
        "amount": "volume",
    }
    out = raw.rename(columns=rename_map)
    keep = [col for col in ["date", "close", "volume"] if col in out.columns]
    if "date" not in keep or "close" not in keep:
        raise ValueError("AkShare price table did not include date and close columns.")

    out = out[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["close", "volume"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_price_akshare(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = "20160101",
    end_date: str = "20261231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch historical A-share prices from AkShare."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AkShare is not installed. Run: pip install akshare") from exc

    exchange_symbol = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
    attempts = [
        (
            "stock_zh_a_hist",
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            ),
        ),
        (
            "stock_zh_a_daily",
            lambda: ak.stock_zh_a_daily(
                symbol=exchange_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            ),
        ),
        (
            "stock_zh_a_hist_tx",
            lambda: ak.stock_zh_a_hist_tx(
                symbol=exchange_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            ),
        ),
    ]

    errors: list[str] = []
    for source_name, get_data in attempts:
        try:
            raw = get_data()
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                prices = _normalize_price_table(raw)
                prices["source"] = source_name
                return prices
        except Exception as exc:  # noqa: BLE001 - API fallback is intentional.
            errors.append(f"{source_name}: {exc}")
    raise RuntimeError("AkShare price fetch failed: " + " | ".join(errors))


def load_sample_financials() -> pd.DataFrame:
    """Small teaching dataset for pipeline demos only, not audited financials."""
    rows = [
        ["2019", 7370000000, 1775000000, 14050000000, 2960000000, 11090000000, 1520000000, 15.9, 12.6, 86.0, 24.1, 21.1, 3.7],
        ["2020", 8580000000, 3047000000, 18030000000, 3780000000, 14250000000, 2860000000, 22.9, 16.9, 86.7, 35.5, 21.0, 3.3],
        ["2021", 10747000000, 3757000000, 22360000000, 4920000000, 17440000000, 3560000000, 22.3, 16.8, 87.6, 35.0, 22.0, 3.0],
        ["2022", 12627000000, 4140000000, 27050000000, 5830000000, 21220000000, 3650000000, 20.8, 15.3, 86.8, 32.8, 21.6, 3.1],
        ["2023", 14566000000, 4532000000, 31980000000, 6710000000, 25270000000, 4740000000, 19.3, 14.2, 85.2, 31.1, 21.0, 3.4],
    ]
    columns = [
        "period",
        "revenue",
        "net_profit",
        "total_assets",
        "total_liabilities",
        "equity",
        "operating_cash_flow",
        "roe",
        "roa",
        "gross_margin",
        "net_margin",
        "asset_liability_ratio",
        "current_ratio",
    ]
    return normalize_financial_columns(pd.DataFrame(rows, columns=columns))


def load_sample_prices() -> pd.DataFrame:
    """Annual-ish teaching price series for chart demos only."""
    rows = [
        ["2019-12-31", 120.4, 421000],
        ["2020-12-31", 236.8, 688000],
        ["2021-12-31", 271.5, 533000],
        ["2022-12-31", 165.2, 612000],
        ["2023-12-31", 145.6, 489000],
    ]
    df = pd.DataFrame(rows, columns=["date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df
