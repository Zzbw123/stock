"""Fetch peer-company market, valuation and financial panel data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import add_technical_features
from fetch_data import fetch_financial_indicators_akshare
from supplement_data import (
    INDEX_SPECS,
    PROCESSED_DIR,
    RAW_DIR,
    _estimated_disclosure_date,
    _period_label,
    _require_akshare,
    build_market_feature_table,
    fetch_index_ohlcv,
    fetch_stock_ohlcv,
    fetch_valuation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PEER_DIR = RAW_DIR / "peers"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_PEER_LIST = RAW_DIR / "peer_list.csv"


def _load_peers(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Peer list not found: {path}")
    peers = pd.read_csv(path, dtype={"symbol": str})
    peers["symbol"] = peers["symbol"].astype(str).str.zfill(6)
    if "name" not in peers:
        peers["name"] = peers["symbol"]
    if "sector" not in peers:
        peers["sector"] = "医药"
    return peers[["symbol", "name", "sector"]].drop_duplicates("symbol")


def _fetch_index_tables(start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    index_tables: dict[str, pd.DataFrame] = {}
    for prefix, spec in INDEX_SPECS.items():
        try:
            index_df = fetch_index_ohlcv(spec["symbol"], prefix, start_date, end_date)
            index_tables[prefix] = index_df
            index_df.to_csv(RAW_DIR / f"index_{prefix}_{spec['symbol']}.csv", index=False, encoding="utf-8-sig")
            print(f"[OK] index {spec['label']}: {len(index_df)} rows")
        except Exception as exc:  # noqa: BLE001 - batch collection should continue.
            print(f"[WARN] index {spec['label']} failed: {exc}")
    return index_tables


def _fetch_disclosure_for_symbol(
    symbol: str,
    name: str,
    financials: pd.DataFrame,
    use_api: bool,
    disclosure_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    period_col = "period_date" if "period_date" in financials.columns else "period"
    periods = pd.to_datetime(financials[period_col], errors="coerce").dropna().drop_duplicates()
    rows: list[dict[str, object]] = []
    ak = _require_akshare() if use_api else None

    for period_date in periods:
        label = _period_label(period_date)
        actual = pd.NaT
        source = "estimated_by_report_type"
        if use_api and ak is not None:
            try:
                if label not in disclosure_cache:
                    disclosure_cache[label] = ak.stock_report_disclosure(market="沪深京", period=label)
                table = disclosure_cache[label]
                hit = table[table["股票代码"].astype(str).str.zfill(6) == symbol]
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
                "symbol": symbol,
                "name": name,
                "period": period_date.strftime("%Y-%m-%d"),
                "year": int(period_date.year),
                "report_type": label,
                "disclosure_date": actual.strftime("%Y-%m-%d") if pd.notna(actual) else np.nan,
                "disclosure_source": source,
            }
        )
    return pd.DataFrame(rows).sort_values(["symbol", "period"]).reset_index(drop=True)


def _prepare_fundamentals(financials: pd.DataFrame, disclosures: pd.DataFrame) -> pd.DataFrame:
    df = financials.copy()
    if "period" not in df and "period_date" in df:
        df["period"] = pd.to_datetime(df["period_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["period"] = pd.to_datetime(df["period"], errors="coerce").dt.strftime("%Y-%m-%d")
    keep = [
        "symbol",
        "name",
        "sector",
        "period",
        "year",
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
    existing = [col for col in keep if col in df.columns]
    df = df[existing].copy()
    disc = disclosures.copy()
    disc["period"] = pd.to_datetime(disc["period"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.merge(
        disc[["symbol", "period", "disclosure_date", "disclosure_source", "report_type"]],
        on=["symbol", "period"],
        how="left",
    )


def build_panel_model_data(
    market_features: pd.DataFrame,
    financials: pd.DataFrame,
    disclosures: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    panel_parts: list[pd.DataFrame] = []
    fundamentals = _prepare_fundamentals(financials, disclosures)
    fundamentals["disclosure_date"] = pd.to_datetime(fundamentals["disclosure_date"], errors="coerce")

    for symbol, group in market_features.groupby("symbol", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        identity = group[["symbol", "name", "sector"]].copy()
        feature_input = group.drop(columns=["symbol", "name", "sector"], errors="ignore")
        features = add_technical_features(feature_input, horizon=horizon)
        features["date"] = pd.to_datetime(features["date"], errors="coerce").astype("datetime64[ns]")
        features[["symbol", "name", "sector"]] = identity.loc[features.index, ["symbol", "name", "sector"]].to_numpy()

        symbol_fundamentals = fundamentals[fundamentals["symbol"] == symbol].dropna(subset=["disclosure_date"])
        if not symbol_fundamentals.empty:
            symbol_fundamentals = symbol_fundamentals.copy()
            symbol_fundamentals["disclosure_date"] = pd.to_datetime(
                symbol_fundamentals["disclosure_date"], errors="coerce"
            ).astype("datetime64[ns]")
            features = pd.merge_asof(
                features.sort_values("date"),
                symbol_fundamentals.sort_values("disclosure_date").drop(columns=["name", "sector"], errors="ignore"),
                left_on="date",
                right_on="disclosure_date",
                by="symbol",
                direction="backward",
            )
            features["fundamental_mapping_method"] = "disclosure_date_asof"
        else:
            features["fundamental_mapping_method"] = "missing_fundamentals"
        panel_parts.append(features)

    return pd.concat(panel_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def fetch_peer_panel(
    peer_list: str | Path,
    start_date: str,
    end_date: str,
    adjust: str,
    horizon: int,
    use_disclosure_api: bool,
    limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    PEER_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    peers = _load_peers(peer_list)
    if limit is not None:
        peers = peers.head(limit)
    peers.to_csv(PROCESSED_DIR / "peer_list.csv", index=False, encoding="utf-8-sig")

    index_tables = _fetch_index_tables(start_date, end_date)
    market_tables: list[pd.DataFrame] = []
    financial_tables: list[pd.DataFrame] = []
    disclosure_tables: list[pd.DataFrame] = []
    statuses: list[dict[str, object]] = []
    disclosure_cache: dict[str, pd.DataFrame] = {}

    for row in peers.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        name = str(row.name)
        sector = str(row.sector)
        status = {"symbol": symbol, "name": name, "sector": sector}
        print(f"\n=== {symbol} {name} ===")

        stock = pd.DataFrame()
        valuation = pd.DataFrame()
        try:
            stock = fetch_stock_ohlcv(symbol, start_date, end_date, adjust)
            stock = stock.drop(columns=["symbol", "name", "sector"], errors="ignore")
            stock_to_save = stock.copy()
            stock_to_save[["symbol", "name", "sector"]] = [symbol, name, sector]
            stock_to_save.to_csv(PEER_DIR / f"{symbol}_prices_full_{adjust or 'none'}.csv", index=False, encoding="utf-8-sig")
            status["price_rows"] = len(stock)
            print(f"[OK] price rows: {len(stock)}")
        except Exception as exc:
            status["price_error"] = str(exc)
            print(f"[WARN] price failed: {exc}")

        try:
            valuation = fetch_valuation(symbol)
            valuation = valuation.drop(columns=["symbol", "name", "sector"], errors="ignore")
            valuation_to_save = valuation.copy()
            valuation_to_save[["symbol", "name", "sector"]] = [symbol, name, sector]
            valuation_to_save.to_csv(PEER_DIR / f"{symbol}_valuation.csv", index=False, encoding="utf-8-sig")
            status["valuation_rows"] = len(valuation)
            print(f"[OK] valuation rows: {len(valuation)}")
        except Exception as exc:
            status["valuation_error"] = str(exc)
            print(f"[WARN] valuation failed: {exc}")

        if not stock.empty:
            market = build_market_feature_table(stock, index_tables, valuation)
            market[["symbol", "name", "sector"]] = [symbol, name, sector]
            market_tables.append(market)

        financials = pd.DataFrame()
        try:
            financials = fetch_financial_indicators_akshare(symbol)
            financials[["symbol", "name", "sector"]] = [symbol, name, sector]
            financials.to_csv(PEER_DIR / f"{symbol}_financial_indicators.csv", index=False, encoding="utf-8-sig")
            financial_tables.append(financials)
            status["financial_rows"] = len(financials)
            print(f"[OK] financial rows: {len(financials)}")
        except Exception as exc:
            status["financial_error"] = str(exc)
            print(f"[WARN] financial failed: {exc}")

        if not financials.empty:
            disclosure = _fetch_disclosure_for_symbol(
                symbol,
                name,
                financials,
                use_api=use_disclosure_api,
                disclosure_cache=disclosure_cache,
            )
            disclosure_tables.append(disclosure)
            disclosure.to_csv(PEER_DIR / f"{symbol}_disclosure_dates.csv", index=False, encoding="utf-8-sig")
            status["disclosure_rows"] = len(disclosure)
        statuses.append(status)

    market_features = pd.concat(market_tables, ignore_index=True) if market_tables else pd.DataFrame()
    financial_panel = pd.concat(financial_tables, ignore_index=True) if financial_tables else pd.DataFrame()
    disclosure_panel = pd.concat(disclosure_tables, ignore_index=True) if disclosure_tables else pd.DataFrame()
    status_df = pd.DataFrame(statuses)

    if not market_features.empty:
        market_features.to_csv(PROCESSED_DIR / "peer_stock_market_features.csv", index=False, encoding="utf-8-sig")
    if not financial_panel.empty:
        financial_panel.to_csv(PROCESSED_DIR / "peer_financial_indicators.csv", index=False, encoding="utf-8-sig")
    if not disclosure_panel.empty:
        disclosure_panel.to_csv(PROCESSED_DIR / "peer_financial_disclosure_dates.csv", index=False, encoding="utf-8-sig")
    status_df.to_csv(TABLE_DIR / "peer_data_fetch_status.csv", index=False, encoding="utf-8-sig")

    panel_model_data = pd.DataFrame()
    if not market_features.empty and not financial_panel.empty and not disclosure_panel.empty:
        panel_model_data = build_panel_model_data(market_features, financial_panel, disclosure_panel, horizon=horizon)
        panel_model_data.to_csv(PROCESSED_DIR / "panel_model_data.csv", index=False, encoding="utf-8-sig")

    return {
        "market_features": market_features,
        "financial_panel": financial_panel,
        "disclosure_panel": disclosure_panel,
        "panel_model_data": panel_model_data,
        "status": status_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch peer-company panel data for model expansion.")
    parser.add_argument("--peer-list", default=str(DEFAULT_PEER_LIST))
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default="20261231")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--use-disclosure-api", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = fetch_peer_panel(
        peer_list=args.peer_list,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        horizon=args.horizon,
        use_disclosure_api=args.use_disclosure_api,
        limit=args.limit,
    )
    print("\nCompleted peer panel fetch.")
    for name, df in outputs.items():
        print(f"{name}: {df.shape}")


if __name__ == "__main__":
    main()
