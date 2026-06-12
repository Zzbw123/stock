"""Command-line pipeline for the Changchun High-Tech performance project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from advanced_visualizations import run_advanced_visualizations
from entropy_topsis import evaluate_performance
from fetch_data import (
    DEFAULT_SYMBOL,
    fetch_financial_indicators_akshare,
    fetch_price_akshare,
    load_financial_file,
    load_price_file,
    load_sample_financials,
    load_sample_prices,
)
from indicators import NEGATIVE_INDICATORS, POSITIVE_INDICATORS, add_derived_indicators, build_indicator_matrix
from pca_robustness import run_pca_robustness
from visualization import (
    plot_price_vs_score,
    plot_radar,
    plot_streamlit_trend_panel,
    plot_topsis_score,
    plot_trends,
    plot_weight_bar,
)
from report_writer import generate_performance_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_XLSX = PROJECT_ROOT / "outputs" / "report_data.xlsx"
REPORT_MD = PROJECT_ROOT / "outputs" / "company_performance_report.md"


def _filter_annual_reports(financials: pd.DataFrame) -> pd.DataFrame:
    """Keep fiscal year-end rows when AkShare returns mixed quarterly data."""
    if "period_date" not in financials:
        return financials
    dates = pd.to_datetime(financials["period_date"], errors="coerce")
    annual = financials.loc[(dates.dt.month == 12) & (dates.dt.day == 31)].copy()
    return annual if not annual.empty else financials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上市公司绩效评价与可视化展示")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="A股股票代码，默认 000661")
    parser.add_argument("--financial-file", help="手动导入财务数据 CSV/XLSX")
    parser.add_argument("--price-file", help="手动导入股价数据 CSV/XLSX")
    parser.add_argument("--use-akshare", action="store_true", help="优先使用 AkShare 获取数据")
    parser.add_argument("--start-date", default="20190101", help="AkShare 股价开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default="20261231", help="AkShare 股价结束日期 YYYYMMDD")
    parser.add_argument("--demo", action="store_true", help="使用教学演示数据跑通流程")
    return parser.parse_args()


def get_financials(args: argparse.Namespace) -> pd.DataFrame:
    if args.financial_file:
        return load_financial_file(args.financial_file)
    if args.use_akshare:
        return fetch_financial_indicators_akshare(args.symbol)
    if args.demo:
        return load_sample_financials()
    sample_path = RAW_DIR / "changchun_gaoxin_demo_financials.csv"
    if sample_path.exists():
        return load_financial_file(sample_path)
    raise RuntimeError("No financial data. Provide --financial-file, --use-akshare or --demo.")


def get_prices(args: argparse.Namespace) -> pd.DataFrame:
    if args.price_file:
        return load_price_file(args.price_file)
    if args.use_akshare:
        return fetch_price_akshare(args.symbol, args.start_date, args.end_date)
    if args.demo:
        return load_sample_prices()
    sample_path = RAW_DIR / "changchun_gaoxin_demo_prices.csv"
    if sample_path.exists():
        return load_price_file(sample_path)
    return pd.DataFrame()


def run_pipeline(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    for stale_figure in FIGURE_DIR.iterdir():
        if stale_figure.is_file() and (
            stale_figure.suffix.lower() in {".png", ".svg", ".pdf"} or stale_figure.name.endswith(".trace.json")
        ):
            stale_figure.unlink()

    raw_financials = get_financials(args)
    prices = get_prices(args)

    if args.use_akshare:
        raw_financials = _filter_annual_reports(raw_financials)
        raw_financials.to_csv(RAW_DIR / f"{args.symbol}_akshare_financials.csv", index=False, encoding="utf-8-sig")
        if not prices.empty:
            prices.to_csv(RAW_DIR / f"{args.symbol}_akshare_prices.csv", index=False, encoding="utf-8-sig")

    financials = add_derived_indicators(raw_financials)
    matrix = build_indicator_matrix(financials)
    standardized, weights, scores = evaluate_performance(
        matrix,
        positive_indicators=POSITIVE_INDICATORS,
        negative_indicators=NEGATIVE_INDICATORS,
    )

    financials.to_csv(PROCESSED_DIR / "financial_indicators.csv", index=False, encoding="utf-8-sig")
    standardized.to_csv(PROCESSED_DIR / "standardized_matrix.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(TABLE_DIR / "entropy_weights.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(TABLE_DIR / "topsis_scores.csv", index=False, encoding="utf-8-sig")
    if not prices.empty:
        prices.to_csv(PROCESSED_DIR / "stock_prices.csv", index=False, encoding="utf-8-sig")

    plot_trends(financials, FIGURE_DIR)
    plot_streamlit_trend_panel(financials, FIGURE_DIR)
    plot_weight_bar(weights, FIGURE_DIR)
    plot_topsis_score(scores, FIGURE_DIR)
    try:
        plot_radar(standardized, FIGURE_DIR)
    except ValueError as exc:
        print(f"[WARN] Radar chart skipped: {exc}")
    plot_price_vs_score(prices, scores, FIGURE_DIR)
    pca_outputs = run_pca_robustness(standardized, scores, FIGURE_DIR, TABLE_DIR, PROJECT_ROOT / "outputs" / "metrics.csv")
    advanced_outputs = run_advanced_visualizations(
        standardized,
        weights,
        scores,
        pca_outputs["loadings"],
        FIGURE_DIR,
        TABLE_DIR,
        PROJECT_ROOT / "outputs" / "metrics.csv",
    )

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        financials.to_excel(writer, sheet_name="financial_indicators", index=False)
        standardized.to_excel(writer, sheet_name="standardized_matrix", index=False)
        weights.to_excel(writer, sheet_name="entropy_weights", index=False)
        scores.to_excel(writer, sheet_name="topsis_scores", index=False)
        pca_outputs["coordinates"].to_excel(writer, sheet_name="pca_coordinates", index=False)
        pca_outputs["loadings"].to_excel(writer, sheet_name="pca_loadings", index=False)
        advanced_outputs["sensitivity"].to_excel(writer, sheet_name="sensitivity", index=False)
        advanced_outputs["contribution"].to_excel(writer, sheet_name="degradation", index=False)
        advanced_outputs["stage_scores"].to_excel(writer, sheet_name="stage_scores", index=False)
        advanced_outputs["rank_curve"].to_excel(writer, sheet_name="rank_curve", index=False)
        advanced_outputs["entropy_pca_compare"].to_excel(writer, sheet_name="entropy_pca", index=False)
        if not prices.empty:
            prices.to_excel(writer, sheet_name="stock_prices", index=False)

    generate_performance_report(financials, prices, weights, scores, REPORT_MD)

    return {
        "financials": financials,
        "prices": prices,
        "standardized": standardized,
        "weights": weights,
        "scores": scores,
    }


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args)
    print("Analysis completed.")
    print(f"Rows: financials={len(outputs['financials'])}, prices={len(outputs['prices'])}")
    print(f"Report workbook: {REPORT_XLSX}")
    print(f"Performance report: {REPORT_MD}")
    print(f"Figures directory: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
