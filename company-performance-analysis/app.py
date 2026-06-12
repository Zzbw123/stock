"""Streamlit dashboard for listed-company performance evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from entropy_topsis import evaluate_performance  # noqa: E402
from fetch_data import load_financial_file  # noqa: E402
from indicators import NEGATIVE_INDICATORS, POSITIVE_INDICATORS, add_derived_indicators, build_indicator_matrix  # noqa: E402


st.set_page_config(page_title="长春高新经营绩效评价", layout="wide")
st.title("长春高新经营绩效评价与股价趋势预测")

DEFAULT_FINANCIALS = ROOT / "data" / "processed" / "financial_indicators.csv"
LSTM_METRICS = ROOT / "outputs" / "tables" / "lstm_metrics.csv"
LSTM_PREDICTIONS = ROOT / "outputs" / "tables" / "lstm_predictions.csv"
ROLLING_METRICS = ROOT / "outputs" / "tables" / "lstm_rolling_metrics.csv"
ROLLING_PREDICTIONS = ROOT / "outputs" / "tables" / "lstm_rolling_predictions.csv"
ROLLING_SIGNALS = ROOT / "outputs" / "tables" / "rolling_prediction_signals.csv"
LATEST_SIGNAL = ROOT / "outputs" / "tables" / "rolling_latest_signal.csv"
SHAP_IMPORTANCE = ROOT / "outputs" / "tables" / "shap_importance.csv"
SHAP_METRICS = ROOT / "outputs" / "tables" / "shap_proxy_metrics.csv"
PANEL_METRICS = ROOT / "outputs" / "tables" / "panel_model_metrics.csv"
PANEL_PREDICTIONS = ROOT / "outputs" / "tables" / "panel_model_predictions.csv"
PANEL_IMPORTANCE = ROOT / "outputs" / "tables" / "panel_model_feature_importance.csv"
DUAL_BRANCH_METRICS = ROOT / "outputs" / "tables" / "dual_branch_metrics.csv"
DUAL_BRANCH_PREDICTIONS = ROOT / "outputs" / "tables" / "dual_branch_predictions.csv"
CROSS_SECTIONAL_METRICS = ROOT / "outputs" / "tables" / "cross_sectional_metrics.csv"
CROSS_SECTIONAL_BACKTEST = ROOT / "outputs" / "tables" / "cross_sectional_backtest.csv"
CROSS_SECTIONAL_DETAIL = ROOT / "outputs" / "tables" / "cross_sectional_selection_detail.csv"
WALK_FORWARD_TOPK_METRICS = ROOT / "outputs" / "tables" / "walk_forward_topk_metrics.csv"
WALK_FORWARD_YEARLY_METRICS = ROOT / "outputs" / "tables" / "walk_forward_topk_yearly_metrics.csv"
WALK_FORWARD_FOLD_METRICS = ROOT / "outputs" / "tables" / "walk_forward_fold_metrics.csv"
WALK_FORWARD_TOPK_BACKTEST = ROOT / "outputs" / "tables" / "walk_forward_topk_backtest.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"


@st.cache_data
def _load_uploaded(file) -> pd.DataFrame:
    tmp = ROOT / "data" / "raw" / f"_streamlit_upload_{file.name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(file.getbuffer())
    return load_financial_file(tmp)


@st.cache_data
def _load_generated_financials() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_FINANCIALS)


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


uploaded = st.sidebar.file_uploader("导入财务数据 CSV/XLSX", type=["csv", "xlsx", "xls"])
use_generated = st.sidebar.checkbox(
    "使用已生成的真实数据",
    value=uploaded is None and DEFAULT_FINANCIALS.exists(),
    disabled=not DEFAULT_FINANCIALS.exists(),
)

if uploaded is not None:
    financials = _load_uploaded(uploaded)
elif use_generated:
    financials = _load_generated_financials()
else:
    st.info("请先运行真实数据生成流程，或导入财务数据 CSV/XLSX。")
    st.stop()

financials = add_derived_indicators(financials)
matrix = build_indicator_matrix(financials)
standardized, weights, scores = evaluate_performance(
    matrix,
    positive_indicators=POSITIVE_INDICATORS,
    negative_indicators=NEGATIVE_INDICATORS,
)

metric_cols = st.columns(4)
latest = scores.sort_values("year").tail(1).iloc[0]
metric_cols[0].metric("最新期间", int(latest["year"]) if pd.notna(latest["year"]) else latest["period"])
metric_cols[1].metric("TOPSIS 得分", f"{latest['topsis_score']:.3f}")
metric_cols[2].metric("绩效排名", int(latest["rank"]))
metric_cols[3].metric("评价指标数", len([c for c in matrix.columns if c not in {"period", "year"}]))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["经营趋势", "权重与排名", "LSTM预测", "面板模型", "数据表"])

with tab1:
    left, right = st.columns(2)
    for col, label in [
        ("revenue", "营业收入"),
        ("net_profit", "净利润"),
        ("roe", "ROE"),
        ("asset_liability_ratio", "资产负债率"),
        ("operating_cash_flow", "经营现金流"),
    ]:
        if col in financials:
            target = left if col in {"revenue", "roe", "operating_cash_flow"} else right
            target.line_chart(financials.set_index("year")[[col]].rename(columns={col: label}))

with tab2:
    c1, c2 = st.columns(2)
    c1.subheader("熵权法权重")
    c1.bar_chart(weights.set_index("indicator")["weight"])
    c2.subheader("TOPSIS 得分")
    c2.line_chart(scores.sort_values("year").set_index("year")["topsis_score"])
    st.dataframe(scores.sort_values("rank"), use_container_width=True)

with tab3:
    if not LSTM_METRICS.exists() or not LSTM_PREDICTIONS.exists():
        st.info(
            "尚未生成 LSTM 结果，请先运行：\n\n"
            "python src/merge_model_data.py\n\n"
            "python src/lstm_predict.py --task classification --window 20 --horizon 5 --model-type both"
        )
    else:
        metrics = _load_csv(LSTM_METRICS)
        predictions = _load_csv(LSTM_PREDICTIONS)

        st.subheader("模型指标对比")
        st.dataframe(metrics, use_container_width=True)
        core_metrics = metrics[metrics["metric"].isin(["Accuracy", "Precision", "Recall", "F1-score", "RMSE", "MAE"])]
        if not core_metrics.empty:
            st.bar_chart(core_metrics.pivot_table(index="model", columns="metric", values="value", aggfunc="first"))

        st.subheader("预测与回测图")
        figure_paths = [
            FIGURE_DIR / "lstm_prediction.png",
            FIGURE_DIR / "lstm_strategy_return.png",
            FIGURE_DIR / "lstm_confusion_matrix.png",
            FIGURE_DIR / "lstm_price_direction.png",
            FIGURE_DIR / "lstm_metrics_comparison.png",
        ]
        cols = st.columns(2)
        for idx, path in enumerate(figure_paths):
            if path.exists():
                cols[idx % 2].image(str(path), use_container_width=True)

        st.subheader("预测结果明细")
        st.dataframe(predictions, use_container_width=True)

        st.subheader("滚动窗口验证")
        if ROLLING_METRICS.exists():
            rolling_metrics = _load_csv(ROLLING_METRICS)
            st.dataframe(rolling_metrics, use_container_width=True)
            rolling_figures = [
                FIGURE_DIR / "lstm_rolling_f1.png",
                FIGURE_DIR / "lstm_rolling_strategy_return.png",
            ]
            cols = st.columns(2)
            for idx, path in enumerate(rolling_figures):
                if path.exists():
                    cols[idx % 2].image(str(path), use_container_width=True)
            if ROLLING_PREDICTIONS.exists():
                with st.expander("查看滚动验证预测明细"):
                    st.dataframe(_load_csv(ROLLING_PREDICTIONS), use_container_width=True)
        else:
            st.info(
                "尚未生成滚动验证结果，请运行：\n\n"
                "python src/rolling_validation.py --epochs 20 --first-test-year 2024 --transaction-cost 0.001"
            )

        st.subheader("滚动预测信号")
        if LATEST_SIGNAL.exists():
            st.dataframe(_load_csv(LATEST_SIGNAL), use_container_width=True)
        if ROLLING_SIGNALS.exists():
            signal_figures = [
                FIGURE_DIR / "rolling_prediction_signals_base.png",
                FIGURE_DIR / "rolling_prediction_signals_fusion.png",
            ]
            cols = st.columns(2)
            for idx, path in enumerate(signal_figures):
                if path.exists():
                    cols[idx % 2].image(str(path), use_container_width=True)
            with st.expander("查看滚动预测信号明细"):
                st.dataframe(_load_csv(ROLLING_SIGNALS), use_container_width=True)
        else:
            st.info(
                "尚未生成滚动预测信号，请运行：\n\n"
                "python src/rolling_prediction.py --epochs 20 --first-test-year 2024 --transaction-cost 0.001"
            )

        st.subheader("SHAP特征贡献")
        if SHAP_IMPORTANCE.exists():
            if SHAP_METRICS.exists():
                st.dataframe(_load_csv(SHAP_METRICS), use_container_width=True)
            st.dataframe(_load_csv(SHAP_IMPORTANCE).head(30), use_container_width=True)
            shap_figures = [
                FIGURE_DIR / "shap_importance_bar.png",
                FIGURE_DIR / "shap_summary.png",
                FIGURE_DIR / "shap_dependence_topsis_score.png",
                FIGURE_DIR / "shap_dependence_pe_ttm.png",
                FIGURE_DIR / "shap_dependence_pb.png",
            ]
            cols = st.columns(2)
            for idx, path in enumerate(shap_figures):
                if path.exists():
                    cols[idx % 2].image(str(path), use_container_width=True)
        else:
            st.info("尚未生成 SHAP 结果，请运行：\n\npython src/shap_analysis.py --model-type fusion --horizon 5")

with tab4:
    st.subheader("同行面板 LightGBM 与基准模型")
    if PANEL_METRICS.exists():
        panel_metrics = _load_csv(PANEL_METRICS)
        st.dataframe(panel_metrics, use_container_width=True)
        core = panel_metrics[panel_metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return"])]
        if not core.empty:
            st.bar_chart(core.pivot_table(index=["model", "feature_set"], columns="metric", values="value", aggfunc="first"))
        for path in [
            FIGURE_DIR / "panel_model_f1_comparison.png",
            FIGURE_DIR / "panel_model_strategy_return.png",
            FIGURE_DIR / "panel_lightgbm_importance.png",
        ]:
            if path.exists():
                st.image(str(path), use_container_width=True)
        if PANEL_IMPORTANCE.exists():
            st.subheader("面板模型特征重要性")
            st.dataframe(_load_csv(PANEL_IMPORTANCE).head(50), use_container_width=True)
        if PANEL_PREDICTIONS.exists():
            with st.expander("查看面板模型预测明细"):
                st.dataframe(_load_csv(PANEL_PREDICTIONS), use_container_width=True)
    else:
        st.info(
            "尚未生成面板模型结果，请运行：\n\n"
            "python src/panel_modeling.py --valid-start 2024-01-01 --test-start 2025-01-01 --transaction-cost 0.001"
        )

    st.subheader("双分支 LSTM + MLP")
    if DUAL_BRANCH_METRICS.exists():
        dual_metrics = _load_csv(DUAL_BRANCH_METRICS)
        st.dataframe(dual_metrics, use_container_width=True)
        for path in [
            FIGURE_DIR / "dual_branch_strategy_return.png",
            FIGURE_DIR / "dual_branch_symbol_f1.png",
        ]:
            if path.exists():
                st.image(str(path), use_container_width=True)
        if DUAL_BRANCH_PREDICTIONS.exists():
            with st.expander("查看双分支模型预测明细"):
                st.dataframe(_load_csv(DUAL_BRANCH_PREDICTIONS), use_container_width=True)
    else:
        st.info(
            "尚未生成双分支模型结果，请运行：\n\n"
            "python src/dual_branch_lstm.py --epochs 30 --window 20 --transaction-cost 0.001"
        )

    st.subheader("截面 TopK 非重叠调仓回测")
    if CROSS_SECTIONAL_METRICS.exists():
        cross_metrics = _load_csv(CROSS_SECTIONAL_METRICS)
        st.dataframe(cross_metrics, use_container_width=True)
        for path in [
            FIGURE_DIR / "cross_sectional_topk_return.png",
            FIGURE_DIR / "cross_sectional_topk_excess_return.png",
        ]:
            if path.exists():
                st.image(str(path), use_container_width=True)
        if CROSS_SECTIONAL_BACKTEST.exists():
            with st.expander("查看组合调仓明细"):
                st.dataframe(_load_csv(CROSS_SECTIONAL_BACKTEST), use_container_width=True)
        if CROSS_SECTIONAL_DETAIL.exists():
            with st.expander("查看入选股票明细"):
                st.dataframe(_load_csv(CROSS_SECTIONAL_DETAIL), use_container_width=True)
    else:
        st.info(
            "尚未生成截面 TopK 回测结果，请运行：\n\n"
            "python src/cross_sectional_backtest.py --top-k 3 --rebalance-step 5 --transaction-cost 0.001"
        )

    st.subheader("Walk-forward 截面 TopK 回测")
    if WALK_FORWARD_TOPK_METRICS.exists():
        walk_metrics = _load_csv(WALK_FORWARD_TOPK_METRICS)
        st.dataframe(walk_metrics, use_container_width=True)
        for path in [
            FIGURE_DIR / "walk_forward_topk_return.png",
            FIGURE_DIR / "walk_forward_topk_excess_return.png",
            FIGURE_DIR / "walk_forward_topk_ranking.png",
        ]:
            if path.exists():
                st.image(str(path), use_container_width=True)
        if WALK_FORWARD_YEARLY_METRICS.exists():
            st.subheader("Walk-forward 年度拆解")
            st.dataframe(_load_csv(WALK_FORWARD_YEARLY_METRICS), use_container_width=True)
        if WALK_FORWARD_FOLD_METRICS.exists():
            with st.expander("查看逐年分类指标"):
                st.dataframe(_load_csv(WALK_FORWARD_FOLD_METRICS), use_container_width=True)
        if WALK_FORWARD_TOPK_BACKTEST.exists():
            with st.expander("查看 Walk-forward 调仓明细"):
                st.dataframe(_load_csv(WALK_FORWARD_TOPK_BACKTEST), use_container_width=True)
    else:
        st.info(
            "尚未生成 Walk-forward TopK 回测结果，请运行：\n\n"
            "python src/walk_forward_panel_backtest.py --first-test-year 2022 --top-k-grid 1 2 3 5 --transaction-cost 0.001"
        )

with tab5:
    st.subheader("原始/衍生指标")
    st.dataframe(financials, use_container_width=True)
    st.subheader("标准化矩阵")
    st.dataframe(standardized, use_container_width=True)
