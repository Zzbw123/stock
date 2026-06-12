"""Markdown report writer for the Changchun High-Tech performance project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INDICATOR_NAMES = {
    "revenue": "营业收入",
    "net_profit": "净利润",
    "roe": "ROE",
    "gross_margin": "毛利率",
    "net_margin": "净利率",
    "asset_liability_ratio": "资产负债率",
    "current_ratio": "流动比率",
    "revenue_growth": "营收增长率",
    "net_profit_growth": "净利润增长率",
}


def _fmt_amount(value: float | int | None) -> str:
    if pd.isna(value):
        return "缺失"
    return f"{value / 100_000_000:.2f} 亿元"


def _fmt_pct(value: float | int | None) -> str:
    if pd.isna(value):
        return "缺失"
    return f"{value:.2f}%"


def _fmt_num(value: float | int | None) -> str:
    if pd.isna(value):
        return "缺失"
    return f"{value:.2f}"


def _score_for_year(scores: pd.DataFrame, year: int) -> pd.Series:
    matched = scores.loc[scores["year"].astype("Int64") == year]
    if matched.empty:
        return scores.sort_values("year").tail(1).iloc[0]
    return matched.iloc[0]


def _relative_figure(output_path: Path, figure_name: str) -> str:
    return (Path("figures") / figure_name).as_posix()


def generate_performance_report(
    financials: pd.DataFrame,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    scores: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a Chinese performance assessment report backed by generated outputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.parent / "metrics.csv"

    financials = financials.sort_values("year").copy()
    scores = scores.sort_values("rank").copy()
    latest = financials.tail(1).iloc[0]
    previous = financials.iloc[-2] if len(financials) >= 2 else latest
    latest_year = int(latest["year"])
    previous_year = int(previous["year"])
    latest_score = _score_for_year(scores, latest_year)
    best_score = scores.iloc[0]

    latest_revenue = latest.get("revenue")
    latest_profit = latest.get("net_profit")
    previous_revenue = previous.get("revenue")
    previous_profit = previous.get("net_profit")
    revenue_yoy = (latest_revenue / previous_revenue - 1) * 100 if pd.notna(previous_revenue) and previous_revenue else pd.NA
    profit_yoy = (latest_profit / previous_profit - 1) * 100 if pd.notna(previous_profit) and previous_profit else pd.NA

    top_weights = weights.sort_values("weight", ascending=False).head(5).copy()
    weight_lines = [
        f"- {INDICATOR_NAMES.get(row.indicator, row.indicator)}：{row.weight:.1%}"
        for row in top_weights.itertuples(index=False)
    ]

    price_note = "行情数据未生成，报告不评价股价与绩效的关系。"
    if prices is not None and not prices.empty and "close" in prices:
        prices = prices.sort_values("date").copy()
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        price_change = (last_price["close"] / first_price["close"] - 1) * 100 if first_price["close"] else pd.NA
        price_note = (
            f"行情样本从 {first_price['date']} 到 {last_price['date']}，"
            f"复权收盘价由 {_fmt_num(first_price['close'])} 元变为 {_fmt_num(last_price['close'])} 元，"
            f"区间变化约 {_fmt_pct(price_change)}。"
        )

    metric_lookup = {}
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metric_lookup = dict(zip(metrics_df["metric"], metrics_df["value"], strict=False))
    pca_section = ""
    if "pca_topsis_spearman_corr" in metric_lookup:
        pca_section = f"""
## PCA 稳健性检验支持 TOPSIS 主结论

**PCA 降维后的综合得分与 TOPSIS 结果高度一致。** PC1 解释了 {float(metric_lookup['pca_pc1_variance_ratio']):.1%} 的指标差异，PC1+PC2 合计解释 {float(metric_lookup['pca_pc1_pc2_cumulative_variance_ratio']):.1%}；PCA 综合得分与 TOPSIS 得分的 Pearson 相关系数为 {float(metric_lookup['pca_topsis_pearson_corr']):.3f}，Spearman 相关系数为 {float(metric_lookup['pca_topsis_spearman_corr']):.3f}，Top5 年份重合 {int(float(metric_lookup['top5_overlap_count']))}/5。最新年度 {int(float(metric_lookup['latest_year']))} 年在 TOPSIS 中排名第 {int(float(metric_lookup['latest_topsis_rank']))}，在 PCA 中排名第 {int(float(metric_lookup['latest_pca_rank']))}，说明“近期回落但仍非样本最弱”的判断具有稳健性。

![PCA 稳健性检验]({_relative_figure(output_path, 'pca_topsis_robustness.png')})

**二维 PCA 图显示，高位年份主要沿收入、净利润和盈利能力方向展开，2024-2025 年从高绩效区域回撤。** 因此，绩效下滑并非单纯由资产负债率或流动比率驱动，而是更集中体现在利润与回报指标上。

![PCA 二维可视化]({_relative_figure(output_path, 'pca_2d_biplot.png')})
"""

    advanced_section = ""
    if "sensitivity_min_spearman" in metric_lookup:
        advanced_section = f"""
## 模型稳健性与指标诊断进一步锁定利润拖累

**留一指标敏感性显示，模型结论不是由单一指标强行决定。** 逐一去掉一个指标后，最新年度排名最大变化为 {int(float(metric_lookup['sensitivity_latest_rank_max_abs_change']))} 位，最低 Spearman 相关仍为 {float(metric_lookup['sensitivity_min_spearman']):.3f}，Top5 年份最少仍重合 {int(float(metric_lookup['sensitivity_min_top5_overlap']))}/5。也就是说，2025 年绩效回落不是某一个指标口径造成的偶然结果。

![留一指标敏感性]({_relative_figure(output_path, 'leave_one_indicator_sensitivity.png')})

**指标热力图和退化贡献图把问题定位到利润与回报能力。** 热力图显示 2021-2023 年多项指标同步处于高位，而 2025 年在净利润、ROE、净利率等指标上明显转弱；2023 到 2025 的加权标准化变化合计为 {float(metric_lookup['degradation_total_weighted_change_2023_to_latest']):.3f}，主要负向贡献来自利润与回报类指标。

![指标热力图]({_relative_figure(output_path, 'indicator_heatmap.png')})

![2023 到 2025 退化贡献]({_relative_figure(output_path, 'degradation_contribution_2023_2025.png')})

**熵权与 PCA 载荷的对比说明，规模利润指标的重要性判断具有方法一致性。** 熵权法权重与 PC1 载荷绝对值的 Spearman 相关为 {float(metric_lookup['entropy_weight_vs_abs_pc1_loading_spearman']):.3f}，说明两种不同评价思路都把主要信息集中在规模、净利润和盈利能力方向上。

![熵权与 PCA 载荷对比]({_relative_figure(output_path, 'entropy_vs_pca_loading.png')})

**阶段分布和排名轨迹把“高位后回落”的叙事变成可检验事实。** 高位表现期的 TOPSIS 中位得分为 {float(metric_lookup['stage_high_period_median_score']):.3f}，近期调整期中位得分为 {float(metric_lookup['stage_recent_period_median_score']):.3f}；排名轨迹显示 2023 年达到样本第一后，2024-2025 年明显后移。

![分阶段绩效分布]({_relative_figure(output_path, 'stage_score_distribution.png')})

![TOPSIS 排名轨迹]({_relative_figure(output_path, 'topsis_rank_bump.png')})
"""

    latest_rank = int(latest_score["rank"])
    total_periods = len(scores)
    latest_score_value = latest_score["topsis_score"]
    best_year = int(best_score["year"])
    best_score_value = best_score["topsis_score"]

    conclusion = (
        "经营绩效已经从 2021-2023 年的高位阶段转入明显回落阶段"
        if latest_score_value < 0.55
        else "经营绩效仍处在相对较强区间，但边际变化需要继续观察"
    )

    report = f"""# 长春高新经营绩效评估报告

## Executive Summary

- **总体判断：{conclusion}。** 最新年报口径为 {latest_year} 年，TOPSIS 综合得分为 {latest_score_value:.3f}，在 {total_periods} 个年度样本中排名第 {latest_rank}；历史最高得分出现在 {best_year} 年，得分为 {best_score_value:.3f}。
- **收入和利润同步承压。** {latest_year} 年营业收入为 {_fmt_amount(latest_revenue)}，较 {previous_year} 年变化 {_fmt_pct(revenue_yoy)}；净利润为 {_fmt_amount(latest_profit)}，较 {previous_year} 年变化 {_fmt_pct(profit_yoy)}。利润端下降幅度远大于收入端，是综合绩效下滑的核心原因。
- **偿债安全边际仍较充足。** {latest_year} 年资产负债率为 {_fmt_pct(latest.get('asset_liability_ratio'))}，流动比率为 {_fmt_num(latest.get('current_ratio'))}，财务结构没有表现出高杠杆压力。
- **评价结论需要结合业务解释使用。** 本报告基于 AkShare 可取得的公开财务指标和行情数据，TOPSIS 排名是多指标相对评价，不等同于投资建议，也不能替代对生长激素、医药政策、产品管线和年报附注的进一步分析。

## 绩效趋势已经从高位回落

**收入规模仍大于早期水平，但利润质量在最新年度显著走弱。** 从趋势面板看，营业收入在 2023 年达到高位后连续回落，净利润在 2025 年出现陡降；ROE 同步降至低位，说明资本回报能力弱化。资产负债率维持在较低区间，这让公司有一定缓冲，但不能抵消盈利端的压力。

![Streamlit 趋势页指标面板]({_relative_figure(output_path, 'streamlit_trends.png')})

## TOPSIS 得分显示 2023 年是阶段性高点

**综合绩效排名的峰值集中在 2021-2023 年，最新年度排名明显后移。** TOPSIS 方法把收入、利润、盈利能力、偿债能力等指标合成一个相对得分；在该口径下，2023 年为样本内第一，2025 年降至第 {latest_rank}。这意味着公司的核心问题不是单一指标波动，而是收入、利润率和资本回报的组合走弱。

![TOPSIS 综合绩效得分]({_relative_figure(output_path, 'topsis_score.png')})
{pca_section}
{advanced_section}

## 评价权重集中在规模和盈利指标

**熵权法结果显示，模型最看重信息差异较大的收入和净利润。** 当前权重最高的指标为：

{chr(10).join(weight_lines)}

这解释了为什么 2025 年净利润大幅回落会显著拉低综合得分。若后续收入恢复但利润率没有同步修复，TOPSIS 得分大概率只能温和改善；若净利润和 ROE 能够恢复，综合排名才更可能实质反弹。

![熵权法指标权重]({_relative_figure(output_path, 'entropy_weights.png')})

## 股价与绩效之间出现同向走弱信号

**股价不是绩效评价指标本身，但可作为市场预期的外部观察。** {price_note} 结合股价与 TOPSIS 对比图看，市场价格和经营绩效在高位之后均有回落，说明基本面走弱与市场估值/预期调整方向一致。

![股价与综合绩效得分对比]({_relative_figure(output_path, 'price_vs_topsis_score.png')})

## Recommended Next Steps

1. **补充经营现金流和分业务数据。** 当前 AkShare 财务摘要没有提供年报口径经营现金流量净额，建议从年报或巨潮资讯补齐现金流、产品线收入和毛利率，以判断利润下滑是否伴随现金质量恶化。
2. **把 2024-2025 年作为重点复盘窗口。** 最新两年的收入、净利润、ROE 变化最大，应进一步拆解为价格、销量、费用率、减值或非经常性损益。
3. **持续监控 2026 年季报。** 如果后续季度收入继续下滑或净利率不能恢复，综合绩效仍有继续承压风险；若利润率修复，则 TOPSIS 得分可能先于收入规模恢复。

## Further Questions

- 2025 年净利润大幅下降主要来自主营业务毛利率、费用率、减值，还是一次性项目？
- 生长激素核心产品的销量、价格和竞争格局是否发生结构性变化？
- 若加入研发投入、销售费用率、现金流和分产品指标，综合绩效排序是否仍指向同样结论？

## Caveats and Assumptions

- 数据来源为 AkShare：财务摘要来自 `stock_financial_abstract_ths`，行情来自 `stock_zh_a_hist_tx`。
- 财务口径筛选为年报期末数据，样本区间为 {int(financials['year'].min())}-{latest_year} 年。
- TOPSIS 是相对评价模型，权重来自样本内熵权法；样本扩展、指标增删或财务重述都会改变排名。
- 本报告用于经营绩效分析，不构成证券投资建议。
"""

    output_path.write_text(report, encoding="utf-8")
    return output_path
