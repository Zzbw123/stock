"""Visualization helpers for performance analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from indicators import INDICATOR_LABELS

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional styling dependency.
    sns = None


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
if sns is not None:
    sns.set_theme(style="whitegrid", font="SimHei")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _x_labels(df: pd.DataFrame) -> pd.Series:
    if "year" in df:
        return df["year"].astype(str)
    return df["period"].astype(str)


def plot_trends(financials: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save line charts for key financial trend indicators."""
    output_dir = ensure_dir(output_dir)
    chart_specs = [
        ("revenue", "营业收入趋势", "金额"),
        ("net_profit", "净利润趋势", "金额"),
        ("roe", "ROE 趋势", "%"),
        ("asset_liability_ratio", "资产负债率趋势", "%"),
        ("operating_cash_flow", "经营现金流趋势", "金额"),
    ]
    saved: list[Path] = []
    labels = _x_labels(financials)

    for col, title, ylabel in chart_specs:
        if col not in financials or financials[col].dropna().empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(labels, financials[col], marker="o", linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("期间")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = output_dir / f"trend_{col}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_streamlit_trend_panel(financials: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Save the multi-chart trend panel shown in the Streamlit dashboard."""
    output_dir = ensure_dir(output_dir)
    chart_specs = [
        ("revenue", "营业收入", "亿元"),
        ("net_profit", "净利润", "亿元"),
        ("roe", "ROE", "%"),
        ("asset_liability_ratio", "资产负债率", "%"),
        ("operating_cash_flow", "经营现金流", "亿元"),
    ]
    available = [
        (col, title, ylabel)
        for col, title, ylabel in chart_specs
        if col in financials and financials[col].dropna().empty is False
    ]
    if not available:
        return None

    x_values = pd.to_numeric(financials["year"], errors="coerce") if "year" in financials else _x_labels(financials)
    rows = int(np.ceil(len(available) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows), squeeze=False)
    palette = ["#2563eb", "#dc2626", "#0f766e", "#7c3aed", "#b45309"]

    for idx, (col, title, ylabel) in enumerate(available):
        ax = axes[idx // 2][idx % 2]
        values = pd.to_numeric(financials[col], errors="coerce")
        if ylabel == "亿元":
            values = values / 100_000_000
        ax.plot(x_values, values, marker="o", linewidth=2.1, color=palette[idx % len(palette)])
        ax.set_title(title)
        ax.set_xlabel("年份")
        ax.set_ylabel(ylabel)
        if "year" in financials:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis="x", rotation=35)

    for idx in range(len(available), rows * 2):
        axes[idx // 2][idx % 2].axis("off")

    fig.suptitle("Streamlit 趋势页指标面板", fontsize=15, y=0.995)
    fig.tight_layout()
    path = output_dir / "streamlit_trends.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weight_bar(weights: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    df = weights.copy()
    df["label"] = df["indicator"].map(INDICATOR_LABELS).fillna(df["indicator"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if sns is not None:
        sns.barplot(data=df, x="weight", y="label", ax=ax, color="#3b82f6")
    else:
        ax.barh(df["label"], df["weight"], color="#3b82f6")
        ax.invert_yaxis()
    ax.set_title("熵权法指标权重")
    ax.set_xlabel("权重")
    ax.set_ylabel("指标")
    fig.tight_layout()
    path = output_dir / "entropy_weights.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_topsis_score(scores: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    df = scores.sort_values("year" if "year" in scores else "period")
    labels = _x_labels(df)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(labels, df["topsis_score"], marker="o", color="#0f766e", linewidth=2.2)
    ax.set_title("TOPSIS 综合绩效得分")
    ax.set_xlabel("期间")
    ax.set_ylabel("绩效得分")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output_dir / "topsis_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_radar(standardized: pd.DataFrame, output_dir: str | Path, max_periods: int = 5) -> Path:
    """Save a radar chart using standardized indicators."""
    output_dir = ensure_dir(output_dir)
    id_cols = [col for col in ["period", "year"] if col in standardized]
    indicators = [col for col in standardized.columns if col not in id_cols]
    if len(indicators) < 3:
        raise ValueError("Radar chart needs at least three indicators.")

    df = standardized.sort_values(id_cols[-1] if id_cols else indicators[0]).tail(max_periods)
    labels = [INDICATOR_LABELS.get(col, col) for col in indicators]
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    for _, row in df.iterrows():
        values = row[indicators].fillna(0).tolist()
        values += values[:1]
        name = str(row["year"] if "year" in row else row["period"])
        ax.plot(angles, values, linewidth=1.8, label=name)
        ax.fill(angles, values, alpha=0.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("各指标标准化雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    path = output_dir / "indicator_radar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_price_vs_score(prices: pd.DataFrame, scores: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Compare annual closing price with TOPSIS score."""
    if prices is None or prices.empty or "close" not in prices:
        return None
    output_dir = ensure_dir(output_dir)
    price_year = prices.copy()
    price_year["year"] = pd.to_datetime(price_year["date"]).dt.year
    price_year = price_year.sort_values("date").groupby("year", as_index=False).tail(1)
    merged = scores.merge(price_year[["year", "close"]], on="year", how="inner")
    if merged.empty:
        return None

    merged = merged.sort_values("year")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(merged["year"].astype(str), merged["topsis_score"], marker="o", color="#0f766e", label="绩效得分")
    ax1.set_ylabel("TOPSIS 得分", color="#0f766e")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(merged["year"].astype(str), merged["close"], marker="s", color="#dc2626", label="年末收盘价")
    ax2.set_ylabel("股价", color="#dc2626")
    ax1.set_title("股价与综合绩效得分对比")
    ax1.set_xlabel("年份")
    fig.tight_layout()
    path = output_dir / "price_vs_topsis_score.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
