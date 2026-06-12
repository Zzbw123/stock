"""SHAP feature attribution for the trend-prediction feature set.

This module uses a LightGBM proxy classifier on the same engineered features
used by the LSTM experiments. Tree SHAP is much faster and more stable than
explaining a sequence model directly, and it answers the core project question:
which market, valuation, financial and TOPSIS features contribute to direction
classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from lstm_predict import (
    FIGURE_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    TARGET_DIRECTION_TEMPLATE,
    TARGET_RETURN_TEMPLATE,
    load_or_build_model_data,
    restrict_to_common_fusion_period,
    select_feature_columns,
)
from model_evaluation import classification_metrics


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _require_shap():
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install it with: pip install shap") from exc
    return shap


def _prepare_tabular_data(df: pd.DataFrame, model_type: str, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    target_direction = TARGET_DIRECTION_TEMPLATE.format(horizon=horizon)
    target_return = TARGET_RETURN_TEMPLATE.format(horizon=horizon)
    feature_cols = select_feature_columns(df, model_type, horizon)
    keep = ["date", target_direction, target_return] + feature_cols
    work = df[keep].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in [target_direction, target_return] + feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", target_direction] + feature_cols)
    work = work.sort_values("date").reset_index(drop=True)
    return work[feature_cols], work[target_direction].astype(int), work["date"]


def _time_split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(int(n * 0.70), 1)
    valid_end = max(int(n * 0.85), train_end + 1)
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:valid_end], idx[valid_end:]


def _positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def run_shap_analysis(
    data_path: str | Path,
    model_type: str,
    horizon: int,
    max_display: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shap = _require_shap()
    df = restrict_to_common_fusion_period(load_or_build_model_data(data_path, horizon), horizon)
    x, y, dates = _prepare_tabular_data(df, model_type, horizon)
    train_idx, valid_idx, test_idx = _time_split(len(x))
    train_valid_idx = np.concatenate([train_idx, valid_idx])

    positives = int(y.iloc[train_valid_idx].sum())
    negatives = int(len(train_valid_idx) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        verbose=-1,
    )
    model.fit(x.iloc[train_valid_idx], y.iloc[train_valid_idx])
    pred_prob = model.predict_proba(x.iloc[test_idx])[:, 1]
    pred = (pred_prob >= 0.5).astype(int)
    metrics = classification_metrics(y.iloc[test_idx].to_numpy(), pred)
    metrics_df = pd.DataFrame(
        [{"model": f"shap_proxy_{model_type}", "metric": metric, "value": value} for metric, value in metrics.items()]
    )

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(x.iloc[test_idx])
    shap_values = _positive_class_shap_values(shap_raw)
    shap_df = pd.DataFrame(shap_values, columns=x.columns)
    shap_df.insert(0, "date", dates.iloc[test_idx].to_numpy())
    shap_df.insert(1, "actual_direction", y.iloc[test_idx].to_numpy())
    shap_df.insert(2, "predicted_probability", pred_prob)

    importance = (
        pd.DataFrame({"feature": x.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(TABLE_DIR / "shap_values.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLE_DIR / "shap_importance.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(TABLE_DIR / "shap_proxy_metrics.csv", index=False, encoding="utf-8-sig")

    shap.summary_plot(shap_values, x.iloc[test_idx], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    top = importance.head(max_display).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2563eb")
    ax.set_title("SHAP mean absolute contribution")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_importance_bar.png", dpi=180)
    plt.close(fig)

    for candidate in ["topsis_score", "rank", "pe_ttm", "pb", "ps", "hs300_return_20d", "csi_pharma_return_20d"]:
        if candidate in x.columns:
            shap.dependence_plot(candidate, shap_values, x.iloc[test_idx], show=False)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"shap_dependence_{candidate}.png", dpi=180, bbox_inches="tight")
            plt.close()

    write_shap_report(importance, metrics_df, model_type)
    return importance, shap_df, metrics_df


def write_shap_report(importance: pd.DataFrame, metrics: pd.DataFrame, model_type: str) -> None:
    top10 = importance.head(10)
    watched = importance[
        importance["feature"].isin(
            [
                "topsis_score",
                "rank",
                "revenue",
                "net_profit",
                "roe",
                "pe_ttm",
                "pb",
                "ps",
                "hs300_return_20d",
                "csi_pharma_return_20d",
            ]
        )
    ]
    lines = [
        "# SHAP 特征贡献分析报告",
        "",
        f"解释模型：LightGBM 代理分类器，特征集合：{model_type}。",
        "",
        "## 样本外代理模型指标",
        "```text",
        metrics.to_string(index=False),
        "```",
        "",
        "## 全局 Top 10 特征",
        "```text",
        top10.to_string(index=False),
        "```",
        "",
        "## 重点关注变量",
        "```text",
        watched.to_string(index=False) if not watched.empty else "未在 Top 特征表中发现重点关注变量。",
        "```",
        "",
        "## 解释口径",
        "SHAP 值为正表示该特征在该样本上推高未来 5 日上涨概率，SHAP 值为负表示压低上涨概率。该分析解释的是 LightGBM 代理模型，不等同于直接解释 LSTM 隐状态，但可作为特征体系贡献判断和消融实验设计依据。",
    ]
    (FIGURE_DIR.parents[0] / "shap_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis on the trend-prediction feature set.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "lstm_model_data.csv"))
    parser.add_argument("--model-type", choices=["base", "fusion"], default="fusion")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-display", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    importance, _shap_values, metrics = run_shap_analysis(
        data_path=args.data,
        model_type=args.model_type,
        horizon=args.horizon,
        max_display=args.max_display,
        seed=args.seed,
    )
    print(f"Saved SHAP importance: {TABLE_DIR / 'shap_importance.csv'}")
    print(f"Saved SHAP values: {TABLE_DIR / 'shap_values.csv'}")
    print(f"Saved SHAP figures: {FIGURE_DIR}")
    print(metrics.to_string(index=False))
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
