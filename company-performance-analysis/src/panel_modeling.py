"""Panel-data baselines and LightGBM ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_evaluation import classification_metrics, max_drawdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

TARGET_RETURN = "future_5d_return"
TARGET_DIRECTION = "future_5d_direction"

PRICE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_change",
    "amplitude",
    "change",
]
TECH_FEATURES = [
    "daily_return",
    "return_5d",
    "return_20d",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "volume_ma5",
    "volume_ma20",
    "volatility_5d",
    "volatility_20d",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "boll_mid",
    "boll_upper",
    "boll_lower",
    "boll_width",
]
INDEX_PREFIXES = ["hs300", "csi_pharma", "chinext"]
VALUATION_FEATURES = [
    "valuation_close",
    "valuation_pct_change",
    "total_market_cap",
    "float_market_cap",
    "total_shares",
    "float_shares",
    "pe_ttm",
    "pe_static",
    "pb",
    "peg",
    "pcf",
    "ps",
]
FINANCIAL_FEATURES = [
    "revenue",
    "net_profit",
    "roe",
    "gross_margin",
    "net_margin",
    "asset_liability_ratio",
    "current_ratio",
]


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _existing(columns: list[str], candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in columns]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    index_features = [
        col
        for col in df.columns
        if any(col.startswith(f"{prefix}_") for prefix in INDEX_PREFIXES)
        and not col.endswith("_source")
    ]
    base = _existing(list(df.columns), PRICE_FEATURES + TECH_FEATURES)
    market = base + _existing(list(df.columns), index_features)
    valuation = market + _existing(list(df.columns), VALUATION_FEATURES)
    financial = valuation + _existing(list(df.columns), FINANCIAL_FEATURES)
    return {
        "base": base,
        "market": list(dict.fromkeys(market)),
        "valuation": list(dict.fromkeys(valuation)),
        "financial": list(dict.fromkeys(financial)),
    }


def load_panel(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "symbol", TARGET_RETURN, TARGET_DIRECTION]).sort_values(["date", "symbol"])
    return df.reset_index(drop=True)


def split_by_time(
    df: pd.DataFrame,
    valid_start: str,
    test_start: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_date = pd.Timestamp(valid_start)
    test_date = pd.Timestamp(test_start)
    train_idx = np.where(df["date"] < valid_date)[0]
    valid_idx = np.where((df["date"] >= valid_date) & (df["date"] < test_date))[0]
    test_idx = np.where(df["date"] >= test_date)[0]
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Time split produced an empty train/valid/test subset.")
    return train_idx, valid_idx, test_idx


def make_design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    identity = pd.get_dummies(df[["symbol", "sector"]].fillna("unknown"), columns=["symbol", "sector"], dtype=float)
    return pd.concat([numeric, identity], axis=1)


def build_model(model_name: str, seed: int):
    if model_name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=None,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=seed,
            objective="binary",
            verbose=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def predict_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict_proba(x)[:, 1]


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_t, best_score = 0.5, -1.0
    for threshold in np.round(np.arange(0.20, 0.81, 0.02), 2):
        pred = (probability >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_t = float(threshold)
            best_score = float(score)
    return best_t, best_score


def add_strategy_returns(predictions: pd.DataFrame, transaction_cost: float) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []
    for _symbol, group in predictions.sort_values("date").groupby("symbol", sort=False):
        group = group.copy()
        signal = group["predicted_direction"].astype(float).clip(lower=0, upper=1)
        turnover = signal.diff().abs().fillna(signal.abs())
        group["position"] = signal
        group["turnover"] = turnover
        group["transaction_cost"] = turnover * transaction_cost
        group["strategy_return"] = group["actual_return"] * signal - group["transaction_cost"]
        group["buy_hold_return"] = group["actual_return"]
        out_parts.append(group)
    return pd.concat(out_parts, ignore_index=True)


def aggregate_strategy_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    daily = (
        predictions.groupby("date", as_index=False)[["strategy_return", "buy_hold_return"]]
        .mean()
        .sort_values("date")
    )
    daily["strategy_cum_return"] = (1 + daily["strategy_return"]).cumprod() - 1
    daily["buy_hold_cum_return"] = (1 + daily["buy_hold_return"]).cumprod() - 1
    return {
        "strategy_cum_return": float(daily["strategy_cum_return"].iloc[-1]),
        "buy_hold_cum_return": float(daily["buy_hold_cum_return"].iloc[-1]),
        "max_drawdown": max_drawdown(daily["strategy_cum_return"]),
        "trade_count": float(predictions["turnover"].sum()),
        "total_transaction_cost": float(predictions["transaction_cost"].sum()),
    }


def extract_feature_importance(model, columns: list[str], model_name: str) -> pd.DataFrame:
    if model_name == "lightgbm":
        values = model.feature_importances_
    elif model_name == "random_forest":
        values = model.named_steps["model"].feature_importances_
    elif model_name == "logistic":
        values = np.abs(model.named_steps["model"].coef_[0])
    else:
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature": columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_experiments(
    data_path: str | Path,
    valid_start: str,
    test_start: str,
    transaction_cost: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_panel(data_path)
    train_idx, valid_idx, test_idx = split_by_time(df, valid_start, test_start)
    y = df[TARGET_DIRECTION].astype(int).to_numpy()
    sets = feature_sets(df)

    experiment_specs: list[tuple[str, str]] = [(name, "lightgbm") for name in sets]
    experiment_specs.extend(
        [
            ("financial", "logistic"),
            ("financial", "random_forest"),
        ]
    )

    all_metrics: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []

    for feature_set, model_name in experiment_specs:
        features = sets[feature_set]
        x = make_design_matrix(df, features)
        model = build_model(model_name, seed)
        model.fit(x.iloc[train_idx], y[train_idx])

        valid_prob = predict_probability(model, x.iloc[valid_idx])
        threshold, valid_f1 = best_threshold(y[valid_idx], valid_prob)
        test_prob = predict_probability(model, x.iloc[test_idx])
        test_pred = (test_prob >= threshold).astype(int)

        pred = df.iloc[test_idx][["date", "symbol", "name", "sector", "close", TARGET_RETURN, TARGET_DIRECTION]].copy()
        pred = pred.rename(columns={TARGET_RETURN: "actual_return", TARGET_DIRECTION: "actual_direction"})
        pred["feature_set"] = feature_set
        pred["model"] = model_name
        pred["predicted_probability"] = test_prob
        pred["threshold"] = threshold
        pred["predicted_direction"] = test_pred
        pred = add_strategy_returns(pred, transaction_cost)
        all_predictions.append(pred)

        metrics = classification_metrics(y[test_idx], test_pred)
        metrics.update(aggregate_strategy_metrics(pred))
        metrics["valid_f1_at_threshold"] = valid_f1
        metrics["threshold"] = threshold
        for metric, value in metrics.items():
            all_metrics.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "metric": metric,
                    "value": value,
                    "valid_start": valid_start,
                    "test_start": test_start,
                    "transaction_cost": transaction_cost,
                }
            )

        importance = extract_feature_importance(model, list(x.columns), model_name)
        if not importance.empty:
            importance["feature_set"] = feature_set
            importance["model"] = model_name
            all_importance.append(importance)
        print(f"[OK] {model_name}/{feature_set}: threshold={threshold:.2f}, F1={metrics['F1-score']:.3f}")

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    importance_df = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    return metrics_df, predictions_df, importance_df


def plot_outputs(metrics: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    core = metrics[metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return"])]
    if not core.empty:
        pivot = core.pivot_table(index=["model", "feature_set"], columns="metric", values="value", aggfunc="first")
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot["F1-score"].sort_values().plot(kind="barh", ax=ax, color="#2563eb")
        ax.set_title("Panel model F1-score comparison")
        ax.set_xlabel("F1-score")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "panel_model_f1_comparison.png", dpi=180)
        plt.close(fig)

    daily_parts = []
    for (model, feature_set), group in predictions.groupby(["model", "feature_set"]):
        daily = group.groupby("date", as_index=False)[["strategy_return", "buy_hold_return"]].mean().sort_values("date")
        daily["strategy_cum_return"] = (1 + daily["strategy_return"]).cumprod() - 1
        daily["series"] = f"{model}_{feature_set}"
        daily_parts.append(daily)
    if daily_parts:
        daily_all = pd.concat(daily_parts, ignore_index=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for series, group in daily_all.groupby("series"):
            ax.plot(pd.to_datetime(group["date"]), group["strategy_cum_return"], label=series, linewidth=1.5)
        ax.set_title("Panel strategy cumulative return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "panel_model_strategy_return.png", dpi=180)
        plt.close(fig)

    top = importance[(importance["model"] == "lightgbm") & (importance["feature_set"] == "financial")].head(25)
    if not top.empty:
        top = top.sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top["feature"], top["importance"], color="#0f766e")
        ax.set_title("Panel LightGBM feature importance")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "panel_lightgbm_importance.png", dpi=180)
        plt.close(fig)


def write_report(metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    report_path = PROJECT_ROOT / "outputs" / "panel_model_report.md"
    summary = metrics[metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return", "max_drawdown"])].pivot_table(
        index=["model", "feature_set"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    top_importance = importance[(importance["model"] == "lightgbm") & (importance["feature_set"] == "financial")].head(15)
    lines = [
        "# 面板模型与消融实验报告",
        "",
        "## 实验设计",
        "使用 8 家医药及相关上市公司构建面板数据，训练集为 2024 年以前，验证集为 2024 年，测试集为 2025 年及以后。验证集用于选择 F1-score 最优分类阈值，测试集只用于最终样本外评估。",
        "",
        "特征消融顺序为：base 行情与技术指标；market 加入指数环境；valuation 加入估值市值；financial 加入财务指标。面板模型没有纳入 TOPSIS，因为同行公司尚未统一重算 TOPSIS 得分。",
        "",
        "## 核心结果",
        "```text",
        summary.round(4).to_string(),
        "```",
        "",
        "## LightGBM Top 特征",
        "```text",
        top_importance.to_string(index=False),
        "```",
        "",
        "## 结论",
        "面板 LightGBM 可以作为 LSTM 之外的重要强基准。若 financial 特征集显著优于 valuation 或 market，说明财务信息在跨公司样本中具有增量解释力；若提升有限，则短期方向更多由市场状态、估值和技术面驱动。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run panel LightGBM ablation and baseline models.")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "panel_model_data.csv"))
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics, predictions, importance = run_experiments(
        data_path=args.data,
        valid_start=args.valid_start,
        test_start=args.test_start,
        transaction_cost=args.transaction_cost,
        seed=args.seed,
    )
    metrics.to_csv(TABLE_DIR / "panel_model_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(TABLE_DIR / "panel_model_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLE_DIR / "panel_model_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_outputs(metrics, predictions, importance)
    write_report(metrics, importance)

    summary = metrics[metrics["metric"].isin(["Accuracy", "F1-score", "strategy_cum_return"])].pivot_table(
        index=["model", "feature_set"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
