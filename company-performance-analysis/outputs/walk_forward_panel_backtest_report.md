# Walk-forward 面板 LightGBM + TopK 回测

## 设置

- 测试年份：2022 至 数据最新年份
- 验证方式：测试年前一年选分类阈值，训练集为验证年前全部历史数据
- 特征集：market, valuation, financial
- TopK：1, 2, 3, 5
- 调仓间隔：每 5 个交易日
- 单边交易成本：0.0010

## 最优组合

- 模型：lightgbm_financial
- 参数：Top5, 最低概率 0.000
- 总收益：-0.3705
- 等权基准总收益：-0.5455
- 超额收益：0.1751
- 最大回撤：-0.4666
- Sharpe：-0.1175

## 年度分类表现

```text
           variant  test_year  Accuracy  Precision  Recall  F1-score  threshold  valid_f1_at_threshold
   lightgbm_market       2022    0.4607     0.4473  0.9682    0.6119       0.20                 0.5893
lightgbm_valuation       2022    0.4535     0.4446  0.9824    0.6122       0.24                 0.5959
lightgbm_financial       2022    0.4499     0.4432  0.9859    0.6115       0.20                 0.5901
   lightgbm_market       2023    0.4525     0.4489  0.9908    0.6179       0.24                 0.6117
lightgbm_valuation       2023    0.4509     0.4481  0.9884    0.6167       0.22                 0.6096
lightgbm_financial       2023    0.4499     0.4480  0.9954    0.6179       0.22                 0.6097
   lightgbm_market       2024    0.4174     0.4174  1.0000    0.5889       0.20                 0.6158
lightgbm_valuation       2024    0.4174     0.4170  0.9950    0.5877       0.20                 0.6063
lightgbm_financial       2024    0.4179     0.4174  0.9975    0.5885       0.22                 0.6089
   lightgbm_market       2025    0.5103     0.5088  0.9980    0.6740       0.36                 0.5914
lightgbm_valuation       2025    0.5098     0.5085  1.0000    0.6742       0.34                 0.5926
lightgbm_financial       2025    0.5103     0.5088  0.9929    0.6729       0.42                 0.5917
   lightgbm_market       2026    0.3860     0.3810  0.9967    0.5513       0.20                 0.6454
lightgbm_valuation       2026    0.3897     0.3813  0.9834    0.5495       0.20                 0.6631
lightgbm_financial       2026    0.3872     0.3800  0.9801    0.5476       0.20                 0.6704
```

## 最优组合年度拆解

```text
      source            variant  top_k  min_probability  year  periods  total_return  benchmark_total_return  excess_total_return  max_drawdown  win_rate  avg_turnover  avg_selected_count
walk_forward lightgbm_financial      5              0.0  2022       49       -0.1007                 -0.2010               0.1003       -0.2979    0.4898        0.4531                 5.0
walk_forward lightgbm_financial      5              0.0  2023       48       -0.0661                 -0.1947               0.1286       -0.2106    0.5208        0.5333                 5.0
walk_forward lightgbm_financial      5              0.0  2024       49       -0.2358                 -0.2323              -0.0035       -0.3289    0.3469        0.2449                 5.0
walk_forward lightgbm_financial      5              0.0  2025       48        0.1905                  0.1471               0.0434       -0.1526    0.5208        0.5417                 5.0
walk_forward lightgbm_financial      5              0.0  2026       20       -0.1761                 -0.1979               0.0219       -0.2060    0.5000        0.4800                 5.0
```

## 结果文件

- outputs/tables/walk_forward_panel_predictions.csv
- outputs/tables/walk_forward_fold_metrics.csv
- outputs/tables/walk_forward_topk_backtest.csv
- outputs/tables/walk_forward_topk_metrics.csv
- outputs/tables/walk_forward_topk_yearly_metrics.csv
- outputs/tables/walk_forward_topk_selection_detail.csv
- outputs/figures/walk_forward_topk_return.png
- outputs/figures/walk_forward_topk_excess_return.png
- outputs/figures/walk_forward_topk_ranking.png