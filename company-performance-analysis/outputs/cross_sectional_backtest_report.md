# 截面 TopK 非重叠调仓回测

- 调仓间隔：每 5 个交易日
- TopK：1, 2, 3, 5
- 最低预测概率：0.000
- 单边交易成本：0.0010

## 最优组合

- 来源/模型：panel / lightgbm_valuation
- 参数：Top5, 最低概率 0.000
- 总收益：0.0881
- 等权基准总收益：-0.0986
- 超额收益：0.1866
- 最大回撤：-0.1903
- Sharpe：0.3988

## 结果文件

- outputs/tables/cross_sectional_backtest.csv
- outputs/tables/cross_sectional_selection_detail.csv
- outputs/tables/cross_sectional_metrics.csv
- outputs/figures/cross_sectional_topk_return.png
- outputs/figures/cross_sectional_topk_excess_return.png