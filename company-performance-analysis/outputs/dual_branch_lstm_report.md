# 双分支 LSTM + MLP 模型报告

## 模型结构
日频行情、技术指标、指数和估值变量进入 LSTM 分支；财务指标、TOPSIS 指标以及公司/行业哑变量进入 MLP 静态分支。两个分支的隐表示拼接后，通过全连接层输出未来 5 日上涨概率。

训练阶段支持 early stopping；阈值可按验证集 F1-score 或验证集策略收益选择；也可设置最低持仓概率形成不交易区间，减少低置信度交易。

## 结果
```text
           model                 metric     value valid_start test_start  window  transaction_cost
dual_branch_lstm               Accuracy  0.494894  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm              Precision  0.480838  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm                 Recall  0.944876  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm               F1-score   0.63734  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm       DirectionHitRate  0.494894  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm    strategy_cum_return  -0.11256  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm    buy_hold_cum_return  -0.37404  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm           max_drawdown -0.836417  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm            trade_count     152.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm total_transaction_cost     0.152  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm  valid_f1_at_threshold  0.593472  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm             valid_loss  0.681034  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm             best_epoch       5.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm              threshold      0.28  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm    threshold_objective        f1  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm         min_hold_proba       0.2  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm       min_valid_trades      20.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm            lstm_hidden      32.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm          static_hidden      32.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm                dropout       0.2  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm          learning_rate     0.001  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm sequence_feature_count      80.0  2024-01-01 2025-01-01      20             0.001
dual_branch_lstm   static_feature_count      21.0  2024-01-01 2025-01-01      20             0.001
```

## 解释
该结构避免将低频财务信息简单复制为日频序列，从建模结构上区分了快变量和慢变量。它适合作为单分支 LSTM 与 LightGBM 面板模型之间的深度学习增强版本。