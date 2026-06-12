# LSTM 滚动窗口验证报告

## 方法说明
本实验按年度进行 expanding-window 滚动验证：测试年前的历史样本用于训练和验证，测试年份完全留作样本外评估。训练尾部 15% 作为验证集，并在验证集上选择 F1-score 最优的分类阈值。分类损失使用训练集正负样本比例自动设置 pos_weight，以缓解涨跌方向不均衡。

回测策略为 long-flat：预测未来 5 个交易日上涨时持有，预测下跌时空仓。交易成本按单边 0.1% 在仓位变化时扣除。

## 分年度结果
```text
metric                    Accuracy  F1-score  strategy_cum_return  trade_count
fold      model                                                               
test_2024 base              0.3884    0.5595              -0.8473          1.0
          fusion            0.4008    0.5511              -0.8553          5.0
          naive_momentum    0.5207    0.3830              -0.2147         50.0
test_2025 base              0.4527    0.5933              -0.1446          7.0
          fusion            0.4486    0.6193               0.1716          1.0
          naive_momentum    0.5185    0.4507               0.1453         52.0
test_2026 base              0.3500    0.5185              -0.8512          1.0
          fusion            0.3500    0.5185              -0.8512          1.0
          naive_momentum    0.3800    0.1143              -0.7167         24.0
```

## 平均结果
```text
metric          Accuracy  F1-score  strategy_cum_return  trade_count
model                                                               
base              0.3970    0.5571              -0.6144       3.0000
fusion            0.3998    0.5630              -0.5116       2.3333
naive_momentum    0.4731    0.3160              -0.2620      42.0000
```

## 结论
滚动验证比单次 holdout 更接近样本外检验。当前结果显示，融合财务绩效和 TOPSIS 后的 F1-score 略高于基础 LSTM，但策略累计收益仍不稳定，说明经营绩效变量可能改善部分方向识别，却尚未形成稳健交易优势。