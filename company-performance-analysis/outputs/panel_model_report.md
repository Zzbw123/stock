# 面板模型与消融实验报告

## 实验设计
使用 8 家医药及相关上市公司构建面板数据，训练集为 2024 年以前，验证集为 2024 年，测试集为 2025 年及以后。验证集用于选择 F1-score 最优分类阈值，测试集只用于最终样本外评估。

特征消融顺序为：base 行情与技术指标；market 加入指数环境；valuation 加入估值市值；financial 加入财务指标。面板模型没有纳入 TOPSIS，因为同行公司尚未统一重算 TOPSIS 得分。

## 核心结果
```text
metric                     Accuracy  F1-score  max_drawdown  strategy_cum_return
model         feature_set                                                       
lightgbm      base           0.4847    0.6271       -0.8022              -0.3065
              financial      0.4726    0.6389       -0.8247              -0.3595
              market         0.4719    0.6398       -0.8252              -0.3638
              valuation      0.4730    0.6405       -0.8239              -0.3514
logistic      financial      0.5419    0.2542       -0.2757               0.1571
random_forest financial      0.5106    0.6222       -0.7331              -0.1417
```

## LightGBM Top 特征
```text
              feature  importance feature_set    model
            macd_hist       448.0   financial lightgbm
                  peg       411.0   financial lightgbm
       volatility_20d       394.0   financial lightgbm
csi_pharma_return_20d       390.0   financial lightgbm
     hs300_return_20d       377.0   financial lightgbm
        volatility_5d       365.0   financial lightgbm
   chinext_return_20d       364.0   financial lightgbm
           boll_width       356.0   financial lightgbm
 csi_pharma_return_5d       347.0   financial lightgbm
          macd_signal       337.0   financial lightgbm
                   pb       319.0   financial lightgbm
                rsi14       315.0   financial lightgbm
                 macd       308.0   financial lightgbm
    chinext_return_5d       300.0   financial lightgbm
                  pcf       288.0   financial lightgbm
```

## 结论
面板 LightGBM 可以作为 LSTM 之外的重要强基准。若 financial 特征集显著优于 valuation 或 market，说明财务信息在跨公司样本中具有增量解释力；若提升有限，则短期方向更多由市场状态、估值和技术面驱动。