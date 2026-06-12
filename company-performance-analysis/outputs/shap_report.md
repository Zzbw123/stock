# SHAP 特征贡献分析报告

解释模型：LightGBM 代理分类器，特征集合：fusion。

## 样本外代理模型指标
```text
            model           metric    value
shap_proxy_fusion         Accuracy 0.580769
shap_proxy_fusion        Precision 0.466019
shap_proxy_fusion           Recall 0.470588
shap_proxy_fusion         F1-score 0.468293
shap_proxy_fusion DirectionHitRate 0.580769
```

## 全局 Top 10 特征
```text
           feature  mean_abs_shap
               peg       0.527686
        boll_width       0.309476
               ma5       0.224772
     volatility_5d       0.221256
         pe_static       0.189239
  hs300_return_20d       0.186653
         return_5d       0.175327
chinext_return_20d       0.158910
    volatility_20d       0.123551
        boll_lower       0.120645
```

## 重点关注变量
```text
              feature  mean_abs_shap
     hs300_return_20d       0.186653
           net_profit       0.070711
               pe_ttm       0.041768
              revenue       0.039589
csi_pharma_return_20d       0.035833
                   pb       0.020797
                   ps       0.014713
         topsis_score       0.003302
                  roe       0.002586
                 rank       0.000000
```

## 解释口径
SHAP 值为正表示该特征在该样本上推高未来 5 日上涨概率，SHAP 值为负表示压低上涨概率。该分析解释的是 LightGBM 代理模型，不等同于直接解释 LSTM 隐状态，但可作为特征体系贡献判断和消融实验设计依据。