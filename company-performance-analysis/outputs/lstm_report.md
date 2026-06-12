# 基于 TOPSIS 经营绩效评价与 LSTM 的股价趋势预测研究

## 1. 研究问题
本文在长春高新经营绩效评价基础上，将年频财务指标、熵权 TOPSIS 综合得分与日频行情技术指标结合，比较基础 LSTM 与融合 LSTM 在未来短期趋势预测中的表现。

## 2. 为什么选择 LSTM
股票收益序列具有明显的时间依赖，LSTM 能通过门控结构保留一段历史窗口中的有效信息，适合处理短期动量、均线、波动率等序列特征。

## 3. 预测目标
本研究优先预测未来 5 个交易日累计收益率方向，而不是直接预测绝对股价。方向标签由 future_5d_return 是否大于 0 得到，能降低股价尺度变化和复权误差带来的解释难度。

## 4. 数据来源
日频股价来自 data/processed/stock_prices.csv，补充后的市场特征来自 data/processed/stock_market_features.csv，包含前复权 OHLCV、成交额、换手率、沪深300、中证医药、创业板指、估值和市值字段；财务指标来自 data/processed/financial_indicators.csv；TOPSIS 得分来自 outputs/tables/topsis_scores.csv；财报披露日来自 data/processed/financial_disclosure_dates.csv。

## 5. 特征体系
基础模型使用 open、high、low、close、volume、amount、turnover、收益率、均线、成交量均线、波动率、RSI、MACD、布林带、市场指数收益率、PE/PB/PS 和市值等行情、技术与市场环境变量；融合模型在此基础上加入 revenue、net_profit、roe、gross_margin、net_margin、asset_liability_ratio、current_ratio、revenue_growth、net_profit_growth、topsis_score、rank 等经营绩效变量。

## 6. 财务绩效与 TOPSIS 融合方法
脚本优先使用 financial_disclosure_dates.csv 中的披露日期，通过 merge_asof 将已经披露的最新财务指标和 TOPSIS 得分映射到每个交易日，从而避免在披露日前使用未来财务信息。部分早期年份若 AkShare 未返回实际披露日，则按年报 4 月 30 日、半年报 8 月 31 日、一季报 4 月 30 日、三季报 10 月 31 日进行估算，并在 disclosure_source 中标记。

## 7. 模型结构与数据切分
每个样本使用过去 20 个交易日作为输入窗口。数据按时间顺序切分为约 70% 训练集、15% 验证集、15% 测试集，不进行随机打乱。特征标准化只在训练集拟合 scaler，验证集和测试集仅 transform。

模型采用单层 PyTorch LSTM、Dropout 和 Dense 输出层。分类任务使用 sigmoid 概率和 BCEWithLogitsLoss；回归任务使用线性输出和 MSELoss。脚本还提供小规模遗传算法搜索 hidden_size、dropout 和 learning_rate。

## 8. 实验结果
```text
metric          F1-score
model                   
base            0.039604
fusion          0.000000
naive_momentum  0.333333
```

## 9. 策略收益对比
```text
metric          buy_hold_cum_return  max_drawdown  strategy_cum_return
model                                                                 
base                      -0.882809      0.000000             0.019491
fusion                    -0.882809     -0.050622            -0.050622
naive_momentum            -0.882809     -0.780631            -0.716885
```

## 10. 模型局限性
第一，虽然已补充完整 OHLCV、指数和估值字段，但样本仍只覆盖单一股票，结论容易受个股阶段行情影响。第二，财务指标以年频为主，无法反映报告期内的实时经营变化。第三，部分早期披露日期为规则估算值，仍需用公告原文进一步校验。第四，模型结果不应直接解释为可交易投资建议。

## 11. 后续改进方向
后续可进一步补充公告原文披露时间、分钟级流动性指标、行业成分股横截面样本和分析师预期数据；还可使用滚动窗口验证、交易成本和滑点假设、阈值优化与类别不均衡处理，提高结论稳健性。