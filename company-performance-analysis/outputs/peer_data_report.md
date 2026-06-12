# 医药同行公司面板数据获取说明

## 股票池

本次扩展抓取 8 家医药及相关上市公司：

| 股票代码 | 公司名称 | 分类 |
|---|---|---|
| 000661 | 长春高新 | 生物制品 |
| 300122 | 智飞生物 | 疫苗 |
| 300142 | 沃森生物 | 疫苗 |
| 600276 | 恒瑞医药 | 创新药 |
| 000538 | 云南白药 | 中药 |
| 300760 | 迈瑞医疗 | 医疗器械 |
| 603259 | 药明康德 | CXO |
| 688185 | 康希诺 | 疫苗 |

## 数据范围

- 主要时间范围：2019-01-02 至 2026-06-12
- 康希诺因上市较晚，覆盖区间为 2020-08-13 至 2026-06-12
- 合并后日频面板样本数：14,019 行
- 面板建模表字段数：99 列

## 数据内容

已抓取并合并：

- 前复权 OHLCV：open、high、low、close、volume、amount、turnover、pct_change
- 市场指数：沪深300、中证医药、创业板指
- 估值市值：PE、PB、PS、PEG、PCF、总市值、流通市值
- 财务指标：revenue、net_profit、roe、gross_margin、net_margin、asset_liability_ratio、current_ratio 等
- 财报披露日期：当前批量脚本默认使用报告类型规则估算披露日，避免未来财务信息直接泄露
- 预测标签：future_5d_return、future_5d_direction

## 输出文件

- `data/raw/peer_list.csv`
- `data/raw/peers/*_prices_full_qfq.csv`
- `data/raw/peers/*_valuation.csv`
- `data/raw/peers/*_financial_indicators.csv`
- `data/raw/peers/*_disclosure_dates.csv`
- `data/processed/peer_stock_market_features.csv`
- `data/processed/peer_financial_indicators.csv`
- `data/processed/peer_financial_disclosure_dates.csv`
- `data/processed/panel_model_data.csv`
- `outputs/tables/peer_data_fetch_status.csv`

## 后续用途

`panel_model_data.csv` 可用于跨公司 LightGBM、RandomForest、Logistic Regression、面板 LSTM 或消融实验。相比单只长春高新时间序列，面板数据能显著增加样本量，并支持比较“公司经营绩效、估值和行业环境”对未来 5 日方向的影响。
