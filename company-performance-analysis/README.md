# 上市公司绩效评价与可视化展示

本项目用于数据科学课程作业：以 **长春高新（000661.SZ）** 为例，构建财务指标体系，使用熵权法和 TOPSIS 评价经营绩效，并导出图表和 Streamlit 看板。

## 项目结构

```text
company-performance-analysis/
├── data/
│   ├── raw/                  # 手动导入 CSV/Excel 或示例数据
│   └── processed/            # 清洗后的指标矩阵
├── src/
│   ├── fetch_data.py         # 手动导入 + AkShare 获取数据
│   ├── indicators.py         # 财务指标计算
│   ├── entropy_topsis.py     # 熵权法 + TOPSIS
│   ├── visualization.py      # 图表绘制
│   ├── lstm_predict.py       # 可选 LSTM 股价预测
│   └── report_charts.py      # 一键导出表格和图表
├── outputs/
│   ├── figures/              # 导出的 PNG 图表
│   ├── tables/               # 权重、得分、排名 CSV
│   └── report_data.xlsx      # 汇总工作簿
├── app.py                    # Streamlit 看板
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

AkShare 会联网获取公开行情和财务指标。如果本机无法安装或接口临时不可用，可以直接使用手动 CSV/Excel。

## 数据格式

手动财务表至少需要包含期间列，例如 `period`、`年份`、`报告期` 或 `日期`。支持以下常见字段，中文或英文列名均可：

- 营业收入：`revenue` / `营业收入` / `营业总收入`
- 净利润：`net_profit` / `净利润` / `归母净利润`
- 总资产：`total_assets` / `总资产` / `资产总计`
- 总负债：`total_liabilities` / `总负债` / `负债合计`
- 股东权益：`equity` / `股东权益` / `所有者权益`
- 现金流：`operating_cash_flow` / `经营活动现金流量净额`
- 财务比率：`roe`、`roa`、`gross_margin`、`net_margin`、`asset_liability_ratio`、`current_ratio`

如果部分比率缺失，程序会根据基础字段自动推导。

## 一键运行

使用教学演示数据跑通流程：

```bash
python src/report_charts.py --demo
```

使用 AkShare 获取长春高新数据：

```bash
python src/report_charts.py --use-akshare --symbol 000661 --start-date 20190101
```

使用手动数据：

```bash
python src/report_charts.py --financial-file data/raw/changchun_gaoxin_financials.xlsx --price-file data/raw/changchun_gaoxin_prices.csv
```

## 输出结果

运行后会生成：

- `outputs/tables/entropy_weights.csv`：熵权法指标权重表
- `outputs/tables/topsis_scores.csv`：综合绩效得分与排名
- `outputs/report_data.xlsx`：汇总工作簿
- `outputs/figures/`：趋势图、雷达图、权重柱状图、TOPSIS 得分图、股价与绩效对比图

## Streamlit 看板

```bash
streamlit run app.py
```

看板支持上传 CSV/XLSX，也可以使用教学演示数据预览流程。

## 方法说明

1. 构建指标体系：规模、盈利能力、偿债能力、现金流质量与成长性指标。
2. 指标正向化：资产负债率为负向指标，其余默认正向指标。
3. 标准化：使用 min-max 标准化到 `[0, 1]`。
4. 熵权法：信息差异越大的指标权重越高。
5. TOPSIS：计算各年度与理想解和负理想解的距离，得到综合绩效得分。

## 说明

`data/raw/changchun_gaoxin_demo_*.csv` 是教学演示数据，只用于验证代码流程。正式作业请使用 AkShare 或从年报、财报平台整理出的真实数据，并在论文/报告中注明来源和日期。
