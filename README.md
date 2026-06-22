# Single Factor Test · A股单因子检验框架

> 从单因子检验蒸馏出的可复用单因子检验操作系统。覆盖从因子构造到QP建仓的完整链路。

---

## 核心提炼

### 因子定义

```
因子值：(close_t0 / close_{t-5}) - 1   ← 5日动量反转因子
收益率：(open_t2 / open_t1) - 1        ← t+1开盘买入，t+2开盘卖出
```

### 三个关键工程决策

**1. 交易日历对齐（非 `pct_change(5)`）**

停牌缺口会让 `pct_change(5)` 静默跨越超过5个交易日，导致因子值偷看未来。
必须用显式日历映射：

```python
date_plus5 = {all_dates[i]: all_dates[i+5] for i in range(len(all_dates)-5)}
lag5['date'] = lag5['date'].map(date_plus5)
```

**2. 过滤基于 t1 日，不是 t0 日**

买入在 t1 日执行，因此涨跌停/停牌过滤必须基于 t1 日状态（`shift(-1)`），用 t0 日过滤会虚高因子有效性。

| 过滤项 | 判断标准 |
|--------|---------|
| 涨停 | pct_chg_{t+1} ≥ 9.5% |
| 跌停 | pct_chg_{t+1} ≤ -9.5% |
| 停牌 | is_suspend_{t+1} == 1 |
| ST | is_st_{t0} == 1 |
| 新股 | days_listed < 60 |

**3. 截面三步处理流水线**

```
原始因子 → 3σ截断去极值 → Z-Score标准化 → OLS中性化（市值+行业）
```

中性化使用 OLS 残差法，控制规模效应和行业暴露，使 IC 反映因子本身的预测能力。

---

## 使用方法

### 作为 Claude Code Skill 安装

```bash
git clone https://github.com/CharlieBlueberry/single-factor-test.git \
  ~/.claude/skills/single-factor-test
```

安装后，在对话中提到以下关键词即可激活：

- `单因子检验` / `因子回测` / `IC分析` / `分层回测`
- `因子中性化` / `行业中性` / `二次规划建仓`
- `帮我检验一个因子` / `这个因子有没有效`

### Skill 覆盖的六个场景

| 场景 | 内容 |
|------|------|
| A · 因子构造 | 价格对齐、收益率定义、winsorize 参数选择 |
| B · 过滤标准 | A股五类过滤的实现细节与陷阱 |
| C · 处理流水线 | 去极值→标准化→中性化，含完整代码 |
| D · IC分析 | Spearman IC、t检验、评判标准表 |
| E · 分层回测 | 市值加权分层、行业内分层、净值曲线构建 |
| F · QP建仓 | 行业偏离约束、单股上限、协方差估计 |

---

## 扩展方向

- 换收益率窗口（t+3/t+5）观察因子 decay 速度
- 按年分段 IC，判断因子在不同市场环境下的稳定性
- 多因子合成：将本因子与其他 alpha 信号 Z-Score 加权
- 替换 Ledoit-Wolf 协方差估计，提升 QP 建仓稳定性
- 加入换手率惩罚项，控制交易成本

---

## 文件结构

```
single-factor-test/
├── SKILL.md          # Claude Code Skill 主文件（含完整框架、代码、决策规则）
├── images/
│   ├── factor_analysis_4in1.png
│   └── industry_neutral_layer_backtest.png
└── README.md
```
