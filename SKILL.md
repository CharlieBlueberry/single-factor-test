---
name: single-factor-test
description: |
  A股单因子检验全流程框架。基于实战代码蒸馏，覆盖因子构造→过滤→标准化→中性化→IC检验
  →分层回测→行业中性分层→二次规划建仓的完整链路。
  内置决策规则：交易日历对齐、三类过滤标准、3σ去极值、Spearman IC、市值加权分层、
  行业偏离约束的QP优化。
  当用户提到「单因子」「因子检验」「IC」「分层回测」「因子中性化」「alpha因子」
  「回测框架」「因子有效性」「IC_IR」「行业中性」「二次规划建仓」时使用。
  即使用户只说「帮我检验一个因子」「这个因子有没有效」「怎么做因子回测」也应触发。
---

# 单因子检验 · 工程操作系统

> 「因子不是发现的，是在噪音里挣出来的。」

## 定位

**我能帮你的**：因子设计、过滤逻辑、标准化/中性化选择、IC分析、分层回测、QP建仓
**我不替代的**：数据获取、因子的经济直觉来源、实盘滑点/冲击成本

---

## 问题路由

| 用户问题 | 执行场景 |
|---------|---------|
| 设计/定义一个新因子 | → 场景A：因子构造 |
| 数据清洗/过滤怎么做 | → 场景B：过滤标准 |
| 因子标准化/中性化 | → 场景C：处理流水线 |
| IC检验/统计显著性 | → 场景D：IC分析 |
| 分层回测怎么做 | → 场景E：分层回测 |
| 建仓/组合优化 | → 场景F：QP建仓 |
| 全流程搭建 | → 顺序执行A→B→C→D→E→F |

---

## 场景A：因子构造

### 关键决策：用什么价格/时间对齐

```
收益率定义（避免未来数据）：
  ret = open_{t+2} / open_{t+1} - 1
  ✅ t+1开盘买入（避免t+1涨跌停无法成交）
  ✅ t+2开盘卖出（避免t日收盘→t+1开盘的隔夜跳空干扰）
  ❌ 不用 close_t / close_{t+1}（包含了当日收盘后的信息）

因子值的时间对齐（核心陷阱）：
  必须用「交易日历」计算滞后，不能用 pct_change(5)
  原因：停牌/退市会导致跨越超过5个自然交易日
  
  正确做法：
    all_dates = sorted(daily['date'].unique())
    date_plus5 = {all_dates[i]: all_dates[i+5] for i in range(len(all_dates)-5)}
    lag5['date'] = lag5['date'].map(date_plus5)  # 原 t-5 → 打标签到 t0
```

### 因子值预处理（构造阶段）

```python
# Winsorize：剔除复权跳变等数据噪音（构造阶段只做粗剪）
factor = raw_factor.clip(-0.5, 0.5)
# 注意：截面精细去极值在场景C做，这里只剪掉数据异常
```

---

## 场景B：过滤标准（A股特有）

### 五类必须过滤

| 过滤项 | 过滤逻辑 | 为什么 |
|--------|---------|--------|
| t1日涨停 | pct_chg_{t+1} ≥ 9.5% | 涨停无法买入，信号无法执行 |
| t1日跌停 | pct_chg_{t+1} ≤ -9.5% | 跌停无法卖出，信号无法执行 |
| t1日停牌 | is_suspend_{t+1} == 1 | 停牌无法成交 |
| ST股 | is_st_{t0} == 1 | 特殊处理，退市风险 |
| 上市<60日 | days_listed < 60 | 新股效应，不稳定 |

### 实现要点

```python
# t1的状态需要shift(-1)映射到t0行
daily['t1_up']   = g['pct_chg'].shift(-1) >= 0.095
daily['t1_down'] = g['pct_chg'].shift(-1) <= -0.095
daily['t1_sus']  = g['is_suspend'].shift(-1).fillna(0)

# 注意：科创板/创业板用20%涨跌停，需要分别处理
# 简化版：统一用9.5%（保守，宁可多过滤）
```

### 截面有效标的数检查

```python
# 每个截面至少需要30只股票才有统计意义
# 分层回测需要每层至少有5-10只
if len(cur) < 30:
    continue
```

---

## 场景C：因子处理流水线

### 三步流水线（每个截面独立执行）

```
原始因子
  → [Step 1] 截面去极值（3σ截断）
  → [Step 2] 截面标准化（Z-Score）
  → [Step 3] 中性化（可选，OLS残差）
  → 处理后因子
```

### Step 1：去极值

```python
mu_f, sd_f = f.mean(), f.std()
f_winsor = f.clip(mu_f - 3*sd_f, mu_f + 3*sd_f) if sd_f > 1e-10 else f
# 注意：是截面内3σ，不是全样本
```

### Step 2：Z-Score标准化

```python
def zscore(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 1e-10 else s * 0.0
# 防止sd≈0时除零（单日截面可能出现）
```

### Step 3：中性化（OLS残差法）

```python
def neutralize(f, mc, ind):
    lmc = np.log(mc.clip(lower=1.0))           # log市值
    dummies = pd.get_dummies(ind, drop_first=True).astype(float)  # 行业哑变量
    X = pd.concat([lmc.rename('lmc'), dummies], axis=1)
    X = add_constant(X, has_constant='add')
    df = pd.concat([f.rename('y'), X], axis=1).dropna()
    if len(df) < 10:
        return pd.Series(np.nan, index=f.index)
    return OLS(df['y'], df.drop(columns='y')).fit().resid.reindex(f.index)
```

### 何时需要中性化

| 场景 | 建议 |
|------|------|
| 因子与市值/行业高度相关 | 必须中性化，否则IC来自规模效应 |
| 做行业内选股 | 中性化后的因子更纯净 |
| 纯动量/技术类因子 | 先看Raw IC，如果Raw≈Neu说明没有风格偏差 |
| 快速验证阶段 | 先跑Raw，再跑Neu，对比判断 |

---

## 场景D：IC分析

### Spearman IC（首选）

```python
def spearman_ic(f, r):
    tmp = pd.concat([f, r], axis=1).dropna()
    if len(tmp) < 10:
        return np.nan
    return stats.spearmanr(tmp.iloc[:,0], tmp.iloc[:,1])[0]
# 用Spearman而非Pearson：对极端值鲁棒，A股小盘股容易出极端收益
```

### IC显著性检验

```python
a = np.array(ic_list, dtype=float)
a = a[~np.isnan(a)]
n = len(a)
mu = a.mean()
sd = a.std(ddof=1)
IR = mu / sd          # 信息比率：越高越好，>0.5值得关注
t  = mu / (sd / np.sqrt(n))
p  = 2 * stats.t.sf(abs(t), df=n-1)
```

### IC评判标准

| 指标 | 弱信号 | 可用信号 | 强信号 |
|------|--------|---------|--------|
| \|IC均值\| | <0.02 | 0.02-0.05 | >0.05 |
| IC_IR | <0.3 | 0.3-0.5 | >0.5 |
| p值 | >0.05 | 0.01-0.05 | <0.01 |
| IC>0占比 | <52% | 52-58% | >58% |

### Cum IC图的解读

```
斜率持续为正 → 因子有持续alpha
频繁震荡/斜率趋于0 → 因子衰减
斜率周期性 → 因子有择时特征，不适合全天候使用
```

---

## 场景E：分层回测

### 市值加权分层（推荐）

```python
def layer_ret(f, r, n=5, w=None):
    tmp = pd.concat([f.rename('f'), r.rename('r'),
                     w.rename('w') if w is not None else pd.Series()], axis=1).dropna()
    if len(tmp) < n:
        return np.full(n, np.nan)
    tmp['lyr'] = pd.qcut(tmp['f'].rank(method='first'), n, labels=False)
    # 市值加权
    def wavg(g):
        wt = g['w'] / g['w'].sum()
        return (g['r'] * wt).sum()
    return tmp.groupby('lyr').apply(wavg).reindex(range(n)).values
# layer 0 = 因子最小（空头端），layer 4 = 因子最大（多头端）
```

### 行业内分层（去行业暴露）

```python
# 在每个行业内部独立分层，再按行业市值权重加权汇总
ind_w = cur.groupby('industry')['mkt_cap'].sum()
ind_w = ind_w / ind_w.sum()

agg, tw = np.zeros(5), 0.0
for iname, grp in cur.groupby('industry'):
    if len(grp) < 5: continue       # 行业内至少5只
    lr = layer_ret(grp['fz'], grp['ret'], w=grp['mkt_cap'])
    if np.isnan(lr).any(): continue
    w = ind_w.get(iname, 0.0)
    agg += lr * w; tw += w
```

### 净值曲线构建

```python
# cumprod（净值曲线）而非cumsum（收益率累加）
cum = (1 + layer_ret_df.fillna(0)).cumprod()
# fillna(0)：无效截面当天收益为0，不是nan
```

### 分层回测评判

| 观察点 | 好因子的表现 |
|--------|------------|
| 层间单调性 | layer 0→4净值应单调递增（或递减，取决于因子方向） |
| 多空对称性 | layer 4与layer 0的走势方向相反且幅度接近 |
| 极端层稳定性 | layer 4（或0）的Sharpe应显著>1 |
| 中间层 | 可以乱，极端层单调就够 |

---

## 场景F：二次规划建仓（QP）

### 适用场景

在因子有效性确认后，对最新截面建组合：
- 选因子得分最高（或最低）的N只股票
- 在控制行业偏离的前提下，最小化组合方差

### 标准QP框架

```python
# 目标：min 0.5 * w^T Σ w
# 约束：
#   sum(w) = 1                   权重归一
#   w @ mu >= target_return      收益约束（因子得分均值）
#   |行业权重 - 基准权重| <= DEV  行业偏离约束（±10%）
#   0 <= w_i <= 0.10              单股上限

from scipy.optimize import minimize

cons = [
    {'type': 'eq',   'fun': lambda w: w.sum() - 1},
    {'type': 'ineq', 'fun': lambda w: w @ mu_v - tgt},
]
for ind in set(inds):
    idx = [i for i, x in enumerate(inds) if x == ind]
    b = float(bench.get(ind, 0.0))
    cons += [
        {'type': 'ineq', 'fun': lambda w, i=idx, b=b: w[i].sum() - (b - DEV)},
        {'type': 'ineq', 'fun': lambda w, i=idx, b=b: (b + DEV) - w[i].sum()},
    ]

res = minimize(lambda w: 0.5 * w @ Sig @ w,
               np.ones(N) / N, method='SLSQP',
               bounds=[(0.0, 0.10)] * N,
               constraints=cons,
               options={'maxiter': 3000, 'ftol': 1e-9})
```

### 协方差矩阵估计

```python
# 用最近60个交易日的历史收益率估计
# 对称化 + 正则化（防止奇异）
Sig = piv.cov().values
Sig = (Sig + Sig.T) / 2 + np.eye(N) * 1e-5
```

### QP结果检查

```python
print(f"状态: {res.message}")           # 必须是 'Optimization terminated successfully'
print(f"最大偏离={偏离.abs().max():.4f}")  # 应 ≤ DEV
print(f"收益={w @ mu_v:.6f}")            # 应 ≥ tgt
print(f"年化波动={np.sqrt(w @ Sig @ w * 252):.4f}")
```

---

## 核心心智模型

### 模型1：时间对齐是因子测试的生死线

**一句话**：pct_change(5) 在有停牌缺口的数据上会静默跨越超过5个交易日，导致因子值偷看未来。

**正确做法**：用交易日历 `date_plus5` 字典显式对齐，每一步 shift 都要思考「这步操作有没有用到未来信息」。

**检验方法**：因子IC异常高（>0.1均值）→ 优先怀疑未来数据泄露。

---

### 模型2：过滤逻辑的先后顺序影响结果

**一句话**：过滤必须基于 t1 日的状态，不是 t0 日。

**原因**：因子在 t0 日计算，买入在 t1 日开盘执行，如果 t1 日涨停/停牌，信号根本无法执行。用 t0 日的状态过滤等于「能执行再过滤」，虚高了因子有效性。

---

### 模型3：Raw IC vs Neutralized IC 的对比本身就是信息

**一句话**：如果 Raw IC ≈ Neutralized IC，说明因子纯净；如果 Raw >> Neu，说明 alpha 来自风格暴露（规模/行业），不是因子本身。

**决策规则**：
- Neu IC 显著 → 因子有真实 alpha，可以直接用
- Raw 显著但 Neu 不显著 → 因子是规模/行业代理，需要重新设计
- 两者都不显著 → 因子无效

---

### 模型4：分层单调性 > IC均值

**一句话**：IC 均值告诉你因子是否有效，分层单调性告诉你因子能不能赚钱。

**原因**：IC 是线性相关系数，但实际收益来自极端分位数。一个因子可以 IC 均值只有 0.03，但最高分位的组合年化超额 15%，这才是可用的因子。

---

### 模型5：协方差矩阵估计决定QP质量

**一句话**：用多少天的历史数据估计 Σ 是重要的超参数，60天是经验值，波动率高的市场环境应缩短。

**实践规则**：
- 股票数 N 远大于历史天数 T → Σ 奇异，必须加正则化 `+ np.eye(N) * 1e-5`
- 样本量不足（N > T/3）→ 考虑缩减估计（Ledoit-Wolf）替代历史 Σ

---

## 决策启发式

1. **截面独立处理**：去极值、标准化、中性化全部在每个截面内独立执行，不跨截面。

2. **factor.clip(-0.5, 0.5) 在构造阶段**：这不是标准化，是剔除复权等数据异常，宽松一些没关系。

3. **市值加权优于等权**：等权分层受小盘股波动影响大，结果虚高；市值加权更接近实际可执行收益。

4. **行业内至少5只才参与加权**：行业只有1-2只时权重过大，引入噪音，直接跳过。

5. **valid_dates 最后2个截面不用**：因为需要 t+2 的 open，最后几个截面的 ret 是 NaN。

6. **QP选最小因子端（nsmallest）做空头**：如果因子方向是反转类（因子小的股票收益高），选 nsmallest。

7. **IC序列中 nan 在统计前过滤**：截面有效标的不足时 IC 为 nan，统计均值/IR 前必须 `a[~np.isnan(a)]`。

8. **OLS中性化用 drop_first=True**：行业哑变量需要去掉一列避免多重共线，且要加 `add_constant`。

---

## 🛑 检查点

### 因子构造后必答
1. **时间对齐方式**：用的是交易日历对齐，还是 `pct_change(n)`？
2. **未来数据检查**：每次 shift 的方向是否确认不会读到未来？
3. **极端值来源**：IC 异常高（>0.08）→ 先查数据，不要庆祝。

### 分层回测后必答
1. **层间单调性**：5层净值从低到高（或高到低）是否单调？
2. **Raw vs Neu 对比**：两者差异大吗？差异来自哪里（规模还是行业）？
3. **有效截面数**：有多少个截面被 `len(cur) < 30` 过滤掉了？太多说明数据质量有问题。

### QP建仓后必答
1. **优化器状态**：`res.message` 是否为 `Optimization terminated successfully`？
2. **行业偏离**：最大偏离是否 ≤ DEV（10%）？
3. **收益约束满足**：`w @ mu_v` 是否 ≥ tgt？

---

## 常见问题与 Fallback

| 问题 | 原因 | 解决 |
|------|------|------|
| IC 均值异常高（>0.08） | 未来数据泄露 / 数据质量问题 | 逐步检查 shift 方向；验证 close_lag5 对齐 |
| 分层不单调但 IC 显著 | 因子对中间分位数噪音大 | 只看极端两层，忽略中间层 |
| QP 优化失败 | 协方差矩阵奇异 / 约束不可行 | 增大正则化系数；放宽 DEV 约束；减少候选股数量 |
| 行业内分层所有截面 tw≈0 | 行业分组后每组 <5 只 | 放宽最小组内数量阈值；或换用更粗的行业分类 |
| 中性化后 IC 变负 | 因子方向跟规模负相关 | 加负号翻转因子方向再做中性化 |
| cum IC 斜率到后期变平 | 因子 alpha 衰减 | 检查 IC 按年分段统计，确认哪一段开始衰减 |

---

## 完整流水线 Checklist

```
[ ] 1. 数据加载：daily / industry / ST / 停牌
[ ] 2. 交易日历构建：date_plus5 映射
[ ] 3. 因子计算：用日历对齐 close_lag5，raw factor + clip
[ ] 4. 收益率：open_t2 / open_t1 - 1
[ ] 5. 过滤标记：涨跌停 / 停牌 / ST / 上市天数 / 行业（全部基于 t1 日）
[ ] 6. MultiIndex 设置：(date, code)
[ ] 7. 截面循环：
    [ ] 7a. 过滤 _keep
    [ ] 7b. 3σ去极值 + Z-Score
    [ ] 7c. 中性化（OLS残差）
    [ ] 7d. Spearman IC（Raw + Neu）
    [ ] 7e. 市值加权分层
    [ ] 7f. 行业内分层
[ ] 8. IC t检验：均值 / IR / t / p
[ ] 9. 画图：Cum IC + 分层净值（Raw / Neu / 行业）
[ ] 10. QP建仓（可选）：选股 → 估计Σ → 优化 → 检查状态
```

---

## 扩展方向

| 方向 | 说明 |
|------|------|
| 换收益率窗口 | 测试 t+3/t+5 开盘，观察因子 decay |
| 分时段IC | 按年或按牛熊市分段统计，判断因子稳定性 |
| 因子合成 | 多因子 Z-Score 加权合成，观察 IC 提升 |
| Ledoit-Wolf 协方差 | 替代历史 Σ，QP 结果更稳定 |
| 换手率约束 | QP 中加入换手率惩罚项，降低交易成本 |

---

**创建时间**：2026年6月
**代码来源**：谭思宇_最终提交.py — 单因子检验实战代码蒸馏
