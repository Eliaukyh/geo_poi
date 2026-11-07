# 方法实现规格（Method Implementation Spec）

> 面向实现者：本文档描述一套**训练无关（training-free）**的下一 POI 推荐方法。
> 只要给定「用户签到序列 + POI 元数据 + 训练集」，即可在任意 POI 数据集上复现。
> 全文不含任何模型训练步骤，所有统计量均为离线计数。

---

## 0. 方法一句话

把可离线计算的移动统计拆成三条**证据线**，以带距离的事实文本注入提示词；
LLM 在三种呈现视角下各生成一次，用 Borda 聚合；
最后做两步后处理：**合规校验**（修复违反客观约束的输出）与**置信度校准**（保留模型首位判断，尾部按检索统计重建）。

---

## 1. 数据要求

### 1.1 必需字段

| 数据 | 格式 | 说明 |
|---|---|---|
| 签到记录 | `(user_id, poi_id, timestamp)` | 按时间排序 |
| POI 元数据 | `poi_id → (lat, lon, category)` | 类别数不限 |
| 训练/测试划分 | 每个测试样本 = 一段历史 + 一个目标 POI | 见下 |

### 1.2 样本构造

每个评测样本包含：

```
sample = {
  user: int,
  pois: [l_1, ..., l_n],      # 历史 POI 序列（不含目标）
  hours: [h_1, ..., h_n],     # 对应小时 (0-23)
  days: [d_1, ..., d_n],      # 对应天索引（用于时间切分）
  target: l_{n+1}             # 待预测的下一个 POI
}
```

**注意**：若原数据是「连续轨迹」，可按窗口切分（如每 6 天一个样本，取最后一天最后一个 POI 为目标）。
切分方式需与原数据集保持一致，否则无法与已有基线对比。

---

## 2. 离线统计（每数据集一次）

记训练集所有样本展平后的签到序列为 $\{(l_i, t_i)\}$。

### 2.1 转移计数（三张表）

```python
t1  = defaultdict(Counter)   # a -> {b: count}
t1h = defaultdict(Counter)   # (a, bucket) -> {b: count}
t2  = defaultdict(Counter)   # (a_prev, a) -> {b: count}

bucket(h) = h // 3           # 3 小时一个桶

for 每条训练序列:
    for i in range(1, len(seq)):
        a, b, h = seq[i-1], seq[i], hour[i]
        t1[a][b] += 1
        t1h[(a, bucket(h))][b] += 1
        if i >= 2:
            t2[(seq[i-2], a)][b] += 1
```

### 2.2 用户访问分布

```python
user_hist = defaultdict(list)   # user -> [(poi, category, hour), ...]
```

### 2.3 类别—时段先验（Laplace 平滑）

```python
cat_hour = zeros(n_categories, 24)
for each (cat, hour) in user_hist:  cat_hour[cat][hour] += 1
prob(c|h) = (cat_hour[c][h] + 0.5) / (sum_c cat_hour[c][h] + 0.5 * n_categories)
```

### 2.4 轨迹 TF-IDF 索引

- 每条训练轨迹 → 字符串 `" ".join(map(str, poi_ids))`
- 以 TF-IDF 拟合，得到轨迹向量矩阵
- 需要同时保存每条训练轨迹的**下一站** `rec[j]`

---

## 3. 证据构造（MMEG）

给定样本，构造三条证据线，每条取 **top-8**。

### 3.1 转移视图（协同信号 · 局部）

$$
s_{\text{tr}}(v)=w_1\hat P(v\mid l_n,b)+w_3\hat P(v\mid l_{n-1},l_n)
$$

- $\hat P(v\mid l_n,b)=t1h[(l_n,b)][v]/\sum_{v'}t1h[(l_n,b)][v']$，二阶项同理
- **默认 $w_1=1.0,\;w_3=1.5$**
- 可选的第三项（无条件下阶 $w_2\hat P(v\mid l_n)$）实测可省，$w_2=0$ 即可
- 若 $(l_n,b)$ 与 $(l_{n-1},l_n)$ 都为空，该视图可返回空

**实测**：该视图是三条证据中最强的单一信号，移除后 H@10 从 68.5 掉到 35.0。

### 3.2 个体视图（个体信号 · 全局）

$$
s_{\text{ind}}(v)=\log\big(1+n_u(v)\big)\cdot\big(0.25+\hat P(c(v)\mid h)\big)\cdot\rho
$$

- $n_u(v)$：用户对 $v$ 的历史访问次数
- $\rho=0.6$ 若 $v$ 出现在最近序列中，否则 1

### 3.3 类比视图（协同信号 · 全局）

1. 把当前轨迹文本与训练轨迹矩阵做余弦相似度
2. 取相似度最高的 **50 条**训练轨迹
3. 把它们的下一站按相似度累加投票：

$$
s_{\text{sem}}(v)=\sum_{j\in\mathcal{N}_{50}}\mathrm{sim}(S_u,S_j)\cdot\mathbb 1[r(S_j)=v]
$$

4. 取 top-8

### 3.4 空间属性（不是独立证据线）

**不要**把「附近的 POI」单列成一行。距离以字段形式附在每条候选后。

> 实测：把它作为独立证据行会使 H@10 从 76.0 降到 66.0（−10pt）。

### 3.5 序列化格式

```
- After POI {l_n}, people most often went to: {p} ({cat}, {d:.2f}km); ...
- This user has visited most often: {p} ({cat}); ...
- Users with similar recent trajectories next visited: {p} ({cat}, {d:.2f}km); ...
```

三条约束：

1. 附数值（距离 km），使空间推理由本地计算承担；
2. 写成**事实陈述**，不要写成候选菜单（不出现 `[C1]`、`id=` 等）；
3. 提示词中明确允许推荐证据之外的 POI。

---

## 4. 用户画像（可选）

**输入统计**（本地计算，不调 LLM）：

```
Total check-ins: {n}
Most frequent places: {top-10 POI, 类别, 次数}
Category mix: {前 4 类占比}
Time-of-day mix: {00-05 / 06-11 / 12-17 / 18-23 四段占比}
```

**输出**（一次调用，按用户缓存）：

```json
{"profile": "≤60 词的行为画像（含回访/探索倾向）", "routine": "一句作息描述"}
```

**实测**：单独增益 +0.3pt（不显著）。若追求简洁可以省略。

---

## 5. 提示词模板（三个变体）

### 5.1 System

```
You are an expert urban mobility analyst. You predict the next point of interest
of a user from their check-in trajectory and reference evidence. Answer with JSON only.
```

### 5.2 User（变体 A，默认顺序）

```
A user in {city} is moving through the city. Predict their next point of interest.

Recent check-ins (oldest to newest): {recent trajectory, 如 39 (Food); 954 (Shop)}
Current hour of day: {h}:00
{profile}

Reference evidence retrieved from historical check-ins (use it as a hint; you may also rely on your own knowledge of this city):
{evidence block}

Instructions:
1. Recommend exactly 10 distinct POI ids, most likely first.
2. Use POI ids that exist in this city. You are not limited to the ids listed above.
3. Prefer places that are plausible given the hour, the recent trajectory and the user's habits.
4. Do not include line breaks inside the JSON.

Return JSON only: {"recommendation": [10 integer ids, most likely first], "reason": "one short sentence"}
```

### 5.3 三个变体

| 变体 | 扰动 | 做法 |
|---|---|---|
| **A** | 无 | 上述默认提示词 |
| **B** | 输入顺序 | 把三行证据**倒序**，并加一句 `Note: the evidence below is ordered by personal relevance first.` |
| **C** | 行为先验 | 加角色前缀 `You are a long-time resident of {city} who knows its neighbourhoods, opening patterns and daily rhythms well. Think about the time of day and the user's routine before deciding.` |

三个变体**必须**共享：证据内容、任务指令、解码参数（温度 0、JSON 模式）。否则实验结果无法归因。

**实测**：变体 C（角色）的独立贡献不显著；收益主要来自 B（顺序扰动）。

---

## 6. 生成与解析

- 温度 0，`response_format={"type":"json_object"}`，`max_tokens` 约 900
- 解析 `recommendation` 字段：转 int、去重、丢弃非整数
- 三个变体各生成一次，得到 $L_A, L_B, L_C$（各 ≤10 项）

---

## 7. 共识聚合（Borda）

$$
\text{score}(v)=\sum_{k\in\{A,B,C\}}\big(10-\text{rank}_k(v)\big)\cdot\mathbb 1[v\in L_k]
$$

取 top-10，记为 $L_0$。

**为什么不用概率加权**：LLM 自报置信度不可用（实测 93.7% 样本自报 ≥0.7）。

---

## 8. 候选池（供修复/校准使用）

用同一套检索构造一个**纯统计**的 10 项有序列表：

```
① 转移视图分数归一化 → 取前 8 作为头部
② 个体视图、类比视图分数归一化后按权重加和（个体 1.0，类比 0.5）
③ 从辅助候选中按分数取未出现在头部的，填满剩余 2 个位置
④ 截断为 10
```

记为 `pool`。

**这个池子是系统性能的上界**：直接输出 pool 的 top-10 即为纯统计方法的成绩。

**在实现前务必先测这个基线**——它可能已经很强（我们在 Foursquare 上测到 H@10 ≈ 0.81）。

---

## 9. CARE-I：合规校验

### 9.1 规则引擎（确定性）

| 规则 | 判据 | 说明 |
|---|---|---|
| R1 | 恰好 10 项、互异、ID 在词表内 | 格式合法性 |
| R2 | 与当前位置直线距离 ≤ 50 km | 物理可达 |
| R3 | 估计行程 ≤ 120 min | 物理可达 |
| R4 | 类别在该时段的先验 ≥ 0.02 | 时间合理性 |

R4 是唯一高频触发的规则（实测约 20% 的样本）。典型错误：正午推荐夜店。

### 9.2 Judge（LLM）

输入：候选逐项事实 + 规则引擎标记。
输出：

```json
{"verdict": "pass" | "fix",
 "issues": [{"kind": "cat_hour", "poi": 1076, "detail": "Nightlife at 11h (p=0.017)"}]}
```

**约束**：Judge 只能补充、不能否定规则引擎的判定。

### 9.3 Fixer（LLM）

输入：当前列表 + 违规清单 + 替换候选池（15 项）。
输出：修复后的 10 项列表。

**约束**：
- 只替换被标记的项，其余位置不动；
- 替换项只能来自给定池（禁止凭空生成）；
- **第 1 位永不替换**。

### 9.4 验收

- 计算修复前后的违规数 $m_{\text{before}}, m_{\text{after}}$
- **仅当 $m_{\text{after}} < m_{\text{before}}$ 时接受**，否则回滚
- 最多 2 轮

**这条单调性保证闭环不会使结果更违规。**

---

## 10. CARE-II：置信度校准

### 10.1 动机（用数据说明，务必先验证）

在目标数据集上检查两件事：

1. **命中来源**：输出中每个 POI 来自哪条证据线？命中主要来自哪条？
   （我们测到：转移视图占输出 45%，却贡献 89% 的命中）
2. **分位命中率**：当目标在转移榜第 1/2/3-8/更后时，模型命中率分别是多少？
   （我们测到：rank1 100%、rank2 100%、rank3-8 56%）

若上述现象成立（模型头部可靠、尾部不可靠），则适用校准；否则跳过本步骤。

### 10.2 做法

```python
def calibrate_tail(ranking, pool, keep=1):
    head = ranking[:keep]                       # 信任模型的首位
    tail = [p for p in pool if p not in head]   # 尾部按统计顺序
    return list(dict.fromkeys(head + tail))[:10]
```

**默认 keep=1**。

### 10.3 边界情况

| 情况 | 处理 |
|---|---|
| pool 不足 9 项 | 用模型输出的剩余项按原顺序补足 |
| 模型首位不在 pool 中 | 保留首位，不强制替换 |
| 首位在 pool 中 | 从 pool 中剔除首位后取前 9 |
| 列表不足 10 项 | 先由 §9 的有效性修复补足，再校准 |

### 10.4 实测效果（Foursquare NYC）

| 配置 | H@1 | H@5 | H@10 | N@10 |
|---|---|---|---|---|
| 无校准 | 0.358 | 0.657 | 0.765 | 0.552 |
| 有校准 | 0.360 | 0.747 | 0.808 | 0.605 |

配对检验：H@10 +4.3pt（CI [+2.2,+6.5]），NDCG +5.3pt，均显著；**H@1 不变**（首位保护）。

---

## 11. 完整流程

```
输入: 轨迹 S_u, 目标时刻 t

# 离线（每数据集一次）
T1, T1h, T2 ← 转移计数;  user_hist;  P(c|h);  TF-IDF

# ① 证据
E ← [transition_top8, memory_top8, semantic_top8]   # 带距离的事实文本
P_u ← profile(stats)                                 # 可选，按用户缓存

# ② 生成
L ← Borda([ LLM(prompt_A), LLM(prompt_B), LLM(prompt_C) ])

# ③ 合规
for r in 1..2:
    I ← rules(L) ∪ judge(L)
    if I is empty: break
    L' ← fixer(L, I, pool);  L'[1] ← L[1]
    if |issues(L')| < |issues(L)|: L ← L'  else break

# ④ 校准
L ← calibrate_tail(L, pool, keep=1)

# ⑤ 兜底
L ← dedup(L);  不足 10 项则由 pool 补足
return L[:10]
```

---

## 12. 默认超参

| 参数 | 值 | 敏感性 |
|---|---|---|
| 每视图证据条数 | 8 | **不敏感**（2–16 间指标平坦） |
| 转移权重 $w_1{:}w_3$ | 1 : 1.5 | **不敏感**（1:1 到 1:50 波动 <2.5pt） |
| 时段桶 | 3 小时 | 未系统扫描 |
| 转移/类比 top-k | 20 / 50 轨迹 | 未系统扫描 |
| 温度 | 0 | — |
| 校准信任位 keep | 1 | keep 越大 H@10 略升、NDCG 下降 |
| 规则阈值 | 50km / 120min / 0.02 | 只有 0.02 高频触发 |

---

## 13. 必须报告的基线

**在实现本方法前，先跑下面两个基线**，否则会误判方法的实际价值：

| 基线 | 实现 | Foursquare NYC H@10 |
|---|---|---|
| **Markov（时段条件二阶）** | 直接取 §3.1 转移视图的 top-10 | **0.775** |
| **候选池 top-10** | §8 的 pool 直接输出 | **0.808** |

若新数据集上 Markov 基线已经很高，说明该任务的可预测性由短期转移主导，LLM 的增量可能为负——这与我们在 Foursquare 上的发现一致（未校准的 LLM 管线 0.765 < Markov 0.775）。

---

## 14. 已知陷阱

1. **不要用「无证据」的结果当作「无模块」的结果。** 无证据但保留候选池时，指标会被池子兜底；要测模块价值必须同时移除池子。
2. **带校准的消融会掩盖证据的作用。** 校准从池子重建尾部，使 H@10 对证据缺失不敏感；要测证据价值必须关闭校准，看 H@1 与 NDCG。
3. **初始化 LLM 的思考模式。** Qwen3.x 默认输出超长推理链，需在请求中传 `enable_thinking: false`，否则连接可能超时断开。
4. **样本量小时注意粒度。** n=200 时单条样本 = 0.5 个百分点，小于 2–3pt 的差异不可靠。
5. **运行间波动。** 即使温度 0，同一配置两次运行可差 0.5–2pt；所有对比必须在同一轮内配对。
