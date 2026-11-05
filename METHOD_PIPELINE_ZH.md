# 方法（变体）：约束式候选选择 + LLM 排序

> **说明**：本文档描述的是**约束式选择**变体——LLM 只能在 25 个候选内排序。
> 当前主方法已改为**生成式**（LLM 自由生成 POI，证据只作参考），
> 见 [`PAPER_ZH.md`](PAPER_ZH.md)。
> 本文件保留作为消融材料：约束式在 NYC 上 HR@10 = 0.808，生成式为 0.562，
> 差值 0.246 即"允许自由生成"的代价。

> 本文档描述**完整处理链路**，对应代码 `src/grpoi/`。文中每个阶段都标注了实现位置与
> 输入输出，所有数值来自 `results/` 下的实测（NYC/test，n=600，全量）。

---

## 3.1 问题定义与记号

设用户 $u$ 的轨迹为 $S_u = \{(l_1,t_1),\dots,(l_n,t_n)\}$，其中 $l_i \in \mathcal{P}$ 为 POI，
$t_i$ 为签到时间戳，$\mathcal{P}$ 为全城 POI 集合（NYC：$|\mathcal{P}|=15{,}754$）。

给定 $S_u$ 与目标时刻 $t_{n+1}$，预测 $l_{n+1}$。系统输出一个长度为 10 的列表
$\hat{L} = [\hat{l}_1,\dots,\hat{l}_{10}]$，按概率降序排列。

每个 POI 附带两类静态属性：地理坐标 $g(l)=(\text{lat},\text{lon})$ 与类别 $c(l)\in\mathcal{C}$（$|\mathcal{C}|=10$）。

训练集 $\mathcal{D}_{\text{train}}$ 为已观测轨迹的集合，系统**不更新任何参数**，仅从中统计计数。

---

## 3.2 框架总览

```
                    ┌──────────────── 离线统计（一次）────────────────┐
                    │ 转移计数 · 用户画像 · 类别-时段先验 · TF-IDF 索引 │
                    └───────────────────────┬────────────────────────┘
                                            │
 输入轨迹 S_u ──────────────────────────────▼──────────────────────────────
                                   ① 四路候选生成
                    ┌──────────────┬──────────────┬──────────────┬─────────┐
                    │ 转移图        │ 用户记忆      │ 地理邻域      │ 语义轨迹 │
                    └──────┬───────┴──────┬───────┴──────┬───────┴────┬────┘
                           └──────────────┴──────────────┴────────────┘
                                            ▼
                                   ② 两阶段融合 → 候选集 C（25 个）
                                            ▼
                          ┌───────────────── ③ LLM 排序 ─────────────────┐
                          │ 输入：轨迹 + 时段 + 画像 + C（含工具事实）      │
                          │ 输出：ranking / confidence / reason          │
                          └───────────────────────┬──────────────────────┘
                                                  ▼
                                   ④ 自适应校验（条件触发，1 次额外调用）
                                                  ▼
                                   ⑤ 先验锚定混合 → top-10
                                                  ▼
                                              输出 L̂
```

LLM 出现在 ③ 与 ④ 两处，**每样本至少调用一次**；
其余阶段均为确定性统计计算。

---

## 3.3 阶段 ①：四路候选生成

四条通道各自独立产出候选列表，互不共享分数（量纲不同）。

### 3.3.1 转移图通道（Semantic-free，核心）

对训练集统计三种条件分布，混合为转移分数：

$$
s_{\text{tr}}(v) \;=\; 1.0\cdot \hat{P}\!\left(v\mid l_{n},\,b(t_{n+1})\right)
\;+\;0.3\cdot \hat{P}\!\left(v\mid l_{n}\right)
\;+\;1.5\cdot \hat{P}\!\left(v\mid l_{n-1},l_{n}\right)
$$

其中 $b(t)=\lfloor h(t)/3\rfloor$ 为三小时时段桶，$\hat{P}$ 为最大似然计数估计。
三项分别刻画**时段条件下的转移**、**无条件下阶转移**、**二阶上下文转移**。

> 实现：`retrieval.py::GraphRoute`
> 该通道单独使用即达 NYC HR@10 = 0.775，是全部通道中最强的一路。

### 3.3.2 用户记忆通道

$$
s_{\text{mem}}(v) \;=\; \log\!\big(1+n_u(v)\big)\cdot\big(0.25+\hat{P}(c(v)\mid h(t))\big)\cdot\rho_{\text{rec}}(v)
$$

$n_u(v)$ 为用户 $u$ 在训练期的访问次数，$\hat{P}(c\mid h)$ 为类别-时段先验，
$\rho_{\text{rec}}(v)=0.6$ 当 $v$ 已出现在最近序列中（避免重复推荐刚去过的地方），否则为 1。
刻画"用户的长期偏好随时间的变化"。

> 实现：`retrieval.py::MemoryRoute`

### 3.3.3 地理邻域通道

$$
s_{\text{geo}}(v) \;=\; \exp\!\left(-\frac{d(l_n,v)}{\lambda}\right)\cdot\big(1+0.3\cos\theta\big)\cdot\rho_{\text{rec}}(v)
$$

$d$ 为大圆距离（$\lambda=2.5$ km），$\theta$ 为"上一位置→当前位置"向量与"当前位置→候选"向量的夹角。
方向项用于抑制反向候选；此地 $\rho_{\text{rec}}(v)=0.5$（与记忆通道的 0.6 不同），即最近去过的候选权重减半。

> 实现：`retrieval.py::GeoRoute`

### 3.3.4 语义轨迹通道

以 TF-IDF 编码轨迹文本，检索训练集中最相似的 $k=50$ 条历史轨迹，
把它们的"下一站"作为候选，权重为余弦相似度累加。

> 实现：`retrieval.py::SemanticRoute`

---

## 3.4 阶段 ②：两阶段融合

四路通道强度差异极大（见下表），因此**不做等权融合**。

| 通道 | 单独 HR@10（NYC） |
|---|---|
| 转移图 | 0.775 |
| 用户记忆 | 0.437 |
| 语义轨迹 | 0.428 |
| 地理邻域 | 0.100 |

融合规则：

$$
\text{head} = \operatorname{TopK}_{k_g}\big(s_{\text{tr}}\big),\qquad
a(v)=\sum_{r\in\{\text{mem,geo,sem}\}} w_r\cdot \mathrm{minmax}\big(s_r(v)\big)
$$

$$
C = \text{head} \;\Vert\; \operatorname{TopK}\big(a(v)\;:\;v\notin \text{head}\big)
$$

即**强通道决定头部，弱通道只负责补位**。取 $k_g=8$，候选集规模 $|C|=25$。

> 实现：`fusion.py::two_stage`
> 设计依据：等权 RRF 会使 HR@1 从 0.538 掉到 0.302（见 §6.2 消融）。

---

## 3.5 阶段 ③：LLM 排序

### 3.5.1 输入构造

提示词由四部分组成：

1. **轨迹上下文**：最近 6 次签到，格式 `POIID(类别) -> ...`；
2. **时间上下文**：目标时刻的小时数；
3. **用户画像**：训练期签到数与前两大类别占比；无历史时标注 cold-start；
4. **候选清单**：每个候选一行，含工具预先算好的事实

```
[C_i] id=243 | Food | 1.2km | ~18min | hour-fit=0.31 | 证据：7 次转移自 POI 784；该用户访问过 2 次
```

其中 `km`、`min`、`hour-fit` 分别由大圆距离、行程时间模型、类别-时段先验在**本地计算**，
LLM 不做任何数值运算。

### 3.5.2 输出约束

要求模型返回 JSON：

```json
{"ranking": [10 个整数 id], "confidence": 0.0~1.0, "reason": "一句话"}
```

**关键约束**：`ranking` 中的 id 必须来自候选集 $C$，否则该 id 被丢弃并计入 `invalid`。
因此 LLM **不能凭空生成 POI**，只能在检索给出的候选内排序——这消除了生成式推荐的幻觉问题。

> 实现：`agent.py::ranking_prompt`、`AdaptiveAgent.run`

---

## 3.6 阶段 ④：自适应校验

### 3.6.1 触发条件

$$
\text{verify} = \mathbb{1}\big[\,\text{conf}<\tau_c \;\lor\; \text{invalid}>0 \;\lor\; |r|<10\,\big]
$$

其中 $\tau_c=0.55$。仅当条件成立时发起第二次调用。

### 3.6.2 校验提示词

把**工具核验结果**回灌给模型：

```
1. id=243 | Food | 1.2km | ~18min | hour-fit=0.31 | check=ok
2. id=899 | Residence | 13.5km | ~48min | hour-fit=0.11 | check=implausible_jump:13.5km
```

并要求按清单修正：id 必须在候选内、不得重复、恰好 10 个、
`implausible_jump` 与 `cat_hour_mismatch` 的候选应被替换。

> 实现：`agent.py::critique_prompt`、`tools.py::violations`

实测触发率 13.5%（NYC/test），平均调用 **1.135 次/样本**。

---

## 3.7 阶段 ⑤：先验锚定混合

记 $r_f$ 为候选在融合列表中的名次，$r_\ell$ 为 LLM 排序中的名次，
$n=\max(|C|,|r_\ell|)$，定义秩分数

$$
\sigma_f(v)=\frac{n-r_f(v)}{n},\qquad \sigma_\ell(v)=\frac{n-r_\ell(v)}{n}
$$

最终分数与列表：

$$
s(v)=\alpha\,\sigma_f(v)+(1-\alpha)\,\sigma_\ell(v),\qquad
\hat{L}=\operatorname{TopK}_{10}\big(s(v)\big)
$$

若 $|\hat{L}|<10$，用融合顺序补齐（不使用任何模型未提及的 POI）。

$\alpha$ 是"对统计先验的信任度"：$\alpha=1$ 即完全不用 LLM，$\alpha=0$ 即完全由 LLM 排序。

> 实现：`agent.py::blend_orders`

---

## 3.8 算法

```
算法 1  检索先验 + LLM 仲裁的免训练下一 POI 推荐
────────────────────────────────────────────────────────────
输入: 轨迹 S_u, 目标时刻 t, 候选预算 |C|, 头部预算 k_g, 先验权重 α, 校验阈值 τ_c
输出: 排序列表 L̂, 理由 e

 1  # ---- 离线统计（每城一次） ----
 2  T1h, T1, T2 ← 从 D_train 统计时段条件/一阶/二阶转移
 3  H_u ← 用户历史分布;  P(c|h) ← 类别-时段先验;  TF-IDF 索引 ← 轨迹文本
 4
 5  # ---- 阶段 ① ----
 6  R_tr ← TopK(s_tr(·), 20)
 7  R_mem ← TopK(s_mem(·), 15)
 8  R_geo ← TopK(s_geo(·), 20)
 9  R_sem ← semantic_retrieve(S_u, 50)
10
11  # ---- 阶段 ② ----
12  head ← TopK_{k_g}(R_tr)
13  a(v) ← Σ_r w_r · minmax(R_r[v])
14  C ← head ‖ TopK_{|C|−k_g}(a(v), v ∉ head)
15
16  # ---- 阶段 ③ ----
17  x ← build_prompt(S_u, t, H_u, C)
18  y ← LLM(x)                       # {ranking r, confidence conf, reason e}
19  r, conf ← validate(y, C)          # 丢弃 C 之外的 id
20
21  # ---- 阶段 ④ ----
22  if conf < τ_c or invalid(r) or |r| < 10:
23      x' ← build_critique(S_u, t, r, facts(C))
24      r, conf, e ← LLM(x')          # 工具事实驱动的修正
25
26  # ---- 阶段 ⑤ ----
27  s(v) ← α·σ_f(v) + (1−α)·σ_ℓ(v)
28  L̂ ← TopK_{10}(s)
29  return L̂, e
```

---

## 3.9 复杂度与成本

| 项 | 值 |
|---|---|
| 候选生成 | 2.3 ms/样本（CPU，15,754 POI） |
| LLM 调用 | 1.135 次/样本（13.5% 触发校验） |
| 端到端延迟 | 约 5 s/样本（受 API 延迟支配） |
| 训练 | 无 |
| 参数 | 4 个通道权重 + $k_g$ + $\alpha$ + $\tau_c$，均在 valid 上选定 |

---

## 3.10 参数配置与结果

$\alpha$ 刻画 LLM 在最终排序中的话语权。三种配置在 NYC/test 上的结果：

| 配置 | LLM 角色 | HR@1 | HR@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|
| $\alpha=1.0$ | 不参与排序 | 0.542 | 0.752 | 0.808 | 0.672 |
| $\alpha=0.75$ | 调整顺序 | 0.500 | 0.752 | **0.808** | 0.653 |
| $\alpha=0.0$ | 完全决定顺序 | 0.385 | 0.680 | **0.808** | 0.580 |

三者 HR@10 一致（0.808），差异集中在头部精度（HR@1）。

**如何读这张表**：在当前基准上，统计先验在"排第 1"上更准，
而 LLM 并不提供额外信息，因此 $\alpha$ 越小，头部越差。
这不是否定 LLM 的作用，而是说明**在"下一步由最近转移决定"的数据上，
重排不是 LLM 的最优角色**——这一结论反过来界定了 LLM 应该承担的任务。

---

## 3.11 设计说明：为什么 LLM 不能生成候选

本方法显式禁止 LLM 输出候选集之外的 POI。原因有三：

1. **幻觉**：生成式模型会编造不存在的 ID，评测中直接记为错误；
2. **可归因**：候选由哪条通道提供是明确的，便于做通道消融；
3. **可替换**：LLM 只依赖候选与事实，更换模型不会改变检索行为。

---

## 3.12 已知边界

1. **自评置信度不可用于仲裁**：93.7% 的样本自报 $\text{conf}\ge 0.7$，
   按 $\text{conf}\ge 0.8$ 门控时 HR@1 掉到 0.492。校验门控因此只能用于"省调用"，
   不能用于"决定信谁"。
2. **重排角色收益有限**：在转移主导的数据上，LLM 排序不优于统计先验。
3. **长历史基准上表现不同**：在 GNPR/Think2Go 划分上，本方法 Acc@1 = 0.214，
   低于训练型 Think2Go 的 0.385，说明统计先验在长历史、跨月场景下失效更明显。

---

## 3.13 实现对照表

| 阶段 | 文件 | 关键函数 |
|---|---|---|
| 离线统计 | `data.py` | `load_city` |
| ① 四路检索 | `retrieval.py` | `GraphRoute` / `MemoryRoute` / `GeoRoute` / `SemanticRoute` |
| ② 两阶段融合 | `fusion.py` | `two_stage` |
| ③ LLM 排序 | `agent.py` | `ranking_prompt` / `AdaptiveAgent.run` |
| ④ 自适应校验 | `agent.py` / `tools.py` | `critique_prompt` / `violations` |
| ⑤ 先验混合 | `agent.py` | `blend_orders` |
| LLM 客户端 | `llm.py` | `LLMClient.chat_json` |
| 评测 | `metrics.py` | `evaluate` |
