# 方法：多视图证据与提示级测试时增强

> 本文档对应代码 `src/grpoi/`，所有模块、参数、成本均与实现一致。
> 图中每个方框标注了数据形态，可直接据此绘制架构图。

---

## 3.1 问题定义

用户 $u$ 的轨迹 $S_u=\{(l_1,t_1),\dots,(l_n,t_n)\}$，其中 $l_i\in\mathcal{P}$ 为 POI、$t_i$ 为时间戳。
每个 POI 具备静态属性：坐标 $g(l)=(\text{lat},\text{lon})$、类别 $c(l)\in\mathcal{C}$（$|\mathcal{C}|=10$）。

给定 $S_u$ 与目标时刻 $t_{n+1}$，输出按概率降序的 10 元列表 $\hat L=[\hat l_1,\dots,\hat l_{10}]$。

系统**不更新任何参数**：训练集 $\mathcal{D}_{\text{train}}$ 只用于离线统计，不参与梯度更新。

---

## 3.2 框架总览

```
┌──────────────────────────── 离线统计层（每城一次，无 LLM）────────────────────────────┐
│                                                                                      │
│   D_train ──┬──▶ 转移计数表 T1h / T1 / T2        （POI × 时段 → POI 频次）            │
│             ├──▶ 用户画像 H_u                    （user → {POI: 次数}）               │
│             ├──▶ POI 属性表                      （id → 坐标 / 类别）                 │
│             └──▶ TF-IDF 索引                     （轨迹文本 → 稀疏向量）              │
└───────────────────────────────────┬──────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────── 在线推理层 ─────────────────────────┐
│                                                                                      │
│  S_u, t_{n+1}                                                                        │
│      │                                                                               │
│      ▼                                                                               │
│  ┌──────────────────────┐   ①证据构造（确定性，<3ms）                                 │
│  │ 转移视图  T_view     │   "从 l_n 出发人群常去哪"     附转移次数                     │
│  │ 个体视图  U_view     │   "该用户常去哪"             附访问次数                     │
│  │ 空间视图  G_view     │   "附近有什么"               附距离(km)                     │
│  │ 语义视图  S_view     │   "相似轨迹的下一站"         附距离(km)                     │
│  └──────────┬───────────┘                                                           │
│             │  各取 top-8，序列化为 4 行事实文本  E                                    │
│             ▼                                                                        │
│  ┌──────────────────────┐   ②提示构造（3 个变体，同一份 E，不同组织）                  │
│  │  A: 默认顺序          │                                                           │
│  │  B: 证据倒序          │                                                           │
│  │  C: 居民视角+显式思考 │                                                           │
│  └──────────┬───────────┘                                                           │
│             ▼                                                                        │
│  ┌──────────────────────┐   ③生成（LLM，温度 0，JSON 模式）                           │
│  │  LLM × 3              │   每变体输出 10 个 POI ID + 一句理由                        │
│  └──────────┬───────────┘                                                           │
│             ▼                                                                        │
│  ┌──────────────────────┐   ④聚合（Borda 计数，无 LLM）                               │
│  │  score(v)=Σ(10-rank) │   → 融合后的 top-10                                          │
│  └──────────┬───────────┘                                                           │
│             ▼                                                                        │
│  ┌──────────────────────┐   ⑤合法性修复（确定性）                                     │
│  │  去越界/重复 → 补位   │   补位池 = 证据的统计排序                                   │
│  └──────────┬───────────┘                                                           │
│             ▼                                                                        │
│           L̂（10 个 POI）+ reason                                                     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**LLM 只出现在 ③**。①②④⑤ 全部是确定性计算，可完全复现。

---

## 3.3 多视图证据构造

原始轨迹是一串 POI ID，转移规律、个体习惯、空间约束都以隐式形式埋在序列里。
我们把这四类规律**显式化为四个视图**，每个视图输出"POI + 可核对数值"的短列表。

记最后位置为 $l_n$、最后时刻小时为 $h(t_{n+1})$、时段桶 $b=\lfloor h/3\rfloor$。

### 3.3.1 转移视图（Transition View）

刻画"从当前 POI 出发，人群通常去哪"。混合三种条件频率：

$$
s_{\text{tr}}(v)=1.0\cdot\hat P\big(v\mid l_n,b\big)+0.3\cdot\hat P\big(v\mid l_n\big)+1.5\cdot\hat P\big(v\mid l_{n-1},l_n\big)
$$

- 第一项：时段条件一阶转移（同一位置、不同时段，去向不同）；
- 第二项：无条件下阶转移（平滑作用）；
- 第三项：二阶转移（利用上一步的方向性）。

$\hat P$ 为最大似然计数估计（加性平滑）。取 top-8 输出。

> 实现：`retrieval.py::GraphRoute`

### 3.3.2 个体视图（Individual View）

刻画"这个用户自己常去哪"，用于捕获长期偏好与回访倾向：

$$
s_{\text{ind}}(v)=\log\big(1+n_u(v)\big)\cdot\big(0.25+\hat P(c(v)\mid h)\big)\cdot\rho_{\text{rec}}(v)
$$

$n_u(v)$ 为用户训练期访问次数，$\hat P(c\mid h)$ 为类别—时段先验（"这个时间点通常会去哪类地方"），
$\rho_{\text{rec}}=0.6$ 表示若该 POI 出现在最近序列中则降权。取 top-8。

> 实现：`retrieval.py::MemoryRoute`

### 3.3.3 空间视图（Spatial View）

刻画"附近有什么"，以最后位置为锚：

$$
s_{\text{geo}}(v)=\exp\!\left(-\frac{d(l_n,v)}{\lambda}\right)\cdot\big(1+0.3\cos\theta\big)\cdot\rho_{\text{rec}}(v)
$$

$d$ 为大圆距离（$\lambda=2.5$ km，仅在 5 km 内检索），$\theta$ 为"上一步→当前"向量与"当前→候选"向量的夹角，
用于抑制与行进方向相反的候选。取 top-8，**输出时附距离**。

> 实现：`retrieval.py::GeoRoute`

### 3.3.4 语义视图（Semantic View）

刻画"轨迹层面相似的用户去了哪"。把每条训练轨迹表示为 ID 串，用 TF-IDF 向量化，
检索余弦相似度最高的 50 条轨迹，取其**下一站**作为候选，权重为相似度累加。取 top-8，附距离。

> 实现：`retrieval.py::SemanticRoute`

### 3.3.5 证据序列化

四视图各自渲染为一行**事实陈述**（而非待选项），拼成证据块 $E$：

```
- After POI 784, people most often went to: 243 (Food, 1.2km); 899 (Shop, 0.3km); 1274 (Drink, 0.07km); ...
- This user has visited most often: 954 (Food); 784 (Outdoors); 382 (Arts & Entertainment); ...
- Near the current location: 237 (Travel & Transport, 0.11km); 899 (Shop & Service, 0.13km); ...
- Users with similar recent trajectories next visited: 39 (Arts & Entertainment, 2.1km); 382 (Arts, 0.32km); ...
```

三个设计要点：

1. **附数值**（次数、距离），让模型不必自己估计；
2. **不附分数**（不暴露各路通道的权重），避免模型被内部标定影响；
3. **不写成菜单**（不出现 `[C1]`、`id=` 这类"从中选择"的暗示），保持生成自由。

> 实现：`generative.py::evidence_block`

---

## 3.4 生成

### 3.4.1 提示结构

提示由五部分组成，按固定顺序拼接：

| 部分 | 内容 | 目的 |
|---|---|---|
| ① 任务陈述 | "Predict their next point of interest" | 固定任务语义 |
| ② 轨迹上下文 | 最近 8 次签到，格式 `POIID(类别) -> ...` | 短期行为 |
| ③ 时间与画像 | 目标小时 + 训练期签到数与前两大类别占比；无历史则标 cold-start | 长期偏好 |
| ④ 证据块 $E$ | 3.3.5 的四行事实 | 统计规律 |
| ⑤ 指令 | 见下 | 输出约束 |

指令部分只有三条实质约束：

```
1. Recommend exactly 10 distinct POI ids, most likely first.
2. Use POI ids that exist in this city. You are not limited to the ids listed above.
3. Do not include line breaks inside the JSON.
```

**第 2 条是方法的关键**：明确允许模型输出证据之外的 POI，因此它必须调用自身的城市知识，
而不是在给定集合内排序。实测 NYC/SIN 上 **54%** 的输出 POI 不在证据中。

> 实现：`generative.py::generation_prompt`

### 3.4.2 输出解析

模型返回：

```json
{"recommendation": [10 个整数 ID], "reason": "一句话"}
```

解析时只做保守处理：非整数跳过、列表内去重、保留原始顺序。
`reason` 仅记录，不参与打分。

---

## 3.5 合法性修复（只修不约束）

自由生成会产生三类无效输出：越界 ID、重复 ID、数量不足。处理规则：

$$
\hat L=\big[\text{valid}(r)\big]\;\Vert\;\big[\text{fill from evidence pool}\big],\qquad|\hat L|=10
$$

1. 丢弃不在 $[0,|\mathcal{P}|)$ 的 ID 与重复 ID（计入 `invalid`）；
2. 不足 10 个时，按证据的融合顺序补足；
3. 模型已给出的有效 ID **一律不改动、不重排**。

补位池由四视图的融合排序提供（`fusion.py::two_stage`：强视图前 8、其余视图补位）。

实测非法率：PHO 0.0%、NYC 0.7%、SIN 0.2%——修复几乎不影响结果，但缺了它，含非法 ID 的样本会整体失效。

> 实现：`generative.py::GenerativeAgent.repair`

---

## 3.6 提示级测试时增强（TTA）

### 3.6.1 三个提示变体

显式规划不可用（模型自述意图与输出的一致率达 71%/97%，见 §4），因此改为在**输入侧**制造多样性：
保持证据内容 $E$ 完全不变，只改变它的组织方式：

| 变体 | 变换 | 作用 |
|---|---|---|
| **A** | 默认顺序（转移 → 个体 → 空间 → 语义） | 群体规律优先 |
| **B** | 证据倒序（个体 → 空间 → 语义 → 转移） | 个人与空间优先 |
| **C** | 追加居民视角 + "先考虑时段与作息再决定" | 激活常识推理 |

三个变体各生成一次（温度 0），得到三个列表 $L_A,L_B,L_C$。

### 3.6.2 Borda 聚合

$$
\text{score}(v)=\sum_{k\in\{A,B,C\}}\big(10-\text{rank}_k(v)\big)\cdot\mathbb 1\big[v\in\text{top-}10_k\big],
\qquad \hat L=\operatorname{TopK}_{10}(\text{score})
$$

即：某 POI 在某变体中排第 1 得 10 分、排第 10 得 1 分，未出现得 0 分，三变体求和。

选择 Borda 的理由：

1. 不需要概率（模型不输出可靠概率）；
2. 对单个异常列表不敏感（一个变体出错不会颠覆结果）；
3. 奖励**共识**而非单次高名次，与"多视角互补"的设计意图一致。

代价：3 次调用/样本。实测相对单次生成 +2.7pt HR@10（显著），
相对同提示词重复采样方向更优（+1.8pt，未达显著），见 §5.5。

> 实现：`run_prompt_ensemble.py`

---

## 3.7 算法

```
算法 1  多视图证据 + 提示级 TTA 的零样本下一 POI 推荐
────────────────────────────────────────────────────────────────
输入: 轨迹 S_u, 目标时刻 t, 变体集合 V={A,B,C}
输出: 推荐列表 L̂, 理由 e

 1  # ---------- 离线（每城一次）----------
 2  T1h,T1,T2 ← 转移计数(D_train)
 3  H_u ← 用户访问分布(D_train);  P(c|h) ← 类别-时段先验
 4  IDF 索引 ← TF-IDF(D_train 轨迹文本)
 5
 6  # ---------- ① 证据构造 ----------
 7  V_tr  ← TopK(s_tr, 8)      # 转移视图
 8  V_ind ← TopK(s_ind, 8)     # 个体视图
 9  V_geo ← TopK(s_geo, 8)     # 空间视图（附距离）
10  V_sem ← TopK(s_sem, 8)     # 语义视图（附距离）
11  E ← serialize(V_tr, V_ind, V_geo, V_sem)
12
13  # ---------- ② 提示变体 ----------
14  for k ∈ V:
15      P_k ← build_prompt(S_u, t, E, variant=k)
16
17  # ---------- ③ 生成 ----------
18  for k ∈ V:
19      L_k, e_k ← LLM(P_k)                  # 温度 0，JSON
20      L_k ← parse(L_k)                     # 去重、丢弃非整数
21
22  # ---------- ④ 聚合 ----------
23  score ← Borda(L_A, L_B, L_C)             # Σ(10-rank)
24  L̂ ← TopK_10(score)
25
26  # ---------- ⑤ 修复 ----------
27  L̂ ← drop_out_of_range(L̂);  L̂ ← dedup(L̂)
28  while |L̂| < 10: L̂.append(next evidence POI not in L̂)
29  return L̂, e
```

---

## 3.8 复杂度与成本

| 阶段 | 复杂度 / 成本 | 说明 |
|---|---|---|
| 离线统计 | O(\|D_train\|) 一次 | NYC 约 10 万条交互 |
| 证据构造 | **2.3 ms/样本** | CPU，15,754 POI |
| LLM 调用 | **3 次/样本** | 单次版本为 1 次 |
| 输入 / 输出 | 约 650 / 68 tokens | 每变体 |
| 端到端延迟 | 约 5–7 s/样本 | 受 API 延迟支配 |
| 单价 | 约 $0.0004/样本（3 次） | gpt-4o-mini |
| 可训练参数 | **0** | 全部为计数统计 |

---

## 3.9 设计要点（每一项都有对应的消融）

| 设计 | 为什么这么做 | 对应实验 |
|---|---|---|
| 四视图而非单通道 | 轨迹把三类规律压成 ID 串，模型需自行统计 | §5.3 无证据 0.205 → 四视图 0.692 |
| 事实陈述而非候选菜单 | 菜单会把生成退化为排序，失去开放输出 | 与约束式对比 §5.5 |
| 允许输出证据外 POI | 保留生成能力，54% 输出为模型自生成 | §5.6 |
| 只修不约束 | 保证可评测，又不改变"谁生成" | 非法率 0.0–0.7% |
| 三变体 TTA | 显式规划不可得，用视角制造隐式多样性 | §5.5 +2.7pt 显著 |
| Borda 聚合 | 无需概率、抗单点异常、奖励共识 | 待补聚合策略对比 |

---

## 3.10 与实现文件的对应

| 阶段 | 文件 | 关键函数 |
|---|---|---|
| 离线统计 | `data.py` | `load_city` |
| 转移视图 | `retrieval.py` | `GraphRoute.retrieve` |
| 个体视图 | `retrieval.py` | `MemoryRoute.retrieve` |
| 空间视图 | `retrieval.py` | `GeoRoute.retrieve` |
| 语义视图 | `retrieval.py` | `SemanticRoute.retrieve` |
| 证据序列化 | `generative.py` | `evidence_block` |
| 提示构造 | `generative.py` / `run_prompt_ensemble.py` | `generation_prompt` / `build_prompt` |
| 生成与解析 | `generative.py` / `llm.py` | `GenerativeAgent.run` / `chat_json` |
| 聚合 | `run_prompt_ensemble.py` | `borda` |
| 修复 | `generative.py` | `GenerativeAgent.repair` |
| 指标 | `metrics.py` | `evaluate` |
