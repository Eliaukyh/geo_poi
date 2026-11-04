# geo-rag-poi

**面向下一 POI 推荐的免训练 Agentic 地理检索增强框架**
*Training-Free Agentic Geo-RAG for Next-POI Recommendation*

[English](README.md) | 中文

---

## 1. Motivation：为什么还要做这件事

### 1.1 问题本身的三个约束

下一 POI 推荐（next-POI recommendation）要回答的是"这个用户接下来会去哪"。在真实部署里，它同时受三个条件挤压：

1. **标注稀缺**。序列模型（LSTM/GRU/Transformer 类）依赖大量用户轨迹训练，而新城市、新用户、新时间段几乎没有可用标注。冷启动场景下，训练范式本身就不成立。
2. **空间约束是硬约束**。一个"语义上很像"但 40 公里外的 POI，在推荐列表里是无效的。语言模型恰恰最不擅长这种连续数值推理。
3. **成本必须可控**。逐用户逐请求调用大模型，调用次数直接等于账单。任何"每次都跑两三轮反思"的方案在工程上很难落地。

这三点共同指向**免训练（training-free）**路线：不更新任何参数，靠检索到的证据 + LLM 的零样本推理完成推荐。这也正是 LLM4POI 之后一批工作的共同前提。

### 1.2 现有免训练方案的两个断点

以 RALLM-POI 为典型代表，当前免训练方案在**检索**和**排序**之间各有一个断点：

**断点一：检索太窄。**
RALLM-POI 只有一条 TF-IDF 轨迹相似度通道。它回答的是"哪条历史轨迹和当前轨迹文本相似"，但回答不了：
- 人流的**转移规律**（大家从当前这个 POI 出发通常去哪）；
- **空间邻接**（步行 10 分钟内有哪些选择）；
- **用户自身习惯**（这个用户每周三晚上固定去哪类地方）。

单通道必然漏召回。而漏召回是排序阶段无法弥补的——候选集里没有正确答案，再强的重排也没用。

**断点二：排序太自由。**
RALLM-POI 让 LLM 从记忆里**直接生成** 10 个 POI ID。这意味着：
- 模型可能编造不存在的 ID；
- 推荐结果无法与检索证据对齐；
- 地理可行性完全交给模型"心算"。

我们把这一点做成了可测量的负结果：**朴素地让 LLM 对融合后的候选列表做 listwise 重排，指标会显著下降**（PHO/valid：Recall@1 0.289 → 0.132，NDCG@10 0.567 → 0.422）。原因是弱模型的重排先验与统计检索信号冲突，且它没有任何机制知道自己错了。这说明"让 LLM 自由重排"不是一个安全的默认设计——必须有锚。

### 1.3 因此本文的设计命题

> 把**检索**做宽，把**地理**交给工具，把 **LLM** 限制为受约束的精排，把**调用次数**做成按需触发。

四个动作分别对应四个贡献点，下面逐条说明，并标注当前代码中的实现位置与实测证据。

---

## 2. Contribution

### C1｜多路混合检索：用四条互补通道替代单通道 TF-IDF

不再依赖单一路径，而是并行构建四个候选来源，再用倒数排名融合（RRF）合并：

| 通道 | 回答的问题 | 实现 |
|---|---|---|
| **语义**（semantic） | 哪条历史轨迹和当前轨迹最像？ | 训练轨迹 TF-IDF + 余弦相似度，取相似轨迹的"下一站"投票 |
| **转移图**（graph） | 从当前这几个 POI 出发，人流通常去哪？ | 训练集连续签到构建有向转移图，按最近 POI 距离衰减加权 |
| **地理邻域**（geo） | 步行可达范围内有哪些选择？ | 以最后位置为锚，距离指数衰减 + 行进方向一致性 |
| **个体记忆**（memory） | 这个用户自己常去哪？ | 用户历史访问频次 × P(类别 \| 时段) |

融合采用 RRF 而非加权分数相加，避免四路分数量纲不一致。候选池取 top-25 交给下游。

> 代码：`src/grpoi/retrieval.py`
> 证据：候选召回率（target 是否落在候选池内）在 PHO 达 0.95、NYC 0.77、SIN 0.74–0.76；纯检索（零 LLM 调用）在 PHO/test 上 HR@10 已达 0.744，**单独就超过 RALLM-POI 的 0.590**。

### C2｜工具约束：地理量由工具算，模型只做取舍

LLM 不做算术，只做判断。距离、行程时间、类别—时段合理性全部由 `tools.py` 计算后以事实形式注入 prompt。

由于所用 Foursquare 三个数据集**不含营业时间字段**，原计划中的"营业状态"约束无法实现，改为从训练签到统计的 **P(category | hour)** 先验：它同样能捕捉"凌晨 3 点推荐商场"这类不合理建议，且完全可量化、可复现。

> 代码：`src/grpoi/tools.py`（`haversine`、`travel_time_min`、`category_hour_fit`、`violations`）
> 说明：行程时间为 `1.3 × 大圆距离 / 速度`（≤1.5 km 按步行 4.5 km/h，以上按 22 km/h），是显式声明的近似，接口留了替换位，可接 OSRM/高德。

### C3｜先验锚定排序 + 自适应校验：给 LLM 装一个"刹车"

针对 1.2 中的负结果，最终排序不是"LLM 说了算"，而是：

```
最终分数 = α · 检索先验 + (1 − α) · LLM 排序      （默认 α = 0.75）
```

LLM 只能在统计先验附近做微调，无法推翻检索结果。α 在 valid 上调、在 test 上报告；实测 α=1.0（纯检索）时 HR@1 0.316，α=0.75 时 0.342，说明模型确实贡献了正向信号，但必须限幅。

在此基础上再叠加**按需校验**：只有当 LLM 自报置信度低于阈值（0.55）或输出未通过校验（ID 非法、数量不足、重复）时，才发起第二次"工具核验"调用。

> 代码：`src/grpoi/agent.py`（`blend_orders`、`AdaptiveAgent.run`）
> 证据：校验触发率约 33%，平均调用从 RALLM-POI 的 2.00 次/样本降至 1.32–1.39 次/样本，指标无损。

### C4｜可分离的评测协议：把"召回问题"和"排序问题"拆开报

现有工作报告 Recall/NDCG/MAP，但这些指标把两件事混在一起：候选没检索到，和检索到了但排错了。本文额外报告 **candidate recall（候选池召回）** 作为独立诊断量，并保证指标实现与 RALLM-POI **逐位一致**（用其原始 `predictions_*.txt` 复算，对齐其 `metrics_*.csv`）。

这直接改变了优化方向：实验显示 NYC/SIN 的瓶颈在召回（0.770 / 0.736–0.763）而非排序，因此下一步该扩候选池而不是调提示词。

> 代码：`src/grpoi/metrics.py`、`compare_rallm.py`、`bootstrap.py`

### 实验层面的贡献

在同一份数据、同一划分、同一指标函数、同一模型（gpt-4o-mini）下，相对 RALLM-POI 的提升：

受控复现（同一套指标代码、同一子集）：

| 设置 | HR@10 提升（相对） | NDCG@10 |
|---|---|---|
| PHO/valid（全量 38） | +36% | 0.362 → 0.599 |
| PHO/test（全量 39） | +43% | 0.340 → 0.467 |
| NYC/valid（前 80 条） | +60% | 0.248 → 0.428 |
| SIN/valid（前 80 条） | +61% | 0.236 → 0.436 |

对 RALLM-POI 论文报告值（全量、80/20 划分，非受控）：PHO H@10 +23.6%、NYC +55.8%、SIN +42.8%。

---

## 3. 与已有工作的差异

### 3.1 与 RALLM-POI（直接对标基线）

| 维度 | RALLM-POI | 本工作 |
|---|---|---|
| 检索通道 | 1 条（TF-IDF 轨迹） | 4 条（语义 / 转移图 / 地理 / 个体记忆）+ RRF |
| 候选约束 | 无，模型自由生成 ID | 强制来自候选池，非法 ID 直接拒绝 |
| 地理计算 | 交给 LLM | 工具计算后作为事实注入 |
| 重排安全 | 无机制 | 先验锚定混合（α），LLM 不可推翻检索 |
| 调用次数 | 固定 2 次 | 1.32–1.39 次（置信度门控） |
| 评测 | 仅 Recall/NDCG/MAP | 增补候选召回、空间违规率、调用/Token/成本 |

### 3.2 与其它零样本 / RAG 路线的定位

下表按各工作公开摘要所体现的**能力定位**做区分，具体实现细节请以原文为准：

| 工作 | 训练需求 | 核心侧重 | 与本工作的关系 |
|---|---|---|---|
| **LLM4POI**（SIGIR'24） | 需训练 | 任务定义、强基线 | 提供问题设定与对比参照；本工作不复现其训练部分 |
| **AgentMove**（NAACL'25） | 免训练 | 零样本移动模式建模 | 同为免训练路线；本工作在其之外强调多路检索与工具约束 |
| **ZeroPOIRec**（DMKD'25） | 免训练 | 零样本用户画像与候选排序 | 同样关注候选排序；本工作把候选来源扩展到转移图与地理邻域 |
| **TP-RAG**（EMNLP'25） | 免训练 | 时空 RAG + 行程规划 | 同为时空 RAG；本工作面向单点下一跳预测而非多日行程 |
| **Think2Go**（KDD'26） | 需 RL | 按不确定性决定推理深度 | 借用"自适应计算"的思想，但用置信度门控实现，不走 RL |
| **RALLM-POI**（PRICAI'25） | 免训练 | TF-IDF + DTW + Agent 自省 | 直接对标基线，本工作逐项替换其薄弱环节 |

**一句话定位**：不是"另一个 LLM 推荐器"，而是**把免训练推荐里的检索层做厚、把 LLM 的自由度收紧**，并用可分离的评测证明增益来自哪里。

---

## 4. 方法框架

```text
样本（6 天 × N 次签到）
        │
        ▼
┌─────────────────── 多路检索（C1）───────────────────┐
│  semantic : TF-IDF 轨迹相似度 → 相似轨迹的下一站投票   │
│  graph    : POI 转移图 → 从最近 3 个 POI 出发的去向    │
│  geo      : 距离衰减 + 行进方向一致性 → 邻域候选       │
│  memory   : 用户历史频次 × P(category|hour)          │
└───────────────────────┬──────────────────────────────┘
                        ▼
                 RRF 融合 → top-25 候选
                        │
                        ▼
        ┌───────── 工具层（C2）计算事实 ─────────┐
        │ 距离 / 行程时间 / P(类别|时段) / 违规标记 │
        └───────────────────┬───────────────────┘
                            ▼
              LLM 受约束精排（1 次调用）+ 自报置信度
                            │
              置信度 < 0.55 或输出非法？
                    │yes              │no
                    ▼                 │
        工具核验二次校验（1 次调用）    │
                    └────────┬────────┘
                             ▼
        最终分 = α·检索先验 + (1−α)·LLM 排序（C3，α=0.75）
                             ▼
                         top-10
```

---

## 5. 环境准备

依赖 uv（已装在 `../.tools/bin/uv`）。本机沙箱下 `~/.cache/uv` 不可写，因此缓存与解释器都放在工作区：

```bash
cd geo-rag-poi
source ../.tools/uv-env.sh    # 设置 UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR / MPLCONFIGDIR
uv sync                       # 按 uv.lock 复原（Python 3.11）
```

API 凭据放在 `../.tools/.env`（权限 600，不要提交）：

```bash
DMX_API_KEY=sk-...
DMX_BASE_URL=https://www.dmxapi.com/v1
```

`config.py` 启动时自动读取，也可用环境变量 `GRPOI_MODEL` 切换模型。

---

## 6. 运行

所有命令在 `geo-rag-poi/` 下执行（源码使用相对路径）。

```bash
# 仅检索，零 API 成本（用于快速自检，也是召回上限诊断）
uv run python run_eval.py --city PHO --n 30 --mode retrieval_only

# 完整方法
uv run python run_eval.py --city PHO --split test --n 39 --mode ours

# 一次跑多个模式做对比
uv run python run_eval.py --city PHO --split test --n 39 --mode retrieval_only,ours,free

# 消融
uv run python run_eval.py --city PHO --n 38 --mode ours,no_agent,no_geo

# 离线调 α（复用 dump，不产生调用）
uv run python tune_alpha.py --file results/dump/<run>_ours.jsonl

# 融合权重网格搜索（validation，零成本）
uv run python tune_fusion.py --city PHO --split valid

# 用同一套指标复算 RALLM-POI 的预测文件
uv run python compare_rallm.py --city PHO --split test \
  --file "../RALLM-POI/experiments/<run>/predictions_<ts>.txt"

# 置信区间 / 配对显著性
uv run python bootstrap.py --file results/dump/PHO_test_39_ours.jsonl \
  --compare results/dump/PHO_test_39_retrieval_only.jsonl
```

**模式说明**

| 模式 | 含义 |
|---|---|
| `ours` | 完整方法（多路检索 + 工具 + 先验锚定 + 自适应校验） |
| `no_agent` | 去掉二次校验，单次调用 |
| `no_geo` | 去掉地理通道（消融） |
| `retrieval_only` | 只用融合排序，零 LLM 调用（召回上限诊断） |
| `free` | 无检索，LLM 自由生成（RALLM-POI 去 RAG 版） |
| `rag_free` | RALLM 式 TF-IDF RAG，但仍自由生成 |

---

## 7. 数据说明

复用 RALLM-POI 的预处理数据（只读，`RALLM-POI/` 保持原样）：

- 每个样本 = 若干天，每天是 6 个并行数组 `[POI_id, category_id, user_id, hour, flag, day_number]`；
- 展平后得到历史序列，**最后一天的最后一个 POI 为预测目标**（与 RALLM-POI 的 `test_model` 完全一致）；
- POI 编号为 `0..N-1`，与 `raw_data/{city}_poi_mapping.csv` 行号、`meta['POI']` 顺序一一对应；
- 类别共 10 类，来自 `raw_data/{city}_poi_full_info.csv` 的 `L1_Category`。

| 城市 | train | valid | test | POI 数 |
|---|---|---|---|---|
| PHO | 307 | 38 | 39 | 1430 |
| NYC | 4792 | 599 | 600 | 15754 |
| SIN | 5056 | 632 | 632 | 8974 |

**已知数据限制**：无营业时间字段（故用 P(category|hour) 替代）；无路网/交通数据（行程时间为几何近似）；跨城市 POI 词表独立、无共享 ID，故当前实验为城市内零样本，跨城市迁移需另设类别级映射，不在本轮范围内。

---

## 8. 评测口径

指标定义与 RALLM-POI 的 `calc_recall / calc_ndcg / calc_map` 逐行对齐：

```
HR@k     = #{label ∈ preds[:k]} / N
NDCG@k   = Σ 1/log2(pos+1) / N        (pos 从 1 开始)
MAP@k    = Σ 1/pos / N
```

**命名说明（重要）**：每个样本只有一个 ground truth，因此

```
HR@k = Recall@k = Accuracy@k
```

三者是同一个量——每个用户贡献一个二值指示 1[rank ≤ k]。RALLM-POI 的**论文**写 HR@k，
其**代码**却把同一函数命名为 `calc_recall`，所以早期版本沿用了 "Recall" 这一列名。
现统一以 **HR@k** 呈现（与论文一致），代码同时输出 `HR@k` / `Recall@k` / `Accuracy@k`
三个键，数值完全相同。

**一致性验证**：用本项目指标代码复算 RALLM-POI 的 `predictions_*.txt`，在 PHO/valid 上复现其 `metrics_*.csv` 的每一位（HR@1 0.15789、N@5 0.29788、M@10 0.27445）。因此下表中两行由同一函数产生。

补充指标（本工作新增）：候选召回率、空间跳变违规率、类别—时段违规率、非法 ID 率、调用次数/Token/延迟/成本。

---

## 9. 实验结果（v3）

> **重要更新（v3）**：检索层已升级为「时段条件一阶 + 二阶转移」的计数模型，
> 数字较早期版本大幅提升；同时发现 **LLM 排序组件为负贡献**。
> 完整对照表、α 扫描与重访分析见 [`results/SUMMARY.md`](results/SUMMARY.md)。

### 9.1 与各方法的差距（test 集，全量）

`HR@10` / `NDCG@10`：

| 方法 | 类型 | PHO (39) | NYC (600) | SIN (632) |
|---|---|---|---|---|
| UPop（用户最热） | 统计 | 0.615 / 0.393 | 0.475 / 0.265 | 0.571 / 0.371 |
| Markov1（一阶） | 统计 | 0.718 / 0.493 | 0.635 / 0.397 | 0.606 / 0.390 |
| ItemKNN | 统计 | 0.718 / 0.427 | 0.592 / 0.357 | 0.560 / 0.357 |
| TemporalMarkov（时段二阶） | 统计 | 0.795 / 0.669 | 0.775 / 0.659 | 0.758 / 0.655 |
| RALLM-POI（论文） | 免训练 LLM | 0.684 / 0.399 | 0.419 / 0.213 | 0.440 / 0.264 |
| **Ours v3（仅检索）** | 混合 | **0.897 / 0.688** | **0.783 / 0.658** | **0.759 / 0.645** |
| Ours v3 + LLM | 混合 | — | 0.777 / 0.609 | — |

### 9.2 结论

1. 相对 RALLM-POI，HR@10 提升 +0.20～+0.31（绝对）。
2. 但相对一个**纯计数的时段二阶 Markov**，多路融合的净收益只有 +0.102 / +0.008 / +0.001。
3. 加入 LLM 排序后指标全面下降；α 扫描单调指向 α=1（禁用 LLM）。
4. 因此：**贡献应归于时序检索建模，而非 LLM**。详见 SUMMARY.md 第 6 节的路线建议。

## 10. 关键发现

1. **最强的单一组件是「时段条件 + 二阶转移」的计数模型**：NYC HR@10 0.775、SIN 0.758，
   零训练、零 LLM 调用。
2. **多路融合的净收益很小**：相对该计数基线，NYC +0.008、SIN +0.001，PHO +0.102。
3. **LLM 重排为负贡献**：α 扫描单调指向 α=1（禁用），α=0.75 时 HR@10 从 0.783 降到 0.777，
   NDCG@10 从 0.658 降到 0.609。
4. **该 split 由重访主导**（NYC 57%、SIN 60% 的目标已在用户历史中），所有方法的 headline
   数字都被抬高，必须同时报「全新 POI」子集。
5. **跨数据集不可比**：Agent4POI 的 NYC 是 1,083 用户 / 11,357 POI，本项目是 2,862 / 15,754。

## 11. 局限与后续

- **单城市设定**：三城各自独立建模，跨城市迁移未实现（POI 词表不共享）。若做跨城，建议先做类别级映射。
- **无真实路网**：行程时间为几何近似，会低估城市内部绕行；接口已预留，可接 OSRM/高德。
- **PHO 样本极小**：valid/test 仅 38/39 条，置信区间宽；主结论应以 NYC/SIN 全量 test 为准。
- **消融未跑全量**：目前消融仍为 80 条子集，建议补全量以保证一致性。
- **单模型**：仅 gpt-4o-mini，建议补 gpt-4o / DeepSeek 的敏感性分析。

---

## 12. 目录结构

```
geo-rag-poi/
├── src/grpoi/
│   ├── config.py       路径、超参、API 配置
│   ├── data.py         pickle 适配、POI 表、转移图、P(cat|hour)
│   ├── retrieval.py    四路检索 + RRF 融合
│   ├── tools.py        距离/行程时间/类别-时段先验/违规检查
│   ├── llm.py          JSON 模式客户端 + 精确成本核算
│   ├── agent.py        提示词、先验锚定混合、自适应校验
│   ├── metrics.py      与 RALLM-POI 对齐的指标 + 空间/成本指标
│   └── baselines.py    free / rag_free 基线
├── run_eval.py         评测入口
├── tune_fusion.py      融合权重网格搜索
├── tune_alpha.py       离线 α 调优
├── compare_rallm.py    复算 RALLM-POI 预测
├── bootstrap.py        置信区间与配对检验
├── make_table.py       合并结果为表格
└── results/            指标 JSON、表格、逐样本 dump
```

---

## 13. 常见问题

**Q：`uv run` 报缓存目录不可写？**
先 `source ../.tools/uv-env.sh`。本机沙箱只允许写工作区，该脚本把 uv 缓存、解释器、matplotlib 缓存都指向 `.tools/`。

**Q：没设 API key 会怎样？**
`config.py` 自动读 `../.tools/.env`。缺失时 `retrieval_only` 仍可运行，其余模式会直接报错提示。

**Q：如何换模型？**
`GRPOI_MODEL=gpt-4o uv run python run_eval.py ...` 或改 `config.AGENT.model`。价格表在 `config.PRICES`。

**Q：结果 JSON 里的 `verified_rate` 是什么？**
触发二次校验的样本比例，当前约 0.33。

**Q：想复现论文表格？**
按第 9 节逐条执行；RALLM-POI 一侧需先在 `RALLM-POI/` 内以其自身 venv 运行 `main_rallm_poi.py`（我为其加了 `--limit` 参数，仅用于截断推理集，不改算法）。
