# geo-rag-poi

**多视图证据 + 用户画像 + 视角增强的零样本下一 POI 推荐**
（training-free · evidence-grounded generation · prompt-level TTA）

不训练任何参数：检索四类证据 → LLM 自由生成 POI → Borda 聚合 → 只做合法性修复。

> 完整方法与实验见 **[PAPER_ZH.md](PAPER_ZH.md)**，
> 方法细节（公式、提示词、算法）见 **[METHOD_ZH.md](METHOD_ZH.md)**。

## 主结果（test 全量，gpt-4o-mini，3 次调用/样本）

`HR@10 / NDCG@10`：

| 方法 | PHO (39) | NYC (600) | SIN (632) |
|---|---|---|---|
| RALLM-POI（论文值） | 0.684 / 0.399 | 0.419 / 0.213 | 0.440 / 0.264 |
| **本方法（证据 + 画像 + TTA）** | **0.692 / 0.484** | **0.587 / 0.389** | **0.644 / 0.453** |
| 单次生成（消融基线） | 0.692 / 0.480 | 0.562 / 0.358 | 0.587 / 0.415 |
| TemporalMarkov（零 LLM） | 0.795 / 0.669 | 0.775 / 0.659 | 0.758 / 0.655 |

相对单次生成，NYC +2.5pt、SIN +5.7pt（配对 bootstrap 显著）。
纯统计基线仍更高——这是自由生成的固有代价，见论文 §4。

## 消融（NYC/test 600）

| 配置 | HR@10 | NDCG@10 |
|---|---|---|
| 单次生成 | 0.562 | 0.358 |
| + LLM 用户画像 | 0.565 | 0.370 |
| + 视角增强（TTA） | 0.588 | 0.387 |
| + 两者 | 0.587 | 0.389 |

画像在 TTA 之上没有叠加（0.587 vs 0.588），其单独贡献亦不显著。

## 快速开始

```bash
cd geo-rag-poi
source ../.tools/uv-env.sh        # uv 缓存 / matplotlib 缓存目录
uv sync

# 最终配置（画像 + TTA，3 次调用/样本）
uv run python run_final.py --city NYC --n 600 --workers 12 \
  --dump results/dump/NYC_test_gen_gen.jsonl

# 单次生成（消融基线）
uv run python run_eval.py --city NYC --split test --n 600 --mode gen --workers 12
```

API 凭据放在 `../.tools/.env`（`DMX_API_KEY` / `DMX_BASE_URL`），不随仓库分发。

## 方法一句话

```
① 证据：转移视图 + 个体视图 + 空间视图 + 语义视图（写成事实，带次数/距离）
② 画像：LLM 把长历史压成"回访倾向 + 作息"（按用户缓存，一次调用）
③ 生成：LLM 自由生成 10 个 POI（允许证据之外）
④ 增强：三种视角提示各生成一次（默认 / 证据倒序 / 居民视角）
⑤ 聚合与修复：Borda 计数 → 丢弃越界与重复 → 证据补位
```

## 目录

```
src/grpoi/
  generative.py   主方法：证据序列化 + 生成 + 修复
  profile.py      LLM 用户画像（按用户缓存）
  retrieval.py    四路证据来源（转移 / 个体 / 空间 / 语义）
  data.py         离线统计（转移计数 / 用户画像 / 类别-时段先验 / 日号）
  fusion.py       候选融合（仅用于修复补位与约束式消融）
  agent.py        约束式选择变体（消融用）
  baselines.py    无检索基线
  classic_baselines.py  统计基线（POP / UPop / Markov / ItemKNN / GeoN）
  metrics.py      指标（与 RALLM-POI 实现逐位对齐）
  llm.py          客户端与成本核算

run_final.py            最终配置：画像 + TTA + 聚合 + 修复
run_eval.py             主评测入口（gen / ours / free / retrieval_only）
run_prompt_ensemble.py  TTA 消融（同提示词 vs 异提示词）
run_soft_hint_full.py   意图干预消融（软提示）
run_profile_ablation.py 画像消融
run_baselines.py        统计基线
compare_rallm.py        用同一套指标复算 RALLM-POI 的预测文件
compare_two_stage.py    约束式选择（消融上界）
run_think2go_bench.py   跨基准评测（GNPR / Think2Go 划分）
```

## 文档

| 文件 | 内容 |
|---|---|
| [PAPER_ZH.md](PAPER_ZH.md) | 背景、缺陷论证、方法、全量结果、消融 |
| [METHOD_ZH.md](METHOD_ZH.md) | 方法细节：四视图公式、提示词、算法、成本 |
| [RELATED_RALLM_ZH.md](RELATED_RALLM_ZH.md) | RALLM-POI 与 RALLM-STAR 的做法与差异 |
| [results/THINK2GO_BENCH.md](results/THINK2GO_BENCH.md) | 跨基准边界条件 |

## 结果文件

`results/` 只保留论文引用到的结果：

- `final_*.json`：最终配置三城结果
- `*_test_gen.json`：单次生成（消融基线）
- `*_test_classic.json`：统计基线
- `*_test_twostage.json`：约束式选择（消融上界）
- `prompt_ensemble_NYC_600.json`：TTA 消融
- `soft_hint_NYC_600.json`：意图干预消融
- `profile_NYC_600.json`：画像消融
- `rallm_*.json`：RALLM-POI 的本地复现
- `PHO_test_39.json`：含无证据基线（`free`）
