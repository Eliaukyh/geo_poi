# geo-rag-poi

**证据驱动的生成式零样本下一 POI 推荐**（training-free, evidence-grounded generation）

检索四类证据 → LLM 自由生成 POI → 只做合法性修复。不训练任何参数，每样本 1 次 LLM 调用。

> 方法与实验的完整说明见 **[PAPER_ZH.md](PAPER_ZH.md)**。

## 主结果（test 全量，gpt-4o-mini）

| 方法 | PHO (39) | NYC (600) | SIN (632) | 调用/样本 |
|---|---|---|---|---|
| RALLM-POI（论文值） | 0.684 | 0.419 | 0.440 | 2.0 |
| **本方法** | **0.692** | **0.562** | **0.587** | **1.0** |

指标为 HR@10（单目标下等于 Recall@10 / Accuracy@10）。

## 快速开始

```bash
cd geo-rag-poi
source ../.tools/uv-env.sh        # uv 缓存 / matplotlib 缓存目录
uv sync

# 生成式主方法（调用 LLM）
uv run python run_eval.py --city NYC --split test --n 600 --mode gen --workers 12

# 零成本对照：无证据生成 / 约束式排序
uv run python run_eval.py --city PHO --split test --n 39 --mode free
uv run python run_eval.py --city PHO --split test --n 39 --mode retrieval_only
```

API 凭据在 `../.tools/.env`（`DMX_API_KEY` / `DMX_BASE_URL`），不随仓库分发。

## 方法一句话

```
四源证据（转移 / 个体记忆 / 地理邻域 / 语义邻居）
        ↓  写成事实，不写成候选菜单
LLM 自由生成 10 个 POI ID（可以是证据里没有的）
        ↓
合法性修复：丢弃越界与重复 → 从证据补足 → 有效 ID 不改动
```

## 目录

```
src/grpoi/
  generative.py   主方法：证据构建 + 生成 + 修复
  retrieval.py    四路证据来源
  data.py         离线统计（转移计数 / 用户画像 / 类别-时段先验）
  fusion.py       候选融合（仅用于修复补位与消融）
  agent.py        约束式选择变体（消融用）
  baselines.py    无检索基线与 RAG 基线
  classic_baselines.py  统计基线（POP / UPop / Markov / ItemKNN / GeoN）
  metrics.py      指标（与 RALLM-POI 实现逐位对齐）
  llm.py          客户端与成本核算

run_eval.py             主评测入口（gen / ours / free / retrieval_only）
run_baselines.py        统计基线
compare_rallm.py        用同一套指标复算 RALLM-POI 的预测文件
compare_fusion.py       融合策略对比（RRF / minmax / union）
compare_two_stage.py    两阶段融合（消融）
run_think2go_bench.py   跨基准评测（GNPR / Think2Go 划分）
```

## 文档

| 文件 | 内容 |
|---|---|
| [PAPER_ZH.md](PAPER_ZH.md) | 方法、贡献、全量对比、消融（主文档） |
| [RELATED_RALLM_ZH.md](RELATED_RALLM_ZH.md) | RALLM-POI 与 RALLM-STAR 的做法与差异 |
| [results/FUSION_ANALYSIS.md](results/FUSION_ANALYSIS.md) | 融合策略分析（RRF 为什么失效） |
| [results/THINK2GO_BENCH.md](results/THINK2GO_BENCH.md) | 跨基准边界条件 |

## 结果文件

`results/` 只保留论文引用到的结果：

- `*_test_gen.json`：主方法三城结果
- `*_test_twostage.json`：约束式消融
- `*_test_classic.json`：统计基线
- `rallm_*.json`：RALLM-POI 的本地复现
- `PHO_test_39.json`：含无证据基线（`free`）
