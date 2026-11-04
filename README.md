# geo-rag-poi

Training-free agentic Geo-RAG for next-POI recommendation — a stronger, transparent
alternative to RALLM-POI, evaluated on **exactly the same protocol** as RALLM-POI
(same pickles, same splits, same metric definitions).

**中文文档（Motivation / Contribution / 与已有工作的差异）：[README.zh-CN.md](README.zh-CN.md)**

## Why not just reproduce RALLM-POI

RALLM-POI ("TF-IDF retrieval + weighted-DTW rerank + one agentic LLM pass") is a fair
baseline but a weak centrepiece: a single retrieval channel, free-form LLM generation with
no candidate constraint, and metrics that cannot separate retrieval quality from ranking
quality. This project keeps the training-free premise and replaces each weak link:

| RALLM-POI | here |
|---|---|
| one TF-IDF trajectory channel | four complementary routes (semantic / transition graph / geo / user memory) fused by RRF |
| LLM does the geography itself | distance, travel time and category-hour plausibility are computed by tools and given to the LLM as facts |
| LLM emits POI ids from memory | LLM re-orders an explicit candidate list; ids outside it are rejected |
| always two LLM calls | critique pass only when the LLM reports low confidence or its output fails validation |
| retrieval quality not measurable | candidate recall is reported separately from ranking quality |

## Pipeline

```text
sample ──► 4 retrieval routes ──► RRF fusion (top-25 candidates)
              ├── semantic : TF-IDF over training trajectories      (RALLM-POI's channel)
              ├── graph    : POI transition counts from train set
              ├── geo      : distance decay + heading consistency
              └── memory   : user's own revisits × P(category|hour)
                                        │
                                        ▼
                        LLM ranks candidates (1 call) + self-confidence
                                        │
                     confidence < τ or invalid output? ──yes──► tool-grounded critique (1 call)
                                        │no
                                        ▼
                 score = α·retrieval_prior + (1-α)·LLM_order   (α tuned on valid)
                                        │
                                        ▼
                                    top-10
```

The convex blend (default α = 0.75) is what stops listwise LLM reranking from destroying a
good retrieval order — an effect we measured directly (see *Findings*).

## Usage

```bash
cd geo-rag-poi
source ../.tools/uv-env.sh     # uv cache/python/matplotlib dirs inside the workspace
uv sync

# retrieval only, no API cost
uv run python run_eval.py --city PHO --n 30 --mode retrieval_only

# full method (uses DMX_API_KEY / DMX_BASE_URL from ../.tools/.env)
uv run python run_eval.py --city PHO --split test --n 39 --mode retrieval_only,ours,free

# ablations
uv run python run_eval.py --city PHO --n 38 --mode ours,no_agent,no_geo

# offline re-tuning of α from a dumped run (free)
uv run python tune_alpha.py --file results/dump/<run>_ours.jsonl

# fusion weight grid search (free, validation split only)
uv run python tune_fusion.py --city PHO --split valid

# re-score a RALLM-POI predictions file with identical metric code
uv run python compare_rallm.py --city PHO --split test \
  --file "../RALLM-POI/experiments/<run>/predictions_<timestamp>.txt"
```

Modes: `ours` · `no_agent` · `no_geo` · `free` (LLM only, no retrieval) ·
`rag_free` (RALLM-style TF-IDF RAG, free-form) · `retrieval_only` (no LLM).

## Metric parity with RALLM-POI

`compare_rallm.py` re-scores RALLM-POI's own `predictions_*.txt` with this project's
metric code; on PHO/valid it reproduces RALLM-POI's `metrics_*.csv` digit for digit
(HR@1 0.15789, N@5 0.29788, M@10 0.27445), so the two methods are scored by one function.

## Data notes

* Reuses RALLM-POI's pickles read-only; that repo stays pristine (one `--limit` flag added
  to its evaluation harness so both methods can run on identical subsets).
* The Foursquare splits contain **no opening-hours field**, so "opening status" is replaced
  by an empirical `P(category | hour)` prior from the training check-ins.
* No routing API: travel time is `1.3 × great-circle distance / speed`, walking ≤1.5 km,
  transit above — a stated approximation with a replaceable interface (`tools.py`).

## Results (v3, test splits, full data)

The retrieval layer was upgraded to a **time-conditioned second-order transition model**
(3-hour buckets) plus three auxiliary routes. Full comparison, alpha sweep and revisit
analysis: `results/SUMMARY.md`.

HR@10 / NDCG@10:

| Method | Type | PHO (39) | NYC (600) | SIN (632) |
|---|---|---|---|---|
| UPop (user's own popular) | statistical | 0.615 / 0.393 | 0.475 / 0.265 | 0.571 / 0.371 |
| Markov1 | statistical | 0.718 / 0.493 | 0.635 / 0.397 | 0.606 / 0.390 |
| ItemKNN | statistical | 0.718 / 0.427 | 0.592 / 0.357 | 0.560 / 0.357 |
| TemporalMarkov (hour + 2nd order) | statistical | 0.795 / 0.669 | 0.775 / 0.659 | 0.758 / 0.655 |
| RALLM-POI (paper) | zero-shot LLM | 0.684 / 0.399 | 0.419 / 0.213 | 0.440 / 0.264 |
| **Ours v3 (retrieval only)** | hybrid | **0.897 / 0.688** | **0.783 / 0.658** | **0.759 / 0.645** |
| Ours v3 + LLM | hybrid | — | 0.777 / 0.609 | — |

Key caveat: the gain over RALLM-POI is large (+0.20 to +0.31 absolute), but the gain over a
pure counting TemporalMarkov baseline is small (+0.102 / +0.008 / +0.001), and the LLM
re-ranking step *degrades* every metric (alpha sweep is monotonic towards alpha = 1, i.e.
no LLM). The honest contribution is the temporal retrieval model, not the LLM.

## Findings

1. A time-conditioned second-order transition model is the single strongest component:
   HR@10 0.775 (NYC) and 0.758 (SIN) with zero training and zero LLM calls.
2. Multi-route fusion adds only +0.008 (NYC) / +0.001 (SIN) over that counting baseline,
   though it helps more on PHO (+0.102).
3. LLM re-ranking *hurts*: the alpha sweep is monotonic towards alpha = 1 (no LLM),
   HR@10 drops from 0.783 to 0.777 and NDCG@10 from 0.658 to 0.609 at alpha = 0.75.
4. These splits are revisit-dominated (NYC 57%, SIN 60% of targets were already in the
   user's own history), which inflates every method's headline number; report the
   novel-POI subset too.
5. Numbers are not comparable across Foursquare-NYC releases (1,083 users / 11,357 POIs in
   Agent4POI versus 2,862 / 15,754 here).

## Layout

```
src/grpoi/
  config.py     paths, hyper-parameters, API client settings
  data.py       RALLM-POI pickle adapter, POI tables, transition graph, P(cat|hour)
  retrieval.py  four routes + RRF fusion
  tools.py      distance / travel time / category-hour plausibility / violations
  llm.py        JSON-mode client with exact cost accounting
  agent.py      prompts, blending, adaptive verification
  metrics.py    RALLM-compatible Recall/NDCG/MAP + spatial/cost metrics
run_eval.py     CLI driver
tune_fusion.py  fusion weight grid search
tune_alpha.py   offline α tuning from dumps
compare_rallm.py re-score RALLM-POI predictions
make_table.py   merge runs into one markdown table
```
