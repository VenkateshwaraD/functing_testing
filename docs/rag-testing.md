# Testing RAG Pipelines End-to-End

How to test a RAG system with `aitestkit`: point a target at your pipeline,
declare golden retrieval labels in the dataset, score with retrieval +
generation + ingestion metrics, gate the run, read the reports.

## How it works

```
 query ─► ┌───────── your pipeline (SUT) ─────────┐
          │  retriever ─► reranker ─► generator   │
          └───────┬───────────────────────┬───────┘
                  │ retrieval_context/ids │ output
                  ▼                       ▼
          ┌───────────────┐       ┌───────────────┐
          │ retrieval     │       │ generation    │
          │ metrics       │       │ metrics       │
          │ recall@k mrr  │       │ faithfulness  │
          │ ndcg hit_rate │       │ groundedness  │
          └───────┬───────┘       │ correctness   │
                  │               └───────┬───────┘
                  ▼                       ▼
              per-case scores ──► gates ──► exit code
```

The key idea: your target returns **both** the final answer *and* the
retrieved chunks/ids for each case. Retrieval metrics score the chunks
against golden labels; generation metrics score the answer — so a failure
tells you *which stage* is broken, not just that the answer is wrong.

---

## Step 1 — Wire the target

### Python target (typical)

Return a dict with `output` plus retrieval fields:

```python
# myapp/rag.py
def answer(input: str, *, trace_sink=None, **kw) -> dict:
    hits = index.search(input, k=5)          # your retriever
    context = [h.text for h in hits]
    ids = [h.doc_id for h in hits]
    if trace_sink:
        trace_sink.retrieval(context)        # shows up in trajectory
    return {
        "output": generate(input, context),  # final answer text
        "retrieval_context": context,        # chunk texts, ranked
        "retrieval_ids": ids,                # chunk ids, ranked (preferred)
        "metadata": {"pre_rerank_ids": pre}, # optional, for rerank_delta
    }
```

### HTTP target

Extract answer + context from the JSON response:

```yaml
target:
  type: http
  url: "https://rag.internal/query"
  method: POST
  headers: {Authorization: "Bearer ${RAG_API_KEY}"}
  request_template: {q: "{input}"}
  response_path: "answer"
  context_path: "sources"        # retrieved passages for RAG metrics
```

### Testing stages separately

Point `target:` at each stage in different suites — e.g. one suite calls
`myapp:retrieve` (returns `retrieval_ids`, scored by `recall_at_k`/`mrr`),
another calls `myapp:generate` with `context` provided in the dataset
(scored by `faithfulness`). This isolates retrieval regressions from
generation regressions.

> Note: `stage_targets:` is declared in the config schema but not yet wired
> into the runner — separate suites per stage is the working approach.

---

## Step 2 — Write the dataset

```yaml
# datasets/rag_cases.yaml
cases:
  - id: paris-rag
    input: "capital of france"
    expected_output: "Paris is the capital of France."
    expected_retrieval_ids: ["paris:0"]        # golden chunk ids — enables
    context:                                  # exact (non-approximate) scoring
      - "Paris is the capital of France."
      - "Tokyo is the capital of Japan."
    tags: [rag, retrieval]

  - id: unanswerable
    input: "price of a manned mars flight"
    expected_retrieval_ids: []                 # should retrieve nothing good
    tags: [rag, adversarial]
```

| Field | Used by |
|-------|---------|
| `expected_retrieval_ids` | `recall_at_k`, `precision_at_k`, `mrr`, `ndcg`, `hit_rate`, `rerank_delta` |
| `context` | fallback relevance labels when ids are absent (metrics report `approximate: true`) |
| `expected_output` | `correctness`, `exact_match`, `contains` |
| `metadata.pre_rerank_ids` | `rerank_delta` — measures reranker lift |

---

## Step 3 — Configure the suite

```yaml
name: rag-regression

target:
  type: python
  function: "myapp.rag:answer"

datasets:
  - datasets/rag_cases.yaml

# Embedder — used by retrieval/vectordb metrics and chunk-compare --------
embedder:
  provider: openai                     # or fake for offline dev
  model: text-embedding-3-small

# Judge — for faithfulness / groundedness / correctness -------------------
judge:
  provider: openai
  model: gpt-4o-mini

# Vector store under test (optional) — enables vectordb_* metrics ---------
vectordb:
  type: chroma                         # memory|chroma|faiss|qdrant|pgvector|pinecone
  path: ./chroma_db                    # chroma/faiss: local path
  collection: docs                     # qdrant/pgvector: collection name
  # host/port/index_name for remote stores; keys via *_API_KEY env vars

metrics:
  # --- retrieval quality ------------------------------------------------
  - name: recall_at_k
    k: 5                               # |relevant ∩ retrieved[:k]| / |relevant|
  - name: precision_at_k
    k: 5
  - name: mrr                          # reciprocal rank of first hit
  - name: ndcg                         # graded ranking quality
  - name: hit_rate
    k: 5                               # fraction of cases with any hit @k
  - name: rerank_delta                 # lift vs metadata.pre_rerank_ids

  # --- generation quality (judge) ----------------------------------------
  - name: faithfulness                 # answer supported by context
    threshold: 0.8
  - name: groundedness                 # no claims outside context
  - name: answer_relevancy             # on-topic vs the question
  - name: contextual_recall            # context covers what's needed
  - name: contextual_precision         # relevant chunks ranked first
  - name: citation_accuracy            # cited sources actually say it
  - name: context_window
    max_tokens: 8000                   # stuffing guard
  - name: multi_turn_coherence         # for conversational RAG

  # --- ingestion / index health ------------------------------------------
  - name: ingestion_parse              # docs parsed cleanly
  - name: ingestion_metadata           # required metadata present
  - name: ingestion_dedup              # no duplicate chunks
  - name: vectordb_count               # index doc count sanity
  - name: embedding_drift              # embedding distribution shift

concurrency: 4
case_timeout_seconds: 120
max_cost_usd: 1.00                     # judge+embedder spend cap → exit 3

gates:
  min_pass_rate: 0.85
  max_error_cases: 0
  per_metric:
    faithfulness: {min_mean: 0.8, min_per_case: 0.5}
    recall_at_k:  {min_mean: 0.7}
```

Optional backends: `aitestkit[ragas]` / `aitestkit[deepeval]` expose the
full RAGAS / DeepEval metric suites via the `ragas` / `deepeval` metrics.

---

## Step 4 — Pick a chunking strategy (optional but recommended)

Before tuning metrics, compare how chunking affects retrieval:

```bash
# golden set — examples/golden/queries.yaml format:
#   schema_version: 1
#   corpus_id: capitals
#   queries:
#     - {id: q1, query: "capital of France", relevant_ids: ["paris:0"]}

aitestkit chunk-compare docs/corpus/ golden/queries.yaml \
    --strategies fixed,semantic,hierarchical \
    --embedder openai:text-embedding-3-small \
    --top-k 5 --chunk-size 512 --chunk-overlap 64
```

It re-chunks the corpus per strategy, runs the golden queries, and reports
retrieval quality per strategy — offline with `--embedder fake:fake`.

---

## Step 5 — Run it

```bash
# offline dev — fake judge + fake embedder, zero spend
aitestkit run rag_suite.yaml --judge fake -o out/

# real run
export OPENAI_API_KEY=...
aitestkit run rag_suite.yaml -o out/

# record once, replay in CI forever (provider calls → cassette)
aitestkit run rag_suite.yaml --cassette cassettes/rag.jsonl --cassette-mode record
aitestkit run rag_suite.yaml --cassette cassettes/rag.jsonl

# baseline the index — catch retrieval regressions on re-ingestion
aitestkit run rag_suite.yaml --baseline-save main
aitestkit run rag_suite.yaml --baseline .aitestkit/baselines/main.json
```

---

## Step 6 — Read the results

Same outputs as any suite (`results.json`, `results.jsonl`,
`aitestkit-report.html`, `junit.xml`). RAG-specific detail:

- **Per-metric `details`** — e.g. `recall_at_k` reports
  `{k, relevant, hits_at_k, approximate}`; `approximate: true` means it fell
  back to text matching because `expected_retrieval_ids` was absent — add
  golden ids for exact scoring.
- **Retrieval events in the trajectory** — `sink.retrieval(chunks)` calls
  appear in the per-case trajectory so you can see exactly what was fetched.
- **Judge rationales** — `faithfulness`/`groundedness` include the judge's
  reason per case, so a low score names the unsupported claim.
- **`trends`** — `embedding_drift` + `recall_at_k` over time is your
  index-rot early warning after re-ingestion or corpus changes.

Exit codes: `0` pass · `1` failures · `2` config · `3` budget · `4`
guardrail abort · `5` gate failure — so CI fails the job on retrieval or
faithfulness regressions automatically (see [ci-cd.md](ci-cd.md)).

---

## Worked example

```bash
cd examples
export PYTHONPATH=.
uv run aitestkit run suites/rag_suite.yaml -o /tmp/rag-out
```

`suites/rag_suite.yaml` runs `demo_app:rag` (canned retrieval context)
against `datasets/rag_cases.yaml` with `recall_at_k`, `hit_rate`,
`context_window`, `ingestion_parse`, `ingestion_dedup` — fully offline.
For a real-pipeline template with stage isolation, add a second suite whose
`target.function` points at your retriever alone.
