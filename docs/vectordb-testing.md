# Vector DB Validation — Index Integrity, Embedding Drift, Consistency

Point a suite at your index and the `vectordb_*` metrics inspect it
directly. The framework validates the store — it does not index into it;
your pipeline (or a setup step) populates the DB first. Self-contained
guide.

```
suite.vectordb ──► build_vector_store ──► ctx.vector_store
suite.embedder ─┘         │                        │
                          ▼                        ▼
              opens your real index      vectordb_count · embedding_drift
              (chroma path / faiss file / remote host)
```

---

## 1. Configure

```yaml
embedder:                      # required — vectordb metrics need it
  provider: openai             # or fake for offline
  model: text-embedding-3-small

vectordb:
  type: faiss                  # memory|chroma|faiss|qdrant|pgvector|pinecone
  dims: 8                      # faiss: index dimension
  path: ./db/index             # chroma/faiss: persisted store location
  # collection: docs           # chroma/qdrant/pgvector: collection name
  # host: ..., port: ...       # remote stores
  # index_name: ...            # pinecone
```

Store types and their extras (`uv sync --extra <name>`):

| `type` | Extra | Key config | Persistence |
|---|---|---|---|
| `memory` | — (built-in) | `model` | ephemeral — empty each run |
| `chroma` | `chroma` | `collection`, `path` | persisted at `path` |
| `faiss` | `faiss` | `dims`, `path` | `<path>.faiss` + `<path>.json` sidecar |
| `qdrant` | `qdrant` | `host`/`url`, `collection` | server |
| `pgvector` | `pgvector` | connection, `collection` | postgres |
| `pinecone` | `pinecone` | `index_name`, keys | cloud |

Missing extras raise a scoped `ConfigError` naming the install command.

## 2. The three checks

**Index integrity — `vectordb_count`**

```yaml
- name: vectordb_count
  expected: 1240          # exact match, or:
  # min_count: 1          # at-least-N sanity (default)
```

Catches failed/partial re-ingestion. `details.count` reports the actual.

**Embedding drift — `embedding_drift`**

```yaml
- name: embedding_drift
  sample_fraction: 0.2    # % of stored chunks re-embedded
  threshold: 0.95         # default; mean cosine old-vs-new
```

Re-embeds stored texts with the configured `embedder:` and computes
cosine similarity against the stored vectors — a silent embedder,
chunker, or preprocessing change drops the score. `details` reports
`sampled`, `mean_similarity`, `min_similarity`, `worst_chunk` (the chunk
id that drifted most). Empty store → skipped, not failed.

**Retrieval consistency** — scored per query against golden ids:
`recall_at_k`, `precision_at_k`, `mrr`, `ndcg`, `hit_rate`,
`rerank_delta` (see `docs/rag-testing.md`). For consistency *over time*,
combine with `repeat:` + baselines/`compare` (`docs/regression-testing.md`).

## 3. Populate the index (your side)

The suite only *reads* the store. Persist the same path/connection your
pipeline uses:

```python
# e.g. ingestion step — same path the suite points at
store = FaissVectorStore(embedder, dims=8, path="evidence/vectordb/index")
await store.upsert([Chunk(id="doc0:0", text="…", doc_id="doc0", metadata={})])
```

`memory` type is ephemeral — only useful for skip-semantics checks or
in-process flows; real validation targets persisted/remote stores.

## 4. Run

Shipped demo — 5-chunk persisted FAISS index + fake embedder, fully
offline (`uv sync --extra faiss` first):

```bash
PYTHONPATH=examples uv run aitestkit run \
    suites/rag/vectordb_suite.yaml -o evidence/vectordb/out
```

Actual result — both checks scored from the real index:

```
PASS index-health
vectordb_count    1.0   {count: 5}                    # expected: 5 ✓
embedding_drift   1.0   {sampled: 5, mean_sim: ≈1.0, worst_chunk: doc3:0}
gate PASS
```

## 5. Validate the results

```bash
# drift detail — which chunks moved most
jq '.results[].metrics[] | select(.metric=="embedding_drift") | .details' \
   evidence/vectordb/out/results.json
# → {sampled, mean_similarity, min_similarity, worst_chunk}
```

A drift score just under threshold + a `worst_chunk` id = re-index that
document (or the whole index if the embedder changed globally).

## Limitations

- Count-level integrity only — no dimension-mismatch or orphan-id check
  (a dimension break surfaces as a store `MetricError`, not a clean score)
- Remote-store `reembed_sample` may be unsupported → scoped
  `MetricError` (embedding_drift errors on that store, other metrics
  unaffected)
- No framework-side ingestion into `vectordb:` — populate first

## Config reference

| Key | Purpose |
|---|---|
| `vectordb.type` | memory\|chroma\|faiss\|qdrant\|pgvector\|pinecone |
| `vectordb.path` | chroma/faiss persisted location |
| `vectordb.dims` | faiss index dimension |
| `embedder.provider/model` | embedder used for drift + retrieval metrics |
| `vectordb_count.expected/min_count` | integrity bound |
| `embedding_drift.sample_fraction` | % of index re-embedded |

## Related

- `docs/rag-testing.md` — retrieval metric family, chunk-compare
- `docs/regression-testing.md` — drift across runs, baselines
- `docs/getting-started.md` — suite anatomy
