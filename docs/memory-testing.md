# Conversation & Memory Testing

How to test multi-turn conversations, follow-up questions, and agent memory
with `aitestkit`: `conversations:` blocks run turns over one session,
memory metrics check what the agent retained — including **direct store
inspection**, not just re-asking the model. Self-contained guide.

```
conversation case ──► turn 0 ──► turn 1 ──► … ──► turn N
                        │ same session_id for every turn │
                        ▼                              ▼
                   target keeps state          metrics score the last answer
                                                   + MemoryStore contents
```

---

## 1. Write a conversation — `conversations:` + `turns:`

`dataset_testcases/memory_cases.yaml`:

```yaml
conversations:
  - id: name-remember
    session_id: sess-demo-1           # optional — defaults to the conv id
    turns:
      - input: "my name is ada"       # turn 0: plant a fact
      - input: "what is my name"      # turn 1: follow-up question
    metadata:
      must_remember: ["ada"]          # drives context_retention
    tags: [memory, conversation]
```

Turns run **sequentially** in order — never in parallel — and each one
produces its own result (`name-remember/turn-0`, `…/turn-1`). The runner
stamps `session_id`, `_conversation`, `_turn`, `_final_turn` into case
metadata; `_final_turn` is what "score the last answer" metrics key on.

Your target gets the session on every call — keep state keyed by it:

```python
def chat(input: str, *, session_id: str | None = None) -> str:
    store.remember(session_id, input)      # your memory logic
    return reply
```

HTTP targets substitute `{session_id}` into `request_template` fields —
a real chat API's conversation/thread id slots straight in.

## 2. Wire the memory store — `memory:`

Suite-level hook that makes the SUT's memory inspectable:

```yaml
memory:
  store: "myapp.memory:store"     # module:attr → a MemoryStore instance
```

`MemoryStore` (`aitestkit.agentic.memory`) interface: `get` / `set` /
`search` / `all` / `clear`, all `session_id`-scoped. Ship
`JsonFileMemoryStore` for restart-persistence testing (does the agent
still remember after the process dies?).

Without `memory:`, `forgetting_check`/`memory_accuracy` report a metric
error ("no memory_store configured"), nothing else.

## 3. Score it — memory metrics

| Metric | Needs | Checks | Judge? |
|---|---|---|---|
| `context_retention` | `metadata.must_remember` | facts present in the final answer | no |
| `contradiction_resolution` | `metadata.superseded` + `current` | newest fact wins; 0.5 if both surface | no |
| `forgetting_check` | `metadata.memory_key` (+`memory_value`) | `store.get(key)` still there, right value | no |
| `memory_accuracy` | populated store | answer consistent with stored facts | **yes** |
| `multi_turn_coherence` | multi-turn output | final answer consistent with earlier turns | **yes** |

Cases missing a metric's metadata are **skipped**, not failed — the same
case list can mix memory and plain cases.

Single (non-conversation) cases can still play: `metadata.session_id`
targets a specific session — the demo's `sess-seeded` exercises
pre-populated memory without a `conversations:` block:

```yaml
  - id: currency-moved
    input: "what is my preferred currency"
    metadata:
      session_id: sess-seeded
      superseded: "eur"         # agent was told eur earlier,
      current: "usd"            # must answer usd now
```

## 4. Run it

```bash
PYTHONPATH=examples uv run aitestkit run suites/memory/memory_suite.yaml \
    -o evidence/memory
```

Fully offline — `context_retention`, `forgetting_check`, and
`contradiction_resolution` are deterministic. Add `memory_accuracy` /
`multi_turn_coherence` only with a judge configured.

## 5. Validate the results

```
name-remember/turn-0  context_retention 1.0  retained 1/1 facts
name-remember/turn-1  context_retention 1.0  retained 1/1 facts
name-recall           forgetting_check  1.0  key 'name' retained
moved-city            contradiction_resolution 1.0  uses updated fact
```

Each turn lands in `results.json` with its `session_id`:

```json
{"case_id": "name-remember/turn-1", "session_id": "sess-demo-1",
 "actual_output": "your name is ada",
 "metrics": [{"metric": "context_retention", "score": 1.0,
   "details": {"retained": ["ada"], "lost": []}}]}
```

What to look for per metric:

- **`context_retention`** — `details.lost` names the forgotten facts
- **`contradiction_resolution`** — `details.used_superseded: true` means
  the stale fact resurfaced (score 0.5 when both old+new appear)
- **`forgetting_check`** — `details` shows the actual stored `value` vs
  expected `memory_value` — a retention failure you can see in the store
- **`memory_accuracy`** — `details.facts_seen` + judge rationale

```bash
# which facts were lost across all turns?
jq '.results[].metrics[] | select(.metric=="context_retention")
    | .details.lost[]?' evidence/memory/results.json
```

## Config reference

| Key | Purpose |
|---|---|
| `conversations[].turns` | ordered inputs; run sequentially, share session |
| `conversations[].session_id` | session key for the target + store |
| `metadata.session_id` | pin a single case to a session |
| `metadata.must_remember` | facts `context_retention` requires |
| `metadata.superseded`/`current` | stale vs updated fact |
| `metadata.memory_key`/`memory_value` | `forgetting_check` store lookup |
| `memory.store` | `module:attr` — MemoryStore for inspection metrics |

## Related

- `docs/agent-testing.md` — trajectories, tool calls, guardrails
- `docs/response-testing.md` — per-turn output metrics
- `docs/rag-testing.md` — `multi_turn_coherence` for conversational RAG
