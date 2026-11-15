# Custom Providers & Target Adapters

How to plug **your own provider class** (judge/embedder/model) and **your
own target adapter** (bespoke SUT client) into `aitestkit` — no fork
required. Self-contained guide.

## How it works

```
suite YAML                      registry/factory              your code
─────────────                   ─────────────────             ─────────
judge:                          providers/registry.py         class MyJudge
  provider: "myco.judge:J"  ──► _load_impl("module:Class") ─►  (ChatProvider)

providers:                                                  class MyEmbedder
  myco: {impl: "myco:E"}     ──► isinstance(Embedder) check ─► (Embedder)

target:
  type: custom                  targets/factory.py            class MyAdapter
  function: "myco:Adapter"   ──► isinstance(TargetAdapter) ─►  (adapter)
```

Two extension points, both resolved via `module:Class` import specs and
checked against a runtime `Protocol` — a wrong-shaped class fails fast with
a `ConfigError`, not a mid-run crash.

---

## Part 1 — Custom provider

### The contracts (`providers/base.py`)

```python
# ChatProvider — used by the judge and judge-backed metrics
class MyChatProvider:
    name = "myco"                              # required attr

    async def complete(
        self,
        messages: list[dict[str, str]],        # [{"role": ..., "content": ...}]
        *,
        model: str,
        temperature: float = 0.0,
        response_format: str | None = None,
        seed: int | None = None,
    ) -> ProviderResponse:                     # text + usage + raw
        ...

# Embedder — used by retrieval/vectordb metrics and chunk-compare
class MyEmbedder:
    name = "myco"

    async def embed(
        self, texts: list[str], *, model: str
    ) -> tuple[list[list[float]], Usage]:
        ...
```

`ProviderResponse` / `Usage` live in `aitestkit.providers.base` /
`aitestkit.models`. Constructor kwargs: the framework may pass
`rate_limiter=` — accept `**kwargs` to stay forward-compatible.

### Example — a provider wrapping your internal gateway

```python
# myco/providers.py
import httpx
from aitestkit.models import Usage
from aitestkit.providers.base import ProviderResponse

class MyCoProvider:
    name = "myco"

    def __init__(self, **kw) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://llm.myco.internal", timeout=30
        )

    async def complete(self, messages, *, model, temperature=0.0,
                       response_format=None, seed=None) -> ProviderResponse:
        r = await self._client.post("/v2/chat", json={
            "model": model, "messages": messages, "temperature": temperature,
        })
        r.raise_for_status()
        body = r.json()
        return ProviderResponse(
            text=body["text"],
            usage=Usage(input_tokens=body["usage"]["in"],
                        output_tokens=body["usage"]["out"]),
            raw=body,
        )
```

### Wiring it into a suite — two equivalent ways

```yaml
# A) direct module:Class spec — judge/embedder fields accept it inline
judge:
  provider: "myco.providers:MyCoProvider"
  model: myco-judge-v1

# B) named provider with impl: — required for the models: matrix, since
#    "provider:model" can't hold a class path
providers:
  myco:
    provider: myco
    model: myco-judge-v1
    impl: "myco.providers:MyCoProvider"
    api_key_env: MYCO_API_KEY     # optional — only if your class needs it
    rpm: 60

judge: {provider: myco}           # resolves via the providers: entry
models: ["myco:myco-judge-v1"]    # matrix entry — provider name = map key
```

Notes:

- **Auth**: a custom provider with no `api_key_env` is trusted to manage
  its own credentials — env validation is skipped for `impl` entries and
  `module:Class` specs. Set `api_key_env` to enforce a specific variable.
- **Verification**: a class missing `name` or `complete()` fails at suite
  load with `ConfigError: custom provider ... must implement the
  ChatProvider protocol`.
- Judge metrics expect JSON verdicts from the judge — a custom judge
  provider must return `{"score": 0..1, "reason": "..."}` text, same as
  any judge backend.

---

## Part 2 — Custom target adapter

Use `type: custom` when `http`/`python`/`command`/`langgraph` can't express
your SUT — e.g. a websocket protocol, a multi-call flow, a SDK client with
rich session state.

### The contract (`targets/base.py` → `TargetAdapter`)

```python
class MyAdapter:
    name = "myrest"                            # required attr

    async def invoke(
        self,
        case: TestCase,                        # input, expected_*, context, metadata
        *,
        model: str | None = None,
        provider: str | None = None,           # models: matrix entry parts
        tool_mocks: dict | None = None,
        session_id: str | None = None,
        trace_sink=None,                       # emit trajectory events
        tool_registry=None,                    # suite tools: registry
        prompt: str | None = None,             # prompts: matrix variant
        prompt_name: str | None = None,
    ) -> TargetOutput:
        ...
```

Return a `TargetOutput` (`aitestkit.models`) — fields: `output`,
`trajectory`, `tool_calls`, `retrieval_context`, `retrieval_ids`, `usage`,
`latency_ms`, `error`, `metadata`.

### Example — adapter for a two-call internal agent API

```python
# myco/adapter.py
import httpx
from aitestkit.models import TargetOutput

class TwoCallAdapter:
    name = "myco-agent"

    def __init__(self, *, base_url: str = "https://agent.myco.internal") -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30)

    async def invoke(self, case, *, trace_sink=None, **kw) -> TargetOutput:
        r1 = await self._client.post("/plan", json={"task": case.input})
        plan = r1.json()["plan"]
        if trace_sink:
            trace_sink.thought(plan["rationale"])
        r2 = await self._client.post("/act", json={"plan_id": plan["id"]})
        result = r2.json()
        for call in result.get("tool_calls", []):
            if trace_sink:
                trace_sink.tool_call(call["name"], args=call.get("args"))
        return TargetOutput(
            output=result["answer"],
            tool_calls=[{"name": c["name"], "args": c.get("args", {})}
                        for c in result.get("tool_calls", [])],
        )
```

### Wiring it

```yaml
target:
  type: custom
  function: "myco.adapter:TwoCallAdapter"   # class (or factory) — instantiated
  kwargs: {base_url: "https://staging.myco.internal"}   # → constructor args
```

- `function` may resolve to a **class** (instantiated with `kwargs:`) or a
  **factory/adapter instance** (used as-is).
- Constructor kwargs come from the `kwargs:` block.
- Runs under the `python`-target security gate: enabled by default,
  `--no-allow-python-targets` blocks it in locked-down CI.
- A class missing `name` or `invoke()` fails at load with a `ConfigError`.

---

## Part 3 — Testing it end-to-end

```yaml
# suite.yaml
name: custom-e2e

target:
  type: custom
  function: "myco.adapter:TwoCallAdapter"

datasets: [datasets/cases.yaml]

judge: {provider: myco, model: myco-judge-v1}
providers:
  myco:
    provider: myco
    model: myco-judge-v1
    impl: "myco.providers:MyCoProvider"

metrics:
  - {name: correctness, threshold: 0.7}   # judged by YOUR provider
  - exact_match

gates:
  min_pass_rate: 0.9
```

```bash
export PYTHONPATH=.                        # your adapter module importable
aitestkit run suite.yaml -o out/           # everything else works as usual:
# cassettes, baselines, repeat/flaky, gates, reports, exit codes — all apply
```

Everything composes: custom providers can back the judge **and** model
matrix, custom adapters get `trace_sink`/`tool_registry`/`prompt` kwargs
like built-in targets, and both are checked at suite load — a malformed
class never makes it past `ConfigError`.

### Minimal offline verification

A do-nothing custom provider + adapter proves the wiring with zero keys:

```python
class Ping:
    name = "ping"
    async def complete(self, messages, *, model, **kw):
        from aitestkit.providers.base import ProviderResponse
        from aitestkit.models import Usage
        return ProviderResponse(text='{"score": 1.0, "reason": "ok"}',
                                usage=Usage())
```

```yaml
judge: {provider: "mypkg.providers:Ping", model: ping-v1}
```
