# API Testing

How to test a REST/JSON API with `aitestkit`: the `type: http` target posts
each case to your endpoint, extracts fields with dotted paths, and the
regular output + agentic metrics score the response. Self-contained guide.

```
case input ──► request_template ({input}) ──► POST endpoint ──► JSON
                                                     │
        response_path ──► output      context_path ──► retrieval_context
        tools_path    ──► trajectory (tool_call events)
```

---

## 1. Configure the target

```yaml
target:
  type: http
  url: "https://api.internal/v1/query"          # ${ENV_VAR} expands here
  method: POST                                   # any verb: GET/PUT/DELETE…
  headers: {Authorization: "Bearer ${API_KEY}"}  # secrets stay in env vars
  request_template: {q: "{input}", session: "{session_id}"}
  response_path: "data.answer"                   # → TargetOutput.output
  context_path: "data.sources"                   # → retrieval_context
  tools_path: "data.tool_calls"                  # → trajectory tool_calls
  kwargs: {timeout_seconds: 30}
```

| Placeholder | Filled from |
|---|---|
| `{input}` | `case.input` |
| `{context}` | `case.context` |
| `{session_id}` | conversation/case session |
| `{model}` / `{provider}` | `models:` matrix entry |

`request_template` interpolates recursively (dicts, lists, strings).
`response_path`/`context_path`/`tools_path` are dotted paths —
`data.answer`, `results.0.text` (numeric parts index into lists).

**No `response_path`?** A JSON dict falls back to its `output`/`text`
key, then the whole payload stringified; a non-JSON body becomes `output`
as raw text — `contains`/`regex_match` still work.

## 2. Write the dataset

```yaml
cases:
  - id: payment-status
    input: "payment status TX-78231"
    expected_tools:
      - {name: lookup, args: {q: "payment status TX-78231"}}  # tools_path
    tags: [api, banking, payments]
```

`expected_tools` pins both the tool name and args your endpoint reports
via `tools_path` — `tool_args_correct` catches wrong-argument bugs, not
just wrong-endpoint bugs.

## 3. Score it

```yaml
metrics:
  - name: contains                    # response-text assertions
    substrings: ["answer for"]
  - name: json_schema                 # when response_path yields JSON text
    schema: {type: object, required: [status]}
  - name: regex_match
    pattern: 'status":\s*"(ok|done)"'
  - name: tool_sequence               # tools_path → agentic metrics
  - name: tool_args_correct
  - name: latency
    max_ms: 2000                      # API responsiveness bound
  - name: max_cost                    # if the API reports usage
    max_usd: 0.01
```

HTTP ≥ 400 is an **infra error**, not a metric failure: the case lands
`status: error` with `reason: http target returned 403: …` — gated by
`max_error_cases`. (Expecting a 4xx as *correct* behavior isn't supported
— that's a custom-adapter job; see `docs/custom-adapters.md`.)

## 4. Run the demo

`suites/api/http_suite.yaml` targets a local stub. Start it (background):

```bash
uv run python -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        q = json.loads(self.rfile.read(
            int(self.headers['Content-Length']))).get('q', '')
        out = json.dumps({'data': {'answer': f'answer for {q}',
            'tool_calls': [{'name': 'lookup', 'args': {'q': q}}]}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass
HTTPServer(('127.0.0.1', 8791), H).serve_forever()
" &
uv run aitestkit run suites/api/http_suite.yaml -o evidence/api
```

```
PASS payment-status   contains 1.0 · tool_sequence 1.0 · latency 1.0
PASS fx-quote         same — all metrics green
```

For a real API: set the key, repoint `url:`:

```bash
API_KEY=sk-… uv run aitestkit run suites/api/http_suite.yaml -o out/
```

## 5. Validate the results

```json
{"case_id": "fx-quote",
 "actual_output": "answer for exchange rate eur",
 "latency_ms": 103.7,
 "trajectory": {"events": [{"type": "tool_call",
   "payload": {"name": "lookup", "args": {"q": "exchange rate eur"}}}]},
 "metrics": [{"metric": "tool_args_correct", "score": 1.0}]}
```

- `actual_output` ← `response_path` extraction
- `trajectory.events[]` ← `tools_path` entries, with `payload.name`/`args`
  feeding the agentic metrics
- `latency_ms` ← wall clock per request — the `latency` metric bound
- `status: error` + `error` text ← timeouts, unreachable host, ≥400

```bash
# which requests were slowest?
jq '.results[] | {case_id, ms: .latency_ms} | select(.ms > 100)' \
    evidence/api/results.json
```

## 6. Offline CI — cassettes

Record the endpoint once, replay byte-identical forever:

```bash
uv run aitestkit run suite.yaml --cassette cassettes/api.jsonl \
    --cassette-mode record
uv run aitestkit run suite.yaml --cassette cassettes/api.jsonl   # no keys
```

## Limits

- No status-code assertions — ≥400 is an infra `error`, you can't
  "expect a 404" as a scored test
- No response-header assertions, no multi-step request chaining
  (auth → call), no file upload — `request_template` is JSON-only
- One endpoint per suite — different routes → separate suites

## Config reference

| Key | Purpose |
|---|---|
| `url` | endpoint; `${ENV_VAR}` expands |
| `method` | HTTP verb (default `POST`) |
| `headers` | `${ENV_VAR}` per value |
| `request_template` | JSON body; `{input}`/`{session_id}`/… |
| `response_path` | dotted path → `output` |
| `context_path` | dotted path → `retrieval_context` |
| `tools_path` | dotted path → tool_call trajectory events |
| `kwargs.timeout_seconds` | request bound → `TargetError` |

## Related

- `docs/response-testing.md` — output metrics applied to API responses
- `docs/rag-testing.md` — `context_path` feeds retrieval metrics
- `docs/custom-adapters.md` — non-JSON protocols, auth flows, chaining
