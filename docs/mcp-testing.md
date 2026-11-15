# MCP Testing

How to test a **Model Context Protocol** server with `aitestkit`: the
`type: mcp` target spawns or connects to your server, calls a configured
tool per case, and records the call so every agentic/response metric
scores real MCP behavior. Requires `aitestkit[mcp]`.

```
suite case ──► mcp target ──► ClientSession ──► call_tool(name, args)
   input          │                                   │
                  ▼                                   ▼
           args_template ({input})        output + ToolCall + trajectory
                  └──────────────► metrics (tool_sequence, forbidden…) ──► gates
```

---

## 1. Install + pick a transport

```bash
uv sync --extra mcp            # or: pip install 'aitestkit[mcp]'
```

Two transports, chosen by which key you set:

| Key | Transport | Security gate |
|---|---|---|
| `command: [argv, …]` | stdio — spawns the server per call | `--allow-command-targets` (same risk class as `type: command`) |
| `url: http://…` | streamable HTTP — remote server | none (like `type: http`) |

```yaml
target:
  type: mcp
  command: ["python3", "my_mcp_server.py"]     # stdio — local server
  # url: "http://localhost:8000/mcp"           # remote alternative
  kwargs:
    tool: transaction_search             # REQUIRED — tool called per case
    args_template: {query: "{input}"}        # arg mapping (below)
    timeout_seconds: 30                      # connect+call bound → TargetError
    env: {MY_SERVER_KEY: "…"}                # stdio: env for the subprocess
```

## 2. `args_template` — map case fields into tool args

Each value is rendered per case; `{input}`, `{session_id}`, `{case_id}`
substitute, anything else passes through literally:

```yaml
args_template:
  query: "{input}"                    # case.input → the query arg
  session: "{session_id}"             # conversation session → a session arg
  limit: 5                            # non-string values pass through as-is
```

Default (no `args_template`): `{"input": "{input}"}` — matches servers
that take a single `input` arg.

## 3. Write the dataset — `expected_tools` pins the call

```yaml
cases:
  - id: transaction-search
    input: "october transaction history"
    expected_tools:
      - {name: transaction_search, args: {query: "october transaction history"}}
    tags: [mcp, banking, tools]
```

`tool_sequence` checks the call happened; `tool_args_correct` compares
`args ⊆ actual args` — a mismatched arg name or value is caught, not
just "a tool ran".

## 4. Score it — agentic + response metrics apply unchanged

```yaml
metrics:
  - name: tool_sequence                # search actually invoked
  - name: tool_args_correct            # with the right query
  - name: forbidden_tools
    tools: [delete_everything]         # MCP tools that must never run
  - name: termination_check            # clean finish_reason
  - name: contains
    substrings: ["TX-78231"]              # result text reached output
```

The adapter emits `thought → tool_call → message` per call plus a
`ToolCall` record — so `loop_detection`, `confirmation_required`,
`trajectory_match`, and the judge metrics work exactly as with any other
agent target. `isError` results land in `ToolCall.error` (a tool failure,
not an infra `error`).

## 5. Run the demo

`suites/mcp/mcp_suite.yaml` is fully self-contained — the MCP server is an
inline `python -c` `MCPServer` exposing `transaction_search` and `delete_everything`:

```bash
uv sync --extra mcp
uv run aitestkit run suites/mcp/mcp_suite.yaml \
    --allow-command-targets -o evidence/mcp
```

```
PASS transaction-search  tool_sequence 1.0 · tool_args 1.0 · termination done
PASS fee-schedule        same — all 5 metrics 1.0
```

## 6. Validate the results

Per case in `results.json`:

```json
{"actual_output": "transactions for 'overdraft fee schedule': TX-78231, TX-78232",
 "trajectory": {"events": [
   {"type": "thought",   "payload": {"text": "mcp: transaction_search({…})"}},
   {"type": "tool_call", "payload": {"name": "transaction_search",
     "args": {"query": "overdraft fee schedule"},
     "result": "transactions for '…': TX-78231, TX-78232"}},
   {"type": "message"}],
  "finish_reason": "done"}}
```

```bash
# which MCP tools were actually invoked, with what args?
jq '.results[].trajectory.events[]? | select(.type=="tool_call")
    | .payload | {name, args}' evidence/mcp/results.json

# tool errors surfaced as isError
jq '.results[] | select(.trajectory.events[]? |
    select(.type=="tool_call") | .payload.error != null) | .case_id' \
    evidence/mcp/results.json
```

`status: error` = connect/timeout/protocol failure (`reason` names it);
tool-level `isError` stays a scored result with `ToolCall.error` set.

## MCP SDK version notes

The adapter uses the **official MCP Python SDK** (`pip install 'aitestkit[mcp]'`,
spec `mcp>=1.9`). Client side: `ClientSession` + `stdio_client` /
`streamable_http_client`. The SDK renamed APIs between 1.x and 2.x — the
adapter handles both names for the HTTP client automatically.

Writing your own stdio server for tests? The import differs:

```python
# mcp>=2:                       # mcp<2:
from mcp.server.mcpserver import MCPServer   from mcp.server.fastmcp import FastMCP
srv = MCPServer("demo")                      srv = FastMCP("demo")

@srv.tool()                                  @srv.tool()
def txn_search(q: str) -> str:             def txn_search(q: str) -> str:
    ...                                          ...
srv.run()                                    srv.run()   # stdio by default
```

## Limitations

- **Fresh session per call** — the server is spawned/connected per case;
  MCP server-side state does not persist across cases or conversation
  turns (a `session_id` can still be passed as a tool arg for your own
  state keying)
- **One tool per suite** — `kwargs.tool` is fixed; use multiple suites
  (or per-case `assert:` + separate datasets) to exercise several tools
- **stdio spawns a subprocess** — needs `--allow-command-targets` and a
  reachable interpreter/SDK in the server environment

## Config reference

| Key | Purpose |
|---|---|
| `command` | stdio server argv — spawns subprocess, gated |
| `url` | streamable-HTTP endpoint — remote server |
| `kwargs.tool` | tool name called per case (required) |
| `kwargs.args_template` | `{input}`/`{session_id}`/`{case_id}` mapping |
| `kwargs.timeout_seconds` | connect+call bound (default 60) |
| `kwargs.env` | env vars for the stdio subprocess |

## Related

- `docs/agent-testing.md` — trajectory metrics that score MCP tool calls
- `docs/custom-adapters.md` — when `type: mcp` doesn't fit (SSE, auth
  flows, multi-call protocols) — write a `TargetAdapter` instead
