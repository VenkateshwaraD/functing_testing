# AGENTS.md — conventions for AI coding agents

## What this repo is

`aitestkit` — a Python AI-testing framework (LLM output, prompt, RAG, agentic,
memory, reliability testing). Promptfoo-style workspace layout at repo root.

## Layout

```
src/aitestkit/        framework code (the package)
suites/               tracked suite YAMLs, grouped by pillar — agent/, ab/,
                      api/, banking/, llm/, mcp/, memory/, prompt/, rag/,
                      reliability/, schema/ (datasets resolve ../../dataset_testcases/)
dataset_testcases/    tracked test-case datasets (.yaml/.json/.jsonl)
prompts/              tracked prompt .txt files (version by filename)
docs/                 standalone per-module guides
tests/                tracked — unit tests (run before pushing)
examples/             gitignored — bundled demo apps, golden + meta-eval sets (local only)
evidence/             run output — tracked (committed evidence)
```

Suite paths are relative to the suite file: `../dataset_testcases/`,
`../prompts/`.

## Commands

```bash
# run a suite (repo root)
aitestkit run suites/prompt/prompt_testing_suite.yaml
# or: uv run aitestkit run ... / python -m aitestkit.cli run ...

# demo suites import bundled apps — they need the path:
PYTHONPATH=examples aitestkit run suites/banking/banking_suite.yaml

# lint / typecheck (must stay clean)
uv run ruff check src/
uv run mypy src/

# local unit tests (not committed)
uv run pytest tests/
```

## Provider / judge configuration

Precedence: **CLI flag > suite YAML > env var > shipped default**.

Judge metrics (`requires_judge = True` on the metric class) require an
**explicit** judge — set `AITESTKIT_JUDGE_PROVIDER` + `AITESTKIT_JUDGE_MODEL`
(non-empty), a `judge:` block, or `--judge`/`--judge-model`; the run errors
otherwise. Judge provider+model must differ from every `models:` SUT entry
(same pair = self-grading → ConfigError).

```bash
aitestkit run X.yaml --judge openai --judge-model gpt-4o-mini
export AITESTKIT_JUDGE_PROVIDER=anthropic
export AITESTKIT_JUDGE_MODEL=claude-haiku-4-5
```

`.env` at repo root is auto-loaded (cwd-upward search); shell env wins.
Provider keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.
Fully offline runs: `AITESTKIT_JUDGE_PROVIDER=fake`.
Env vars also: `AITESTKIT_EMBED_PROVIDER`, `AITESTKIT_EMBED_MODEL`,
`AITESTKIT_MAX_COST_USD`.

## Conventions

- **Independence**: suites run standalone; optional deps (deepeval, ragas,
  litellm...) are lazy-imported and must fail as scoped metric errors, never
  break unrelated suites. Keep it that way.
- **Custom integrations** via `module:Class`: providers `impl:`, targets
  `type: custom`, metrics via `@register`.
- **Security boundary**: `--no-allow-python-targets` rejects every
  code-loading key — `memory.store`, `tools.*.function`, `providers.*.impl`,
  `module:Class` judge/embedder names — not just `target.type`. Command/mcp
  subprocesses get a minimal env (PATH/HOME/`LC_*`/`AITESTKIT_*` + explicit
  `env:`); provider keys never reach a child implicitly. `${VAR}` in
  headers/URLs/`env:` fails loudly when unset.
- Prompt variants accept `file:` (relative to suite) or inline `template:`.
- `ToolCall` args key is `args:` (not `arguments:` — silently ignored).
- Per-case `assert:` in datasets overrides suite `metrics:` for that case —
  bare names reuse suite config, `{name, threshold, ...params}` dicts build
  fresh; unknown names fail at load.
- `expect: fail` on a case marks it a detection probe: scored-metric
  failure → `xfailed` (counts as judged+passing in `min_pass_rate`);
  all-pass → `failed` with `unexpected_pass` detail. Never use
  `threshold: 0.0` to hide failures — it's rejected at load unless the
  metric sets `informational: true` (non-gating, excluded from verdict).
- `required: true` on a metric means a skip is an authoring bug — it
  reports `error`, never a silent pass-by-omission.
- Gates `max_error_cases`, `max_flaky`, `max_skip_rate` are mandatory on
  every suite. `min_pass_rate` judges only
  passed/failed/xfailed — zero judged cases fails the gate.
  A metric crash marks the case `error`, not `skipped`.
- Live calls: `--cassette FILE --cassette-mode record` needs the real
  provider key (loud `config error` without it); `--cassette FILE`
  replays offline. Sanitized cassettes live in `cassettes/` (tracked).
- Don't commit `examples/`, `.env`, `results` —
  gitignored on purpose. `evidence/` IS committed — it's the run-evidence
  trail reviewers look at.
- Exit codes: `0` gates pass, `1` metric/gate failure, `2` config error.
- Commit style: conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
