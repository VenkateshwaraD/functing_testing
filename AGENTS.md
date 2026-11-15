# AGENTS.md — conventions for AI coding agents

## What this repo is

`aitestkit` — a Python AI-testing framework (LLM output, prompt, RAG, agentic,
memory, reliability testing). Promptfoo-style workspace layout at repo root.

## Layout

```
src/aitestkit/        framework code (the package)
suites/               tracked suite YAMLs — target + datasets + metrics + gates
dataset_testcases/    tracked test-case YAMLs
prompts/              tracked prompt .txt files (version by filename)
docs/                 standalone per-module guides
tests/                gitignored — local unit tests only
examples/             gitignored — bundled demo apps + corpus (local only)
scratch/              gitignored — run output
```

Suite paths are relative to the suite file: `../dataset_testcases/`,
`../prompts/`.

## Commands

```bash
# run a suite (repo root)
aitestkit run suites/prompt_testing_suite.yaml
# or: uv run aitestkit run ... / python -m aitestkit.cli run ...

# demo suites import bundled apps — they need the path:
PYTHONPATH=examples aitestkit run suites/banking_suite.yaml

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
- Prompt variants accept `file:` (relative to suite) or inline `template:`.
- `ToolCall` args key is `args:` (not `arguments:` — silently ignored).
- Per-case `assert:` in datasets overrides suite `metrics:` for that case —
  bare names reuse suite config, `{name, threshold, ...params}` dicts build
  fresh; unknown names fail at load.
- Don't commit `tests/`, `examples/`, `scratch/`, `.env`, `results` —
  gitignored on purpose.
- Exit codes: `0` gates pass, `1` metric/gate failure, `2` config error.
- Commit style: conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
