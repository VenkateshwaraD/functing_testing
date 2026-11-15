# CI/CD

This project uses **GitLab CI/CD**, configured in [`.gitlab-ci.yml`](../.gitlab-ci.yml)
at the repo root. It replaces the previous GitHub Actions workflows.

> **Note:** unit tests (`tests/`) are intentionally *not* committed to this
> repository, so there is no `pytest` job in the pipeline. Run tests locally
> before pushing:
>
> ```bash
> uv sync --frozen
> uv run coverage run -m pytest tests/ -m "not live"
> uv run coverage report --fail-under=80
> ```

## Architecture

The pipeline has four stages. Jobs in the same stage run in parallel; a stage
only starts after the previous stage succeeds.

```
push / MR / schedule / tag
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  check                                                      │
│  └─ check  (matrix: Python 3.13 / 3.14 / 3.15)              │
│       ruff → mypy → offline example suites                  │
├─────────────────────────────────────────────────────────────┤
│  audit                                                      │
│  └─ audit  (pip-audit — schedule, lockfile change, manual)  │
├─────────────────────────────────────────────────────────────┤
│  live                                                       │
│  └─ live-smoke  (scheduled / manual, needs API keys)        │
├─────────────────────────────────────────────────────────────┤
│  release         (tag pipelines only: v*)                   │
│  ├─ build        uv build + twine check                     │
│  ├─ publish-pypi twine upload to PyPI  (manual)             │
│  └─ release      GitLab release for the tag                 │
└─────────────────────────────────────────────────────────────┘
```

All jobs run in the `ghcr.io/astral-sh/uv:debian` image and use `uv` for
Python installs and dependency sync (`uv sync --frozen` against `uv.lock`).

## Jobs in detail

### `check` — lint, typecheck, example suites

Runs on every push and merge request. Fanned out across a matrix of
`PY = 3.13 / 3.14 / 3.15` via `parallel:matrix`, `fail-fast` off by default.

Steps, in order:

1. `uv python install $PY` — fetch the matrix Python.
2. `uv sync --frozen --python $PY` — install deps from `uv.lock`.
3. `uv run ruff check src examples` — lint (rule set in `pyproject.toml`).
4. `uv run mypy` — typecheck `src/aitestkit` (config in `pyproject.toml`).
5. Runs every suite in `suites/*/*.yaml` through
   `uv run aitestkit run <suite>` with `PYTHONPATH=.` set from `examples/`.
   These suites are replay-only — they use fake providers and cassettes, so
   they need **no secrets** and work on forks.

Artifacts: `example-out/` reports are uploaded on every run
(`when: always`, 1-week expiry). The uv cache (`.uv-cache`) is cached per
Python version.

### `audit` — dependency vulnerability scan

Runs `uvx pip-audit --strict` against the synced environment. Triggers:

- Any **scheduled** pipeline,
- Pushes to the default branch **only when** `uv.lock` or
  `pyproject.toml` changed (`rules:changes`),
- Otherwise available as a **manual** job.

### `live-smoke` — live-provider smoke suite

Replacement for the old `nightly-live` workflow. Runs the spend-capped suite
[`ci/live_suite.yaml`](../ci/live_suite.yaml) (`max_cost_usd: 0.50`, hard
abort on breach) against the real OpenAI API.

Triggers: **scheduled pipelines** and **manual web runs** only — never on
normal pushes, so it can't burn API budget on every commit.

The job checks `OPENAI_API_KEY` first and exits 0 ("skipping") when the
variable isn't set — safe on forks/mirrors without secrets.

Artifacts: `live-out/` report directory, uploaded always.

### `build` / `publish-pypi` / `release` — tagged releases

These only exist in **tag pipelines** matching `v*`:

- **`build`** — `uv build` then `uvx twine check dist/*`; uploads `dist/`
  as an artifact.
- **`publish-pypi`** — *manual* job; uploads `dist/` to PyPI with twine
  using `PYPI_TOKEN` (see configuration below).
- **`release`** — creates a GitLab Release for the tag via `release-cli`.

## Configuration

### CI/CD variables

Set under **Settings → CI/CD → Variables** (mask the secrets):

| Variable             | Used by        | Required | Notes                                |
|----------------------|----------------|----------|--------------------------------------|
| `OPENAI_API_KEY`     | `live-smoke`   | For live | Masked; job skips cleanly if absent. |
| `ANTHROPIC_API_KEY`  | `live-smoke`   | Optional | Masked.                              |
| `PYPI_TOKEN`         | `publish-pypi` | For tags | Masked + protected; PyPI API token.  |

### Schedules

**Build → Pipeline schedules → New schedule**:

- **Live smoke:** e.g. `17 3 * * *` (daily ~03:17 UTC) on the default
  branch — equivalent to the old `nightly-live` cron.
- **Audit:** e.g. `42 6 * * 1` (weekly, Monday ~06:42 UTC) — equivalent to
  the old weekly `audit` cron. Any scheduled pipeline also runs `audit`
  automatically via its first rule.

### Protected tags

Restrict `v*` tags (**Settings → Repository → Protected tags**) so only
maintainers can cut releases — this also gates `publish-pypi`.

## How to run

| What                      | How                                                        |
|---------------------------|------------------------------------------------------------|
| Lint/typecheck/examples   | Push a commit or open an MR — `check` runs automatically.  |
| Dependency audit          | Push a change to `uv.lock`/`pyproject.toml` on main, run a schedule, or trigger the manual `audit` job in the pipeline view. |
| Live smoke                | Create a pipeline schedule, or **Build → Pipelines → Run pipeline** then play the manual `live-smoke` job. |
| Release                   | `git tag v0.2.0 && git push origin v0.2.0` — `build` runs automatically; play `publish-pypi` manually to upload to PyPI; `release` creates the GitLab release. |

## Validating results

Suite runs in CI are gated by exit code — `0` passes the job, anything
non-zero fails it (`1` case failures, `2` config, `3` budget, `4`
guardrail, `5` gate). To diagnose a failure:

- **`example-out/` / `live-out/` artifacts** — uploaded on every run
  (`when: always`, 1-week expiry). Open `aitestkit-report.html` for
  per-case metric detail, or `results.json` for raw verdicts
  (`metrics[].reason` + `details` carry the evidence).
- **`junit.xml`** — consumed by GitLab's test report UI on MRs.
- **Job log** — the console report prints `PASS`/`FAIL` per case plus the
  gate verdicts (`gate FAIL min_pass_rate: 0.8 (< 0.9)`).

A `2` exit means the suite never ran — check for missing CI variables or
bad YAML before chasing metric scores.

## Local equivalents

Everything the pipeline does can be reproduced locally:

```bash
uv sync --frozen

# check
uv run ruff check src examples
uv run mypy
cd examples && PYTHONPATH=. sh -c \
  'for s in suites/*/*.yaml; do uv run aitestkit run "$s" -o /tmp/out || exit $?; done'

# audit
uvx pip-audit --strict

# live smoke (needs real keys in env)
cd examples && PYTHONPATH=. uv run aitestkit run ../ci/live_suite.yaml \
    -o /tmp/live-out --report console,json

# release build
uv build && uvx twine check dist/*
```
