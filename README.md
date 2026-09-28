# Alpha CI

A backtest framework for research: an alpha claim counts only after it passes CI, the same way code does. The kernel does three things and nothing else: time, holdout data and labels, and the verdict.

It is a library plus the `aci` CLI. Any Python 3.11+ environment that installs it can host research ideas; plugins live in `alpha-ci-lab`.

```sh
uv sync
uv run aci --help
uv run pytest
uv run ruff check
uv run lint-imports
```
