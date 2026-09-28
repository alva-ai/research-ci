# Alpha CI

A backtest framework for research: an alpha claim counts only after it passes CI, the same way code does. The kernel does three things and nothing else: time, holdout data and labels, and the verdict.

```sh
uv sync
uv run aci --help
uv run pytest
uv run ruff check
uv run lint-imports
```
