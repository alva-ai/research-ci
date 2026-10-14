# Evaluation layout

`protocol.py` defines the four fixed slots. `runner.py` executes an external
plan against a read-only Polars table and writes sidecar records. `baseline.py`
contains the Engine floor that runs for every stage.

The methods in `library/` are reusable general methodology. They are grouped
by what they measure, not by stage. Their metadata declares which stages and
input shapes are meaningful. A stage may select a method through a default
catalog or an external Lab plan; an inapplicable method returns
`not_applicable` with a reason.

Case-specific rules remain in the Lab's `evals/<stage>.py` files.
