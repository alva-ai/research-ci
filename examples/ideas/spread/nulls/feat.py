from research_ci.api import Null, PermuteWithin

NULLS = [
    Null(
        "tail",
        PermuteWithin("time", clusters="symbol", block="horizon"),
        evals=("rank ic without top symbol",),
        reps=60,
    ),
]
