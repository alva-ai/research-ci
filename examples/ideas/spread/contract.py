from research_ci.api import Alternative, Contract, Effect, Execution, Label, Universe, Use

CONTRACT = Contract(
    claim="cross-venue funding spreads persist",
    decisive=(Effect(on="carry", target=Use("ref.mean", {"cluster": "symbol", "top": 0.34})),),
    alternatives=(Alternative("tail", "a few symbols carry the result"),),
    used=(),
    data="toy.data",
    costs=Use("ref.flat_costs", {"commission_bps": 1, "slippage_bps": 1, "legs": 2}),
    execution=Execution(every="1h"),
    universe=Universe("toy.pairs", clusters="symbol", groups=("venue_pair",)),
    labels=(Label("carry", Use("ref.carry", {"flow": "funding_rate"}), horizons=("8h", "24h")),),
    explore_end="2026-01-15T00:00:00Z",
    seal_end="2026-01-20T00:00:00Z",
    budget=3,
)
