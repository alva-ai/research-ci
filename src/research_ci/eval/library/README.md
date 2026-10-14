# General methodology library

The library is grouped by the question a method answers. `DataSet` and
`FeatureSet` declare the available table structure. The external case agent
then uses `LibrarySelection` to choose methods and supply columns, keys,
windows, horizons, thresholds, and a reason. The Engine validates and expands
that selection; it does not run every method and does not choose the case's
methodology.

| family | question | examples |
| --- | --- | --- |
| `quality` | Can the declared columns be used? | required columns, finite values, coverage, dispersion |
| `structure` | How are features related to one another? | correlation, VIF, PCA, effective rank, autocorrelation |
| `association` | How does a feature relate to an explicit target? | Pearson IC, rank IC, ICIR, cumulative IC, bucket spread |
| `information` | Is there nonlinear dependence? | k-NN mutual information |
| `stability` | Does the evidence persist across ordered data? | split association, rank persistence, distribution shift |

The four slots remain separate:

1. `baseline` checks table and stage correctness.
2. `library` runs the reusable diagnostics selected by the external agent.
3. `case` states the research case's definitions and assumptions.
4. `case_eval` supplies fixtures and counterexamples for those definitions.

A metric with `passed` status only means that the calculation completed. A
plan can pass an explicit threshold and still require target, out-of-sample,
null-world, or cost evidence later. Thresholds belong in the external plan so
the case agent can explain why a threshold applies.

Feat normally selects target-free methods: correctness, coverage, dispersion,
feature-feature correlation and mutual information, VIF, PCA, persistence, and
drift. Sig owns recipe-specific feature screening and final signal evaluation
against an explicit target and horizon. That includes IC, rank IC, ICIR,
cumulative IC, bucket spread, feature-target mutual information, horizon
decay, out-of-sample stability, and null or permutation evidence.

The current API exposes target sidecars through `FeatureSet.targets` and
`EvalContext`. This is a transitional low-level mechanism. The planned
`SignalSet` will own the public target, horizon, targetless-hypothesis, and
signal-recipe declaration. Until it lands, selecting a target-dependent method
still requires an explicit target mapping; an empty declaration is a plan
error rather than an invented or empty evaluation.
