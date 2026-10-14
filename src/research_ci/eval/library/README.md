# General methodology library

The library is grouped by the question a method answers. `DataSet`, target-free
`FeatureSet`, and `SignalSet` declare the available table structure. The
external case agent reads the compiled stage menu, then uses
`LibrarySelection` to choose methods and supply columns, keys, windows,
horizons, thresholds, and a reason. The Engine validates the choice against
the stage policy and declaration before expanding it. It does not run every
method or choose the case's methodology.

| family | question | examples |
| --- | --- | --- |
| `quality` | Can the declared columns be used? | required columns, finite values, coverage, dispersion |
| `structure` | How are features related to one another? | correlation, VIF, PCA, effective rank, autocorrelation |
| `association` | How does a feature relate to an explicit target? | Pearson IC, rank IC, ICIR, cumulative IC, bucket spread |
| `information` | Is there nonlinear dependence? | feature-pair MI in Feat, feature-target MI in Sig |
| `stability` | Does the evidence persist across ordered data? | split association, rank persistence, distribution shift |

The four slots remain separate:

1. `baseline` checks table and stage correctness.
2. `library` runs the reusable diagnostics selected by the external agent.
3. `case` states the research case's definitions and assumptions.
4. `case_eval` supplies fixtures and counterexamples for those definitions.

Slot 2 accepts only methods registered in this library. A case-specific
expression or custom runner belongs in slot 3, even when its implementation is
small enough to resemble a general check. This keeps the menu auditable and
prevents an agent from bypassing the stage boundary with a raw `EvalSpec`.

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

`SignalSet` owns the complete feature universe, candidate and final signal
columns, target, horizon, targetless hypothesis, and fit-policy declaration.
`EvalContext` carries the named label sidecar at execution time. Selecting a
target-dependent method requires
an explicit horizon-to-label mapping, and an absent target keeps that method unavailable
instead of inventing an empty evaluation. `FeatureSet` has no target field, so
Feat cannot accidentally absorb Sig research.
