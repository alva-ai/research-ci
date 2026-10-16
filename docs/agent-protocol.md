# Agent protocol

This document is the operating contract for an external research Agent. It is
deliberately more concrete than a prompt: the Engine owns the state machine,
while the Agent owns exploration, construction, and the explanation of the
result.

## The boundary

An Agent may create or update files in its Lab and may run the Engine commands
that `next --json` authorizes. It must not edit `workflow.json`, `candidate.json`,
`approved.json`, or a published stage directory by hand. Every stage produces a
numbered attempt. An invalid, rejected, or stale attempt remains audit history;
the next attempt gets a new number.

The order is fixed:

```text
Contract -> Data -> Feat -> Sig -> Algo
```

The reviewer sees a Card only after the Agent has run the stage, prepared a
review brief, and submitted a recommendation. A human can use the terminal
yes/no handshake; an external review Agent supplies the explicit `--by agent`,
`--decision`, and `--card` arguments. JSON output is the automation interface,
not a bypass around review.

## What the Agent does at each stage

### Contract

Write the smallest useful mission, scope, stage list, and invariants. Keep
labels, features, signals, and outcomes out of the Contract unless the mission
requires their existence. Derive the Card, inspect it, and wait for approval.

### Data

Explore freely inside the Lab, including downloading additional sources when a
missing field blocks the question. Preserve native-frequency tables and a
point-in-time aligned wide `main.parquet`. Record source, frequency, event time,
availability time, freshness, proxy status, gaps, and every treatment. Data
must not contain engineered features, labels, signals, positions, or PnL.

Populate all four eval slots:

1. **baseline** — Engine floor checks, including finite values, `NaN`/`NAN`,
   nulls, duplicate keys, time ordering, and point-in-time leakage.
2. **library** — compatible general Data methods selected from the Data menu.
3. **case** — source- and question-specific checks written in the Lab.
4. **case_eval** — valid fixtures must pass and invalid/counterexample fixtures
   must fail.

If the result is insufficient, write that conclusion and run another attempt or
return to the source exploration. Do not hide a gap with a silent fill.

### Feat

Read only the approved Data handoff. Build the complete target-free feature
universe with explicit lineage: rolling/lagged transforms, normalization,
regimes, cross-sectional transforms, and warm-up policy. Use the stage kit for
name-free Polars expressions when useful. If a raw field is missing, go back to
Data rather than inventing a proxy.

Use the Feat menu to evaluate coverage, dispersion, redundancy, correlation,
VIF, PCA concentration, mutual information between features, stability, and
drift where the data supports them. Every feature receives a disposition and a
reason. Target-aware IC, labels, and signal selection belong to Sig.

### Sig

Declare the complete approved Feat universe, labels, horizons, chronological
splits, and the selection rule. Screen every feature for every declared target;
then compare candidate recipes such as a rule, weighted combination, matrix,
cluster, or model. Keep candidate scores separate from the final wide signal
handoff. A confirmation/holdout, when declared, is consumed only after the
development recipe is frozen.

The Sig menu may include Pearson/Spearman IC, Rank IC, ICIR, cumulative IC,
bucket spread, target mutual information, lead/lag, rolling stability, split
stability, bootstrap, PCA/cluster context, and model diagnostics. These are
evidence for a decision, not automatic proof of causality. Preserve labels,
feature dispositions, lineage, candidate comparison, counterevidence, and plots.

### Algo

Consume only the approved Sig handoff. Declare how a score becomes a weight,
position, order, or trade. Include lag, fees, slippage, funding, leverage,
turnover, liquidity, drawdown, and exposure assumptions. Run the case-owned
backtester and preserve both a wide position/trade table and a timeseries
backtest result. Algo may be rejected even when Sig is attractive if execution
or risk evidence is insufficient.

## The loop

After every run, follow the machine-readable next action:

```sh
uv run rci next --out .rci/runs/<run> --json
```

The legal progression is `run -> prepare -> submit -> review`. A reviewer can
approve and continue, approve and stop, or reject with feedback. A rejection
requires a new attempt. An upstream `revise` stales and locks downstream
attempts, so the Agent must rebuild the affected handoff and then revisit each
downstream stage. The Agent may iterate as many times as the evidence requires;
the reviewer should only decide from the Card and its linked evidence.

Before preparing a Card, check the stage completion list printed by:

```sh
uv run rci guide --stage <data|feat|sig|algo>
```

The review brief should answer five questions in plain language:

- What question did this attempt answer?
- What evidence supports the conclusion?
- What independent evidence or counterexample weakens it?
- Which files and plots let a reviewer reproduce the claim?
- What changed from the previous attempt, and what should happen next?

`evidence_refs` and `key_plots` are relative paths inside the attempt. The
Engine checks that they exist before it lets the Card enter review.

## What the reviewer decides

The reviewer does not inspect implementation details to rescue a weak result.
The Card must make the result, evidence, limitations, and next action clear.

- **continue**: the stage is sufficient and the next stage may start;
- **stop**: the research is a valid no-go or terminal result;
- **reject**: the Agent must address concrete feedback and create a new attempt.

An Agent reviewer is an explicit role with the same choices. A terminal human
review defaults to no and binds the approval to the Card ID displayed on screen.
