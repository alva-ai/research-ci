# Example: one idea, end to end

A small lab and one idea in it, run through the contract, data and feat layers from a fresh ledger in about 40 seconds. It
shows every slot of every layer on a synthetic market, and the three ways a submission ends: FAILED, TAINTED and PASSED.

```bash
cd examples
./run.sh
```

`run.sh` needs [uv](https://docs.astral.sh/uv/). It installs this lab with the engine from the repository, keeps the
engine's home in `examples/.rci-home`, and runs every step below. Approvals here are automatic (`--auto`); in normal use
a person reads each card and approves it, and an agent relays the words with the card's id
(`rci approve --stage <stage> --relayed "<words>" --card <id>`).

## The lab

| Path | What it is |
| --- | --- |
| `pyproject.toml`, `src/toy_lab/` | The lab's plugins, registered as entry points: `toy.data`, two sources for three symbols on three venues over 25 days, `candles` with an hourly `close` and `funding` with a `funding_rate` and its `funding_interval_hours` every 8 hours; `toy.pairs`, every venue pair of a symbol, read from `wide` |
| `ideas/spread/contract.py` | The claim (cross-venue funding spreads persist), the decisive effect (the top third of pairs by score earns more carry than random), the alternative (`tail`: a few symbols carry the result), the outcome `ref.carry` at 8h and 24h, costs and the budget |
| `ideas/spread/data.py`, `evals/toy.py` | Slot 3 for data: the lab eval `closes_at_zero`, which reads the sources and flags cells of `candles`, with its h0, and its CASES (slot 4) next to it |
| `ideas/spread/feat.py`, `evals/toy.py` | Slot 3 for feat: the lab eval `funding_ic`, on the funding part of the outcome, `y_flow`, with its h0, and its CASES |
| `ideas/spread/nulls/feat.py` | The null world of the contract's alternative, `tail`, which can veto |
| `ideas/spread/research/` | Three features files, each `features(wide, candidates)`: `noise.py` (random scores), `peek.py` (reads each venue's average funding over every row of `wide`, so it looks ahead) and `spread.py` (the latest funding spread) |
| `ideas/spread/.rci/` | The ledger and the cards a run writes; not committed, since it records the absolute paths of the files it judged |

`rci init` also wrote `START.md`, `research/example.py` and `evals/example.py`: what a fresh idea starts from, refused until they are rewritten.

## The flow

1. `rci contract derive`: the contract's premises (point-in-time universe, an outcome that does not read past t + h) and the power
   at each horizon. DECIDABLE.
2. `rci data check`: the source's admission, the default checks, the lab eval `closes_at_zero` and the calibration of the
   decisive test. READY.
3. `rci feat submit research/noise.py`: random scores do not beat random. FAILED, slot 1.
4. `rci feat submit research/peek.py`: the scores change when the data is cut early, so they read the future. TAINTED.
5. `rci feat submit research/spread.py`: beats random at 8h and 24h with z above the bar of the third submission, and
   neither the lab eval nor the `tail` world vetoes. PASSED at 8h, 24h.

## Transcript

The transcript below was recorded before the sources, `wide` and the outcome replaced the field table and labels; `run.sh` prints the current cards.

```text
$ rci init ideas/spread
created ideas/spread; read ideas/spread/START.md first: it says how the work goes
then fill in ideas/spread/contract.py and run `rci contract derive -i ideas/spread`
lab evals start from ideas/spread/evals/example.py, features from ideas/spread/research/example.py
`rci next -i ideas/spread` says what to do after every step

$ rci contract derive -i ideas/spread
ContractCard · spread · v1
claim      cross-venue funding spreads persist
decisive   carry: beats random significantly · target ref.mean(cluster=symbol, top=0.34)
decisions  every 1h from execution · 335 decision times
data       toy.data afa06051cf36 · 3,780 rows · 9 insts · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 259.4 days old
universe   toy.pairs · median 9 candidates in 3 symbol clusters per decision · membership pit
costs      ref.flat_costs(commission_bps=1, legs=2, slippage_bps=1) · median round trip 8.0 bp
labels     carry: ref.carry · H 8h 24h
checks     [!]  data is current: latest data is 259.4 days old
           ok   universe point-in-time: 3 cut points give the same candidates
           ok   universe causal: values after 2026-01-08T04:00Z leave the candidates at it unchanged
           ok   label carry: worked examples: 2 ok
           ok   label carry: examples bite: shift, in-window changes and missing fields move it; later data does not
           ok   label carry: no data after t + h: 2 cut points, horizon 8h
           ok   label carry: no data after t + h: 2 cut points, horizon 24h
           ok   target reads the score: ref.mean moves with the scores
power      carry: the smallest gain over random the data can detect (MDE)
           H        decisions  blocks  MDE bp/entry  if rolled
           8h             321    41.9           5.9  ≈ 64.8%/yr
           24h            305    14.0          19.5  ≈ 71.1%/yr
alts       tail  (engine floor: chance, timing)
           each needs a null world in nulls/feat.py, or feat refuses a submission before it is counted; a world can veto
used       nothing declared
history    history can confirm
budget     3
verdict    DECIDABLE
           carry: noise measured at 8h, 24h; feat passes it where it beats random and no lab eval or null world rejects it
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
           [!] carry: symbol Z carries 55% at 24h of the noise the MDE rests on; rci data check looks at whether its data is fit for research
----------------------------------------------------------------------------------------------------
card 6d77654b
up for review; a person approves it with `rci approve --stage contract`

$ rci approve -i ideas/spread --stage contract --auto
    contract b418c3c098fe
contract v1 PASSED: freeze 94da1a669339 · approved by the agent (automatic research)

$ rci data check -i ideas/spread
DataCard · spread · v1
data       toy.data afa06051cf36 · 3,780 rows · 9 insts · 2026-01-01T01:00Z → 2026-01-15T00:00Z · 259.4 days old
1 floor · the source's admission, and what the engine weighs on every idea's data
source     toy.data · admission 04a8425fcb2b · not yet admitted: approving this card admits it for every idea [!]
           code 82522fb96764 · 9 insts · 6,750 rows
           ok   values are finite: none: values null, NaN or infinite
           ok   known no earlier than the event: none: rows known before their event time
           ok   no long gaps: none: gaps longer than 3× the usual spacing
           ok   no near-duplicate events: none: events within 60s of the last
           ok   no level shifts: none: lasting moves of 5× or more
           ok   replay as of past times: 5 cut points agree with the full fetch
           ok   canary: replay catches a hindsight deletion: deleting a:Y close from 2026-01-15 18:00 was caught
           ok   independent source agrees: 24 of 24 checked rows agree with the source (12 it could not check)
                reference: the toy venue
           ok   canary: reconciliation catches altered values: 3 of 3 altered rows caught
checks     ok   field grades: unverified fields are not read: funding_interval_hours
grades     verified: close funding_rate · unverified: funding_interval_hours
treatments none needed
power      H            MDE before   after
           carry 8h            5.9     5.9
           carry 24h          19.5    19.5
calibrate  ok   carry at 8h vs floor: stand-ins die 98% (needs ≥ 92%) · a signal planted at 1.5 MDE (3.8 bp/entry) lives 97% (needs ≥ 80%)
2 engine lists · the default checks and the library evals data.py lists; what they flag is treated by default
checks     default  [!]  data is current: latest data is 259.4 days old
           default  ok   level sanity: no declared level at zero or below, stale, or spiking
           default  ok   identity breaks: no declared level comes back 5x away after a long gap
           default  ok   coverage: every inst covers 24h
           default  ok   hedged legs are one asset: every candidate row holds (close within 1.2x and returns correlated ≥ 0.3)
           default  [!]  influence concentration: more than 50% of the noise from one symbol: Z 55% at carry 24h; shown, not treated
clusters   share of the noise that goes away without each symbol, largest first
           carry 8h    X 0% · Y 0% · Z 0%
           carry 24h   Z 55% · X 0% · Y 0%
3 lab evals · tests of this idea's data: what each flags is treated as its on_fail says; with on_fail=block any row it flags in the data after the treatments before it vetoes the data
lab evals  ok   closes_at_zero (evals/toy.py) · H0: every close is above zero
                on_fail mask: valid 1/1 · flagged 0 rows
4 tests of lab evals · their CASES and PROPERTIES before the data was fetched, then the pits each covers
tests      ok   closes_at_zero (evals/toy.py): valid 1/1
decide     nothing
verdict    READY
           approving freezes the treated data; feat reads only that
           approving also admits toy.data for every idea
warnings   [!] the data source has nothing newer than 2026-01-26T00:00Z; collect live data first
----------------------------------------------------------------------------------------------------
card e2643979
up for review; a person approves it with `rci approve --stage data`

$ rci approve -i ideas/spread --stage data --auto
    contract        b418c3c098fe
    data_check      fe8f9119d507
    engine          6b8f272a9de3
    engine_packages 4cd20008b60b
    exclusions      47c61605ee6a
    panel           b0a41d046f24
    python          CPython 3.13.13
data v1 PASSED: freeze 47170e1f1ddc · approved by the agent (automatic research)

$ rci feat submit -i ideas/spread ideas/spread/research/noise.py
ResultCard · spread · feat v1 · submission 1 of 3
features   ideas/spread/research/noise.py · 719e506e5111
scores     2,961 rows · 100% of candidate rows · 9de768645c89
cost       full call 0.4 ms, peak memory +0.2 MB (process 244 MB) · on cut data 0.1 ms, 0.2 ms, 0.2 ms · not judged
1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim
premises   floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.00 on submission 1
decisive   carry: gain over random 0.4 bp/entry ± 1.1 at 8h (≈ 4.6%/yr if rolled) · z 0.2119 < 2.00 · 50 shuffles · FAILED
             8h     gain       0.4 ± 1.1       MDE      1.8    41 blocks  z 0.6682   vs the family's null 0.2119 ✗
             24h    gain      -0.4 ± 3.2       MDE      2.7    13 blocks  z -0.3733  vs the family's null -0.9936 ✗
2 engine lists · the default diagnostics and the library evals feat.py lists, vs the floor world; shown only
shown      [!]  rank ic (default) vs floor (chance, timing): z 0.2856 < 2.00 at 8h · best of 2 horizons · 50 shuffles
                8h     value    0.01931 ± 0.02301    null  0.0005803 sd 0.02252      41 blocks  z 0.8318
                24h    value    0.01631 ± 0.02284    null   0.003891 sd 0.01752      13 blocks  z 0.7087
           [!]  rank ic without top symbol (default) vs floor (chance, timing): z 0.9262 < 2.00 at 24h · best of 2 horizons · 50 shuffles
                8h     value    0.01272 ± 0.02767    null  -0.002568 sd 0.0284       41 blocks  z 0.5382
                24h    value    0.03459 ± 0.02657    null    0.00207 sd 0.02405      13 blocks  z 1.352
           [!]  top minus bottom fifth (default) vs floor (chance, timing): z 0.6421 < 2.00 at 8h · best of 2 horizons · 50 shuffles
                8h     value  0.0002145 ± 0.0002237  null -7.473e-06 sd 0.0002028    41 blocks  z 1.094
                24h    value  6.908e-05 ± 0.0003167  null  2.166e-05 sd 0.0002648    13 blocks  z 0.1791
3 lab evals · tests of the claim: each lab eval, and each null world, must reach z ≥ 2.00 at every horizon it covers, or the claim fails there
lab evals  ✗ funding_ic (evals/toy.py) · H0: the scores do not rank the funding part of the carry
               vs floor (chance, timing): z 0.8329 < 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value    0.01885 ± 0.01678    null   0.002411 sd 0.01757      41 blocks  MDE 0.04993    z 0.9356 ✗
                  24h    value    0.02063 ± 0.01141    null  -0.001713 sd 0.01798      13 blocks  MDE 0.05109    z 1.243 ✗
worlds     ✗ null 1 (tail: a few symbols carry the result) judges rank ic without top symbol (default)
               vs null 1 (tail): z 0.9262 < 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value    0.01272 ± 0.02767    null  -0.002568 sd 0.0284       41 blocks  MDE 0.0807     z 0.5382 ✗
                  24h    value    0.03459 ± 0.02657    null    0.00207 sd 0.02405      13 blocks  MDE 0.06835    z 1.352 ✗
           ok   alternatives covered: chance: floor · timing: floor · tail: null 1
4 tests of lab evals · run before the submission was counted
tests      ok   funding_ic (evals/toy.py): valid 1/1
verdict    FAILED
           carry: does not beat random, z 0.2119 < 2.00 (decisive test, slot 1): gain over random 0.417 bp per entry ± 1.06 at 8h
----------------------------------------------------------------------------------------------------
feat OPEN: not up for review; `rci next` says what to do and how many submissions are left

$ rci feat submit -i ideas/spread ideas/spread/research/peek.py
ResultCard · spread · feat v1 · submission 2 of 3
features   ideas/spread/research/peek.py · 2d50ba5b2d97
scores     2,961 rows · 100% of candidate rows · 5d7436c3d3f8
cost       full call 2.1 ms, peak memory +0.4 MB (process 244 MB) · on cut data 1.6 ms, 1.4 ms, 1.5 ms · not judged
1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim
premises   floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    FAIL scores read nothing ahead: 2026-01-04T18:00Z: 747 values of score differ; 2026-01-08T04:00Z: 1485 values of score differ; 2026-01-11T14:00Z: 2223 values of score differ
need       z ≥ 2.60 on submission 2
4 tests of lab evals · run before the submission was counted
tests      ok   funding_ic (evals/toy.py): valid 1/1
verdict    TAINTED
           the scores read data from after their decision time (scores read nothing ahead, floor): 2026-01-04T18:00Z: 747 values of score differ; 2026-01-08T04:00Z: 1485 values of score differ; 2026-01-11T14:00Z: 2223 values of score differ; compute each score from rows with available_at <= t
----------------------------------------------------------------------------------------------------
feat OPEN: not up for review; `rci next` says what to do and how many submissions are left

$ rci feat submit -i ideas/spread ideas/spread/research/spread.py
ResultCard · spread · feat v1 · submission 3 of 3
features   ideas/spread/research/spread.py · a5a0d9d57f06
scores     2,961 rows · 100% of candidate rows · 8af96e36d3cc
cost       full call 0.3 ms, peak memory +0.0 MB (process 243 MB) · on cut data 0.2 ms, 0.2 ms, 0.4 ms · not judged
1 floor · the premises and the decisive test, fixed before any result; only the decisive test can pass a claim
premises   floor    ok   scores rank candidates: scores 100% of candidate rows; ranks candidates at 100% of decision times
           floor    ok   scores read nothing ahead: the same scores with the data cut at 3 decision times
need       z ≥ 2.90 on submission 3
decisive   carry: gain over random 5.4 bp/entry ± 2.0 at 8h (≈ 59.2%/yr if rolled) · z 4.5 ≥ 2.90 · 150 shuffles · PASSED at 8h, 24h
             8h     gain       5.4 ± 2.0       MDE      4.7    41 blocks  z 4.316    vs the family's null 4.5 ✓
             24h    gain      17.7 ± 6.4       MDE     17.8    13 blocks  z 3.723    vs the family's null 3.802 ✓
2 engine lists · the default diagnostics and the library evals feat.py lists, vs the floor world; shown only
shown      ok   rank ic (default) vs floor (chance, timing): z 4.755 at 8h · best of 2 horizons · 50 shuffles
                8h     value     0.1699 ± 0.03826    null  -0.003232 sd 0.04105      41 blocks  z 4.217
                24h    value      0.269 ± 0.05959    null   -0.01264 sd 0.08481      13 blocks  z 3.321
           ok   rank ic without top symbol (default) vs floor (chance, timing): z 5.245 at 8h · best of 2 horizons · 50 shuffles
                8h     value     0.2087 ± 0.04598    null -0.0003659 sd 0.05008      41 blocks  z 4.175
                24h    value     0.3152 ± 0.07336    null   -0.01904 sd 0.09195      13 blocks  z 3.635
           ok   top minus bottom fifth (default) vs floor (chance, timing): z 3.231 at 8h · best of 2 horizons · 50 shuffles
                8h     value   0.001056 ± 0.0003151  null -3.357e-05 sd 0.0003375    41 blocks  z 3.228
                24h    value   0.002651 ± 0.0008472  null -0.0001715 sd 0.001116     13 blocks  z 2.528
3 lab evals · tests of the claim: each lab eval, and each null world, must reach z ≥ 2.00 at every horizon it covers, or the claim fails there
lab evals  ✓ funding_ic (evals/toy.py) · H0: the scores do not rank the funding part of the carry
               vs floor (chance, timing): z 18.07 ≥ 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value     0.8683 ± 0.01915    null  -0.005829 sd 0.05101      41 blocks  MDE 0.1449     z 17.14 ✓
                  24h    value     0.9041 ± 0.0169     null  -0.008726 sd 0.07559      13 blocks  MDE 0.2148     z 12.08 ✓
worlds     ✓ null 1 (tail: a few symbols carry the result) judges rank ic without top symbol (default)
               vs null 1 (tail): z 5.245 ≥ 2.00 · one vote for all its horizons · best of 2 horizons · 50 shuffles
                  8h     value     0.2087 ± 0.04598    null -0.0003659 sd 0.05008      41 blocks  MDE 0.1423     z 4.175 ✓
                  24h    value     0.3152 ± 0.07336    null   -0.01904 sd 0.09195      13 blocks  MDE 0.2613     z 3.635 ✓
           ok   alternatives covered: chance: floor · timing: floor · tail: null 1
4 tests of lab evals · run before the submission was counted
tests      ok   funding_ic (evals/toy.py): valid 1/1
verdict    PASSED at carry 8h, 24h
           carry: passes at 8h, 24h: the decisive test beats random there, z 4.5 ≥ 2.90 (slot 1), and every lab eval and null world that covers them reaches z ≥ 2.00 (slot 3)
----------------------------------------------------------------------------------------------------
card 695e37bf
up for review; a person approves it with `rci approve --stage feat`

$ rci approve -i ideas/spread --stage feat --auto
    calibration           fe8f9119d507
    contract              b418c3c098fe
    engine                6b8f272a9de3
    engine_packages       4cd20008b60b
    eval:evals/toy.py     410359887149
    feat_file             a057f301ead0
    feat_submission       6c2aee0a9013
    features              a5a0d9d57f06
    nulls_file            80cf6ded980f
    plugin:ref.carry      599e2f89646d
    plugin:ref.flat_costs 599e2f89646d
    plugin:ref.mean       599e2f89646d
    plugin:toy.pairs      82522fb96764
    python                CPython 3.13.13
    upstream:data         47170e1f1ddc
feat v1 PASSED: freeze c53ca1403df6 · approved by the agent (automatic research)

$ rci status -i ideas/spread
spread
  contract  PASSED  v1           approved by the agent (automatic research)
  data      PASSED  v1           approved by the agent (automatic research)
  feat      PASSED  v1   3/3     approved by the agent (automatic research)
  signal    OPEN    v1   0/3
  algo      OPEN    v1   0/3
  seal      OPEN    v1
  forward   OPEN    v1
  capital   OPEN    v1
  monitor   OPEN    v1
```
