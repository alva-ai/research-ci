# Default method proposals

The files in this package will select the Engine's general methods for each
stage as their input contracts become explicit. A proposal supplies method
names only; the Lab or stage runner still supplies columns, targets, horizons,
and any other context. This prevents a generic library method from guessing
case semantics.

Feat proposals cover correctness, coverage, feature structure, redundancy,
and target-free stability. Sig proposals cover target-aware feature selection,
signal construction evidence, out-of-sample behavior, and null evidence. The
current proposal files predate the final `SignalSet` boundary; narrowing the
Feat list and expanding the Sig list is an implementation task tracked in the
project TODO.
