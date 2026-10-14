# Default method proposals

The files in this package will select the Engine's general methods for each
stage as their input contracts become explicit. A proposal supplies method
names only; the Lab or stage runner still supplies columns, targets, horizons,
and any other context. This prevents a generic library method from guessing
case semantics.
