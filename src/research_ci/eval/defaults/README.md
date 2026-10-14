# Default method proposals

The files in this package list the Engine's recommended starting methods for
each stage. A proposal supplies method names only; the Lab or stage runner
still supplies columns, targets, horizons, and any other context. This prevents
a generic library method from guessing case semantics.

Defaults are recommendations rather than permissions. The stage policy and
compiled menu remain authoritative, and an external agent may select a
different available method when it records a reason.

Feat proposals cover correctness, coverage, feature structure, redundancy,
feature-pair mutual information, and target-free stability. Sig proposals
cover explicit target association, feature-target mutual information,
out-of-sample behavior, and signal stability. Every current default is valid
under its stage's static menu; declaration requirements may narrow that menu
further for one case.
