#!/usr/bin/env bash
# The whole flow of the example idea from a fresh ledger: the contract, the data, and three feat submissions (noise
# fails, a peek at the future is tainted, the spread passes). Approvals here are automatic; a person relays them in
# normal use.
set -euo pipefail
cd "$(dirname "$0")"
export RCI_HOME="$PWD/.rci-home"
IDEA=ideas/spread
rm -rf "$RCI_HOME" "$IDEA/.rci"
uv sync -q
step() { echo "$ rci $*"; uv run rci "$@" || true; echo; }
step init "$IDEA"
step contract derive -i "$IDEA"
step approve -i "$IDEA" --stage contract --auto
step data check -i "$IDEA"
step approve -i "$IDEA" --stage data --auto
step feat submit -i "$IDEA" "$IDEA/research/noise.py"
step feat submit -i "$IDEA" "$IDEA/research/peek.py"
step feat submit -i "$IDEA" "$IDEA/research/spread.py"
step approve -i "$IDEA" --stage feat --auto
step status -i "$IDEA"
