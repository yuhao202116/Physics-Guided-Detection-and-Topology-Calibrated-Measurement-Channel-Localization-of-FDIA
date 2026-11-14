#!/bin/bash
# Run localization analysis for all cases sequentially
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/../.."

# Ensure MPLCONFIGDIR is writable
export MPLCONFIGDIR="/tmp/matplotlib-$USER"
mkdir -p "$MPLCONFIGDIR"

for CASE in case14 case30 case57; do
    echo ""
    echo "============================================================"
    echo "  Running $CASE"
    echo "============================================================"
    export PGAE_CASE="$CASE"
    python3 experiments/diagnostics/residual_localization.py "$CASE"
done

echo ""
echo "All cases complete."
