#!/bin/bash

# Load pre-trained NIFTY50 market models and simulate the last month.
# Assumes artifacts already exist in ARTIFACT_DIR (from train_nifty50_market_models.sh).

set -e

SYMBOL="${1:-^NSEI}"
TOTAL_DAYS="${2:-210}"
SIMULATE_DAYS="${3:-30}"
ARTIFACT_DIR="${4:-results/nifty50_market_only_artifacts}"
OUTPUT_CSV="${5:-results/nifty50_last_month_simulation_loaded.csv}"
SUMMARY_JSON="${6:-results/nifty50_last_month_summary_loaded.json}"

if [ ! -f "$ARTIFACT_DIR/models.joblib" ]; then
    echo "❌ Error: Artifacts not found at $ARTIFACT_DIR"
    echo "Run 'bash train_nifty50_market_models.sh' first to train models."
    exit 1
fi

echo "Loading pre-trained NIFTY50 models and simulating..."
echo "  Symbol: $SYMBOL"
echo "  Total history days: $TOTAL_DAYS"
echo "  Simulation window: $SIMULATE_DAYS days"
echo "  Artifact directory: $ARTIFACT_DIR"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Summary JSON: $SUMMARY_JSON"
echo ""

./.venv/bin/python pipelines/train_market_models.py \
  --mode simulate-only \
  --symbol "$SYMBOL" \
  --total-days "$TOTAL_DAYS" \
  --simulate-days "$SIMULATE_DAYS" \
  --artifact-dir "$ARTIFACT_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --summary-json "$SUMMARY_JSON"

echo ""
echo "✅ Simulation complete!"
echo "Simulation results saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_JSON"
