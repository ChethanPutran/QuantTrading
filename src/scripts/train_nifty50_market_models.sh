#!/bin/bash

# Train all market-only models on 7 months of NIFTY50 option-chain data
# and simulate the last 1 month. Saves artifacts for later reuse.

set -e

SYMBOL="${1:-^NSEI}"
TOTAL_DAYS="${2:-210}"
SIMULATE_DAYS="${3:-30}"
ARTIFACT_DIR="${4:-results/nifty50_market_only_artifacts}"
OUTPUT_CSV="${5:-results/nifty50_last_month_simulation.csv}"
SUMMARY_JSON="${6:-results/nifty50_last_month_summary.json}"

echo "Training NIFTY50 market-only models..."
echo "  Symbol: $SYMBOL"
echo "  Total history days: $TOTAL_DAYS (~7 months)"
echo "  Simulation window: $SIMULATE_DAYS days"
echo "  Artifact directory: $ARTIFACT_DIR"
echo "  Output CSV: $OUTPUT_CSV"
echo "  Summary JSON: $SUMMARY_JSON"
echo ""

./.venv/bin/python pipelines/train_market_models.py \
  --mode train-simulate \
  --symbol "$SYMBOL" \
  --total-days "$TOTAL_DAYS" \
  --simulate-days "$SIMULATE_DAYS" \
  --artifact-dir "$ARTIFACT_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --summary-json "$SUMMARY_JSON"

echo ""
echo "✅ Training and simulation complete!"
echo "Simulation results saved to: $OUTPUT_CSV"
echo "Summary saved to: $SUMMARY_JSON"
echo ""
echo "To simulate again with the same trained models, run:"
echo "  bash simulate_nifty50_last_month.sh $SYMBOL $TOTAL_DAYS $SIMULATE_DAYS $ARTIFACT_DIR results/nifty50_last_month_simulation_rerun.csv results/nifty50_last_month_summary_rerun.json"
