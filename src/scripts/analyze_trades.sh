#!/bin/bash

# ============================================================================
# Trade Analysis & Supervised Data Generation Only
# ============================================================================
# This script runs only the trade analysis phase:
# 1. Analyzes 6 months of historical data
# 2. Identifies 10 most profitable trades per day
# 3. Generates supervised training data in CSV format

set -e

SYMBOL="${1:-^NSEI}"
LOOKBACK_DAYS="${2:-180}"
TOP_TRADES="${3:-10}"
OUTPUT_DIR="${4:-results/supervised_data}"

echo ""
echo "============================================================================"
echo "📊 TRADE ANALYSIS - GENERATE SUPERVISED DATA"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Symbol: $SYMBOL"
echo "  Lookback: $LOOKBACK_DAYS days"
echo "  Top trades per day: $TOP_TRADES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

./.venv/bin/python -m analysis.trade_analyzer \
  --symbol "$SYMBOL" \
  --days "$LOOKBACK_DAYS" \
  --top-trades "$TOP_TRADES" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "✅ TRADE ANALYSIS COMPLETE!"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  CSV data: $OUTPUT_DIR/supervised_training_data.csv"
echo "  Top trades: $OUTPUT_DIR/top_trades_per_day.csv"
echo "  Summary: $OUTPUT_DIR/trade_analysis_summary.json"
echo ""
echo "📈 Statistics:"
#jq '.' "$OUTPUT_DIR/trade_analysis_summary.json" 2>/dev/null || cat "$OUTPUT_DIR/trade_analysis_summary.json"
echo ""
