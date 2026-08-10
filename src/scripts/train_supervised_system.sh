#!/bin/bash

# ============================================================================
# Full Supervised Trading Pipeline: Analyze Top Trades + Train Models
# ============================================================================
# This script executes the complete workflow:
# 1. Identifies 10 most profitable trades per day from 6 months of history
# 2. Generates supervised training data
# 3. Trains action classifier and profit regressor
# 4. Generates comprehensive performance report

set -e

SYMBOL="${1:-^NSEI}"
LOOKBACK_DAYS="${2:-180}"
TOP_TRADES="${3:-10}"
OUTPUT_DIR="${4:-results/supervised_trading}"

echo ""
echo "============================================================================"
echo "🎯 SUPERVISED TRADING SYSTEM - FULL PIPELINE"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Symbol: $SYMBOL"
echo "  Lookback: $LOOKBACK_DAYS days (~6 months)"
echo "  Top trades per day: $TOP_TRADES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

./.venv/bin/python -m pipelines.supervised_trading_orchestrator \
  --symbol "$SYMBOL" \
  --days "$LOOKBACK_DAYS" \
  --top-trades "$TOP_TRADES" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "✅ FULL PIPELINE COMPLETE!"
echo "============================================================================"
echo ""
echo "Output location: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  1. Trade analysis:"
echo "     - $OUTPUT_DIR/01_trade_analysis/supervised_training_data.csv"
echo "     - $OUTPUT_DIR/01_trade_analysis/top_trades_per_day.csv"
echo "     - $OUTPUT_DIR/01_trade_analysis/trade_analysis_summary.json"
echo ""
echo "  2. Trained models:"
echo "     - $OUTPUT_DIR/02_supervised_models/supervised_models.joblib"
echo "     - $OUTPUT_DIR/02_supervised_models/training_metrics.json"
echo ""
echo "  3. Performance report:"
echo "     - $OUTPUT_DIR/03_reports/supervised_training_report.json"
echo ""
echo "💡 Next steps:"
echo "  • Review the report: cat $OUTPUT_DIR/03_reports/supervised_training_report.json"
echo "  • Analyze top trades: cat $OUTPUT_DIR/01_trade_analysis/top_trades_per_day.csv | head -20"
echo "  • Check model metrics: cat $OUTPUT_DIR/02_supervised_models/training_metrics.json"
echo ""
