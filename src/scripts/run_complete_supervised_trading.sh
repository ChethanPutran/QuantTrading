#!/bin/bash

# ============================================================================
# MASTER SCRIPT: Complete Supervised Trading System
# ============================================================================
# Complete end-to-end workflow:
# 1. Analyze 6 months of trades to identify top 10 profitable trades per day
# 2. Generate supervised training data
# 3. Train supervised models (action classifier + profit regressor)
# 4. Simulate trading using trained models
# 5. Generate comprehensive performance report

set -e

SYMBOL="${1:-^NSEI}"
LOOKBACK_DAYS="${2:-180}"
SIMULATE_DAYS="${3:-30}"
TOP_TRADES="${4:-10}"
OUTPUT_DIR="${5:-results/complete_supervised_trading}"

echo ""
echo "============================================================================"
echo "🎯 COMPLETE SUPERVISED TRADING SYSTEM - MASTER PIPELINE"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Symbol: $SYMBOL"
echo "  Analysis period: $LOOKBACK_DAYS days (~6 months)"
echo "  Simulation period: $SIMULATE_DAYS days (~1 month)"
echo "  Top trades per day: $TOP_TRADES"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Steps:"
echo "  1️⃣  Analyze historical trades (identify top performers)"
echo "  2️⃣  Train supervised models (action classifier + profit regressor)"
echo "  3️⃣  Simulate trading using trained models"
echo "  4️⃣  Generate performance report"
echo ""

./.venv/bin/python -m pipelines.complete_supervised_trading_system \
  --symbol "$SYMBOL" \
  --lookback-days "$LOOKBACK_DAYS" \
  --simulate-days "$SIMULATE_DAYS" \
  --top-trades "$TOP_TRADES" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "✅ COMPLETE PIPELINE FINISHED!"
echo "============================================================================"
echo ""
echo "📂 Output files organized in: $OUTPUT_DIR/"
echo ""
echo "📊 Report locations:"
echo "   • Analysis: $OUTPUT_DIR/01_trade_analysis/"
echo "   • Models: $OUTPUT_DIR/02_supervised_models/"
echo "   • Simulation: $OUTPUT_DIR/03_simulation_results/"
echo "   • Report: $OUTPUT_DIR/04_reports/"
echo ""
echo "💡 Quick commands:"
echo "   View report: cat $OUTPUT_DIR/04_reports/complete_trading_report.json | python -m json.tool"
echo "   View trades: cat $OUTPUT_DIR/01_trade_analysis/top_trades_per_day.csv | head -20"
echo "   View simulation: cat $OUTPUT_DIR/03_simulation_results/supervised_trading_simulation.csv | head -20"
echo ""
