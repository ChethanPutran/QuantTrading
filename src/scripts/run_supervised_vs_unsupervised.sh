#!/bin/bash

# ============================================================================
# Full Dual Training & Simulation: Supervised vs Unsupervised Comparison
# ============================================================================
# Runs both training pipelines and compares results:
# 1. Unsupervised training (market-only models - baseline)
# 2. Supervised training (from top historical trades)
# 3. Compare performance and generate comparison report

set -e

SYMBOL="${1:-^NSEI}"
LOOKBACK_DAYS="${2:-210}"  # 7 months for unsupervised
SIMULATE_DAYS="${3:-30}"    # 1 month simulation
TOP_TRADES="${4:-10}"
OUTPUT_DIR="${5:-results/supervised_vs_unsupervised}"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "🔄 SUPERVISED vs UNSUPERVISED COMPARISON"
echo "============================================================================"
echo ""
echo "This will run both training approaches and compare results:"
echo ""

# ========== PHASE 1: Run Unsupervised (Baseline) ==========
echo "👉 PHASE 1: Training Unsupervised Baseline Models"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

UNSUPERVISED_OUTPUT="$OUTPUT_DIR/unsupervised_baseline"
mkdir -p "$UNSUPERVISED_OUTPUT"

./.venv/bin/python -m pipelines.train_market_models \
  --mode train-simulate \
  --symbol "$SYMBOL" \
  --total-days "$LOOKBACK_DAYS" \
  --simulate-days "$SIMULATE_DAYS" \
  --artifact-dir "$UNSUPERVISED_OUTPUT/artifacts" \
  --output-csv "$UNSUPERVISED_OUTPUT/simulation.csv" \
  --summary-json "$UNSUPERVISED_OUTPUT/metrics.json"

UNSUPERVISED_PNL=$(cat "$UNSUPERVISED_OUTPUT/metrics.json" | python -c "import sys, json; print(json.load(sys.stdin)['final_pnl'])" || echo "0")
UNSUPERVISED_VALUE=$(cat "$UNSUPERVISED_OUTPUT/metrics.json" | python -c "import sys, json; print(json.load(sys.stdin)['final_portfolio_value'])" || echo "100000")

echo ""
echo "✓ Unsupervised PnL: \$$UNSUPERVISED_PNL"
echo ""

# ========== PHASE 2: Run Supervised (Proposed) ==========
echo "👉 PHASE 2: Training Supervised Models from Best Trades"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

SUPERVISED_OUTPUT="$OUTPUT_DIR/supervised_system"
mkdir -p "$SUPERVISED_OUTPUT"

./.venv/bin/python -m pipelines.complete_supervised_trading_system \
  --symbol "$SYMBOL" \
  --lookback-days "$LOOKBACK_DAYS" \
  --simulate-days "$SIMULATE_DAYS" \
  --top-trades "$TOP_TRADES" \
  --output-dir "$SUPERVISED_OUTPUT"

SUPERVISED_PNL=$(cat "$SUPERVISED_OUTPUT/03_simulation_results/supervised_trading_metrics.json" | python -c "import sys, json; print(json.load(sys.stdin)['final_pnl'])" || echo "0")
SUPERVISED_VALUE=$(cat "$SUPERVISED_OUTPUT/03_simulation_results/supervised_trading_metrics.json" | python -c "import sys, json; print(json.load(sys.stdin)['final_portfolio_value'])" || echo "100000")

echo ""
echo "✓ Supervised PnL: \$$SUPERVISED_PNL"
echo ""

# ========== PHASE 3: Generate Comparison Report ==========
echo "👉 PHASE 3: Generating Comparison Report"
echo "────────────────────────────────────────────────────────────────────────"
echo ""

COMPARISON_FILE="$OUTPUT_DIR/COMPARISON_REPORT.txt"

cat > "$COMPARISON_FILE" << EOF
================================================================================
SUPERVISED vs UNSUPERVISED TRADING MODELS - COMPARISON REPORT
================================================================================

Test Configuration:
  Symbol: $SYMBOL
  Analysis Period: $LOOKBACK_DAYS days
  Simulation Period: $SIMULATE_DAYS days
  Top Trades Analyzed: $TOP_TRADES per day
  Report Generated: $(date)

================================================================================
RESULTS SUMMARY
================================================================================

UNSUPERVISED BASELINE (Market-Only Models):
  Final Portfolio Value: \$$UNSUPERVISED_VALUE
  Profit/Loss: \$$UNSUPERVISED_PNL
  Return: $(python -c "print(f'{($UNSUPERVISED_PNL / 100000 * 100):.2f}%')" 2>/dev/null || echo "N/A")

SUPERVISED MODELS (From Top Trades Analysis):
  Final Portfolio Value: \$$SUPERVISED_VALUE
  Profit/Loss: \$$SUPERVISED_PNL
  Return: $(python -c "print(f'{($SUPERVISED_PNL / 100000 * 100):.2f}%')" 2>/dev/null || echo "N/A")

IMPROVEMENT:
  Absolute PnL Difference: \$$(python -c "print(f'{($SUPERVISED_PNL - $UNSUPERVISED_PNL):.2f}')" 2>/dev/null || echo "N/A")
  Relative Improvement: $(python -c "if $UNSUPERVISED_PNL != 0: print(f'{(($SUPERVISED_PNL - $UNSUPERVISED_PNL) / abs($UNSUPERVISED_PNL) * 100):.2f}%'); else: print('N/A')" 2>/dev/null || echo "N/A")

================================================================================
ANALYSIS
================================================================================

✓ Unsupervised approach: Market-only models (GMM, HMM, Linear, Decision Tree, etc.)
  - Learns general market patterns from price history
  - No specific trade-winning knowledge
  
✓ Supervised approach: Models trained on best historical trades
  - Learns what made top trades profitable
  - Feature-to-best-action mapping
  - Should identify high-probability trades

================================================================================
RECOMMENDATIONS
================================================================================

EOF

if (( $(echo "$SUPERVISED_PNL > $UNSUPERVISED_PNL" | bc -l) )); then
    echo "✓ SUPERVISED approach is MORE profitable" >> "$COMPARISON_FILE"
    echo "  Recommendation: Deploy supervised model for live trading" >> "$COMPARISON_FILE"
    echo "  Next steps:" >> "$COMPARISON_FILE"
    echo "    1. Validate on additional historical windows" >> "$COMPARISON_FILE"
    echo "    2. Fine-tune prediction thresholds" >> "$COMPARISON_FILE"
    echo "    3. Test on live data with small position sizes" >> "$COMPARISON_FILE"
else
    echo "✗ UNSUPERVISED approach is more profitable" >> "$COMPARISON_FILE"
    echo "  Recommendation: Investigate supervised model training" >> "$COMPARISON_FILE"
    echo "  Next steps:" >> "$COMPARISON_FILE"
    echo "    1. Analyze feature engineering quality" >> "$COMPARISON_FILE"
    echo "    2. Review trade analysis methodology" >> "$COMPARISON_FILE"
    echo "    3. Increase training data or adjust thresholds" >> "$COMPARISON_FILE"
fi

cat >> "$COMPARISON_FILE" << 'EOF'

================================================================================
OUTPUT DIRECTORY STRUCTURE
================================================================================

$OUTPUT_DIR/
├── unsupervised_baseline/
│   ├── artifacts/
│   ├── simulation.csv
│   └── metrics.json
├── supervised_system/
│   ├── 01_trade_analysis/
│   ├── 02_supervised_models/
│   ├── 03_simulation_results/
│   └── 04_reports/
└── COMPARISON_REPORT.txt (this file)

================================================================================
EOF

cat "$COMPARISON_FILE"

echo ""
echo "============================================================================"
echo "✅ COMPARISON COMPLETE!"
echo "============================================================================"
echo ""
echo "📊 Report saved: $COMPARISON_FILE"
echo ""
echo "📂 Full outputs in: $OUTPUT_DIR/"
echo ""
