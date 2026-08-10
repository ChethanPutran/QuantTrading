#!/bin/bash

# ============================================================================
# Train Supervised Models from Existing Data
# ============================================================================
# This script trains supervised models from pre-generated supervised data:
# 1. Loads supervised training data (CSV)
# 2. Trains action classifier (BUY/SELL/HOLD)
# 3. Trains profit regressor
# 4. Saves models and metrics

set -e

SUPERVISED_DATA="${1:-results/supervised_data/supervised_training_data.csv}"
SYMBOL="${2:-^NSEI}"
OUTPUT_DIR="${3:-results/supervised_models}"

if [ ! -f "$SUPERVISED_DATA" ]; then
    echo "❌ Error: Supervised data file not found: $SUPERVISED_DATA"
    echo ""
    echo "Generate supervised data first by running:"
    echo "  bash analyze_trades.sh"
    exit 1
fi

echo ""
echo "============================================================================"
echo "🤖 TRAIN SUPERVISED MODELS"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Symbol: $SYMBOL"
echo "  Supervised data: $SUPERVISED_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo ""

./.venv/bin/python -m pipelines.train_supervised_models \
  --symbol "$SYMBOL" \
  --supervised-data "$SUPERVISED_DATA" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "✅ MODEL TRAINING COMPLETE!"
echo "============================================================================"
echo ""
echo "Output files:"
echo "  Models: $OUTPUT_DIR/supervised_models.joblib"
echo "  Metrics: $OUTPUT_DIR/training_metrics.json"
echo ""
echo "📊 Model Performance:"
cat "$OUTPUT_DIR/training_metrics.json" | python -m json.tool 2>/dev/null | head -30
echo ""
