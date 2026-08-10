#!/bin/bash

PROJECT_NAME="trading_system"

echo "Creating UV project..."

# Create project using uv
uv init $PROJECT_NAME
cd $PROJECT_NAME || exit

# Create folders
mkdir -p config data features models memory control execution learning evaluation utils pipelines tests

# Create sub-files
touch config/config.yaml config/logging.yaml

touch data/loaders.py data/stream.py data/preprocessing.py

touch features/kalman.py features/indicators.py features/feature_pipeline.py

touch models/gmm.py models/hmm.py models/linear_model.py models/base_model.py

touch memory/pattern_db.py memory/pattern_node.py memory/encoder.py

touch control/mpc.py control/constraints.py

touch execution/simulator.py execution/order_manager.py execution/cost_model.py

touch learning/trainer.py learning/hidden_state.py learning/branching.py

touch evaluation/metrics.py evaluation/backtest.py evaluation/reports.py

touch utils/logger.py utils/helpers.py utils/math_utils.py

touch pipelines/live_pipeline.py pipelines/backtest_pipeline.py

touch tests/test_gmm.py tests/test_kalman.py tests/test_pipeline.py

# Ensure main entry exists
touch main.py

# Add dependencies
echo "Adding dependencies..."
uv add numpy pandas scipy scikit-learn

# (Optional but recommended later)
# uv add matplotlib seaborn
# uv add rich loguru
# uv add pytest

echo "UV project setup complete!"
echo "Run your project with:"
echo "  uv run python main.py"