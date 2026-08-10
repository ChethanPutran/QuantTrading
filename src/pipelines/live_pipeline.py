from ..learning import BaseTrainer
from ..execution import BaseExecutionEngine, BaseOrderManager
from ..features import BaseFeaturePipeline
from ..models import BaseModel, BaseRegimeModel, BaseTransitionModel
from ..control import BaseConstraints, BaseController
from ..memory import BasePatternDB, BasePatternEncoder

class LivePipeline:
    """
    Orchestrates the real-time flow.
    Uses dependency injection (SOLID).
    """

    def __init__(
        self,
        feature_pipeline : BaseFeaturePipeline,
        regime_model : BaseRegimeModel,
        transition_model : BaseTransitionModel,
        pattern_db : BasePatternDB,
        encoder : BasePatternEncoder,
        model : BaseModel,
        controller : BaseController,
        execution_engine : BaseExecutionEngine,
        trainer: BaseTrainer,
    ):
        self.feature_pipeline = feature_pipeline
        self.regime_model = regime_model
        self.transition_model = transition_model
        self.pattern_db = pattern_db
        self.encoder = encoder
        self.model = model
        self.controller = controller
        self.execution_engine = execution_engine
        self.trainer = trainer

    def step(self, data: dict)-> None:
        """
        Executes one step of the pipeline:
        1. Extract features
        2. Update regime and transition models
        3. Query pattern DB
        4. Make prediction
        5. Decide action
        6. Execute action
        7. Update trainer


        Args: 
        ----
        data: dict
            Raw market data for the current step.
        
        Returns:
        -------
            None    
        """

        # 1. Extract features
        features = self.feature_pipeline.transform(data)

        # 2. Update regime and transition models
        regime_probs = self.regime_model.update(features)
        regime_state = self.transition_model.update(regime_probs)

        # 3. Query pattern DB
        key = self.encoder.encode(features, regime_state)
        node = self.pattern_db.get(key)

        # 4. Make prediction
        prediction = self.model.predict(features)

        # 5. Decide action
        action = self.controller.decide(features)

        # 6. Execute action
        result = self.execution_engine.step(action)

        # 7. Update trainer
        self.trainer.update({
            "features": features,
            "prediction": prediction,
            "result": result,
            "pattern": key,
        })


        

"""
This LSTM model takes the last 60 days of closing prices to predict the next day's closing price.
"""
### Step 2: Set Up MPC to Optimize Buy/Sell/Hold
"""

We will now set up an **MPC controller** that uses the
 **LSTM predictions** for stock prices to decide on **buy/sell/hold** actions.
   The MPC will optimize the future reward, which is based on the predicted stock price.

"""
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_stock_model.h5')

# Function to predict the next stock price using the LSTM model
def predict_stock_price(data):
    prediction = model.predict(data)
    return prediction[0][0]
