import asyncio
import numpy as np

from smart.trading_system.comm.event_bus import init_event_bus
from features.feature_pipeline import FeaturePipeline
from pipelines.async_pipeline import FeatureEngine, RegimeEngine, PredictionEngine
from regime.gmm import GaussianMixtureRegimeModel
from regime.hmm import MarkovTransitionModel


class DummyModel:
    def predict(self, x):
        return 0.01


def test_feature_regime_prediction_handlers():
    bus = init_event_bus()
    pipeline = FeaturePipeline(feature_dim=4)
    gmm = GaussianMixtureRegimeModel(n_regimes=2, feature_dim=4, learning_rate=0.1)
    hmm = MarkovTransitionModel(n_regimes=2, smoothing=0.5)

    feat_engine = FeatureEngine(bus, pipeline)
    regime_engine = RegimeEngine(bus, gmm, hmm)
    pred_engine = PredictionEngine(bus, DummyModel())

    # Simulate producing a feature event by calling engine directly
    async def run():
        # create a tick -> feature processing
        feat = pipeline.update(100.0)
        filtered_price, velocity = pipeline.get_state()
        from smart.trading_system.comm.events import FeatureEvent
        fe = FeatureEvent(features=feat, filtered_price=filtered_price, velocity=velocity, timestamp=0.0)
        # Call regime handler directly
        await regime_engine.handle_features(fe)
        # create a dummy pattern event to drive prediction handler
        from memory.pattern_db import PatternDB
        from memory.encoder import PatternEncoder
        encoder = PatternEncoder(feature_bin_size=0.1, regime_dim=2)
        key = encoder.encode(feat, np.array([0.5,0.5]))
        pdb = PatternDB()
        node = pdb.get(key)
        from smart.trading_system.comm.events import PatternEvent
        pe = PatternEvent(pattern_key=key, hidden_state=node.hidden_state, node_stats=node.get_stats(), timestamp=0.0)
        # store last features/regime in prediction engine and call handler
        await pred_engine.store_features(fe)
        # create regime event
        from smart.trading_system.comm.events import RegimeEvent
        re = RegimeEvent(gmm_probs=np.array([0.5,0.5]), hmm_probs=np.array([0.5,0.5]), regime_id=0, timestamp=0.0)
        await pred_engine.store_regime(re)
        await pred_engine.handle_pattern(pe)

    asyncio.run(run())
