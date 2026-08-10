import time

from utils.latency_monitor import LatencyTracker, init_latency_tracker
from utils.latency_monitor import ModuleType


def test_latency_tracker_measure_and_stats():
    tracker = LatencyTracker(enable_alerts=False)
    with tracker.measure(ModuleType.KALMAN):
        time.sleep(0.0001)
    stats = tracker.get_statistics()
    assert 'kalman' in stats
    m = stats['kalman']
    assert 'mean' in m

    # test global init
    t2 = init_latency_tracker(enable_alerts=False)
    t2.measure(ModuleType.PREDICTION)
