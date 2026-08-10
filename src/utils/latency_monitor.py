"""
Latency monitoring and profiling utilities.
Tracks module-level latencies against budget.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from enum import Enum


class ModuleType(Enum):
    """Module types for latency tracking."""
    KALMAN = "kalman"
    FEATURES = "features"
    GMM = "gmm"
    HMM = "hmm"
    MEMORY = "memory"
    PREDICTION = "prediction"
    MPC = "mpc"
    EXECUTION = "execution"
    LEARNING = "learning"
    LOGGING = "logging"


# Latency budgets (in milliseconds) from latency_budget.md
LATENCY_BUDGETS = {
    ModuleType.KALMAN: 2,
    ModuleType.FEATURES: 5,
    ModuleType.GMM: 8,
    ModuleType.HMM: 5,
    ModuleType.MEMORY: 4,
    ModuleType.PREDICTION: 2,
    ModuleType.MPC: 25,
    ModuleType.EXECUTION: 5,
    ModuleType.LEARNING: 5,
    ModuleType.LOGGING: 3
}

TOTAL_BUDGET = 100  # milliseconds per tick


class LatencyTracker:
    """
    Tracks module-level latencies and alerts on violations.
    """
    
    def __init__(self, enable_alerts: bool = True):
        """
        Initialize latency tracker.
        
        Args:
            enable_alerts: Whether to log latency violations
        """
        self.enable_alerts = enable_alerts
        self.measurements: Dict[ModuleType, List[float]] = {
            module: [] for module in ModuleType
        }
        self.total_latencies: List[float] = []
        self.violations: Dict[ModuleType, int] = {
            module: 0 for module in ModuleType
        }
    
    @contextmanager
    def measure(self, module: ModuleType):
        """
        Context manager for measuring module latency.
        
        Usage:
            with tracker.measure(ModuleType.GMM):
                gmm.update(x)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            self.measurements[module].append(elapsed_ms)
            
            # Check against budget
            budget = LATENCY_BUDGETS[module]
            if elapsed_ms > budget:
                self.violations[module] += 1
                if self.enable_alerts:
                    print(f"⚠️  {module.value} EXCEEDED: {elapsed_ms:.2f}ms > {budget}ms budget")
    
    def update_tick_latency(self, elapsed_ms: float) -> None:
        """Record total tick latency."""
        self.total_latencies.append(elapsed_ms)
        
        if elapsed_ms > TOTAL_BUDGET * 0.9:  # 90% of budget
            if self.enable_alerts:
                print(f"⚠️  TICK LATENCY HIGH: {elapsed_ms:.2f}ms > 90% of {TOTAL_BUDGET}ms budget")
    
    def get_statistics(self, module: Optional[ModuleType] = None) -> Dict[str, Any]:
        """
        Get latency statistics.
        
        Args:
            module: Specific module (None for all)
        
        Returns:
            Dictionary of statistics
        """
        if module is None:
            # All modules
            stats = {}
            for m in ModuleType:
                stats[m.value] = self._module_stats(m)
            
            # Add total
            if self.total_latencies:
                total_array = np.array(self.total_latencies)
                stats['total'] = {
                    'mean': float(np.mean(total_array)),
                    'std': float(np.std(total_array)),
                    'p50': float(np.percentile(total_array, 50)),
                    'p95': float(np.percentile(total_array, 95)),
                    'p99': float(np.percentile(total_array, 99)),
                    'max': float(np.max(total_array)),
                    'min': float(np.min(total_array)),
                    'budget': TOTAL_BUDGET
                }
            
            return stats
        else:
            return self._module_stats(module)
    
    def _module_stats(self, module: ModuleType) -> Dict[str, Any]:
        """Get statistics for a specific module."""
        measurements = self.measurements[module]
        
        if not measurements:
            return {
                'n_calls': 0,
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'budget': LATENCY_BUDGETS[module],
                'violations': 0
            }
        
        array = np.array(measurements)
        budget = LATENCY_BUDGETS[module]
        
        return {
            'n_calls': len(measurements),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'p50': float(np.percentile(array, 50)),
            'p95': float(np.percentile(array, 95)),
            'p99': float(np.percentile(array, 99)),
            'max': float(np.max(array)),
            'min': float(np.min(array)),
            'budget': budget,
            'budget_exceeded': self.violations[module],
            'violation_ratio': self.violations[module] / len(measurements)
        }
    
    def print_report(self) -> None:
        """Print formatted latency report."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print(f"{'MODULE':<20} {'MEAN':<10} {'P95':<10} {'P99':<10} {'BUDGET':<10} {'VIOLATIONS':<10}")
        print("="*70)
        
        for module in ModuleType:
            s = stats[module.value]
            print(
                f"{module.value:<20} "
                f"{s.get('mean', 0.0):>8.2f}ms {s.get('p95', 0.0):>8.2f}ms {s.get('p99', 0.0):>8.2f}ms "
                f"{s.get('budget', 0):>8.0f}ms {int(s.get('budget_exceeded', s.get('violations', 0))):>10d}"
            )
        
        print("="*70)
        
        if 'total' in stats:
            s = stats['total']
            print(f"{'TOTAL':<20} {s.get('mean', 0.0):>8.2f}ms {s.get('p95', 0.0):>8.2f}ms "
                f"{s.get('p99', 0.0):>8.2f}ms {s.get('budget', 0):>8.0f}ms")
        
        print("="*70 + "\n")
    
    def reset(self) -> None:
        """Reset all measurements."""
        for module in ModuleType:
            self.measurements[module] = []
            self.violations[module] = 0
        self.total_latencies = []
    
    def get_violations_summary(self) -> Dict[str, int]:
        """Get summary of violations."""
        return {
            module.value: count
            for module, count in self.violations.items()
            if count > 0
        }


# Global tracker instance
_tracker: Optional[LatencyTracker] = None


def init_latency_tracker(enable_alerts: bool = True) -> LatencyTracker:
    """Initialize global latency tracker."""
    global _tracker
    _tracker = LatencyTracker(enable_alerts=enable_alerts)
    return _tracker


def get_latency_tracker() -> LatencyTracker:
    """Get global latency tracker."""
    global _tracker
    if _tracker is None:
        _tracker = LatencyTracker()
    return _tracker
