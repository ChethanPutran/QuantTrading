"""
Kalman Filter for signal filtering and state estimation.
Removes noise from price stream and estimates velocity.
"""

from dataclasses import dataclass

import numpy as np
from typing import Tuple, Optional

from pydantic_settings import BaseSettings


class KalmanFilter:
    """
    1D Kalman filter for price tracking.
    
    State: [price, velocity]
    Observation: price
    
    Theory:
    -------
    State: x_t = [p_t, v_t]^T where p_t is price, v_t is velocity
    Dynamics: x_t = A @ x_{t-1}  (constant velocity model)
    Observation: y_t = H @ x_t + noise
    
    Algorithm:
    1. Predict: x_pred = A @ x
    2. Update: x = x_pred + K @ (y - H @ x_pred)
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-2,
        initial_price: float = 0.0,
        dt: float = 1.0
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise variance (Q)
            measurement_noise: Measurement noise variance (R)
            initial_price: Initial price estimate
            dt: Time step (for velocity scaling)
        """
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.dt = dt
        
        # State: [price, velocity]
        self.x = np.array([initial_price, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 0.1
        
        # State transition matrix (constant velocity)
        self.A = np.array([
            [1.0, dt],    # price_t = price_{t-1} + velocity * dt
            [0.0, 1.0]    # velocity_t = velocity_{t-1}
        ])
        
        # Observation matrix (we only measure price)
        self.H = np.array([[1.0, 0.0]])
        
        # Innovation (for diagnostics)
        self.innovation = 0.0
        self.innovation_covariance = self.R
    
    def predict(self) -> np.ndarray:
        """
        Predict step: project state forward.
        
        Returns:
            Predicted state [price, velocity]
        """
        # Predict state
        self.x_pred = self.A @ self.x
        
        # Predict covariance
        self.P_pred = self.A @ self.P @ self.A.T
        self.P_pred[0, 0] += self.Q  # Add process noise to price dimension
        self.P_pred[1, 1] += self.Q * 0.1  # Less process noise on velocity
        
        return self.x_pred
    
    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update step: incorporate new measurement.
        
        Args:
            measurement: Observed price
        
        Returns:
            Tuple of (filtered_price, estimated_velocity)
        """
        # Predict
        self.predict()
        
        # Innovation (measurement residual)
        self.innovation = measurement - (self.H @ self.x_pred)[0]
        
        # Innovation covariance
        self.innovation_covariance = (self.H @ self.P_pred @ self.H.T)[0, 0] + self.R
        
        # Kalman gain
        K = (self.P_pred @ self.H.T) / self.innovation_covariance
        
        # Update state
        self.x = self.x_pred + K.flatten() * self.innovation
        
        # Update covariance
        self.P = (np.eye(2) - K * self.H) @ self.P_pred
        
        return float(self.x[0]), float(self.x[1])
    
    def get_state(self) -> Tuple[float, float]:
        """
        Get current state.
        
        Returns:
            Tuple of (price, velocity)
        """
        return float(self.x[0]), float(self.x[1])
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()
    
    def reset(self, initial_price: float = 0.0) -> None:
        """Reset filter to initial state."""
        self.x = np.array([initial_price, 0.0])
        self.P = np.eye(2) * 0.1
        self.innovation = 0.0


class KalmanFilter1DConfig(BaseSettings):
    process_noise: float = 1e-5
    measurement_noise: float = 1e-2
    initial_price: float = 0.0
    dt: float = 1.0

@dataclass
class KalmanFilter1D:
    config: KalmanFilter1DConfig = None

    def __init__(self, config: KalmanFilter1DConfig = None):
        self.config = config or KalmanFilter1DConfig()
        self.measurement_variance = self.config.measurement_noise
        self.estimate = self.config.initial_price
        self.error_variance = 1.0

    def update(self, measurement: float) -> tuple[float, float]:
        self.error_variance += self.config.process_noise
        kalman_gain = self.error_variance / (self.error_variance + self.measurement_variance)
        residual = measurement - self.estimate
        self.estimate += kalman_gain * residual
        self.error_variance *= 1.0 - kalman_gain
        velocity = kalman_gain * residual
        return self.estimate, velocity

    def reset(self, estimate: float = 0.0) -> None:
        self.estimate = float(estimate)
        self.error_variance = 1.0
