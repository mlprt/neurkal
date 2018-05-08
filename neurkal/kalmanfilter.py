"""Conventional Kalman Filter implementation.
"""

import numpy as np


class KalmanFilter:
    """Estimate the state of a linear dynamical system with Gaussian noise."""

    def __init__(self, M, B, Z, sigma_0=None, estimate_0=None):
        """
        Args:
            M (np.ndarray): State transition matrix (`N*N`).
            B (np.ndarray): Control effect matrix (`N*1`).
            Z (np.ndarray, float): Process noise covariance matrix (`N*N`), or
                scalar standard deviation (if univariate normal `noise_dist`).
            sigma_0 (np.ndarray): Initial sigma matrix ("Kalman covariance").
            estimate_0 (np.ndarray): Initial estimate (`N`-vector).
                Defaults to zero(s).

        TODO:
            * matrix shape checks
        """
        self._M = np.array(M)
        self._B = np.array(B)
        self._Z = np.array(Z)

        if sigma_0 is None:
            sigma_0 = np.eye(self._M.shape[1]) * 1e12
        self._sigma = sigma_0  # (prior) estimate covariance
        self._gain = np.zeros_like(sigma_0)  # kalman gain, K
        self._I = np.eye(self._gain.shape[1])

        if estimate_0 is None:
            estimate_0 = np.zeros((self._M.shape[1], 1))
        self._estimate = estimate_0

    def step(self, c, x_s, Q, kalman_estimate=None):
        self._update_gain(Q)
        self._update_estimate(c=c, x_s=x_s, kalman_estimate=kalman_estimate)
        self._update_sigma(Q)

    def _update_gain(self, Q):
        Q = np.array(Q)
        self._gain = self._sigma @ np.linalg.inv(self._sigma + Q)

    def _update_sigma(self, Q):
        gain = self._gain
        gain_sub = self._I - self._gain
        self._sigma = self._M @ (gain_sub @ self._sigma @ gain_sub.T
                                 + gain @ Q @ gain.T) @ self._M.T
        self._sigma += self._Z

    def _update_estimate(self, c, x_s, kalman_estimate=None):
        if kalman_estimate is None:
            kalman_estimate = np.copy(self._estimate)
        self._estimate = (self._M @ kalman_estimate + self._B @ c)
        self._estimate = (self._I - self._gain) @ self._estimate
        if x_s is not None:
            self._estimate += self._gain @ x_s

    @property
    def estimate(self):
        return self._estimate

    @property
    def gain(self):
        return self._gain

    @property
    def sigma(self):
        return self._sigma
