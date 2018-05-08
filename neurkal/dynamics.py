"""Classes for managing system state dynamics.
"""

import neurkal.utils as utils

import numpy as np


class LinearDynamics:
    """Manage the evolution of a linear dynamical system with noise.
    """

    _noise_dist_err = "noise_dist takes two arguments, mean and covariance/std"

    def __init__(self, M, B, Z, x0=None, noise_dist=None):
        """
        Args:
            M (np.ndarray): State transition matrix (`N*N`).
            B (np.ndarray): Control effect matrix (`N*1`).
            Z (np.ndarray, float): Process noise covariance matrix (`N*N`), or
                scalar standard deviation (if univariate normal `noise_dist`).
            x0 (np.ndarray, float): Initial state (`N`-vector).
            noise_dist (function): Process noise distribution function.
                Should take either scalar mean and standard deviation, or
                an `N`-vector of means and an `N*N` covariance matrix.
        """
        # dynamical parameter matrices
        try:
            np.hstack([M, B])
        except ValueError:
            raise ValueError("Matrices M and B have different number of rows")

        self._M = np.array(M)
        self._B = np.array(B)

        # make sure noise distribution is well-behaved and configure parameters
        if noise_dist is None:
            noise_dist = np.random.multivariate_normal
            mu = np.array([0.0])
        else:
            try:
                if not noise_dist(0, 0) == 0:
                    raise TypeError()
                mu = 0
                Z = np.ravel(Z)[0]

            except ValueError:
                if not noise_dist([0], [[0]]) == 1:
                    raise TypeError(LinearDynamics._noise_dist_err)
                mu = np.zeros(M.shape[1])

            except TypeError:
                raise TypeError(LinearDynamics._noise_dist_err)

        self._mu = mu  # noise mean (0)
        self._Z = Z  # noise covariance
        self._noise_dist = noise_dist  # noise distribution

        self._x = np.zeros((self._M.shape[1], 1))  # state vector

        if x0 is not None:
            self.initialize(x0)

    def update(self, c, n=1):
        """Iterate the dynamical equations.

        Args:
            c: `N`-vector of control commands.
            n (int): Number of iterations. Defaults to 1.
        """
        for i in range(n):
            noise = utils.colvec(self._noise_dist(self._mu, self._Z))
            self._x = self._M @ self._x + self._B @ c + noise

    def initialize(self, x0):
        """Replace the state vector with `x0`."""
        try:
            self._x[:] = x0
        except ValueError:
            raise ValueError("Initial state vector is wrong length")

    @property
    def x(self):
        return self._x
