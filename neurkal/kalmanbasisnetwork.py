"""Approximation of the Kalman Filter as a basis function network.
"""

import neurkal.utils as utils

from itertools import product

from numba import njit, prange
import numpy as np


class KalmanBasisNetwork:
    """Basis function network covering an arbitrary number of input `PopCode`s.
    """

    def __init__(self, sensory_inputs, motor_inputs, M, B, Z, K_w=3, mu=0.001,
                 eta=0.002, sigma=None):
        """
        Args:
            sensory_inputs (List[PopCode]): Sensory variable codes.
            motor_inputs (List[PopCode]): Motor variable codes.

        TODO:
            * prevent blowups/failures due to bad sensory estimates
        """
        self._all_inputs = sensory_inputs + motor_inputs
        self._D, self._C = len(sensory_inputs), len(motor_inputs)
        self._M = np.array(M, dtype=np.float64)
        self._B = np.array(B, dtype=np.float64)
        self._Z = np.array(Z, dtype=np.float64)

        shape = [len(l) for l in self._all_inputs]
        self._N = np.prod(shape)
        # indices for referring to input units
        self._idx = np.array(np.meshgrid(*[range(n) for n in shape]))
        self._idx = self._idx.T.reshape(-1, self._D + self._C)
        self._pairs = np.array(list(product(range(self._N), repeat=2)))
        # TODO: different input network lengths
        self._input_acts = np.zeros((shape[0], self._D))

        if sigma is None:
            sigma = np.eye(self._D)
        self._sigma = np.array(sigma, dtype=np.float64)
        self._lambda = np.ones(self._D)  # TODO: prior gains?

        # self._prefs contains actual preferences
        self._set_prefs()

        # initialize activity
        self.clear()

        # divisive normalization parameters
        self._mu = mu
        self._eta = eta

        # lateral weights, including weights for readout
        self._K_w = K_w
        self.set_weights(K_w, L=np.hstack([M, B]))  # (mu, eta)
        self._readout_weights = _set_weights(self._prefs, self._K_w,
                                             np.eye(self._D + self._C),
                                             self._D, self._N, self._pairs)

        # useful for calculations
        self._d = np.arange(self._D)
        self._I = np.eye(self._D, dtype=np.float64)
        self._f_c_default = np.ones(self._N)

    def update(self, estimate=True, first=False):
        """
        TODO:
            * Pass new state vector
        """
        # update activation function
        self._h = _calc_h(self._w, self._activity,
                          self._mu, self._eta)

        # calculate new activities
        if self._C:
            # TODO: multiple motor commands
            f_c = self._all_inputs[self._D].activity[self._idx[:, self._D]]
        else:
            f_c = self._f_c_default

        # TODO: allow input activities of different lengths
        # (Numba doesn't like a list of ndarrays passed to calc_activity)
        for i, inp in enumerate(self._all_inputs):
            self._input_acts[:, i] = inp.activity
        S = self._input_acts[self._idx].diagonal(axis1=1, axis2=2)
        self._activity = _calc_activity(S, self._h, f_c, self._lambda)

        Q = [self._all_inputs[d].cr_bound for d in range(self._D)]
        Q = self._I * Q  # alternative to np.diag(Q)

        if estimate:
            self._estimate = self.readout()
            self._sigma = _calc_sigma(self._sigma, Q, self._I, self._M,
                                      self._Z)

            # update sensory gains
            self._lambda = _calc_lambda(self._sigma, Q)

    def set_weights(self, K_w, L):
        if not self._C:
            prefs = np.ones((self._N, self._D + 1))
            prefs[:, :-1] = self._prefs
        else:
            prefs = self._prefs
        self._w = _set_weights(prefs, K_w, L, self._D, self._N, self._pairs)

    def _set_prefs(self):
        self._prefs = np.zeros((self._N, self._D + self._C))
        for n in range(self._N):
            pref_spec = zip(self._all_inputs, self._idx[n])
            self._prefs[n, :] = [inp._prefs[i] for inp, i in pref_spec]

    def readout(self, iterations=15):
        # store state
        activity = np.copy(self._activity)
        weights = np.copy(self._w)

        self.readout_activity = np.zeros((iterations, self._N))
        self.readout_activity[0, :] = self._activity
        # converge on D-dim. stable manifold
        self._weights = self._readout_weights
        for i in range(iterations):
            self.update(estimate=False)
            self.readout_activity[i, :] = self._activity
        # center of mass estimate
        com = utils.arg_popvector(self.readout_activity[-1],
                                  self._prefs[:, :self._D])

        # reset
        self._w = weights
        self._activity = activity

        return com

    def clear(self):
        self._activity = np.zeros(self._N)

    @property
    def activity(self):
        return np.copy(self._activity)

    @property
    def prefs(self):
        return self._prefs

    @property
    def lam(self):
        return np.copy(self._lambda)

    @property
    def estimate(self):
        return np.copy(self._estimate)

    @property
    def weights(self):
        return np.copy(self._w)


@njit(cache=True)
def _calc_h(w, act, mu, eta):
    usq = (w @ act) ** 2
    u_den = mu + eta * np.sum(usq)
    h = usq / u_den
    return h


@njit(cache=True)
def _calc_sigma(sigma, Q, I, M, Z):
    gain = sigma @ np.linalg.inv(sigma @ Q)
    gain_sub = I - gain
    sigma = M @ (gain_sub @ sigma @ gain_sub.transpose()
                 + gain @ Q @ gain.transpose()) @ M.transpose() + Z
    return sigma


@njit(cache=True)
def _calc_lambda(sigma, Q):
    lambda_ = np.diag(sigma / Q)
    return lambda_


@njit(cache=True)
def _set_weights(prefs, K_w, L, D, N, pairs):
    w = np.zeros((N, N))
    for k in prange(len(pairs)):
        i, j = pairs[k]
        dx_d = (L @ prefs[i]) - prefs[j][:D]
        w_raw = np.sum(np.cos(np.deg2rad(dx_d))) - D
        w[i, j] = np.exp(K_w * w_raw)
    w = w.T
    return w


@njit(cache=True)
def _calc_activity(S, h, f_c, lambda_):
    # TODO: how slow is transpose? use row format anyway?
    act = h * f_c + np.dot(lambda_, S.transpose())
    return act
