from itertools import product
from math import exp
import threading

import numpy as np
from scipy.misc import derivative

class PopCode():
    def __init__(self, shape, act_func, dist, space=None):
        """
        Args:
            shape: Network dimensions.
            space: Range of preferred inputs for each dimension.
            act: Activity function, giving average unit rate for an input.
            dist: Unit activity distribution given average activity.
        """

        self.act_func = act_func
        self.dist = np.vectorize(dist)

        try:
            len(shape)
        except TypeError:
            shape = [shape]

        if space is None:
            # assume 360 deg periodic coverage in each dimension
            space = [(-np.pi, np.pi) for _ in shape]
            # assume preferred stimuli are evenly spaced across range
            self._prefs = [np.linspace(*r, p + 1)[:-1]
                           for r, p in zip(space, shape)]
        else:
            self._prefs = [np.linspace(*r, p) for r, p in zip(space, shape)]
            try:
                self._prefs[0][0]
            except IndexError:
                raise ValueError("Network must have non-trivial shape and \
                                  range of preferred values")

        try:
            # should be a function taking input and preferred input
            exp(act_func(0.0, 0.0))
        except TypeError:
            raise TypeError("`act` not a function with 2 inputs and 1 output")

        # TODO: multiple dimensions
        self._act_func = lambda x: [self.act_func(x, x_i)
                                    for x_i in self._prefs[0]]

    def __call__(self, x, cr_bound=True):
        # TODO: better naming? e.g. activity changes with recurrent connections
        # but mean_activity and noise are based on input
        self.mean_activity = self._act_func(x)
        self.activity = self.dist(self.mean_activity)
        self.noise = self.activity - self.mean_activity
        self._calc_cr_bound(x)
        return self.activity

    def __len__(self):
        return len(self._prefs[0])

    def _calc_cr_bound(self, x, dx=0.01):
        fs = [(lambda x_i: (lambda x_: self.act_func(x_, x_i)))(x_i)
              for x_i in self.prefs[0]]
        dx_f = np.array([derivative(f, x, dx=dx) for f in fs])
        q = 1 / np.matmul(np.matmul(dx_f, np.diag(self.mean_activity)), dx_f.T)
        self._cr_bound = q

    @property
    def prefs(self):
        return self._prefs

    @property
    def cr_bound(self):
        return self._cr_bound



class RecurrentPopCode(PopCode):
    """
    TODO:
       * weight_func same number of dimensions as pop code (1D only currently)
    """
    def __init__(self, weight_func, S, mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_func = np.vectorize(weight_func)
        self.set_weights()

        self._S = S
        self._mu = mu

    def set_weights(self):
        shape = [self._prefs[0].shape[0]] * 2
        self._w = self._weight_func(*np.indices(shape))

    def step(self):
        u = np.matmul(self._w, self.activity)
        u_sq = u ** 2
        self.activity = u_sq / (self._S + self._mu * np.sum(u_sq))


class KalmanBasisNetwork:

    def __init__(self, sensory_inputs, motor_inputs, M, B, K_w=3, mu=0.0,
                 eta=0.002):
        """
        Args:
            sensory_inputs (PopCode):
            motor_inputs (PopCode):

        TODO:
            * pass KalmanFilter rather than M, B?
        """
        self._D, self._C = len(sensory_inputs), len(motor_inputs)
        self._all_inputs = sensory_inputs + motor_inputs
        shape = [len(l) for l in self._all_inputs]
        self._N = np.prod(shape)
        # indices for referring to input units
        self._idx = np.array(np.meshgrid(*[range(n) for n in shape]))
        self._idx = self._idx.T.reshape(-1, self._D + self._C)
        # self._prefs contains actual preferences
        self._set_prefs()

        self.act = np.zeros(self._N)

        # divisive normalization parameters
        self._mu = mu
        self._eta = eta
        #self.set_weights(K_w, M, B)  # (mu, eta)

    def update(self):
        # self.act =
        for idx in self.prefs:
            self.act[idx] = 0

    def h_i(self):
        pass

    def set_weights(self, K_w, M, B):
        N = self.prefs.shape[0]
        L = np.hstack([M, B])
        self._w = np.zeros((N, N))
        for i, j in product(self.prefs, 2):
            dx_d = np.matmul(L, self._prefs[i]) - self._prefs[j]
            w_raw = np.sum(np.cos(dx) for dx in dx_d) - self._D
            self._w[i, j] = np.exp(K_w * w_raw)

    def _set_prefs(self):
        self._prefs = np.zeros((self._N, self._D + self._C))
        for n in range(self._N):
            pref_spec = zip(self._all_inputs, self._idx[n])
            self._prefs[n, :] = [inp._prefs[0][i] for inp, i in pref_spec]


class KalmanFilter:

    def __init__(self, M, B, Z, sigma_0=1e12):
        self._M = M
        self._B = B
        self._Z = Z
        self._I = np.eye(1)
        self._sigma = sigma_0  # (prior) estimate covariance
        self._q = 0  # covariance of feedback estimates, Q
        self._gain = 0  # kalman gain, K

    def step(self):
        # TODO: proper matrix multiplication in these routines
        self._update_gain()
        self._update_estimate(0.0)
        self._update_sigma()

    def _update_gain(self):
        self._gain = self._sigma * np.linalg.inv(self._sigma + self._q)

    def _update_sigma(self):
        gain = self._gain
        gain_sub = self._I - self._gain
        self._sigma = self._M * (gain_sub * self._sigma * gain_sub.T
                                 + gain * self._Q * gain.T) * self._M.T
        self._sigma += self._Z

    def _update_estimate(self, c, x_s):
        self._estimate = (self._I - self._gain) * (self._M * self._estimate
                                                   + self._B * c)
        self._estimate += self._gain * x_s
