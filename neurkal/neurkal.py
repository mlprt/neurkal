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

    def __init__(self, sensory_inputs, motor_inputs, mu, eta):
        """
        Args:
            sensory_inputs (PopCode):
            motor_inputs (PopCode):
        """
        shape = [len(l) for l in sensory_inputs + motor_inputs]
        # for identifying control variables
        self._control_idx = len(sensory_inputs)
        # TODO: use np.indices?
        self.prefs = np.fromfunction(lambda *args: np.dstack(args), shape)
        self.prefs = self.prefs.astype(int)

        self.act = np.zeros(shape)

        # divisive normalization parameters
        self._mu = mu
        self._eta = eta
        self.set_weights()  # (mu, eta)

    def update(self):
        # self.act =
        for idx in self.prefs:
            self.act[idx] = 0

    def h_i(self):
        pass

    def set_weights(self):
        pass


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
