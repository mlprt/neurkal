import neurkal.utils as utils

from itertools import product
from math import exp

import numba as nb
from numba import njit
import numpy as np


class PopCode():
    """
    TODO:
        * non-Poisson CR bounds
    """

    def __init__(self, n, act_func, dist, space=None, ds=None):
        """
        Args:
            shape: Network dimensions.
            space: Range of preferred inputs for each dimension.
            act: Activity function, giving average unit rate for an input.
            dist: Unit activity distribution given average activity.
        """

        self.act_func = act_func
        self._act_func = njit(act_func)
        self.dist = dist
        self._weight_func = None
        self._activity = np.zeros(n, dtype=np.float64)
        self._mean_activity = np.zeros(n)
        self._noise = np.zeros(n)
        self._space = space

        if space is None:
            # assume 360 deg periodic coverage in each dimension
            space = (-180, 180)
        # assume preferred stimuli are evenly spaced across range
        self._prefs = np.linspace(*space, n)
        if ds is None:
            ds = _get_derivatives_func(self._act_func, self._prefs)
        self._ds = ds

        try:
            # should be a function taking input and preferred input
            exp(act_func(0.0, 0.0))
        except TypeError:
            raise TypeError("`act` not a function with 2 inputs and 1 output")

    def __call__(self, x, cr_bound=True, certain=False):
        # TODO: better naming? e.g. activity changes with recurrent connections
        # but mean_activity and noise are based on input
        self._mean_activity = self._act_func(x, self._prefs)
        self._x = x
        if certain:
            self._activity = self._mean_activity
            self._noise = np.zeros_like(self._activity)
        else:
            self._activity = self.dist(self._mean_activity).astype(np.float64)
            self._noise = self._activity - self._mean_activity
        self._calc_cr_bound(x)
        return self._activity

    def __len__(self):
        return len(self._prefs)

    def _calc_cr_bound(self, x, dx=0.01):
        dx_f = self._ds(x0=x, dx=dx)
        self._cr_bound = _calc_q(dx_f, self._mean_activity)

    def readout(self, iterations=15, weight_func=None, S=0.001, mu=0.002):
        if weight_func is None and self._weight_func is None:
            self._weight_func = utils.gaussian_filter(p=len(self._prefs),
                                                      K_w=1, delta=0.7)
        recnet = RecurrentPopCode.from_popcode(self,
                                               weight_func=self._weight_func,
                                               mu=mu, S=S)
        recnet._activity = np.copy(self._activity)
        for _ in range(iterations):
            recnet.step()

        # center of mass estimate
        com = utils.arg_popvector(self._activity, self._prefs)
        return com

    def clear(self):
        self._activity = np.zeros_like(self._prefs)

    @property
    def prefs(self):
        return self._prefs

    @property
    def space(self):
        return self._space

    @property
    def cr_bound(self):
        return self._cr_bound

    @property
    def activity(self):
        return self._activity

    @property
    def mean_activity(self):
        return self._mean_activity

    @property
    def noise(self):
        return self._noise


@njit(cache=True)
def _calc_q(dx_f, mean_activity):
    q = dx_f @ np.diag(1 / mean_activity) @ np.transpose(dx_f)
    return 1 / q


def _get_derivatives_func(f, prefs):
    weights = np.array([-0.5, 0, 0.5])
    steps = utils.colvec(np.arange(3) - 1)

    @njit(cache=True)
    def derivatives(x0, dx):
        dfs = weights @ f(x0 + steps * dx, prefs)
        return dfs / dx
    return derivatives


class RecurrentPopCode(PopCode):
    """
    TODO:
       * weight_func same number of dimensions as pop code (1D only currently)
    """

    def __init__(self, weight_func, mu, S=1e-6, normalize=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_func = weight_func
        self.set_weights()

        self._S = S
        self._mu = mu

    @classmethod
    def from_popcode(cls, popcode, weight_func, mu, S=1e-6, *args, **kwargs):
        return cls(weight_func=weight_func, mu=mu, S=S, ds=popcode._ds,
                   n=len(popcode._prefs), act_func=popcode.act_func,
                   dist=popcode.dist, space=popcode.space, *args, **kwargs)

    def set_weights(self):
        shape = [self._prefs.shape[0]] * 2
        self._w = self._weight_func(*np.indices(shape))

    def step(self):
        self._activity = _popcode_calc_activity(self._w, self._activity,
                                                self._S, self._mu)


@njit(cache=True)
def _popcode_calc_activity(w, activity, S, mu):
    u_sq = (w @ np.transpose(activity)) ** 2
    updated = u_sq / (S + mu * np.sum(u_sq))
    return updated


class KalmanBasisNetwork:

    def __init__(self, sensory_inputs, motor_inputs, M, B, Z, K_w=3, mu=0.001,
                 eta=0.002, sigma=None, n_var=10):
        """
        Args:
            sensory_inputs (PopCode):
            motor_inputs (PopCode):

        TODO:
            * prevent blowups/failures due to bad sensory estimates
            * clean up __init__?
        """
        self._all_inputs = sensory_inputs + motor_inputs
        self._D, self._C = len(sensory_inputs), len(motor_inputs)
        self._d = np.arange(self._D)
        self._M, self._B, self._Z = np.array(M), np.array(B), np.array(Z)

        shape = [len(l) for l in self._all_inputs]
        self._N = np.prod(shape)
        # indices for referring to input units
        self._idx = np.array(np.meshgrid(*[range(n) for n in shape]))
        self._idx = self._idx.T.reshape(-1, self._D + self._C)
        self._pairs = tuple(product(range(self._N), repeat=2))

        if sigma is None:
            sigma = np.eye(self._D)
        self._sigma = sigma
        self._lambda = np.ones(self._D)  # TODO: prior gains?
        self._I = np.eye(self._D)
        self._h = np.zeros(self._N)
        self._activity = np.zeros(self._N)
        self._estimates = np.full((n_var, self._D), np.nan)
        self._inputs = np.full((n_var, self._D), np.nan)
        self._n_var = n_var

        # self._prefs contains actual preferences
        self._set_prefs()

        # divisive normalization parameters
        self._mu = mu
        self._eta = eta

        # lateral weights, including weights for readout
        self._K_w = K_w
        self.set_weights(K_w, L=np.hstack([M, B]))  # (mu, eta)
        self._readout_weights = _set_weights(self._prefs, self._K_w,
                                             np.eye(self._D + self._C),
                                             self._D, self._N, self._pairs)

    def update(self, estimate=True, first=False):
        """
        TODO:
            * Pass new state vector
        """
        # update activation function
        self._calc_h()

        # calculate new activities
        if self._C:
            # TODO: multiple motor commands
            f_c = self._all_inputs[self._D].activity[self._idx[:, self._D]]
        else:
            f_c = np.ones(self._N)

        # TODO: allow input activities of different lengths
        # (Numba doesn't like a list of ndarrays passed to calc_activity)
        input_acts = np.vstack([inp.activity for inp in self._all_inputs])
        self._activity = _calc_activity(self._idx, input_acts,
                                        self._h, f_c, self._lambda)

        Q = np.diag([self._all_inputs[d].cr_bound for d in range(self._D)])

        if estimate:
            self._estimate = self.readout()
            gain = self._sigma @ np.linalg.inv(self._sigma @ Q)
            gain_sub = self._I - gain
            self._sigma = self._M @ (gain_sub @ self._sigma @ gain_sub.T
                                     + gain @ Q @ gain.T)
            self._sigma = self._sigma @ self._M.T
            self._sigma += self._Z

            # update sensory gains
            for d in range(self._D):
                self._lambda[d] = self._sigma[d, d] / Q[d, d]

    def _calc_h(self):
        # normalized activation function for each unit
        self._h = _calc_h(self._w, self._activity,
                          self._mu, self._eta)

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
        # TODO:
        com = utils.arg_popvector(self.readout_activity[-1],
                                  self._prefs[:, :self._D])

        # reset
        self._w = weights
        self._activity = activity

        return com

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
def _set_weights(prefs, K_w, L, D, N, pairs):
    w = np.zeros((N, N))
    for k in range(len(pairs)):
        i, j = pairs[k]
        dx_d = (L @ prefs[i]) - prefs[j][:D]
        w_raw = np.sum(np.cos(np.deg2rad(dx_d))) - D
        w[i, j] = np.exp(K_w * w_raw)
    w = w.T
    return w


#@njit(cache=True)
def _calc_activity(idxs, input_activities, h, f_c, lambda_):
    #
    S = np.transpose(np.transpose(input_activities)[idxs].diagonal(axis1=1,
                                                                   axis2=2))
    act = h * f_c + np.dot(lambda_, S)
    return act


class KalmanFilter:

    def __init__(self, M, B, Z, sigma_0=None, estimate_0=None):
        """
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


class StateDynamics:
    def __init__(self, M, B, Z, x0=None):
        try:
            np.hstack([M, B])
        except ValueError:
            raise ValueError("Matrices M and B have different number of rows")

        # dynamical parameter matrices
        self._M = np.array(M)
        self._B = np.array(B)

        self._mu = np.zeros(self._M.shape[1])  # noise mean (0)
        # noise covariance matrix
        self._Z = Z

        self._x = np.zeros((self._M.shape[1], 1))  # state vector

        if x0 is not None:
            try:
                self._x[:] = x0
            except ValueError:
                raise ValueError("Initial state vector is wrong length")

    def update(self, c):
        noise = utils.colvec(np.random.multivariate_normal(self._mu, self._Z))
        self._x = self._M @ self._x + self._B @ c + noise

    @property
    def x(self):
        return self._x

    def change(self, x):
        self._x = x
