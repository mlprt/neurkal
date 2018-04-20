import neurkal.utils as utils

from itertools import product
from math import exp
import threading

import numpy as np
from scipy.misc import derivative
from scipy.integrate import quad

class PopCode():
    """
    TODO:
        * non-Poisson CR bounds
    """
    def __init__(self, n, act_func, dist, space=None):
        """
        Args:
            shape: Network dimensions.
            space: Range of preferred inputs for each dimension.
            act: Activity function, giving average unit rate for an input.
            dist: Unit activity distribution given average activity.
        """

        self.act_func = act_func
        self.dist = np.vectorize(dist)
        self.activity = np.zeros(n)
        self.mean_activity = np.zeros(n)
        self.noise = np.zeros(n)

        if space is None:
            # assume 360 deg periodic coverage in each dimension
            space = (-np.pi, np.pi)
            # assume preferred stimuli are evenly spaced across range
        self._prefs = np.linspace(*space, n + 1)[:-1]

        try:
            # should be a function taking input and preferred input
            exp(act_func(0.0, 0.0))
        except TypeError:
            raise TypeError("`act` not a function with 2 inputs and 1 output")

        # TODO: multiple dimensions
        self._act_func = lambda x: [self.act_func(x, x_i)
                                    for x_i in self._prefs]

    def __call__(self, x, cr_bound=True):
        # TODO: better naming? e.g. activity changes with recurrent connections
        # but mean_activity and noise are based on input
        self.mean_activity = self._act_func(x)
        self.activity = self.dist(self.mean_activity)
        self.noise = self.activity - self.mean_activity
        self._calc_cr_bound(x)
        return self.activity

    def __len__(self):
        return len(self._prefs)

    def _calc_cr_bound(self, x, dx=0.01):
        fs = [(lambda x_i: (lambda x_: self.act_func(x_, x_i)))(x_i)
              for x_i in self.prefs]
        dx_f = np.array([derivative(f, x, dx=dx) for f in fs])
        q = 1 / (dx_f @ np.diag(self.mean_activity) @ dx_f.T)
        self._cr_bound = q

    def readout(self, iterations=100, weight_func=None, S=0.001, mu=0.002):
        if weight_func is None:
            weight_func = utils.gaussian_filter(p=len(self._prefs), K_w=1,
                                                delta=0.7)
        recnet = RecurrentPopCode(weight_func=weight_func, mu=mu, S=S,
                                  n=len(self._prefs),
                                  act_func=self.act_func, dist=self.dist)
        recnet.activity = np.copy(self.activity)
        for _ in range(iterations):
            recnet.step()

        # center of mass estimate
        com = utils.arg_popvector(self.activity, self._prefs)
        return com

    def clear(self):
        self.activity = np.zeros_like(self._prefs)

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
    def __init__(self, weight_func, mu, S=0.001, normalize=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if normalize:
            # TODO: fix? does not do what I expected...
            bounds = np.min(self._prefs), np.max(self._prefs)
            integral = quad(lambda j: weight_func(0.0, j), *bounds)[0]
        else:
            integral = 1.0
        weight_func_ = lambda i, j: weight_func(i, j) / integral
        self._weight_func = np.vectorize(weight_func_)
        self.set_weights()

        self._S = S
        self._mu = mu

    def set_weights(self):
        shape = [self._prefs.shape[0]] * 2
        self._w = self._weight_func(*np.indices(shape))

    def step(self):
        u = self._w @ self.activity.T
        u_sq = u ** 2
        self.activity = u_sq / (self._S + self._mu * np.sum(u_sq))


class KalmanBasisNetwork:

    def __init__(self, sensory_inputs, motor_inputs, M, B, K_w=3, mu=0.01,
                 eta=0.002, prior=None):
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
        self._h = np.zeros(self._N)
        self._sigma = np.eye(self._D)
        self._lambda = np.zeros(self._D)  # TODO: prior gains?
        self._activity = np.zeros(self._N)

        # divisive normalization parameters
        self._mu = mu
        self._eta = eta
        self._K_w = K_w
        self.set_weights(K_w, L=np.hstack([M, B]))  # (mu, eta)

    def update(self, sigma=None):
        """
        TODO:
            * Pass new state vector
            * Get sigma from Kalman Filter equations
        """
        if sigma is not None:
            self._sigma = sigma
        # update activation function
        self._calc_h()

        # update sensory gains
        for d in range(self._D):
            q = self._all_inputs[d].cr_bound
            self._lambda[d] = self._sigma[d] / q

        # calculate new activities
        if self._C:
            f_c = [self._all_inputs[self._D].activity[self._idx[i, self._D]]
                   for i in range(self._N)]
        else:
            f_c = np.ones(self._N)
        for i in range(self._N):
            idx = self._idx[i]
            S_d = [self._all_inputs[d].activity[idx[d]]
                   for d in range(self._D)]
            # TODO: multiple motor commands...
            self._activity[i] = self._h[i] * f_c[i] + np.dot(self._lambda, S_d)

    def _calc_h(self):
        # normalized activation function for each unit
        self._calc_u()
        for i in range(self._N):
            self._h[i] = (self._u[i] ** 2) / self._u_den


    def _calc_u(self):
        # raw activation for each unit
        self._u = np.zeros(self._N)
        for i, j in product(range(self._N), repeat=2):
            self._u[i] += self._w[i, j] * self._activity[j]
        self._u_den = self._mu + self._eta * np.sum(self._u ** 2)

    def set_weights(self, K_w, L, readout=False):
        self._w = np.zeros((self._N, self._N))
        if not self._C and not readout:
            prefs = np.ones((self._N, 2))
            prefs[:, 0] = self._prefs.T
        else:
            prefs = self._prefs
        for i, j in product(range(self._N), repeat=2):
            #print('L: ', L, ', prefs: ', prefs[i])
            dx_d = (L @ prefs[i]) - self._prefs[j][:self._D]
            w_raw = np.sum(np.cos(dx) for dx in dx_d) - self._D
            self._w[i, j] = np.exp(K_w * w_raw)
        #self._w = self._w.T

    def _set_prefs(self):
        self._prefs = np.zeros((self._N, self._D + self._C))
        for n in range(self._N):
            pref_spec = zip(self._all_inputs, self._idx[n])
            self._prefs[n, :] = [inp._prefs[i] for inp, i in pref_spec]

    def readout(self, iterations=100):
        # store state
        activity = np.copy(self._activity)
        weights = np.copy(self._w)

        # converge on D-dim. stable manifold
        self.set_weights(self._K_w, L=np.eye(self._D + self._C), readout=True)
        for _ in range(iterations):
            self.update(self._sigma)
        # center of mass estimate
        # TODO: pop. vector? multidimensional?
        nm = np.zeros(self._D)
        dm = 0
        for i in range(self._N):
            nm += self._activity[i] * self._prefs[i, :self._D]
            dm += self._activity[i]

        # reset
        self._w = weights
        self._activity = activity

        return nm / dm
        #return self._prefs[np.argmax(self._activity)]

    @property
    def activity(self):
        return self._activity

    @property
    def prefs(self):
        return self._prefs

    @property
    def lam(self):
        return self._lambda


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
            sigma_0 = [[1e12]]
        self._sigma = sigma_0  # (prior) estimate covariance
        self._gain = np.zeros_like(sigma_0)  # kalman gain, K
        self._I = np.eye(self._gain.shape[0])

        if estimate_0 is None:
            estimate_0 = np.zeros_like(Z)
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

        self._x = np.zeros(self._M.shape[1])  # state vector

        if x0 is not None:
            try:
                self._x[:] = x0
            except ValueError:
                raise ValueError("Initial state vector is wrong length")

    def update(self, c):
        noise = np.random.multivariate_normal(self._mu, self._Z)
        self._x = np.ravel(self._M @ self._x + self._B @ c + noise)

    @property
    def x(self):
        return self._x
