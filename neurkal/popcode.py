"""Classes managing neural population codes.
"""

import neurkal.utils as utils

from math import exp

from numba import njit
import numpy as np


class PopCode():
    """A 1D population of neurons capable of coding a probability distribution.

    TODO:
        * non-Poisson stuff
        * non-periodic readout
        * better solution for ds (don't clone?)
    """

    def __init__(self, n, act_func, dist, space=None, ds=None):
        """
        Args:
            n: Number of units/"neurons".
            act_func: Activity function, giving average unit rate for an input.
                First argument is unit input, second is unit preferred input.
            dist: Unit activity distribution given average activity.
            space: Range of preferred inputs.
                Unit preferences will be uniformly placed over this range.
            ds: Returns derivatives of act_func for each unit.
                (Used to prevent recompilation by Numba on cloning.)
        """
        if space is None:
            # assume 360 deg periodic coverage in each dimension
            space = (-180, 180)
        # assume preferred stimuli are evenly spaced across range

        self.act_func = act_func
        self._act_func = njit(act_func)
        self.dist = dist
        self._space = space

        self._prefs = np.linspace(*space, n)
        if ds is None:
            ds = _get_derivatives_func(self._act_func, self._prefs)
        self._ds = ds

        self._activity = np.zeros(n, dtype=np.float64)
        self._mean_activity = np.zeros(n)
        self._noise = np.zeros(n)

        # for readout
        self._weight_func = None

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
        """Return maximum likelihood estimate of activity maximum.

        Uses recursive connections to converge activity onto a smooth hill
        (stable manifold), then read out the peak as a population vector.
        """
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
    q = dx_f @ np.diag(1 / mean_activity) @ dx_f.transpose()
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
    """Population code with recurrent connections and divisive normalization.
    """

    def __init__(self, weight_func, mu, S=1e-6, *args, **kwargs):
        """
        Args:
            weight_func: Returns lateral weights given two index arrays.
            mu (float): Divisive normalization scaling parameter.
            S (float): Divisive normalization constant parameter.
                Defaults to very low value to avoid division by zero.
        """
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
        self._activity = _calc_activity(self._w, self._activity,
                                        self._S, self._mu)


@njit(cache=True)
def _calc_activity(w, activity, S, mu):
    u_sq = (w @ activity.transpose()) ** 2
    updated = u_sq / (S + mu * np.sum(u_sq))
    return updated
