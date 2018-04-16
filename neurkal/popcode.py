from math import exp
import threading

import numpy as np


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
            space = [(0, 2 * np.pi) for _ in shape]
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

        self._lock = threading.Lock()

    def __call__(self, x):
        self.mean_activity = self._act_func(x)
        self.activity = self.dist(self.mean_activity)
        self.noise = self.activity - self.mean_activity
        return self.activity

    def __len__(self):
        return len(self._prefs[0])

    @property
    def prefs(self):
        return self._prefs


class KalmanBasisNetwork():

    def __init__(self, sensory_inputs, motor_inputs):
        """
        Args:
            sensory_inputs (PopCode):
            motor_inputs (PopCode):
        """
        shape = [len(l) for l in sensory_inputs + motor_inputs]
        motor_start = len(sensory_inputs)
        # TODO: use np.indices?
        self.prefs = np.fromfunction(lambda *args: np.dstack(args), shape)
        self.act = np.zeros(shape)

    def update(self):
        # self.act =
        pass
