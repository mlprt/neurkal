from numba import njit
import numpy as np


def colvec(rowvec):
    v = np.asarray(rowvec)
    return v.reshape(v.size, 1)


def gaussian_filter(p, K_w, delta):
    @njit(cache=True)
    def filt(i, k):
        return K_w * np.exp((np.cos(2*np.pi*(i - k)/p) - 1) / delta**2)
    return filt


@njit(cache=True)
def arg_popvector(activity, prefs):
    """Return monotonically increasing angle, i.e. (-pi, pi)"""
    pv_x = np.dot(activity, np.cos(np.deg2rad(prefs)))
    pv_y = np.dot(activity, np.sin(np.deg2rad(prefs)))
    arg = np.rad2deg(np.arctan2(pv_y, pv_x))
    return arg
