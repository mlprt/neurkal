import numpy as np

def colvec(rowvec):
    v = np.asarray(rowvec)
    return v.reshape(v.size, 1)

def gaussian_filter(p, K_w, delta):
    def filt(i, k):
        return K_w * np.exp((np.cos(2*np.pi*(i - k)/p) - 1) / delta**2)
    return filt

def arg_popvector(activity, prefs):
    """Return monotonically increasing angle, i.e. (-pi, pi)"""
    act_sum = np.sum(activity)
    pv_x = np.dot(activity, np.cos(np.deg2rad(prefs))) / act_sum
    pv_y = np.dot(activity, np.sin(np.deg2rad(prefs))) / act_sum
    arg =  np.arctan2(pv_y, pv_x)
    return arg
