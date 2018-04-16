
from neurkal import popcode

import numpy as np


def input_act(x, x_i):
    return 3 * (np.exp(2 * (np.cos(x - x_i) - 1)) + 0.01)


if __name__ == '__main__':
    # simulation parameters
    p = 20  # units in input networks
    t_f = 10  # total time
    dt = 0.1  # timestep

    # dynamical parameters
    M = 1  # internal model dynamics
    Bc = 0.003  # constant "motor" dynamics
    Z = 0.001  # motor noise variance

    sensory_input = popcode.PopCode(p, act_func=input_act,
                                    dist=np.random.poisson)
    # motor_input = popcode.PopCode(p, act_func=input_act, dist=lambda x: x)
    kalman_network = popcode.KalmanBasisNetwork(sensory_inputs=[sensory_input],
                                                motor_inputs=[])

    # initial state
    x = 0

    for t in np.arange(0, t_f, dt):
        sensory_input(x)
        kalman_network.update()
        x = M*x + Bc + np.random.normal(0, Z)  # step dynamics
