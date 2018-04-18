
import neurkal

import numpy as np


def input_act(x, x_i):
    return 3 * (np.exp(2 * (np.cos(x - x_i) - 1)) + 0.01)


if __name__ == '__main__':
    # simulation parameters
    p = 20  # units in input networks
    t_f = 10  # total time
    dt = 0.1  # timestep

    # dynamical parameters
    M = [[1]]  # internal model dynamics
    B = [[1]]   # constant "motor" dynamics
    Z = 0.001  # motor noise variance
    c = 0.003

    # initial state
    x0 = 0
    state = neurkal.StateDynamics(M, B, Z, x0=x0)
    sensory_input = neurkal.PopCode(p, act_func=input_act,
                                    dist=np.random.poisson)
    # motor_input = popcode.PopCode(p, act_func=input_act, dist=lambda x: x)
    kalman_network = neurkal.KalmanBasisNetwork(sensory_inputs=[sensory_input],
                                                motor_inputs=[])

    for t in np.arange(0, t_f, dt):
        # update input population code with state
        sensory_input(state.x)
        # update activity in basis network implementing Kalman filter
        kalman_network.update()
        # update real state (+ control noise)
        state.update(c)
