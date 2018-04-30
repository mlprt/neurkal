
import neurkal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def input_act(x, x_i):
    return 7 * (np.exp(20 * (np.cos(np.deg2rad(x - x_i)) - 1)) + 0.01)


if __name__ == '__main__':
    # plot options 
    sns.set_style("ticks", {'font_scale': 1.5})
    set_palette = lambda: itertools.cycle(sns.color_palette('deep'))
    
    # simulation parameters
    p = 30  # units in input networks
    th_r = [-180, 180]
    d_th = [20, 30]
    steps = 1000
    t_f = 100
    ts = np.linspace(0, t_f, steps)
    
    K_w = 3
    mu = 0.001
    eta = 0.01
    
    # etc
    change = True
    dx_change = 25

    prior = True
    x_prior = 0.0
    
    weird_act_cutoff = 40

    # dynamical parameters
    M = [[1]]  # internal model dynamics
    B = [[0.03]]   # constant "motor" dynamics
    Z = [[0.03]]  # motor noise variance
    c = np.array([[1]])


    # initial state
    state = neurkal.StateDynamics(M, B, Z, x0=d_th[0])
    
    # network initialization
    sensory_input = neurkal.PopCode(p, space=th_r, act_func=input_act,
                                    dist=np.random.poisson)
    # motor_input = popcode.PopCode(p, act_func=input_act, dist=lambda x: x)
    kalman_network = neurkal.KalmanBasisNetwork(sensory_inputs=[sensory_input], 
                                                motor_inputs=[], M=M, B=np.array(B), Z=Z,
                                                mu=mu, eta=eta, K_w=K_w)

    states = []
    inputs = []
    activities = []
    estimates = []
    meas = []
    gains = []

    weird_ros = []

    for i, t in enumerate(ts):
        if prior and not i:
            sensory_input(np.array([x_prior]))  # prior
            kalman_network.update(first=True)
        else:
            sensory_input(state.x[0])
            kalman_network.update()
        states.append(state.x[0])
        inputs.append(sensory_input.activity)
        # update activity in basis network implementing Kalman filter
        #print(sensory_input.cr_bound)
        activities.append(kalman_network.activity)
        estimates.append(kalman_network.estimate)
        gains.append(kalman_network.lam[0])
        meas.append(sensory_input.readout())
        if meas[-1] - state.x[0] > weird_act_cutoff:
            weird_ros.append([state.x[0], sensory_input.activity])

        # update real state (+ control noise)
        if change and i == int(0.5 * steps):
            state.change(state.x - dx_change)
        state.update(c)
        
    # plotting
    plot_step = 5
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    prefs_, ts_ = np.meshgrid(range(p), range(0, steps, plot_step))
    act = np.array(activities)[::plot_step]
    surf = ax.plot_surface(prefs_, ts_, act, linewidth=0)
    plt.show()