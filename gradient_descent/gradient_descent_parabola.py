#Gradient Descent illustration with learning rate and number of steps

import numpy as np
import matplotlib.pyplot as plt

def loss_fn(x,x_pred):
    return  np.sum( ( x - x_pred ) ** 2 ) / np.size( x )

def dummy_fn(x):
    expr=x**2
    gradient=2*x

    return gradient 

def sim_fn(nsteps=1,alpha=0.1):
    weight_init=2
    weight_change=[]

    for _ in range(nsteps):
        weight_init=weight_init-alpha*(dummy_fn(weight_init))
        weight_change.append(weight_init)

    return weight_change

def plot_vs_nsteps():

    nsteps=[50,30,20,10]
    
    for nstep in nsteps:
        weights= sim_fn(nsteps=nstep)
        plt.plot(weights,label=f'Steps={nstep}')
        plt.legend()
    
    plt.show()

def plot_vs_alpha():

    alpha_vals=[0.005,0.05,0.1,0.5,0.9]

    for alpha in alpha_vals:
        weights = sim_fn(nsteps=50,alpha=alpha)
        plt.plot(weights,label=f'Alpha={alpha}')
        plt.legend()

    plt.show()

plot_vs_nsteps()
