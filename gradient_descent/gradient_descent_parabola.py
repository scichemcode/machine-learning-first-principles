#Gradient Descent illustration with learning rate and number of steps

import numpy as np
import matplotlib.pyplot as plt

def loss_fn(x,x_pred):
    return  np.sum( ( x - x_pred ) ** 2 ) / np.size( x )

def dummy_fn(x):
    expr=x**2
    gradient=2*x

    return gradient 

def sim_fn(nsteps=1,alpha=0.5):
    values=np.linspace(-1,1,nsteps)

    weight_init=2
    weight_change=[]

    for val in values:
        weight_init=weight_init-alpha*(dummy_fn(val))
        weight_change.append(weight_init)

    return weight_change

def plot_vs_nsteps():

    nsteps=[10000,20000,30000,50000]
    
    for nstep in nsteps:
        weights= sim_fn(nsteps=nstep)
        plt.plot(weights,label=f'Steps={nstep}')
        plt.legend()
    
    plt.show()

def plot_vs_alpha():

    alpha_vals=[0.005,0.5,1.0,1.5,2.0]

    for alpha in alpha_vals:
        weights = sim_fn(nsteps=100000,alpha=alpha)
        plt.plot(weights,label=f'Alpha={alpha}')
        plt.legend()

    plt.show()

plot_vs_nsteps()
