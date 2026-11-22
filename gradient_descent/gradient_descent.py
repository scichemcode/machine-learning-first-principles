#Gradient Descent illustration with learning rate and number of steps

#TODO:

# Change the mathematical function

import numpy as np
import matplotlib.pyplot as plt

def loss_fn(x,x_pred):
    return  np.sum( ( x - x_pred ) ** 2 ) / np.size( x )

def dummy_fn(x):
    expr=x**2+3*x+6
    gradient=2*x+3

    return gradient 

def sim_fn(nsteps=1,alpha=0.1):
    weight_init=2
    weight_change=[]

    for _ in range(nsteps):
        weight_init=weight_init-alpha*(dummy_fn(weight_init))
        weight_change.append(weight_init)

    return weight_change,nsteps

def plot_vs_nsteps():
    weights_1,nsteps_1= sim_fn(nsteps=50)
    weights_2,nsteps_2= sim_fn(nsteps=20)
    weights_3,nsteps_3= sim_fn(nsteps=10)

    plt.plot(weights_1,label=f'Steps={nsteps_1}')
    plt.plot(weights_2,label=f'Steps={nsteps_2}')
    plt.plot(weights_3,label=f'Steps={nsteps_3}')

    plt.legend()
    plt.show()

def plot_vs_alpha():

    alpha_vals=[0.005,0.05,0.1,0.5,0.9]

    for alpha in alpha_vals:
        weights,nsteps = sim_fn(nsteps=50,alpha=alpha)
        plt.plot(weights,label=f'Alpha={alpha}')
        plt.legend()

    plt.show()

plot_vs_alpha()
