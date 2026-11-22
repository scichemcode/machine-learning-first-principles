#Single neuron illustration: input -> output -> loss -> gradient -> update

import numpy as np
from neuron_plots import show_plots


# 1. Input to output

def weighted_sum(x_inp,weights,bias):  # linear regression part
    return weights@x_inp+bias          # s = w1*x1 + w2*x2 + ... + b

def sigmoid(s):  # activation function
    return 1/(1+np.exp(-s))            # squashes s to between 0 and 1


# 2. Loss function and its gradient

def sigmoid_loss(s,target):
    m=2*target-1                       # target 0 or 1 becomes -1 or +1
    return sigmoid(-m*s)               # L = sigmoid(-m*s)

def sigmoid_loss_gradient(s,target):
    m=2*target-1
    loss=sigmoid_loss(s,target)
    return -m*loss*(1-loss)            # dL/ds = -m * L * (1-L)


# 3. Settings

x_inp=np.array([1,2,3,4])
weights=np.array([0.11,0.12,0.13,0.14])  # large weights flatten the sigmoid and stall training
bias=0

target=0    # what the output should become
nsteps=50
alpha=0.05  # learning rate

assert target in (0,1), "sigmoid loss needs a target of 0 or 1"


# 4. Training loop

history=[]  # saved for the plots

for step in range(nsteps+1):
    # forward: input -> output -> loss
    s=weighted_sum(x_inp,weights,bias)
    output=sigmoid(s)
    loss=sigmoid_loss(s,target)

    history.append(dict(weights=weights.copy(),bias=bias,s=s,output=output,loss=loss))
    if step==nsteps:
        break  # the last pass only measures the trained neuron

    # backward: chain rule
    dL_ds=sigmoid_loss_gradient(s,target)
    dL_dw=dL_ds*x_inp                  # because ds/dw = x
    dL_db=dL_ds*1                      # because ds/db = 1

    # update: step against the gradient
    weights=weights-alpha*dL_dw
    bias=bias-alpha*dL_db


# 5. Results

print(f"Output: {history[0]['output']}")
print(f"After training: {history[-1]['output']}")

show_plots(x_inp,target,history,sigmoid_loss,sigmoid_loss_gradient)
