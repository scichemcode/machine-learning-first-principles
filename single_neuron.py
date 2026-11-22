import numpy as np

def sigmoid(x):  # activation function
    expr=1/(1+np.exp(-x))
    return expr

s=0
x_inp=[1,2,3,4]
weight_matrix=[1.1,1.2,1.3,1.4]

for (w,x) in zip(weight_matrix,x_inp):
    p=w*x
    s+=p


neuron_output=sigmoid(s)

print (f"Output: {neuron_output}")
