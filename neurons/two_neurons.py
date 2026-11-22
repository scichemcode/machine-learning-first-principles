import numpy as np

def sigmoid(x):  # activation function
    expr=1/(1+np.exp(-x))
    return expr

def neuron(weight_matrix,x_inp,bias):
    s=x_inp@weight_matrix.T+bias
    return sigmoid(s)

x_inp=np.array([1,2,3,4])

weights_1=np.array([1.1,1.2,1.3,1.4])
weights_2=np.array([2.1,2.2,2.3,2.4])

final_weights=np.array([2.7,1.3])

bias_1=3
bias_2=4
bias_output=5

out_1=neuron(weights_1,x_inp,bias=bias_1)
out_2=neuron(weights_2,x_inp,bias=bias_2)

hidden_1=[out_1,out_2]

to_final= np.array(hidden_1)

final_output = neuron(final_weights,to_final,bias=bias_output)

print(f"Final Layer {final_output}")
