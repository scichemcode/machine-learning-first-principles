# Machine Learning from First Principles

Small, standalone scripts that build machine learning basics with NumPy.

## Contents

```
.
├── neurons/
│   ├── single_neuron.py              # one sigmoid neuron with four inputs
│   └── two_neurons.py                # two hidden neurons feeding one output neuron
└── gradient_descent/
    ├── gradient_descent.py           # weight updates using the gradient of x² + 3x + 6
    └── gradient_descent_parabola.py  # weight updates using the gradient of x²
```

## Requirements

Python 3.6+, NumPy and Matplotlib (for the gradient descent plots).

```bash
pip install numpy matplotlib
```

## Running

From the repository root:

```bash
python neurons/single_neuron.py
python neurons/two_neurons.py
python gradient_descent/gradient_descent.py           # plots different learning rates
python gradient_descent/gradient_descent_parabola.py  # plots different step counts
```
