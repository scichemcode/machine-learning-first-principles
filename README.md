# Machine Learning from First Principles

Small, standalone scripts that build machine learning basics with NumPy.

## Contents

```
.
├── neurons/
│   ├── single_neuron.py              # one sigmoid neuron: network, loss and training animation
│   ├── neuron_plots.py               # plots and animation used by single_neuron.py
│   ├── two_neurons.py                # two hidden neurons feeding one output neuron
│   └── neuron_playground.ipynb       # experiments: learning rate, nsteps and adding neurons
└── gradient_descent/
    ├── gradient_descent.py           # weight updates using the gradient of x² + 3x + 6
    └── gradient_descent_parabola.py  # weight updates using the gradient of x²
```

## Requirements

Python 3.6+, NumPy and Matplotlib (for the plots and animation).

```bash
pip install numpy matplotlib
```

## Running

From the repository root:

```bash
python neurons/single_neuron.py                # shows network, loss and training animation
python neurons/two_neurons.py
python gradient_descent/gradient_descent.py           # plots different learning rates
python gradient_descent/gradient_descent_parabola.py  # plots different step counts
```

The notebook needs Jupyter (and optionally `ipywidgets` for its sliders):

```bash
pip install jupyter ipywidgets
jupyter notebook neurons/neuron_playground.ipynb
```
