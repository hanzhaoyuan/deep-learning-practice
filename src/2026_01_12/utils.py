import numpy as np
from numpy.random import default_rng
from numpy import sqrt, exp


def neuron_clas_2d(w, x):
    return (x@w > 0).astype(int)


def train_w_value(x, y_gt):
    num_samples = len(x)
    num_train_iterations = 100
    eta = .1  # Learning rate

    rng = default_rng()
    w = rng.standard_normal(size=(2,))
    b = rng.standard_normal()

    for i in range(num_train_iterations):
        selected = rng.integers(0, num_samples)  # Select random sample
        x_selected = x[selected]
        y_gt_selected = y_gt[selected]

        # y_p_selected = neuron_clas_2d(w, x_selected)  # Neuron prediction
        y_p_selected = neuron_clas_2d_bias(
            w, b, x_selected)  # Neuron prediction

        error = y_p_selected - y_gt_selected  # Calculate error
        w = w - eta*error*x_selected  # Update neuron weight
        w = w / sqrt(w[0] ** 2 + w[1] ** 2)  # Weight regularization
        b = b - eta * error  # Update neuron bias

        print(f"i={i} w0={w[0]:.2f} w1={w[1]:.2f} error={error[0]:.2f}")
    return w


def neuron_clas_2d_bias(w, b, x):
    """Artificial neuron with a bias for multidimensional classification."""
    return (x @ w + b > 0).astype(int)


def sigmoid(x):
    """Sigmoid function."""
    return 1/(1+exp(-x))


def dnn2_clas(wa, wb, x):
    """Two-layer dense neural network for classification."""
    x_a = x  # Input layer 1
    p_a = x_a @ wa  # Activation potential layer 1
    y_a = sigmoid(p_a)  # Output layer 1

    x_b = y_a  # Input layer 2
    p_b = x_b @ wb  # Activation potential layer 2
    y_b = sigmoid(p_b)  # Output layer 2 (output neuron)

    y_p = y_b  # Network prediction

    return y_p


def dnn2_clas_new(wa, wb, x):
    """Two-layer dense neural network for classification."""
    return sigmoid(sigmoid(x@wa)@wb)
