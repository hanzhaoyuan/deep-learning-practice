from numpy.random import default_rng
from numpy import reshape, sum, transpose, exp


def neuron_reg_1d(w0, x):
    """Artificial neuron for 1D regression."""
    return w0 * x


def neuron_reg_2d(w, x):
    """Artificial neuron for multidimensional regression"""
    return x @ w


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + exp(-x))


def dnn2_reg(wa, wb, x):
    """Two-layer dense neural network for classification."""
    return sigmoid(x @ wa) @ wb


def train_single_neuron(x, y_gt, w0):
    rng = default_rng()

    num_samples = len(x)
    num_train_iterations = 100
    eta = .1  # Learning rate

    for i in range(num_train_iterations):
        # Select random sample
        selected = rng.integers(0, num_samples)
        x0_selected = x[selected]
        y_gt_selected = y_gt[selected]

        # Neuron prediction
        y_p_selected = neuron_reg_1d(w0, x0_selected)

        # Calculate error
        error = y_p_selected - y_gt_selected
        # Update neuron weight
        w0 = w0 - eta * error * x0_selected

        print(f"i={i} w0={w0[0]:.2f} error={error[0]:.2f}")

    return w0


def train_single_neuron_2d(x, y_gt, w):
    rng = default_rng()

    num_samples = len(x)
    num_train_iterations = 100
    eta = .1  # Learning rate

    for i in range(num_train_iterations):
        # Select random sample
        selected = rng.integers(0, num_samples)
        x_selected = x[selected]
        y_gt_selected = y_gt[selected]

        # Neuron prediction
        y_p_selected = neuron_reg_2d(w, x_selected)

        # Calculate error
        error = y_p_selected - y_gt_selected
        # Update neuron weight
        w = w - eta * error * x_selected

        print(f"i={i} w0={w[0]:.2f} w1={w[1]:.2f} error={error[0]:.2f}")

    return w


def d_sigmoid(x):
    """Derivative of sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))


def train_w_value(x, y_gt, wa, wb):
    num_samples = len(x)
    num_train_iterations = 10 ** 5
    eta = .1  # Learning rate
    rng = default_rng()

    for i in range(num_train_iterations):
        # Select random sample.
        selected = rng.integers(0, num_samples)
        x_selected = reshape(x[selected], (1, -1))
        y_gt_selected = reshape(y_gt[selected], (1, -1))

        # Detailed neural network calculation.
        x_selected_a = x_selected  # Input layer 1
        p_a = x_selected_a @ wa  # Activation potential layer 1
        y_selected_a = sigmoid(p_a)  # Output layer 1

        x_selected_b = y_selected_a  # Input layer 2
        p_b = x_selected_b @ wb  # Activation potential layer 2
        y_selected_b = p_b  # Output layer 2 (output neuron)

        y_p_selected = y_selected_b # Prediction

        # Update weights.
        error = y_p_selected - y_gt_selected  # Calculate error

        delta_b = error * 1
        # delta_b = error * 1
        wb = wb - eta * delta_b * transpose(x_selected_b)

        delta_a = sum(wb * delta_b, axis=1) * d_sigmoid(p_a)
        wa = wa - eta * delta_a * transpose(x_selected_a)

        if i % 100 == 0:
            print(f"{i} y_p={y_p_selected[0, 0]:.2f} error={error[0, 0]:.2f}")
    return (wa, wb)
