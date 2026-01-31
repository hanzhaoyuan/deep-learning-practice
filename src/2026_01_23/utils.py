from numpy import reshape, sum, transpose, exp
from numpy.random import default_rng


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + exp(-x))


def dnn3_clas(wa, wb, wc, x):
    return sigmoid(sigmoid(sigmoid(x @ wa) @ wb) @ wc)


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
    return sigmoid(sigmoid(x @ wa) @ wb)


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
        y_selected_b = sigmoid(p_b)  # Output layer 2 (output neuron)

        y_p_selected = y_selected_b

        # Update weights.
        error = y_p_selected - y_gt_selected  # Calculate error

        delta_b = error * d_sigmoid(p_b)
        wb = wb - eta * delta_b * transpose(x_selected_b)

        delta_a = sum(wb * delta_b, axis=1) * d_sigmoid(p_a)
        wa = wa - eta * delta_a * transpose(x_selected_a)

        if i % 100 == 0:
            print(f"{i} y_p={y_p_selected[0, 0]:.2f} error={error[0, 0]:.2f}")
    return (wa, wb)


def train_w_value(x, y_gt, wa, wb, wc):
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
        y_selected_b = sigmoid(p_b)  # Output layer 2 (output neuron)

        x_selected_c = y_selected_b # Input layer 3
        p_c = x_selected_c @ wc # Activation potential layer 3
        y_selected_c = sigmoid(p_c) # Output layer 3 (output neuron)
        
        y_p_selected = y_selected_c

        # Update weights.
        error = y_p_selected - y_gt_selected  # Calculate error

        delta_c = error * d_sigmoid(p_c)
        wc = wc - eta * delta_c * transpose(x_selected_c)

        delta_b = error * d_sigmoid(p_b)
        wb = wb - eta * delta_b * transpose(x_selected_b)

        delta_a = sum(wb * delta_b, axis=1) * d_sigmoid(p_a)
        wa = wa - eta * delta_a * transpose(x_selected_a)

        if i % 100 == 0:
            print(f"{i} y_p={y_p_selected[0, 0]:.2f} error={error[0, 0]:.2f}")
    return (wa, wb, wc)
