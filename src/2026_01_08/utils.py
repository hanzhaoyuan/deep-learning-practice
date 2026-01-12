from numpy.random import default_rng


def neuron_clas_1d(w0, b, x):
    """Artificial neuron for 1D classification."""
    return (w0 * x + b > 0).astype(int)


def train_w0_value(x, y_gt):
    num_samples = len(x)
    num_train_iterations = 100
    eta = .1  # Learning rate

    rng = default_rng()
    w0 = rng.standard_normal()
    b = rng.standard_normal()

    for i in range(num_train_iterations):
        selected = rng.integers(0, num_samples)  # Select random sample
        x0_selected = x[selected]
        y_gt_selected = y_gt[selected]

        y_p_selected = neuron_clas_1d(w0, b, x0_selected)  # Neuron prediction

        error = y_p_selected - y_gt_selected  # Calculate error

        w0 = w0 - eta * error * x0_selected  # Update neuron weight
        b = b - eta * error    # Update neuron bais

        print(f"i={i} w0={w0:.2f} b={b:.2f} error={error:.2f}")

    return (w0, b)
