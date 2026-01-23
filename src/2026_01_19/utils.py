from numpy.random import default_rng


def neuron_clas_2d(w, x):
    return (x @ w > 0).astype(int)


def train_w_value(x, y_gt):
    num_samples = len(x)
    num_train_iterations = 100
    eta = .1

    rng = default_rng()
    w = rng.standard_normal(size=(2,))

    for i in range(num_train_iterations):
        selected = rng.integers(0, num_samples)
        x_selected = x[selected]
        y_gt_selected = y_gt[selected]
        y_p_selected = neuron_clas_2d(w, x_selected)

        error = y_p_selected - y_gt_selected
        w = w - eta * error * x_selected

        print(f"i={1} w0={w[0]:.2f} w1{w[1]:.2f} error={error[0]:.2f}")
    return w