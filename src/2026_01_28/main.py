from loader import load_data
from plotting import plot_pred_2d
from numpy.random import default_rng
from utils import dnn2_reg, train_single_neuron_2d, train_w_value, neuron_reg_2d


def main():
    (x, y_gt) = load_data("data_reg_2d_linear.csv")
    rng = default_rng()
    w = rng.standard_normal(size=(2,))
    num_neurons = 3
    wa = rng.standard_normal(size=(2, num_neurons))  # Input weights layer 1
    wb = rng.standard_normal(size=(num_neurons, 1))  # Input weights layer 2
    (wa, wb) = train_w_value(x, y_gt, wa, wb)
    plot_pred_2d(x, y_gt, y_p=dnn2_reg(wa, wb, x))


if __name__ == "__main__":
    main()
