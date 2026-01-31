from loader import load_data
from plotting import plot_pred_2d, plot_data_2d
from numpy.random import default_rng
from utils import dnn3_clas, train_w_value


def main():
    (x, y_gt) = load_data("data_class_2d_nonconvex.csv")
    num_neurons_1 = 7
    num_neurons_2 = 5

    rng = default_rng()
    # Input weights layer 1
    wa = rng.standard_normal(size=(2, num_neurons_1))
    # Input weights layer 2
    wb = rng.standard_normal(size=(num_neurons_1, num_neurons_2))
    # Input weights layer 3
    wc = rng.standard_normal(size=(num_neurons_2, 1))

    (wa, wb, wc) = train_w_value(x, y_gt, wa, wb, wc)
    plot_pred_2d(x, y_gt, y_p=dnn3_clas(wa, wb, wc, x))


if __name__ == "__main__":
    main()
