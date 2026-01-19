from loader import load_data
from plotting import plot_pred_2d
from numpy.random import default_rng
from utils import dnn2_clas_new


def main():
    (x, y_gt) = load_data('data_class_2d_convex_clean.csv')

    num_neurons = 3
    rng = default_rng()
    wa = rng.standard_normal(size=(2, num_neurons))  # Input weights layer 1
    wb = rng.standard_normal(size=(num_neurons, 1))  # Input weights layer 2

    plot_pred_2d(x, y_gt, y_p=dnn2_clas_new(wa, wb, x))


if __name__ == '__main__':
    main()
