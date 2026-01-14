from loader import load_data
from plotting import plot_pred_2d
from numpy.random import default_rng
from utils import neuron_clas_2d, train_w_value


def main():
    (x, y_gt) = load_data('data_class_2d_clean.csv')
    # rng = default_rng()
    # w = rng.standard_normal(size=(2,))
    w = train_w_value(x, y_gt)

    plot_pred_2d(x, y_gt, y_p=neuron_clas_2d(w, x))


if __name__ == '__main__':
    main()
