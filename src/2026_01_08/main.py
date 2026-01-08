from loader import load_1d_csv_data
from plotting import plot_data_1d, plot_pred_1d
from numpy.random import default_rng
from utils import neuron_clas_1d


def main():
    (x, y_gt) = load_1d_csv_data('data_class_1d_clean.csv')
    rng = default_rng()
    w0 = rng.standard_normal()
    print('x: ', x)
    print('y_gt: ', y_gt)
    print('w0: ', w0)
    y_p = neuron_clas_1d(w0, x)
    print('y_p: ', y_p)
    plot_pred_1d(x, y_gt, y_p)


if __name__ == '__main__':
    main()
