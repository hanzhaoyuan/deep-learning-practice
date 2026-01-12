from loader import load_1d_csv_data
from plotting import plot_data_1d, plot_pred_1d
from numpy.random import default_rng
from utils import neuron_clas_1d, train_w0_value


def main():
    # (x, y_gt) = load_1d_csv_data('data_class_1d_clean.csv')
    # w0 = train_w0_value(x, y_gt)
    # y_p = neuron_clas_1d(w0, x)

    # plot_pred_1d(x, y_gt, y_p)

    # (x_test, y_gt_test) = load_1d_csv_data("data_class_1d_clean_test.csv")
    # (w0, b) = train_w0_value(x_test, y_gt_test)
    # plot_pred_1d(x_test, y_gt_test, y_p=neuron_clas_1d(w0, b, x_test))

    # (x, y_gt) = load_1d_csv_data(filename="data_class_1d_noisy.csv")
    # plot_data_1d(x,y_gt)

    # (x_test, y_gt_test) = load_1d_csv_data(filename="data_class_1d_noisy_test.csv")
    # (w0, b) = train_w0_value(x_test, y_gt_test)
    # plot_pred_1d(x_test, y_gt_test, y_p=neuron_clas_1d(w0, b, x_test))

    # (x, y_gt) = load_1d_csv_data(filename="data_class_1d_nonconvex.csv")
    # plot_data_1d(x,y_gt)

    (x_test, y_gt_test) = load_1d_csv_data(filename="data_class_1d_nonconvex_test.csv")
    (w0, b) = train_w0_value(x_test, y_gt_test)
    plot_pred_1d(x_test, y_gt_test, y_p=neuron_clas_1d(w0, b, x_test))


if __name__ == '__main__':
    main()
