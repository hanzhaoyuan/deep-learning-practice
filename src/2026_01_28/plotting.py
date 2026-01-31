import matplotlib.pyplot as plt


def plot_data_1d(x, y_gt):
    """Plot 1D data."""
    plt.scatter(x, y_gt, s=20, c="k")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_pred_1d(x, y_gt, y_p):
    """Plot 1D data and predictions."""
    plt.scatter(x, y_gt, s=20, c="k", label="ground truth")
    plt.scatter(x, y_p, s=100, c="tab:orange", marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_data_2d(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.show()


def plot_pred_2d(x, y_gt, y_p):
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50, label="ground turch")
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100,
                marker="x", label="predicted")
    plt.legend(fontsize="15")
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.show()
