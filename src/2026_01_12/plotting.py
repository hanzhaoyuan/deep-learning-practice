import matplotlib.pyplot as plt


def plot_data_2d(x, y_gt):
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=15)
    plt.ylabel("x1", fontsize=15)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_pred_2d(x, y_gt, y_p):
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50, label='ground truth')
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100, marker='x', label='predicted')
    plt.colorbar()
    plt.legend(fontsize=20)
    plt.axis("equal")
    plt.xlabel("x0", fontsize=15)
    plt.ylabel("x1", fontsize=15)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()
