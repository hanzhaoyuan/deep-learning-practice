import matplotlib.pyplot as plt


def plot_2d_data(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.show()


def plot_2d_data(x, y_gt, y_p):
    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50, label="ground turch")
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100,
                marker="x", label="ground turch")
    plt.legend(fontsize="20")
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.show()
