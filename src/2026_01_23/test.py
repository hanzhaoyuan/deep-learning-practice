import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]])

selected = 2  # 随机选择第三个样本

x_selected = np.reshape(x[selected], (1, -1))

print(x_selected)
