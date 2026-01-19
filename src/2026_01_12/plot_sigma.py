import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 创建x范围
x = np.linspace(-10, 10, 400)

# 计算Sigmoid值
y = sigmoid(x)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', color='b')

# 添加标题和标签
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True)

# 显示图例
plt.legend()

# 显示图形
plt.show()
