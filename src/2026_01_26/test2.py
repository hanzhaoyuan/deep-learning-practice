import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 模拟一个混淆矩阵数据 (10x10)
# 行(y轴)代表实际类别 (True Label)，列(x轴)代表预测类别 (Predicted Label)
# 大部分数据应该在对角线上 (预测正确)
confusion_matrix = np.array([
    [98,  0,  1,  0,  0,  0,  0,  1,  0,  0],  # 数字 0
    [0, 99,  0,  0,  0,  0,  0,  0,  1,  0],  # 数字 1
    [1,  1, 95,  1,  0,  0,  0,  2,  0,  0],  # 数字 2
    [0,  0,  1, 90,  0,  8,  0,  1,  0,  0],  # 数字 3 (注意这里：把3错认为5的有8次)
    [0,  0,  0,  0, 96,  0,  0,  1,  0,  3],  # 数字 4
    [0,  0,  0,  3,  0, 92,  0,  1,  2,  2],  # 数字 5
    [1,  1,  0,  0,  1,  1, 96,  0,  0,  0],  # 数字 6
    [0,  0,  1,  0,  1,  0,  0, 97,  0,  1],  # 数字 7
    [0,  2,  1,  2,  0,  1,  0,  0, 93,  1],  # 数字 8
    [1,  0,  0,  1,  5,  2,  0,  1,  1, 89]  # 数字 9
])

# 2. 设置图像大小
plt.figure(figsize=(10, 8))

# 3. 使用 Seaborn 绘制热力图 (Heatmap)
# annot=True: 在格子里显示数值
# fmt='d': 数值格式为整数
# cmap='Blues': 颜色主题
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])

# 4. 添加标签 (对应您文本中的描述)
plt.ylabel('True Label (Actual)')    # 实际类别 i
plt.xlabel('Predicted Label')        # 预测类别 j
plt.title('Confusion Matrix (MNIST Digits)')

# 5. 显示图像 (阻塞模式，代替 plt.ion)
plt.show()
