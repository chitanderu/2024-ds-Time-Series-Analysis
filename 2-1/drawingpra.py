import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label='linear')
# plt.plot(x, x**2, label='quadratic')
# plt.plot(x, x**3, label='cubic')
#
# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title("Simple Plot")
#
# plt.legend()
#
# plt.show()
#
# # 创建数据
# x = np.linspace(0, 2*np.pi, 10)
# print(x)
# y = np.sin(x)
#
# # 绘制茎状图
# plt.stem(x, y, linefmt='--', markerfmt='o', basefmt='r-')
#
# # 添加标题和标签
# plt.title('Stem Plot Example')
# plt.xlabel('x')
# plt.ylabel('sin(x)')
#
# # 显示图表
# plt.show()
k=2
b = 1  # 截距

# 生成 x 的取值范围
x = np.linspace(-10, 10, 100)  # 从 -10 到 10 之间生成 100 个点
print(x)
# 计算对应的 y 值
y = k * x + b
print(y)
# 绘制直线
plt.plot(x, y, label=f'y = {k}x + {b}', color='blue')

# 添加标题和标签
plt.title('Line Plot of y = kx + b')
plt.xlabel('x')
plt.ylabel('y')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


def calculate_acf(y, lags):
    n = len(y)
    mean_y = np.mean(y)
    acf_values = []

    for k in range(lags + 1):
        y1 = y[:n - k]
        y2 = y[k:]
        acf_k = np.sum((y1 - mean_y) * (y2 - mean_y)) / np.sum((y - mean_y) ** 2)
        acf_values.append(acf_k)

    return np.array(acf_values)


def calculate_pacf(y, lags):
    pacf_values = np.zeros(lags + 1)
    pacf_values[0] = 1

    for k in range(1, lags + 1):
        r = calculate_acf(y, k)
        R = np.linalg.inv(np.array([[r[abs(i - j)] for i in range(k)] for j in range(k)]))
        pacf_values[k] = R[-1].dot(r[1:k + 1])

    return pacf_values


def manual_pacf_plot(y, lags=30):
    pacf_values = calculate_pacf(y, lags)

    plt.figure(figsize=(10, 4))
    plt.stem(range(lags + 1), pacf_values, use_line_collection=True)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title('Manual Partial Autocorrelation Plot')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()

# 示例调用
manual_pacf_plot(y, lags=30)
