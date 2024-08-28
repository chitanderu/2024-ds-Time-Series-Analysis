import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
dfname='附录1.2'
y=pd.read_csv('%s.csv'%dfname,header=None)
y.iloc[:,0]=y.iloc[:,0]
y=y.values[:,0]

acf=np.ones(18)
g0=np.sum((y-y.mean())**2)

for i in range(1,18):
    y1 = y[:-i]
    y2 = y[i:]
    print(y1)
    print(y2)

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


# 示例调用manual_pacf_plot(y, lags=30)


# 示例调用s
manual_pacf_plot(y, lags=17)