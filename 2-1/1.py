import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

#本代码中延迟为17
from statsmodels.tsa.stattools import pacf

dfname='附录1.2'
y=pd.read_csv('%s.csv'%dfname,header=None)
y.iloc[:,0]=y.iloc[:,0]
y=y.values[:,0]

acf=np.ones(18)
g0=np.sum((y-y.mean())**2)



for i in range(1,18):
    y1=y[:-i]
    y2=y[i:]
    gk=np.sum((y1-y.mean())*(y2-y.mean()))
    acf[i]=gk/g0


print(acf)


def drawts(y,pname):
    ##draw ax
    fig = plt.figure(figsize=(10,8))
    ts_ax=plt.subplot2grid((2,2),(0,0),colspan=2)
    acf_ax=plt.subplot2grid((2,2),(1,0))
    pacf_ax=plt.subplot2grid((2,2),(1,1))
    ##draw plot
    ts_ax.plot(y,'*-')
    ts_ax.set_title('Time Series Analysis Plots')
    smt.graphics.plot_acf(y,lags=None,ax=acf_ax,alpha=0.05) ##2sigma
    smt.graphics.plot_pacf(y,lags=None,ax=pacf_ax,alpha=0.05)  ##2sigma
    #plt.savefig('%s.jpg'%pname,dpi=256)
    plt.show()
    plt.close()

drawts(y,"zxk")

def mydrawts(y, pname):
    lags=17
    # 绘制时间序列分析图
    global pacf_k
    plt.figure(figsize=(10, 8))
    plt.plot(y, label='Time Series')
    plt.title(f'{pname} - Time Series Analysis')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    n = len(y)
    conf_interval1= 1.96 / np.sqrt(np.arange(n, n-lags-1, -1))
    conf_interval = 1.96 / np.sqrt(n)
    print(conf_interval)
    conf_intervals = np.array(
        [conf_interval * np.sqrt(1 + 2 * np.sum(acf[1:k] ** 2)) for k in range(0, lags + 1)])
    print( conf_intervals)

    #绘制自相关图
    lags=17 #误差
    acf_values = smt.acf(y, nlags=17)
    plt.figure(figsize=(10, 8))
    plt.stem(range(1+lags),acf_values, use_line_collection=True)
    plt.fill_between(range(1,lags + 1), conf_intervals[1:], -conf_intervals[1:], color='blue', alpha=0.2)
    plt.title(str(pname)+'Autocorrelation')
    plt.show()

    # 绘制偏自相关图

    pacf_values = np.zeros(lags + 1)
    pacf_values[0] = 1

    for k in range(1, lags + 1):
        r = acf
        R = np.linalg.inv(np.array([[r[abs(i - j)] for i in range(k)] for j in range(k)]))
        pacf_values[k] = R[-1].dot(r[1:k + 1])
    plt.figure(figsize=(10, 4))
    plt.stem(range(lags + 1), pacf_values, use_line_collection=True)
    plt.fill_between(range(lags + 1),  conf_interval,  -conf_interval, color='blue', alpha=0.2)
    plt.title('paritial Autocorrelation Plot')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()
    plt.show()



mydrawts(y,"1")
