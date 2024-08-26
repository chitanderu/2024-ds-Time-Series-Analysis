import statsmodels.tsa.api as smt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


dfname='附录1.2'
y=pd.read_csv('%s.csv'%dfname,header=None)
y.iloc[:,0]=y.iloc[:,0].astype('float')
y=y.values[:,0]

acf=np.ones(4)
g0=np.sum((y-y.mean())**2)

for i in range(1,4):
    y1 = y[:-i]
    y2 = y[i:]
    print(y1)
    print(y2)


#测试样例
# A=[2,4,6,8,10]
# for i in range(1,4):
#     y1 = A[:-i]
#     y2 = A[i:]
#     print(y1)
#     print(y2)

for i in range(1,4):
    y1=y[:-i]
    #print(y1)
    y2=y[i:]
   # print(y2)
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




