import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
y1=[1.0, 1.0, 1.0, 0.96, 0.96, 0.94, 0.74, 0.56, 0.48]
y2=[0.9210526315789473, 1.0, 0.8421052631578947, 0.8947368421052632, 0.7894736842105263, 0.4473684210526316, 0.13157894736842105, 0.07894736842105263, 0.05263157894736842]
x=[0, 1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()














with open('ai_res.txt','r')as f :
    ls=f.readlines()
    x=[]
    y=[]
    for i,l in enumerate(ls):
        l=float(l)
        x.append(i)
        y.append(l)
    plt.xlabel(r"")
    plt.ylabel(r"识别率")
    plt.plot(x,y)
    print(x,y)
with open('person_res.txt','r')as f :
    ls=f.readlines()
    x=[]
    y=[]
    for i,l in enumerate(ls):
        l=float(l)
        x.append(i)
        y.append(l)
    plt.xlabel(r"噪声次数")
    plt.ylabel(r"识别率")
    plt.plot(x,y)
    print(x,y)
plt.show()