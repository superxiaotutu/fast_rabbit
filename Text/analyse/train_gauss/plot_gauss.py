import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
with open('gauss_result.txt','r')as f :
    ls=f.readlines()
    x=[]
    y=[]
    for i,l in enumerate(ls[:40]):
        l=float(l.split()[1])
        x.append(i)
        y.append(l)
    plt.xlabel(r"噪声次数")
    plt.ylabel(r"识别率")

    plt.plot(x,y)
    plt.show()