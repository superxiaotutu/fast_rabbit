import tensorflow as tf
tf.Session()
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
with open('gauss_result.txt','r')as f :
    ls=f.readlines()
    x=[]
    y=[]
    for i,l in enumerate(ls):
        l=float(l.split()[1])
        x.append(i)
        y.append(l)
    plt.xlabel(r"噪声次数")
    plt.ylabel(r"识别率")

    plt.plot(x,y)
    plt.show()