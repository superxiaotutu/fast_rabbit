import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
ax_size=16
leg_size=15
x=[0, 1, 2, 3, 4, 5, 6, 7, 8]
y=[1.0, 0.138, 0.022, 0.004, 0.006, 0.0, 0.002, 0.0, 0.0]
plt.xlabel(r"noise level ",fontsize=ax_size)
plt.ylabel(r"accuracy (%)",fontsize=ax_size)
plt.plot(x,y)
plt.show()




def plot():
    with open('res.txt','r')as f :
        ls=f.readlines()
        y=[]
        for i in range(0,45,5):
            l=(float(ls[i]))/500
            print(l)
            y.append(l)
        x=[i for i in range(0,9)]
        print(x)
        print(y)
        plt.xlabel(r"noise level ",fontsize=ax_size)
        plt.ylabel(r"accuracy (%)",fontsize=ax_size)

        plt.plot(x,y)
        plt.show()