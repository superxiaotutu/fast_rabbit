import matplotlib.pyplot as plt

n_y = [0.928, 0.662, 0.494, 0.326, 0.27, 0.184, 0.17, 0.156, 0.124, 0.102]
n_yb=[0.9, 0.91, 0.896, 0.908, 0.904, 0.918, 0.904, 0.892, 0.908, 0.91]
x = [i for i in range(0, 10)]
print(n_y)
print(n_yb)
plt.plot(x, n_y)
plt.plot(x, n_yb)
plt.show()
