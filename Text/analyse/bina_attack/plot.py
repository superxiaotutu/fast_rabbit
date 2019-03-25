import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax_size = 16
leg_size = 12
fig, ax = plt.subplots()

with open('res.txt', 'r')as f:
    ls = f.readlines()
    x = []
    y = []
    for i, l in enumerate(ls):
        l = (int(l)) / 500
        print(l)
        x.append(i)
        y.append(l)
with open('res_bin.txt', 'r')as f:
    ls = f.readlines()
    x = []
    bin_y = []
    for i, l in enumerate(ls):
        l = (int(l)) / 500
        print(l)
        x.append(i)
        bin_y.append(l)
print(x)
print(y)
print(bin_y)
plt.xlabel(r"noise level ", fontsize=ax_size)
plt.ylabel(r"accuracy (%)", fontsize=ax_size)
plt.grid(axis="y")
a = plt.plot(x, y,'x-', label="no_bina", linewidth=2.0, ms=5)
b = plt.plot(x, bin_y,'+-', label="bina", linewidth=2.0, ms=5)

plt.legend(bbox_to_anchor=(1.0, 1), fontsize=leg_size, loc=1, borderaxespad=0.)
plt.show()
