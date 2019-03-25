from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    imgs=[]
    a = np.asarray(Image.open('images/bin_adv_0.png', 'r').convert('L'))
    b = np.asarray(Image.open('images/bin_adv_49.png', 'r').convert('L'))
    b_bin = b.copy()
    b_bin[b_bin >= np.mean(b_bin)] = 255
    b_bin[b_bin < np.mean(b_bin)] = 0
    c = b-a
    c_bin =b_bin-a
    imgs.append(a)
    imgs.append(b)
    imgs.append(b_bin)
    imgs.append(c)
    imgs.append(c_bin)

    for i,v in enumerate(imgs):
        plt.subplot(1, 5, i+1)
        plt.imshow(v)
        plt.axis("off")
    plt.show()
