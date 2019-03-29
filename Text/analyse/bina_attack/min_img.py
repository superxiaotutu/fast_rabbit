import cv2

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    imgs = []
    ori = cv2.imread('images/46_A.png')
    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    adv = cv2.imread('images/23_40_A.png')
    adv = cv2.cvtColor(adv, cv2.COLOR_BGR2RGB)
    b_bin = cv2.imread('images/9.png')
    c = (ori-adv)*2

    imgs.append(c)
    print(c)
    c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    retval, c_bin = cv2.threshold(c, 0, 255, cv2.THRESH_OTSU)
    imgs.append(ori)
    imgs.append(adv)
    imgs.append(b_bin)
    # print(c_bin)
    imgs.append(c_bin)

    # cv2.imshow('s',c)
    # cv2.waitKey(0)
    for i, v in enumerate(imgs):
        plt.subplot(1, 5, i + 1)
        plt.imshow(v,cmap='gray')
        plt.axis("off")
    plt.show()
