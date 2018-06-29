import cv2
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib ipympl
import glob as gb
import csv

from sklearn import manifold


def data_load(data_folders, res, states, file_ext="JPG", exclude=None, include=None):
    total = []
    total_lb = []
    for df in data_folders:
        img_path = gb.glob(df + '/*.' + file_ext)

        for path in img_path:
            clb = -1
            pl = path.lower()

            if exclude is not None and exclude in pl:
                continue

            if include is not None and include not in pl:
                continue

            for lb in states:
                if lb in pl:
                    clb = states.index(lb)
                    total_lb.append(clb)
                    break

            if clb == -1:
                continue

            img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 'data/Resized_256_256xBroken_0.JPG', cv2.IMREAD_GRAYSCALE)
            if img1 is None:  # 判断读入的img1是否为空，为空就继续下一轮循环
                continue
            res1 = cv2.resize(img1, (res, res))  # 对图片进行缩放，第一个参数是读入的图片，第二个是制定的缩放大小
            res1 = cv2.equalizeHist(res1)
            res1_1 = res1.flatten() / 255.0  # res1_1 = res1.reshape(1,784)/255.0       #将表示图片的二维矩阵转换成一维
            # im_data = np.concatenate((im_data, res1_1))
            res1_1_1 = res1_1.tolist()  # 将numpy.narray类型的矩阵转换成list
            total.append(res1_1_1)

    return np.array(total), np.array(total_lb)






