#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
import sys
import math
import random
import cv2
import gc
import time
import scipy.io as scio
from get_t import *
import threading
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
用于超像素合并
1.根据块间颜色距离合并。
2.根据块间颜色距离和梯度合并。
"""
path = "/home/liu/Nutstore Files/科研/毕业设计/大论文/用到的图片/fiveK样例/"
label_path = path + "test_mat/"
image_path = path + "test/"
merge_label_path = path + "merge_label/"
merge_label_img_path = path + "merge_label_img/"
threshold = 50


def compute_color_d(a, b):
    d = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
    return d


def save_mat(mat, path):
    scio.savemat(path, {'label': mat})


def save_mat_asimg(mat, path):
    mat = mat.astype(np.uint8)
    mat = Image.fromarray(mat)
    mat.save(path)


def merge_pixel(label, img, threshold, flag=True):
    height, width = label.shape

    n = np.max(label) + 1
    s = [[0] * 3] * n
    num = [0] * n
    b = np.arange(0, n)
    # 获得每个超像素的颜色向量
    for i in range(n):
        result = np.where(label == i)
        for j in range(len(result[0])):
            s[i] += img[result[0][j]][result[1][j]]
        s[i] = s[i] / len(result[0])
        num[i] = len(result[0])
    # 计算相邻的超像素之间的颜色距离，对小于阈值的超像素块对记录到b中
    for i in range(n):
        result = np.where(label == i)
        for j in range(len(result[0])):
            if result[1][j] < width - 1 and result[0][j] < height - 1:
                if label[result[0][j]][result[1][j]] != label[result[0][j] + 1][result[1][j]] and \
                        b[label[result[0][j] + 1][result[1][j]]] != b[i]:
                    d = compute_color_d(s[i], s[label[result[0][j] + 1][result[1][j]]])
                    if d <= threshold:
                        b[label[result[0][j] + 1][result[1][j]]] = b[i]
                        flag = False
                if label[result[0][j]][result[1][j]] != label[result[0][j]][result[1][j] + 1] and \
                        b[label[result[0][j]][result[1][j] + 1]] != b[i]:
                    d = compute_color_d(s[i], s[label[result[0][j]][result[1][j] + 1]])
                    if d <= threshold:
                        b[label[result[0][j]][result[1][j] + 1]] = b[i]
                        flag = False
    count = -1
    # 根据数组b的结果来修改label
    for i in range(n):
        l = np.where(b == i)
        l = l[0]
        if len(l) == 0:
            continue
        else:
            count += 1
            for j in l:
                b[j] = count
    for i in range(height):
        for j in range(width):
            label[i][j] = b[label[i][j]]

    # print('当前图像有%d个块' % count)
    return label, flag, count + 1


def work(i, image_list, label_list):
    # 获得图像和标签的路径
    print("deal: %d" % i)
    image = read_image(image_path + image_list[i])
    label = read_label(label_path + label_list[i])
    while 1:
        label, flag, count = merge_pixel(label, image, threshold=50)
        if flag:
            break
    save_mat(label, merge_label_path + label_list[i])
    label = label / label.max() * 255
    save_mat_asimg(label, merge_label_img_path + image_list[i])
    return count


"""
需要统计的指标
平均一张图有几个块
"""
if __name__ == '__main__':

    utils.create_dir(merge_label_path)
    utils.create_dir(merge_label_img_path)

    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)
    image_list.sort(key=lambda x: int(x[:-4]))
    label_list.sort(key=lambda x: int(x[:-4]))
    length = len(image_list)

    with ThreadPoolExecutor(max_workers=6) as t:
        obj_list = []
        for i in range(length):
            obj = t.submit(work, i, image_list, label_list)
            obj_list.append(obj)

        for future in as_completed(obj_list):
            print(future.result())

    '''
    for i in range(length):
        obj = work(i, image_list, label_list)
    '''
