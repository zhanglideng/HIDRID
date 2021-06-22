# -*- coding: utf-8 -*-
import sys

sys.path.append("../../")
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
from get_t import *
from merge_superpixel import *

data_path = '/home/liu/Nutstore Files/科研/毕业设计/大论文/用到的图片/fiveK样例/'

sigma_range = [0, 0]  # 高斯噪声的方差
air_light_range = [0.6, 0.9]
color_shift = 0.05  # 大气光值的偏差程度
is_mini = False
if is_mini:
    haze_path = data_path + 'mini_haze/'
    image_path = data_path + 'mini_gt/'
    t_path = data_path + 'mini_guide_t/'
else:
    haze_path = data_path + 'merge_haze/'
    image_path = data_path + 'image/'
    t_path = data_path + 'merge_t/'


def save_haze(img, t_name, fog):
    # t_name = '08407_32_t=[0.09,0.51].png'
    name_list = t_name.split('_')
    save_path = '%s%s_%s_a=[%.02f,%.02f,%.02f]%s' % (
        haze_path, name_list[0], name_list[1], fog[0], fog[1], fog[2], name_list[2])
    image_out = Image.fromarray(img.astype('uint8')).convert('RGB')
    image_out.save(save_path, 'png')


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def get_A(image, r, shift, type='normal'):
    h, w, _ = image.shape
    if type == 'normal':
        h, w, _ = image.shape
        A = round(random.uniform(r[0] + shift, r[1] - shift), 2)
        A_R = round(A + random.uniform(-1, 1) * shift, 2)
        A_G = round(A + random.uniform(-1, 1) * shift, 2)
        A_B = round(A + random.uniform(-1, 1) * shift, 2)
        map_A = np.ones((3, h, w))
        map_A[0] = map_A[0] * A_R
        map_A[1] = map_A[1] * A_G
        map_A[2] = map_A[2] * A_B
        map_A = map_A.swapaxes(0, 2).swapaxes(0, 1)
        return map_A
    else:
        # dark = DarkChannel(image, 15)

        imsz = h * w
        # n = int(max(math.floor(imsz / 1000), 1))
        # darkvec = dark.reshape(imsz, 1)
        # image = image.reshape(imsz, 3)
        print(image.shape)
        R = max(image[:, :, 0])
        G = max(image[:, :, 1])
        B = max(image[:, :, 2])

        '''
        indices = np.argsort(darkvec, axis=0)
        indices = indices[imsz - numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        A = cv2.max(A, r[0])
        '''


def make_haze_image(image, t, air_light_range):
    """
    根据给出的深度图像、RGB图像、大气光范围、散射系数范围合成对应的有雾图像
    :param image: RGB图像
    :param t: 传输图
    :param air_light_range: 大气光范围
    :param count: 有雾图像计数
    :return:
    """
    height, width, channel = image.shape
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    noise = np.random.randn(1, height, width) * sigma
    noise = np.concatenate((noise, noise, noise))
    noise = noise.swapaxes(0, 2).swapaxes(0, 1)

    h, w, _ = image.shape
    A = round(random.uniform(air_light_range[0] + color_shift, air_light_range[1] - color_shift), 2)
    A_R = round(A + random.uniform(-1, 1) * color_shift, 2)
    A_G = round(A + random.uniform(-1, 1) * color_shift, 2)
    A_B = round(A + random.uniform(-1, 1) * color_shift, 2)
    map_A = np.ones((3, h, w))
    map_A[0] = map_A[0] * A_R
    map_A[1] = map_A[1] * A_G
    map_A[2] = map_A[2] * A_B
    map_A = map_A.swapaxes(0, 2).swapaxes(0, 1)

    # map_A = get_A(image, air_light_range, color_shift, type='un')

    t = np.expand_dims(t, axis=0)
    t = np.concatenate((t, t, t))
    t = t.swapaxes(0, 2).swapaxes(0, 1)
    noise = 1 + noise
    t = np.multiply(t, noise)
    haze = np.add(np.multiply(image, t), np.add(255 * np.multiply(map_A, (1 - t)), noise))
    # haze = np.add(np.multiply(image, t), 255 * np.multiply(map_A, (1 - t)))
    haze[haze < 0] = 0
    haze[haze > 255] = 255
    return haze, [A_R, A_G, A_B]


if __name__ == '__main__':
    utils.create_dir(haze_path)
    image_list = os.listdir(image_path)
    t_list = os.listdir(t_path)
    image_list.sort(key=lambda x: int(x[:5]))
    t_list.sort(key=lambda x: int(x[:5]))
    t1_list = []
    for i, name in enumerate(t_list):
        t1_list.append(name[:5])
    haze_num = int(len(t_list) / len(image_list))
    length = len(image_list)
    start = time.time()
    for i, image_name in enumerate(image_list):
        # print('dealing: %s' % image_list[i])
        image = read_image(image_path + image_name)
        find_t_index = utils.find_n(t1_list, image_name[:5])
        for j, t_index in enumerate(find_t_index):
            t = read_image(t_path + t_list[t_index])
            t = t.astype(np.float16) / 255
            haze, fog = make_haze_image(image, t, air_light_range)
            save_haze(haze, t_list[t_index], fog)
        end = time.time()
        s = (end - start) / (i + 1) * (length - i - 1)
        if i % 100 == 0:
            print('%d:%02d:%02d' % (s // 3600, s // 60 - s // 3600 * 60, s % 60))
