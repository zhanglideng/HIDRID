import sys

sys.path.append("../../")
import cv2
import utils
from get_t import *
import numpy as np
import os
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

path = "/home/liu/Nutstore Files/科研/毕业设计/大论文/用到的图片/fiveK样例/"
image_path = path + "test/"
label_path = path + "test_mat/"
guide_path = path + "t/"

r = [32, 64]
# hazy_range = [0.06, 0.07, 0.09, 0.11, 0.13, 0.16, 0.21, 0.26, 0.33, 0.41, 0.51, 0.64, 0.80, 1.0]
hazy_range = [0.06, 0.07, 0.09, 0.11, 0.13, 0.16, 0.21, 0.26, 0.33, 0.41, 0.51, 0.64, 0.80]


def Guidedfilter(im, p, r, eps):
    im = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    q[q < 0] = 0
    q[q > 255] = 255
    return q


def work(i, image_list, label_list):
    print("deal: %d" % i)
    image = read_image(image_path + image_list[i])
    label = read_label(label_path + label_list[i])

    for j in r:
        a1 = random.randint(0, len(hazy_range) - 1)
        a2 = random.randint(0, len(hazy_range) - 1)
        if a1 < a2:
            t = label_to_t(label, [hazy_range[a1], hazy_range[a2]])
            save_name = '%s%s_%d_t=[%.2f,%.2f].png' % (
                guide_path, image_list[i][:-4], j, hazy_range[a1], hazy_range[a2])
        else:
            t = label_to_t(label, [hazy_range[a2], hazy_range[a1]])
            save_name = '%s%s_%d_t=[%.2f,%.2f].png' % (
                guide_path, image_list[i][:-4], j, hazy_range[a2], hazy_range[a1])
        t = t * 255
        t = t.astype(np.uint8)
        # guide_t = Guidedfilter(image, t, j, 0.0001)
        guide_t = Image.fromarray(t.astype(np.uint8))

        guide_t.save(save_name, 'png')


def find(i, guide_path):
    # /home/liu/zhanglideng/data/FIVEK/fiveK4.1/guide_t/00000_64_t=[0.41,0.64].png
    return glob.glob("%s%05d*" % (guide_path, i))


if __name__ == '__main__':

    utils.create_dir(guide_path)

    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)
    image_list.sort(key=lambda x: int(x[:-4]))
    label_list.sort(key=lambda x: int(x[:-4]))

    length = len(image_list)
    start = time.time()
    """
    for i in range(length):
        work(i, image_list, label_list)
        end = time.time()
        s = (end - start) / (i + 1) * (length - i - 1)
        print('%d:%02d:%02d' % (s // 3600, s // 60 - s // 3600 * 60, s % 60))
    """
    with ThreadPoolExecutor(max_workers=6) as t:
        obj_list = []
        begin = time.time()
        for i in range(length):
            obj = t.submit(work, i, image_list, label_list)
            obj_list.append(obj)

        for future in as_completed(obj_list):
            print(future.result())
        times = time.time() - begin
        print(times)
