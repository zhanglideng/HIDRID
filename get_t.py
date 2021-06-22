from PIL import Image
import random
import numpy as np
import scipy.io as scio
import cv2


def label_to_t(label, hazy_range):
    b = label
    b = b.astype(np.float16)
    for i in range(np.max(label) + 1):
        result = np.where(label == i)
        r = round(random.uniform(hazy_range[0], hazy_range[1]), 2)
        for j in range(len(result[0])):
            # ran = random.uniform(110, 90)
            # b[result[0][j]][result[1][j]] = r * ran / 100
            b[result[0][j]][result[1][j]] = r
    return b


def read_label(path):
    data = scio.loadmat(path)
    a = np.array(data['label'])
    return a


def read_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img


if __name__ == '__main__':
    b = read_label('/media/liu/新加卷/数据集/fiveK/fiveK4/merge_label/06684.mat')
