#1.Crop the dataset

import os
import cv2
import numpy as np
import shutil


def Scharr_demo(image):
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return int(np.mean(gradxy))


ori_path = '' # origin path
cut_path = '' # patch path

size = 320 # patch size

if not os.path.exists(cut_path):
    os.makedirs(cut_path)

count = 0

data_list = os.listdir(ori_path)
data_list.sort(key=lambda x: int(x[:-4]))
for j in range(len(data_list)):
    print(data_list[j])
    image = cv2.imread(ori_path + data_list[j])
    height, width, channel = image.shape
    for n in [0.5, 0.25, 0.125]:
        if int(width * n) < size or int(height * n) < size:
            continue
        re_image = cv2.resize(image, (int(width * n), int(height * n)), interpolation=cv2.INTER_CUBIC)

        re_height, re_width, channel = re_image.shape
        height_num = re_height // size
        width_num = re_width // size
        height_border = (size * (height_num + 1) - re_height) // height_num
        width_border = (size * (width_num + 1) - re_width) // width_num
        for k in range(height_num + 1):
            for m in range(width_num + 1):
                # print(count)
                patch = re_image[k * (size - height_border - 1):k * (size - height_border - 1) + size,
                        m * (size - width_border - 1):m * (size - width_border - 1) + size]
                # print(patch.shape)
                (mean, stddv) = cv2.meanStdDev(patch)
                mean = int(np.mean(mean))
                stddv = int(np.mean(stddv))

                if stddv >= 30 and 230 >= mean >= 25:
                    name = cut_path + '0' * (5 - len(str(count))) + str(count) + '.png'
                    cv2.imwrite(name, patch)
                    count += 1
                    if count % 1000 == 0:
                        print(count)
