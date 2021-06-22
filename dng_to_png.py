# coding=utf-8
import sys

sys.path.append("..")
import utils
import os
import rawpy
import imageio
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def work(i, dng_path, dng_list, png_path):
    print(i)
    dng = rawpy.imread(dng_path + dng_list[i])
    img = dng.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    img = np.float32(img / 65535.0 * 255.0)
    img = Image.fromarray(img.astype(np.uint8))
    img_name = '%s%d.png' % (png_path, i)
    img.save(img_name)
    return 0


if __name__ == '__main__':
    dng_path = ' ' # dng image path
    png_path = ' ' # result path
    img_list = [] # image list
    img_list.sort(key=lambda x: int(x[:-4]))
    new_img_list = []
    for img_name in img_list:
        new_img_list.append(int(img_name[:-4]))
    count = 0
    utils.create_dir(png_path)
    dng_list = os.listdir(dng_path)
    length = len(dng_list)

    with ThreadPoolExecutor(max_workers=6) as t:
        obj_list = []
        begin = time.time()
        for i in range(length):
            obj = t.submit(work, i, dng_path, dng_list, png_path)
            obj_list.append(obj)

        for future in as_completed(obj_list):
            print(future.result())
        times = time.time() - begin
        print(times)
