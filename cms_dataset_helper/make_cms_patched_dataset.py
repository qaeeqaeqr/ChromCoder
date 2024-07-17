import os
import shutil
import sys
sys.path.append('..')
from data.cms_patch.patching import single_patching

import cv2
import matplotlib.pyplot as plt
import numpy as np

# goal:将原单根数据集拷贝到新位置，并将新位置的数据集的每章图片进行cms_patch处理，变成*新的格式*的数据集

# note：新格式：图像名称不变，分辨率变成(grid_size*patch_size, grid_size*patch_size, channel)；
# note：       图像中的第一列中含有m个染色体patch（m < grid_size时使用空白patch补全），其他列都是空白patch
# note：       在CPCv2中，patch_size设定为32，grid_size设定为13
patch_size = 32
patch_num = 13

def dst_path_init(dst_path):
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    for i in range(24):
        os.mkdir(os.path.join(dst_path, str(i)))


def process_single_img(img, img_path=None):
    # some images are completely empty!
    patches: list = single_patching(img, patch_size=patch_size, max_patch_num=patch_num)

    col1 = np.concatenate([*patches], axis=0)
    blanks = np.full(shape=(patch_size*patch_num, patch_size*(patch_num-1), 3),
                     fill_value=255)
    new_img = np.concatenate([col1, blanks], axis=1)

    return new_img.astype(dtype=np.uint8)


def make_patched_dataset(src_path='/shared/users/cuit/wy/dtst/taiwan_huaxi_40w',
                         dst_path='/shared/users/cuit/wy/dtst/patched_single_data'):
    print('Initializing destination directory..\n')
    dst_path_init(dst_path)
    num_finished = 0
    num_error_sample = 0

    for i in range(24):  # for each number of chromosome
        sub_src_path = os.path.join(src_path, str(i))
        sub_dst_path = os.path.join(dst_path, str(i))

        img_names = os.listdir(sub_src_path)
        for img_name in img_names:  # for each image in specific number of chromosome
            # 跳过一些taiwan数据
            if img_name[0] != 'G':
                pass
            else:
                img_src_path = os.path.join(sub_src_path, img_name)
                img_dst_path = os.path.join(sub_dst_path, img_name)

                img = cv2.imread(img_src_path)
                try:
                    new_img = process_single_img(img, img_src_path)
                    plt.imsave(img_dst_path, new_img)
                except Exception as e:
                    num_error_sample += 1

            num_finished += 1
            if num_finished % 100 == 0:
                print(f'\r{num_finished} samples finished.. (total {num_error_sample} problemed images)', end='')


if __name__ == '__main__':
    make_patched_dataset()
