import os
import random
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt

from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

from data.cms_patch.patching import CMSPatching, single_patching
from cpc_models.ResNetV2_Encoder import PreActResNet14_Encoder
from cpc_models.ViT_Encoder import ViT_Encoder
from argparser.train_CPC_argparser import argparser
from anomaly_detection import Detector, PatchDetector

import torch
from torch import nn

grid_size = 13


def encoder_test():
    """
    Encoder负责对每个patch的染色体进行编码，可能需要挑选合适的encoder对3*32*32的染色体patch进行编码。
    """
    x = torch.randn(size=(4, 13, 1, 3, 32, 32), ).to('cuda')  # (batch_size, grids, 1, channel, patch_size, patch_size)
    args = argparser()
    args.batch_size = 4  # for testing
    use_classifier = False
    # encoder = PreActResNet14_Encoder(args, use_classifier).to('cuda')
    encoder = ViT_Encoder(args, use_classifier).to('cuda')
    print('num_params:', sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    print(encoder(x).shape)


def _draw_theoratical_fp_fn():
    """

    draw the results of different model.

    """
    fp = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    resnet50_fn = [0.011, 0.004, 0.002, 0.002, 0.002, 0.002, 0.002]
    resnet34_fn = [0.066, 0.035, 0.020, 0.007, 0.006, 0.006, 0.002]
    resnet14_fn = [0.118, 0.064, 0.035, 0.028, 0.015, 0.009, 0.007]

    plt.plot(fp, resnet50_fn, c='g', label='resnet50')
    plt.scatter(fp, resnet50_fn, c='g')
    plt.plot(fp, resnet34_fn, c='b', label='resnet34')
    plt.scatter(fp, resnet34_fn, c='b')
    plt.plot(fp, resnet14_fn, c='r', label='resnet14')
    plt.scatter(fp, resnet14_fn, c='r')

    plt.xticks(fp)
    plt.xlabel('patch-wise false positive')
    plt.ylabel('false negative')
    plt.legend()
    plt.show()


def _draw_pratical_fp_fn():
    """

    draw the results tested on practical dataset.

    according to the sequence:
    patch_threshold[0.05, 0.075, 0.1, 0.125, 0.15, 0.2] tolarance[0, 1, 2]
    """
    patch_thresh = [0.05, 0.05, 0.05, 0.075, 0.075, 0.075, 0.1, 0.1, 0.1, 0.125, 0.125, 0.125, 0.15, 0.15, 0.15, 0.2,
                    0.2, 0.2]
    tolarance = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    fp = [0.2224, 0.0599, 0.0210, 0.3066, 0.0984, 0.0368, 0.3789, 0.1377, 0.0560, 0.4430, 0.1784, 0.0770, 0.4998,
          0.2217, 0.1003, 0.5967, 0.3026, 0.1519]
    fn = [0.0110, 0.0349, 0.1360, 0.0037, 0.0165, 0.0643, 0.0018, 0.0073, 0.0386, 0.0018, 0.0073, 0.0239, 0.0018,
          0.0073, 0.0184, 0.0018, 0.0055, 0.0110]
    pareto_fp = []
    pareto_fn = []

    num_data = len(fp)
    for i in range(num_data):
        is_pareto = True
        for j in range(num_data):
            if not (fp[j] == fp[i] and fn[j] == fn[i]) and fp[j] <= fp[i] and fn[j] <= fn[i]:
                is_pareto = False
                break
        if is_pareto:
            pareto_fp.append(fp[i])
            pareto_fn.append(fn[i])

    plt.scatter(fp, fn, c='#aaaaaa')
    plt.scatter(pareto_fp, pareto_fn, c='r')

    plt.xlabel('False Positive')
    plt.ylabel('False Negative')

    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    plt.show()


def patchwise_anomaly_detection_test_A(target='fp'):
    root_path = '/data/home/cuitor/wangyu/datasets/A_paired_data'
    total_cases = len(os.listdir(root_path))
    finished_cases = 0
    total_samples = 0
    FP = 0
    TN = 0
    detector = PatchDetector()
    detector.build()

    for case_name in os.listdir(root_path):
        finished_cases += 1
        print('\r', finished_cases, '/', total_cases, 'cases finished..', end='')
        case_path = os.path.join(root_path, case_name)

        for number in range(0, 22):
            cur_number_path = os.path.join(case_path, str(number))

            if not (os.path.exists(os.path.join(cur_number_path, '0.png')) and
                    os.path.exists(os.path.join(cur_number_path, '1.png'))):
                continue

            img1 = Image.open(os.path.join(cur_number_path, '0.png'))
            img2 = Image.open(os.path.join(cur_number_path, '1.png'))

            res = detector.detect(img1, img2, number, 'A')

            TN = TN + 1 if not res else TN
            FP = FP + 1 if res else FP
            total_samples += 1

    print('总共正常的样本数量(对)：', total_samples)
    print('预测正确的:', TN, '预测错误的:', FP)
    print('假阳性率：', FP / total_samples)
    return FP / total_samples


def patchwise_anomaly_detection_test_L(target='fp'):
    root_path = '/data/home/cuitor/wangyu/datasets/L_paired_data'
    total_cases = 1000
    finished_cases = 0
    total_samples = 0
    FP = 0
    TN = 0
    detector = PatchDetector()
    detector.build()

    for case_name in os.listdir(root_path)[:total_cases]:
        finished_cases += 1
        print('\r', finished_cases, '/', total_cases, 'cases finished..', end='')
        case_path = os.path.join(root_path, case_name)

        for number in range(0, 22):
            cur_number_path = os.path.join(case_path, str(number))

            if not (os.path.exists(os.path.join(cur_number_path, '0.png')) and
                    os.path.exists(os.path.join(cur_number_path, '1.png'))):
                continue

            img1 = Image.open(os.path.join(cur_number_path, '0.png'))
            img2 = Image.open(os.path.join(cur_number_path, '1.png'))

            res = detector.detect(img1, img2, number, 'A')

            TN = TN + 1 if not res else TN
            FP = FP + 1 if res else FP
            total_samples += 1

    print('总共正常的样本数量(对)：', total_samples)
    print('预测正确的:', TN, '预测错误的:', FP)
    print('假阳性率：', FP / total_samples)
    return FP / total_samples


def patchwise_anomaly_detection_test_G(target='fp', tolarance=0):
    '''

    Args:
        target: 检测目标，测试假阴性/假阳性
        tolarance: 容忍度。异常的块的数量小于等于￥tolarance￥时，这对染色体被判定为正常。

    Returns:

    '''
    if target == 'fn':
        root_path = '/data/home/cuitor/wangyu/datasets/extracted-yiwei'
        total_samples = 0
        TP = 0
        FN = 0
        detector = PatchDetector()
        detector.build()

        for i in range(22):
            print('\r', i, '/ 21 on processing..', end='')
            folder_path = os.path.join(root_path, str(i))
            img_names = os.listdir(folder_path)
            img_names = sorted(img_names)
            assert len(img_names) % 2 == 0
            for j in range(0, len(img_names), 2):
                pair0_path = os.path.join(folder_path, img_names[j])
                pair1_path = os.path.join(folder_path, img_names[j + 1])
                res = detector._detect(Image.open(pair0_path),
                                       Image.open(pair1_path),
                                       i,
                                       ctype='G',
                                       ret_type='lst')
                total_samples += 1

                # TP = TP + 1 if len(res) > tolarance else TP
                # FN = FN + 1 if not len(res) > tolarance else FN
                # 1-18号染色体在报告图的前三排，也就是长度较长
                TP = TP + 1 if not ((len(res) <= tolarance and i <= 17) or (len(res) == 0 and i > 17)) else TP
                FN = FN + 1 if ((len(res) <= tolarance and i <= 17) or (len(res) == 0 and i > 17)) else FN

        print('总共异常样本数量（对）：', total_samples)
        print('预测正确：', TP, '\t预测错误：', FN)
        print('假阴性率:', FN / total_samples)
        return FN / total_samples

    elif target == 'fp':
        root_path = '/data/home/cuitor/wangyu/datasets/single_data_200k'
        total_samples = 0
        FP = 0
        TN = 0
        detector = PatchDetector()
        detector.build()

        for i in range(22):  # 1到22号染色体
            print('\r', i, '/ 21 on processing..', end='')
            folder_path = os.path.join(root_path, str(i))
            img_names = os.listdir(folder_path)
            img_names = [img_name for img_name in img_names if img_name[0] == 'G']  # 模型只针对低分辨的染色体
            img_names.sort()

            skip_current = False
            len_img_names = len(img_names) - 1
            for num_img in range(len_img_names):
                if skip_current:  # 是一对中的第二章图片
                    skip_current = False
                    continue

                if img_names[num_img].split('.')[0] != img_names[num_img + 1].split('.')[0] or \
                        img_names[num_img].split('.')[1] != img_names[num_img + 1].split('.')[1]:  # 只有单张
                    continue

                img0 = os.path.join(folder_path, img_names[num_img])
                img1 = os.path.join(folder_path, img_names[num_img + 1])
                # When trying new detection methods, remove "try...except..."
                try:
                    res = detector._detect(Image.open(img0),
                                           Image.open(img1),
                                           i,
                                           ctype='G',
                                           ret_type='lst')
                except Exception:
                    continue

                skip_current = True
                total_samples += 1
                TN = TN + 1 if ((len(res) <= tolarance and i <= 17) or (len(res) == 0 and i > 17)) else TN
                FP = FP + 1 if not ((len(res) <= tolarance and i <= 17) or (len(res) == 0 and i > 17)) else FP

        print('总共正常的样本数量(对)：', total_samples)
        print('预测正确的:', TN, '预测错误的:', FP)
        print('假阳性率：', FP / total_samples)
        return FP / total_samples

    else:
        raise ValueError('unexpected target')


def visualize_patchwise_anomaly_detection(img_path1='/home/kms/wangyu_dtst/extracted-yiwei/0/G2105202393-0-0-a.jpg',
                                          img_path2='/home/kms/wangyu_dtst/extracted-yiwei/0/G2105202393-0-0-n.jpg',
                                          save_fig=True,
                                          ):
    """
    将patchwise异常检测结果展示（用红色区域表明哪里有异常）
    :return:
    """
    from data.cms_patch.SKELETONlib.MySkeletonize import get_skeleton
    from data.cms_patch.SKELETONlib.Graph import Graph

    patch_size = 32
    detector = PatchDetector()
    detector.build()

    def get_center_points(img, patch_size=32):
        # 得到每个有内容的patch的中心点
        im_skeleton = get_skeleton(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        graph = Graph(im_skeleton)
        if len(graph.nodes) > 1:
            paths = graph.endpoint_paths
            longest_path = graph.get_longest_merged_path(paths)
            rows, cols = graph.nodes_to_numpy(longest_path)
        else:
            for key in graph.nodes.keys():
                r, c = key.split(',')
            rows = [eval(r)]
            cols = [eval(c)]
        center_points = []
        for i in range(len(rows)):
            if i % (patch_size / 2) == 0 or i == len(rows) - 1:
                center_points.append((rows[i], cols[i]))
        return center_points

    def draw_anomaly(center_points, anomaly_patch_index, img, pixel_offset=30, cms_gray_threshold=252):
        for i in range(len(center_points)):
            if i in anomaly_patch_index:
                left = center_points[i][0] - int(patch_size / 2)
                top = center_points[i][1] - int(patch_size / 2)
                # 将不是白色的像素点变得红一些，以提示异常
                # 使用numpy完成两层for循环的逻辑，以加速计算
                '''
                patch = img[left:left + patch_size, top:top + patch_size]

                logical_index = np.all(np.logical_and(patch[:, :, 0].reshape(patch_size, patch_size, 1) < cms_gray_threshold,
                                                      np.all(patch[:, :, 0:2] == patch[:, :, 0].reshape(patch_size,
                                                                                                        patch_size, 1),
                                                             axis=2)),
                                       axis=2)
                patch[logical_index, 0] = patch[logical_index, 0] + pixel_offset
                patch[logical_index, 1:] = patch[logical_index, 1:] - pixel_offset
                '''
                for j in range(left, left + patch_size):
                    for k in range(top, top + patch_size):
                        if ((float(img[j][k][0]) + float(img[j][k][1]) + float(img[j][k][2])) / 3 < 254) \
                                and img[j][k][0] == img[j][k][1] == img[j][k][2]:  # 不是白色且没有被上过色
                            img[j][k][0] = min(img[j][k][0] + pixel_offset, 255)
                            img[j][k][1] = max(img[j][k][1] - pixel_offset, 0)
                            img[j][k][2] = max(img[j][k][2] - pixel_offset, 0)

    def show(img1, img2, save_fig, target_dir='./anomalies'):
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.xticks([])
        plt.yticks([])

        if save_fig:
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            plt.savefig(os.path.join(target_dir, img_path1.split('/')[-1][:-6] + '.jpg'), dpi=200)
        else:
            plt.show()

    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    img1, img2 = np.array(img1, dtype=np.uint8), np.array(img2, dtype=np.uint8)
    center_points1, center_points2 = get_center_points(img1), get_center_points(img2)  # 每个patch的中心点(y, x)
    anomaly_patch_index = detector.detect(img1, img2, n=eval(img_path1.split('/')[-1].split('-')[2]),
                                          ret_type='lst')  # 第i个patch是异常的
    inverse1 = True if center_points1[0][0] > center_points1[-1][0] else False
    inverse2 = True if center_points2[0][0] > center_points2[-1][0] else False
    center_points1 = list(reversed(center_points1)) if inverse1 else center_points1
    center_points2 = list(reversed(center_points2)) if inverse2 else center_points2
    draw_anomaly(center_points1, anomaly_patch_index, img1)
    draw_anomaly(center_points2, anomaly_patch_index, img2)
    show(img1, img2, save_fig)


def save_visualization_results():
    root_path = '/home/kms/wangyu_dtst/extracted-yiwei'
    for i in range(10, 22):
        sub_dir = os.path.join(root_path, str(i))
        img_names = os.listdir(sub_dir)
        img_names = sorted(img_names)
        for j in range(len(img_names)):
            if j % 2 == 1:
                continue
            visualize_patchwise_anomaly_detection(os.path.join(sub_dir, img_names[j]),
                                                  os.path.join(sub_dir, img_names[j + 1]))
            print('\rnumber', i, '\t', j, 'finished.', end='')


def experiment_fp_fn_on_practical_dataset():
    """
    根据patch被判定为异常的阈值，以及允许出现异常的块数，进行更多实验。
    """
    patch_thresh_modifier = PatchDetector()
    for patch_thresh in [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]:
        patch_thresh_modifier._save_patch_thresholds_G(fp=patch_thresh)
        for tolarance in [0, 1, 2]:
            fp = patchwise_anomaly_detection_test_G(target='fp', tolarance=tolarance)
            fn = patchwise_anomaly_detection_test_G(target='fn', tolarance=tolarance)
            print('patch thresh:', patch_thresh, '  ', 'tolarance:', tolarance,
                  '\t', 'fp:', fp, '  ', 'fn:', fn)
            print('-' * 30)


def API_test(img_path1='/data/home/cuitor/wangyu/datasets/A_paired_data/A2401150001.001/1/0.png',
             img_path2='/data/home/cuitor/wangyu/datasets/A_paired_data/A2401150001.001/1/1.png',
             n=0,
             ctype='A', ):
    """

    Args:
        img_path1:
        img_path2:
        n: 0 - 22，不包含性染色体
        ctype: 'G', 'A', 'L' 三者之一，其它会报错。

    Returns:
        True or False

    """
    import requests

    with open(img_path1, 'rb') as f1, open(img_path2, 'rb') as f2:
        img1_bytes = f1.read()
        img2_bytes = f2.read()

    num_data = {'n': n, 'ctype': ctype}
    img_files = {'img1': ('image1.jpg', img1_bytes), 'img2': ('image2.jpg', img2_bytes)}

    response = requests.post('http://172.17.0.2:5000/anomaly_detection/', data=num_data, files=img_files)

    result = response.json()['result']  # result是一个布尔值，true代表异常
    return result


def API_test1():
    import requests
    root_path = '/data/home/cuitor/wangyu/datasets/single_data_200k'
    for i in range(22):  # 1到22号染色体
        print('\r', i, '/ 21 on processing..', end='')
        folder_path = os.path.join(root_path, str(i))
        img_names = os.listdir(folder_path)
        img_names = [img_name for img_name in img_names if img_name[0] == 'G']  # 模型只针对低分辨的染色体
        img_names.sort()

        skip_current = False
        len_img_names = len(img_names) - 1
        for num_img in range(len_img_names):
            if skip_current:  # 是一对中的第二章图片
                skip_current = False
                continue

            if img_names[num_img].split('.')[0] != img_names[num_img + 1].split('.')[0] or \
                    img_names[num_img].split('.')[1] != img_names[num_img + 1].split('.')[1]:  # 只有单张
                continue

            img_path1 = os.path.join(folder_path, img_names[num_img])
            img_path2 = os.path.join(folder_path, img_names[num_img + 1])

            with open(img_path1, 'rb') as f1, open(img_path2, 'rb') as f2:
                img1_bytes = f1.read()
                img2_bytes = f2.read()

            num_data = {'n': i}
            img_files = {'img1': ('image1.jpg', img1_bytes), 'img2': ('image2.jpg', img2_bytes)}
            response = requests.post('http://172.17.0.2:5000/anomaly_detection/', data=num_data, files=img_files)

            result = response.json()['result']  # result是一个布尔值，true代表异常
            print(result)


class Paper(object):
    def __init__(self):
        pass

    def get_number_of_data_in_dataset(self):
        root_path = '/data/home/cuitor/wangyu/datasets/cms_patched_single_data_200k'
        for number in sorted(os.listdir(root_path)):
            sub_root_path = os.path.join(root_path, number)
            paired = 0
            for i in range(len(os.listdir(sub_root_path))):
                if os.listdir(sub_root_path)[i][0] == 'G':
                    paired += 1

            print('number', str(eval(number) + 1), ':', len(os.listdir(sub_root_path)), ' ', paired)


    def statistic_similarities_draw(self):
        with open('./patch_normal_similarities_G.json') as f:
            all_similarities = json.load(f)

        number = 0
        patch = 3
        sim_0_0 = all_similarities[str(number)][patch]

        plt.hist(sim_0_0, bins=40, edgecolor='black')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(3)
        plt.gca().spines['bottom'].set_linewidth(3)
        plt.xticks(fontname='Arial', fontsize=15)
        plt.yticks(fontname='Arial', fontsize=15)
        plt.title(f'b={number + 1}, i={patch}')
        plt.show()


if __name__ == '__main__':
    detector = PatchDetector()
    detector.build()
    patchwise_anomaly_detection_test_G(target='fp', tolarance=1)
    patchwise_anomaly_detection_test_G(target='fn', tolarance=1)
