import os
import warnings
import logging

import torchvision.transforms

import time
import json
import typing
import io

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torchvision import transforms

from cpc_models.ResNetV2_Encoder import PreActResNetN_Encoder
from cpc_models.PixelCNN_GIM import PixelCNN
from cpc_models.CPC import CPC
from argparser.train_CPC_argparser import argparser
from data.image_preprocessing import CMS_Trans_for_inference
from data.data_handlers import aug
from data.cms_patch.SKELETONlib.MySkeletonize import get_skeleton
from data.cms_patch.SKELETONlib.Graph import Graph

from flask import Flask, request, jsonify


class Detector(object):
    def __init__(self):
        self.args = argparser()
        self.model_path = os.path.join('./TrainedModels', self.args.dataset,
                                       f'trained_encoder_{self.args.encoder}_crop{self.args.crop}'
                                       f'{"_colour" if (not self.args.gray) else ""}_grid{self.args.grid_size}_'
                                       f'{self.args.norm}Norm_{self.args.pred_directions}dir_aug'
                                       f'{self.args.patch_aug}_{self.args.epochs}{self.args.model_name_ext}.pt')
        self.normal_similarities_path = './normal_similarities.json'
        self.thresholds_path = './similarity_threshold.json'

        self.encoder: torch.nn.Module = None
        self.trans: torchvision.transforms.Compose = None
        self.use_gpu = True

    def build(self, use_gpu=True):
        """
        need to execute this function right after create an object of class Detector
        :param use_gpu: whether deploy the model to GPU
        :return: None
        """
        trans = transforms.Compose([CMS_Trans_for_inference(self.args, eval=True, aug=aug[self.args.dataset])])
        if self.args.encoder[:6] == 'resnet':
            encoder = PreActResNetN_Encoder(self.args, use_classifier=False)
        elif self.args.encoder == 'vit':
            encoder = ViT_Encoder(self.args, use_classifier=False)
        else:
            raise ValueError('Unexpected encoder name.')

        # load trained model
        if os.path.exists(self.model_path):
            encoder.load_state_dict(torch.load(self.model_path))
            print('load pretrained encoder')
        else:
            warnings.warn('Invalid path of trained encoder!!')
        if use_gpu:
            encoder.to('cuda')
        encoder.eval()

        self.trans = trans
        self.encoder = encoder
        self.use_gpu = use_gpu

    def encode(self, x, use_gpu=True):
        """

        :param use_gpu:
        :param x:list of images (images is read by PIL.Image.open) or a single image
        :return: latent representation of the image
        """
        if self.encoder is None or self.trans is None:
            raise BrokenPipeError('Please build the model first using Detector.build()')

        if use_gpu != self.use_gpu:
            raise ValueError(f'When building model, gpu is {"" if self.use_gpu else "not"} used.'
                             f'inconsistance gpu specification detected.')

        if hasattr(x, '__iter__'):
            x1 = []
            for img in x:
                x1.append(self.trans(img))
            x1 = torch.cat(x1, dim=0)
        else:
            x1 = self.trans(x)

        if use_gpu:
            x1 = x1.to('cuda')

        x1 = self.encoder(x1)
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[3])  # shape: (batch_size, 13, 1024)

        return x1

    def _cosine_similarity_from_feature(self, feature1, feature2):
        dot_products = np.sum(feature1 * feature2, axis=1)
        norms_feature1 = np.sqrt(np.sum(feature1 ** 2, axis=1))
        norms_feature2 = np.sqrt(np.sum(feature2 ** 2, axis=1))
        cosine_similarities = dot_products / (norms_feature1 * norms_feature2)
        # 染色体patch后面的白的patch的相似度接近1，没有计算意义.
        for i in range(len(cosine_similarities) - 1, -1, -1):
            if cosine_similarities[i] > 0.99:
                cosine_similarities = np.delete(cosine_similarities, i)
            else:
                break

        # 如果两根染色体完全相同（显然不应该这样），则返回-1（异常的相似度值）
        if cosine_similarities.size == 0:
            return -0.999
        else:
            return np.mean(cosine_similarities)

    def _save_similarity_between_normal_pair(self):
        # todo 注意染色体分为G L（高分辨 低分辨）两种。
        root_path = '/home/kms/wangyu_dtst/single_data_200k/single_data_200k'
        target_file = self.normal_similarities_path

        total_lst = []
        for i in range(22):  # X Y not included.
            total_lst.append([])
        finished_count = 0

        for i in range(0, 22):
            sub_folder_path = os.path.join(root_path, str(i))
            # 得到数据集中所有以G开头的染色体
            g_chromo = []
            for img_name in os.listdir(sub_folder_path):
                if img_name[0] == 'G':
                    g_chromo.append(img_name)

            # 不一定所有染色体都成对出现，所以这里需设置逻辑跳过单根的染色体
            paired = False
            g_chromo.sort()
            for j in range(len(g_chromo) - 1):
                if paired:
                    paired = False
                    continue
                else:
                    cur_prefix = g_chromo[j].split('.')[0:2]
                    nxt_prefix = g_chromo[j + 1].split('.')[0:2]
                    if cur_prefix == nxt_prefix:
                        paired = True
                        # 找到一对G染色体，存储他们之间的相似度。
                        img1 = Image.open(os.path.join(sub_folder_path, g_chromo[j]))
                        img2 = Image.open(os.path.join(sub_folder_path, g_chromo[j + 1]))
                        try:
                            encodings = self.encode([img1, img2])
                        except Exception as e:
                            continue
                        encodings = encodings.detach().cpu().numpy()
                        feature1, feature2 = encodings[0], encodings[1]
                        total_lst[i].append(float(self._cosine_similarity_from_feature(feature1, feature2)))
                        finished_count += 1
                        print('\r', finished_count, 'pairs finished..', end='')

        res = {}
        for i in range(22):
            res[str(i)] = total_lst[i]

        if os.path.exists(target_file):
            os.remove(target_file)
        with open(target_file, 'w') as f:
            json.dump(res, f)
        print('\nDone')

    def _plot_normal_similarities(self):
        import matplotlib.pyplot as plt
        if not os.path.exists(self.normal_similarities_path):
            print('no normal similarities counted.')
            return

        with open(self.normal_similarities_path, 'r') as f:
            res = json.load(f)
        plt.hist(res[str(1)], bins=20)
        plt.show()

    def _save_calculated_threshold(self, fp=0.2):
        """可以根据fp（假阳性率）确定一个合适的阈值，阈值之下的样本即被判断为阳性。"""
        if not os.path.exists(self.normal_similarities_path):
            print('no normal similarities counted.')
            return
        assert 0 <= fp < 1

        with open(self.normal_similarities_path, 'r') as f:
            all_similarities = json.load(f)
        res = {'fp': fp}

        for i in range(22):
            similarities = all_similarities[str(i)]
            similarities.sort()
            threshold = similarities[int(len(similarities) * fp)]
            res[str(i)] = threshold

        if os.path.exists(self.thresholds_path):
            os.remove(self.thresholds_path)

        with open(self.thresholds_path, 'w') as f:
            json.dump(res, f)

    def detect(self, img1, img2, n, use_gpu=True):
        """

        :param use_gpu: must be the same as in "build" function
        :param img1: image from PIL.Image.open(path/to/image). not image path
        :param img2: image. not image path
        :param n:
        :return:
        """
        if not os.path.exists(self.thresholds_path):
            raise FileNotFoundError('missing file (thresholds): ' + self.thresholds_path)

        try:
            encodings = self.encode([img1, img2], use_gpu=use_gpu)
        except Exception as e:
            raise ValueError('image type not supported')

        if use_gpu:
            encodings = encodings.detach().cpu().numpy()
        else:
            encodings = encodings.detach().numpy()

        # 得到图像特征，判断相似度
        feature1, feature2 = encodings[0], encodings[1]
        similarity = self._cosine_similarity_from_feature(feature1, feature2)

        # 根据相似度及其阈值判定是否异常
        with open(self.thresholds_path, 'r') as f:
            thresholds = json.load(f)
        if similarity <= thresholds[str(n)]:
            return True
        return False


class PatchDetector(object):
    def __init__(self):
        self.args = argparser()
        self.model_path = os.path.join('./TrainedModels', self.args.dataset,
                                       f'trained_encoder_{self.args.encoder}_crop{self.args.crop}'
                                       f'{"_colour" if (not self.args.gray) else ""}_grid{self.args.grid_size}_'
                                       f'{self.args.norm}Norm_{self.args.pred_directions}dir_aug'
                                       f'{self.args.patch_aug}_{self.args.epochs}{self.args.model_name_ext}.pt')
        self.full_CPC_path = os.path.join('./TrainedModels', self.args.dataset,
                                          f'trained_cpc_{self.args.encoder}_crop{self.args.crop}'
                                          f'{"_colour" if (not self.args.gray) else ""}_grid{self.args.grid_size}_'
                                          f'{self.args.norm}Norm_{self.args.pred_directions}dir_aug'
                                          f'{self.args.patch_aug}_{self.args.epochs}{self.args.model_name_ext}.pt')

        self.normal_similarities_path_G = './patch_normal_similarities_G.json'
        self.thresholds_path_G = './patch_similarity_threshold_G.json'
        self.normal_similarities_path_A = './patch_normal_similarities_A.json'
        self.thresholds_path_A = './patch_similarity_threshold_A.json'
        self.normal_similarities_path_L = './patch_normal_similarities_L.json'
        self.thresholds_path_L = './patch_similarity_threshold_L.json'

        self.encoder: torch.nn.Module = None
        self.auto_regressor: torch.nn.Module = None
        self.trans: torchvision.transforms.Compose = None
        self.use_gpu = True
        self.cuda_num = '1'

    def build(self, use_gpu=True):
        """
        need to execute this function right after create an object of class Detector
        :param use_gpu: whether deploy the model to GPU
        :return: None
        """
        self.trans = transforms.Compose([CMS_Trans_for_inference(self.args, eval=True, aug=aug[self.args.dataset])])
        if self.args.encoder[:6] == 'resnet':
            self.encoder = PreActResNetN_Encoder(self.args, use_classifier=False)
            self.auto_regressor = PixelCNN(self.encoder.encoding_size)
        else:
            raise ValueError('Unexpected encoder name.')

        # load trained model
        if os.path.exists(self.model_path) and os.path.exists(self.full_CPC_path):
            if not use_gpu:  # 将模型加载到CPU上
                self.encoder.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
                trained_dict = torch.load(self.full_CPC_path, map_location=lambda storage, loc: storage)
                trained_dict = {'.'.join(k.split('.')[1:]): v for k, v in trained_dict.items() if
                                '.'.join(k.split('.')[1:]) in self.auto_regressor.state_dict()}
                self.auto_regressor.load_state_dict(trained_dict)
            else:
                self.encoder.load_state_dict(torch.load(self.model_path))
                trained_dict = torch.load(self.full_CPC_path)
                trained_dict = {'.'.join(k.split('.')[1:]): v for k, v in trained_dict.items() if
                                '.'.join(k.split('.')[1:]) in self.auto_regressor.state_dict()}
                self.auto_regressor.load_state_dict(trained_dict)
        else:
            warnings.warn('Invalid path of trained encoder!!')
        if use_gpu:
            self.encoder.to('cuda')
            self.auto_regressor.to('cuda')
        self.encoder.eval()
        self.auto_regressor.eval()
        self.use_gpu = use_gpu

    def _encode(self, x, use_gpu=True):
        """

        :param use_gpu:
        :param x:list of images (images is read by PIL.Image.open) or a single image
        :return: latent representation of the image
        """
        if self.encoder is None or self.trans is None:
            raise BrokenPipeError('Please build the model first using Detector.build()')

        if use_gpu != self.use_gpu:
            raise ValueError(f'When building model, gpu is {"" if self.use_gpu else "not"} used.'
                             f'inconsistance gpu specification detected.')

        if hasattr(x, '__iter__'):
            x1 = []
            for img in x:
                x1.append(self.trans(img))
            x1 = torch.cat(x1, dim=0)
        else:
            x1 = self.trans(x)

        if use_gpu:
            x1 = x1.to('cuda')

        with torch.no_grad():
            x1 = self.encoder(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = self.auto_regressor(x1)
            x1 = x1.permute(0, 2, 1, 3)
        x1 = x1.view(x1.shape[0], x1.shape[1], x1.shape[2])  # shape: (batch_size, 13, encoding_size)

        return x1

    def _cosine_similarity_from_feature(self, feature1, feature2):
        dot_products = np.sum(feature1 * feature2, axis=1)
        norms_feature1 = np.sqrt(np.sum(feature1 ** 2, axis=1))
        norms_feature2 = np.sqrt(np.sum(feature2 ** 2, axis=1))
        cosine_similarities = dot_products / (norms_feature1 * norms_feature2)

        return cosine_similarities

    def _save_similarities_between_normal_patch_G(self):
        root_path = '/data/home/cuitor/wangyu/datasets/single_data_200k'
        target_file = self.normal_similarities_path_G

        total_lst = []
        for i in range(22):  # X Y not included.
            total_lst.append([])
            for j in range(self.args.grid_size):
                total_lst[i].append([])
        finished_count = 0

        for i in range(0, 22):
            sub_folder_path = os.path.join(root_path, str(i))
            # 得到数据集中所有以G开头的ssh -L 16006:127.0.0.1:6006 -i ~/.ssh/kms01.pem -p 2201 kms@lab.kemoshen.com染色体
            g_chromo = []
            for img_name in os.listdir(sub_folder_path):
                if img_name[0] == 'G':
                    g_chromo.append(img_name)

            # 不一定所有染色体都成对出现，所以这里需设置逻辑跳过单根的染色体
            paired = False
            g_chromo.sort()
            for j in range(len(g_chromo) - 1):
                if paired:
                    paired = False
                    continue
                else:
                    cur_prefix = g_chromo[j].split('.')[0:2]
                    nxt_prefix = g_chromo[j + 1].split('.')[0:2]
                    if cur_prefix == nxt_prefix:
                        paired = True
                        # 找到一对G染色体，存储他们之间的相似度。
                        img1 = Image.open(os.path.join(sub_folder_path, g_chromo[j]))
                        img2 = Image.open(os.path.join(sub_folder_path, g_chromo[j + 1]))
                        try:
                            encodings = self._encode([img1, img2])
                        except Exception as e:
                            continue
                        encodings = encodings.detach().cpu().numpy()
                        feature1, feature2 = encodings[0], encodings[1]
                        similarities = (self._cosine_similarity_from_feature(feature1, feature2))
                        for k in range(self.args.grid_size):
                            total_lst[i][k].append(round(float(similarities[k]), 8))
                        finished_count += 1
                        print('\r', finished_count, 'pairs finished..', end='')

        res = {}
        for i in range(22):
            res[str(i)] = total_lst[i]

        if os.path.exists(target_file):
            os.remove(target_file)
        with open(target_file, 'w') as f:
            json.dump(res, f)
        print('\nDone')

    def _save_patch_thresholds_G(self, fp=0.075):
        if not os.path.exists(self.normal_similarities_path_G):
            print('no normal similarities counted.')
            return
        assert 0 <= fp < 1

        with open(self.normal_similarities_path_G, 'r') as f:
            all_similarities = json.load(f)
        res = {'fp': fp}

        total_lst = []
        for i in range(22):  # X Y not included.
            total_lst.append([])

        for i in range(22):
            for j in range(self.args.grid_size):
                similarities = all_similarities[str(i)][j]
                similarities.sort()
                similarities = [x for x in similarities]  # todo post-process may be needed.
                threshold = round(similarities[int(len(similarities) * fp)], 6) if similarities else 1.000
                total_lst[i].append(threshold)

        for i in range(22):
            res[str(i)] = total_lst[i]

        if os.path.exists(self.thresholds_path_G):
            os.remove(self.thresholds_path_G)
        with open(self.thresholds_path_G, 'w') as f:
            json.dump(res, f)

    def _save_similarities_between_normal_patch_A(self):
        root_path = '/data/home/cuitor/wangyu/datasets/A_paired_data'
        # 22 numbers of chromsomes & 13 patches for each number
        all_similarities = [[[] for i in range(self.args.grid_size)] for j in range(22)]

        finished_case_count = 0
        total_cases_count = len(os.listdir(root_path))

        for case_name in os.listdir(root_path):
            case_path = os.path.join(root_path, case_name)

            for number in range(0, 22):
                cur_number_path = os.path.join(case_path, str(number))

                if not (os.path.exists(os.path.join(cur_number_path, '0.png')) and
                        os.path.exists(os.path.join(cur_number_path, '1.png'))):
                    continue

                img1 = Image.open(os.path.join(cur_number_path, '0.png'))
                img2 = Image.open(os.path.join(cur_number_path, '1.png'))

                try:
                    encodings = self._encode([img1, img2])
                except Exception as e:
                    continue

                encodings = encodings.detach().cpu().numpy()
                feature1, feature2 = encodings[0], encodings[1]
                similarities = (self._cosine_similarity_from_feature(feature1, feature2))

                for i in range(self.args.grid_size):
                    all_similarities[number][i].append(round(float(similarities[i]), ndigits=8))

            finished_case_count += 1
            print('\r', finished_case_count, '/', total_cases_count, 'finished.')

        res = {}
        for i in range(22):
            res[str(i)] = all_similarities[i]

        if os.path.exists(self.normal_similarities_path_A):
            os.remove(self.normal_similarities_path_A)
        with open(self.normal_similarities_path_A, 'w') as f:
            json.dump(res, f)
        print('\nDone')

    def _save_patch_thresholds_A(self, fp=0.075):
        if not os.path.exists(self.normal_similarities_path_A):
            print('no normal similarities counted.')
            return
        assert 0 <= fp < 1

        with open(self.normal_similarities_path_A, 'r') as f:
            all_similarities = json.load(f)
        res = {'fp': fp}

        total_lst = []
        for i in range(22):  # X Y not included.
            total_lst.append([])

        for i in range(22):
            for j in range(self.args.grid_size):
                similarities = all_similarities[str(i)][j]
                similarities.sort()
                similarities = [x for x in similarities]  # todo post-process may be needed.
                threshold = round(similarities[int(len(similarities) * fp)], 6) if similarities else 1.000
                total_lst[i].append(threshold)

        for i in range(22):
            res[str(i)] = total_lst[i]

        if os.path.exists(self.thresholds_path_A):
            os.remove(self.thresholds_path_A)
        with open(self.thresholds_path_A, 'w') as f:
            json.dump(res, f)

    def _save_similarities_between_normal_patch_L(self):
        root_path = '/data/home/cuitor/wangyu/datasets/L_paired_data'
        # 22 numbers of chromsomes & 13 patches for each number
        all_similarities = [[[] for i in range(self.args.grid_size)] for j in range(22)]

        finished_case_count = 0
        total_cases_count = len(os.listdir(root_path))

        for case_name in os.listdir(root_path):
            case_path = os.path.join(root_path, case_name)

            for number in range(0, 22):
                cur_number_path = os.path.join(case_path, str(number))

                if not (os.path.exists(os.path.join(cur_number_path, '0.png')) and
                        os.path.exists(os.path.join(cur_number_path, '1.png'))):
                    continue

                img1 = Image.open(os.path.join(cur_number_path, '0.png'))
                img2 = Image.open(os.path.join(cur_number_path, '1.png'))

                try:
                    encodings = self._encode([img1, img2])
                except Exception as e:
                    continue

                encodings = encodings.detach().cpu().numpy()
                feature1, feature2 = encodings[0], encodings[1]
                similarities = (self._cosine_similarity_from_feature(feature1, feature2))

                for i in range(self.args.grid_size):
                    all_similarities[number][i].append(round(float(similarities[i]), ndigits=8))

            finished_case_count += 1
            print('\r', finished_case_count, '/', total_cases_count, 'finished.')

        res = {}
        for i in range(22):
            res[str(i)] = all_similarities[i]

        if os.path.exists(self.normal_similarities_path_L):
            os.remove(self.normal_similarities_path_L)
        with open(self.normal_similarities_path_L, 'w') as f:
            json.dump(res, f)
        print('\nDone')

    def _save_patch_thresholds_L(self, fp=0.075):
        if not os.path.exists(self.normal_similarities_path_L):
            print('no normal similarities counted.')
            return
        assert 0 <= fp < 1

        with open(self.normal_similarities_path_L, 'r') as f:
            all_similarities = json.load(f)
        res = {'fp': fp}

        total_lst = []
        for i in range(22):  # X Y not included.
            total_lst.append([])

        for i in range(22):
            for j in range(self.args.grid_size):
                similarities = all_similarities[str(i)][j]
                similarities.sort()
                similarities = [x for x in similarities]  # todo post-process may be needed.
                threshold = round(similarities[int(len(similarities) * fp)], 6) if similarities else 1.000
                total_lst[i].append(threshold)

        for i in range(22):
            res[str(i)] = total_lst[i]

        if os.path.exists(self.thresholds_path_L):
            os.remove(self.thresholds_path_L)
        with open(self.thresholds_path_L, 'w') as f:
            json.dump(res, f)

    def _get_center_points(self, img, patch_size=32):
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

    def _draw_anomaly_patch(self, center_points, anomaly_patch_index, img,
                            pixel_offset=30, cms_gray_threshold=250, patch_size=32):
        for i in range(len(center_points)):  # 对于每个patch
            if i not in anomaly_patch_index:
                continue
            left = center_points[i][0] - int(patch_size / 2)
            top = center_points[i][1] - int(patch_size / 2)
            # 将不是白色的像素点变得红一些，以提示异常
            patch = img[left:left + patch_size, top:top + patch_size].astype(np.float32)  # patch: 32*32*3

            equal_values_mask = (patch[:, :, 0] == patch[:, :, 1]) & (patch[:, :, 1] == patch[:, :, 2])
            smaller_than_thresh_mask = ((patch[:, :, 0] + patch[:, :, 1] + patch[:, :, 2]) / 3 < cms_gray_threshold)
            target_area = equal_values_mask & smaller_than_thresh_mask
            patch[target_area, 0] += pixel_offset
            patch[target_area, 1] -= pixel_offset
            patch[target_area, 2] -= pixel_offset

            patch[patch > 255] = 255
            patch[patch < 0] = 0

            img[left:left + patch_size, top:top + patch_size] = patch.astype(np.uint8)

            '''
            for j in range(left, left + patch_size):
                for k in range(top, top + patch_size):
                    if ((float(img[j][k][0]) + float(img[j][k][1]) + float(img[j][k][2])) / 3 < cms_gray_threshold) \
                            and img[j][k][0] == img[j][k][1] == img[j][k][2]:  # 不是白色且没有被上过色
                        img[j][k][0] = min(img[j][k][0] + pixel_offset, 255)
                        img[j][k][1] = max(img[j][k][1] - pixel_offset, 0)
                        img[j][k][2] = max(img[j][k][2] - pixel_offset, 0)'''

    def _draw_anomaly(self, img1, img2, anomalies):
        img1, img2 = np.array(img1, dtype=np.uint8), np.array(img2, dtype=np.uint8)
        center_points1, center_points2 = self._get_center_points(img1), self._get_center_points(img2)
        anomaly_patch_index = anomalies
        inverse1 = True if center_points1[0][0] > center_points1[-1][0] else False
        inverse2 = True if center_points2[0][0] > center_points2[-1][0] else False
        center_points1 = list(reversed(center_points1)) if inverse1 else center_points1
        center_points2 = list(reversed(center_points2)) if inverse2 else center_points2
        self._draw_anomaly_patch(center_points1, anomaly_patch_index, img1)
        self._draw_anomaly_patch(center_points2, anomaly_patch_index, img2)
        return img1, img2

    def _detect(self, img1, img2, n, ctype, use_gpu=True, ret_type='img'):
        """

        :param ret_type: 'lst' for a list of indexes of anomaly patches; 'img' for two processed img.
        :param use_gpu: must be the same as in "build" function
        :param img1: image. not image path
        :param img2: image. not image path
        :param ctype: 'A' or 'G' or 'L'
        :param n: number of chromsome
        :return: ref :param ret_type
        """
        if not os.path.exists(self.thresholds_path_G):
            raise FileNotFoundError('missing file (thresholds): ' + self.thresholds_path_G)

        encodings1 = self._encode(img1, use_gpu=use_gpu)
        encodings2 = self._encode(img2, use_gpu=use_gpu)

        if use_gpu:
            encodings1 = encodings1.detach().cpu().numpy()
            encodings2 = encodings2.detach().cpu().numpy()
        else:
            encodings1 = encodings1.detach().numpy()
            encodings2 = encodings2.detach().numpy()

        # 得到图像特征，判断每个patch的相似度
        similarity = self._cosine_similarity_from_feature(encodings1[0], encodings2[0])

        # G、A、L三类染色体有不同的统计学规律。
        if ctype == 'G':
            with open(self.thresholds_path_G, 'r') as f:
                thresholds = json.load(f)
        elif ctype == 'L':
            with open(self.thresholds_path_L, 'r') as f:
                thresholds = json.load(f)
        elif ctype == 'A':
            with open(self.thresholds_path_A, 'r') as f:
                thresholds = json.load(f)
        else:
            raise ValueError('not a valid choice for parameter "ctype".')

        # 逐个patch进行检测，并把异常patch的序号添加到结果。
        anomalies = [i for i in range(self.args.grid_size) if round(similarity[i], 6) < thresholds[str(n)][i]]

        if ret_type == 'lst':
            return anomalies
        elif ret_type == 'img':
            ret = self._draw_anomaly(img1, img2, anomalies)
            return ret
        else:
            raise ValueError('unexpected param value: ret_type')

    def detect(self, img1, img2, n, ctype, use_gpu=True):
        '''

        Args:
            img1: ndarray, image
            img2: ndarray, image
            n: int, number of chromosome
            ctype: str, 'A' or 'G' or 'L'
            use_gpu: boolean, whether to use gpu or not.

        Returns:
            A boolean value. True for anomaly and False for normal.
        '''
        try:
            res = self._detect(img1, img2, n, ctype, ret_type='lst')
        except Exception as e:
            return True

        if len(res) <= 1:
            return False
        else:
            return True


app = Flask(__name__)
detector = PatchDetector()


@app.route('/anomaly_detection/', methods=['POST'])
def detect():
    global detector
    n = int(request.form['n'])
    ctype = str(request.form['ctype'])

    img1_file = request.files['img1']
    img2_file = request.files['img2']

    img1 = Image.open(io.BytesIO(img1_file.read()))
    img2 = Image.open(io.BytesIO(img2_file.read()))

    result = detector.detect(img1, img2, n, ctype)
    return {'result': result}


if __name__ == '__main__':
    detector.build()
    app.run(host='0.0.0.0', port=5000)
