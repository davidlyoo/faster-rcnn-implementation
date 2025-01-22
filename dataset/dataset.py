from __future__ import  absolute_import, division

import numpy as np
import torch
from torchvision import transforms as tvtsf
from skimage import transform as sktsf

from dataset.voc_dataset import VOCBboxDataset
from dataset import util
from utils.config import opt


# 데이터를 시각화할 때 사용되는 역정규화 함수.
def inverse_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


# PyTorch 표준 정규화 방식으로 이미지를 정규화.
def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


# 이미지 전처리 함수.
def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)

    # pytorch 정규화
    normalize = pytorch_normalize

    return normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # 수평 뒤집기 적용
        img, params = util.random_flip(img, x_random=True, return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, return_difficult=True)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))

        # Negative stride 확인 후 필요할 때만 복사
        if img.strides[1] < 0 or img.strides[2] < 0:
            img = img.copy()
        if bbox.strides[0] < 0:
            bbox = bbox.copy()
        if label.strides[0] < 0:
            label = label.copy()

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult