from __future__ import absolute_import

import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


# VGG-16 모델을 분해하여 feature extractor와 classifier를 반환하는 함수
def decom_vgg16():
    model = vgg16(pretrained=not opt.load_path)

    # Feature Extractor 설정 (Conv5_3까지 사용)
    features = list(model.features)[:30]
    classifier = list(model.classifier)

    # 마지막 분류 레이어 제거
    del classifier[6]

    # Dropout 레이어 제거 (옵션 적용)
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]

    classifier = nn.Sequential(*classifier)

    # Conv1~Conv4까지의 가중치를 고정 (Freeze)
    for layer in features[:10]:
        for param in layer.parameters():
            param.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """
    VGG-16을 기반으로 한 Faster R-CNN 모델

    Args:
        n_fg_class (int): 배경을 제외한 클래스 개수
        ratios (list of floats): Anchor의 가로세로 비율
        anchor_scales (list of numbers): Anchor의 크기 (스케일)
    """

    feat_stride = 16  # VGG-16 Conv5 레이어의 다운샘플링 비율 (16배 축소)

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,  # 배경 클래스 포함
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)


class VGG16RoIHead(nn.Module):
    """
    VGG-16 기반 Faster R-CNN의 RoI Head 모듈

    Args:
        n_class (int): 배경 포함 클래스 개수
        roi_size (int): RoI-Pooling 이후 특징 맵 크기
        spatial_scale (float): RoI 크기 조정 비율
        classifier (nn.Module): VGG-16 분류기 (FC Layer)
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier # VGG16의 FCL 사용
        self.cls_loc = nn.Linear(4096, n_class * 4)  # Bounding-box Regression (Localization)
        self.score = nn.Linear(4096, n_class)  # Class 점수 예측 (Softmax Classification)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """
        RoI Head의 순전파 (Forward Propagation)

        Args:
            x (torch.Tensor): 입력 특징 맵
            rois (torch.Tensor): RoI 바운딩 박스 좌표 (N, 4)
            roi_indices (torch.Tensor): RoI가 속한 이미지 인덱스 (N,)

        Returns:
            roi_cls_locs (torch.Tensor): RoI 별 Bounding-box 위치 예측값
            roi_scores (torch.Tensor): RoI 별 클래스 분류 점수
        """
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)

        # 좌표 순서 변환 (YX → XY)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        # RoI Pooling 적용
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)

        # Fully Connected Layers 통과
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


# 가중치 초기화 함수
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
