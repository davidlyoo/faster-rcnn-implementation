from __future__ import absolute_import

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnet.meter.ConfusionMeter, AverageValueMeter
from collections import namedtuple

from utils import array_tool as at
from utils.vis_tool import Visualizer
from utils.config import opt
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """
    Faster R-CNN 훈련을 위한 래퍼 클래스
    Faster R-CNN 모델을 훈련, 다음과 같은 loss를 계산합니다.

    * `rpn_loc_loss`: RPN에서의 Localization loss
    * `rpn_cls_loss`: RPN에서의 Classification loss
    * `roi_loc_loss`: RoI에서의 위치 loss
    * `roi_cls_loss`: RoI에서의 Classification loss
    * `total_loss`: 위 네 가지 손실의 총합
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # Target 생성기 (Ground Truth와 매칭되는 바운딩 박스와 레이블 생성)
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        self.vis = Visualizer(env=opt.env)

        # 훈련 상태를 나타내는 지표들 (Confusion Matrix, Average Loss)
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, imgs, bboxes, labels, scale):
        """
        Faster R-CNN을 실행하고 loss를 계산하는 함수

        Args:
            imgs (torch.Tensor): 배치 이미지 데이터
            bboxes (torch.Tensor): 배치 바운딩 박스 데이터
            labels (torch.Tensor): 배치 라벨 데이터
            scale (float): 입력 이미지의 스케일

        Returns:
            LossTuple: 5가지 손실값을 포함한 namedtuple
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('배치 크기 1만 지원됩니다.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # 샘플 RoI 생성
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std
        )

        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index
        )

        # ------------------ RPN 손실 계산 -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size
        )
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)

        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma
        )

        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        self.rpn_cm.add(at.totensor(rpn_score, False), gt_rpn_label.data.long())

        # ------------------ ROI 손실 계산 -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(),
                                at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma
        )

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses.append(sum(losses))

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        
        return losses

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    
    return loc_loss
