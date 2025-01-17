from __future__ import absolute_import, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from dataset.dataset import preprocess
from utils import array_tool as at
from utils.config import opt
from model.utils.bbox_tools import loc2bbox


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f


class FasterRCNN(nn.Module):
    """
    Faster R-CNN 메인 모델 클래스
    
    Backbone Feature Extractor, RPN, Head를 조합해 객체 탐지 수행
    : BBox regression + Classification

    Args:
        extractor (nn.Module): Backbone Feature Extractor
        rpn (nn.Module): Region Proposal Network
        head (nn.Module): Fast R-CNN Head
        loc_normalize_mean (tuple): BBox 회귀의 평균
        loc_normalize_std (tuple): BBox 회귀의 표준편차
    """

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std= (0.1, 0.1, 0.2, 0.2)
                 ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        """
        Head에서 정의한 클래스 수를 받아옴 (배경 포함)
        """
        return self.head.n_class

    def forward(self, x, scale=1.):
        """
        Faster R-CNN의 순전파
        1) Feature Extractor를 통해 Feature Map 추출
        2) RPN을 거쳐 Proposal(RoI) 생성
        3) RoI + Feature Map을 Head에 전달 -> 최종 cls_loc, cls_score 산출

        Args:
            x (torch.Tensor): 입력 이미지 (N, C, H, W) 형태
            scale (float, optional): 이미지 resize 비율

        Returns:
            roi_cls_locs (torch.Tensor): (R, n_class * 4) 형태, RoI별 클래스별 bbox regression
            roi_scores (torch.Tensor): (R, n_class) 형태, RoI별 클래스 점수
            rois (np.ndarray): (R, 4) 형태, Proposal들의 좌표
            roi_indices (np.ndarray): (R,) 배치 인덱스
        """
        img_size = x.shape[2:] # (H, W)

        # 1) Feature Extractor
        h = self.extractor(x)

        # 2) RPN -> (loc, score, rois, roi_indices, anchor)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)

        # 3) Head
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices)
        
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):
        """
        NMS 및 score에 대한 threshold 설정

        Args:
            preset (str): 'visualize' or 'evaluate'
        """
        # 노이즈 제거, 시각화에 좋게 설정
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset은 visualize, evaluate 둘 중에 하나여야 합니다.')

    def _suppress(self, raw_cls_bbox, raw_prob):
        """
        NMS 등으로 최종 필터링 (BBox, label, score)

        Args:
            raw_cls_bbox (torch.Tensor or np.ndarray):
                (R, n_class*4) 형태, 모든 RoI에 대한 bbox 예측 결과.
            raw_prob (torch.Tensor or np.ndarray):
                (R, n_class) 형태, 모든 RoI에 대한 클래스 점수.

        Returns:
            bbox (np.ndarray): 최종 NMS 후 BBox 좌표
            label (np.ndarray): 최종 label
            score (np.ndarray): 최종 score
        """
        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]

            # score threshold 필터링
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            
            # NMS 수행 후 NMS 결과만 모으기
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score
    
    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        """
        실제 이미지에 대한 Detection 수행

        Args:
            imgs (list or np.ndarray): 입력 이미지 리스트 혹은 단일 이미지
            sizes (list of tuple, optional): 각 이미지의 (H, W), None이면 내부 계산
            visualize (bool, optional): visualize 모드인지 여부

        Returns:
            bboxes (list of np.ndarray): 각 이미지별 탐지된 BBox들
            labels (list of np.ndarray): 각 이미지별 Label들
            scores (list of np.ndarray): 각 이미지별 Score들
        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            
            # Faster R-CNN forward
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            roi_cls_loc = roi_cls_loc.data
            roi_scores = roi_scores.data
            roi = at.totensor(rois) / scale

            # BBox 회귀 값 복원
            mean = torch.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)

            # loc2bbox 통해 최종 예측 박스 좌표 - 실제 좌표로 De-normalize
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            
            # 이미지 경계 넘지 않도록 clamp(제한)
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2].clamp(min=0, max=size[0]))
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])
            
            prob = (F.softmax(at.totensor(roi_scores), dim=1))
            
            # NMS + score threshold
            bbox, label, score = self._suppress(cls_bbox, prob)
            
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
            
        self.use_preset('evaluate')
        self.train()
        
        return bboxes, labels, scores
    
    def get_optimizer(self):
        """_summary_
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
            
        return self.optimizer