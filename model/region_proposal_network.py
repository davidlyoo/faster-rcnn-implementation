import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN) for Faster R-CNN

    RPN은 입력 Feature Map에서 다양한 크기, 비율의 Anchor를 기반으로
    물체가 있을 법한 영역(Region of Interest, RoI)을 예측한다. 

    Args:
        in_channels (int): 입력 Feature Map의 채널 수
        mid_channels (int): Conv 레이어의 중간 채널 수
        ratios (list of float): Anchor 비율 목록
        anchor_scales (list of int): Anchor 크기 목록
        feat_stride (int): Feature Map과 원본 이미지의 stride
        proposal_creator_params (dict): ProposalCreator에서 사용하는 파라미터들
                # ProposalCreator: NMS 등으로 최종 RoI를 결정하는 모듈

    """

    def __init__(
        self,
        in_channels=512, mid_channels=512,
        ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
        feat_stride=16,
        proposal_creator_params=dict()
        ):

        super(RegionProposalNetwork, self).__init__()
        # 기본 anchor 설정 (y1, x1, y2, x2)
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0] # 각 위치에서 생성되는 anchor 개수
        
        # RPN 레이어 구성
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        
        # 레이어 초기화
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """
        RPN의 순전파 과정
        
        입력 Feature Map x로부터 각 Anchor별 Bounding-box 회귀 값(loc)과 배경/전경 분류 점수(score)를 예측
        ProposalCreator를 통해 최종 RoI 산출

        Args:
            x (torch.Tensor): Backbone CNN에서 출력된 Feature Map
                                (N, C, H, W) 형태
            img_size (tuple): (원본 이미지의 height, width)
            scale (float, optional): 이미지 resize scale
            
        Returns:
            rpn_locs (torch.Tensor): (N, (H*W*n_anchor), 4) 형태의 Bounding-box 회귀 예측 값
            rpn_scores (torch.Tensor): (N, (H*W*n_anchor), 2) 형태의 배경/전경 분류 점수
            rois (np.ndarray): 모든 배치에 Non-Maximum Suppression (NMS)를 거쳐 결정된 RoI 좌표 (y1, x1, y2, x2)
            roi_indices (np.ndarray): rois의 각 index (배치 인덱스)
            anchor (np.ndarray): Feature Map에서 생성된 모든 anchor 좌표 
        """

        # x의 Shape 정보: (N, C, H, W)
        n, _, hh, ww = x.shape
        
        # Feature Map 크기에 맞춰 shift된 anchor 생성
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, hh, ww
        )

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        # bbox 회귀 예측(conv -> permute -> view) - 4k
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # 분류 점수(conv → permute → view) - 2k
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale=scale
            )
            batch_idx = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_idx)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Feature Map 각 위치에서 anchor_base를 얼마나 이동해야 하는지 계산해
    최종적인 anchor 좌표 생성

    Args:
        anchor_base (np.ndarray): 기본 Anchor 배열, (A, 4) 형태로 각 행이 (y1, x1, y2, x2)
        feat_stride (_type_): 몇 픽셀 간격으로 shift할지 / Feature Map 축소 비율
        height (_type_): Feature Map height
        width (_type_): Feature Map width
        
    Returns:
        anchor (np.ndarray): 모든 위치에 Shift된 Anchor 좌표 (K*A, 4)
                            K = H*W, A = anchor_base.shape[0]
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    # shift = (K, 4) 형태 → [shift_y, shift_x, shift_y, shift_x]
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0] # anchor_base 내 anchor 개수: 3x3 = 9 (scale 개수 X ratio 개수)
    K = shift.shape[0]       # 전체 위치 개수 (H * W)

    # anchor_base(1, A, 4) + shift(K, 1, 4) → (K, A, 4) by np broadcasting
    # Feature Map의 각 위치(K)에 대해 anchor_base의 A개 좌표를 모두 더함: (K X A)개의 Anchor를 한 번에 계산
    anchor = anchor_base.reshape((1, A, 4)) + \
            shift.reshape((1, K, 4)).transpose((1, 0, 2))

    # (K*A, 4) 형태로 Flatten
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    레이어에 대해 weight/bias 초기화 수행

    Args:
        m (nn.Module): 초기화할 레이어
        mean (float): 정규분포 평균
        stddev (float): 정규분포 표준편차
        truncated (bool, optional): True일 경우, 2표준편차 범위를 벗어난 값은 재샘플링
    """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()




