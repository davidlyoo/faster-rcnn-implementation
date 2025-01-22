import numpy as np

import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox


class ProposalTargetCreator:
    """
    Faster R-CNN 학습용:
        주어진 RoI에 대해, 실제 BBox(ground truth) 정보를 할당하고
        전경/배경으로 샘플링한 뒤 학습에 필요한 오프셋(loc)과 라벨 생성
    
    Args:
        n_sample (int): 한 번에 샘플링할 RoI 수 (전경+배경)
        pos_ratio (float): 전경 RoI 비율(예: 0.25)
        pos_iou_thresh (float): IoU가 이 값 이상이면 전경으로 간주
        neg_iou_thresh_hi (float): IoU가 이 값 미만이면 배경 후보
        neg_iou_thresh_lo (float): 배경으로 인정할 때, IoU가 이 값 이상이도록 제한
    """

    def __init__(
        self,
        n_sample=128,
        pos_ratio=0.25,
        pos_iou_thresh=0.5,
        neg_iou_thresh_hi=0.5,
        neg_iou_thresh_lo=0.0
    ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo


    def __call__(
        self,
        roi,
        bbox,
        label,
        loc_normalize_mean=(0., 0., 0., 0.),
        loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
    ):
        """
        최종적으로 (sampled_roi, gt_roi_loc, gt_roi_label) 반환

        Args:
            roi (ndarray): (R, 4), RPN이 제안한 RoI들
            bbox (ndarray): (R', 4), 실제 GT BBox
            label (ndarray): (R',), GT BBox 라벨(0~클래스-1)
            loc_normalize_mean (tuple): BBox 오프셋 정규화 평균
            loc_normalize_std (tuple): BBox 오프셋 정규화 표준편차

        Returns:
            sampled_roi (ndarray): (S, 4), 샘플링된 RoI 좌표
            gt_roi_loc (ndarray): (S, 4), 해당 RoI를 GT에 맞추기 위한 오프셋
            gt_roi_label (ndarray): (S,), 0=배경, 1~K=클래스 라벨
        """
        n_bbox = bbox.shape[0]

        # RoI + GT 합치기
        roi = np.concatenate((roi, bbox), axis=0)

        # 전경 RoI 수
        pos_roi_per_image = int(round(self.n_sample * self.pos_ratio))

        # IoU 계산
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)

        # 라벨 0을 배경으로 하기 위해 +1
        gt_roi_label = label[gt_assignment] + 1

        # 전경(positive) 선택
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_image = min(pos_roi_per_image, len(pos_index))
        if len(pos_index) > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_image, replace=False)

        # 배경(negative) 선택
        neg_index = np.where(
            (max_iou < self.neg_iou_thresh_hi) &
            (max_iou >= self.neg_iou_thresh_lo)
        )[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_image
        neg_roi_per_this_image = min(neg_roi_per_this_image, len(neg_index))
        if len(neg_index) > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False
            )

        # 전경 + 배경
        keep_index = np.append(pos_index, neg_index)

        # 라벨 설정(배경=0)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_image:] = 0
        sample_roi = roi[keep_index]

        # 샘플링된 RoI를 실제 GT에 맞추기 위한 오프셋 계산
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (
            gt_roi_loc - np.array(loc_normalize_mean, np.float32)
        ) / np.array(loc_normalize_std, np.float32)

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator:
    """
    RPN 학습용:
        입력된 Anchor에 대해 GT BBox를 할당(전경/배경/무시)하고, bbox 오프셋(loc)을 계산
    """

    def __init__(
        self,
        n_sample=256,
        pos_iou_thresh=0.7,
        neg_iou_thresh=0.3,
        pos_ratio=0.5
    ):
        """
        Args:
            n_sample (int): 샘플링할 anchor 수
            pos_iou_thresh (float): IoU >= pos_iou_thresh -> 전경
            neg_iou_thresh (float): IoU < neg_iou_thresh -> 배경
            pos_ratio (float): 전경 anchor 비율
        """
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """
        최종 (loc, label) 반환
        - loc: (S,4), 전경 anchor 대비 GT bbox 오프셋  
        - label: (S,), {1=전경, 0=배경, -1=무시}
        """
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]

        argmax_ious, label = self._create_label(inside_index, anchor, bbox)
        loc = bbox2loc(anchor, bbox[argmax_ious])

        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        label = np.full((len(inside_index),), -1, dtype=np.int32)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        # 배경/전경 설정
        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        # 전경 제한
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False
            )
            label[disable_index] = -1

        # 배경 제한
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False
            )
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    """
    data(크기 작은 배열)을 원래 크기(count)에 맞춰 확장하여 배치
    index 외 위치는 fill로 채움
    """
    if data.ndim == 1:
        ret = np.full((count,), fill, dtype=data.dtype)
        ret[index] = data
    else:
        ret = np.full((count,) + data.shape[1:], fill, dtype=data.dtype)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    """
    anchor가 이미지 범위 내에 완전히 들어 있는 인덱스만 반환
    """
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:
    """
    (RPN 추론) loc, score, anchor -> 최종 RoI 생성
    NMS 등으로 후보 bbox를 걸러서 반환
    """

    def __init__(
        self,
        parent_model,
        nms_thresh=0.7,
        n_train_pre_nms=12000,
        n_train_post_nms=2000,
        n_test_pre_nms=6000,
        n_test_post_nms=300,
        min_size=16
    ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        Args:
            loc (ndarray): (R,4) 예측된 오프셋
            score (ndarray): (R,) 전경 확률
            anchor (ndarray): (R,4)
            img_size (tuple): (H,W)
            scale (float): 이미지 스케일
        
        Returns:
            (ndarray): (N,4) NMS 통과 후 최종 RoI
        """
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)

        # 이미지 범위 내로 클리핑
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # 크기가 너무 작은 bbox 제거
        min_size_ = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size_) & (ws >= min_size_))[0]
        roi = roi[keep]
        score = score[keep]

        # 스코어 내림차순 정렬
        order = score.argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order]
        score = score[order]

        # NMS 수행
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh
        )
        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        return roi[keep.cpu().numpy()]