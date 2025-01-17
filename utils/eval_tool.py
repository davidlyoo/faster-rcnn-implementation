from __future__ import division

from collections import defaultdict
import itertools

import numpy as np
import six

from model.utils.bbox_tools import bbox_iou

def eval_detection_voc(
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
    gt_difficults=None, iou_thresh=0.5, use_07_metric=False
):
    """
    전체적인 객체 탐지 성능 평가 (mAP 계산)
    - pred_bboxes: 예측된 바운딩 박스 리스트
    - pred_labels: 예측된 클래스 라벨 리스트
    - pred_scores: 예측된 신뢰도 점수 리스트
    - gt_bboxes: 실제 바운딩 박스 리스트
    - gt_labels: 실제 클래스 라벨 리스트
    - gt_difficults: 어려운 샘플 여부 (VOC 데이터셋 기준, None 가능)
    - iou_thresh: IoU 임계값 (기본값: 0.5)
    - use_07_metric: VOC 2007 방식 사용 여부

    반환값:
    - {'ap': 각 클래스별 AP, 'map': 전체 mAP}
    """
    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh
    )

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
    gt_difficults=None, iou_thresh=0.5
):
    """
    Precision과 Recall을 계산하는 함수
    - IoU 임계값을 기준으로 예측된 객체가 GT와 얼마나 일치하는지 평가
    """
    
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)  # 클래스별 실제 객체 수
    score = defaultdict(list)  # 클래스별 예측 점수
    match = defaultdict(list)  # 예측된 객체와 GT의 매칭 결과 (1: 정답, 0: 오답, -1: 어려운 샘플)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in six.moves.zip(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults
    ):
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend(list(np.zeros(pred_bbox_l.shape[0])))
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('입력된 리스트들의 길이가 동일해야 합니다')

    if n_pos:
        n_fg_class = max(n_pos.keys()) + 1
    else:
        n_fg_class = 0

    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """
    VOC AP (Average Precision) 계산
    - 11-point interpolated AP (VOC 2007) 또는 standard AP 방식 사용 가능
    """
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)

    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
