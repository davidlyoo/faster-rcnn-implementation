import numpy as np
import six


def loc2bbox(src_bbox, loc):
    """
    BBox 오프셋(loc)을 활용해 원본 BBox(src_bbox)를 디코딩하면 최종 BBox 좌표를 구함
    
    Args:
        src_bbox (ndarray): (R, 4) shape, 각 요소는 [y_min, x_min, y_max, x_max]
        loc (ndarray): (R, 4k) 형태 (k=클래스 개수 등), bbox offsets
                        보통 (R, 4) 혹은 (R, n_class*4)

    Returns:
        (R, 4k) shape:
            디코딩된 bounding box.
            각 4개의 값은 [y_min, x_min, y_max, x_max] 순
    """
    if src_bbox.shape[0] == 0:
        # Bounding-box가 없는 경우
        return np.zeros((0, loc.shape[1]), dtype=loc.dtype)

    # float 변환
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """
    두 BBox(원본·대상)를 비교해, 오프셋(Δ)을 계산함(Encode)

    Args:
        src_bbox (ndarray): (R, 4) shape, 원본 bbox
        dst_bbox (ndarray): (R, 4) shape, 타겟 bbox

    Returns:
        (R, 4) shape ndarray:
            [t_y, t_x, t_h, t_w]
    """
    # src_bbox, dst_bbox 모두 (R, 4)라고 가정
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """
    두 집합의 BBox 간 IoU 행렬 계산

    Args:
        bbox_a (ndarray): (N, 4) shape [y_min, x_min, y_max, x_max]
        bbox_b (ndarray): (K, 4) shape [y_min, x_min, y_max, x_max]

    Returns:
        ious (ndarray): (N, K) shape, IoU 값
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError("Bbox shape must be (?, 4).")

    # top-left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom-right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    # intersection
    inter_area = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)

    # area of a, b
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return inter_area / (area_a[:, np.newaxis] + area_b - inter_area)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                        anchor_scales=[8, 16, 32]):
    """
    Faster R-CNN에서 사용할 기본 Anchor 생성

    Args:
        base_size (int or float): 기본 anchor의 (가로, 세로) 크기
        ratios (list[float]): width:height 비율
        anchor_scales (list[int]): scale 배수

    Returns:
        anchor_base (ndarray): (R, 4) shape,
            [y_min, x_min, y_max, x_max]
    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    index = 0
    for r in ratios:
        for s in anchor_scales:
            h = base_size * s * np.sqrt(r)
            w = base_size * s * np.sqrt(1. / r)

            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
            index += 1

    return anchor_base
