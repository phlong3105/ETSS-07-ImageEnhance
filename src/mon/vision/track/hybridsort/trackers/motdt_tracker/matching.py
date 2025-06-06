import lap
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from trackers.motdt_tracker import kalman_filter


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def hmiou(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    ious = np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float)
    if ious.size == 0:
        return ious
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    iou[wc <= 0] = 0
    iou[hc <= 0] = 0
    iou *= o
    return iou

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def hmiou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = hmiou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def nearest_reid_distance(tracks, detections, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.features, det_features, metric).min(axis=0))

    return cost_matrix

def nearest_reid_distance_with_score(tracks, detections, metric='cosine', min_cls_score=0.4):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    det_score = np.asarray([track.score_kalman for track in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.features, det_features, metric).min(axis=0) + abs(np.clip(track.score_kalman, min_cls_score, 1.0)-det_score))

    return cost_matrix


def mean_reid_distance(tracks, detections, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type metric: str
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.empty((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    cost_matrix = cdist(track_features, det_features, metric)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix
