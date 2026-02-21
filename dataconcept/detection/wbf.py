"""weighted box fusion utilities for merging multi-scale detection results."""

import numpy as np
from ensemble_boxes import weighted_boxes_fusion


def iou(box1, box2):
    """compute intersection over union between two xyxy boxes."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter_area
    return inter_area / union_area if union_area != 0 else 0


def filter_overlapping_bboxes(bboxes, scores, labels, iou_thr=0.5):
    """remove the lower-scoring box when two different-class boxes overlap above threshold."""
    keep = [True] * len(bboxes)
    for i in range(len(bboxes)):
        for j in range(i):
            if labels[i] != labels[j] and iou(bboxes[i], bboxes[j]) > iou_thr:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False
    return bboxes[keep], scores[keep], [labels[i] for i in range(len(labels)) if keep[i]]


def bbox_ensemble(bboxes, scores, labels, iou_thr=0.3):
    """fuse boxes from multiple detection scales using weighted box fusion, then filter overlaps."""
    unique_labels = list(set(sum(labels, [])))
    label_to_idx = {label: i + 1 for i, label in enumerate(unique_labels)}
    idx_to_label = {i + 1: label for i, label in enumerate(unique_labels)}

    labels_numeric = [[label_to_idx[l] for l in label_set] for label_set in labels]

    final_bboxes, final_scores, final_labels = weighted_boxes_fusion(
        bboxes, scores, labels_numeric, iou_thr=iou_thr, skip_box_thr=0.26,
    )

    final_labels = [idx_to_label[int(l)] for l in final_labels]
    final_bboxes, final_scores, final_labels = filter_overlapping_bboxes(
        np.array(final_bboxes), np.array(final_scores), final_labels, iou_thr=0.5,
    )

    return final_bboxes, final_scores, final_labels
