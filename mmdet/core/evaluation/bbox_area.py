# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def bbox_area(bboxes1, area_size):
    """Calculate the area each bbox of bboxes1 and bboxes2.

    Args:
    bboxes1 (ndarray): Shape (k, 4)
    """
    bboxes = bboxes1[:, :4]
    bboxes = bboxes.astype(np.float32)

    det_bbox = []

    area1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    for i in range(bboxes1.shape[0]):

        if area_size[0] <= area1[i] <= area_size[1]:

            det_bbox.append(bboxes1[i])

    return np.array(det_bbox)
