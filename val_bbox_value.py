import math

import matplotlib
matplotlib.use('TkAgg')
import mmcv
import numpy as np
import torch.nn as nn
from mmcv import Config

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root
pair = nn.PairwiseDistance(p=2)

def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0.3,
                               tp_iou_thr=0.5):
    num_classes = len(dataset.CLASSES)
    result_matrix = np.zeros(shape=[num_classes, 2])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        analyze_per_img_dets(result_matrix, gt_bboxes, labels, res_bboxes,score_thr, tp_iou_thr)
        prog_bar.update()
    return result_matrix

def analyze_per_img_dets(result_matrix,
                         gt_bboxes,
                         gt_labels,
                         result,
                         score_thr=0.3,
                         tp_iou_thr=0.5,):
    for det_label, det_bboxes in enumerate(result):
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr and gt_label == det_label:
                        det_point_y = det_bbox[1] + ((det_bbox[3] - det_bbox[1]) / 2)
                        det_point_x = det_bbox[2]

                        gt_point_y = gt_bboxes[j][1] + ((gt_bboxes[j][3] - gt_bboxes[j][1]) / 2)
                        gt_point_x = gt_bboxes[j][2]

                        point_val_result = math.sqrt(
                            ((gt_point_x - det_point_x) ** 2) + ((gt_point_y - det_point_y) ** 2))
                        line_val_result = gt_bboxes[j][2] - det_bbox[2]

                        result_matrix[gt_label, 0] += abs(point_val_result)
                        result_matrix[gt_label, 1] += abs(line_val_result)




def main():
    prediction_path = "/home/chenzhen/dt_mmdetection/tools/2399_results.pkl"
    # prediction_path = "/home/chenzhen/dt_mmdetection/tools/2399_40_results.pkl"
    config = "/home/chenzhen/dt_mmdetection/configs/yolox/yolox_s_temp.py"
    cfg = Config.fromfile(config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    results = mmcv.load(prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)
    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  score_thr=0.75,
                                                  tp_iou_thr=0.75)
    np.set_printoptions(precision=4, suppress=True)
    print(list(confusion_matrix[:,0]))
    print(list(confusion_matrix[:,1]))


if __name__ == '__main__':
    main()