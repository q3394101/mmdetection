import matplotlib
import mmcv
import numpy as np
from mmcv import Config
from mmcv.ops import nms

from mmdet.core.evaluation.bbox_area import bbox_area
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

matplotlib.use('TkAgg')


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0.3,
                               nms_iou_thr=None,
                               area_size=None,
                               tp_iou_thr=0.5):
    num_classes = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
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
        analyze_per_img_dets(confusion_matrix, gt_bboxes, labels, res_bboxes,
                             score_thr, tp_iou_thr, nms_iou_thr, area_size)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gt_bboxes,
                         gt_labels,
                         result,
                         score_thr=0.3,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None,
                         area_size=None):
    if area_size:
        area_gt = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        mask = (area_gt > area_size[0]) & (area_gt <= area_size[1])
        gt_bboxes = gt_bboxes[mask]
        gt_labels = gt_labels[mask]
    true_positives = np.zeros_like(gt_labels)
    for det_label, det_bboxes in enumerate(result):
        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes[:, :4],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        if area_size:
            det_bboxes = bbox_area(det_bboxes, area_size)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def main():
    # prediction_path = "/home/chenzhen/dt_mmdetection/tools/2399_40_results.pkl" # noqa E501
    config = '/home/chenzhen/code/detection/mmdetection/configs/datang_detection/yolox_s_8x8_300e_coco.py'  # noqa E501
    prediction_path = '/home/chenzhen/code/detection/mmdetection/result/jichu_result.pkl'  # noqa E501
    area_size = None
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
    confusion_matrix = calculate_confusion_matrix(
        dataset,
        results,
        score_thr=0.3,
        nms_iou_thr=None,
        tp_iou_thr=0.75,
        area_size=area_size)
    np.set_printoptions(precision=4, suppress=True)
    tp = confusion_matrix.diagonal()
    fp = confusion_matrix.sum(0) - tp  # false positives
    fn = confusion_matrix[:, -1]  # false negatives (missed detections)
    cla_num = tp + fn
    print('\nconfusion_matrix', confusion_matrix)
    print('\ntp', tp, '\nfp', fp, '\nfn', fn, '\ncla_num', cla_num)


if __name__ == '__main__':
    main()
