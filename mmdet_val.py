import argparse
import math

import mmcv
import numpy as np
import torch.nn as nn
from mmcv import Config
from mmcv.ops import nms
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_map
from mmdet.core.evaluation.bbox_area import bbox_area
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

pair = nn.PairwiseDistance(p=2)


def calculate_num_confusion_matrix(dataset,
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
        num_analyze_per_img_dets(confusion_matrix, gt_bboxes, labels,
                                 res_bboxes, score_thr, tp_iou_thr,
                                 nms_iou_thr, area_size)
        prog_bar.update()
    return confusion_matrix


def calculate_dis_confusion_matrix(dataset,
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
        dis_analyze_per_img_dets(
            result_matrix,
            gt_bboxes,
            labels,
            res_bboxes,
            score_thr,
            tp_iou_thr,
        )
        prog_bar.update()
    return result_matrix


def voc_eval(result, dataset, iou_thr=0.5, nproc=4):
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    _, eval_results = eval_map(
        result,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc)
    return eval_results


def dis_analyze_per_img_dets(
    result_matrix,
    gt_bboxes,
    gt_labels,
    result,
    score_thr=0.3,
    tp_iou_thr=0.5,
):
    for det_label, det_bboxes in enumerate(result):
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr and gt_label == det_label:
                        det_point_y = det_bbox[1] + (
                            (det_bbox[3] - det_bbox[1]) / 2)
                        det_point_x = det_bbox[2]

                        gt_point_y = gt_bboxes[j][1] + (
                            (gt_bboxes[j][3] - gt_bboxes[j][1]) / 2)
                        gt_point_x = gt_bboxes[j][2]

                        point_val_result = math.sqrt((
                            (gt_point_x - det_point_x)**2) + (
                                (gt_point_y - det_point_y)**2))
                        line_val_result = gt_bboxes[j][2] - det_bbox[2]

                        result_matrix[gt_label, 0] += abs(point_val_result)
                        result_matrix[gt_label, 1] += abs(line_val_result)


def num_analyze_per_img_dets(confusion_matrix,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='DT_detection test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--tp_iou_thr',
        type=float,
        default=0.5,
        help='iou threshold (default: 0.5)')
    parser.add_argument(
        '--area_size',
        type=float,
        default=None,
        help='test(val) bbox area range (default: None)')
    parser.add_argument(
        '--nms_iou_thr',
        type=float,
        default=None,
        help='nms iou  threshold (default: None)')

    args = parser.parse_args()

    return args


def main():
    exp = 1e-7
    args = parse_args()
    config = args.config
    checkpoint_path = args.checkpoint
    score_thr = args.score_thr
    tp_iou_thr = args.tp_iou_thr
    area_size = args.area_size
    nms_iou_thr = args.nms_iou_thr
    cfg = Config.fromfile(config)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    results = mmcv.load(checkpoint_path)
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
    dataset_name = list(dataset.CLASSES)
    # print('\n----------------datasets-------------------\n')
    # print(dataset)

    print('\n------------cal_val-----------------------\n')
    # 1.map
    eval_results = voc_eval(results, dataset, tp_iou_thr, 4)
    num_gts = np.zeros((1, len(dataset_name)), dtype=int)
    for i, cls_result in enumerate(eval_results):
        num_gts[:, i] = cls_result['num_gts']

    # 2. cal tp fp
    TP_confusion_matrix = calculate_num_confusion_matrix(
        dataset,
        results,
        score_thr=score_thr,
        nms_iou_thr=nms_iou_thr,
        tp_iou_thr=tp_iou_thr,
        area_size=area_size)
    np.set_printoptions(precision=4, suppress=True)
    tp = TP_confusion_matrix.diagonal()
    fp = TP_confusion_matrix.sum(0) - tp  # false positives
    pure_fp = TP_confusion_matrix[-1, :]
    confusion_fp = fp - pure_fp
    # fn = TP_confusion_matrix[:, -1]  # false negatives (missed detections)
    fn = num_gts[0] - tp[:-1]
    print('\n----------------cal_tp_fp_fn-------------------\n')

    header1 = ['class', 'tp', 'pure_fp', 'conf_fp', 'fn']
    table_data1 = [header1]
    for i in range(len(dataset_name)):
        row_data1 = [
            dataset_name[i], tp[i], pure_fp[i], confusion_fp[i], fn[i]
        ]
        table_data1.append(row_data1)
    table1 = AsciiTable(table_data1)
    print_log('\n' + table1.table)

    # 3. cal dis loss
    DIS_confusion_matrix = calculate_dis_confusion_matrix(
        dataset, results, score_thr=score_thr, tp_iou_thr=tp_iou_thr)
    np.set_printoptions(precision=4, suppress=True)

    print('\n-------------cal_point_line----------------------\n')
    point_result_normal = list(
        map(lambda x: x[0] / (x[1] + exp),
            zip(list(DIS_confusion_matrix[:, 0]), tp)))
    line_result_normal = list(
        map(lambda x: x[0] / (x[1] + exp),
            zip(list(DIS_confusion_matrix[:, 1]), tp)))

    header2 = ['class', 'point_result', 'line_result']
    table_data2 = [header2]
    for i in range(len(dataset_name)):
        row_data2 = [
            dataset_name[i], f'{point_result_normal[i]:.3f}',
            f'{line_result_normal[i]:.3f}'
        ]
        table_data2.append(row_data2)
    table2 = AsciiTable(table_data2)
    print_log('\n' + table2.table)


if __name__ == '__main__':
    main()
