import mmcv
import numpy as np
from mmcv import Config

from mmcv.visualization import Color, color_val

from mmdet.datasets import build_dataloader, build_dataset
import matplotlib.pyplot as plt

import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cfg = Config.fromfile('/home/chenzhen/dt_code/mmdetection/configs/datang_detection/yolox_s_temp.py')
# print(cfg)
# cfg.gpu_ids = [0]
cfg.gpu_ids = range(0, 1)
cfg.seed = None

# Build dataset
dataset = build_dataset(cfg.data.train)
# prepare data loaders

data_loader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,
    # cfg.gpus will be ignored if distributed
    num_gpus=len(cfg.gpu_ids),
    dist=False,
    seed=cfg.seed)
print('build_dataloader finished')

for i, data_batch in enumerate(data_loader):
    # print(list(data_batch.keys()))
    img_batch = data_batch['img']._data[0]
    gt_label = data_batch['gt_labels']._data[0]
    gt_bbox = data_batch['gt_bboxes']._data[0]

    for batch_i in range(len(img_batch)):
        img = img_batch[batch_i]
        labels = gt_label[batch_i].numpy()
        bboxes = gt_bbox[batch_i].numpy()
        mean_value = np.array(cfg.img_norm_cfg['mean'])
        std_value = np.array(cfg.img_norm_cfg['std'])
        img_hwc = np.transpose(img.numpy(), [1, 2, 0])
        img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
        img_numpy_uint8 = np.array(img_numpy_float, np.uint8)
        # print(labels)
        # 参考mmcv.imshow_bboxes

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        # colors = ['green', 'red', 'blue', 'cyan', 'yellow', 'magenta', 'white', 'black']
        class_names = None
        score_thr = 0
        bbox_color = 'green'
        text_color = 'green'
        font_scale = 1
        thickness = 3
        img = np.ascontiguousarray(img_numpy_uint8)
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        bbox_color = color_val(bbox_color)
        text_color = color_val(text_color)

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
    if len(bbox) > 4:
        label_text += f'|{bbox[-1]: .02f}'
    cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    plt.imshow(img)
    plt.show()