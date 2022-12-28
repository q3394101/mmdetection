import mmcv
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.fileio import FileClient
from mmcv.visualization import  color_val
import os.path as osp

save = True
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])


@HOOKS.register_module()
class BaseShowDataPipline(Hook):

    def before_run(self, runner):
        runner.logger.info(f'BaseShowDataPipline will be runing!!')
        if save:
            self.out_dir = runner.work_dir
            self.file_client = FileClient.infer_client(None,
                                                       self.out_dir)
            self.out_dir = self.file_client.join_path(self.out_dir, "DataPipline")
            runner.logger.info(f'img  Visualization will be saved to {self.out_dir} by '
                               f'{self.file_client.name}.')
            os.makedirs(self.out_dir, exist_ok=True)

    def before_train_epoch(self, runner):
        train_loader = runner.data_loader
        for i, data_batch in enumerate(train_loader):
            img_batch = data_batch['img']._data[0]
            # gt_label = data_batch['gt_labels']._data[0]
            gt_bbox = data_batch['gt_bboxes']._data[0]
            image_name = osp.splitext(data_batch['img_metas']._data[0][1]['ori_filename'])[0]
            for batch_i in range(len(img_batch)):
                img = img_batch[batch_i]
                # labels = gt_label[batch_i].numpy()
                bboxes = gt_bbox[batch_i].numpy()
                mean_value = np.array(img_norm_cfg['mean'])
                std_value = np.array(img_norm_cfg['std'])
                img_hwc = np.transpose(img.numpy(), [1, 2, 0])
                img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
                img_numpy_uint8 = np.array(img_numpy_float, np.uint8)
                # print(labels)
                # 参考mmcv.imshow_bboxes

                assert bboxes.ndim == 2
                # assert labels.ndim == 1
                # assert bboxes.shape[0] == labels.shape[0]
                assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
                # colors = ['green', 'red', 'blue', 'cyan', 'yellow', 'magenta', 'white', 'black']
                # class_names = None
                score_thr = 0
                bbox_color = 'green'
                # text_color = 'green'
                # font_scale = 1
                thickness = 3
                img = np.ascontiguousarray(img_numpy_uint8)
                if score_thr > 0:
                    assert bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    # labels = labels[inds]

                bbox_color = color_val(bbox_color)
                # text_color = color_val(text_color)

                # for bbox, label in zip(bboxes, labels):
                for bbox in bboxes:
                    bbox_int = bbox.astype(np.int32)
                    left_top = (bbox_int[0], bbox_int[1])
                    right_bottom = (bbox_int[2], bbox_int[3])
                    cv2.rectangle(
                        img, left_top, right_bottom, bbox_color, thickness=thickness)
                    # label_text = class_names[
                    #     label] if class_names is not None else f'cls {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]: .02f}'
            # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
            #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
            if save:
                cv2.imwrite(osp.join(self.out_dir, image_name + ".jpg"), img)
            else:
                plt.imshow(img)
                plt.show()

