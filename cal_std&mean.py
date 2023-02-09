import os

import mmcv
import numpy as np
from mmcv import Config

from mmdet.datasets import build_dataloader, build_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

cfg = Config.fromfile('/home/chenzhen/code/detection/mmdetection/configs/'
                      'datang_detection/yolox_s_temp.py')
# print(cfg)
# cfg.gpu_ids = [0]
cfg.gpu_ids = range(0, 1)
cfg.seed = None

img_list = []

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
    print(i)
    img_batch = data_batch['img']._data[0]
    # print(img_batch._data)
    gt_label = data_batch['gt_labels']._data[0]
    for batch_i in range(len(img_batch)):
        img = img_batch[batch_i]
        gt = gt_label[batch_i]

        mean_value = np.array(cfg.img_norm_cfg['mean'])
        std_value = np.array(cfg.img_norm_cfg['std'])
        img_hwc = np.transpose(img.numpy(), [1, 2, 0])
        img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
        img_numpy_uint8 = np.array(img_numpy_float, np.uint8)

        img_cal = img_numpy_uint8[np.newaxis, :, :, :]
        img_list.append(img_cal)

        # print(gt.numpy())
        # XXX: 该函数会调用cv2.imshow导致报错
        # mmcv.imshow(img_numpy_uint8, 'img', 0)

        # plt.imshow(img_numpy_uint8)
        # plt.show()
means, stdevs = [], []

imgs = np.concatenate(img_list, axis=0)
for i in range(3):
    pixels = imgs[:, :, :, i].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print('normMean = {}'.format(means))
print('normStd = {}'.format(stdevs))
