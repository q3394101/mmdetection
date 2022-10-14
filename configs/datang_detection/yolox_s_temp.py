# _base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'] # noqa E501

########################### DaTang Mobile YoloX v1.1 2022-10-08 ################################### noqa E501,E266
# 1. add occ estimation and weight (check v1.1-1)
# 2. guarantee the validation of bbox_ignore (check v1.1-2)
# 3. judge crop ratio for data augmentation : mosaic (check v1.1-3)
# 4. specify data augmentation (filter, pad ) (check v1.1-4)
# 5. strength lower-bound regression (check v1.1-5)
################################################################################################### noqa E501,E266

########################### model setting init 2022-10-08 ######################################### noqa E501,E266
import os

DEBUG = True
if os.environ.get('DEBUG', False):
    DEBUG = True
batch_size = 2
img_scale = (640, 640)  # height, width
CLASSES = ('Car', 'Bus', 'Cyclist', 'Pedestrian', 'driverless_car', 'Truck',
           'Tricyclist', 'Trafficcone')

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    # bbox_head=dict(
    #     type='YOLOXHead', num_classes=8, in_channels=128, feat_channels=128),
    bbox_head=dict(
        type='YOLOXHead_DT',
        num_classes=len(CLASSES),
        in_channels=128,
        feat_channels=128),
    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5),
        occ_cls_weight_type=  # noqa E251
        'Linear',  # v1.1-1  current types: 'None','Linear',etc.
        occ_reg_weight_type=  # noqa E251
        'Linear',  # v1.1-1  current types: 'None','Linear',etc.
        with_ignore=True,  # v1.1-2
        bound_weight=[1.0, 2.0, 1.0, 1.0]  # up low left right v1.1-5
    ),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.65),
    ))
###################################################################################################### noqa E501,E266

########################### data loading pipeline 2022-10-08 ######################################### noqa E501,E266

# dataset settings
data_root = 'data/coco/'
dataset_type = 'CocoDataset_datang'

train_pipeline = [
    #################### TODO:ratio  v1.1-3 # noqa E266
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        center_ratio_range=(0.5, 1.5)),
    ##############
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(5, 5),
         keep_empty=False),  # v1.1-1  v1.1-2
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_occs'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg'))  # v1.1-1  v1.1-2
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_occ=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0 if DEBUG else 4,
    persistent_workers=False if DEBUG else True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))

######################################################################################################  # noqa E501,266

########################### learning parameters 2022-10-08 ###########################################  # noqa E501,266
# optimizer
# default 8 gpu
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 15
# resume_from = None
interval = 999  # debug: 999  use: 1 or 2  (save checkpoint, eval checkpoint)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# learning policy
lr_config = dict(
    # _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)
# auto_scale_lr = dict(enable=False, base_batch_size=64)  # revise according to real situation 2022-10-08  # noqa E501,251

######################################################################################################  # noqa E501,266
