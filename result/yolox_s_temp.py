DEBUG = True
batch_size = 2
img_scale = (640, 640)
CLASSES =  ('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding')
model = dict(
    type='YOLOX',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead_DT', num_classes=8, in_channels=128,
        feat_channels=128),
    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5),
        occ_cls_weight_type='Linear',
        occ_reg_weight_type='Linear',
        with_ignore=True,
        bound_weight=[1.0, 2.0, 1.0, 1.0],
        with_occ=True,
        with_direct=True),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
# data_root = '/home/chenzhen/code/detection/datasets/coco100/'
data_root = '/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/'
dataset_type = 'CocoDataset_datang'

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        center_ratio_range=(1.0, 1.0)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(5, 5), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_occs',
            'gt_direct'
        ],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg'))
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset_datang',
        classes= ('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding'),
        ann_file=
        '/home/chenzhen/code/detection/datasets/union2voc_multiClass/coco/annotations/train.json',
        img_prefix='/home/chenzhen/code/detection/datasets/union2voc_multiClass/coco/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_occ=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(
            type='Mosaic',
            img_scale=(640, 640),
            pad_val=114.0,
            center_ratio_range=(1.0, 1.0)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(5, 5), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_occs',
                'gt_direct'
            ],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                       'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                       'img_norm_cfg'))
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
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
    samples_per_gpu=2,
    workers_per_gpu=0,
    persistent_workers=False,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset_datang',
            classes=('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding'),
            ann_file=
            '/home/chenzhen/code/detection/datasets/union2voc_multiClass/coco/annotations/train.json',
            img_prefix='/home/chenzhen/code/detection/datasets/union2voc_multiClass/coco/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_occ=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                center_ratio_range=(1.0, 1.0)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(5, 5),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
                    'gt_occs', 'gt_direct'
                ],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg'))
        ]),
    val=dict(
        type='CocoDataset_datang',
        classes= ('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding'),
        ann_file=
        '/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/annotations/val.json',
        img_prefix='/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
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
        ]),
    test=dict(
        type='CocoDataset_datang',
        classes= ('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding'),
        ann_file=
        '/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/annotations/val.json',
        img_prefix='/home/chenzhen/code/detection/datasets/dt_hangzhou/coco_dt_with_date_captured/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
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
        ]))
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
max_epochs = 300
num_last_epochs = 15
interval = 999
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=999, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=999)
evaluation = dict(
    save_best='auto',
    interval=999,
    dynamic_intervals=[(285, 1)],
    metric='bbox')
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = './work_dirs/yolox_s_temp'
auto_resume = False
gpu_ids = [0]
