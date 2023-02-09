optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=25,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=5)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=25, priority=48),
    dict(type='SyncNormHook', num_last_epochs=25, interval=5, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
img_scale = (800, 800)
CLASSES = ('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car', 'Truck',
           'Animal', 'Obstacle', 'Special_Target', 'Other_Objects',
           'Unmanned_riding')
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
model = dict(
    type='YOLOX',
    input_size=(800, 800),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(
        type='mmcls.RepVGG',
        arch='yolox-pai-small',
        add_ppf=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        out_indices=(1, 2, 3)),
    neck=dict(
        type='YOLOXASFFPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        act_cfg=dict(type='SiLU'),
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXTOODHead',
        num_classes=11,
        in_channels=128,
        feat_channels=128,
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
work_dir = './work_dir/yolox-repvgg/coco-hz-x2x-repo/asff_tood'
data_root = '/app/dataset/coco-hz-x2x-repo/'
dataset_type = 'CocoDataset'
train_pipeline = [
    dict(
        type='RoadPaste',
        img_scale=(800, 800),
        pad_val=114,
        center_ratio_range=(0.5, 1.5),
        prob=0.5,
        roadname='date_captured'),
    dict(type='Mosaic', img_scale=(800, 800), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-400, -400)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        classes=('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car',
                 'Truck', 'Animal', 'Obstacle', 'Special_Target',
                 'Other_Objects', 'Unmanned_riding'),
        ann_file='/app/dataset/coco-hz-x2x-repo/annotations/train.json',
        img_prefix='/app/dataset/coco-hz-x2x-repo/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(
            type='RoadPaste',
            img_scale=(800, 800),
            pad_val=114,
            center_ratio_range=(0.5, 1.5),
            prob=0.5,
            roadname='date_captured'),
        dict(type='Mosaic', img_scale=(800, 800), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-400, -400)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
        dict(
            type='Pad',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
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
    samples_per_gpu=32,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car',
                     'Truck', 'Animal', 'Obstacle', 'Special_Target',
                     'Other_Objects', 'Unmanned_riding'),
            ann_file='/app/dataset/coco-hz-x2x-repo/annotations/train.json',
            img_prefix='/app/dataset/coco-hz-x2x-repo/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(
                type='RoadPaste',
                img_scale=(800, 800),
                pad_val=114,
                center_ratio_range=(0.5, 1.5),
                prob=0.5,
                roadname='date_captured'),
            dict(type='Mosaic', img_scale=(800, 800), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-400, -400)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car',
                 'Truck', 'Animal', 'Obstacle', 'Special_Target',
                 'Other_Objects', 'Unmanned_riding'),
        ann_file='/app/dataset/coco-hz-x2x-repo/annotations/val.json',
        img_prefix='/app/dataset/coco-hz-x2x-repo/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
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
        type='CocoDataset',
        classes=('Car', 'Bus', 'Cycling', 'Pedestrian', 'driverless_Car',
                 'Truck', 'Animal', 'Obstacle', 'Special_Target',
                 'Other_Objects', 'Unmanned_riding'),
        ann_file='/app/dataset/coco-hz-x2x-repo/annotations/val.json',
        img_prefix='/app/dataset/coco-hz-x2x-repo/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
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
max_epochs = 300
num_last_epochs = 25
interval = 5
evaluation = dict(
    save_best='auto', interval=5, dynamic_intervals=[(275, 1)], metric='bbox')
auto_resume = False
gpu_ids = range(0, 4)
