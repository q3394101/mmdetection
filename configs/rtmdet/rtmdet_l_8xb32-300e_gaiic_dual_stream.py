from doctest import debug


_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    # '../_base_/datasets/coco_detection.py'
]
# , './rtmdet_tta.py'
custom_imports = dict(imports=[
                               'mmdet.datasets.transforms.my_loading',
                               'mmdet.datasets.transforms.my_wrapper',
                               'mmdet.datasets.transforms.my_formatting',
                               'mmdet.models.data_preprocessors.my_data_preprocessor',
                               'mmdet.datasets.my_coco',
                               'mmdet.models.detectors.rtmdet_dual_stream',
                               'mmdet.datasets.transforms.my_transforms'
                               ], allow_failed_imports=False)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
debug_flag = False

dataset_type = 'DualStreamCocoDataset'
data_root = '/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/'
num_classes = 5
classes = ('car', 'truck', 'bus', 'van', 'freight_car')


model = dict(
    type='RTMDet_Dual',
    data_preprocessor=dict(
        type='DoubleInputDetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=True,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    
    dict(type='CachedMosaic2Images', img_scale=(640, 640), pad_val=114.0),

    dict(
        type='Image2Broadcaster',
        transforms=[
            
            dict(
                type='RandomResize',
                scale=(1280, 1280),
                ratio_range=(0.8, 1.3),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640), recompute_bbox=True,
                 crop_type='absolute',
                    allow_negative_crop=True),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            # dict(
            #     type='CachedMixUp',
            #     img_scale=(640, 640),
            #     ratio_range=(1.0, 1.0),
            #     max_cached_images=20,
            #     pad_val=(114, 114, 114),
            #     prob=1),
        ]
    ),
    
    dict(
        type='Branch',
        transforms=[
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    dict(type='DoublePackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'img_path2', 'ori_shape2', 'img_shape2',
                   'scale_factor'))
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    dict(type='Image2Broadcaster',
        transforms=[
            dict(
                type='RandomResize',
                scale=(640, 640),
                ratio_range=(0.1, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640), recompute_bbox=True,
                    crop_type='absolute',
                    allow_negative_crop=True),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
        ]
    ),
    
    dict(
        type='Branch',
        transforms=[
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        ]
    ),
    
    dict(type='DoublePackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'img_path2', 'ori_shape2', 'img_shape2',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
        
    dict(type='LoadAnnotations', with_bbox=True,),
    dict(type='Branch',
        transforms=[
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        ]
    ),

    dict(
        type='DoublePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'img_path2', 'ori_shape2', 'img_shape2',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=16,
    num_workers=10,
    batch_sampler=None,
    pin_memory=True,
    # dataset=dict(pipeline=train_pipeline),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json' if debug_flag else 'train.json',
        data_prefix=dict(img='val/rgb'if debug_flag else 'train/rgb'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=5, num_workers=10, 
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/rgb'),
        test_mode=True,
        pipeline=test_pipeline,
        ))

test_dataloader = val_dataloader

max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004
interval = 1 if debug_flag else 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    proposal_nums=(100, 1, 10),
    format_only=False,
    )

test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

visualization = _base_.default_hooks.visualization
visualization.update(dict(draw=True, show=debug_flag))