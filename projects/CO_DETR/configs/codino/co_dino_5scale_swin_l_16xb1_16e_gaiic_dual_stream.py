_base_ = ['co_dino_5scale_r50_8xb2_1x_coco.py']


## Custom ##
# from .my_loading import LoadImageFromFile2
# from .my_wrapper import Image2Broadcaster, Branch
# from .my_formatting import DoublePackDetInputs
custom_imports = dict(imports=['projects.CO_DETR.codetr',
                               'mmdet.datasets.transforms.my_loading',
                               'mmdet.datasets.transforms.my_wrapper',
                               'mmdet.datasets.transforms.my_formatting',
                               'mmdet.models.data_preprocessors.my_data_preprocessor',
                               'mmdet.datasets.my_coco'
                               ], allow_failed_imports=False)

dataset_type = 'DualStreamCocoDataset'
data_root = '/nasdata/private/zwlu/detection/Gaiic1/projects/data/mmdet/gaiic/GAIIC2024/'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

image_size = (1024, 1024)
num_classes = 5
classes = ('car', 'truck', 'bus', 'van', 'freight_car')

batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size)
]
# model settings
model = dict(
    # Dual Stream model
    type='CoDETR_Dual',

    data_preprocessor=dict(
        type='DoubleInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192, # Dual-Stream features combined before neck
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    
    dict(type='TransformBroadcaster',
        mapping={'img': ['img', 'img2']},
        auto_remap=True,
        # share_random_param=True,
        transforms=[
             dict(type='RandomFlip', prob=0.5),
             dict(
                type='RandomChoice',
                transforms=[
                        [
                            dict(
                            type='RandomChoiceResize',
                            scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                                    (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                                    (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                                    (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                                    (992, 2048), (1024, 2048), (1056, 2048),
                                    (1088, 2048), (1120, 2048), (1152, 2048),
                                    (1184, 2048), (1216, 2048), (1248, 2048),
                                    (1280, 2048), (1312, 2048), (1344, 2048),
                                    (1376, 2048), (1408, 2048), (1440, 2048),
                                    (1472, 2048), (1504, 2048), (1536, 2048)],
                            keep_ratio=True)
                        ],
                        [
                        dict(
                            type='RandomChoiceResize',
                            # The radio of all image in train dataset < 7
                            # follow the original implement
                            scales=[(400, 4200), (500, 4200), (600, 4200)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                                    (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                                    (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                                    (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                                    (992, 2048), (1024, 2048), (1056, 2048),
                                    (1088, 2048), (1120, 2048), (1152, 2048),
                                    (1184, 2048), (1216, 2048), (1248, 2048),
                                    (1280, 2048), (1312, 2048), (1344, 2048),
                                    (1376, 2048), (1408, 2048), (1440, 2048),
                                    (1472, 2048), (1504, 2048), (1536, 2048)],
                            keep_ratio=True)
                        ]
                    ]),
                ]
         
         ),
    
    dict(type='DoublePackDetInputs')
]

train_dataloader = dict(
        batch_size=1, num_workers=1, 
        dataset=dict(
            type=dataset_type,
            metainfo=dict(classes=classes),
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img='train/rgb'),
            pipeline=train_pipeline
            )
        )

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadImageFromFile2'),
    dict(type='LoadAnnotations'),
    
    dict(type='Branch',
         transforms=[
             dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
         ]),
    
    
    dict(
        type='DoublePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_evaluator = dict(
    ann_file=data_root + 'val.json')

val_dataloader = dict(dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/rgb'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=1e-4))

max_epochs = 16
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
