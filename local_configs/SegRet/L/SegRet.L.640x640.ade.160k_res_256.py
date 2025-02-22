_base_ = [
    '../../_base_/models/SegRet.py',
    # '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# data settings
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RMT_L_pre.pth',
    backbone=dict(
        type='RMT_L',
        norm_layer=norm_cfg),
    decode_head=dict(
        type='SegFormerHeadWithRes',
        in_channels=[112, 224, 448, 640],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256, use_relu=True),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # mode='slide', crop_size=(640, 640), stride=(608, 608)
    test_cfg=dict(mode='whole'))

dist_params = dict(backend='nccl', port=29518)  #

# optimizer
# todo: most papers only used the trick in the backbone,excluding the decoder head
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {}
# custom_keys.update({'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
custom_keys.update({
    f'backbone.layers.{stage_id}.blocks.{block_id}.retention_layer_norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate([4, 8, 25, 8])
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.layers.{stage_id}.blocks.{block_id}.final_layer_norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate([4, 8, 25, 8])
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.patch_embed.proj.{block_id}': backbone_norm_multi
    for block_id in [1, 4, 7, 10]
})
custom_keys.update({
    f'backbone.layers.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in [0, 1, 2]
})

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

lr_config = dict(_delete_=True, policy='poly',
                 # warmup='linear',
                 # warmup_iters=0,
                 # warmup_ratio=1e-6,
                 power=0.9, by_epoch=False)

evaluation = dict(interval=4000, metric='mIoU')

