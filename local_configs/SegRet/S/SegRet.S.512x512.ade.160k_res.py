_base_ = [
    '../../_base_/models/SegRet.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # SyncBN
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RMT_S_pre.pth',
    backbone=dict(
        type='RMT_S',
        norm_layer=norm_cfg),
    decode_head=dict(
        type='SegFormerHeadWithRes',
        in_channels=[64, 128, 256, 512],
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
    # mode='slide', crop_size=(512, 512), stride=(480, 480)  # mode='whole'
    test_cfg=dict(mode='whole'))

dist_params = dict(port=29518)

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
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=16)
evaluation = dict(interval=4000, metric='mIoU')
