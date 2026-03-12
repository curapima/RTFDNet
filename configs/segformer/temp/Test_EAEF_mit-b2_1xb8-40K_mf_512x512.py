norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        123.675, 116.28, 103.53, 123.675, 116.28, 103.53
    ],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(480,640))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            123.675, 116.28, 103.53, 123.675, 116.28, 103.53
        ],
        std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(480,640)),
    pretrained='D:/UniEAEF/pretrain/mmseg_mit_b2.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head =dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'MSRS_ADE20KDataset'
data_root = 'D:/mmsegmentation/dataset/ade_mf/'
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MSRS_ADE20KDataset',
        data_root='D:/mmsegmentation/dataset/ade_mf/',
        data_prefix=dict(
            img_path='rgbx_np/validation',
            seg_map_path='annotations/validation'),
        pipeline=[
            dict(type='LoadImageFromFile_rgbx'),
            dict(type='Resize', scale=(480, 640), keep_ratio=True),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

test_dataloader = val_dataloader
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=True)

log_level = 'INFO'

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=10,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

launcher = 'none'
work_dir = './work_dirs\\rgb_segformer_MF_RGB_mit-b2_bit'
