ann_file_test = 'data_preprocessed_full/val.txt'
ann_file_train = 'data_preprocessed_full/train_val_decrease_sample.txt'
ann_file_val = 'data_preprocessed_full/val.txt'
data_root = 'data_preprocessed_full/train'
data_root_val = 'data_preprocessed_full/val'
auto_scale_lr = dict(base_batch_size=400, enable=False)
base_lr = 0.0016
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = '/mlcv/WorkingSpace/Personals/tuongbck/VNDH/mmaction2/train_mvit_3/best_acc_top1_epoch_3.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.3,
        spatial_size=400,
        temporal_size=32,
        type='MViT'),
    cls_head=dict(
        average_clips='prob',
        in_channels=768,
        label_smooth_eps=0.1,
        num_classes=6,
        type='MViTHead'),
    data_preprocessor=dict(
        blending=dict(
            augments=[
                dict(alpha=0.8, num_classes=6, type='MixupBlending'),
                dict(alpha=1, num_classes=6, type='CutmixBlending'),
            ],
            type='RandomBatchAugment'),
        format_shape='NCTHW',
        mean=[
            114.75,
            114.75,
            114.75,
        ],
        std=[
            57.375,
            57.375,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0016, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0.01, norm_decay_mult=0.01))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.01,
        type='LinearLR'),
    dict(
        T_max=200,
        begin=6,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        eta_min=1.6e-05,
        type='CosineAnnealingLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            30,
            60,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
repeat_sample = 2
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data_preprocessed_full/val.txt',
        data_prefix=dict(video='data_preprocessed_full/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=4,
                frame_interval=3,
                num_clips=1,
                test_mode=True,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                400,
            ), type='Resize'),
            dict(crop_size=400, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        400,
    ), type='Resize'),
    dict(crop_size=400, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=200, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    collate_fn=dict(type='repeat_pseudo_collate'),
    dataset=dict(
        ann_file='data_preprocessed_full/train_val_decrease_sample.txt',
        data_prefix=dict(video='data_preprocessed_full/train'),
        num_repeats=1,
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=4,
                frame_interval=3,
                num_clips=1,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                400,
            ), type='Resize'),
            dict(
                magnitude=7,
                num_layers=4,
                op='RandAugment',
                type='PytorchVideoWrapper'),
            dict(keep_ratio=False, scale=(
                400,
                400,
            ), type='Resize'),
            dict(crop_size=400, type='CenterCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(erase_prob=0.25, mode='rand', type='RandomErasing'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        sample_once=True,
        type='RepeatAugDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_evaluator = dict(type='AccMetric')
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=4, frame_interval=3, num_clips=1, type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        400,
    ), type='Resize'),
    dict(
        magnitude=7,
        num_layers=4,
        op='RandAugment',
        type='PytorchVideoWrapper'),
    dict(keep_ratio=False, scale=(
        400,
        400,
    ), type='Resize'),
    dict(crop_size=400, type='CenterCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(erase_prob=0.25, mode='rand', type='RandomErasing'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data_preprocessed_full/val.txt',
        data_prefix=dict(video='data_preprocessed_full/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=4,
                frame_interval=3,
                num_clips=1,
                test_mode=True,
                type='DenseSampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                400,
            ), type='Resize'),
            dict(crop_size=400, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=4,
        frame_interval=3,
        num_clips=1,
        test_mode=True,
        type='DenseSampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        400,
    ), type='Resize'),
    dict(crop_size=400, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'train_mvit_3/'
