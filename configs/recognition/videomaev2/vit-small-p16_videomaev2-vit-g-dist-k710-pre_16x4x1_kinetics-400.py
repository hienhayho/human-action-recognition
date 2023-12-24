_base_ = ['vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py']

# model settings
model = dict(
    backbone=dict(embed_dims=384, depth=12, num_heads=6),
    cls_head=dict(in_channels=384))

dataset_type = 'VideoDataset'
data_root = 'data_preprocessed_full/train'
data_root_val = 'data_preprocessed_full/val'
ann_file_train = 'data_preprocessed_full/train_new_after.txt'
ann_file_val = 'data_preprocessed_full/val.txt'
ann_file_test = 'data_preprocessed_full/val.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='repeat_pseudo_collate'),
    dataset=dict(
        type='RepeatAugDataset',
        num_repeats=repeat_sample,
        sample_once=True,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

train_evaluator = dict(type='AccMetric')
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1.6e-3
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=30,
        end=200,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=512 // 2)