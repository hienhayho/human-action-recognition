model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHeadFocalLoss',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    train_cfg=None,
    test_cfg=None)