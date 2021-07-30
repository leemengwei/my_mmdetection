_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
# model settings
checkpoint_config = dict(interval=10)
workflow = [('train', 3)]
log_config = dict(
    interval=10)

model = dict(
    bbox_head=dict(
        num_classes=1),
    train_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.6), 
        max_per_img=200),  
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.5), 
        max_per_img=200),  
    )
load_from='./checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

# dataset settings
dataset_type = 'CocoDataset'
train_dir = "../dataset/出猪通道-泉州/train"
val_dir = "../dataset/出猪通道-泉州/val"
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(800, 448), (1000, 550), (608, 608), (800, 800), (448, 448)], keep_ratio=True, multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=train_dir + '/annotation_coco.json',
        img_prefix=train_dir,
        pipeline=train_pipeline,
        classes=('fisheye_pig',)
        ),
    val=dict(
        type=dataset_type,
        ann_file=val_dir + '/annotation_coco.json',
        img_prefix=val_dir,
        pipeline=test_pipeline,
        classes=('fisheye_pig',)
        ),
    test=dict(
        type=dataset_type,
        ann_file=val_dir + '/annotation_coco.json',
        img_prefix=val_dir,
        pipeline=test_pipeline,
        classes=('fisheye_pig',)
        )
    )

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[36, 72])
runner = dict(type='EpochBasedRunner', max_epochs=96)
