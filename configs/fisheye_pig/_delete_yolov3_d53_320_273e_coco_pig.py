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
dataset_type = 'COCODataset'
train_dir = "../dataset/fish_eye_pig/train"
val_dir = "../dataset/fish_eye_pig/val"
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=train_dir + '/annotation_coco.json',
        img_prefix=train_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('fisheye_pig',)
        ),
    val=dict(
        type='CocoDataset',
        ann_file=val_dir + '/annotation_coco.json',
        img_prefix=val_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('fisheye_pig',)
        ),
    test=dict(
        type='CocoDataset',
        ann_file=val_dir + '/annotation_coco.json',
        img_prefix=val_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('fisheye_pig',)
        ))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[36, 72])
runner = dict(type='EpochBasedRunner', max_epochs=96)