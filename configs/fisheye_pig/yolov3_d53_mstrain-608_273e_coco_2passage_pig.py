_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
# model settings
model = dict(
    bbox_head=dict(
        num_classes=2),
    train_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.6), 
        max_per_img=200),  
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.5), 
        max_per_img=200),  
    )
#load_from='./checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'

# dataset settings
dataset_type = 'CocoDataset'
train_dir_1 = "../dataset/1猪舍-汇研+剪裁ok_cat_and_dogs/train"
val_dir_1 = "../dataset/1猪舍-汇研+剪裁ok_cat_and_dogs/val"
train_dir_2 = "../dataset/2出猪通道-泉州safe/train"
val_dir_2 = "../dataset/2出猪通道-泉州safe/val"
train_dir_3 = "../dataset/3出猪台-泉州+剪裁safe/train"
val_dir_3 = "../dataset/3出猪台-泉州+剪裁safe/val"
train_dir_3 = "../dataset/4称重台-泉州safe/train"
val_dir_3 = "../dataset/4称重台-泉州safe/val"
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='MinIoURandomCrop', min_ious=(0.8, 0.9, 1.0), min_crop_size=0.7),
    #dict(type='Resize', img_scale=[(1366, 768), (990, 540), (1115, 608)], keep_ratio=True, multiscale_mode='value'),
    dict(type='Resize', img_scale=[(608,608)], keep_ratio=True, multiscale_mode='value'),
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
        img_scale=(608,608),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=[train_dir_1 + '/annotation_coco.json', train_dir_2 + '/annotation_coco.json'],
        img_prefix=[train_dir_1, train_dir_2],
        pipeline=train_pipeline,
        classes=('pig', 'person')
        ),
    val=dict(
        type=dataset_type,
        ann_file=[val_dir_1 + '/annotation_coco.json', val_dir_2 + '/annotation_coco.json'],
        img_prefix=[val_dir_1, val_dir_2],
        pipeline=test_pipeline,
        classes=('pig', 'person')
        ),
    test=dict(
        type=dataset_type,
        ann_file=[val_dir_1 + '/annotation_coco.json', val_dir_2 + '/annotation_coco.json'],
        img_prefix=[val_dir_1, val_dir_2],
        pipeline=test_pipeline,
        classes=('pig', 'person')
        )
    )

# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    #_delete_=True,
    #policy='CosineAnnealing',
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    #warmup_ratio=1e-6,
    warmup_ratio=0.1,
    step=[150, 180],
    #min_lr_ratio=1e-10
    )
runner = dict(type='EpochBasedRunner', max_epochs=200)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10)
workflow = [('train', 1), ('val', 1)]