_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint_config = dict(interval=10)
workflow = [('train', 3)]
log_config = dict(
    interval=10)

model = dict(
    #pretrained='./checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth',
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

# dataset settings
# img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile', to_float32=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='PhotoMetricDistortion'),
#     dict(
#         type='Expand',
#         mean=img_norm_cfg['mean'],
#         to_rgb=img_norm_cfg['to_rgb'],
#         ratio_range=(1, 2)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#         min_crop_size=0.3),
#     dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(320, 320),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

# train_dir = "../dataset/fish_eye_pig/train"
# val_dir = "../dataset/fish_eye_pig/val"

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=1,
#     train=dict(
#         ann_file=train_dir + '/annotation_coco.json',
#         img_prefix=train_dir,
#         pipeline=train_pipeline,
#         classes=('fisheye_pig',)
#         ),
#     val=dict(
#         ann_file=val_dir + '/annotation_coco.json',
#         img_prefix=val_dir,
#         pipeline=test_pipeline,
#         classes=('fisheye_pig',)
#         ),
#     test=dict(
#         ann_file=val_dir + '/annotation_coco.json',
#         img_prefix=val_dir,
#         pipeline=test_pipeline,
#         classes=('fisheye_pig',)
#         ))


dataset_type = 'COCODataset'
# classes = ('balloon',)
train_dir = "../dataset/fish_eye_pig/train"
val_dir = "../dataset/fish_eye_pig/val"

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
data = dict(
    samples_per_gpu=1,
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
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=36)