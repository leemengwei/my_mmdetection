_base_ = '../cascade_rcnn/cascade_mask_rcnn_eff-B7_nasfpn_1x_coco.py'
fp16 = dict(loss_scale=512.)
# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=[dict(num_classes=1), dict(num_classes=1), dict(num_classes=1)],
#         mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
train_dir = "../dataset/coco/train2017"
val_dir = "../dataset/coco/val2017"
#anno_dir = "/media/feng/584A3B6F4A3B4950/dataset/coco/ann2017"

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = '/home/feng/Desktop/projects/animal/mmdetection/tools/work_dirs/cascade_mask_rcnn_eff-B3_nasfpn_2017/latest.pth'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        ann_file=train_dir + '/instances_train2017.json',
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
                #img_scale=[(1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='MyCopyPaste'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            #dict(type='Pad', size_divisor=32),
            dict(type='Pad', size_divisor=128),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        #classes=('fisheye_pig',)
    ),
    val=dict(
        type='CocoDataset',
        ann_file=val_dir + '/instances_val2017.json',
        img_prefix=val_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                #img_scale=(658, 492),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    #dict(type='Pad', size_divisor=32),
                    dict(type='Pad', size_divisor=128),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        #classes=('fisheye_pig',)
    ),
    test=dict(
        type='CocoDataset',
        ann_file=val_dir + '/instances_val2017.json',
        img_prefix=val_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                #img_scale=(658, 492),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    #dict(type='Pad', size_divisor=32),
                    dict(type='Pad', size_divisor=128),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        #classes=('fisheye_pig',)
    ))

runner = dict(type='EpochBasedRunner', max_epochs=200)
total_epochs = 200
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=100, norm_type=2))
#classes = ('fisheye_pig',)
gpu_ids = range(0, 2)