_base_ = '../cascade_rcnn/cascade_mask_rcnn_eff-B7_nasfpn_1x_coco.py'
fp16 = dict(loss_scale=512.)
# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=[dict(num_classes=1), dict(num_classes=1), dict(num_classes=1)],
#         mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
# classes = ('balloon',)
train_dir = "/media/feng/584A3B6F4A3B4950/dataset/fish_eye_pig/train"
val_dir = "/media/feng/584A3B6F4A3B4950/dataset/fish_eye_pig/val"

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
                #img_scale=[(658, 492)],
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
        classes=('fisheye_pig',)),
    val=dict(
        type='CocoDataset',
        ann_file=val_dir + '/annotation_coco.json',
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
        classes=('fisheye_pig',)),
    test=dict(
        type='CocoDataset',
        ann_file=val_dir + '/annotation_coco.json',
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
        classes=('fisheye_pig',)))

runner = dict(type='EpochBasedRunner', max_epochs=200)
total_epochs = 200
classes = ('fisheye_pig',)
gpu_ids = range(0, 2)