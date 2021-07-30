_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
classes=('pig', 'person') 

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.6,
            nms=dict(type='nms', iou_threshold=0.5),          # NOTE : base 训练先不跟猪舍盘估项目改head参数
            max_per_img=100,
            mask_thr_binary=0.5))
        )

# Modify dataset related settings
dataset_type = 'COCODataset'
dir_1 = "../dataset/1all_dorm+cut_safe/"
dir_1_1 = "../dataset/1all_dorm_BYZ+cut_safe/"
dir_roi = "../dataset/1all_dorm_BYZ_roi+cut_safe/"   #有头有尾
dir_1_2 = "../dataset/1huiyan_dorm_raw+cut_safe/"
dir_roi_1 = "../dataset/1huiyan_dorm_roi+cut_safe/"   #有头有尾
dir_2 = "../dataset/2all_passage+cut_safe/"
dir_3 = "../dataset/3all_stage+cut_safe/"
dir_4 = "../dataset/4all_weights+cut_safe/"   #有头有尾
dir_4_1 = "../dataset/4all_weights_BYZ_roi+cut_safe/"   #有头有尾

# pigs
pig_dirs = [dir_1, dir_1_1, dir_1_2, dir_2, dir_3, dir_4, dir_4_1]
pig_head_hip_dirs = [dir_roi, dir_roi_1, dir_4, dir_4_1]
safe_pigs_prefix_train = [i+'/train/' for i in pig_dirs]
safe_pigs_prefix_val = [i+'/val/' for i in pig_dirs]
safe_pigs_ann_train = [i+'/annotation_coco.json' for i in safe_pigs_prefix_train]   # NOTE: negative samples, empty gt, 负样本手动加到ann了吗！
safe_pigs_ann_val = [i+'/annotation_coco.json' for i in safe_pigs_prefix_val]
# head and hip
safe_head_and_hip_prefix_train = [i+'/train/' for i in pig_head_hip_dirs]
safe_head_and_hip_prefix_val = [i+'/val/' for i in pig_head_hip_dirs]
safe_head_and_hip_ann_train = [i+'/annotation_coco.json' for i in safe_head_and_hip_prefix_train]
safe_head_and_hip_ann_val = [i+'/annotation_coco.json' for i in safe_head_and_hip_prefix_val]

test_dir = "/home/lmw/leemengwei/dataset/images_nolabel_useful/panzhong/"
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.1),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.1),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.1)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.01),
            dict(type='MedianBlur', blur_limit=3, p=0.01)
        ],
        p=0.1),
    #dict(type='CopyPaste', blend=True, sigma=1, pct_objects_paste=0.5, p=1),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file = safe_pigs_ann_train,
        # ann_file= safe_head_and_hip_ann_train,
        img_prefix=safe_pigs_prefix_train,
        # img_prefix=safe_head_and_hip_prefix_train,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=True),
            dict(
                type='Resize',
                img_scale=[(1080, 1080)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Albu',
                transforms=albu_train_transforms,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap={
                    'img': 'image',
                    'gt_masks': 'masks',
                    'gt_bboxes': 'bboxes'
                },
                update_pad_shape=False,
                skip_img_without_anno=True),
            #dict(type='GlanceOnData', wait_sec=0.2),   # Glance 在normalize之前，否则就看不到了，很多是黑的（负值）
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=classes),
    val=dict(
        type='CocoDataset',
        ann_file = safe_pigs_ann_val,
        # ann_file = safe_head_and_hip_ann_val,
        img_prefix=safe_pigs_prefix_val,
        # img_prefix=safe_head_and_hip_prefix_val,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1080, 1080),
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
                    #dict(type='ImageToTensor', keys=['img']),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=classes),
    test=dict(
        type='CocoDataset',
        #ann_file = safe_pigs_ann_val,
        #ann_file = safe_head_and_hip_ann_val,
        #img_prefix=safe_pigs_prefix_val,
        #img_prefix=safe_head_and_hip_prefix_val,
        ann_file=test_dir + '/test_loop.json',
        img_prefix=test_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1080,1080)],
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
                    #dict(type='ImageToTensor', keys=['img']),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=classes))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[150, 175, 185])
#fp16=True
runner = dict(type='EpochBasedRunner', max_epochs=200)

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10)
workflow = [('train', 1), ('val', 1)]
