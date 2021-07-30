_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# We also need to change the num_classes in head to match the dataset's annotation
#classes=('pig_head', 'pig_hip')
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
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
        )

# Modify dataset related settings
dataset_type = 'COCODataset'
dir_1 = "../dataset/1all_dorm+cut_safe/"
dir_1_2 = "../dataset/1huiyan_dorm_raw+cut_safe/"
dir_1_3 = "../dataset/1huiyan_dorm_roi+cut_safe/"
dir_2 = "../dataset/2all_passage+cut_safe/"
dir_3 = "../dataset/3all_stage+cut_safe/"
dir_4 = "../dataset/4all_weights+cut_safe/"
# pigs
safe_pigs_prefix_train = [i+'/train/' for i in [dir_1, dir_1_2, dir_1_3, dir_2, dir_3, dir_4]]        # 错误！ roi+cut_safe 不能用
safe_pigs_prefix_val = [i+'/val/' for i in [dir_1, dir_1_2, dir_1_3, dir_2, dir_3, dir_4]]
safe_pigs_ann_train = [i+'/annotation_coco.json' for i in safe_pigs_prefix_train]
safe_pigs_ann_val = [i+'/annotation_coco.json' for i in safe_pigs_prefix_val]
# head and hip
safe_head_and_hip_prefix_train = [i+'/train/' for i in [dir_1_3, dir_4]]
safe_head_and_hip_prefix_val = [i+'/val/' for i in [dir_1_3, dir_4]]
safe_head_and_hip_ann_train = [i+'/annotation_coco.json' for i in safe_head_and_hip_prefix_train]
safe_head_and_hip_ann_val = [i+'/annotation_coco.json' for i in safe_head_and_hip_prefix_val]

test_dir = "/home/lmw/leemengwei/dataset/images_nolabel_useful/panzhong/"
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file = safe_pigs_ann_train,
        #ann_file= safe_head_and_hip_ann_train,
        img_prefix=safe_pigs_prefix_train,
        #img_prefix=safe_head_and_hip_prefix_train,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='Resize',
                img_scale=[(1080, 1080)],
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
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        classes=classes),
    val=dict(
        type='CocoDataset',
        ann_file = safe_pigs_ann_val,
        #ann_file = safe_head_and_hip_ann_val,
        img_prefix=safe_pigs_prefix_val,
        #img_prefix=safe_head_and_hip_prefix_val,
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
