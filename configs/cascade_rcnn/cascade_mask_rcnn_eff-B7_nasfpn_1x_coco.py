_base_ = './cascade_mask_rcnn_eff-B7_nasfpn_101x_coco.py'
#fp16 = dict(loss_scale=512.)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    #pretrained='open-mmlab://resnext101_32x4d',
    # backbone=dict(
    #     #_delete_=True,
    #     type='EfficientNet',
    #     model_type='efficientnet-b0',
    #     out_indices=(0, 1, 3, 6)),
    neck=dict(
        type='NASFPN',
        #in_channels=[256, 512, 1024, 2048],
        #in_channels=[24, 40, 112, 1280],  #b0
        in_channels=[48, 80, 224, 2560], #b7
        #in_channels=[32, 48, 136, 1536],  #b3
        out_channels=256,
        num_outs=5,
        stack_times=7,
        norm_cfg=norm_cfg),
    )
