from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from IPython import embed
#config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
#checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

config_file = '../configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
checkpoint_file = '../checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = 'demo1.jpg'
result = inference_detector(model, img)
#model.show_result(img, result, score_thr=0.3)
show_result_pyplot(model, img, result, score_thr=0.3)
print(type(model))
#embed()

# test a video and show the results
video = mmcv.VideoReader('demo.mp4')
for idx,frame in enumerate(video):
    print(idx)
    result = inference_detector(model, frame)
    #model.show_result(frame, result, wait_time=1, out_file='result.jpg')
    show_result_pyplot(model, frame, result, wait_time=1)
