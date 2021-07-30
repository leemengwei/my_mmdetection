# CHECKS
# check setup (quick run):
python demo/my_image_demo.py /home/lmw/leemengwei/dataset/2出猪通道-泉州safe/val/passage_pig_1619081779123.jpg configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_2passage_pig.py work_dirs/yolov3_d53_mstrain-608_273e_coco_2passage_pig/latest.pth
python demo/my_image_demo.py ../dataset/2all_passage+cut_safe/val/passage_pig_1619081781450.jpg configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig/epoch_20_0123.pth

python demo/my_video_demo.py  ../dataset/videos/record.mp4 configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_fisheye_pig/latest.pth --show --out out.mp4
python demo/my_video_demo.py  /home/lmw/leemengwei/dataset/videos/泉州出猪通道/wave1.mp4 configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_2passage_pig.py work_dirs/yolov3_d53_mstrain-608_273e_coco_2passage_pig/latest.pth --show --out out.mp4
# check dataset:
python tools/misc/browse_dataset.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py
python tools/misc/browse_dataset.py configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_pig.py
# check config:
python tools/misc/print_config.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py

# TRAINS
# single train:
python tools/train.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py
python tools/train.py configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_pig.py  #

# multiple train:
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_pig.py 2  #

# Glance on train
# analyze train curve
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig/20210408_192018.log.json  --keys 0_bbox_mAP 0_bbox_mAP_50 # loss_rpn_cls loss_rpn_bbox loss_cls acc loss_bbox loss_mask loss
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/yolov3_d53_320_273e_coco_pig/20210414_195416.log.json  --keys bbox_mAP bbox_mAP_50 

# TESTS
# single test:
python tools/test.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py/latest.pth --eval bbox segm --out out.pkl --eval-options jsonfile_prefix=./tmp/tmp classwise=True
python tools/test.py configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_pig.py work_dirs/yolov3_d53_320_273e_coco_pig/latest.pth --eval bbox --out out.pkl --eval-options jsonfile_prefix=./tmp/tmp classwise=True
python tools/test.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig/latest.pth --eval segm --out seg.pkl --eval-options jsonfile_prefix=./tmp/tmp classwise=True  # or switch off 'eval' and specify: --format-only to get output json loop!

# ANALYSIS
# analyze Error
python tools/analysis_tools/coco_error_analysis.py tmp/tmp.segm.json results --ann=../dataset/4all_weights+cut_safe/val/annotation_coco.json --types='segm'
# analyze bad & good (convinient)
python tools/analysis_tools/analyze_results.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py out.pkl work_dirs/ #be sure update pkl
python tools/analysis_tools/analyze_results.py configs/fisheye_pig/yolov3_d53_mstrain-608_273e_coco_pig.py out.pkl work_dirs/ #be sure update pkl
python tools/analysis_tools/analyze_results.py configs/fisheye_pig/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_all_pig.py seg.pkl work_dirs/


# Publish
python tools/model_converters/publish_model.py work_dirs/yolov3_d53_mstrain-608_273e_coco_pig_path/latest.pth work_dirs/yolov3_d53_mstrain-608_273e_coco_pig_path/publish.pth



