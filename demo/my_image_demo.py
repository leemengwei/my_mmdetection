from argparse import ArgumentParser
from mmdet.core import post_processing
import sys, os
from IPython import embed
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import tqdm
import re
import pickle
import seaborn as sns
from scipy import stats
import numpy as np
# from NXIN import mask_to_poly
# from NXIN import InfoRecorder
# from NXIN import my_show_images
import matplotlib.pyplot as plt
import cv2
import dill

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # pig_index = model.CLASSES.index('pig')
    # # test a single image
    # Recorder = InfoRecorder.InfoRecorder()
    idx = 0
    if os.path.isdir(args.img):
        for root, dirs, files in os.walk(args.img):
            print("Working on %s"%root)
            for img in tqdm.tqdm(files):
                path_img = os.path.join(root, img)
                result = inference_detector(model, path_img)
                # show the results
                show_result_pyplot(model, path_img, result, score_thr=args.score_thr)
    else:
        result = inference_detector(model, args.img)
        # show the results
        show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
