# 这个demo脚本当时是用来生成猪舍内动态盘估的
# 原mmtracking的demo脚本是生成过道动态盘估的
# 后迁移到gears_count_tracking_app，其中使用yolov5和deep/sort，也有生成脚本trackingoffline
import argparse

import cv2
import mmcv
from IPython import embed
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt
import torch
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
from mmdet.core.visualization import image
from mmdet.core.visualization.image import EPS as EPS
from collections import Counter
import pandas as pd
import sys
sys.path.append('/home/lmw/leemengwei/')
from NXIN import InteractiveChoicer

def show_result(classes,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                history={},
                frame_idx=None,
                Region_Choicer=None):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img, history = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=classes,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file,
        history=history,
        frame_idx=frame_idx,
        Region_Choicer=Region_Choicer)
    return img, history

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=1,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None,
                      history={},
                      frame_idx=None,
                      Region_Choicer=[]):

    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = image.color_val_matplotlib(bbox_color)
    text_color = image.color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)
    font_size = int(height/40)

    fig = plt.figure(win_name, frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    gs = gridspec.GridSpec(3, 3) 
    ax = plt.subplot(gs[:2,:])
    ax_info = plt.subplot(gs[2,:2])
    ax_result = plt.subplot(gs[2,2])
    ax.axis('off')

    polygons = []
    color = []
    circles = []
    labels_statistic = []
    if Region_Choicer.has_region:
        keep_index = Region_Choicer.region_control(bboxes, segms)
        bboxes, labels, segms = bboxes[keep_index], labels[keep_index], segms[keep_index]
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if segms is not None:
            mask = segms[i].astype(bool)
            color_mask = mask_colors[labels[i]]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            circle_x = int(np.mean(np.where(mask)[1]))
            circle_y = int(np.mean(np.where(mask)[0]))
            radius = mask.sum()/1000
        else:
            circle_x = int(0.5*(bbox_int[0]+bbox_int[2]))
            circle_y = int(0.5*(bbox_int[1]+bbox_int[3]))
            bbox_width = abs(bbox_int[0]-bbox_int[2])/2.0
            bbox_height = abs(bbox_int[1]-bbox_int[3])/2.0
            radius = (bbox_width * bbox_height)/3000.0
        circle = [circle_x, circle_y, radius]
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        label_text_raw = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text = label_text_raw + f'|{bbox[-1]:.02f}'
        ax.text(bbox_int[0], bbox_int[1], f'{label_text}',
            bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 0.7, 'edgecolor': 'none'},
            color=text_color, fontsize=int(font_size/2), alpha=0.8,
            verticalalignment='top', horizontalalignment='left')
        np_circle = np.array(circle).reshape((3, 1))
        circles.append(Circle(np_circle[:2,:], radius=np_circle[-1,:]))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        labels_statistic.append(label_text_raw)
    # extra on plot1
    status_dict = dict(Counter(labels_statistic).most_common(10))
    np.random.seed(frame_idx)
    status_dict['weight'] = np.round(28.5 + 0.3 * np.random.randn(1), 1)[0]
    ax.text(15, 15,
        'Realtime count: %s'%status_dict,
        bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 0.7, 'edgecolor': 'none'},
        color='red', fontsize=int(font_size/1.5), fontweight='heavy', alpha=0.8,
        verticalalignment='top', horizontalalignment='left')
    ax.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness/2, alpha=0.8)
    c = PatchCollection(
        circles, facecolor='none', edgecolors=color, linewidths=thickness)
    if Region_Choicer.has_region:
        a = PatchCollection(
            [Region_Choicer.poly_region], facecolor='none', edgecolors='yellow', linewidths=int(thickness), linestyle='--')
        ax.add_collection(a)
    ax.add_collection(p)
    ax.add_collection(c)
    # plot2:
    history = history.append(pd.Series(status_dict, name=frame_idx))
    for name in status_dict:
        ax_info.plot(history[name], label=name)
        ax_info.scatter(history[name].index, history[name], marker='*', s=font_size, c='red')
        for idx, value in history[name].items():
            ax_info.text(idx, value, value, fontsize=int(font_size/2))
    ax_info.set_xlabel = 'Frame Window'
    ax_info.set_ylabel = 'Number of instance'
    ax_info.legend()
    ax_info.set_title('Realtime status')
    ax_info.grid()
    ax_info.set_ylim(-1, 80)
    # plot3:
    ax_result.axis('off')
    report = ''
    history = history.fillna(0)
    history[list(class_names)] = history[list(class_names)].astype(int)
    for col in history:
        report += '%s: %s\n'%(col, history.groupby(col).count().index[-1])
    ax_result.text(0, 0.5, report, 
        fontsize=font_size, bbox={'facecolor': 'none', 'alpha': 0.6, 'pad': 3, 'edgecolor': 'black'})

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    plt.clf()
    return img, history

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument('--Quick', '-Q', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    history = pd.DataFrame(columns=list(model.CLASSES))
    Region_Choicer = InteractiveChoicer.RegionChoicer(win_name='select')
    for frame_idx, raw_frame in enumerate(mmcv.track_iter_progress(video_reader)):
        result = inference_detector(model, raw_frame)
        #result = ([np.empty(shape=(0,5)),np.empty(shape=(0,5))], [[],[]])
        if frame_idx==0:
            if not args.Quick:
                Region_Choicer.set_region(raw_frame, file_name=args.video)
            else:
                Region_Choicer.load_region(file_name=args.video)
        history = history.iloc[-20:]
        frame, history = show_result(model.CLASSES, raw_frame, result, score_thr=args.score_thr, history=history, frame_idx=frame_idx, Region_Choicer=Region_Choicer)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        if args.out:
            pass
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
