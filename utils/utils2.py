import cv2
from queue import Queue
from threading import Thread
import os
from config import imshape
import json
import numpy as np
from config import hues, labels, imshape, mode
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from tensorflow.keras.utils import to_categorical


class VideoStream:
    def __init__(self, device=0, size=100):
        self.stream = cv2.VideoCapture(device)
        self.stream.set(cv2.CAP_PROP_FPS, 15)
        self.stopped = False
        self.queue = Queue(maxsize=size)

    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while self.stopped is False:

            if not self.queue.full():
                (grabbed, frame) = self.stream.read()

            if not grabbed:
                self.stop()
                return

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def check_queue(self):
        return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True
        self.stream.release()

def load_img_from_json(annot_path):
    with open(annot_path) as handle:
        data = json.load(handle)

    return data['imagePath']


def generate_missing_json():

    # creates a background json for the entire image if missing
    # this assumes you will never annotate a background class

    for im in os.listdir('dataset/QualiCarnes/images'):
        fn = im.split('.')[0]+'.json'
        path = os.path.join('dataset/QualiCarnes/annotated', fn)

        if os.path.exists(path) is False:
            json_dict = {}

            # these points might be reversed if not using a square image (idk)
            json_dict['shapes'] = [{"label": "background",
                                    "points": [[0,0],
                                               [0, imshape[0]-1],
                                               [imshape[0]-1, imshape[1]-1],
                                               [imshape[0]-1, 0]]
                                    }]
            with open(path, 'w') as handle:
                json.dump(json_dict, handle, indent=2)


def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    print(f'len labels: {len(labels)}')

    for i, label in enumerate(labels):
        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,i].astype(np.uint8)

        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)

    return blank


def crf(im_softmax, im_rgb):
    n_classes = im_softmax.shape[2]
    feat_first = im_softmax.transpose((2, 0, 1)).reshape(n_classes, -1)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    im_rgb = np.ascontiguousarray(im_rgb)

    d = dcrf.DenseCRF2D(im_rgb.shape[1], im_rgb.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=im_rgb,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((im_rgb.shape[0], im_rgb.shape[1]))
    if mode is 'binary':
        return res * 255.0
    if mode is 'multi':
        res_hot = to_categorical(res) * 255.0
        res_crf = add_masks(res_hot)
        return res_crf

def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    print(f'size segm: {segm.shape}')
    print(f'mask segm: {masks.shape}')
    for i, c in enumerate(cl):
        masks[i, :, :] = segm[:, :] == c
    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise
    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

def get_not_borders(curr_gt_mask):
    borders_dilation = binary_dilation(curr_gt_mask, iterations=3).astype(curr_gt_mask.dtype)
    borders_erosion = binary_erosion(curr_gt_mask, iterations=3).astype(curr_gt_mask.dtype)
    borders = np.bitwise_not(np.logical_or(borders_dilation != curr_gt_mask, borders_erosion != curr_gt_mask))
    return borders

def pixel_area(eval_segm, gt_segm, cl=None):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = extract_classes(gt_segm)
    else:
        n_cl = len(cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    area_gt = list([0]) * n_cl
    area_eval = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        area_eval[i] = np.sum(curr_eval_mask)
        area_gt[i] = np.sum(curr_gt_mask)

    return area_eval, area_gt

def mean_accuracy(eval_segm, gt_segm, ignore_border=False, cl=None):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl = extract_classes(gt_segm)
    else:
        n_cl = len(cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if ignore_border:
            not_borders = get_not_borders(curr_gt_mask)

            n_ii = np.logical_and(curr_eval_mask, curr_gt_mask)
            n_ii = np.sum(n_ii[not_borders])
            t_i  = np.sum(curr_gt_mask[not_borders])
        else:
            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i
        else:
            accuracy[i] = 1.

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_, accuracy

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_IU(eval_segm, gt_segm, ignore_border=False, cl=None):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    if cl is None:
        cl, n_cl   = union_classes(eval_segm, gt_segm)
    else:
        n_cl = len(cl)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if ignore_border:
            not_borders = get_not_borders(curr_gt_mask)
            n_ii = np.logical_and(curr_eval_mask, curr_gt_mask)
            n_ii = np.sum(n_ii[not_borders])

            t_i  = np.sum(curr_gt_mask[not_borders])
            n_ij = np.sum(curr_eval_mask[not_borders])
        else:
            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

        if (n_ij == 0) or (t_i) == 0:
            IU[i] = 1.
            continue

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, IU

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_