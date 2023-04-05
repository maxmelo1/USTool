import os
import cv2
import numpy as np


def overlay_mask(img, masks, colorMap, alpha = 0.7):
    '''
    img: source image colour usually
    mask: mask to overlay in image, black and white usually
    colorMap[0]: color of background
    alpha: alpha channels of image and masks
    '''

    #img = cv2.imread("frame1.bmp", 1)
    #mask = cv2.imread("mask1.bmp", 0)
    #mask = mask / 255.0

    if np.shape(img)[2] == 1:
        img = np.dstack((img,img,img))

    maskSum = np.zeros(np.shape(img))
    
    for i in range(len(masks)):
        maskColor = np.dstack((masks[i], masks[i], masks[i]))
        maskColor = colorMap[i]*maskColor
        
        maskSum += np.minimum(maskColor, 255)*alpha

    new_img = ( np.minimum(img + maskSum, 255) ).astype("uint8")

    return new_img 