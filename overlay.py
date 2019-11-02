# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np

def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name

def mask(raw, cornea, ulcer):
    # resize
    origin = raw
    #origin = cv2.resize(origin, (648, 432), interpolation=cv2.INTER_AREA)
    #cornea = np.expand_dims(cornea[:,:,0], -1)

    # draw a mask for a cornea
    ret, thresh = cv2.threshold(255-cornea, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(origin, contours, -1, (0, 0, 255), 4)

    # draw a mask for ulcers
    if ulcer is not None:    
        #ulcer = np.expand_dims(ulcer[:,:,0], -1)
        ret, thresh = cv2.threshold(255-ulcer, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = cv2.drawContours(img, contours, -1, (255, 255, 255), 4)

    return img

if __name__ == '__main__':
    img_path = './rawImages/'
    cornea_path = './corneaLabels/'
    ulcer_path = './ulcerLabels/'
    cornea_overlay_path = './corneaOverlay/'
    ulcer_overlay_path = './ulcerOverlay/'

    # generate cornea overlay images
    cornea_list = glob.glob(cornea_path + '*.png')
    for path in cornea_list:
        name = analyze_name(path)
        raw = cv2.imread(img_path + name + '.jpg')
        cornea = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        overlay = mask(raw, cornea, None)
        cv2.imwrite(cornea_overlay_path + name + '.jpg', overlay)
        print(name)

    # generate ulcer overlay images
    ulcer_list = glob.glob(ulcer_path + '*.png')
    for path in ulcer_list:
        name = analyze_name(path)
        raw = cv2.imread(img_path + name + '.jpg')
        cornea = cv2.imread(cornea_path + name + '.png', cv2.IMREAD_GRAYSCALE)
        ulcer = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        overlay = mask(raw, cornea, ulcer)
        cv2.imwrite(ulcer_overlay_path + name + '.jpg', overlay)
        print(name)

