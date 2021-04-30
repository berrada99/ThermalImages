#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 23:35:35 2021

@author: hassan
"""

import cv2
import numpy as np



#%%
def areaCalc(segemented_img):
    segemented_img = segemented_img//255
    shape = np.shape(segemented_img)
    Um = shape[0]
    Vm = shape[0]
    u0 = Um//2
    v0 = Vm//2
    area = 0
    for u in range(0, Um):
        for v in range(0, Vm):
            if (u != u0 and v != v0):    
                area += segemented_img[u, v]/(((u - u0)**3)*((v - v0)**2))
            
    return area

def calibrate(img_calib):
    c = (1/16)/areaCalc(img_calib)
    
def realAreaCalc(img_calib, segmented_img):
    calibrate(im)*areaCalc(segmented_img)
#%%
calib_image = cv2.imread(r'FLIR0172.jpg')
thermal_image = cv2.imread(r'FLIR0173.jpg')
cv2.imshow('image', thermal_image)


gray_img = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
print(np.shape(gray_img))
gray_img = gray_img[:225]

ret, segmented_img = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY)
cv2.imshow('image', segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

c = calibrate(segmented_img)
print(c)
