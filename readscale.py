#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:09:40 2021

@author: hassan
"""
import cv2 
from pytesseract import pytesseract 
import numpy as np 
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#%%
def crop(im, x_min, x_max, y_min, y_max):
    return im[x_min:x_max, y_min:y_max]

def read_scale(im):
    
    """ Takes a cv opened gray image of a flir camera and returns temperatures"""
    im = cv2.bitwise_not(im)

    ret, im = cv2.threshold(im, 25, 255, cv2.THRESH_BINARY)

    im = cv2.erode(im, np.ones((2, 2), np.uint8))
    
    
    im_shape = np.shape(im)
    print(im_shape)
    
    im = crop(im, 0, 240, 270, 320)
    im[30:200] = [255, 255, 255]
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    list_element = pytesseract.image_to_string(im, lang='eng').split()
    
    """ Elimination des °C """
    for i in range(len(list_element)):
        element = list_element[i]
        if element[-2:] == '°C':
            list_element[i] = element[:-2]
    
    """ Retrieving temperatures """
    temps = []
    for element in list_element:
        try:
            temps.append(float(element))
        except:
            pass
        
    print(temps)
    temp_low = min(temps)
    temp_high = max(temps)
    return temp_low,temp_high

#%%

print(cv2.__version__)
im = cv2.imread(r'./gray_data.jpg')



# im = cv2.imread('testtext1.png')

# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
read_scale(im)