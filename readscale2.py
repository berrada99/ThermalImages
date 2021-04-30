#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:17:47 2021

@author: hassan
"""
import os 
import cv2 
from pytesseract import pytesseract 
import numpy as np 
pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#%%
def crop(im, x_min, x_max, y_min, y_max):
    return im[x_min:x_max, y_min:y_max]

def read_scale(im):
    
    """ Takes a cv opened gray image of a flir camera and returns temperatures """
    im_shape = np.shape(im)
    
    """ First cropping image to keep the scale only """
    im = crop(im, 0, im_shape[0], (im_shape[1]*4)//5, im_shape[1])
    
    """ inverting image in order to have a black over wight font """
    im = cv2.bitwise_not(im)
    
    """ Manually putting white on the scale to prevent threshold from not whitening the
    top of the temp scale """
    im[im_shape[0]//8:im_shape[0]*7//8] = 255

    """ Applying the threshold """
    ret, im = cv2.threshold(im, 25, 255, cv2.THRESH_BINARY)
    
    """ Eroding the image so that temperatures are more visible """
    im = cv2.erode(im, np.ones((2, 2), np.uint8))
    
    """ Expanding the image to facilitate detection """
    im_shape = np.shape(im)
    im = cv2.resize(im, (2*im_shape[1], 2*im_shape[0]))
    
    """ This will allow you to test the image feel free to copy paste it anywhere
    in the code to see the image """
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    """ Using pytesseract to detect temperatures """
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    list_element = pytesseract.image_to_string(im, lang='eng').split()
    

    """ Retrieving temperatures """
    temps = []
    for element in list_element:
        try:
            temps.append(float(element))
        except:
            pass
        
    try:
        temp_low = min(temps)
        temp_high = max(temps)
    except:
        temp_low = -400
        temp_high = -400
    
    print(temps)
    return temp_low,temp_high

#%%

print(cv2.__version__)
im = cv2.imread(r'./gray_data.jpg')

# im = cv2.imread('testtext1.png')

# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# read_scale(im)

#%% Testing read scale on all images

basepath = r'./'
list_images = []
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)) and str(entry).endswith('.jpg'):
        list_images.append(entry)
        
for im_path in list_images:
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    read_scale(im)

# im = cv2.imread(list_images[2])
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# read_scale(im)

# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()