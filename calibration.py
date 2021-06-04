#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:39:59 2021

@author: hassan
"""
#%%
import numpy as np
import cv2
import glob

# %%

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# %%

def linear_transform(image, alpha, beta):
    new_image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)
    return new_image
    
# %%
# termination criteria
def calibration():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # chess = (7, 7)
    # chess = (7, 4)
    # chess = (6, 9)
    chess = (7, 9)
    tile_size = 2
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess[0]*chess[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)*tile_size
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    # gray_imgs = []
    # img_names = []
    
    images = glob.glob('/home/hassan/ThermalImagesCv/calibrationImages6/*.jpg')
    # images  = glob.glob('/home/hassan/ThermalImagesCv/calibrationImages/Working/*.jpg')
    # print(images)
    for fname in images:
        print("Processing image %s:" %fname)
        img = cv2.imread(fname)
        img = ResizeWithAspectRatio(img, 576)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Show Images
        print("OK")
        # cv2.imshow('gray %s' %fname,gray)
        # cv2.waitKey(0)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chess,None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            print(True)
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # gray_imgs.append(gray)
            # img_names.append(fname)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chess, corners2,ret)
            # cv2.imwrite('calibrationImages/Chessboards/image.jpg')
            # cv2.imshow('img %s' %fname,img)
            # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()
    return objpoints, imgpoints


#%% 
# objpoints, imgpoints = calibration()
# shape = gray_imgs[-1].shape[::-1]
shape = (576, 432)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
