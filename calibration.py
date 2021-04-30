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

def linear_transform(image, alpha, beta):
    new_image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)
    return new_image
    
#%%
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chess = (7, 7)
tile_size = 29
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess[0]*chess[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrationImages/Working/*.jpg')

for fname in images:
    print("Processing image %s:" %fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, gray = cv2.threshold(gray, 100 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # gray = linear_transform(gray, 1.5, 0)
    cv2.imshow('gray %s' %fname,gray)
    cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chess,None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, chess, corners2,ret)
        cv2.imshow('img %s' %fname,img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

#%% 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(mtx)
print(dist)
print(rvecs)
print(tvecs)