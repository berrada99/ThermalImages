#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:44:34 2021

@author: hassan
"""

import cv2
import numpy as np
import calibration as cl

#%%
def areaCalc(segmented_img):
    pass


#%%

objpoints, imgpoints, gray = cl.calibration()
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
rvec = rvecs[-1]
tvec = tvecs[-1]


rmat = cv2.Rodrigues(rvec)
      
# Could remove
T = np.zeros((3, 1), dtype = np.float32)
T[2, 0] = tvec[2, 0]

R = np.reshape(rmat[:1], (3, 3))

print(np.shape(T))
print(np.shape(R))
M = np.concatenate((R, T), axis = 1)
A = np.linalg.inv(K*M)
print(A)

