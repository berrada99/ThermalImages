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

    area = 0
    vmax = len(segmented_img)
    umax = len(segmented_img[0])
    for v in range(0, vmax):
        for u in range(0, umax):
            if segmented_img[v, u] == 255:
                s = np.dot(C, np.array([u, v, 0, 1]))
                print("s = " + str(s))
                
                
                
#%%

objpoints, imgpoints, gray = cl.calibration()
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
rvec = rvecs[-1]
tvec = tvecs[-1]


R = cv2.Rodrigues(rvec)[0]
print("R" + str(R))
      
# Could remove
T = np.zeros((3, 1), dtype = np.float32)
T[2, 0] = tvec[2, 0]


print(np.shape(R))
Rxy = R[:, :2]
print("R: " + str(R))
print("T: " + str(T))
print("K: " + str(K))

RxyT = np.concatenate((Rxy, T), axis = 1)
print("RxyT: " + str(RxyT))
print("K*M: " + str(K*RxyT))
A = np.linalg.inv(K*RxyT)
print("A: " + str(A))
print(A*K*RxyT)

