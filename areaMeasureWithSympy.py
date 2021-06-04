#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:09:41 2021

@author: hassan
"""

import cv2
import numpy as np
import calibration as cl
from sympy import *
init_printing()

#%%
def areaCalc(segmented_img, A, C):
    
    # f = open('sfile.txt', 'w')
    umax = len(segmented_img)
    vmax = len(segmented_img[0])
    area = 0
    determinants = np.zeros((umax, vmax))
    detJacobPhi = getDetJacobPhi(A, C)
    for v in range(0, vmax):
        for u in range(0, umax):
            if segmented_img[u, v] == 255:
                s = float(calculateDetJacobPhi(detJacobPhi, u, v))
                determinants[u, v] = s
                area += s
                # f.write("%s " %str(s))
        # f.write("\n")
    # f.close()
    m = np.max(determinants)
    determinants *= 255.0/m
    determinants = determinants.astype('uint8')
    print(determinants)
    cv2.imshow('image', determinants)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return determinants


#%% Retrieving parameters of the camera

def getAC(objpoints, imgpoints, gray_imgs):
    
    index = -1
    gray = gray_imgs[index]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    rvec = rvecs[index]
    tvec = tvecs[index]
    print(tvecs)


    R = cv2.Rodrigues(rvec)[0]
    print("R" + str(R))
      
    T = tvec


    print(np.shape(R))
    print("R: " + str(R))
    print("T: " + str(T))
    print("K: " + str(K))

    M = np.concatenate((R, T), axis = 1)
    """ s(u, v) = A(Xw, Yw)"""
    A = np.linalg.inv(M[:2, :2])
    C = M[-1]
    print("A: " + str(A))
    return A, C, gray

def getBH(A, C, u, v):
    f1 = c[0]*A[0, 0]*u + c[0]*A[0, 1]*v - 1
    f2 = c[1]*A[0, 0]*u + c[1]*A[0, 1]*v
    f3 = c[3]*A[0, 0]*u + c[3]*A[0, 1]*v
    h1 = c[0]*A[1, 0]*u + c[0]*A[1, 1]*v
    h2 = c[1]*A[1, 1]*u + c[1]*A[1, 1]*v - 1
    h3 = c[3]*A[1, 0]*u + c[3]*A[1, 1]*v
    B = np.array([[f1, f2], [f3, f4]])
    H = np.array([[f3], [h3]])
    print("B: " + str(B))
    print("H: " + str(H))
    return B, H

def getDetJacobPhi(A, C):
    u, v = symbols('u v')
    phi = Matrix(
        [((C[1]*A[1, 0]*u + C[1]*A[1, 1]*v - 1)*(C[3]*A[0, 0]*u + C[3]*A[0, 1]*v) - (C[0]*A[1, 0]*u + C[0]*A[1, 1]*v)*(C[3]*A[1, 0]*u + C[3]*A[1, 1]*v))/((C[1]*A[1, 0]*u + C[1]*A[1, 1]*v - 1)*(C[0]*A[0, 0]*u + C[0]*A[0, 1]*v - 1) - (C[1]*A[0, 0]*u + C[1]*A[0, 1]*v)*(C[0]*A[1, 0]*u + C[0]*A[1, 1]*v)), 
         ((C[0]*A[0, 0]*u + C[0]*A[0, 1]*v - 1)*(C[3]*A[1, 0]*u + C[3]*A[1, 1]*v) - (C[1]*A[0, 0]*u + C[1]*A[0, 1]*v)*(C[3]*A[0, 0]*u + C[3]*A[0, 1]*v))/((C[1]*A[1, 0]*u + C[1]*A[1, 1]*v - 1)*(C[0]*A[0, 0]*u + C[0]*A[0, 1]*v - 1) - (C[1]*A[0, 0]*u + C[1]*A[0, 1]*v)*(C[0]*A[1, 0]*u + C[0]*A[1, 1]*v))
         ]
        )
    Jacobphi = phi.jacobian(Matrix([u, v]))
    detJacobphi = Jacobphi.det()
    return detJacobphi
    
def calculateDetJacobPhi(detJacobphi, U, V):
    u, v = symbols('u v')
    
    detJacobphiUV = detJacobphi.subs(u, U).subs([(u, U), (v, V)])

    return detJacobphiUV
    

objpoints, imgpoints, gray_imgs = cl.calibration()
A, C, gray = getAC(objpoints, imgpoints, gray_imgs)
    
#%% Segementing image

"""
cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


ret, segmented_img = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inv_img = 255 - segmented_img
ret, segmented_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


"""
cv2.imshow('image', inv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

determinants = areaCalc(inv_img, A, C)
#%% Tests

"""
A = np.array([[1, 0], [0, 1]])
C = np.array([0, 0, 0, 1])
detJacobPhi = getDetJacobPhi(A, C)

print(calculateDetJacobPhi(detJacobPhi, 1, 1))

A = np.array([[0, 1], [1, 0]])
C = np.array([0, 0, 0, 1])
detJacobPhi = getDetJacobPhi(A, C)

print(calculateDetJacobPhi(detJacobPhi, 1, 1))


A = np.array([[1, 0], [0, 1]])
C = np.array([1, 0, 0, 1])
detJacobPhi = getDetJacobPhi(A, C)

print(calculateDetJacobPhi(detJacobPhi, 2, 2))


A = np.array([[1, 0], [0, 1]])
C = np.array([0, 1, 0, 1])
detJacobPhi = getDetJacobPhi(A, C)

print(calculateDetJacobPhi(detJacobPhi, 2, 2))

A = np.array([[1, 0], [0, 1]])
C = np.array([1, 1, 0, 1])
detJacobPhi = getDetJacobPhi(A, C)

print(calculateDetJacobPhi(detJacobPhi, 2, 2))
"""

