#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:17:25 2021

@author: hassan
"""

import cv2
import numpy as np
import calibration as cl



#%%

def areaCalc(segmented_img, A, C):
    
    # f = open('sfile.txt', 'w')
    umax = len(segmented_img)
    vmax = len(segmented_img[0])
    area = 0
    determinants = np.zeros((umax, vmax))
    for v in range(0, vmax):
        for u in range(0, umax):
            if segmented_img[u, v] == 255:
                # s = getDetJacobPhi(A, C, u, v)
                # determinants[u, v] = s
                area += getDetJacobPhi(A, C, u, v)
                # f.write("%s " %str(s))
        # f.write("\n")
    # f.close()
    # m = np.max(determinants)
    # determinants *= 255.0/m
    # determinants = determinants.astype('uint8')
    # print(determinants)
    cv2.imshow('image', segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return area

def getAC(objpoints, imgpoints):
    
    index = 0
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (576, 432),None,None)
    rvec = rvecs[index]
    # tvec = tvecs[index]
    tvec = np.mean(tvecs, axis = 0)
    rvec = np.mean(rvecs, axis = 0)

    R = cv2.Rodrigues(rvec)[0]
    # print("R" + str(R))
      
    T = tvec

    # MULTIPLICATION AVEC K
    
    M = np.matmul(K, np.concatenate((R, T), axis = 1))
    """ s(u, v) = A(Xw, Yw)"""
    A = np.linalg.inv(M[:2, :2])
    C = M[-1]
    # print("A: " + str(A))
    return A, C

def getDetJacobPhi(A, C, u, v):
    
    # Le calcul est correct, la formule est incorrecte. 
    b11 = C[0]*A[0, 0]*u + C[0]*A[0, 1]*v - 1
    b12 = C[1]*A[0, 0]*u + C[1]*A[0, 1]*v
    b21 = C[0]*A[1, 0]*u + C[0]*A[1, 1]*v
    b22 = C[1]*A[1, 0]*u + C[1]*A[1, 1]*v - 1
    
    B = np.array([[b11, b12], [b21, b22]])
    # print("B: " + str(B))
    
    f3 = C[3]*A[0, 0]*u + C[3]*A[0, 1]*v
    h3 = C[3]*A[1, 0]*u + C[3]*A[1, 1]*v
    
    db11u = C[0]*A[0, 0]
    db12u = C[1]*A[0, 0]
    db21u = C[0]*A[1, 0]
    db22u = C[1]*A[1, 0]
    df3u  = C[3]*A[0, 0]
    dh3u  = C[3]*A[1, 0]
    db11v = C[0]*A[0, 1]
    db12v = C[1]*A[0, 1]
    db21v = C[0]*A[1, 1]
    db22v = C[1]*A[1, 1]
    df3v  = C[3]*A[0, 1]
    dh3v  = C[3]*A[1, 1]

    
    
    det = b11*b22 - b12*b21
    det2 = det**2
    ddetu = db11u*b22 + b11*db22u - db12u*b21 - b12*db21u
    ddetv = db11v*b22 + b11*db22v - db12v*b21 - b12*db21v
    
    j11 = - (1/det2)*((df3u*b22 + f3*db22u - dh3u*b21 - h3*db21u)*det
                    - (ddetu*(f3*b22 - h3*b21)) ) 
    j12 = - (1/det2)*((df3v*b22 + f3*db22v - dh3v*b21 - h3*db21v)*det
                    - (ddetv*(f3*b22 - h3*b21))) 
    
    j21 = - (1/det2)*((-df3u*b12 - f3*db12u + dh3u*b11 + h3*db11u)*det
                    + (ddetu*(f3*b12 - h3*b11))) 
    j22 = - (1/det2)*((-df3v*b12 - f3*db12v + dh3v*b11 + h3*db11v)*det
                    + (ddetv*(f3*b12 - h3*b11))) 
    
    JacobPhi = np.array([[j11, j12], [j21, j22]])
    # print("JacobPhi: " + str(JacobPhi))
    detJacobPhi = np.abs(np.linalg.det(JacobPhi))
    
    return detJacobPhi

#%%
objpoints, imgpoints= cl.calibration()
A, C = getAC(objpoints, imgpoints)

#%%
"""
cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
img = cv2.imread("/home/hassan/ThermalImagesCv/calibrationImages5/20210602_100906.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cl.ResizeWithAspectRatio(gray, 576)

# ret, segmented_img = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, segmented_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
inv_img = 255 - segmented_img


area = areaCalc(inv_img, A, C)
# print("Aire des carreaux blancs calculée : " + str(area) + " cm2")
# print("Aire des carreaux blancs réelle   : " + str(4*8*2.9**2) + " cm2")
# print("Aire des carreaux blancs réelle   : " + str(5*8*2**2) + " cm2")
print("Aire de l'objet estimée ': " + str(area) + " cm2")
# print("Aire de l'objet réelle': " + str(13*12 - 9*8) + " cm2")
# print("Aire de l'objet réelle': " + str(13.8*7) + " cm2")
# print("Aire de l'objet réelle': " + str(31*21) + " cm2")
print("Aire de l'objet réelle': " + str(13.5*12.7 - 9.5*8 + 14.5*13.5 - 11*8 + 13.5*15.1 - 8*11) + " cm2")
