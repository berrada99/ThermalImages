#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:09:41 2021

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


#%% Retrieving parameters of the camera

def getAC(objpoints, imgpoints):
    
    index = 15
    # gray = gray_imgs[index]
    # img_name = img_names[index]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1]),None,None)
    rvec = rvecs[index]
    tvec = tvecs[index]
    # print(tvecs)


    R = cv2.Rodrigues(rvec)[0]
    # print("R" + str(R))
      
    T = tvec


    # print(np.shape(R))
    # print("R: " + str(R))
    # print("T: " + str(T))
    # print("K: " + str(K))
    
    # GROSSE ERREUR : IL FAUT MULTIPLIER PAR K !!!
    
    M = np.matmul(K, np.concatenate((R, T), axis = 1))
    """ s(u, v) = A(Xw, Yw)"""
    A = np.linalg.inv(M[:2, :2])
    C = M[-1]
    # print("A: " + str(A))
    return A, C, gray, img_name

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
    
    
#%% Segementing image

objpoints, imgpoints = cl.calibration()
A, C, gray, img_name = getAC(objpoints, imgpoints, gray_imgs, img_names)


print(img_name)
"""
cv2.imshow('image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


# ret, segmented_img = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, segmented_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
inv_img = 255 - segmented_img



"""
cv2.imshow('image', inv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
area = areaCalc(inv_img, A, C)
print("Aire des carreaux blancs calculée : " + str(area) + " cm2")
# print("Aire des carreaux blancs réelle   : " + str(4*8*2.9**2) + " cm2")
print("Aire des carreaux blancs réelle   : " + str(5*8*2**2) + " cm2")

#%% Tests

"""
A = np.array([[1, 0], [0, 1]])
C = np.array([0, 0, 0, 1])
print(getDetJacobPhi(A, C, 1, 1))

A = np.array([[0, 1], [1, 0]])
C = np.array([0, 0, 0, 1])
print(getDetJacobPhi(A, C, 1, 1))


A = np.array([[1, 0], [0, 1]])
C = np.array([1, 0, 0, 1])
print(getDetJacobPhi(A, C, 2, 2))


A = np.array([[1, 0], [0, 1]])
C = np.array([0, 1, 0, 1])
print(getDetJacobPhi(A, C, 2, 2))

A = np.array([[1, 0], [0, 1]])
C = np.array([1, 1, 0, 1])
print(getDetJacobPhi(A, C, 2, 2))

"""
