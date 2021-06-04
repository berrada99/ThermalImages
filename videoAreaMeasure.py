#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:25:58 2021

@author: hassan
"""

import cv2 
import numpy as np 
import glob
GRAY_SHAPE = (576, 324)

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
    
    images = glob.glob('/home/hassan/ThermalImagesCv/calibrationImages5/calibrate/*.jpg')
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


# %% 

def getAC(objpoints, imgpoints):
    
    # index = 15
    # gray = gray_imgs[index]
    # img_name = img_names[index]
    global GRAY_SHAPE
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, GRAY_SHAPE, None, None)
    rvec = np.mean(rvecs, axis = 0)
    tvec = np.mean(tvecs, axis = 0)


    R = cv2.Rodrigues(rvec)[0]
    T = tvec

    # GROSSE ERREUR : IL FAUT MULTIPLIER PAR K !!!
    
    M = np.matmul(K, np.concatenate((R, T), axis = 1))
    """ s(u, v) = A(Xw, Yw)"""
    A = np.linalg.inv(M[:2, :2])
    C = M[-1]
    return A, C


# %%
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

def getDetJacobPhi2(A, C, U, V):
    
    # Le calcul est correct, la formule est incorrecte. 
    
    
    b11 = C[0]*A[0, 0]*U + C[0]*A[0, 1]*V - 1
    b12 = C[1]*A[0, 0]*U + C[1]*A[0, 1]*V
    b21 = C[0]*A[1, 0]*U + C[0]*A[1, 1]*V
    b22 = C[1]*A[1, 0]*U + C[1]*A[1, 1]*V - 1
    
    # B = np.array([[b11, b12], [b21, b22]])
    # print("B: " + str(B))
    
    f3 = C[3]*A[0, 0]*U + C[3]*A[0, 1]*V
    h3 = C[3]*A[1, 0]*V + C[3]*A[1, 1]*V
    
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
    det2 = np.square(det)
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
    
    
    # print("JacobPhi: " + str(JacobPhi))
    detJacobPhi = np.abs(j11*j22 - j21*j22)
    
    return detJacobPhi

# %% 

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
    # area2 = np.sum(np.multiply((segmented_img == 255), getDetJacobPhi2(segmented_img, A, C)))
    # f.close()
    # m = np.max(determinants)
    # determinants *= 255.0/m
    # determinants = determinants.astype('uint8')
    # print(determinants)
    # cv2.imshow('image', segmented_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(area)
    return area


def areaCalc2(segmented_img, A, C, detJacobPhi):
    area = np.sum(np.multiply((segmented_img == 255), detJacobPhi)
    return area

# %% Reading video file 

font = cv2.FONT_HERSHEY_SIMPLEX

# calibrating camera and getting parameters
objpoints, imgpoints = calibration()

# Getting intrinsic and extrinsic parameters
A, C = getAC(objpoints, imgpoints)
# Getting matrixes of pixels
umax = GRAY_SHAPE[0]
vmax = GRAY_SHAPE[1]

U = np.tile(np.arange(0, umax), (vmax,1))
V = np.tile(np.reshape(np.arange(0, vmax), (vmax, 1)), (1, umax))

detJacobPhi = getDetJacobPhi2(A, C, U, V)


# Opening video 
cap = cv2.VideoCapture('/home/hassan/ThermalImagesCv/ThermalVideos/test.avi')

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Processing image
    frame = ResizeWithAspectRatio(frame, 576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, segmented_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area = areaCalc2(segmented_img, A, C, detJacobPhi)
    print(area)
    print(A, C)
    
    # Putting text
    image = cv2.putText(frame, "Area of Water = " + str(area) + " cm2",(300,300), font, 0.5, (0, 0, 255),2,cv2.LINE_AA)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# for "each" image : calculate area 
# draw the results in output 

