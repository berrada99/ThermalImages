#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:58:17 2021

@author: hassan
"""

import sys, traceback
import cv2
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
# from readscale import read_scale

#%%
def qsigmoid(image, q = 0.35, alpha = 45, beta = 65, L = 255): 
    
    # q in (0, 1) : higher q means spikier function
    # alpha serves as the interval "accepted"  
    # beta the temperature target
    
    # Ajout de *2 par rapport à la version original de l'article (Version trouvée dans un autre article)
    return 2*L/(1 + np.power(1 + (1 - q)*np.abs((image - beta)/alpha), 1/(1 - q)))

#%%¨
thermal_data,path,filename = pcv.readimage(filename='./calibrationImages/Working/FLIR0213.jpg', mode="flir") #mode="native"
# thermal_data = cv2.imread('segemented_img.jpg')


gray_img = cv2.cvtColor(thermal_data, cv2.COLOR_BGR2GRAY)
# gray_img = thermal_data

_ = 0
ret, segmented_img = cv2.threshold(gray_img, _ , 255 , cv2.THRESH_OTSU)
segmented_img = 255 - segmented_img
cv2.imshow('image', segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Pourcentage de blanc dans la photo : " + str((np.sum(segmented_img)/(255*240*320))*100))

plt.hist(gray_img.ravel(),256,[0,256])
# plt.hist(sigmoid_img.ravel(),256,[0,256])
# plt.hist(segmented_img.ravel(),256,[0,256])
plt.show()

#%%
# cv2.imwrite('calibrationImages/Chessboards/P001.jpg', gray_img)
# cv2.imwrite('FLIR0223_sigmoid_data.jpg', sigmoid_img)
cv2.imwrite('calibrationImages/Chessboards/P007.jpg', segmented_img)


# img = cv2.imread('FLIR_0223_gray_data.jpg')
# img = cv2.imread('calibrationImages/Chessboards/P001.jpg')
# cv2.imshow('image', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#%% sigmoid plotting

# X = np.linspace(0, 255, 1000)
# y = qsigmoid(X)


# plt.plot(X, y, '-r')
# plt.show()