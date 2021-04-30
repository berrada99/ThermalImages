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
from readscale import read_scale

#%%
def qsigmoid(image, q = 0.35, alpha = 45, beta = 65, L = 255): 
    
    # q in (0, 1) : higher q means spikier function
    # alpha serves as the interval "accepted"  
    # beta the temperature target
    
    # Ajout de *2 par rapport à la version original de l'article (Version trouvée dans un autre article)
    return 2*L/(1 + np.power(1 + (1 - q)*np.abs((image - beta)/alpha), 1/(1 - q)))

#%%¨
thermal_data,path,filename = pcv.readimage(filename='FLIR0142.jpg', mode="flir") #mode="native"
# thermal_data = cv2.imread('segemented_img.jpg')
print(np.shape(thermal_data))


gray_img = cv2.cvtColor(thermal_data, cv2.COLOR_BGR2GRAY)
# gray_img = gray_img.astype('float64')

# sigmoid_img = qsigmoid(gray_img)


# ret, segmented_img = cv2.threshold(sigmoid_img, 1, 255, cv2.THRESH_BINARY)
_ = 0
ret, segmented_img = cv2.threshold(sigmoid_img, _ ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('image', segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
print("Pourcentage de blanc dans la photo : " + str((np.sum(segmented_img)/(255*240*320))*100))

plt.hist(gray_img.ravel(),256,[0,256])
# plt.hist(sigmoid_img.ravel(),256,[0,256])
# plt.hist(segmented_img.ravel(),256,[0,256])
plt.show()

#%%
# cv2.imwrite('gray_data.jpg', gray_img)
# cv2.imwrite('sigmoid_data.jpg', sigmoid_img)
# cv2.imwrite('segmented_img.jpg', segmented_img)


img = cv2.imread('gray_data.jpg')
img = cv2.imread('segmented_img.jpg')
cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% sigmoid plotting

X = np.linspace(0, 255, 1000)
y = qsigmoid(X)


plt.plot(X, y, '-r')
plt.show()