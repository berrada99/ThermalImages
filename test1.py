#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:05:20 2021

@author: hassan
"""
# run ./test1.py -i FLIR0084.jpg

import sys, traceback
import cv2
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.", default=None)
    args = parser.parse_args()
    return args


### Main workflow
def main():
    # Get options
    args = options()

    pcv.params.debug=args.debug #set debug mode
    pcv.params.debug_outdir=args.outdir #set output directory

    # Read raw thermal data

    # Inputs:
    #   filename - Image file to be read (possibly including a path)
    #   mode - Return mode of image ("native," "rgb,", "rgba", "gray", or "flir"), defaults to "native"
    
    thermal_data,path,filename = pcv.readimage(filename='FLIR0084.jpg', mode="flir") #mode="native"
    gray_img  = thermal_data # cv2.cvtColor(thermal_data, cv2.COLOR_BGR2GRAY)

    # Rescale the thermal data to a colorspace with range 0-255

    # Inputs:
    #   gray_img - Grayscale image data 
    #   min_value - New minimum value for range of interest. default = 0
    #   max_value - New maximum value for range of interest. default = 255
    scaled_thermal_img = pcv.transform.rescale(gray_img=gray_img)
    # Works
    
    
    # Threshold the thermal data to make a binary mask

    # Inputs:
    #   gray_img - Grayscale image data 
    #   threshold- Threshold value (between 0-255)
    #   max_value - Value to apply above threshold (255 = white) 
    #   object_type - 'light' (default) or 'dark'. If the object is lighter than the background then standard 
    #                 threshold is done. If the object is darker than the background then inverse thresholding is done. 
    bin_mask = pcv.threshold.binary(gray_img=thermal_data, threshold=35, max_value=255, object_type='dark')


    # Identify objects

    # Inputs: 
    #   img - RGB or grayscale image data for plotting 
    #   mask - Binary mask used for detecting contours 
    id_objects, obj_hierarchy = pcv.find_objects(img=scaled_thermal_img, mask=bin_mask)


    # Define the region of interest (ROI) 

    # Inputs: 
    #   img - RGB or grayscale image to plot the ROI on 
    #   x - The x-coordinate of the upper left corner of the rectangle 
    #   y - The y-coordinate of the upper left corner of the rectangle 
    #   h - The height of the rectangle 
    #   w - The width of the rectangle 
    roi, roi_hierarchy= pcv.roi.rectangle(img=scaled_thermal_img, x=20, y=20, h=50, w=50)


    # Decide which objects to keep

    # Inputs:
    #   img - RGB or grayscale image data to display kept objects on 
    #   roi_contour - contour of ROI, output from pcv.roi.rectangle in this case
    #   object_contour - Contour of objects, output from pcv.roi.rectangle in this case 
    #   obj_hierarchy - Hierarchy of objects, output from pcv.find_objects function
    #   roi_type - 'partial' (for partially inside, default), 'cutto', or 'largest' (keep only the largest contour)
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=scaled_thermal_img,roi_contour=roi,
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy, 
                                                                  roi_type='cutto')
                                                                  
    # kept_mask = None : doesn't work


                                                       
    ##### Analysis #####

    # Analyze thermal data 

    # Inputs:
    #   img - Array of thermal values
    #   mask - Binary mask made from selected contours
    #   histplot - If True plots histogram of intensity values (default histplot = False)
    #   label - Optional label parameter, modifies the variable name of observations recorded 
    analysis_img = pcv.analyze_thermal_values(thermal_array=thermal_data.astype('uint8'), mask=kept_mask, histplot=True, label="default")


    # Pseudocolor the thermal data 

    # Inputs:
    #     gray_img - Grayscale image data
    #     obj - Single or grouped contour object (optional), if provided the pseudocolored image gets 
    #           cropped down to the region of interest.
    #     mask - Binary mask (optional) 
    #     background - Background color/type. Options are "image" (gray_img, default), "white", or "black". A mask 
    #                  must be supplied.
    #     cmap - Colormap
    #     min_value - Minimum value for range of interest
    #     max_value - Maximum value for range of interest
    #     dpi - Dots per inch for image if printed out (optional, if dpi=None then the default is set to 100 dpi).
    #     axes - If False then the title, x-axis, and y-axis won't be displayed (default axes=True).
    #     colorbar - If False then the colorbar won't be displayed (default colorbar=True)
    pseudo_img = pcv.visualize.pseudocolor(gray_img = thermal_data, mask=kept_mask, cmap='viridis', 
                                           min_value=31, max_value=35)

    # Write shape and thermal data to results file
    pcv.outputs.save_results(filename=args.result)

if __name__ == '__main__':
    main()