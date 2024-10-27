import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from helpers import display_image

img_path = 'data/messi.jpg'
messi_img = cv.imread(img_path)

img_path = 'data/ronaldo.jpeg'
ronaldo_img = cv.imread(img_path)

#############################################################
##                    Pyramid functions                    ##
#############################################################
def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(levels - 1):
        image = cv.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)

    for i in range(levels - 1):        
        gaussian_expanded = cv.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))

        laplacian = cv.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid

#############################################################
##                     Cropping images                     ##
#############################################################
messi_img_mid_width = messi_img.shape[1]//2
left_part_messi_img = messi_img[:, :messi_img_mid_width]

ronaldo_img_mid_width = ronaldo_img.shape[1]//2
right_part_ronaldo_img = ronaldo_img[:, ronaldo_img_mid_width:]

