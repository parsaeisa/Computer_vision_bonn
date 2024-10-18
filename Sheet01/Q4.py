import numpy as np
import cv2 as cv
from ex_1_1 import display_image
import matplotlib.pyplot as plt

SIZE = 7

#############################################################
##         Reading the image and make it grayscale         ##
#############################################################
img_path = 'bonn.png'
img = cv.imread(img_path)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
display_image(gray_img)

#############################################################
##                   Using gaussianBlur                    ##
#############################################################
sigma = 2 * np.sqrt(2)
blurred = cv.GaussianBlur(gray_img, (SIZE, SIZE), sigmaX=sigma)

#############################################################
#               implement getGaussianKernel                 #
#############################################################
def getGaussianKernel(size):
    mid = size/2
    k = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x = i - mid
            y = j - mid

            k[i,j] = np.exp(-1 * (x**2 + y**2) / (2 * sigma**2))

    k /= np.sum(k)
    return k

#############################################################
#      Using filter2D without using getGaussianKernel       #
#############################################################
cv.filter2D(gray_img, -1, getGaussianKernel(SIZE))

#############################################################
#      Using sepFilter2D without using getGaussianKernel       #
#############################################################
cv.sepFilter2D(gray_img, -1, getGaussianKernel(SIZE))
