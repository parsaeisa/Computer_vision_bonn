import numpy as np
import cv2 as cv
from ex_1_1 import display_image
import matplotlib.pyplot as plt
from Sheet01.helper_methods import difference

SIZE = 7

#############################################################
##         Reading the image and make it grayscale         ##
#############################################################
img_path = 'bonn.png'
img = cv.imread(img_path)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#############################################################
##                   Using gaussianBlur                    ##
#############################################################
sigma = 2 * np.sqrt(2)
blurred = cv.GaussianBlur(gray_img, (SIZE, SIZE), sigmaX=sigma)

#############################################################
#      Using filter2D without using getGaussianKernel       #
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

filtered = cv.filter2D(gray_img, -1, getGaussianKernel(SIZE))

#############################################################
#     Using sepFilter2D without using getGaussianKernel     #
#############################################################
def getGaussian1DKernel(size):
    mid = size/2
    k = np.zeros((size))

    for i in range(size):        
            x = i - mid            

            k[i] = np.exp(-1 * (x**2) / (2 * sigma**2))

    k /= np.sum(k)
    return k

kernelX = getGaussian1DKernel(SIZE)
kernelY = getGaussian1DKernel(SIZE)

sepFiltered = cv.sepFilter2D(gray_img, -1, kernelX, kernelY)

#############################################################
##                     Showing results                     ##
#############################################################
display_image("filtered without using getGaussianKernel", filtered)
display_image("sepFiltered without using getGaussianKernel", sepFiltered)

#############################################################
##          Computing difference between images            ##
#############################################################
print(difference(blurred, filtered))
print(difference(blurred, sepFiltered))
print(difference(sepFiltered, filtered))