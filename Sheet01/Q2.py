import numpy as np
import cv2 as cv
from ex_1_1 import display_image
import matplotlib.pyplot as plt

#####           Reading the image          #####
img_path = 'bonn.png'
img = cv.imread(img_path)

#############################################################
##               Using equalizeHist methid                 ##
#############################################################
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
equ_opencv = cv.equalizeHist(gray_img)

#############################################################
##           My implementation of equalizeHist             ##
#############################################################
BRIGHTNESS_LAYERS = 256

def hist_equ(image):
    '''
    input:
    image (ndarray): input image
    output:
    output_image (ndarray): enhanced image
    '''
    
    pdf = np.zeros(BRIGHTNESS_LAYERS)

    for pixel in np.nditer(image) :
      pdf[np.round_(pixel)] += 1

    cdf = np.zeros(BRIGHTNESS_LAYERS)
    cdf[0] = pdf[0]
    for pixel_iterator in range(1,BRIGHTNESS_LAYERS) :
      cdf[pixel_iterator] = cdf[pixel_iterator-1] + pdf [pixel_iterator]

    height , width = image.shape
    n = height * width
    tr = (cdf / n) * (BRIGHTNESS_LAYERS - 1)

    output_image = np.zeros((1,height * width))

    output_index = 0 
    for pixel in np.nditer(image) :
      output_image[0,output_index]= tr[np.round_(pixel)]
      output_index += 1 

    output_image = output_image.reshape(image.shape)
    assert image.shape == output_image.shape
    # End
    
    return output_image

my_equ = hist_equ(gray_img)
#############################################################
##                    Getting output                       ##
#############################################################

plt.hist(gray_img.flatten(),256,[0,256], color = 'r')
plt.show()

plt.hist(my_equ.flatten(),256,[0,256], color = 'r')
plt.show()

res = np.hstack((gray_img, my_equ)) #stacking images side-by-side

# plt.figure(figsize=(16, 16))
plt.imshow(res, cmap='gray')

plt.show() 
