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


kernel1 = [
 0.0113, 0.0838, 0.0113,
 0.0838, 0.6193, 0.0838,
 0.0113, 0.0838, 0.0113,
]

kernel2 = [
    -1.7497, 0.3426, 1.1530, -0.2524,  0.9813,
 0.5142, 0.2211, -1.0700, -0.1894,  0.2550,
 -0.4580, 0.4351, -0.5835, 0.8168,  0.6727,
 0.1044, -0.5312, 1.0297, -0.4381,  -1.1183,
 1.6189, 1.5416, -0.2518, -0.8424,  0.1845,
]