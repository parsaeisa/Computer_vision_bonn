import numpy as np
import cv2 as cv
from helper_methods import difference
from ex_1_1 import display_image

SIZE = 7

#############################################################
##         Reading the image and make it grayscale         ##
#############################################################
img_path = 'bonn.png'
img = cv.imread(img_path)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
display_image("gray scaled image", gray_img)

#############################################################
##             Filter images with two kernels              ##
#############################################################
kernel1 = np.array([
 [0.0113, 0.0838, 0.0113],
 [0.0838, 0.6193, 0.0838],
 [0.0113, 0.0838, 0.0113],
], dtype=np.float64)

kernel2 = np.array([
    [-1.7497, 0.3426, 1.1530, -0.2524,  0.9813],
 [0.5142, 0.2211, -1.0700, -0.1894,  0.2550],
 [-0.4580, 0.4351, -0.5835, 0.8168,  0.6727],
 [0.1044, -0.5312, 1.0297, -0.4381,  -1.1183],
 [1.6189, 1.5416, -0.2518, -0.8424,  0.1845],
], dtype=np.float64)

filteredWithKernel1 = cv.filter2D(gray_img, -1, kernel1)
filteredWithKernel2 = cv.filter2D(gray_img, -1, kernel2)

display_image("filteredWithKernel1", filteredWithKernel1)
# display_image("filteredWithKernel2", filteredWithKernel2)

#############################################################
##          Use OpenCV SVD class with Kernel 1             ##
#############################################################
w1, u1, vt1 = cv.SVDecomp(kernel1)

max_singular_value = w1.max()
max_index = w1.argmax()

KernelX = u1[:, max_index] * np.sqrt(max_singular_value)
KernelY = vt1[max_index, :] * np.sqrt(max_singular_value)

KernelX /= np.sum(KernelX)
KernelY /= np.sum(KernelY)

sepFilteredWithKernel1 = cv.sepFilter2D(gray_img, -1, KernelX, KernelY)

display_image("used kernel1 with seperated filters", sepFilteredWithKernel1)

#############################################################
##          Use OpenCV SVD class with Kernel 2             ##
#############################################################
w2, u2, vt2 = cv.SVDecomp(kernel2)

max_singular_value = w2.max()
max_index = w2.argmax()

print("w2: \n", w2, "\n")

second_max_singular_value = np.partition(w2[:, 0], -2)[-2]
second_max_index = np.where(w2[:, 0] == second_max_singular_value)[0]

KernelX1 = u2[:, max_index] * np.sqrt(max_singular_value)
KernelY1 = vt2[max_index, :] * np.sqrt(max_singular_value)

KernelX2 = u2[:, second_max_index] * np.sqrt(second_max_singular_value)
KernelY2 = vt2[second_max_index, :] * np.sqrt(second_max_singular_value)

KernelX1 /= np.sum(KernelX1)
KernelY1 /= np.sum(KernelY1)

KernelX2 /= np.sum(KernelX2)
KernelY2 /= np.sum(KernelY2)

sepFilteredWithKernel2 = cv.sepFilter2D(gray_img, -1, KernelX1, KernelY1)
sepFilteredWithKernel2 = cv.sepFilter2D(sepFilteredWithKernel2, -1, KernelX2, KernelY2)

# display_image("used kernel2 with seperated filters", sepFilteredWithKernel)

#############################################################
##          Computing difference between images            ##
#############################################################
print(difference(filteredWithKernel1, sepFilteredWithKernel1))