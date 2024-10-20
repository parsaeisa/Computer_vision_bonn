import numpy as np
import cv2 as cv
from helper_methods import difference
import matplotlib.pyplot as plt
SIZE = 7

#############################################################
##         Reading the image and make it grayscale         ##
#############################################################
img_path = 'bonn.png'
img = cv.imread(img_path)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#############################################################
##                    Defining kernels                     ##
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

#############################################################
##             Filter images with two kernels              ##
#############################################################
filteredWithKernel1 = cv.filter2D(gray_img, -1, kernel1)
filteredWithKernel2 = cv.filter2D(gray_img, -1, kernel2)

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

#############################################################
##          Use OpenCV SVD class with Kernel 2             ##
#############################################################
w2, u2, vt2 = cv.SVDecomp(kernel2)

max_singular_value = w2.max()
max_index = w2.argmax()

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

sepFilteredWithApprox1 = cv.sepFilter2D(gray_img, cv.CV_64F, KernelX1, KernelY1)
sepFilteredWithApprox2 = cv.sepFilter2D(gray_img, cv.CV_64F, KernelX2, KernelY2)

sepFilteredWithKernel2 = sepFilteredWithApprox1 + sepFilteredWithApprox2

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

#############################################################
###                   Displaying images                    ##
#############################################################
axes[0, 0].imshow(filteredWithKernel1)
axes[0, 0].set_title('Filtered with Kernel 1')
axes[0, 0].axis('off')  # Hide axis

axes[0, 1].imshow(sepFilteredWithKernel1)
axes[0, 1].set_title('SepFiltered with Kernel 1')
axes[0, 1].axis('off')

axes[1, 0].imshow(filteredWithKernel2)
axes[1, 0].set_title('Filtered with Kernel 2')
axes[1, 0].axis('off')

axes[1, 1].imshow(sepFilteredWithKernel2)
axes[1, 1].set_title('SepFiltered with Kernel 2')
axes[1, 1].axis('off')

# Display the images
plt.show()

#############################################################
##          Computing difference between images            ##
#############################################################
print("Difference between two ways with Kernel 1", difference(filteredWithKernel1, sepFilteredWithKernel1))
print("Difference between two ways with Kernel 2", difference(filteredWithKernel2, sepFilteredWithKernel2))