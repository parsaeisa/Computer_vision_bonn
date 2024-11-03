import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from helpers import display_image

#############################################################
##            Reading images and converting them           ##
#############################################################
img_path = 'data/einstein.jpeg'
f = cv.imread(img_path)
f_gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
f_gray = f_gray.astype('float32') / 255

img_path = 'data/einstein_eye.jpeg'
g = cv.imread(img_path)
g_gray = cv.cvtColor(g, cv.COLOR_BGR2GRAY)
g_gray = g_gray.astype('float32') / 255

#############################################################
##                  Sum Squared Difference                 ##
#############################################################
def SSD(f, g, m, n):
    k, l = g.shape

    if m+k >= f.shape[0] or n+l >= f.shape[1]:
        return 1

    patch = f[m:m+k, n:n+l]    

    return np.sum((g-patch)**2)

SSD_output = np.zeros(f_gray.shape, dtype=np.float64)

image_length, image_width = f_gray.shape

for i in range(image_length):    
    for j in range(image_width):        
        SSD_output[i,j] = SSD(f_gray, g_gray, i, j)

SSD_output_normalized = cv.normalize(SSD_output, None, 0, 255, cv.NORM_MINMAX)
SSD_output_normalized = SSD_output_normalized.astype(np.float32)

#############################################################
##             Normalized Cross Correlation                ##
#############################################################
def NCC(f, g, mean_g, m, n):
    k, l = g.shape

    if m+k >= f.shape[0] or n+l >= f.shape[1]:
        return 0
    
    patch = f[m:m+k, n:n+l]
    mean_patch = np.mean(patch)
    
    numerator = np.sum(np.multiply(g - mean_g, patch - mean_patch))    
    denominator = np.sqrt(np.sum((g - mean_g) ** 2) * np.sum((patch - mean_patch) ** 2))
    
    if denominator == 0:
        return 0
    
    h_mn = numerator / denominator

    return h_mn

NCC_output = np.zeros(f_gray.shape, dtype=np.float64)

mean_g = np.mean(g)

for i in range(image_length):    
    for j in range(image_width):        
        NCC_output[i,j] = NCC(f_gray, g_gray, mean_g, i, j)

NCC_output_normalized = cv.normalize(NCC_output, None, 0, 255, cv.NORM_MINMAX)
NCC_output_normalized = NCC_output_normalized.astype(np.uint8)

#############################################################
###                   Displaying images                    ##
#############################################################
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(SSD_output_normalized)
axes[0].set_title('SSD')
axes[0].axis('off')  # Hide axis

axes[1].imshow(NCC_output_normalized)
axes[1].set_title('NCC')
axes[1].axis('off')  # Hide axis

plt.show()

#############################################################
###                  Drawing rectangles                    ##
#############################################################
RECT_LENGTH= 40
RECT_WIDTH= 40

def draw_rectangle(y_indices, x_indices, showable_image):    

    print(len(y_indices))

    image_colored = cv.cvtColor(showable_image, cv.COLOR_GRAY2BGR)  # Convert to BGR for color drawing

    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        # print(img[x,y])
        # Draw the rectangle on the img (in-place)
        cv.rectangle(image_colored, (int(x - RECT_LENGTH/2), int(y - RECT_WIDTH/2)), (int(x + RECT_LENGTH/2), int(y + RECT_WIDTH/2)), (0, 255, 0), 2)

    display_image("rectangled image", image_colored)

y_indices, x_indices = np.where((NCC_output_normalized)//255 >= np.float32(0.7))
draw_rectangle(y_indices, x_indices, NCC_output_normalized)

y_indices, x_indices = np.where((SSD_output_normalized)//255 <= np.float32(0.1))
draw_rectangle(y_indices, x_indices, SSD_output_normalized)

#############################################################
##                    Subtracting 0.5                      ##
#############################################################
f_gray -= 0.5
g_gray -= 0.5

SSD_output = np.zeros(f_gray.shape, dtype=np.float64)

image_length, image_width = f_gray.shape

for i in range(image_length):    
    for j in range(image_width):        
        SSD_output[i,j] = SSD(f_gray, g_gray, i, j)

SSD_output_normalized = cv.normalize(SSD_output, None, 0, 255, cv.NORM_MINMAX)
SSD_output_normalized = SSD_output_normalized.astype(np.float32)

NCC_output = np.zeros(f_gray.shape, dtype=np.float64)

mean_g = np.mean(g)

for i in range(image_length):    
    for j in range(image_width):        
        NCC_output[i,j] = NCC(f_gray, g_gray, mean_g, i, j)

NCC_output_normalized = cv.normalize(NCC_output, None, 0, 255, cv.NORM_MINMAX)
NCC_output_normalized = NCC_output_normalized.astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(SSD_output_normalized)
axes[0].set_title('SSD')
axes[0].axis('off')  # Hide axis

axes[1].imshow(NCC_output_normalized)
axes[1].set_title('NCC')
axes[1].axis('off')  # Hide axis

plt.show()

y_indices, x_indices = np.where((NCC_output_normalized)//255 >= np.float32(0.7))
draw_rectangle(y_indices, x_indices, NCC_output_normalized)

y_indices, x_indices = np.where((SSD_output_normalized)//255 <= np.float32(0.1))
draw_rectangle(y_indices, x_indices, SSD_output_normalized)
