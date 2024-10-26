import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

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
        return 0

    patch = f[m:m+k, n:n+l]    

    return np.sum((g-patch)**2)

output = np.ones(f_gray.shape, dtype=np.float64)

image_length, image_width = f_gray.shape

for i in range(image_length):    
    for j in range(image_width):        
        output[i,j] = 1-SSD(f_gray, g_gray, i, j)

SSD_output_normalized = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
SSD_output_normalized = SSD_output_normalized.astype(np.uint8)

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

output = np.ones(f_gray.shape, dtype=np.float64)

mean_g = np.mean(g)

for i in range(image_length):    
    for j in range(image_width):        
        output[i,j] = NCC(f_gray, g_gray, mean_g, i, j)

NCC_output_normalized = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
NCC_output_normalized = NCC_output_normalized.astype(np.uint8)

#############################################################
###                   Displaying images                    ##
#############################################################
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(SSD_output_normalized)
# axes[0].set_title('SSD')
# axes[0].axis('off')  # Hide axis

# axes[1].imshow(NCC_output_normalized)
# axes[1].set_title('NCC')
# axes[1].axis('off')  # Hide axis

# plt.show()

#############################################################
###                  Drawing rectangles                    ##
#############################################################

# print(np.where(SSD_output_normalized >= 0.7))
# print(">>>>>>>>>>>>>>>>>>>>>")
# Example image (grayscale)
def draw_rectangle(threshhold, img):
    # Find the indices of pixels that meet the condition
    y_indices, x_indices = np.where(img >= threshhold)

    # Check if there are any matching pixels
    if y_indices.size > 0 and x_indices.size > 0:
        # Get the bounding box around the matching pixels
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Draw the rectangle on the img (in-place)
        image_colored = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
        cv.rectangle(image_colored, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Show the result
        cv.imshow("Rectangle around matching pixels", image_colored)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("No pixels meet the condition.")

draw_rectangle(0.95, SSD_output_normalized)
draw_rectangle(0.7, NCC_output_normalized)