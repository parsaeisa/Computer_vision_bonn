import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from helpers import display_image

LEVELS_COUNT=5

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

_, axarr = plt.subplots(LEVELS_COUNT-1,2, figsize=(10, 10))

def build_laplacian_pyramid(gaussian_pyramid, plot_col):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)    

    for i in range(levels - 1):        
        gaussian_expanded = cv.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))

        laplacian = cv.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

        # Plotting each level
        axarr[i, plot_col].imshow(laplacian)
        axarr[i, plot_col].set_title(f"level: {i}")
        axarr[i, plot_col].axis('off')

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid

#############################################################
##                   Combining pyramids                    ##
#############################################################
def combine_laplacian_pyramids(laplacian_pyramid1, laplacian_pyramid2):
    combined_pyramid = []
    
    for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2):
        cols = lap1.shape[1]
        laplacian_combined = np.hstack((lap1[:, :cols // 2], lap2[:, cols // 2:]))
        combined_pyramid.append(laplacian_combined)
        
    return combined_pyramid

#############################################################
##                   Recostructing image                   ##
#############################################################
def reconstruct(laplacian_pyramid):    
    reconstructed_image = laplacian_pyramid[-1]
    
    for level in range(len(laplacian_pyramid) - 2, -1, -1):
        reconstructed_image = cv.pyrUp(reconstructed_image, dstsize=(laplacian_pyramid[level].shape[1], laplacian_pyramid[level].shape[0]))
        reconstructed_image = cv.add(reconstructed_image, laplacian_pyramid[level])
        
    return reconstructed_image

# Resizing ronaldo image to size (500, 500)
ronaldo_img = cv.resize(ronaldo_img, (500, 500))

laplacian_pyramid1 = build_laplacian_pyramid(
        build_gaussian_pyramid(cv.cvtColor(messi_img, cv.COLOR_BGR2GRAY), LEVELS_COUNT), 0
    )
laplacian_pyramid2 = build_laplacian_pyramid(
        build_gaussian_pyramid(cv.cvtColor(ronaldo_img, cv.COLOR_BGR2GRAY), LEVELS_COUNT), 1
    )

plt.show()

# Combine the two Laplacian pyramids
combined_laplacian_pyramid = combine_laplacian_pyramids(laplacian_pyramid1, laplacian_pyramid2)

# Reconstruct the final blended image from the combined Laplacian pyramid
blended_image = reconstruct(combined_laplacian_pyramid)

display_image("blended image", blended_image)