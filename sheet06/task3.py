import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

class MOG():
    def __init__(self, height=360, width=640, number_of_gaussians=3, background_thresh=0.6, lr=0.01):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.lr = lr
        self.height = height
        self.width = width
        
        # Initialize parameters for each Gaussian
        self.mus = np.zeros((self.height, self.width, self.number_of_gaussians, 3))
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians))
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))
        
        # Initial values for means, variances, and weights
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i, j] = np.array([[122, 122, 122]] * self.number_of_gaussians)
                self.sigmaSQs[i, j] = [36.0] * self.number_of_gaussians
                self.omegas[i, j] = [1.0 / self.number_of_gaussians] * self.number_of_gaussians

    def updateParam(self, img, BG_pivot):
        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask
        
        for i in range(height):
            for j in range(width):
                pixel = img[i, j]
                match_found = False
                match_prob = 0
                match_index = -1

                # Match the pixel with existing Gaussians
                for k in range(self.number_of_gaussians):
                    diff = pixel - self.mus[i, j, k]
                    dist = np.sum(diff**2)

                    # Match condition
                    if dist < 3.0 * self.sigmaSQs[i, j, k]:
                        if self.omegas[i, j, k] > match_prob:
                            match_prob = self.omegas[i, j, k]
                            match_index = k
                        match_found = True

                # Update the best-matching Gaussian
                if match_found:
                    match_gaussian = match_index
                    adaptive_lr = self.lr * (1 - match_prob)  # Adjust learning rate by match confidence
                    self.mus[i, j, match_gaussian] = (
                        (1 - adaptive_lr) * self.mus[i, j, match_gaussian] + adaptive_lr * pixel
                    )
                    diff = pixel - self.mus[i, j, match_gaussian]
                    self.sigmaSQs[i, j, match_gaussian] = (
                        (1 - adaptive_lr) * self.sigmaSQs[i, j, match_gaussian]
                        + adaptive_lr * np.sum(diff**2)
                    )
                    self.omegas[i, j, match_gaussian] += self.lr * (1 - self.omegas[i, j, match_gaussian])
                else:
                    # Replace the least probable Gaussian if no match
                    least_probable = np.argmin(self.omegas[i, j])
                    self.mus[i, j, least_probable] = pixel
                    self.sigmaSQs[i, j, least_probable] = 36.0
                    self.omegas[i, j, least_probable] = 0.01

                # Normalize weights
                self.omegas[i, j] /= np.sum(self.omegas[i, j])

                # Compute match probability for the pixel
                total_prob = 0
                for k in range(self.number_of_gaussians):
                    diff = pixel - self.mus[i, j, k]
                    dist = np.sum(diff**2)
                    if dist < 3.0 * self.sigmaSQs[i, j, k]:
                        total_prob += self.omegas[i, j, k]

                # Foreground/Background classification with soft threshold
                if total_prob < self.background_thresh:
                    mask[i, j] = 255  # Foreground
                else:
                    mask[i, j] = 0  # Background
        
        return mask


# Main script
frame = cv2.imread('person.jpg')
frame = cv2.resize(frame, (640, 360))  # Resize for demonstration purposes
subtractor = MOG(height=360, width=640, number_of_gaussians=3, background_thresh=0.6, lr=0.01)

# Apply background subtraction
BG_pivot = np.ones(frame.shape[:2])  # Initial pivot mask
mask = subtractor.updateParam(frame, BG_pivot)

# Display the resulting mask and foreground
result = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("Foreground Mask", mask)
cv2.imshow("Extracted Foreground", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
