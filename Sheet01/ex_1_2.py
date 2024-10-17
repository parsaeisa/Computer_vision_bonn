import cv2
import numpy as np
import time
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def hist_equalization(img):
    # Initialize cumulative histogram
    hist = np.zeros(shape = 256, dtype = int)

    # Fill histogram
    for row in img:
        for val in row:
            hist[val] += 1

    # Accumulate histogram
    for i in np.arange(1, hist.shape[0]):
        hist[i] += hist[i-1]

    # Normalize histogram
    for i in range(hist.shape[0]):
        hist[i] = 255 * hist[i] / hist[-1]

    # Remap values
    img_new = np.zeros(shape = img.shape, dtype = np.uint8)
    for x, row in enumerate(img):
        for y, val in enumerate(row):
            img_new[x, y] = hist[val]

    return img_new

def main():
    # Load image and convert to gray
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Task 3.2
    img_eq_cv = cv2.equalizeHist(img)
    display_image("3.2 - Histogram Equalization using OpenCV", img_eq_cv)

    img_eq_own = hist_equalization(img)
    display_image("3.2 - Histogram Equalization using own implementation", img_eq_own)
    
    # Calculate abs pixel difference
    diff = cv2.absdiff(img_eq_cv, img_eq_own)
    print(f"Maximum pixel error: {np.max(diff)}")


if __name__ == "__main__":
    main()
