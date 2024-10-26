import cv2 as cv
from helpers import display_image
import matplotlib.pyplot as plt

img_path = 'data/messi.jpg'
messi_img = cv.imread(img_path)

img_path = 'data/ronaldo.jpeg'
ronaldo_img = cv.imread(img_path)

display_image("test", ronaldo_img)

def build_gaussian_pyramid(image):
    pass

def build_laplacian_pyramid(image):
    pass
