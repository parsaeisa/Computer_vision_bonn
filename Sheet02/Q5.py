import cv2 as cv
from helpers import display_image

img_path = 'data/messi.jpg'
messi_img = cv.imread(img_path)

img_path = 'data/ronaldo.jpeg'
ronaldo_img = cv.imread(img_path)

display_image("test", ronaldo_img)