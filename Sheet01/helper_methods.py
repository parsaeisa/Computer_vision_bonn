import numpy as np

def difference(img1, img2):
   return np.max(np.abs(img1 - img2))