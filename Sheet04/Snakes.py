import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Read the images
ball_image = io.imread('ball.png')
coffee_image = io.imread('coffee.png')

# Convert to RGB if the image has an alpha channel
if ball_image.shape[-1] == 4:
    ball_image = color.rgba2rgb(ball_image)
if coffee_image.shape[-1] == 4:
    coffee_image = color.rgba2rgb(coffee_image)

# Convert to grayscale
ball_gray = color.rgb2gray(ball_image)
coffee_gray = color.rgb2gray(coffee_image)

# Initialize the snake
s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T

# Optimize the snake for ball image
snake_ball = active_contour(gaussian(ball_gray, 3), init, alpha=0.015, beta=10, gamma=0.001)

# Optimize the snake for coffee image
snake_coffee = active_contour(gaussian(coffee_gray, 3), init, alpha=0.015, beta=10, gamma=0.001)

# Visualize the result for ball image
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ball_gray, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake_ball[:, 0], snake_ball[:, 1], '-b', lw=3)
ax.set_title('Ball Image')
plt.show()

# Visualize the result for coffee image
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(coffee_gray, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake_coffee[:, 0], snake_coffee[:, 1], '-b', lw=3)
ax.set_title('Coffee Image')
plt.show()
