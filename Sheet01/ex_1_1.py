import cv2 as cv
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
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


def recursive_integral(img, current_index):
    global img_integral
    global flag

    if current_index == (0, 0): # Left upper corner case
        img_integral[0, 0] = img[0, 0]
        flag = True
    elif current_index[0] == 0: # Upper edge case
        next_index = (0, current_index[1] - 1)
        img_integral[0, current_index[1]] = img[-1, -1] + recursive_integral(img[:, :-1], next_index)
    elif current_index[1] == 0: # Left edge case
        next_index = (current_index[0] - 1, 0)
        img_integral[current_index[0], 0] = img[-1, -1] + recursive_integral(img[:-1, :], next_index)
    else: # Usual case
        if flag:
            img_integral[current_index[0], current_index[1]] = \
                img[-1, -1] + \
                img_integral[current_index[0], current_index[1] - 1] + \
                recursive_integral(img[:-1, :], (current_index[0] - 1, current_index[1])) - \
                img_integral[current_index[0] - 1, current_index[1] - 1]
        else:
            img_integral[current_index[0], current_index[1]] = \
                img[-1, -1] + \
                recursive_integral(img[:, :-1], (current_index[0], current_index[1] - 1)) + \
                recursive_integral(img[:-1, :], (current_index[0] - 1, current_index[1])) - \
                img_integral[current_index[0] - 1, current_index[1] - 1]

    return img_integral[current_index[0], current_index[1]]


def get_integral_img(img):
    img_integral = np.zeros(shape=img.shape, dtype=int)
    
    # Fill the first row
    img_integral[0, 0] = img[0, 0]
    for j in range(1, img.shape[1]):
        img_integral[0, j] = img_integral[0, j - 1] + img[0, j]
    
    # Fill the first column
    for i in range(1, img.shape[0]):
        img_integral[i, 0] = img_integral[i - 1, 0] + img[i, 0]
    
    # Fill the rest of the integral image
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            img_integral[i, j] = img[i, j] + img_integral[i - 1, j] + img_integral[i, j - 1] - img_integral[i - 1, j - 1]
    
    return img_integral



def get_mean_val_naive(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Run through randomized coords
    for (x, y) in rand_coords:
        # Sum up values
        sum = 0
        for row in img[x : x + size[0], y : y + size[1]]:
            for val in row:
                sum += val
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using simple summation ({x}, {y}): {mean}")


def get_mean_val_own(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = get_integral_img(img)
    img_integral = np.pad(img_integral, ((1, 0), (1, 0)), 'constant')

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum = img_integral[x + size[0], y + size[1]] - img_integral[x, y]
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using own integral implementation ({x}, {y}): {mean}")


def get_mean_val_cv(img, rand_coords = [[0, 0]], size = None, verbose = True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = cv.integral(img)

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum = img_integral[x + size[0], y + size[1]] - img_integral[x, y]
        mean = sum / (size[0] * size[1])
        if verbose:
            print(f"Mean using cv integral implementation ({x}, {y}): {mean}")


def time_function(func, img):
    # Get randomized patches
    np.random.seed(19)
    patch_size = (100, 100)
    n = 10
    rand_coords = np.dstack(
        [randint(low = 0, high = img.shape[0] - patch_size[0], size = n), 
         randint(low = 0, high = img.shape[1] - patch_size[1], size = n)]
         )[0]
    
    # Time function
    start = time.time()
    func(img, rand_coords, patch_size, verbose = False)
    end = time.time()
    print(f"Elapsed time for {func.__name__}: {end - start}")

def apply_gaussian_filters(img):
    # Apply Gaussian filter twice with σ = 2
    gaussian_twice = cv.GaussianBlur(img, (0, 0), 2)
    gaussian_twice = cv.GaussianBlur(gaussian_twice, (0, 0), 2)
    
    # Apply Gaussian filter once with σ = 2√2
    sigma = 2 * np.sqrt(2)
    gaussian_once = cv.GaussianBlur(img, (0, 0), sigma)
    
    # Display results
    display_image("Gaussian Filter Twice σ=2", gaussian_twice)
    display_image("Gaussian Filter Once σ=2√2", gaussian_once)
    
    # Compute absolute pixelwise difference
    abs_diff = cv.absdiff(gaussian_twice, gaussian_once)
    max_pixel_error = np.max(abs_diff)
    print(f"Maximum pixel error: {max_pixel_error}")

def add_salt_and_pepper_noise(img, noise_level=0.3):
    noisy_img = img.copy()
    num_salt = np.ceil(noise_level * img.size * 0.5)
    num_pepper = np.ceil(noise_level * img.size * 0.5)
    
    # Add salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 1
    
    # Add pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0
    
    return noisy_img

def denoise_image(img):
    # Add salt and pepper noise
    noisy_img = add_salt_and_pepper_noise(img)
    display_image("Noisy Image", noisy_img)
    
    # Apply Gaussian filter
    gaussian_filtered = cv.GaussianBlur(noisy_img, (5, 5), 0)
    display_image("Gaussian Filtered", gaussian_filtered)
    
    # Apply stack blur (assuming stackBlur is defined elsewhere)
    stack_blurred = stackBlur(noisy_img, 5)
    display_image("Stack Blurred", stack_blurred)
    
    # Apply bilateral filter
    bilateral_filtered = cv.bilateralFilter(noisy_img, 9, 75, 75)
    display_image("Bilateral Filtered", bilateral_filtered)
    
    # Compute mean gray value distance to the original image
    filters = [gaussian_filtered, stack_blurred, bilateral_filtered]
    filter_names = ["Gaussian", "Stack Blur", "Bilateral"]
    original_mean = np.mean(img)
    
    for i, filtered_img in enumerate(filters):
        mean_val = np.mean(filtered_img)
        mean_distance = abs(mean_val - original_mean)
        print(f"Mean gray value distance for {filter_names[i]}: {mean_distance}")

def stackBlur(img, radius):
    # Placeholder implementation of stackBlur
    # Replace this with the actual implementation
    return cv.GaussianBlur(img, (radius, radius), 0)




def main():
    # Load image and convert to gray
    img_path = 'bonn.png'
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Task 3.1a
    img_integral_own = get_integral_img(img)
    img_integral_display = cv.normalize(img_integral_own, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    display_image("3.1 - Rectangles and Integral Images", img_integral_display)

    # Task 3.1b
    get_mean_val_naive(img, verbose=True)
    get_mean_val_cv(img, verbose=True)
    get_mean_val_own(img, verbose=True)

    # Task 3.1c
    time_function(get_mean_val_naive, img)
    time_function(get_mean_val_cv, img)
    time_function(get_mean_val_own, img)

    # Task 5
    apply_gaussian_filters(img)

    # Task 7
    denoise_image(img)

if __name__ == "__main__":
    main()
