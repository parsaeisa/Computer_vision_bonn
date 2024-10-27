import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################################
#                                                         #
#                        TASK 4                           #
#                                                         #  
###########################################################


def build_gaussian_pyramid_opencv(image, num_levels):
    pyramid = [image]
    for _ in range(1, num_levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_gaussian_pyramid(image, num_levels):
    pyramid = [image]
    for _ in range(1, num_levels):
        image = cv2.GaussianBlur(image, (5, 5), 1)
        image = image[::2, ::2]  # Downsample by factor of 2
        pyramid.append(image)
    return pyramid


def template_matching_normalized_cross_correlation(image, template):
    result = np.zeros_like(image, dtype=np.float32)
    template_mean = np.mean(template)
    template_std = np.std(template)
    template = (template - template_mean) / template_std

    for y in range(image.shape[0] - template.shape[0] + 1):
        for x in range(image.shape[1] - template.shape[1] + 1):
            patch = image[y:y + template.shape[0], x:x + template.shape[1]]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            if patch_std == 0:
                continue
            patch = (patch - patch_mean) / patch_std
            result[y, x] = np.sum(patch * template)

    return result


def template_matching_multiple_scales(pyramid_image, pyramid_template):
    best_match = None
    best_value = -1
    for level in range(len(pyramid_image)):
        result = template_matching_normalized_cross_correlation(pyramid_image[level], pyramid_template[level])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_value:
            best_value = max_val
            best_match = (level, max_loc)
    return best_match


def task4():
    print("Loading images...")
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/traffic-template.png", cv2.IMREAD_GRAYSCALE)

    print("Build pyramids...")
    start_time = time.time()
    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
    end_time = time.time()
    print(f"Time taken to build OpenCV pyramid: {end_time - start_time} seconds")

    start_time = time.time()
    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)
    end_time = time.time()
    print(f"Time taken to build custom pyramid: {end_time - start_time} seconds")

    print("Comparing and printing mean absolute difference at each level...")
    for i in range(4):
        diff = np.mean(np.abs(cv_pyramid[i] - my_pyramid[i]))
        print(f"Mean absolute difference at level {i}: {diff}")

    print("Calculating the time needed for template matching without the pyramid...")
    start_time = time.time()
    result = template_matching_normalized_cross_correlation(image, template)
    end_time = time.time()
    print(f"Time taken for template matching without pyramid: {end_time - start_time} seconds")

    print("Calculating the time needed for template matching with the pyramid...")
    start_time = time.time()
    best_match = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    end_time = time.time()
    print(f"Time taken for template matching with pyramid: {end_time - start_time} seconds")

    print("Showing the template matching results using the pyramid...")
    level, (x, y) = best_match
    scale = 2 ** level
    top_left = (x * scale, y * scale)
    bottom_right = (top_left[0] + template.shape[1] * scale, top_left[1] + template.shape[0] * scale)
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    plt.imshow(image, cmap='gray')
    plt.title("Template Matching Result using Pyramid")
    plt.show()

if __name__ == "__main__":
    task4()
