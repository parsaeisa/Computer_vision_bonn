import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################################
#                                                         #
#                        TASK 2                           #
#                                                         #  
###########################################################

def get_convolution_using_fourier_transform(image, kernel):
    # TODO: implement
    raise NotImplementedError


def get_convolution(image, kernel):
    # TODO: implement
    raise NotImplementedError


def task2():
    # Load image
    image = cv2.imread("./data/oldtown.jpg", cv2.IMREAD_GRAYSCALE)
    kernel = None  # TODO: calculate kernel

    cv_result = None # TODO: cv2.filter2D
    conv_result = get_convolution(image, kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    # TODO: compare results

###########################################################
#                                                         #
#                        TASK 3                           #
#                                                         #  
###########################################################

def normalized_cross_correlation(image, template):
    # TODO: implement
    raise NotImplementedError

def ssd(image, template):
    # TODO: implement
    raise NotImplementedError

def draw_rectangle_at_matches(image, template_h, template_w, matches):
    # TODO: implement
    raise NotImplementedError

def task3():
    # Load images
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/eye.jpeg", cv2.IMREAD_GRAYSCALE)

    # convert to float and apply intensity transformation to image

    result_ncc = normalized_cross_correlation(image, template)
    result_ssd = ssd(image, template)

    # TODO: draw rectangle around found locations
    # TODO: show the results


###########################################################
#                                                         #
#                        TASK 4                           #
#                                                         #  
###########################################################


def build_gaussian_pyramid_opencv(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def build_gaussian_pyramid(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def template_matching_multiple_scales(pyramid_image, pyramid_template):
    # TODO: implement
    raise NotImplementedError


def task4():
    # Load images
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/traffic-template.png", cv2.IMREAD_GRAYSCALE)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    # TODO: compare and print mean absolute difference at each level

    # TODO: calculate the time needed for template matching without the pyramid

    result = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid


###########################################################
#                                                         #
#                        TASK 5                           #
#                                                         #  
###########################################################

def build_laplacian_pyramid(gaussian_pyramid):
    # TODO: implement
    raise NotImplementedError

def task5():
    # Load images
    messi = cv2.imread('data/messi.jpg')
    ronaldo = cv2.imread('data/ronaldo.jpeg')
    
    # TODO: build pyramids
    # TODO: blend Laplacian pyramids
    # TODO: collapse the combined pyramid
    # TODO: show the result


if __name__ == "__main__":
    task2()
    task3()
    task4()
    task5()
