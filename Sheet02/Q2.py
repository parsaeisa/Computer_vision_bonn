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
    # Get the size of the image and kernel
    image_shape = image.shape
    kernel_shape = kernel.shape

    # Pad the kernel to the size of the image
    pad_kernel = np.zeros_like(image)
    pad_kernel[:kernel_shape[0], :kernel_shape[1]] = kernel

    # Perform FFT on the image and the padded kernel
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(pad_kernel)

    # Multiply the FFTs and perform inverse FFT
    conv_fft = np.fft.ifft2(image_fft * kernel_fft)

    # Take the real part of the result
    conv_result = np.real(conv_fft)

    return conv_result

def get_convolution(image, kernel):
    # Get the size of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output image
    output_image = np.zeros_like(image)

    # Perform convolution
    # Flip the kernel (for convolution operation, not cross-correlation)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Get the shape of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Pad the image with zeros around the border
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Create a sliding window view of the padded image
    window_shape = (image_height, image_width, kernel_height, kernel_width)
    sliding_windows = np.lib.stride_tricks.sliding_window_view(padded_image, (kernel_height, kernel_width))
    
    # Perform element-wise multiplication and sum along the last two axes
    output_image = np.sum(sliding_windows * kernel, axis=(-2, -1))

    return output_image

def task2():
    print("Loading image...")
    image = cv2.imread("./data/oldtown.jpg", cv2.IMREAD_GRAYSCALE)

    print("Defining the 7x7 Sobel filter kernel...")
    kernel = np.array([
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1,  0,  0,  0, -1, -1],
        [-1, -1,  0,  0,  0, -1, -1],
        [-1, -1,  0,  0,  0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
    ])

    print("Filtering the image using custom convolution...")
    start_time = time.time()
    conv_result = get_convolution(image, kernel)
    conv_time = time.time() - start_time

    print("Filtering the image using cv2.filter2D...")
    start_time = time.time()
    cv_result = cv2.filter2D(image, -1, kernel)
    cv_time = time.time() - start_time

    print("Filtering the image using Fourier transform...")
    start_time = time.time()
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    fft_time = time.time() - start_time

    print("Ploting the images...")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Custom Convolution")
    plt.imshow(conv_result, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("cv2.filter2D")
    plt.imshow(cv_result, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Fourier Transform")
    plt.imshow(fft_result, cmap='gray')
    plt.show()

    print("Computing the mean absolute differences...")
    diff_conv_cv = np.mean(np.abs(conv_result - cv_result))
    diff_conv_fft = np.mean(np.abs(conv_result - fft_result))
    diff_cv_fft = np.mean(np.abs(cv_result - fft_result))

    print("Printing the results...")
    print(f"Time taken for custom convolution: {conv_time:.4f} seconds")
    print(f"Time taken for cv2.filter2D: {cv_time:.4f} seconds")
    print(f"Time taken for Fourier transform: {fft_time:.4f} seconds")
    print(f"Mean absolute difference (custom vs cv2): {diff_conv_cv:.4f}")
    print(f"Mean absolute difference (custom vs fft): {diff_conv_fft:.4f}")
    print(f"Mean absolute difference (cv2 vs fft): {diff_cv_fft:.4f}")

if __name__ == "__main__":
    task2()
