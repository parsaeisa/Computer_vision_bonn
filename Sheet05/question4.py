import numpy as np
import cv2
import maxflow
import matplotlib.pyplot as plt


def plot_result(img, denoised_img, rho, pairwise_same, pairwise_diff, figsize=(15, 7)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    fig.suptitle(f"Result for rho={rho} pairwise_cost_same={pairwise_same} and pairwise_cost_diff={pairwise_diff} ", fontsize=16)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Noisy Image")
    axes[0].tick_params(labelbottom=False, labelleft=False)
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title("Denoised Image")
    axes[1].tick_params(labelbottom=False, labelleft=False)

    plt.savefig(f"result_question4.png")
    plt.show()



def binary_img_denoiser(img, rho, pairwise_same, pairwise_diff):
    # TODO: Change to binary image
    img = ... # Binary image

    # Ensure the input image is binary
    assert np.array_equal(np.unique(img), [0, 1]), "Input image must be binary (0 or 1)."

    # TODO: Define Graph and add pixels as nodes


    # TODO: Add unary costs to the graph


    # TODO: Add pairwise costs to the graph


    # TODO: Perform Maxflow optimization


    # TODO: Extract labels and construct the denoised image
    denoised_img = np.zeros_like(img, dtype=np.uint8)
    ...
    
    return denoised_img


if __name__ == "__main__":
    # TODO: Read the noisy binary image
    image = ... # File: './images/noisy_binary.png'

    rho = ... # Set parameter
    pairwise_same = ... # Set parameter
    pairwise_diff = ... # Set parameter

    result = binary_img_denoiser(image, rho=rho, pairwise_same=pairwise_same, pairwise_diff=pairwise_diff)

    plot_result(image, 
                result,
                rho=rho, 
                pairwise_same=pairwise_same, 
                pairwise_diff=pairwise_diff)