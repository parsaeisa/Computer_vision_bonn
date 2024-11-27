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
    # Convert to binary image (0 or 1)
    img = (img > 128).astype(np.uint8)

    # Ensure the input image is binary
    assert np.array_equal(np.unique(img), [0, 1]), "Input image must be binary (0 or 1)."

    # Define graph and add pixels as nodes
    graph = maxflow.Graph[int]()
    node_ids = graph.add_grid_nodes(img.shape)

    # Add unary costs to the graph
    unary_cost_source = -np.log(rho) * img - np.log(1 - rho) * (1 - img)
    unary_cost_sink = -np.log(1 - rho) * img - np.log(rho) * (1 - img)
    graph.add_grid_tedges(node_ids, unary_cost_source, unary_cost_sink)

    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.int32)
    graph.add_grid_edges(node_ids, structure=structure, weights=pairwise_same, symmetric=True)
    graph.add_grid_edges(node_ids, structure=np.ones_like(structure), weights=pairwise_diff, symmetric=True)

    # Perform Maxflow optimization
    graph.maxflow()

    denoised_img = graph.get_grid_segments(node_ids).astype(np.float16)

    return denoised_img

if __name__ == "__main__":
    image = cv2.imread('./noisy_binary.png', cv2.IMREAD_GRAYSCALE)

    rho = 0.99
    pairwise_same = 0.1
    pairwise_diff = 1

    result = binary_img_denoiser(image, rho=rho, pairwise_same=pairwise_same, pairwise_diff=pairwise_diff)

    plot_result(image, 
                result,
                rho=rho, 
                pairwise_same=pairwise_same, 
                pairwise_diff=pairwise_diff)
