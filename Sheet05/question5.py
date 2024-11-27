import numpy as np
import cv2
import maxflow
import matplotlib.pyplot as plt

# Plotting the noisy image and the denoised image
def plot_result(img, denoised_img, rho, pairwise_cost_type, figsize=(15, 7)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    fig.suptitle(f"Result for rho={rho} and pairwise_cost_type={pairwise_cost_type}", fontsize=16)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Noisy Image")
    axes[0].tick_params(labelbottom=False, labelleft=False)
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title("Denoised Image")
    axes[1].tick_params(labelbottom=False, labelleft=False)

    plt.savefig("result_question5.png")
    plt.show()

# Compute pairwise cost based on the chosen type
def compute_pairwise_cost(wm, wn, cost_type, params):
    if cost_type == "quadratic":
        κ = params.get("kappa", 1)
        return κ * (wm - wn) ** 2
    elif cost_type == "truncated_quadratic":
        κ1, κ2 = params.get("kappa1", 50), params.get("kappa2", 5)
        return min(κ1, κ2 * (wm - wn) ** 2)
    elif cost_type == "potts":
        κ = params.get("kappa", 1)
        return κ * (1 - int(wm == wn))
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

# Get neighbors of a pixel in the image (8-connectivity)
def get_neighbors(img, i, j):
    neighbors = []
    rows, cols = img.shape
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            neighbors.append((ni, nj))
    return neighbors

# Alpha Expansion for a single label
def alpha_expansion(I, label, alpha, rho, pairwise_cost_type, params):
    rows, cols = I.shape
    
    # Initialize graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((rows, cols))
    
    # Add unary costs
    for i in range(rows):
        for j in range(cols):
            current_label = label[i, j]
            g.add_tedge(
                nodeids[i, j],
                rho if current_label == alpha else 0,  # Source cost
                rho if current_label != alpha else 0   # Sink cost
            )
    
    # Add pairwise costs
    for i in range(rows):
        for j in range(cols):
            for ni, nj in get_neighbors(I, i, j):
                pairwise_cost = compute_pairwise_cost(
                    label[i, j], label[ni, nj], pairwise_cost_type, params
                )
                g.add_edge(nodeids[i, j], nodeids[ni, nj], pairwise_cost, pairwise_cost)
    
    # Perform optimization
    g.maxflow()
    
    # Extract new labels
    new_label = label.copy()
    for i in range(rows):
        for j in range(cols):
            new_label[i, j] = alpha if g.get_segment(nodeids[i, j]) == 0 else label[i, j]
    
    return new_label

# Perform denoising on grayscale image
def denoise_grayscale_image(img, rho, pairwise_cost_type, params):
    labels = np.zeros_like(img)
    for i, val in enumerate([0, 128, 255]):
        labels[img == val] = i  # Initialize labels based on grayscale intensities

    for alpha in range(3):
        labels = alpha_expansion(img, labels, alpha, rho, pairwise_cost_type, params)
    
    # Map labels back to grayscale values
    denoised_img = np.zeros_like(img)
    for i, val in enumerate([0, 128, 255]):
        denoised_img[labels == i] = val
    
    return denoised_img

# Apply morphological post-processing to clean spots
def apply_morphological_processing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel)  
    return cleaned_img

if __name__ == "__main__":
    # Read noisy image
    image = cv2.imread('./images/noisy_grayscale.png', cv2.IMREAD_GRAYSCALE)
    rho = 180  # Higher rho for stronger smoothing
    pairwise_cost_type = "truncated_quadratic"  # Options: "quadratic", "truncated_quadratic", "potts"
    params = {"kappa": 2, "kappa1": 150, "kappa2": 20}  
    
    # Apply denoising
    denoised_result = denoise_grayscale_image(image, rho=rho, pairwise_cost_type=pairwise_cost_type, params=params)
    
    # Apply morphological post-processing
    cleaned_result = apply_morphological_processing(denoised_result)
    
    # Plot results
    plot_result(image, cleaned_result, rho=rho, pairwise_cost_type=pairwise_cost_type, figsize=(10, 5))