import numpy as np
import matplotlib.pyplot as plt

def distance_transform(binary_image):
    """
    Compute the distance transform of a binary image using two-pass algorithm.
    
    Args:
        binary_image (np.ndarray): Binary input image (0: background, 1: foreground)
    
    Returns:
        np.ndarray: Distance transform map
    """
    if not isinstance(binary_image, np.ndarray) or binary_image.dtype != bool:
        binary_image = binary_image.astype(bool)
    
    height, width = binary_image.shape
    dist_map = np.zeros_like(binary_image, dtype=float)
    
    # TODO: Initialize distance map (set to inf for foreground, 0 for background)        
    B = np.argwhere(binary_image == False)
    for i,j in np.argwhere(binary_image == True):
        dist_map[i, j] = np.inf

    for x in range(height):
        for y in range(width):
            min_dist = np.min([
                np.sqrt((x - i) ** 2 + (y - j) ** 2) for i, j in B
                ])
            dist_map[x, y] = min_dist        
    
    # TODO: Implement forward pass (top-left to bottom-right)
    forward_pass_offsets  = [(0,-1), (-1,1), (-1,0), (-1,-1)]
    forward_pass = np.zeros_like(binary_image, dtype=float)

    for x in range(height):
        for y in range(width):
            if binary_image[x, y] == 0:
                continue
            neighbours = []

            for i, j in forward_pass_offsets:
                if 0 <= x + i < height and 0 <= y + j < width:
                    neighbours.append(dist_map[x + i, y + j] + np.sqrt(i**2 + j**2))
                else:
                    neighbours.append(np.inf)
            
            forward_pass[x, y] = np.min(neighbours)
                
    # TODO: Implement backward pass (bottom-right to top-left)
    backward_pass_offsets  = [(1,-1), (1,0), (1,1), (0,1)]
    backward_pass = np.zeros_like(binary_image, dtype=float)

    for x in range(height-1, 0, -1):
        for y in range(width-1, 0, -1):
            if binary_image[x, y] == 0:
                continue
            backward_pass[x, y] = np.min([
                forward_pass[x, y], 
                np.min([
                    forward_pass[x + i, y + j] + np.sqrt(i**2 + j**2) if 0 <= x + i < height and 0 <= y + j < width
                    else np.inf
                    for i, j in backward_pass_offsets
                ])
            ])

    dist_map = backward_pass
    
    return dist_map

def evaluate_distance_transform(dist_map, ground_truth):
    """
    Evaluate the accuracy of the distance transform.
    
    Args:
        dist_map (np.ndarray): Computed distance transform
        ground_truth (np.ndarray): Ground truth distance transform
    
    Returns:
        float: Mean absolute error between computed and ground truth
    """
    # TODO: Implement evaluation metric
    abs_diff = np.abs(dist_map - ground_truth)
    
    return np.mean(abs_diff)

def visualize(img, ground_truth, calculated, name, error):
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(name)
    
    # plot original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='binary_r')
    plt.title("Binary Image")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Ground Truth Heatmap")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 3, 3)
    plt.imshow(calculated, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f"Calculated Heatmap: Error = {error}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(name)

if __name__ == "__main__":
    for i, name in enumerate(['square', 'circle', 'triangle']):
        img = np.load(f"test_images/task1/{name}.npy")
        ground_truth = np.load(f"test_images/task1/{name}_ground_truth_dist_map.npy")
        calculated = distance_transform(img)
        error = evaluate_distance_transform(ground_truth, calculated)
        visualize(img, ground_truth, calculated, name, error)