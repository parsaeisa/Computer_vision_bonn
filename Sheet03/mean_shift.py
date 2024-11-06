import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(distance, bandwidth):
    """
    Compute Gaussian kernel weight.
    
    Args:
        distance (float): Distance between points
        bandwidth (float): Kernel bandwidth parameter
    
    Returns:
        float: Kernel weight
    """
    # TODO: Implement Gaussian kernel
    return np.exp(-1 * distance / (2 * bandwidth**2))

def mean_shift_step(point, points, bandwidth):
    """
    Perform one step of mean shift.
    
    Args:
        point (np.ndarray): Current point
        points (np.ndarray): All points in the feature space
        bandwidth (float): Kernel bandwidth
    
    Returns:
        np.ndarray: New position after one step
    """
    # TODO: Implement single mean shift step
    distances = np.linalg.norm(points - point, axis=1)
    
    weights = gaussian_kernel(distances, bandwidth)
    
    numerator = np.sum(weights[:, np.newaxis] * points, axis=0)
    denominator = np.sum(weights)

    return numerator / denominator

def mean_shift_segmentation(image, bandwidth, max_iter=50):
    """
    Perform mean shift segmentation on an RGB image.
    
    Args:
        image (np.ndarray): RGB image normalized to [0,1]
        bandwidth (float): Kernel bandwidth
        max_iter (int): Maximum number of iterations
    
    Returns:
        np.ndarray: Segmented image
    """
    # TODO: Implement mean shift segmentation
    # TODO: Implement convergence check
    # TODO: Create final segmentation

    flat_image = image.reshape(-1, 3)
    segmented_image = np.copy(flat_image)
    
    for i, point in enumerate(segmented_image):
        for _ in range(max_iter):
            new_position = mean_shift_step(point, flat_image, bandwidth)
            
            if np.linalg.norm(new_position - point) < 0.5:
                break
            
            point = new_position
        
        segmented_image[i] = point

    return segmented_image.reshape(image.shape)

def normalize_image(image):
    # Convert the image to float to avoid integer division issues
    image = image.astype(np.float32)
    # Subtract the minimum and divide by the range
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image

def visualize(original, segmented, name):
    
    plt.figure(figsize=(10, 5))
    plt.suptitle(name)
    
    # plot original
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    # plot segmented
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title("Segmented")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(name)

if __name__ == "__main__":
    bandwidth = 0.5 # SET PARAMETER
    
    for i, name in enumerate(['simple', 'gradient', 'concentric']):
        original = np.load(f"test_images/task3/{name}.npy")
        segmented = mean_shift_segmentation(original, bandwidth=bandwidth, max_iter=50)
        segmented_normalized = normalize_image(segmented)
        visualize(original, segmented_normalized, name)