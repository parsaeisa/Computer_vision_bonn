import numpy as np
import matplotlib.pyplot as plt

POINTS_COUNT = 56

def read_data(file_path):
    """Reads the hands_orig_train.txt.new file and extracts key points."""
    data = np.loadtxt(file_path)
    x_coords = data[:POINTS_COUNT, :]
    y_coords = data[POINTS_COUNT:, :]
    keypoints = np.stack((x_coords, y_coords), axis=-1)  # Shape: (56, 39, 2)
    return keypoints

# ========================== Mean =============================
def calculate_mean_shape(kpts):
    """Calculates the mean shape of keypoints."""
    return np.mean(kpts, axis=1)  # Average across all images

# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    """Aligns each shape to the reference mean using Procrustes alignment."""
    aligned_kpts = []

    print(">>>>>>>>>>>>>>>>>>", kpts.shape)
    kpts = kpts.transpose(1, 0, 2)  # Rearrange axes to (39, 56, 2)
    print(">>>>>>>>>>>>>>>>>>", kpts.shape)

    for shape in kpts:
        # Center the shapes
        centered_shape = shape - np.mean(shape, axis=0)
        centered_ref = reference_mean - np.mean(reference_mean, axis=0)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(centered_shape.T @ centered_ref)
        R = U @ Vt  # Rotation matrix

        # Apply transformation
        aligned_shape = centered_shape @ R
        aligned_kpts.append(aligned_shape)

    return np.stack(aligned_kpts, axis=1)  # Shape: (56, 39, 2)

# =========================== Error ====================================
def compute_avg_error(kpts, mean_shape):
    """Computes the average RMS error between shapes and the mean shape."""
    errors = np.linalg.norm(kpts - mean_shape[:, np.newaxis, :], axis=(0, 2))
    return np.mean(errors)

# ============================ Procrustres ===============================
def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):
    """Performs Procrustes Analysis on the given keypoints."""
    aligned_kpts = kpts.copy()

    for iter in range(max_iter):
        # Compute the reference mean shape
        reference_mean = calculate_mean_shape(aligned_kpts)

        # Align shapes to the mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        # Compute RMS error
        avg_error = compute_avg_error(aligned_kpts, reference_mean)

        print(f"Iteration {iter + 1}: RMS Error = {avg_error}")

        # Check for convergence
        if avg_error < min_error:
            print("Convergence reached.")
            break

    return aligned_kpts, reference_mean

#############################################################################
####                           Execution                                #####
#############################################################################
file_path = "data\hands_orig_train.txt.new"
keypoints = read_data(file_path)
aligned_kpts, mean_shape = procrustres_analysis(keypoints)

#############################################################################
####                           Visualization                            #####
#############################################################################
plt.figure(figsize=(12, 6))

# Plot original shapes
plt.subplot(1, 2, 1)
for shape in keypoints.T:
    plt.plot(shape[:, 0], shape[:, 1], 'b-', alpha=0.3)
plt.title("Original Shapes")
plt.gca().invert_yaxis()

# Plot aligned shapes
plt.subplot(1, 2, 2)
for shape in aligned_kpts.T:
    plt.plot(shape[:, 0], shape[:, 1], 'g-', alpha=0.3)
plt.plot(mean_shape[:, 0], mean_shape[:, 1], 'r-', linewidth=2, label='Mean Shape')
plt.legend()
plt.title("Aligned Shapes and Mean Shape")
plt.gca().invert_yaxis()

plt.show()