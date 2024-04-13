import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


def transform_s2dest(source_lm, dest_lm):
    # Procrustes analysis function as described previously
    source_center = np.mean(source_lm, axis=0)
    dest_center = np.mean(dest_lm, axis=0)
    source_lm_centered = source_lm - source_center
    dest_lm_centered = dest_lm - dest_center

    source_norm = np.linalg.norm(source_lm_centered, 'fro')
    dest_norm = np.linalg.norm(dest_lm_centered, 'fro')
    source_lm_normalized = source_lm_centered / source_norm
    dest_lm_normalized = dest_lm_centered / dest_norm

    R, s = orthogonal_procrustes(dest_lm_normalized, source_lm_normalized)
    translation = dest_center - (source_center * dest_norm / source_norm) @ R.T
    scale = dest_norm / source_norm

    return {
        'scale': scale,
        'rotation': R,
        'translation': translation
    }


def apply_transformation(source_lm, R, scale, translation):
    """Apply the calculated transformation parameters to the source landmarks."""
    # transformed = (source_lm - np.mean(source_lm, axis=0)) / np.linalg.norm(source_lm - np.mean(source_lm, axis=0), 'fro')
    transformed = source_lm @ R.T * scale
    transformed = transformed + translation
    return transformed


if __name__ == '__main__':
    # Example usage
    source_landmarks = np.array([[1, 2], [3, 4], [5, 6]])
    destination_landmarks = np.array([[3, 4.5], [5, 6.5], [7, 8.5]])

    transform = transform_s2dest(source_landmarks, destination_landmarks)
    print(transform)
    R, scale, translation = transform['rotation'], transform['scale'], transform['translation']
    transformed_source = apply_transformation(source_landmarks, R, scale, translation)

    # Plotting the landmarks
    plt.figure(figsize=(8, 8))
    plt.scatter(source_landmarks[:, 0], source_landmarks[:, 1], c='red', label='Source')
    plt.scatter(destination_landmarks[:, 0], destination_landmarks[:, 1], c='blue', label='Destination')
    plt.scatter(transformed_source[:, 0], transformed_source[:, 1], c='magenta', label='Transformed Source', marker='x')
    plt.legend()
    plt.show()
