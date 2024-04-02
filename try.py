import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Define your quality criteria weights
weight_resolution = 1.0
weight_sharpness = 1.0
# Add weights for other criteria as needed

# Define your quality criteria (modify these functions as needed)
def evaluate_resolution(image):
    # Implement logic to assess image resolution (e.g., minimum pixel count)
    # Return a score (higher score indicates better resolution)
    return image.shape[0] * image.shape[1]

def evaluate_sharpness(image):
    # Implement logic to assess image sharpness (e.g., variance of Laplacian filter)
    # Return a score (higher score indicates better sharpness)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# ... Define similar functions for other quality criteria

def quality_score(image):
    # Combine individual quality scores into a single metric (weighted average is common)
    resolution_score = evaluate_resolution(image)
    sharpness_score = evaluate_sharpness(image)
    # ... Add scores from other criteria
    combined_score = (resolution_score * weight_resolution) + (sharpness_score * weight_sharpness)
    return combined_score

def select_best_quality(cluster):
    best_image = None
    best_score = -float('inf')  # Initialize with negative infinity
    for image in cluster:
        score = quality_score(image)
        if score > best_score:
            best_image = image
            best_score = score
    return best_image

# Define input and output folders
input_folder = "input images"
output_folder = "output images"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load your image dataset from input folder
image_list = os.listdir(input_folder)
images = [cv2.imread(os.path.join(input_folder, image_path)) for image_path in image_list]

# Clustering (replace with your preferred clustering algorithm)
num_clusters = 10  # Example number of clusters
features = np.array([image.flatten() for image in images])
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)  # Replace 'features' with actual feature vectors
cluster_labels = kmeans.labels_

# Select best image from each cluster and save to output folder
for cluster_id in range(num_clusters):
    cluster = [images[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
    best_image = select_best_quality(cluster)
    output_path = os.path.join(output_folder, f"best_image_cluster_{cluster_id}.jpg")
    cv2.imwrite(output_path, best_image)
