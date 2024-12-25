import cv2
import numpy as np
import os

# Create output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Function for Niblack Thresholding
def niblack_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    return (image > mean + k * std).astype(np.uint8) * 255

# Function for Sauvola Thresholding
def sauvola_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    R = std.max()
    return (image > mean * (1 + k * (std / R - 1))).astype(np.uint8) * 255

# Function for Christian Thresholding
def christian_threshold(image, window_size, k):
    mean = cv2.blur(image, (window_size, window_size))
    sq_mean = cv2.blur(image ** 2, (window_size, window_size))
    std = np.sqrt(sq_mean - mean ** 2)
    R = std.max()
    return (image > mean - k * (std / R)).astype(np.uint8) * 255

# Load images
images = ["54b27291c24f7552d24d298609e2b628.jpg", "71546899382953f1f375c34c4a99f276.jpg", "image.png", "завантаження.jpg"]
window_size = 128
k_values = [0.2, 0.5]

for img_name in images:
    # Read and convert to grayscale
    print(f"Processing {img_name}...")
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading {img_name}. Ensure the file exists and the name is correct.")
        continue

    # Global Thresholding (Otsu)
    _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_otsu.jpg")
    cv2.imwrite(otsu_path, otsu_thresh)
    print(f"Saved Otsu result: {otsu_path}")

    # Adaptive Thresholding (Niblack)
    niblack_thresh = niblack_threshold(img, window_size, k=0.2)
    niblack_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_niblack.jpg")
    cv2.imwrite(niblack_path, niblack_thresh)
    print(f"Saved Niblack result: {niblack_path}")

    # Adaptive Thresholding (Sauvola)
    sauvola_thresh = sauvola_threshold(img, window_size, k=0.5)
    sauvola_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_sauvola.jpg")
    cv2.imwrite(sauvola_path, sauvola_thresh)
    print(f"Saved Sauvola result: {sauvola_path}")

    # Adaptive Thresholding (Christian)
    christian_thresh = christian_threshold(img, window_size, k=0.5)
    christian_path = os.path.join(output_dir, f"{img_name.split('.')[0]}_christian.jpg")
    cv2.imwrite(christian_path, christian_thresh)
    print(f"Saved Christian result: {christian_path}")

print("Processing complete. Results are saved in the 'results' folder.")

# Save requirements.txt
with open("requirements.txt", "w") as f:
    f.write("opencv-python\n")
    f.write("numpy\n")

# Save requirements.txt
with open("requirements.txt", "w") as f:
    f.write("opencv-python\n")
    f.write("numpy\n")
