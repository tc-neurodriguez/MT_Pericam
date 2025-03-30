import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from skimage.measure import regionprops
import cv2

# Load the BF channel image
bf_image_path = r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\SY57_Jan_27_2025\SY5Y_Pericam_BF_128MOI.tif"
bf_image = io.imread(bf_image_path)

# Check if the image was loaded successfully
if bf_image is None:
    print("Error: Could not load the BF channel image.")
    exit()

# Convert to grayscale if it's not already
if len(bf_image.shape) == 3:  # If the image has multiple channels
    bf_image = np.mean(bf_image, axis=2).astype(np.uint8)  # Convert to grayscale by averaging channels

# Convert to 8-bit if necessary (e.g., for 16-bit TIFF images)
if bf_image.dtype == np.uint16:
    bf_image = (bf_image / 256).astype(np.uint8)  # Scale 16-bit to 8-bit

# Initialize the Cellpose model
model = models.Cellpose(gpu=False, model_type='cyto')  # Use 'cyto' for cytoplasm segmentation

# Set the diameter parameter (adjust based on your cells' size)
cell_diameter = 30  # Example: adjust this value based on your data

# Run Cellpose on the grayscale image
masks, flows, styles, diams = model.eval(bf_image, diameter=cell_diameter, channels=[0, 0])  # [0, 0] for grayscale images

# Calculate properties of detected objects
props = regionprops(masks)

# Create an overlay image
overlay_image = bf_image.copy()
if len(overlay_image.shape) == 2:  # Ensure the image is 3-channel for color annotations
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

# Overlay the segmentation mask on the original image
mask_overlay = np.zeros_like(overlay_image)
mask_overlay[masks > 0] = [255, 0, 0]  # Color the mask in red

# Blend the mask with the original image
alpha = 0.5  # Transparency of the mask
overlay_image = cv2.addWeighted(overlay_image, 1, mask_overlay, alpha, 0)

# Overlay the diameter of each object on the image
for prop in props:
    y, x = prop.centroid  # Get the centroid of the object
    equivalent_diameter = prop.equivalent_diameter  # Calculate the equivalent diameter
    cv2.putText(overlay_image, f"{equivalent_diameter:.1f}", (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)  # Draw the diameter value

# Display the results
plt.figure(figsize=(12, 6))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(bf_image, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

# Overlay image with segmentation mask and diameters
plt.subplot(1, 2, 2)
plt.imshow(overlay_image)
plt.title(f"Segmentation Mask with Diameters (Diameter={cell_diameter})")
plt.axis('off')

plt.tight_layout()
plt.show()