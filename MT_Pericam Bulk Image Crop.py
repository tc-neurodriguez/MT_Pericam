import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Define the base path and the channels
base_path = r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\SY57_Jan_27_2025"
channels = ["RFP", "GFP", "BFP", "BF"]

# Define ROI coordinates
roi_x, roi_y, roi_width, roi_height = 1300, 750, 500, 500
print("ROI coordinates defined.")

# Loop through each channel
for channel in channels:
    # Construct the file path for the current channel
    file_pattern = f"SY5Y_Pericam_{channel}_128MOI.tif"
    file_path = os.path.join(base_path, file_pattern)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {file_path}")
        continue

    # Convert to RGB for display
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw rectangle on the original image to highlight the ROI
    cv2.rectangle(RGB_img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    # Crop the ROI from the original image
    roi = RGB_img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Display the original image with the ROI highlighted and the cropped image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(RGB_img)
    plt.title(f"Original Image with ROI ({channel})")

    plt.subplot(1, 2, 2)
    plt.imshow(roi)
    plt.title(f"Cropped Image ({channel})")

    plt.show()