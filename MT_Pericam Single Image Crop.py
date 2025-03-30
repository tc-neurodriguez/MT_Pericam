import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import image
image = cv2.imread(r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\SY57_Jan_27_2025\SY5Y_Pericam_RFP_128MOI.tif")
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define ROI coordinates
roi_x, roi_y, roi_width, roi_height = 1300, 750, 500, 500
print("ROI coordinates defined.")

# Draw rectangle on the original image to highlight the ROI
cv2.rectangle(RGB_img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

# Crop the ROI from the original image
roi = RGB_img[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
print(roi)

# Display the original image with the ROI highlighted and the cropped image
plt.subplot(1, 2, 1)
plt.imshow(RGB_img)
plt.title("Original Image with ROI")

plt.subplot(1, 2, 2)
plt.imshow(roi)
plt.title("Cropped Image")

plt.show()