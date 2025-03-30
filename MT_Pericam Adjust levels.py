import cv2
import numpy as np


# Define a callback function for the trackbars (does nothing but is required)
def nothing(x):
    pass


# Function to adjust image levels manually while preserving the black background
def adjust_levels_manual(image, low, high):
    # Create a mask for the background (pixels close to 0)
    background_mask = image <= low

    # Clip the image to the low and high values
    clipped_image = np.clip(image, low, high)

    # Normalize the non-background pixels to the range [0, 255]
    adjusted_image = np.zeros_like(image, dtype=np.uint8)
    if high > low:
        adjusted_image[~background_mask] = ((clipped_image[~background_mask] - low) * (255.0 / (high - low))).astype(
            np.uint8)

    return adjusted_image


# Load the image
image_path = r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\SY57_Jan_27_2025\SY5Y_Pericam_BF_128MOI.tif"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if the image is single-channel (grayscale)
if len(image.shape) == 2:
    print("Image is grayscale.")
else:
    print("Image is not grayscale. Converting to grayscale.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to 8-bit if necessary (e.g., for 16-bit TIFF images)
if image.dtype == np.uint16:
    image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

# Create a window for the trackbars
cv2.namedWindow("Adjust Levels", cv2.WINDOW_NORMAL)

# Create trackbars for low and high values
cv2.createTrackbar("Low", "Adjust Levels", 0, 255, nothing)
cv2.createTrackbar("High", "Adjust Levels", 255, 255, nothing)

# Loop for manual adjustment
while True:
    # Get current trackbar positions
    low = cv2.getTrackbarPos("Low", "Adjust Levels")
    high = cv2.getTrackbarPos("High", "Adjust Levels")

    # Ensure high > low to avoid errors
    if high <= low:
        high = low + 1

    # Adjust the image levels
    adjusted_image = adjust_levels_manual(image, low, high)

    # Apply a colormap for visualization
    colored_image = cv2.applyColorMap(adjusted_image, cv2.COLORMAP_HOT)  # Use "HOT" colormap for red intensities

    # Display the adjusted image
    cv2.imshow("Adjust Levels", colored_image)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break

# Save the final low and high values
print(f"Final Low Value: {low}")
print(f"Final High Value: {high}")

# Apply the same values to another image (example)
another_image_path = r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\SY57_Jan_27_2025\SY5Y_Pericam_RFP_64MOI.tif"
another_image = cv2.imread(another_image_path, cv2.IMREAD_UNCHANGED)

if len(another_image.shape) == 3:  # Convert to grayscale if necessary
    another_image = cv2.cvtColor(another_image, cv2.COLOR_BGR2GRAY)

if another_image.dtype == np.uint16:
    another_image = cv2.convertScaleAbs(another_image, alpha=(255.0 / 65535.0))

# Apply the saved levels
final_adjusted_image = adjust_levels_manual(another_image, low, high)

# Apply a colormap for visualization
final_colored_image = cv2.applyColorMap(final_adjusted_image, cv2.COLORMAP_HOT)

# Display the final adjusted image
cv2.imshow("Final Adjusted Image", final_colored_image)
cv2.waitKey(0)
cv2.destroyAllWindows()