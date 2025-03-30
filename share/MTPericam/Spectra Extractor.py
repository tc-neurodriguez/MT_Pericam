import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\Spectral Comparions\ratiometric_pericam_abs2.png", cv2.IMREAD_GRAYSCALE)


# Thresholding to isolate the spectra
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours of the spectra
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to remove small noise
min_contour_area = 100  # Adjust this value based on your image
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Sort contours by their x-coordinate (left to right)
filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])

# Function to merge broken segments of a contour
def merge_broken_segments(contour, kernel_size=(1, 1), iterations=1):
    # Create a blank image to draw the contour
    contour_image = np.zeros_like(binary)
    cv2.drawContours(contour_image, [contour], -1, 255, thickness=cv2.FILLED)

    # Dilate the contour to merge broken segments
    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(contour_image, kernel, iterations=iterations)

    # Find the merged contour
    merged_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(merged_contours) == 0:
        return contour  # Return the original contour if no merged contour is found
    return merged_contours[0]  # Return the largest merged contour

# Merge broken segments for each contour
merged_contours = [merge_broken_segments(cnt) for cnt in filtered_contours]

# Check if exactly two contours are detected
if len(merged_contours) != 2:
    raise ValueError(f"Expected 2 contours, but found {len(merged_contours)}. Adjust the thresholding or preprocessing.")

# Function to extract and interpolate spectrum data
def extract_spectrum(contour, wavelength_min, wavelength_max, absorbance_min, absorbance_max):
    # Extract x and y coordinates from the contour
    x = [point[0][0] for point in contour]
    y = [point[0][1] for point in contour]

    # Map pixel x-coordinates to wavelength
    wavelength = np.linspace(wavelength_min, wavelength_max, len(x))

    # Map pixel y-coordinates to absorbance
    absorbance = np.interp(y, [min(y), max(y)], [absorbance_min, absorbance_max])

    # Interpolate to 1 nm increments
    f = interp1d(wavelength, absorbance, kind='linear', fill_value='extrapolate')
    wavelength_new = np.arange(wavelength_min, wavelength_max + 1, 1)
    absorbance_new = f(wavelength_new)

    return wavelength_new, absorbance_new

# Extract data for both spectra
wavelength_min = 300  # Example: minimum wavelength in nm
wavelength_max = 600  # Example: maximum wavelength in nm
absorbance_min = 0.0   # Example: minimum absorbance
absorbance_max = 1.0   # Example: maximum absorbance

# Extract spectrum 1 (e.g., without Calcium)
wavelength_new1, absorbance_new1 = extract_spectrum(merged_contours[0], wavelength_min, wavelength_max, absorbance_min, absorbance_max)

# Extract spectrum 2 (e.g., with Calcium)
wavelength_new2, absorbance_new2 = extract_spectrum(merged_contours[1], wavelength_min, wavelength_max, absorbance_min, absorbance_max)

# Save to CSV files
data1 = np.column_stack((wavelength_new1, absorbance_new1))
data2 = np.column_stack((wavelength_new2, absorbance_new2))
np.savetxt('absorbance_data_without_calcium.csv', data1, delimiter=',', header='Wavelength (nm),Absorbance', comments='')
np.savetxt('absorbance_data_with_calcium.csv', data2, delimiter=',', header='Wavelength (nm),Absorbance', comments='')

# Create a figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot the original image
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')  # Hide axes for the image

# Plot the binary image
ax2.imshow(binary, cmap='gray')
ax2.set_title('Binary Image')
ax2.axis('off')  # Hide axes for the binary image

# Plot the merged contours on the binary image
contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to color image for drawing contours
cv2.drawContours(contour_image, merged_contours, -1, (0, 255, 0), 2)  # Draw contours in green
ax3.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
ax3.set_title('Merged Contours on Binary Image')
ax3.axis('off')  # Hide axes for the contour image

# Plot both extracted absorbance spectra
ax4.plot(wavelength_new1, absorbance_new1, label='Without Calcium')
ax4.plot(wavelength_new2, absorbance_new2, label='With Calcium')
ax4.set_xlabel('Wavelength (nm)')
ax4.set_ylabel('Absorbance')
ax4.set_title('Extracted Absorbance Spectra')
ax4.legend()

# Adjust layout and display the figure
plt.tight_layout()
plt.show()