import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
# Replace 'your_file.xlsx' with the actual file path
file_path = r'C:\Users\t_rod\Box\SY5Y transduction Images\BioSpa_Data\MT-PericamTestData3_TPP_ON.xlsx'

# Assuming the data for each wavelength combination is in separate sheets
data_405_470 = pd.read_excel(file_path, sheet_name='Sheet1')  # Replace with the correct sheet name
data_405_520 = pd.read_excel(file_path, sheet_name='Sheet2')  # Replace with the correct sheet name
data_488_520 = pd.read_excel(file_path, sheet_name='Sheet3')  # Replace with the correct sheet name
data_540_580 = pd.read_excel(file_path, sheet_name='Sheet4')  # Replace with the correct sheet name

# Combine the data into a single DataFrame
# Add a column to indicate the wavelength combination
data_405_470['Wavelength'] = '405_470'
data_405_520['Wavelength'] = '405_520'
data_488_520['Wavelength'] = '488_520'
data_540_580['Wavelength'] = '540_580'
combined_data = pd.concat([data_405_470, data_405_520, data_488_520, data_540_580], ignore_index=True)

# Define the wells to analyze
wells = ['B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']

# Convert the Time column to numeric values (hours since the start)
# Assuming the Time column is in datetime.time format
combined_data['Time'] = combined_data['Time'].apply(
    lambda x: x.hour + x.minute / 60 + x.second / 3600
)

# Subtract background from each measurement
# Assuming the background column is named 'Background'
if 'Background' not in combined_data.columns:
    raise KeyError("Background column not found. Please ensure the column is named 'Background'.")

for well in wells:
    combined_data[well] = combined_data[well] - combined_data['Background']

# Define groups
groups = {
    '16_TPP': ['B3', 'B4'],
    '16_Con': ['B5', 'B6'],
    '32_TPP': ['B7', 'B8'],
    '32_Con': ['B9', 'B10'],
}

# Calculate group averages for background-subtracted data
background_subtracted_averages = combined_data.copy()
for group, wells_in_group in groups.items():
    background_subtracted_averages[group] = combined_data[wells_in_group].mean(axis=1)

# Plot background-subtracted data in a multipanel figure
wavelengths = ['405_470', '405_520', '488_520', '540_580']
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns
fig1.suptitle('Background-Subtracted Fluorescence Over Time', fontsize=16)

for i, wavelength in enumerate(wavelengths):
    ax = axes1[i // 2, i % 2]  # Select the appropriate subplot
    for group in groups.keys():
        subset = background_subtracted_averages[background_subtracted_averages['Wavelength'] == wavelength]
        ax.plot(subset['Time'], subset[group], label=group)

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Fluorescence Intensity (Background-Subtracted)')
    ax.set_title(wavelength)
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()

# Calculate F0 (baseline fluorescence) for each well and wavelength combination
F0 = combined_data.groupby('Wavelength')[wells].first()

# Calculate ΔF/F0 for each well and time point
delta_F_over_F0 = combined_data.copy()
for well in wells:
    for wavelength in ['405_470', '405_520', '488_520', '540_580']:
        F0_value = F0.loc[wavelength, well]
        delta_F_over_F0.loc[delta_F_over_F0['Wavelength'] == wavelength, well] = (
                (delta_F_over_F0[well] - F0_value) / F0_value
        )

# Add the Time column back for plotting
delta_F_over_F0['Time'] = combined_data['Time']

# Calculate group averages for ΔF/F0
group_averages = delta_F_over_F0.copy()
for group, wells_in_group in groups.items():
    group_averages[group] = delta_F_over_F0[wells_in_group].mean(axis=1)

# Plot ΔF/F0 in a multipanel figure
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns
fig2.suptitle('ΔF/F₀ Over Time for Each Wavelength Combination', fontsize=16)

for i, wavelength in enumerate(wavelengths):
    ax = axes2[i // 2, i % 2]  # Select the appropriate subplot
    for group in groups.keys():
        subset = group_averages[group_averages['Wavelength'] == wavelength]
        ax.plot(subset['Time'], subset[group], label=group)

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('ΔF/F₀')
    ax.set_title(wavelength)
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()