import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the Excel file
# Replace 'your_file.xlsx' with the actual file path
file_path = r"C:\Users\t_rod\Box\SY5Y transduction Images\Experiment 3_9_25\MT-Pericam_3_9_25_FCCP_python.xlsx"

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
wells = ['B4', 'B8', 'B10','C3','C5', 'C6', 'C7', 'C8', 'C9', 'C10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10','E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
background_wells = [f'G{i}' for i in range(3, 11)]
# Convert the Time column to numeric values (hours since the start)
# Assuming the Time column is in datetime.time format

# Calculate the average of each row (timepoint) for wells G3-10
combined_data['Background_Avg'] = combined_data[background_wells].mean(axis=1)
# Subtract background from each measurement
# Subtract the background average from each well in the wells_to_analyze list
for well in wells:
    if well in combined_data.columns:
        combined_data[well] = combined_data[well] - combined_data['Background_Avg']
    else:
        print(f"Warning: Well {well} not found in the data.")
# Define groups
groups = {
    'Vehicle (0.1% DMSO)': ['B4','B8','B10'],
    '0.1_uM_FCCP': ['C3','C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
    '1_uM_FCCP': ['D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
    '10_uM_FCCP': ['E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
}

# Calculate group averages for background-subtracted data
background_subtracted_averages = combined_data.copy()
for group, wells_in_group in groups.items():
    background_subtracted_averages[group] = combined_data[wells_in_group].mean(axis=1)

# Plot background-subtracted data in a multipanel figure
wavelengths = ['405_470', '405_520', '488_520', '540_580']
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))  # 2 rows, 2 columns
fig1.suptitle('FCCP Background-Subtracted Fluorescence Over Time', fontsize=16)

for i, wavelength in enumerate(wavelengths):
    ax = axes1[i // 2, i % 2]  # Select the appropriate subplot
    for group in groups.keys():
        subset = background_subtracted_averages[background_subtracted_averages['Wavelength'] == wavelength]
        ax.plot(subset['Time'], subset[group], label=group)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Fluorescence Intensity (Background-Subtracted)')
    ax.set_title(wavelength)
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the subtitle
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

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('ΔF/F₀')
    ax.set_title(wavelength)
    ax.legend()
    ax.grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()