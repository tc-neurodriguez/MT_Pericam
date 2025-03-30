import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = r"C:\Users\t_rod\Box\SY5Y transduction Images\Experiment 3_9_25\MT-Pericam_3_9_25_FCCP_python.xlsx"
wells = ['B3','B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10','C3', 'C4','C5', 'C6', 'C7', 'C8', 'C9', 'C10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10','E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']

groups = {
    'Vehicle (0.1% DMSO)': ['B4','B8','B10'],
    '0.1_uM_FCCP': ['C3','C5', 'C6', 'C7', 'C8', 'C9', 'C10'],
    '1_uM_FCCP': ['D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
    '10_uM_FCCP': ['E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
}
background_wells = [f'G{i}' for i in range(3, 11)]
# 1. Data Loading with Debugging
def load_and_process(sheet_name, wavelength):
    print(f"\n⏳ Loading {wavelength} from {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Clean columns
    df.columns = df.columns.str.strip()
    print(f"📊 Initial columns: {df.columns.tolist()}")

    # Calculate the average of each row (timepoint) for wells G3-10
    df['Background_Avg'] = df[background_wells].mean(axis=1)

    for well in wells:
        if well in df.columns:
            df[well] = df[well] - df['Background_Avg']
        else:
            print(f"Warning: Well {well} not found in the data.")

    # Melt data
    melted = df.melt(
        id_vars=['Time'],
        value_vars=wells,
        var_name='Well',
        value_name=wavelength
    )
    print(f"🔄 Melted data shape: {melted.shape}")

    return melted


# 2. Load and merge data
print("\n" + "=" * 50)
print("🚀 STARTING DATA PROCESSING")
print("=" * 50)

wavelengths = {
    'Sheet1': '405_470',
    'Sheet2': '405_520',
    'Sheet3': '488_520',
    'Sheet4': '540_580'
}

merged = None
for sheet, name in wavelengths.items():
    df = load_and_process(sheet, name)
    if merged is None:
        merged = df
    else:
        merged = pd.merge(merged, df, on=['Time', 'Well'], how='outer')
    print(f"🔗 Merged shape after {name}: {merged.shape}")
    print(f"🔍 NaN count: {merged.isna().sum().sum()}")

print("\n🔎 Final merged data preview:")
print(merged.head())
print(f"\n📉 Missing values per column:")
print(merged.isna().sum())

# 3. Calculate ratios
print("\n" + "=" * 50)
print("🧮 CALCULATING RATIOS")
print("=" * 50)

ratio_config = {
    '405_520/540_580': ('405_520', '540_580'),
    '488_520/540_580': ('488_520', '540_580'),
    '540_580/(405_470+540_580)': (('405_470', '540_580'), '540_580')
}

# ratio_config = {
#     '405_520/540_580': ('405_520', '540_580'),
#     '488_520/540_580': ('488_520', '540_580'),
#     '540_580/(405_470+540_580)': (('405_470', '540_580'), '540_580')
# }
# Ensure the DataFrame is sorted by time (if not already)
merged = merged.sort_values(by='Time')

# Define the baseline time point (time = 0)
baseline_time_point = 0


# Calculate F₀ for each fluorescence channel at time = 0
F0_405_520 = merged.loc[merged['Time'] == baseline_time_point, '405_520'].values[0]
F0_488_520 = merged.loc[merged['Time'] == baseline_time_point, '488_520'].values[0]
F0_540_580 = merged.loc[merged['Time'] == baseline_time_point, '540_580'].values[0]
F0_405_470 = merged.loc[merged['Time'] == baseline_time_point, '405_470'].values[0]

# Update the ratio_config to include ΔF/F₀ for the first time point (5 minutes)
ratio_config = {
    '488_520/405_520': ('488_520', '405_520'),
    '405_520/540_580': ('405_520', '540_580'),
    '488_520/540_580': ('488_520', '540_580'),
    '540_580/(405_470+540_580)': (('405_470', '540_580'), '540_580'),
    'Delta_F405_520/F0_405_520': ('405_520', F0_405_520),
    'Delta_F488_520/F0_488_520': ('488_520', F0_488_520),
    'Delta_F540_580/F0_540_580': ('540_580', F0_540_580),
    'Delta_F405_470/F0_405_470': ('405_470', F0_405_470)
}

# Calculate the ratios and ΔF/F₀
for ratio, parts in ratio_config.items():
    if isinstance(parts[0], tuple):
        # Handle complex ratio: denominator is sum
        num = merged[parts[1]]
        denom = merged[parts[0][0]] + merged[parts[0][1]]
    elif 'Delta' in ratio:
        # Handle ΔF/F₀
        num = -(merged[parts[0]] - parts[1])  # ΔF = F - F₀
        denom = parts[1]  # F₀
    else:
        # Handle simple ratio
        num = merged[parts[0]]
        denom = merged[parts[1]]

    merged[ratio] = num / denom
    merged[ratio].replace([np.inf, -np.inf], np.nan, inplace=True)

# Filter the DataFrame to only include time points after the baseline (time > 0)
merged = merged[merged['Time'] > baseline_time_point]
wells = ['B4', 'B8', 'B10','C3','C5', 'C6', 'C7', 'C8', 'C9', 'C10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10','E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']

# Now 'merged' contains the new ratios and ΔF/F₀ values for time points after the baseline
print(merged)
merged_5 = merged[merged['Time'] == 5]
merged_5trim = merged_5[merged_5['Well'].isin(wells)]
print(merged_5trim)
merged_5= merged_5trim

merged_5 = merged_5.reset_index(drop=True)
# Calculate z-scores independently for each column
z_scores_488_520_405_520 = stats.zscore(merged_5['488_520/405_520'])
z_scores_405_520_540_580 = stats.zscore(merged_5['405_520/540_580'])
z_scores_488_520_540_580 = stats.zscore(merged_5['488_520/540_580'])
z_scores_540_580_405_470 = stats.zscore(merged_5['540_580/(405_470+540_580)'])
z_scores_Delta_F405_520 = stats.zscore(merged_5['Delta_F405_520/F0_405_520'])
z_scores_Delta_F488_520 = stats.zscore(merged_5['Delta_F488_520/F0_488_520'])

# Store the z-scores in a DataFrame (optional)
z_scores_df = pd.DataFrame({
    '488_520/405_520_zscore': z_scores_488_520_405_520,
    '405_520/540_580_zscore': z_scores_405_520_540_580,
    '488_520/540_580_zscore': z_scores_488_520_540_580,
    '540_580/(405_470+540_580)_zscore': z_scores_540_580_405_470,
    'Delta_F405_520/F0_405_520_zscore': z_scores_Delta_F405_520,
    'Delta_F488_520/F0_488_520_zscore': z_scores_Delta_F488_520
})

# Print the z-scores DataFrame
print(z_scores_df)
# Perform a left join to merge the 'Well' column from merged_5 with z_scores_df
merged_with_wells = pd.merge(
    merged_5[['Well']],  # Left DataFrame: Only the 'Well' column from merged_5
    z_scores_df,         # Right DataFrame: z_scores_df
    left_index=True,      # Use the index of the left DataFrame for merging
    right_index=True,     # Use the index of the right DataFrame for merging
    how='left'            # Perform a left join
)
print(merged_with_wells)


# Create a 12x8 grid (96-well plate) for each z-score column
def create_plate_grid(data, well_column, value_column):
    plate = np.full((8, 12), np.nan)  # 8 rows (A-H) x 12 columns (1-12)
    for _, row in data.iterrows():
        well = row[well_column]
        row_idx = ord(well[0].upper()) - ord('A')  # Convert A-H to 0-7
        col_idx = int(well[1:]) - 1  # Convert 1-12 to 0-11
        plate[row_idx, col_idx] = row[value_column]
    return plate

# Create grids for each z-score column
plate_488_520_405_520 = create_plate_grid(merged_with_wells, 'Well', '488_520/405_520_zscore')
plate_405_520_540_580 = create_plate_grid(merged_with_wells, 'Well', '405_520/540_580_zscore')
plate_488_520_540_580 = create_plate_grid(merged_with_wells, 'Well', '488_520/540_580_zscore')
plate_540_580_405_470 = create_plate_grid(merged_with_wells, 'Well', '540_580/(405_470+540_580)_zscore')
plate_Delta_F405_520 = create_plate_grid(merged_with_wells, 'Well', 'Delta_F405_520/F0_405_520_zscore')
plate_Delta_F488_520 = create_plate_grid(merged_with_wells, 'Well', 'Delta_F488_520/F0_488_520_zscore')

import seaborn as sns
import matplotlib.pyplot as plt

# # Function to plot a heatmap for a single plate
# def plot_plate_heatmap(plate, title, cmap='RdBu'):
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(plate, annot=True, fmt=".2f", cmap=cmap, cbar=True,
#                 square=True, linewidths=0.5, linecolor='black')
#     plt.title(title)
#     plt.xlabel('Column (1-12)')
#     plt.ylabel('Row (A-H)')
#     plt.xticks(np.arange(12) + 0.5, labels=np.arange(1, 13))
#     plt.yticks(np.arange(8) + 0.5, labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],rotation=0)
#     # Adjust tick spacing from the axes
#     # Adjust the spacing between the heatmap and the axes
#     plt.xlim(-0.25, 12.25)  # Extend x-axis limits
#     plt.ylim(8.25,-0.25)  # Extend y-axis limits (inverted for heatmap)
#
#     plt.show()
#
# # Plot heatmaps for each z-score column
# plot_plate_heatmap(plate_488_520_405_520, 'FCCP 488_520/405_520 Z-Scores')
# plot_plate_heatmap(plate_405_520_540_580, 'FCCP 405_520/540_580 Z-Scores')
# plot_plate_heatmap(plate_488_520_540_580, 'FCCP 488_520/540_580 Z-Scores')
# plot_plate_heatmap(plate_540_580_405_470, 'FCCP 540_580/(405_470+540_580) Z-scores')
# plot_plate_heatmap(plate_Delta_F405_520, 'FCCP Delta_F405_520/F0_405_520 Z-Scores')
# plot_plate_heatmap(plate_Delta_F488_520, 'FCCP Delta_F488_520/F0_488_520 Z-Scores')


# Function to plot a heatmap for a single plate
def plot_plate_heatmap(ax, plate, title, cmap='RdBu', ylabel_rotation=0):
    sns.heatmap(plate, annot=True, fmt=".2f", cmap=cmap, cbar=False,  # Disable individual colorbars
                square=True, linewidths=0.5, linecolor='black', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Column (1-12)')
    ax.set_ylabel('Row (A-H)')

    # Set custom x and y ticks
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], rotation=ylabel_rotation)

    # Adjust the spacing between the heatmap and the axes
    ax.set_xlim(-0.5, 12.5)  # Extend x-axis limits
    ax.set_ylim(8.5, -0.5)  # Extend y-axis limits (inverted for heatmap)


# Create a multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows, 2 columns of subplots
axes = axes.ravel()  # Flatten the 2x2 array of axes for easy iteration

plot_plate_heatmap(axes[0],plate_488_520_405_520, 'FCCP 488_520/405_520 Z-Scores')
plot_plate_heatmap(axes[1],plate_405_520_540_580, 'FCCP 405_520/540_580 Z-Scores')
plot_plate_heatmap(axes[2],plate_488_520_540_580, 'FCCP 488_520/540_580 Z-Scores')
plot_plate_heatmap(axes[3],plate_540_580_405_470, 'FCCP 540_580/(405_470+540_580) Z-scores')

# Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position as needed
fig.colorbar(axes[0].collections[0], cax=cbar_ax)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
plt.show()




















# 4. Group analysis
print("\n" + "=" * 50)
print("👥 GROUP ANALYSIS")
print("=" * 50)

group_means = []
for group_name, wells in groups.items():
    print(f"\n📊 Processing {group_name} ({', '.join(wells)})")
    group_data = merged[merged['Well'].isin(wells)]

    print(f"🔍 Group data shape: {group_data.shape}")
    print(f"📅 Time points: {len(group_data['Time'].unique())}")

    group_mean = group_data.groupby('Time').mean(numeric_only=True).reset_index()
    group_mean['Group'] = group_name

    print(f"📈 Group means shape: {group_mean.shape}")
    print(f"📅 Mean time points: {len(group_mean['Time'])}")

    group_means.append(group_mean)

final_df = pd.concat(group_means)
print("\n🔍 Final combined data:")
print(final_df.head())
print(f"\n📊 Data counts per group:\n{final_df['Group'].value_counts()}")

# 5. Plotting
print("\n" + "=" * 50)
print("🎨 PLOTTING RESULTS")
print("=" * 50)

ratios = ['405_520/540_580', '488_520/540_580', '540_580/(405_470+540_580)']
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

for idx, ratio in enumerate(ratios):
    ax = axs[idx]
    print(f"\n📈 Plotting {ratio}")

    valid_groups = 0
    for group in groups:
        plot_df = final_df[(final_df['Group'] == group) &
                           (final_df[ratio].notna())]

        if plot_df.empty:
            print(f"⚠️ No data for {group} in {ratio}")
            continue

        print(f"📌 {group}: {len(plot_df)} data points")
        ax.plot(plot_df['Time'], plot_df[ratio],
                label=group)
        valid_groups += 1

    if valid_groups == 0:
        print(f"🚨 No valid data for {ratio}")
        continue

    ax.set_title(ratio, fontsize=14)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Ratio', fontsize=12)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
print("\n✅ Processing complete!")