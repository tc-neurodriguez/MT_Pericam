import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
file_path = r"C:\Users\Tyler\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\2025-03-28\Sweeptest_FCCP_python.xlsx"
wells = ['G3','G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10','F3', 'F4','F5', 'F6', 'F7', 'F8', 'F9', 'F10','D3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']

# Define groups
groups = {
    'Vehicle (0.1% DMSO)': ['G3','G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'F3', 'F10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10','E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'],
    '1_uM_FCCP': ['F4','F5', 'F6', 'F7', 'F8', 'F9']
}




# 1. Data Loading with Debugging
def load_and_process(sheet_name, wavelength):
    print(f"\n⏳ Loading {wavelength} from {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Clean columns
    df.columns = df.columns.str.strip()
    print(f"📊 Initial columns: {df.columns.tolist()}")

    # # Calculate the average of each row (timepoint) for wells G3-10
    # df['Gackground_Avg'] = df[background_wells].mean(axis=1)

    for well in wells:
        if well in df.columns:
            df[well] = df[well]
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
    'Sheet3': '370_470',
    'Sheet1': '415_518',
    'Sheet2': '485_525',
    'Sheet4': '555_586'
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

# ratio_config = {
#     '405_520/555_586': ('405_520', '555_586'),
#     '488_520/555_586': ('488_520', '555_586'),
#     '555_586/(370_470+555_586)': (('370_470', '555_586'), '555_586')
# }
# Ensure the DataFrame is sorted by time (if not already)
merged = merged.sort_values(by='Time')

# Define the baseline time point (time = 0)
baseline_time_point = 0


# Calculate F₀ for each fluorescence channel at time = 0
F0_415_518 = merged.loc[merged['Time'] == baseline_time_point, '415_518'].values[0]
F0_485_525 = merged.loc[merged['Time'] == baseline_time_point, '485_525'].values[0]
F0_555_586 = merged.loc[merged['Time'] == baseline_time_point, '555_586'].values[0]
F0_370_470 = merged.loc[merged['Time'] == baseline_time_point, '370_470'].values[0]
# Add F₀ columns to the DataFrame
merged['F0_415_518'] = F0_415_518
merged['F0_485_525'] = F0_485_525
merged['F0_555_586'] = F0_555_586
merged['F0_370_470'] = F0_370_470
# Update the ratio_config to include ΔF/F₀ and normalized ratios
ratio_config = {
    '485_525/415_518': ('485_525', '415_518'),
    '415_518/555_586': ('415_518', '555_586'),
    '485_525/555_586': ('485_525', '555_586'),
    '555_586/(370_470+555_586)': (('370_470', '555_586'), '555_586'),
    'Delta_F415_518/F0_415_518': ('415_518', 'F0_415_518'),
    'Delta_F485_525/F0_485_525': ('485_525', 'F0_485_525'),
    'Delta_F555_586/F0_555_586': ('555_586', 'F0_555_586'),
    'Delta_F370_470/F0_370_470': ('370_470', 'F0_370_470'),
    'Norm_485_525/415_518': ('Delta_F485_525/F0_485_525', 'Delta_F415_518/F0_415_518'),
    'Norm_415_518/555_586': ('Delta_F415_518/F0_415_518', 'Delta_F555_586/F0_555_586'),
    'Norm_485_525/555_586': ('Delta_F485_525/F0_485_525', 'Delta_F555_586/F0_555_586'),
    'Norm_555_586/(370_470+555_586)': (('Delta_F370_470/F0_370_470', 'Delta_F555_586/F0_555_586'), 'Delta_F555_586/F0_555_586')
}

# Calculate the ratios and ΔF/F₀
for ratio, parts in ratio_config.items():
    try:
        if isinstance(parts[0], tuple):
            # Handle complex ratio: denominator is sum
            if 'Norm' in ratio:
                # Handle normalized complex ratios
                num = merged[parts[1]]
                denom = merged[parts[0][0]] + merged[parts[0][1]]
            else:
                # Handle regular complex ratios
                num = merged[parts[1]]
                denom = merged[parts[0][0]] + merged[parts[0][1]]
        elif 'Delta' in ratio:
            # Handle ΔF/F₀
            num = -(merged[parts[0]] - merged[parts[1]])  # ΔF = F - F₀
            denom = merged[parts[1]]  # F₀
        elif 'Norm' in ratio:
            # Handle normalized ratios
            num = merged[parts[0]]
            denom = merged[parts[1]]
        else:
            # Handle simple ratio
            num = merged[parts[0]]
            denom = merged[parts[1]]

        # Calculate the ratio and handle infinities
        merged[ratio] = num / denom
        merged[ratio].replace([np.inf, -np.inf], np.nan, inplace=True)
    except KeyError as e:
        print(f"Error calculating ratio '{ratio}': {e}. Check if the columns exist in the DataFrame.")
    except Exception as e:
        print(f"Unexpected error calculating ratio '{ratio}': {e}")

# Display the updated DataFrame with the new ratios
print(merged.head())

# Filter the DataFrame to only include time points after the baseline (time > 0)
merged = merged[merged['Time'] > baseline_time_point]
wells = ['G3','G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10','F3', 'F4','F5', 'F6', 'F7', 'F8', 'F9', 'F10','D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10','E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']

# Now 'merged' contains the new ratios and ΔF/F₀ values for time points after the baseline
print(merged)
merged_5 = merged[merged['Time'] == 5]
merged_5trim = merged_5[merged_5['Well'].isin(wells)]
print(merged_5trim)
merged_5= merged_5trim

merged_5 = merged_5.reset_index(drop=True)


# Extract Vehicle group data
vehicle_wells = groups['Vehicle (0.1% DMSO)']
vehicle_data = merged_5[merged_5['Well'].isin(vehicle_wells)]

# Columns to compute z-scores for
columns = [
    '485_525/415_518',
    '415_518/555_586',
    '485_525/555_586',
    '555_586/(370_470+555_586)',
    'Delta_F415_518/F0_415_518',
    'Delta_F485_525/F0_485_525',
    'Norm_485_525/415_518',
    'Norm_415_518/555_586',
    'Norm_485_525/555_586',
    'Norm_555_586/(370_470+555_586)'
]

# Calculate z-scores using Vehicle group's mean and std
z_scores_data = {}
for col in columns:
    vehicle_mean = vehicle_data[col].mean()
    vehicle_std = vehicle_data[col].std()
    z_scores = (merged_5[col] - vehicle_mean) / vehicle_std
    z_scores_data[f'{col}_zscore'] = z_scores

# Create DataFrame and print
z_scores_df = pd.DataFrame(z_scores_data)
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
plate_485_525_415_518 = create_plate_grid(merged_with_wells, 'Well', '485_525/415_518_zscore')
plate_415_518_555_586 = create_plate_grid(merged_with_wells, 'Well', '415_518/555_586_zscore')
plate_485_525_555_586 = create_plate_grid(merged_with_wells, 'Well', '485_525/555_586_zscore')
plate_555_586_370_470 = create_plate_grid(merged_with_wells, 'Well', '555_586/(370_470+555_586)_zscore')
# Create grids for each Norm z-score column
plate_Norm_485_525_415_518 = create_plate_grid(merged_with_wells, 'Well', 'Norm_485_525/415_518_zscore')
plate_Norm_415_518_555_586 = create_plate_grid(merged_with_wells, 'Well', 'Norm_415_518/555_586_zscore')
plate_Norm_485_525_555_586 = create_plate_grid(merged_with_wells, 'Well', 'Norm_485_525/555_586_zscore')
plate_Norm_555_586_370_470 = create_plate_grid(merged_with_wells, 'Well', 'Norm_555_586/(370_470+555_586)_zscore')
plate_Delta_F415_518 = create_plate_grid(merged_with_wells, 'Well', 'Delta_F415_518/F0_415_518_zscore')
plate_Delta_F485_525 = create_plate_grid(merged_with_wells, 'Well', 'Delta_F485_525/F0_485_525_zscore')

import seaborn as sns
import matplotlib.pyplot as plt
#
# # Function to plot a heatmap for a single plate
# def plot_plate_heatmap(ax, plate, title, cmap='RdBu'):
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(plate, annot=True, fmt=".2f", cmap=cmap, cbar=True,
#                 square=True, linewidths=0.5, linecolor='Black',ax=ax)
#     plt.title(title)
#     plt.xlabel('Folumn (1-12)')
#     plt.ylabel('Row (A-H)')
#     plt.xticks(np.arange(12) + 0.5, labels=np.arange(1, 13))
#     plt.yticks(np.arange(8) + 0.5, labels=['A', 'G', 'F', 'D', 'E', 'F', 'G', 'H'],rotation=0)
#     # Adjust tick spacing from the axes
#     # Adjust the spacing between the heatmap and the axes
#     plt.xlim(-0.25, 12.25)  # Extend x-axis limits
#     plt.ylim(8.25,-0.25)  # Extend y-axis limits (inverted for heatmap)
#
#     # plt.show()
# # Create a multi-panel figure
# fig, axes = plt.subplots(4, 4, figsize=(16, 12))  # 2 rows, 2 columns of subplots
# axes = axes.ravel()  # Flatten the 2x2 array of axes for easy iteration
#
# # Plot heatmaps for each z-score column
# plot_plate_heatmap(axes[0], plate_485_525_415_518, 'FCCP 485_525/415_518 Z-Scores')
# plot_plate_heatmap(axes[1],plate_415_518_555_586, 'FCCP 415_518/555_586 Z-Scores')
# plot_plate_heatmap(axes[2],plate_485_525_555_586, 'FCCP 485_525/555_586 Z-Scores')
# plot_plate_heatmap(axes[3],plate_555_586_370_470, 'FCCP 555_586/(370_470+555_586) Z-scores')
# plot_plate_heatmap(axes[4], plate_Norm_485_525_415_518, 'Norm FCCP 485_525/415_518 Z-Scores')
# plot_plate_heatmap(axes[5],plate_Norm_415_518_555_586, 'Norm FCCP 415_518/555_586 Z-Scores')
# plot_plate_heatmap(axes[6],plate_Norm_485_525_555_586, 'Norm FCCP 485_525/555_586 Z-Scores')
# plot_plate_heatmap(axes[7],plate_Norm_555_586_370_470, 'Norm FCCP 555_586/(370_470+555_586) Z-scores')
# # plot_plate_heatmap(plate_Delta_F415_518, 'Thaps Delta_F415_518/F0_415_518 Z-Scores')
# # plot_plate_heatmap(plate_Delta_F485_525, 'Thaps Delta_F485_525/F0_485_525 Z-Scores')
# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()


# Function to plot a heatmap for a single plate
def plot_plate_heatmap(ax, plate, title, vmin=-10, vmax=10, cmap='RdBu', ylabel_rotation=0):
    sns.heatmap(plate, center = 0, vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cmap=cmap, cbar=False,  # Disable individual colorbars
                square=True, linewidths=0.5, linecolor='Black', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Folumn (1-12)')
    ax.set_ylabel('Row (A-H)')

    # Set custom x and y ticks
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_yticklabels(['A', 'G', 'F', 'D', 'E', 'F', 'G', 'H'], rotation=ylabel_rotation)

    # Adjust the spacing between the heatmap and the axes
    ax.set_xlim(-0.5, 12.5)  # Extend x-axis limits
    ax.set_ylim(8.5, -0.5)  # Extend y-axis limits (inverted for heatmap)


# Create a multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 rows, 2 columns of subplots
axes = axes.ravel()  # Flatten the 2x2 array of axes for easy iteration

# Plot each heatmap in a separate subplot
plot_plate_heatmap(axes[0], plate_485_525_415_518, 'FCCP Mito Function 485_525/415_518 Z-Scores')
plot_plate_heatmap(axes[1], plate_415_518_555_586, 'FCCP Mito Calcium 415_518/555_586 Z-Scores')
plot_plate_heatmap(axes[2], plate_485_525_555_586, 'FCCP Mito pH 485_525/555_586 Z-Scores')
plot_plate_heatmap(axes[3], plate_555_586_370_470, 'FCCP Mito Volume 555_586/(370_470+555_586) Z-scores')
# plot_plate_heatmap(axes[4], plate_Norm_485_525_415_518, 'Norm FCCP 485_525/415_518 Z-Scores')
# plot_plate_heatmap(axes[5],plate_Norm_415_518_555_586, 'Norm FCCP 415_518/555_586 Z-Scores')
# plot_plate_heatmap(axes[6],plate_Norm_485_525_555_586, 'Norm FCCP 485_525/555_586 Z-Scores')
# plot_plate_heatmap(axes[7],plate_Norm_555_586_370_470, 'Norm FCCP 555_586/(370_470+555_586) Z-scores')
# # Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position as needed
fig.colorbar(axes[0].collections[0], cax=cbar_ax)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
# Save the figure before showing it
fig.savefig('zscore_heatmaps.png', dpi=300, bbox_inches='tight')  # Adjust filename/path as needed

plt.show()

#############KW#############


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

# Convert the groups dictionary into a DataFrame
group_df = pd.DataFrame([(well, group) for group, wells in groups.items() for well in wells],
                        columns=['Well', 'Group'])

# Merge the group information into the target DataFrame
gm_df = pd.merge(merged_5trim, group_df, on='Well', how='left')

# Function to create a single plot
def create_plot(ax, data, y_col, title, ylabel):
    # Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
    groups = [data[data['Group'] == group][y_col] for group in data['Group'].unique()]
    h_stat, p_value = kruskal(*groups)
    print(f"{title} - Kruskal-Wallis p-value: {p_value}")

    # Perform Dunn's post hoc test
    dunn_results = sp.posthoc_dunn(data, val_col=y_col, group_col='Group', p_adjust='Bonferroni')
    print(f"{title} - Dunn's test results:")
    print(dunn_results)

    # Define your desired group order (replace with your actual group names)
    group_order = ["Vehicle (0.1% DMSO)", "0.1_uM_FCCP", "1_uM_FCCP","10_uM_FCCP"]  # Example


    sns.boxplot(
        data=data,
        x='Group',
        y=y_col,
        palette='husl',
        ax=ax,
        order=group_order  # <-- Control order here
    )
    sns.swarmplot(
        data=data,
        x='Group',
        y=y_col,
        color='Black',
        ax=ax,
        size=4,
        order=group_order  # <-- Must match boxplot order
    )

    # Function to convert p-values to asterisks
    def p_value_to_asterisks(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # Extract significant pairs and p-values from Dunn's test results
    significant_pairs = []
    groups = data['Group'].unique()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Avoid duplicate comparisons
                p_adj = dunn_results.loc[group1, group2]
                if p_adj < 0.05:  # Only include significant pairs
                    significant_pairs.append(((group1, group2), p_adj))

    # Get the y-coordinates of the top of each box
    box_tops = [max(data[data['Group'] == group][y_col]) for group in groups]

    # Adjust the y-axis scale to make more room for annotations
    y_max = max(box_tops)  # Maximum y-value of the boxes
    ax.set_ylim(top=y_max * 1.2)  # Increase the upper limit by 30%

    # Manually annotate the plot with significant pairs
    spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
    for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
        x1 = groups.tolist().index(group1)  # Get x-position of group1
        x2 = groups.tolist().index(group2)  # Get x-position of group2
        y1 = box_tops[x1]  # Top of the box for group1
        y2 = box_tops[x2]  # Top of the box for group2
        y_max_box = max(y1, y2)  # Use the higher of the two box tops
        y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

        # Convert p-value to asterisks
        asterisks = p_value_to_asterisks(p_adj)

        # Draw the line and annotation
        ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
        ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Group')
    ax.set_ylabel(ylabel)

# Create a multipanel figure
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 2 rows, 2 columns
fig.suptitle('Multipanel Figure: Mitochondrial Function via Ratiometric Pericam', fontsize=16)

# Define the data and parameters for each subplot
plots = [
    {
        'y_col': '485_525/415_518',
        'title': 'Mitochondrial Function 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/415_518)'
    },
    {
        'y_col': '415_518/555_586',
        'title': 'Mitochondrial Calcium 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (415_518/555_586)'
    },
    {
        'y_col': '485_525/555_586',
        'title': 'Mitochondrial pH 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/555_586)'
    },
    {
        'y_col': '555_586/(370_470+555_586)',
        'title': 'Mitochondrial volume 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (555_586/(370_470+555_586))'
    }
]

# Populate each subplot with a plot
for ax, plot in zip(axes.flat, plots):
    create_plot(ax, gm_df, plot['y_col'], plot['title'], plot['ylabel'])

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()






##########ANOVA_Oneway###########






import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway
import scikit_posthocs as sp

# Convert the groups dictionary into a DataFrame
group_df = pd.DataFrame([(well, group) for group, wells in groups.items() for well in wells],
                        columns=['Well', 'Group'])

# Merge the group information into the target DataFrame
gm_df = pd.merge(merged_5trim, group_df, on='Well', how='left')
# Function to create a single plot
def create_plot(ax, data, y_col, title, ylabel):
    # Perform one-way ANOVA
    groups = [data[data['Group'] == group][y_col] for group in data['Group'].unique()]
    f_stat, p_value = f_oneway(*groups)
    print(f"{title} - One-Way ANOVA p-value: {p_value}")

    # Perform pairwise t-tests with Holm-Sidak correction
    holm_sidak_results = sp.posthoc_ttest(data, val_col=y_col, group_col='Group', p_adjust='holm-sidak')
    print(f"{title} - Holm-Sidak post hoc results:")
    print(holm_sidak_results)

    # Define your desired group order (replace with your actual group names)



    sns.boxplot(
        data=data,
        x='Group',
        y=y_col,
        palette='husl',
        ax=ax

    )
    sns.swarmplot(
        data=data,
        x='Group',
        y=y_col,
        color='Black',
        ax=ax,
        size=4
    )

    # Function to convert p-values to asterisks
    def p_value_to_asterisks(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # Extract significant pairs and p-values from Holm-Sidak results
    significant_pairs = []
    groups = data['Group'].unique()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Avoid duplicate comparisons
                p_adj = holm_sidak_results.loc[group1, group2]
                if p_adj < 0.05:  # Only include significant pairs
                    significant_pairs.append(((group1, group2), p_adj))

    # Get the y-coordinates of the top of each box
    box_tops = [max(data[data['Group'] == group][y_col]) for group in groups]

    # Adjust the y-axis scale to make more room for annotations
    y_max = max(box_tops)  # Maximum y-value of the boxes
    ax.set_ylim(top=y_max * 1.3)  # Increase the upper limit by 30%

    # Manually annotate the plot with significant pairs
    spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
    for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
        x1 = groups.tolist().index(group1)  # Get x-position of group1
        x2 = groups.tolist().index(group2)  # Get x-position of group2
        y1 = box_tops[x1]  # Top of the box for group1
        y2 = box_tops[x2]  # Top of the box for group2
        y_max_box = max(y1, y2)  # Use the higher of the two box tops
        y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

        # Convert p-value to asterisks
        asterisks = p_value_to_asterisks(p_adj)

        # Draw the line and annotation
        ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
        ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Group')
    ax.set_ylabel(ylabel)

# Create a multipanel figure
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 2 rows, 2 columns
fig.suptitle('Multipanel Figure: Mitochondrial Function and Calcium', fontsize=16)

# Define the data and parameters for each subplot
plots = [
    {
        'y_col': '485_525/415_518',
        'title': 'Mitochondrial Function 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/415_518)'
    },
    {
        'y_col': '415_518/555_586',
        'title': 'Mitochondrial Calcium 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (415_518/555_586)'
    },
    {
        'y_col': '485_525/555_586',
        'title': 'Mitochondrial pH 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/555_586)'
    },
    {
        'y_col': '555_586/(370_470+555_586)',
        'title': 'Mitochondrial volume 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (555_586/(370_470+555_586))'
    }
]

# Populate each subplot with a plot
for ax, plot in zip(axes.flat, plots):
    create_plot(ax, gm_df, plot['y_col'], plot['title'], plot['ylabel'])

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()


# Convert the groups dictionary into a DataFrame
group_df = pd.DataFrame([(well, group) for group, wells in groups.items() for well in wells],
                        columns=['Well', 'Group'])
# Convert the moi groups dictionary into a DataFrame
moi_df = pd.DataFrame([(well, group) for group, wells in moi_groups.items() for well in wells],
                        columns=['Well', 'Moi'])


# Merge the group information into the target DataFrame
gm_dfb = pd.merge(merged_5trim, group_df, on='Well', how='left')
gm_df = pd.merge(gm_dfb, moi_df, on='Well', how='left')

########## Boxplot################
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

# Example DataFrame (replace this with your actual data)
# Assuming gm_df is already defined and contains the necessary columns
# gm_df = pd.read_csv('your_data.csv')  # Load your data here
#from patsy import Q
def create_plot(ax, data, y_col, title, ylabel):
    # Perform two-way ANOVA with escaped column names using backticks
    formula = f"Q('{y_col}') ~ C(Group) * C(Moi)"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA table
    print(f"{title} - Two-way ANOVA results:")
    print(anova_table)

    # Initialize holm_sidak_results as None
    holm_sidak_results = None

    # Check for significant interaction effects
    interaction_p = anova_table.loc['F(Group):C(Moi)', 'PR(>F)']
    if interaction_p < 0.05:
        print(f"Significant interaction effect between Group and Moi (p = {interaction_p})")
        # Perform pairwise t-tests with Holm-Sidak correction for interaction
        holm_sidak_results = sp.posthoc_ttest(data, val_col=y_col, group_col='Group', p_adjust='holm-sidak')
        print(f"{title} - Holm-Sidak post hoc results:")
        print(holm_sidak_results)

    # Create the boxplot with 'Moi' as the hue
    sns.boxplot(data=data, x='Group', y=y_col, palette='husl', hue='Moi', ax=ax)

    # Overlay swarmplot to show individual data points in black
    sns.swarmplot(data=data, x='Group', y=y_col, color='Black', hue='Moi', legend=False, ax=ax, size=4, dodge=True)

    # Function to convert p-values to asterisks
    def p_value_to_asterisks(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'  # Not significant

    # Extract significant pairs and p-values from Holm-Sidak results (if they exist)
    significant_pairs = []
    if holm_sidak_results is not None:
        groups = data['Group'].unique()
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                if i < j:  # Avoid duplicate comparisons
                    p_adj = holm_sidak_results.loc[group1, group2]
                    if p_adj < 0.05:  # Only include significant pairs
                        significant_pairs.append(((group1, group2), p_adj))

        # Get the y-coordinates of the top of each box
        box_tops = [max(data[data['Group'] == group][y_col]) for group in groups]

        # Adjust the y-axis scale to make more room for annotations
        y_max = max(box_tops)  # Maximum y-value of the boxes
        ax.set_ylim(top=y_max * 1.3)  # Increase the upper limit by 30%

        # Manually annotate the plot with significant pairs
        spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
        for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
            x1 = groups.tolist().index(group1)  # Get x-position of group1
            x2 = groups.tolist().index(group2)  # Get x-position of group2
            y1 = box_tops[x1]  # Top of the box for group1
            y2 = box_tops[x2]  # Top of the box for group2
            y_max_box = max(y1, y2)  # Use the higher of the two box tops
            y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

            # Convert p-value to asterisks
            asterisks = p_value_to_asterisks(p_adj)

            # Draw the line and annotation
            ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
            ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Group')
    ax.set_ylabel(ylabel)
# Create a multipanel figure
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 2 rows, 2 columns
fig.suptitle('Multipanel Figure: Mitochondrial Function via Ratiometric Pericam', fontsize=16)

plots = [
    {
        'y_col': '485_525/415_518',
        'title': 'Mitochondrial Function 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/415_518)'
    },
    {
        'y_col': '415_518/555_586',
        'title': 'Mitochondrial Calcium 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (415_518/555_586)'
    },
    {
        'y_col': '485_525/555_586',
        'title': 'Mitochondrial pH 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (485_525/555_586)'
    },
    {
        'y_col': '555_586/(370_470+555_586)',
        'title': 'Mitochondrial pH 3min exposure to FCCP',
        'ylabel': 'Pericam Fluorescence Ratio (555_586/(370_470+555_586))'
    }
]

# Populate each subplot with a plot
for ax, plot in zip(axes.flat, plots):
    create_plot(ax, gm_df, plot['y_col'], plot['title'], plot['ylabel'])

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()





#############485/415###################







from scipy.stats import f_oneway
from scipy.stats import kruskal
import scikit_posthocs as sp

# Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
groups2 = [gm_df[gm_df['Group'] == group]['485_525/415_518'] for group in gm_df['Group'].unique()]
h_stat, p_value = kruskal(*groups2)
print(f"Kruskal-Wallis p-value: {p_value}")

# Perform Dunn's post hoc test
dunn_results = sp.posthoc_dunn(gm_df, val_col='485_525/415_518', group_col='Group', p_adjust='Bonferroni')
print("Dunn's test results:")
print(dunn_results)

# Create the boxplot
plt.figure(figsize=(10, 8))
ax = sns.boxplot(data=gm_df, x='Group', y='485_525/415_518', palette='husl')

# Function to convert p-values to asterisks
def p_value_to_asterisks(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'  # Not significant

# Extract significant pairs and p-values from Dunn's test results
significant_pairs = []
groups = gm_df['Group'].unique()
for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:  # Avoid duplicate comparisons
            p_adj = dunn_results.loc[group1, group2]
            if p_adj < 0.05:  # Only include significant pairs
                significant_pairs.append(((group1, group2), p_adj))

# Get the y-coordinates of the top of each box
box_tops = [max(gm_df[gm_df['Group'] == group]['485_525/415_518']) for group in groups]
# Adjust the y-axis scale to make more room for annotations
y_max = max(box_tops)  # Maximum y-value of the boxes
plt.ylim(top=y_max * 1.2)  # Increase the upper limit by 20%

# Manually annotate the plot with significant pairs
spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
    x1 = groups.tolist().index(group1)  # Get x-position of group1
    x2 = groups.tolist().index(group2)  # Get x-position of group2
    y1 = box_tops[x1]  # Top of the box for group1
    y2 = box_tops[x2]  # Top of the box for group2
    y_max_box = max(y1, y2)  # Use the higher of the two box tops
    y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

    # Convert p-value to asterisks
    asterisks = p_value_to_asterisks(p_adj)

    # Draw the line and annotation
    ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
    ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

# Add title and labels
plt.title('Mitochondrial Function 3min exposure to FCCP')
plt.xlabel('Group')
plt.ylabel('Pericam Fluorescence Ratio (485_525/415_518)')

# Show the plot
plt.tight_layout()
plt.show()


##################415/555#################




from scipy.stats import f_oneway
from scipy.stats import kruskal
import scikit_posthocs as sp

# Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
groups2 = [gm_df[gm_df['Group'] == group]['415_518/555_586'] for group in gm_df['Group'].unique()]
h_stat, p_value = kruskal(*groups2)
print(f"Kruskal-Wallis p-value: {p_value}")

# Perform Dunn's post hoc test
dunn_results = sp.posthoc_dunn(gm_df, val_col='415_518/555_586', group_col='Group', p_adjust='Bonferroni')
print("Dunn's test results:")
print(dunn_results)

# Create the boxplot
plt.figure(figsize=(10, 8))
ax = sns.boxplot(data=gm_df, x='Group', y='415_518/555_586', palette='husl')

# Function to convert p-values to asterisks
def p_value_to_asterisks(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'  # Not significant

# Extract significant pairs and p-values from Dunn's test results
significant_pairs = []
groups = gm_df['Group'].unique()
for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:  # Avoid duplicate comparisons
            p_adj = dunn_results.loc[group1, group2]
            if p_adj < 0.05:  # Only include significant pairs
                significant_pairs.append(((group1, group2), p_adj))

# Get the y-coordinates of the top of each box
box_tops = [max(gm_df[gm_df['Group'] == group]['415_518/555_586']) for group in groups]
# Adjust the y-axis scale to make more room for annotations
y_max = max(box_tops)  # Maximum y-value of the boxes
plt.ylim(top=y_max * 1.2)  # Increase the upper limit by 20%

# Manually annotate the plot with significant pairs
spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
    x1 = groups.tolist().index(group1)  # Get x-position of group1
    x2 = groups.tolist().index(group2)  # Get x-position of group2
    y1 = box_tops[x1]  # Top of the box for group1
    y2 = box_tops[x2]  # Top of the box for group2
    y_max_box = max(y1, y2)  # Use the higher of the two box tops
    y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

    # Convert p-value to asterisks
    asterisks = p_value_to_asterisks(p_adj)

    # Draw the line and annotation
    ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
    ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

# Add title and labels
plt.title('Mitochondrial Calcium 3min exposure to FCCP')
plt.xlabel('Group')
plt.ylabel('Pericam Fluorescence Ratio (415_518/555_586)')

# Show the plot
plt.tight_layout()
plt.show()





##################485/555#################




from scipy.stats import f_oneway
from scipy.stats import kruskal
import scikit_posthocs as sp

# Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
groups2 = [gm_df[gm_df['Group'] == group]['485_525/555_586'] for group in gm_df['Group'].unique()]
h_stat, p_value = kruskal(*groups2)
print(f"Kruskal-Wallis p-value: {p_value}")

# Perform Dunn's post hoc test
dunn_results = sp.posthoc_dunn(gm_df, val_col='485_525/555_586', group_col='Group', p_adjust='Bonferroni')
print("Dunn's test results:")
print(dunn_results)

# Create the boxplot
plt.figure(figsize=(10, 8))
ax = sns.boxplot(data=gm_df, x='Group', y='485_525/555_586', palette='husl')

# Function to convert p-values to asterisks
def p_value_to_asterisks(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'  # Not significant

# Extract significant pairs and p-values from Dunn's test results
significant_pairs = []
groups = gm_df['Group'].unique()
for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:  # Avoid duplicate comparisons
            p_adj = dunn_results.loc[group1, group2]
            if p_adj < 0.05:  # Only include significant pairs
                significant_pairs.append(((group1, group2), p_adj))

# Get the y-coordinates of the top of each box
box_tops = [max(gm_df[gm_df['Group'] == group]['485_525/555_586']) for group in groups]
# Adjust the y-axis scale to make more room for annotations
y_max = max(box_tops)  # Maximum y-value of the boxes
plt.ylim(top=y_max * 1.2)  # Increase the upper limit by 20%

# Manually annotate the plot with significant pairs
spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
    x1 = groups.tolist().index(group1)  # Get x-position of group1
    x2 = groups.tolist().index(group2)  # Get x-position of group2
    y1 = box_tops[x1]  # Top of the box for group1
    y2 = box_tops[x2]  # Top of the box for group2
    y_max_box = max(y1, y2)  # Use the higher of the two box tops
    y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

    # Convert p-value to asterisks
    asterisks = p_value_to_asterisks(p_adj)

    # Draw the line and annotation
    ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
    ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

# Add title and labels
plt.title('Mitochondrial pH 3min exposure to FCCP')
plt.xlabel('Group')
plt.ylabel('Pericam Fluorescence Ratio (485_525/555_586)')

# Show the plot
plt.tight_layout()
plt.show()



##################485/555#################




from scipy.stats import f_oneway
from scipy.stats import kruskal
import scikit_posthocs as sp

# Perform Kruskal-Wallis test (non-parametric alternative to ANOVA)
groups2 = [gm_df[gm_df['Group'] == group]['555_586/(370_470+555_586)'] for group in gm_df['Group'].unique()]
h_stat, p_value = kruskal(*groups2)
print(f"Kruskal-Wallis p-value: {p_value}")

# Perform Dunn's post hoc test
dunn_results = sp.posthoc_dunn(gm_df, val_col='555_586/(370_470+555_586)', group_col='Group', p_adjust='Bonferroni')
print("Dunn's test results:")
print(dunn_results)

# Create the boxplot
plt.figure(figsize=(10, 8))
ax = sns.boxplot(data=gm_df, x='Group', y='555_586/(370_470+555_586)', palette='husl')

# Function to convert p-values to asterisks
def p_value_to_asterisks(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'  # Not significant

# Extract significant pairs and p-values from Dunn's test results
significant_pairs = []
groups = gm_df['Group'].unique()
for i, group1 in enumerate(groups):
    for j, group2 in enumerate(groups):
        if i < j:  # Avoid duplicate comparisons
            p_adj = dunn_results.loc[group1, group2]
            if p_adj < 0.05:  # Only include significant pairs
                significant_pairs.append(((group1, group2), p_adj))

# Get the y-coordinates of the top of each box
box_tops = [max(gm_df[gm_df['Group'] == group]['555_586/(370_470+555_586)']) for group in groups]
# Adjust the y-axis scale to make more room for annotations
y_max = max(box_tops)  # Maximum y-value of the boxes
plt.ylim(top=y_max * 1.2)  # Increase the upper limit by 20%

# Manually annotate the plot with significant pairs
spacing_factor = 0.04  # Vertical spacing between lines (adjust as needed)
for idx, ((group1, group2), p_adj) in enumerate(significant_pairs):
    x1 = groups.tolist().index(group1)  # Get x-position of group1
    x2 = groups.tolist().index(group2)  # Get x-position of group2
    y1 = box_tops[x1]  # Top of the box for group1
    y2 = box_tops[x2]  # Top of the box for group2
    y_max_box = max(y1, y2)  # Use the higher of the two box tops
    y = y_max_box + (idx + 1) * spacing_factor * y_max  # Adjust y-position for annotation

    # Convert p-value to asterisks
    asterisks = p_value_to_asterisks(p_adj)

    # Draw the line and annotation
    ax.plot([x1, x1, x2, x2], [y, y + 0.02 * y_max, y + 0.02 * y_max, y], lw=1.5, color='Black')
    ax.text((x1 + x2) * 0.5, y + 0.03 * y_max, asterisks, ha='center', va='bottom', color='Black', fontsize=12)

# Add title and labels
plt.title('Mitochondrial pH 3min exposure to FCCP')
plt.xlabel('Group')
plt.ylabel('Pericam Fluorescence Ratio (555_586/(370_470+555_586))')

# Show the plot
plt.tight_layout()
plt.show()














########## Boxplot################
import seaborn as sns
import matplotlib.pyplot as plt

husl_palette = sns.husl_palette(n_colors=4, s=1)

sns.boxplot(data=gm_df, x='Group', y='485_525/415_518', hue=None,
                order=None, hue_order=None, orient=None,
                color=None, palette=husl_palette, saturation=0.75,
                fill=True, dodge='auto', width=0.8, gap=0,
                whis=1.5, linecolor='auto', linewidth=None,
                fliersize=None, hue_norm=None, native_scale=False,
                log_scale=None, formatter=None, legend='auto',
                ax=None)
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