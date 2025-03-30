import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = r"C:\Users\t_rod\Box\SY5Y transduction Images\Experiment 3_9_25\MT-Pericam_3_9_25_Thapsigargin_python.xlsx"
wells = ['B7', 'B8', 'B9', 'B10','C7', 'C8', 'C9', 'C10','D3', 'D7', 'D8', 'D9', 'D10','E7', 'E8','E10']

groups = {
    '25mM KCl(0.01% DMSO)': ['B7', 'B8', 'B9', 'B10'],
    '0.01_uM_Hist': ['C7', 'C8', 'C9', 'C10'],
    '0.1_uM_Hist': ['D7', 'D8', 'D9', 'D10'],
    '1_uM_Hist': ['E7', 'E8', 'E10'],
}
background_wells = [f'G{i}' for i in range(7, 11)]
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

for ratio, parts in ratio_config.items():
    if isinstance(parts[0], tuple):
        # Handle complex ratio: denominator is sum
        num = merged[parts[1]]
        denom = merged[parts[0][0]] + merged[parts[0][1]]
    else:
        num = merged[parts[0]]
        denom = merged[parts[1]]

    merged[ratio] = num / denom
    merged[ratio].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Ratio statistics
    print(f"\n📐 {ratio} statistics:")
    print(f"✅ Valid values: {merged[ratio].notna().sum()}")
    print(f"❌ NaN values: {merged[ratio].isna().sum()}")
    print(f"📈 Mean ratio: {merged[ratio].mean():.2f} ± {merged[ratio].std():.2f}")

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