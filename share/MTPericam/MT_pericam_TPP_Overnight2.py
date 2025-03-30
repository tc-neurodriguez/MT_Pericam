import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
file_path = r'C:\Users\t_rod\Box\SY5Y transduction Images\BioSpa_Data\MT-PericamTestData3_TPP_ON_ForPython.xlsx'
# Define groups
groups = {
    '16_TPP': ['B3', 'B4'],
    '16_Con': ['B5', 'B6'],
    '32_TPP': ['B7', 'B8'],
    '32_Con': ['B9', 'B10'],
}


# 1. Data Loading with Debugging
def load_and_process(sheet_name, wavelength):
    print(f"\n⏳ Loading {wavelength} from {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Clean columns
    df.columns = df.columns.str.strip()
    print(f"📊 Initial columns: {df.columns.tolist()}")

    # Convert and round time
    df['Time'] = df['Time'].apply(
        lambda x: round(x.hour + x.minute / 60 + x.second / 3600, 4))
    print(f"🕒 Time range: {df['Time'].min()} to {df['Time'].max()} hours")

    # Verify background
    if 'Background' not in df.columns:
        raise KeyError(f"🚨 Missing Background column in {wavelength}")
    print(
        f"🔦 Background stats: Mean={df['Background'].mean():.2f}, Range={df['Background'].min()}-{df['Background'].max()}")

    # Subtract background
    wells = [w for group in groups.values() for w in group]
    df[wells] = df[wells].sub(df['Background'], axis=0)
    print(f"🧮 Background subtracted. Sample values:")
    print(df[wells].head(2))

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
                label=group, marker='o', linestyle='-', markersize=8)
        valid_groups += 1

    if valid_groups == 0:
        print(f"🚨 No valid data for {ratio}")
        continue

    ax.set_title(ratio, fontsize=14)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Ratio', fontsize=12)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
print("\n✅ Processing complete!")