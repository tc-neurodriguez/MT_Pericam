import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

pd.set_option('display.max_columns', None) # Displays all columns
def load_spectra_from_csv(folder_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load fluorescence spectra from CSV files.
    Expected format: wavelength, [Fluorophore] ex, [Fluorophore] em
    """
    spectra = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            fluor_name = os.path.splitext(filename)[0]
            try:
                df = pd.read_csv(os.path.join(folder_path, filename)).fillna(0)

                # Validate columns
                ex_col = f"{fluor_name} ex"
                em_col = f"{fluor_name} em"

                if not all(col in df.columns for col in ['wavelength', ex_col, em_col]):
                    print(f"Skipping {filename}: missing columns")
                    continue

                # Store spectra with NaN protection
                spectra[fluor_name] = {
                    'ex': (df['wavelength'].values, np.nan_to_num(df[ex_col].values)),
                    'em': (df['wavelength'].values, np.nan_to_num(df[em_col].values))
                }

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    return spectra


def calculate_overlap(
        target: Dict[str, Tuple[np.ndarray, np.ndarray]],
        reference: Dict[str, Tuple[np.ndarray, np.ndarray]],
        filters: Dict[str, float],
        wavelength_range: np.ndarray = np.arange(350, 700, 1)
) -> float:
    """
    Calculate effective spectral overlap percentage accounting for:
    1. Excitation crosstalk (must be excited first)
    2. Emission bleed-through (only if excited)
    Returns the effective overlap percentage.
    """
    # Interpolate all spectra to common wavelengths
    target_ex = np.interp(wavelength_range, *target['ex'])
    target_em = np.interp(wavelength_range, *target['em'])
    ref_ex = np.interp(wavelength_range, *reference['ex'])
    ref_em = np.interp(wavelength_range, *reference['em'])

    # Create filter masks (18nm bandwidth)
    ex_filter = ((wavelength_range >= filters['ex_center'] - 9) &
                 (wavelength_range <= filters['ex_center'] + 9))
    em_filter = ((wavelength_range >= filters['em_center'] - 9) &
                 (wavelength_range <= filters['em_center'] + 9))

    # Calculate reference signal in its own windows
    ref_ex_signal = np.trapz(ref_ex * ex_filter, wavelength_range)
    ref_em_signal = np.trapz(ref_em * em_filter, wavelength_range)

    # Calculate raw overlaps
    excitation_crosstalk = np.trapz(target_ex * ex_filter, wavelength_range)
    emission_bleedthrough = np.trapz(target_em * em_filter, wavelength_range)

    # Normalize to reference signals
    excitation_pct = (excitation_crosstalk / ref_ex_signal * 100) if ref_ex_signal > 0 else 0
    emission_pct = (emission_bleedthrough / ref_em_signal * 100) if ref_em_signal > 0 else 0

    # Calculate effective emission bleed-through (only applies if excited)
    effective_emission = (emission_pct * excitation_pct) / 100

    print(
        f"Breakdown - Excitation: {excitation_pct:.1f}%, Emission: {emission_pct:.1f}% → Effective: {effective_emission:.1f}%")

    return effective_emission


def analyze_spectral_overlaps(
        spectra: Dict[str, Dict[str, np.ndarray]],
        reference_sensors: Dict[str, Dict[str, np.ndarray]],
        sensor_configs: List[Dict[str, Dict[str, float]]],
        sensor_names: List[str],
        exclude: List[str] = []
) -> pd.DataFrame:
    """Analyze all fluorophores with effective overlap calculation"""
    results = []

    for name, spec in spectra.items():
        if name in exclude:
            continue

        overlaps = {}
        max_overlap = 0

        for config, s_name in zip(sensor_configs, sensor_names):
            ref_name = config['reference']
            if ref_name not in reference_sensors:
                overlap = 0
            else:
                overlap = calculate_overlap(
                    target=spec,
                    reference=reference_sensors[ref_name],
                    filters=config['filters']
                )

            overlaps[f'{s_name}_Overlap_%'] = overlap
            max_overlap = max(max_overlap, overlap)

        results.append({
            'Fluorophore': name,
            'Max_Effective_Overlap_%': max_overlap,
            **overlaps
        })

    return pd.DataFrame(results).sort_values('Max_Effective_Overlap_%', ascending=False)


def plot_spectra_comparison(
        spectra: Dict[str, Dict[str, np.ndarray]],
        sensor_configs: List[Dict[str, Dict[str, float]]],
        sensor_names: List[str],
        fluorophores: List[str],
        wavelength_range: Tuple[int, int] = (350, 700)
):
    """
    Visualize spectra with uniquely colored filter windows and spectral lines.

    Parameters:
    - spectra: Dictionary of fluorophore spectra
    - sensor_configs: List of filter configurations
    - sensor_names: List of sensor names
    - fluorophores: List of fluorophores to plot
    - wavelength_range: Tuple of (min, max) wavelengths to display
    """
    plt.figure(figsize=(16, 8))

    # Create colormaps - one for filters, one for fluorophores
    filter_colors = plt.cm.tab20(np.linspace(0, 1, len(sensor_configs) * 2))  # 2 colors per sensor
    fluor_colors = plt.cm.Dark2(np.linspace(0, 1, len(fluorophores) * 2))  # 2 colors per fluorophore

    # Set wavelength range
    plt.xlim(*wavelength_range)
    plt.ylim(0, 1.2)  # Slightly above 1 for legend space

    # Plot sensor filter ranges with unique colors
    for i, (config, name) in enumerate(zip(sensor_configs, sensor_names)):
        f = config['filters']

        # Plot excitation filter (solid fill)
        plt.axvspan(
            max(f['ex_center'] - 9, wavelength_range[0]),
            min(f['ex_center'] + 9, wavelength_range[1]),
            alpha=0.15, color=filter_colors[i * 2],
            label=f'{name} Ex ({f["ex_center"]}±9 nm)'
        )

        # Plot emission filter (hatched fill)
        plt.axvspan(
            max(f['em_center'] - 9, wavelength_range[0]),
            min(f['em_center'] + 9, wavelength_range[1]),
            alpha=0.15, color=filter_colors[i * 2 + 1], hatch='//',
            label=f'{name} Em ({f["em_center"]}±9 nm)'
        )

    # Plot fluorophore spectra with unique colors
    for j, name in enumerate(fluorophores):
        if name in spectra:
            wl, ex = spectra[name]['ex']
            _, em = spectra[name]['em']

            # Apply wavelength range mask
            mask = (wl >= wavelength_range[0]) & (wl <= wavelength_range[1])
            wl = wl[mask]
            ex = ex[mask]
            em = em[mask]

            # Plot with unique colors and styles
            plt.plot(wl, ex / np.max(ex),
                     color=fluor_colors[j * 2], lw=2.5, ls='-',
                     label=f'{name} Ex')
            plt.plot(wl, em / np.max(em),
                     color=fluor_colors[j * 2 + 1], lw=2.5, ls='--',
                     label=f'{name} Em')

    # Formatting
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.title('Spectral Overlap Analysis', fontsize=14, pad=20)

    # Create a comprehensive legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               framealpha=0.9,
               fontsize=10)

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
def plot_overlap_results(results: pd.DataFrame, top_n: int = 10):
    """
    Create a 2x2 grid of bar plots showing overlap percentages
    for the top N fluorophores across all sensor configurations.
    """
    # Get top N fluorophores
    top_fluorophores = results.head(top_n)

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Top {top_n} Fluorophore Spectral Overlaps', fontsize=16, y=1.02)

    # Plot settings
    bar_width = 0.6
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors

    # 1. Max Effective Overlap
    axs[0, 0].barh(top_fluorophores['Fluorophore'],
                   top_fluorophores['Max_Effective_Overlap_%'],
                   color=colors[0], height=bar_width)
    axs[0, 0].set_title('Maximum Effective Overlap')
    axs[0, 0].set_xlabel('Overlap Percentage (%)')
    axs[0, 0].set_xlim(0, 100)

    # 2. Pericam Ca Overlap
    axs[0, 1].barh(top_fluorophores['Fluorophore'],
                   top_fluorophores['Pericam_Ca_Overlap_%'],
                   color=colors[1], height=bar_width)
    axs[0, 1].set_title('Pericam (Calcium) Overlap')
    axs[0, 1].set_xlabel('Overlap Percentage (%)')
    axs[0, 1].set_xlim(0, 100)

    # 3. Pericam pH Overlap
    axs[1, 0].barh(top_fluorophores['Fluorophore'],
                   top_fluorophores['Pericam_pH_Overlap_%'],
                   color=colors[2], height=bar_width)
    axs[1, 0].set_title('Pericam (pH) Overlap')
    axs[1, 0].set_xlabel('Overlap Percentage (%)')
    axs[1, 0].set_xlim(0, 100)

    # 4. TagRFP-T Overlap
    axs[1, 1].barh(top_fluorophores['Fluorophore'],
                   top_fluorophores['mScarlet_Overlap_%'],
                   color=colors[3], height=bar_width)
    axs[1, 1].set_title('mScarlet Overlap')
    axs[1, 1].set_xlabel('Overlap Percentage (%)')
    axs[1, 1].set_xlim(0, 100)

    # Adjust layout and add grid
    for ax in axs.flat:
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()  # Show highest at top

    plt.tight_layout()
    plt.savefig('fluorophore_overlaps.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Configuration
    DATA_FOLDER = r'C:\Users\t_rod\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\Spectral Comparisons\spectra_csv'

    # 1. Load all spectra
    print("Loading spectra...")
    all_spectra = load_spectra_from_csv(DATA_FOLDER)

    # 2. Define reference sensors and their filter configurations
    reference_sensors = {
        'ratiometric_pericam': all_spectra['ratiometric_pericam'],
        'mScarlet': all_spectra['mScarlet']
    }

    sensor_configs = [
        {'filters': {'ex_center': 415, 'em_center': 518}, 'reference': 'ratiometric_pericam'},  # Pericam Ca-free
        {'filters': {'ex_center': 485, 'em_center': 525}, 'reference': 'ratiometric_pericam'},  # Pericam Ca-bound
        {'filters': {'ex_center': 570, 'em_center': 593}, 'reference': 'mScarlet'}  # mScarlet
    ]

    sensor_names = ['Pericam_Ca', 'Pericam_pH', 'mScarlet']

    # 3. Analyze overlaps
    print("\nAnalyzing spectral overlaps...")
    results = analyze_spectral_overlaps(
        all_spectra,
        reference_sensors,
        sensor_configs,
        sensor_names,
        exclude=['ratiometric_pericam', 'mScarlet']
    )
    # Generate the plots
    plot_overlap_results(results, top_n=6)  # Use 6 for your example data

    # 4. Save and display results
    results.to_csv('fluorophore_overlaps.csv', index=False)
    print("\nTop overlapping fluorophores:")
    print(results.head(10))

    # 5. Visualize top 3 fluorophores
    top_fluorophores = results.head(3)['Fluorophore'].tolist()
    print(f"\nVisualizing top fluorophores: {top_fluorophores}")
    plot_spectra_comparison(
        all_spectra,
        sensor_configs,
        sensor_names,
        top_fluorophores
    )





if __name__ == "__main__":
    main()