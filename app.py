import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_location_data(data_dir='./locations_data'):
    """Load product data for Kuwait and Jeju."""
    data_dir = Path(data_dir)
    location_data = {'Kuwait': {}, 'Jeju': {}}

    # Files to load for each location
    file_patterns = {
        'Kuwait': ['Kuwait_product_analysis_PMI_Products.csv',
                   'Kuwait_product_analysis_Top_90pct_Products.csv',
                   'Kuwait_product_analysis_Summary.csv'],
        'Jeju': ['jeju_product_analysis_PMI_Products.csv',
                 'jeju_product_analysis_Top_90pct_Products.csv',
                 'jeju_product_analysis_Summary.csv']
    }

    # Load files
    for location, patterns in file_patterns.items():
        for pattern in patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                key = pattern.replace(f"{location}_product_analysis_", "").replace(".csv", "").lower()
                location_data[location][key] = pd.read_csv(file_path)
            else:
                print(f"Warning: File not found - {file_path}")

    return location_data


def create_advanced_product_shelf(location_data, location, cmap_name='Blues'):
    """Create an advanced product shelf visualization."""
    # Get PMI product data
    products_df = None
    for data_type in location_data[location]:
        if 'pmi_products' in data_type.lower():
            products_df = location_data[location][data_type]
            break

    if products_df is None:
        print(f"No PMI product data found for {location}")
        return None

    # Get unique values for each attribute (with new mappings)
    tastes = sorted(products_df['Taste'].unique())  # X-axis
    flavors = sorted(products_df['Flavor'].unique())  # Y-axis
    thicknesses = sorted(products_df['Thickness'].unique())  # Box size

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Calculate product distribution by flavor-taste combinations
    cell_volumes = {}
    for flavor in flavors:
        flavor_data = products_df[products_df['Flavor'] == flavor]
        flavor_total = flavor_data['DF_Vol'].sum()

        for taste in tastes:
            cell_key = (taste, flavor)  # (X, Y)
            cell_df = flavor_data[flavor_data['Taste'] == taste]
            cell_volume = cell_df['DF_Vol'].sum()
            cell_volumes[cell_key] = cell_volume

    # Create gradient matrix with 3 intensity levels per row
    gradient_matrix = np.zeros((len(flavors), len(tastes)))

    for y_idx, flavor in enumerate(flavors):
        # Get all volumes for this flavor row
        row_volumes = [cell_volumes.get((taste, flavor), 0) for taste in tastes]

        if sum(row_volumes) > 0:
            # Create 3 discrete levels based on percentages within the row
            normalized_volumes = [vol / sum(row_volumes) if sum(row_volumes) > 0 else 0 for vol in row_volumes]

            # Map to 3 intensity levels (0.2, 0.5, 0.8)
            for x_idx, vol_pct in enumerate(normalized_volumes):
                if vol_pct == 0:
                    gradient_matrix[y_idx, x_idx] = 0
                elif vol_pct < 0.2:
                    gradient_matrix[y_idx, x_idx] = 0.2
                elif vol_pct < 0.5:
                    gradient_matrix[y_idx, x_idx] = 0.5
                else:
                    gradient_matrix[y_idx, x_idx] = 0.8

    # Plot background gradient
    im = ax.imshow(gradient_matrix, cmap=plt.colormaps['Greys'],
                   alpha=0.3, aspect='auto', origin='lower',
                   extent=(-0.5, len(tastes) - 0.5, -0.5, len(flavors) - 0.5))

    # Map thickness to visual properties
    size_map = {thickness: (i + 1) * 150 for i, thickness in enumerate(thicknesses)}

    # Group products by Taste-Flavor cell
    cell_products = {}
    for taste in tastes:
        for flavor in flavors:
            cell_key = (taste, flavor)
            cell_products[cell_key] = products_df[(products_df['Taste'] == taste) &
                                                  (products_df['Flavor'] == flavor)]

    # Plot products as boxes
    for cell_key, cell_df in cell_products.items():
        if len(cell_df) == 0:
            continue

        taste, flavor = cell_key
        x_pos = tastes.index(taste)
        y_pos = flavors.index(flavor)

        # Calculate total volume for this cell
        total_cell_volume = cell_df['DF_Vol'].sum()

        # Plot each product in this cell
        for idx, (_, product) in enumerate(cell_df.iterrows()):
            thickness = product['Thickness']
            volume = product['DF_Vol']

            # Size based on thickness
            size = size_map.get(thickness, 100)

            # Use standard color (from colormap)
            color = plt.colormaps[cmap_name](0.6)

            # Position within cell
            products_in_cell = len(cell_df)
            grid_size = int(np.ceil(np.sqrt(products_in_cell)))

            # Calculate grid position
            grid_x = idx % grid_size
            grid_y = idx // grid_size

            # Calculate offsets within cell
            x_offset = (grid_x - grid_size / 2 + 0.5) * 0.7 / grid_size
            y_offset = (grid_y - grid_size / 2 + 0.5) * 0.7 / grid_size

            # Scale size by volume ratio
            volume_ratio = volume / total_cell_volume if total_cell_volume > 0 else 0.5
            scaled_size = size * (0.5 + 0.5 * volume_ratio)

            # Create circle representing product
            circle = plt.Circle((x_pos + x_offset, y_pos + y_offset),
                                scaled_size / 1000,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1,
                                alpha=0.8)
            ax.add_patch(circle)

    # Set up axes
    ax.set_xticks(range(len(tastes)))
    ax.set_yticks(range(len(flavors)))
    ax.set_xticklabels(tastes)
    ax.set_yticklabels(flavors)
    ax.set_xlabel('Taste', fontsize=14)
    ax.set_ylabel('Flavor', fontsize=14)

    # Set grid
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title(f'{location} PMI Products Portfolio Shelf View', fontsize=16)

    # Create legends
    # Thickness legend
    thickness_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=plt.colormaps[cmap_name](0.6), markersize=(i + 1) * 6,
                                    label=f"{t}") for i, t in enumerate(thicknesses)]
    thickness_legend = ax.legend(handles=thickness_handles, title="Thickness",
                                 loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.add_artist(thickness_legend)

    # Background gradient legend
    gradient_handles = [
        plt.Rectangle((0, 0), 1, 1, color=plt.colormaps['Greys'](0.2), alpha=0.3, label='Low Market Share'),
        plt.Rectangle((0, 0), 1, 1, color=plt.colormaps['Greys'](0.5), alpha=0.3, label='Medium Market Share'),
        plt.Rectangle((0, 0), 1, 1, color=plt.colormaps['Greys'](0.8), alpha=0.3, label='High Market Share')
    ]
    gradient_legend = ax.legend(handles=gradient_handles, title="Market Share",
                                loc='upper left', bbox_to_anchor=(1.02, 0.7))
    ax.add_artist(gradient_legend)

    plt.tight_layout()
    return fig

def create_advanced_product_shelf_subplot(ax, location_data, location, cmap_name='Blues'):
    """Subplot version of the shelf visualization for comparisons"""
    # Get product data
    products_df = None
    for data_type in location_data[location]:
        if 'pmi_products' in data_type.lower():
            products_df = location_data[location][data_type]
            break

    if products_df is None:
        print(f"No PMI product data found for {location}")
        return None

    # Get unique values for each attribute (with new mappings)
    tastes = sorted(products_df['Taste'].unique())  # X-axis
    flavors = sorted(products_df['Flavor'].unique())  # Y-axis
    thicknesses = sorted(products_df['Thickness'].unique())  # Box size

    # Calculate product distribution by flavor-taste combinations
    cell_volumes = {}
    for flavor in flavors:
        flavor_data = products_df[products_df['Flavor'] == flavor]
        flavor_total = flavor_data['DF_Vol'].sum()

        for taste in tastes:
            cell_key = (taste, flavor)  # (X, Y)
            cell_df = flavor_data[flavor_data['Taste'] == taste]
            cell_volume = cell_df['DF_Vol'].sum()
            cell_volumes[cell_key] = cell_volume

    # Create gradient matrix with 3 intensity levels per row
    gradient_matrix = np.zeros((len(flavors), len(tastes)))

    for y_idx, flavor in enumerate(flavors):
        # Get all volumes for this flavor row
        row_volumes = [cell_volumes.get((taste, flavor), 0) for taste in tastes]

        if sum(row_volumes) > 0:
            # Create 3 discrete levels based on percentages within the row
            normalized_volumes = [vol / sum(row_volumes) if sum(row_volumes) > 0 else 0 for vol in row_volumes]

            # Map to 3 intensity levels (0.2, 0.5, 0.8)
            for x_idx, vol_pct in enumerate(normalized_volumes):
                if vol_pct == 0:
                    gradient_matrix[y_idx, x_idx] = 0
                elif vol_pct < 0.2:
                    gradient_matrix[y_idx, x_idx] = 0.2
                elif vol_pct < 0.5:
                    gradient_matrix[y_idx, x_idx] = 0.5
                else:
                    gradient_matrix[y_idx, x_idx] = 0.8

    # Plot background gradient
    im = ax.imshow(gradient_matrix, cmap=plt.colormaps['Greys'],
                   alpha=0.3, aspect='auto', origin='lower',
                   extent=(-0.5, len(tastes) - 0.5, -0.5, len(flavors) - 0.5))

    # Map thickness to visual properties
    size_map = {thickness: (i + 1) * 150 for i, thickness in enumerate(thicknesses)}

    # Group products by Taste-Flavor cell
    cell_products = {}
    for taste in tastes:
        for flavor in flavors:
            cell_key = (taste, flavor)
            cell_products[cell_key] = products_df[(products_df['Taste'] == taste) &
                                                  (products_df['Flavor'] == flavor)]

    # Plot products as boxes
    for cell_key, cell_df in cell_products.items():
        if len(cell_df) == 0:
            continue

        taste, flavor = cell_key
        x_pos = tastes.index(taste)
        y_pos = flavors.index(flavor)

        # Calculate total volume for this cell
        total_cell_volume = cell_df['DF_Vol'].sum()

        # Plot each product in this cell
        for idx, (_, product) in enumerate(cell_df.iterrows()):
            thickness = product['Thickness']
            volume = product['DF_Vol']

            # Size based on thickness
            size = size_map.get(thickness, 100)

            # Use standard color
            color = plt.colormaps[cmap_name](0.6)

            # Position within cell
            products_in_cell = len(cell_df)
            grid_size = int(np.ceil(np.sqrt(products_in_cell)))

            grid_x = idx % grid_size
            grid_y = idx // grid_size

            x_offset = (grid_x - grid_size / 2 + 0.5) * 0.7 / grid_size
            y_offset = (grid_y - grid_size / 2 + 0.5) * 0.7 / grid_size

            volume_ratio = volume / total_cell_volume if total_cell_volume > 0 else 0.5
            scaled_size = size * (0.5 + 0.5 * volume_ratio)

            circle = plt.Circle((x_pos + x_offset, y_pos + y_offset),
                                scaled_size / 1000,
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1,
                                alpha=0.8)
            ax.add_patch(circle)

    # Set up axes
    ax.set_xticks(range(len(tastes)))
    ax.set_yticks(range(len(flavors)))
    ax.set_xticklabels(tastes)
    ax.set_yticklabels(flavors)
    ax.set_xlabel('Taste', fontsize=12)
    ax.set_ylabel('Flavor', fontsize=12)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    return ax

def create_market_comparison(location_data, location, ax=None):
    """Create comparison of PMI vs total market"""
    pmi_df = None
    market_df = None

    for data_type in location_data[location]:
        if 'pmi_products' in data_type.lower():
            pmi_df = location_data[location][data_type]
        if 'top_90pct_products' in data_type.lower():
            market_df = location_data[location][data_type]

    if pmi_df is None or market_df is None:
        print(f"Missing data for {location}")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
        standalone = True
    else:
        standalone = False

    # Analyze by flavor and length - Fix type mismatch by converting to strings
    pmi_flavors = [str(f) for f in pmi_df['Flavor'].unique()]
    market_flavors = [str(f) for f in market_df['Flavor'].unique()]
    flavors = sorted(set(pmi_flavors + market_flavors))

    pmi_lengths = [str(l) for l in pmi_df['Length'].unique()]
    market_lengths = [str(l) for l in market_df['Length'].unique()]
    lengths = sorted(set(pmi_lengths + market_lengths))

    # Create comparison grid
    comparison_data = []
    for flavor in flavors:
        for length in lengths:
            pmi_vol = pmi_df[(pmi_df['Flavor'].astype(str) == flavor) &
                             (pmi_df['Length'].astype(str) == length)]['DF_Vol'].sum()
            market_vol = market_df[(market_df['Flavor'].astype(str) == flavor) &
                                   (market_df['Length'].astype(str) == length)]['DF_Vol'].sum()

            # Calculate gap
            if market_vol > 0:
                pmi_share = (pmi_vol / market_vol) * 100
            else:
                pmi_share = 0 if pmi_vol == 0 else 100

            comparison_data.append({
                'Flavor': flavor,
                'Length': length,
                'PMI_Volume': pmi_vol,
                'Market_Volume': market_vol,
                'PMI_Share': pmi_share
            })

    # Create a DataFrame for easier analysis
    comparison_df = pd.DataFrame(comparison_data)

    # Create matrix for visualization
    matrix_values = np.zeros((len(flavors), len(lengths)))
    for i, flavor in enumerate(flavors):
        for j, length in enumerate(lengths):
            row = comparison_df[(comparison_df['Flavor'] == flavor) &
                                (comparison_df['Length'] == length)]
            if not row.empty:
                matrix_values[i, j] = row['PMI_Share'].values[0]

    # Plot heatmap
    im = ax.imshow(matrix_values, cmap=plt.colormaps['RdYlGn'],
                   vmin=0, vmax=100, aspect='auto', origin='lower',
                   extent=(-0.5, len(lengths) - 0.5, -0.5, len(flavors) - 0.5))

    # Set up axes
    ax.set_xticks(range(len(lengths)))
    ax.set_yticks(range(len(flavors)))
    ax.set_xticklabels(lengths)
    ax.set_yticklabels(flavors)
    ax.set_xlabel('Length', fontsize=12)
    ax.set_ylabel('Flavor', fontsize=12)
    ax.set_title(f'{location} PMI Market Coverage', fontsize=14)
    ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PMI Market Share %')

    if standalone:
        plt.tight_layout()
        return fig
    else:
        return ax

def main():
    """Main function to load data and create visualizations"""
    # Set up directories
    data_dir = './locations_data'
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load location data
    location_data = load_location_data(data_dir)

    # 1. Create individual shelf visualizations for PMI products
    kuwait_shelf = create_advanced_product_shelf(location_data, 'Kuwait', cmap_name='Blues')
    if kuwait_shelf:
        kuwait_shelf.savefig(os.path.join(output_dir, "kuwait_pmi_shelf_view.png"), dpi=300, bbox_inches='tight')
        print("Kuwait PMI shelf visualization created")

    jeju_shelf = create_advanced_product_shelf(location_data, 'Jeju', cmap_name='Oranges')
    if jeju_shelf:
        jeju_shelf.savefig(os.path.join(output_dir, "jeju_pmi_shelf_view.png"), dpi=300, bbox_inches='tight')
        print("Jeju PMI shelf visualization created")

    # 2. Create comparison view of both locations
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for i, (loc, cmap) in enumerate([('Kuwait', 'Blues'), ('Jeju', 'Oranges')]):
        create_advanced_product_shelf_subplot(axes[i], location_data, loc, cmap_name=cmap)
        axes[i].set_title(f'{loc} PMI Product Portfolio', fontsize=16)

    plt.suptitle('PMI Portfolio Comparison: Kuwait vs Jeju', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    fig.savefig(os.path.join(output_dir, "pmi_shelf_comparison.png"), dpi=300, bbox_inches='tight')
    print("PMI shelf comparison visualization created")

    # 3. Create market coverage comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    create_market_comparison(location_data, 'Kuwait', axes[0])
    create_market_comparison(location_data, 'Jeju', axes[1])

    plt.suptitle('Market Coverage Analysis: Kuwait vs Jeju', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    fig.savefig(os.path.join(output_dir, "pmi_market_coverage.png"), dpi=300, bbox_inches='tight')
    print("PMI market coverage visualization created")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()