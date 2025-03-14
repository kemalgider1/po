import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_location_data(data_dir='./locations_data'):
    """
    Load product data for Kuwait and Jeju.

    Args:
        data_dir (str): Path to the directory containing location data files

    Returns:
        dict: Dictionary containing location-specific data
    """
    data_dir = Path(data_dir)
    location_data = {'Kuwait': {}, 'Jeju': {}}

    # Files to load for each location
    file_patterns = {
        'Kuwait': ['Kuwait_product_analysis_PMI_Products.csv',
                   'Kuwait_product_analysis_*_Distribution.csv',
                   'Kuwait_product_analysis_Top_90pct_Products.csv'],
        'Jeju': ['jeju_product_analysis_PMI_Products.csv',
                 'jeju_product_analysis_*_Distribution.csv',
                 'jeju_product_analysis_Top_90pct_Products.csv']
    }

    # Load files
    for location, patterns in file_patterns.items():
        for pattern in patterns:
            for file_path in data_dir.glob(pattern):
                file_name = file_path.name
                data_type = file_name.replace(f"{location.lower()}_product_analysis_", "").replace(".csv", "")
                try:
                    df = pd.read_csv(file_path)
                    location_data[location][data_type] = df
                    print(f"Loaded {location} {data_type}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return location_data

def create_advanced_product_shelf(location_data, location, cmap_name='viridis'):
    """
    Create an advanced product shelf visualization with multiple dimensions:
    - X-axis: Length
    - Y-axis: Flavor
    - Size: Thickness
    - Color shade: Taste
    - Background gradient: Market share per row
    - Position: Volume within category

    Args:
        location_data (dict): Dictionary containing location-specific data
        location (str): Location to visualize ('Kuwait' or 'Jeju')
        cmap_name (str): Matplotlib colormap name

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Get product data
    products_df = None
    for data_type in location_data[location]:
        if 'pmi_products' in data_type.lower():
            products_df = location_data[location][data_type]
            break

    if products_df is None:
        print(f"No product data found for {location}")
        return None

    # Get unique values for each attribute
    lengths = sorted(products_df['Length'].unique())
    flavors = sorted(products_df['Flavor'].unique())
    thicknesses = sorted(products_df['Thickness'].unique())
    tastes = sorted(products_df['Taste'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Calculate market share by flavor (for row gradient)
    flavor_volumes = {}
    for flavor in flavors:
        flavor_volumes[flavor] = products_df[products_df['Flavor'] == flavor]['DF_Vol'].sum()

    # Create background gradient for each row
    max_volume = max(flavor_volumes.values()) if flavor_volumes else 1

    # Create gradient matrix
    gradient_matrix = np.zeros((len(flavors), len(lengths)))
    for y_idx, flavor in enumerate(flavors):
        relative_volume = flavor_volumes[flavor] / max_volume
        gradient_matrix[y_idx, :] = np.linspace(0.1, relative_volume, len(lengths))

    # Plot background gradient
    im = ax.imshow(gradient_matrix, cmap=plt.cm.get_cmap('Greys'),
                  alpha=0.3, aspect='auto', origin='lower',
                  extent=(-0.5, len(lengths)-0.5, -0.5, len(flavors)-0.5))

    # Map thickness and taste to visual properties
    size_map = {thickness: (i+1)*150 for i, thickness in enumerate(thicknesses)}
    color_map = {taste: plt.cm.get_cmap(cmap_name)(i/len(tastes)) for i, taste in enumerate(tastes)}

    # Group products by Length-Flavor cell
    cell_products = {}
    for length in lengths:
        for flavor in flavors:
            cell_key = (length, flavor)
            cell_products[cell_key] = products_df[(products_df['Length'] == length) &
                                                 (products_df['Flavor'] == flavor)]

    # Plot products as boxes
    for cell_key, cell_df in cell_products.items():
        if len(cell_df) == 0:
            continue

        length, flavor = cell_key
        x_pos = lengths.index(length)
        y_pos = flavors.index(flavor)

        # Calculate total volume for this cell
        total_cell_volume = cell_df['DF_Vol'].sum()

        # Plot each product in this cell
        for idx, (_, product) in enumerate(cell_df.iterrows()):
            thickness = product['Thickness']
            taste = product['Taste']
            volume = product['DF_Vol']

            # Size based on thickness
            size = size_map.get(thickness, 100)

            # Color based on taste
            color = color_map.get(taste, 'blue')

            # Position within cell based on volume ratio
            # (arrange products in a grid within the cell)
            products_in_cell = len(cell_df)
            grid_size = int(np.ceil(np.sqrt(products_in_cell)))

            # Calculate grid position
            grid_x = idx % grid_size
            grid_y = idx // grid_size

            # Calculate offsets within cell
            x_offset = (grid_x - grid_size/2 + 0.5) * 0.7 / grid_size
            y_offset = (grid_y - grid_size/2 + 0.5) * 0.7 / grid_size

            # Scale size by volume ratio
            volume_ratio = volume / total_cell_volume if total_cell_volume > 0 else 0.5
            scaled_size = size * (0.5 + 0.5 * volume_ratio)

            # Create circle representing product
            circle = plt.Circle((x_pos + x_offset, y_pos + y_offset),
                               scaled_size/1000,  # Scale down size
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1,
                               alpha=0.8)
            ax.add_patch(circle)

            # Add product label for significant products
            if volume_ratio > 0.5 and scaled_size > 100:
                product_name = product['CR_BrandId'] if 'CR_BrandId' in product else f"Product {idx}"
                ax.text(x_pos + x_offset, y_pos + y_offset, product_name,
                       ha='center', va='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))

    # Set up axes
    ax.set_xticks(range(len(lengths)))
    ax.set_yticks(range(len(flavors)))
    ax.set_xticklabels(lengths)
    ax.set_yticklabels(flavors)
    ax.set_xlabel('Length', fontsize=14)
    ax.set_ylabel('Flavor', fontsize=14)

    # Set grid
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add title
    ax.set_title(f'{location} Product Portfolio Shelf View', fontsize=16)

    # Create legends
    # Thickness legend
    thickness_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='gray', markersize=(i+1)*6,
                                  label=f"{t}") for i, t in enumerate(thicknesses)]
    thickness_legend = ax.legend(handles=thickness_handles, title="Thickness",
                              loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.add_artist(thickness_legend)

    # Taste legend
    taste_handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color_map[t], markersize=10,
                              label=f"{t}") for t in tastes]
    taste_legend = ax.legend(handles=taste_handles, title="Taste",
                          loc='upper left', bbox_to_anchor=(1.02, 0.7))
    ax.add_artist(taste_legend)

    # Background gradient legend
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Market Share Intensity')

    plt.tight_layout()

    return fig

def create_advanced_product_shelf_subplot(ax, location_data, location, cmap_name='viridis'):
    """Subplot version of the shelf visualization for comparisons"""
    # Get product data
    products_df = None
    for data_type in location_data[location]:
        if 'pmi_products' in data_type.lower():
            products_df = location_data[location][data_type]
            break

    if products_df is None:
        print(f"No product data found for {location}")
        return None

    # Get unique values for each attribute
    lengths = sorted(products_df['Length'].unique())
    flavors = sorted(products_df['Flavor'].unique())
    thicknesses = sorted(products_df['Thickness'].unique())
    tastes = sorted(products_df['Taste'].unique())

    # Calculate market share by flavor (for row gradient)
    flavor_volumes = {}
    for flavor in flavors:
        flavor_volumes[flavor] = products_df[products_df['Flavor'] == flavor]['DF_Vol'].sum()

    # Create background gradient for each row
    max_volume = max(flavor_volumes.values()) if flavor_volumes else 1

    # Create gradient matrix
    gradient_matrix = np.zeros((len(flavors), len(lengths)))
    for y_idx, flavor in enumerate(flavors):
        relative_volume = flavor_volumes[flavor] / max_volume
        gradient_matrix[y_idx, :] = np.linspace(0.1, relative_volume, len(lengths))

    # Plot background gradient
    im = ax.imshow(gradient_matrix, cmap=plt.cm.get_cmap('Greys'),
                  alpha=0.3, aspect='auto', origin='lower',
                  extent=(-0.5, len(lengths)-0.5, -0.5, len(flavors)-0.5))

    # Map thickness and taste to visual properties
    size_map = {thickness: (i+1)*150 for i, thickness in enumerate(thicknesses)}
    color_map = {taste: plt.cm.get_cmap(cmap_name)(i/len(tastes)) for i, taste in enumerate(tastes)}

    # Group products by Length-Flavor cell
    cell_products = {}
    for length in lengths:
        for flavor in flavors:
            cell_key = (length, flavor)
            cell_products[cell_key] = products_df[(products_df['Length'] == length) &
                                                 (products_df['Flavor'] == flavor)]

    # Plot products as boxes
    for cell_key, cell_df in cell_products.items():
        if len(cell_df) == 0:
            continue

        length, flavor = cell_key
        x_pos = lengths.index(length)
        y_pos = flavors.index(flavor)

        # Calculate total volume for this cell
        total_cell_volume = cell_df['DF_Vol'].sum()

        # Plot each product in this cell
        for idx, (_, product) in enumerate(cell_df.iterrows()):
            thickness = product['Thickness']
            taste = product['Taste']
            volume = product['DF_Vol']

            # Size based on thickness
            size = size_map.get(thickness, 100)

            # Color based on taste
            color = color_map.get(taste, 'blue')

            # Position within cell based on volume ratio
            products_in_cell = len(cell_df)
            grid_size = int(np.ceil(np.sqrt(products_in_cell)))

            # Calculate grid position
            grid_x = idx % grid_size
            grid_y = idx // grid_size

            # Calculate offsets within cell
            x_offset = (grid_x - grid_size/2 + 0.5) * 0.7 / grid_size
            y_offset = (grid_y - grid_size/2 + 0.5) * 0.7 / grid_size

            # Scale size by volume ratio
            volume_ratio = volume / total_cell_volume if total_cell_volume > 0 else 0.5
            scaled_size = size * (0.5 + 0.5 * volume_ratio)

            # Create circle representing product
            circle = plt.Circle((x_pos + x_offset, y_pos + y_offset),
                               scaled_size/1000,
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1,
                               alpha=0.8)
            ax.add_patch(circle)

    # Set up axes
    ax.set_xticks(range(len(lengths)))
    ax.set_yticks(range(len(flavors)))
    ax.set_xticklabels(lengths)
    ax.set_yticklabels(flavors)
    ax.set_xlabel('Length', fontsize=12)
    ax.set_ylabel('Flavor', fontsize=12)
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    return ax

def main():
    """Main function to load data and create visualizations"""
    # Set up directories
    data_dir = './locations_data'
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load location data
    location_data = load_location_data(data_dir)

    # Create individual shelf visualizations
    kuwait_shelf = create_advanced_product_shelf(location_data, 'Kuwait', cmap_name='Blues')
    if kuwait_shelf:
        kuwait_shelf.savefig(os.path.join(output_dir, "kuwait_shelf_view.png"), dpi=300, bbox_inches='tight')
        print("Kuwait shelf visualization created")

    jeju_shelf = create_advanced_product_shelf(location_data, 'Jeju', cmap_name='Oranges')
    if jeju_shelf:
        jeju_shelf.savefig(os.path.join(output_dir, "jeju_shelf_view.png"), dpi=300, bbox_inches='tight')
        print("Jeju shelf visualization created")

    # Create comparison view
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for i, (loc, cmap) in enumerate([('Kuwait', 'Blues'), ('Jeju', 'Oranges')]):
        create_advanced_product_shelf_subplot(axes[i], location_data, loc, cmap_name=cmap)
        axes[i].set_title(f'{loc} Product Portfolio', fontsize=16)

        # Add alignment score
        alignment_scores = {
            'Kuwait': {'Overall': 7.73, 'Flavor': 9.64, 'Taste': 8.10, 'Thickness': 5.03, 'Length': 8.17},
            'Jeju': {'Overall': 6.02, 'Flavor': 7.53, 'Taste': 4.37, 'Thickness': 5.82, 'Length': 6.38}
        }

        score_text = f"Overall Alignment: {alignment_scores[loc]['Overall']:.2f}/10\n"
        score_text += f"Flavor: {alignment_scores[loc]['Flavor']:.2f}/10\n"
        score_text += f"Taste: {alignment_scores[loc]['Taste']:.2f}/10\n"
        score_text += f"Thickness: {alignment_scores[loc]['Thickness']:.2f}/10\n"
        score_text += f"Length: {alignment_scores[loc]['Length']:.2f}/10"

        axes[i].text(0.02, 0.97, score_text, transform=axes[i].transAxes,
                    fontsize=10, va='top',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    plt.suptitle('Portfolio Comparison: Kuwait (Well-Aligned) vs Jeju (Misaligned)', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    fig.savefig(os.path.join(output_dir, "shelf_comparison.png"), dpi=300, bbox_inches='tight')
    print("Shelf comparison visualization created")

    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()