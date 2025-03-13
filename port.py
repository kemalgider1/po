import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import os
import json


def load_comparison_data(comparison_file_path):
    """
    Load and process the comparison data between Kuwait and Jeju.

    Args:
        comparison_file_path (str): Path to the comparison CSV file

    Returns:
        DataFrame: Processed comparison data
    """
    print("Loading comparison data...")
    try:
        comp_df = pd.read_csv(comparison_file_path)
        return comp_df
    except Exception as e:
        print(f"Error loading comparison data: {e}")
        return None


def load_product_data(kw_products_path, jj_products_path):
    """
    Load the product data for Kuwait and Jeju.

    Args:
        kw_products_path (str): Path to Kuwait products data
        jj_products_path (str): Path to Jeju products data

    Returns:
        tuple: (kw_df, jj_df) Kuwait and Jeju product dataframes
    """
    print("Loading product data...")
    try:
        kw_df = pd.read_csv(kw_products_path)
        jj_df = pd.read_csv(jj_products_path)
        return kw_df, jj_df
    except Exception as e:
        print(f"Error loading product data: {e}")
        return None, None


def load_attribute_analysis(kw_product_based_path, jj_product_based_path):
    """
    Load the attribute analysis data for Kuwait and Jeju.

    Args:
        kw_product_based_path (str): Path to Kuwait product-based analysis
        jj_product_based_path (str): Path to Jeju product-based analysis

    Returns:
        tuple: (kw_attr_df, jj_attr_df) Kuwait and Jeju attribute analysis dataframes
    """
    print("Loading attribute analysis data...")
    try:
        kw_attr_df = pd.read_csv(kw_product_based_path)
        jj_attr_df = pd.read_csv(jj_product_based_path)
        return kw_attr_df, jj_attr_df
    except Exception as e:
        print(f"Error loading attribute analysis data: {e}")
        return None, None


def load_paris_output(paris_output_path):
    """
    Load PARIS Output data containing ideal vs. real segment distribution.

    Args:
        paris_output_path (str): Path to PARIS_Output data file

    Returns:
        DataFrame: Loaded PARIS data
    """
    print("Loading PARIS Output data...")
    try:
        paris_df = pd.read_csv(paris_output_path)
        required_cols = ["Location", "Real_So_Segment", "Ideal_So_Segment", "Delta_SoS"]
        missing_cols = [col for col in required_cols if col not in paris_df.columns]

        if missing_cols:
            print(f"Warning: PARIS data is missing required columns: {missing_cols}")
            print("Available columns:", paris_df.columns.tolist())

        return paris_df
    except Exception as e:
        print(f"Error loading PARIS data: {e}")
        return None


def create_attribute_grid_visualization(comp_df, attribute, location1='Kuwait', location2='Jeju', paris_df=None):
    """
    Create a grid visualization comparing attribute distribution for two locations.

    Args:
        comp_df (DataFrame): Comparison data
        attribute (str): Attribute to visualize (Flavor, Taste, Thickness, Length)
        location1 (str): First location to compare (should be Kuwait - well aligned)
        location2 (str): Second location to compare (should be Jeju - misaligned)
        paris_df (DataFrame, optional): PARIS Output data for more accurate ideal values

    Returns:
        matplotlib.figure.Figure: The figure containing the grid visualization
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Extract attribute distribution data from comparison data
    locations = [location1, location2]
    location_data = {}

    for loc in locations:
        attr_values = []
        actual_vals = []
        pmi_vals = []
        ideal_vals = []
        gaps = []

        # First try to extract from PARIS data if available
        if paris_df is not None and attribute in paris_df.columns:
            loc_paris = paris_df[paris_df['Location'] == loc]

            if not loc_paris.empty:
                # Group by attribute to get real and ideal segment shares
                attr_group = loc_paris.groupby(attribute).agg({
                    'Real_So_Segment': 'sum',
                    'Ideal_So_Segment': 'sum',
                    'Delta_SoS': 'sum'
                }).reset_index()

                for _, row in attr_group.iterrows():
                    attr_values.append(row[attribute])
                    # Convert segment shares to percentages
                    actual_vals.append(row['Real_So_Segment'] * 100)
                    ideal_vals.append(row['Ideal_So_Segment'] * 100)
                    gaps.append(row['Delta_SoS'] * 100)

                    # Estimate PMI share (this might need to be improved with actual data)
                    # For now, we'll use a placeholder estimate based on real share
                    pmi_vals.append(row['Real_So_Segment'] * 100 * 0.8)  # Assuming PMI has 80% of real

        # If we couldn't get the data from PARIS, try from comparison data
        if not attr_values:
            # Find rows for this location in the comparison data
            for i, row in comp_df.iterrows():
                if pd.notna(row[0]) and attribute in str(row[0]):
                    # Found the attribute section, now get data for this location
                    idx = i + 1
                    while idx < len(comp_df) and pd.isna(comp_df.iloc[idx, 0]):
                        row_data = comp_df.iloc[idx]
                        attr_val = str(row_data.iloc[0]) if pd.notna(row_data.iloc[0]) else None

                        if attr_val and attr_val.strip():
                            # Get column indices for this location
                            if loc == location1:
                                col_offset = 1  # Columns for location1
                            else:
                                col_offset = 4  # Columns for location2

                            # Extract values if available
                            if col_offset < len(row_data):
                                attr_values.append(attr_val)

                                # Parse actual, ideal, and gap values
                                actual = float(row_data.iloc[col_offset]) if pd.notna(row_data.iloc[col_offset]) else 0
                                ideal = float(row_data.iloc[col_offset + 1]) if pd.notna(
                                    row_data.iloc[col_offset + 1]) and col_offset + 1 < len(row_data) else 0
                                gap = float(row_data.iloc[col_offset + 2]) if pd.notna(
                                    row_data.iloc[col_offset + 2]) and col_offset + 2 < len(row_data) else 0

                                actual_vals.append(actual)
                                ideal_vals.append(ideal)
                                gaps.append(gap)

                                # Estimate PMI share if not directly available
                                if col_offset + 3 < len(row_data) and pd.notna(row_data.iloc[col_offset + 3]):
                                    pmi_vals.append(float(row_data.iloc[col_offset + 3]))
                                else:
                                    pmi_vals.append(actual * 0.8)  # Placeholder estimate

                        idx += 1
                    break

        location_data[loc] = {
            'values': attr_values,
            'actual': actual_vals,
            'ideal': ideal_vals,
            'gaps': gaps,
            'pmi': pmi_vals
        }

    # Define a diverging colormap for gaps
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    # Create grid visualization for each location
    for i, loc in enumerate(locations):
        ax = axes[i]
        ax.set_title(f"{loc} - {attribute} Distribution", fontsize=14)
        ax.axis('off')

        if loc not in location_data or not location_data[loc]['values']:
            ax.text(0.5, 0.5, f"No data for {loc} - {attribute}", ha='center', va='center', fontsize=14)
            continue

        loc_values = location_data[loc]['values']
        actual_vals = location_data[loc]['actual']
        ideal_vals = location_data[loc]['ideal']
        gaps = location_data[loc]['gaps']
        pmi_vals = location_data[loc]['pmi']

        # Calculate grid layout
        n_values = len(loc_values)
        grid_cols = min(3, n_values)
        grid_rows = (n_values + grid_cols - 1) // grid_cols

        # Calculate cell size and spacing
        cell_width = 0.8 / grid_cols
        cell_height = 0.8 / grid_rows
        x_spacing = 0.05
        y_spacing = 0.05

        # Create color-coded grid cells
        for j, (value, actual, ideal, gap, pmi) in enumerate(zip(loc_values, actual_vals, ideal_vals, gaps, pmi_vals)):
            row = j // grid_cols
            col = j % grid_cols

            # Calculate cell position
            x = col * (cell_width + x_spacing) + 0.1
            y = 1 - (row * (cell_height + y_spacing) + 0.1 + cell_height)

            # Normalize gap for color mapping (-100 to 100 scale)
            norm_gap = min(1, max(0, (gap + 30) / 60))  # Center around 0 with +/-30% range
            color = cmap(norm_gap)

            # Create cell rectangle
            rect = plt.Rectangle((x, y), cell_width, cell_height,
                                 facecolor=color, alpha=0.8,
                                 transform=ax.transAxes)
            ax.add_patch(rect)

            # Add text for attribute value and metrics
            ax.text(x + cell_width / 2, y + cell_height * 0.85, value,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, fontweight='bold')

            ax.text(x + cell_width / 2, y + cell_height * 0.65, f"Market: {actual:.1f}%",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)

            ax.text(x + cell_width / 2, y + cell_height * 0.5, f"PMI: {pmi:.1f}%",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)

            ax.text(x + cell_width / 2, y + cell_height * 0.35, f"Ideal: {ideal:.1f}%",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)

            # Add gap text with appropriate color
            gap_text = f"Gap: {gap:.1f}%"
            gap_color = 'red' if gap < -5 else ('green' if gap > 5 else 'black')
            ax.text(x + cell_width / 2, y + cell_height * 0.15, gap_text,
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=gap_color, fontweight='bold')

        # Calculate alignment score
        alignment_score = 10 - min(10, sum(abs(g) for g in gaps) / 10)
        ax.text(0.02, 0.97, f"Alignment Score: {alignment_score:.1f}/10",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Gap: Underrepresented (Green) vs. Overrepresented (Red)')

    # Add market share information if available
    market_shares = get_market_shares(comp_df)
    if market_shares:
        for i, loc in enumerate(locations):
            if loc in market_shares:
                share = market_shares[loc]
                axes[i].text(0.02, 0.02, f"Market Share: {share:.1f}%",
                             transform=axes[i].transAxes, fontsize=11, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.suptitle(f"Portfolio Alignment Analysis: {attribute}", fontsize=16, y=1.05)
    return fig


def get_market_shares(comp_df):
    """Extract market share information from comparison data if available."""
    market_shares = {}

    # Default market shares based on project data
    market_shares['Kuwait'] = 75.0
    market_shares['Jeju'] = 11.4

    # Look for market share data in the comparison file
    for i, row in comp_df.iterrows():
        if pd.notna(row[0]) and 'Market Share' in str(row[0]):
            # Next rows might contain location-specific shares
            for j in range(i + 1, min(i + 5, len(comp_df))):
                if pd.notna(comp_df.iloc[j, 0]):
                    loc = comp_df.iloc[j, 0]
                    # Try to find share value in one of the numeric columns
                    for col in range(1, min(10, len(comp_df.columns))):
                        val = comp_df.iloc[j, col]
                        if pd.notna(val) and isinstance(val, (int, float)) and 0 <= val <= 100:
                            market_shares[loc] = val
                            break

    return market_shares


def create_attribute_heatmap(attr_df, location, attribute):
    """
    Create a heatmap visualization for a specific attribute.

    Args:
        attr_df (DataFrame): Attribute analysis data
        location (str): Location to visualize
        attribute (str): Attribute to visualize (Flavor, Taste, Thickness, Length)

    Returns:
        matplotlib.figure.Figure: The figure containing the heatmap
    """
    # Extract relevant columns based on attribute
    cols = [col for col in attr_df.columns if attribute in col and ('Actual' in col or 'Ideal' in col or 'Gap' in col)]

    if not cols:
        print(f"No data found for {attribute} in {location}")
        return None

    # Extract attribute values (rows)
    attr_values = []
    actual_vals = []
    ideal_vals = []
    gaps = []

    # Find the section for this attribute
    found_section = False
    for i in range(len(attr_df)):
        if pd.notna(attr_df.iloc[i, 0]) and attribute in str(attr_df.iloc[i, 0]):
            found_section = True
            continue

        if found_section:
            # Stop when we hit another attribute section
            if pd.notna(attr_df.iloc[i, 0]) and any(
                    attr in str(attr_df.iloc[i, 0]) for attr in ['Flavor', 'Taste', 'Thickness', 'Length']):
                break

            # Extract values
            row = attr_df.iloc[i]
            if pd.notna(row.iloc[0]) and row.iloc[0].strip():
                attr_values.append(row.iloc[0])

                # Find actual, ideal, and gap values based on column names
                actual_col = next((col for col in cols if 'Actual' in col), None)
                ideal_col = next((col for col in cols if 'Ideal' in col), None)
                gap_col = next((col for col in cols if 'Gap' in col), None)

                actual_idx = attr_df.columns.get_loc(actual_col) if actual_col else -1
                ideal_idx = attr_df.columns.get_loc(ideal_col) if ideal_col else -1
                gap_idx = attr_df.columns.get_loc(gap_col) if gap_col else -1

                actual_vals.append(
                    float(row.iloc[actual_idx]) if actual_idx >= 0 and pd.notna(row.iloc[actual_idx]) else 0)
                ideal_vals.append(float(row.iloc[ideal_idx]) if ideal_idx >= 0 and pd.notna(row.iloc[ideal_idx]) else 0)
                gaps.append(float(row.iloc[gap_idx]) if gap_idx >= 0 and pd.notna(row.iloc[gap_idx]) else 0)

    if not attr_values:
        print(f"No values found for {attribute} in {location}")
        return None

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # Create data for heatmaps
    heatmap_data = {
        'Actual': pd.DataFrame({'Values': actual_vals}, index=attr_values),
        'Ideal': pd.DataFrame({'Values': ideal_vals}, index=attr_values),
        'Gap': pd.DataFrame({'Values': gaps}, index=attr_values)
    }

    # Define colormaps
    cmap_actual = 'Blues'
    cmap_ideal = 'Greens'
    cmap_gap = 'RdYlGn'  # Red-Yellow-Green

    titles = ['Actual Distribution (%)', 'Ideal Distribution (%)', 'Gap (Ideal - Actual)']
    cmaps = [cmap_actual, cmap_ideal, cmap_gap]

    # Create heatmaps
    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        key = title.split()[0]
        data = heatmap_data[key]

        # Determine center for diverging colormap
        center = 0 if key == 'Gap' else None

        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap,
                    ax=axes[i], cbar=True, center=center,
                    linewidths=0.5, linecolor='white')

        axes[i].set_title(f"{location} - {title}")
        axes[i].set_ylabel(attribute)
        axes[i].set_xlabel('')

        # Rotate y-axis labels for better readability
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.suptitle(f"{location} - {attribute} Distribution Analysis", fontsize=14, y=1.05)
    return fig


def create_portfolio_alignment_visualization(kw_attr_df, jj_attr_df, attributes=None):
    """
    Create a comprehensive portfolio alignment visualization comparing Kuwait and Jeju.

    Args:
        kw_attr_df (DataFrame): Kuwait attribute analysis data
        jj_attr_df (DataFrame): Jeju attribute analysis data
        attributes (list, optional): List of attributes to include, defaults to all

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    if attributes is None:
        attributes = ['Flavor', 'Taste', 'Thickness', 'Length']

    # Create figure with subplots
    fig, axes = plt.subplots(len(attributes), 2, figsize=(16, 4 * len(attributes)))

    # Handle single attribute case
    if len(attributes) == 1:
        axes = axes.reshape(1, 2)

    # Define color maps
    cmap_actual = plt.cm.Blues
    cmap_ideal = plt.cm.Greens

    # Process each attribute
    for i, attribute in enumerate(attributes):
        for j, (attr_df, location) in enumerate([(kw_attr_df, 'Kuwait'), (jj_attr_df, 'Jeju')]):
            # Extract data for this attribute
            attr_values = []
            actual_vals = []
            ideal_vals = []
            gaps = []

            # Find the section for this attribute
            found_section = False
            for row_idx in range(len(attr_df)):
                if pd.notna(attr_df.iloc[row_idx, 0]) and attribute in str(attr_df.iloc[row_idx, 0]):
                    found_section = True
                    continue

                if found_section:
                    # Stop when we hit another attribute section
                    if pd.notna(attr_df.iloc[row_idx, 0]) and any(attr in str(attr_df.iloc[row_idx, 0]) for attr in
                                                                  ['Flavor', 'Taste', 'Thickness', 'Length']):
                        break

                    # Extract values
                    row = attr_df.iloc[row_idx]
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        # Find columns with actual, ideal values
                        actual_col = 1  # Assuming fixed positions based on observed patterns
                        ideal_col = 2

                        if pd.notna(row.iloc[actual_col]) and pd.notna(row.iloc[ideal_col]):
                            attr_values.append(str(row.iloc[0]))
                            actual_vals.append(float(row.iloc[actual_col]))
                            ideal_vals.append(float(row.iloc[ideal_col]))
                            gaps.append(float(row.iloc[ideal_col]) - float(row.iloc[actual_col]))

            if not attr_values:
                print(f"No values found for {attribute} in {location}")
                continue

            ax = axes[i, j]

            # Create bar chart
            x = np.arange(len(attr_values))
            width = 0.35

            # Sort data by ideal values for better visualization
            if ideal_vals:
                sorted_indices = np.argsort(ideal_vals)[::-1]  # Sort by ideal values (descending)
                attr_values = [attr_values[idx] for idx in sorted_indices]
                actual_vals = [actual_vals[idx] for idx in sorted_indices]
                ideal_vals = [ideal_vals[idx] for idx in sorted_indices]
                gaps = [gaps[idx] for idx in sorted_indices]

            # Plot actual distribution
            actual_bars = ax.bar(x - width / 2, actual_vals, width, label='Actual', color=cmap_actual(0.6))

            # Plot ideal distribution
            ideal_bars = ax.bar(x + width / 2, ideal_vals, width, label='Ideal', color=cmap_ideal(0.6))

            # Add gap indicators
            for idx, (actual, ideal, gap) in enumerate(zip(actual_vals, ideal_vals, gaps)):
                # Determine color based on gap
                color = 'green' if gap > 0 else 'red'

                # Add arrow or connector to show gap
                if abs(gap) > 5:  # Only show significant gaps
                    y_pos = max(actual, ideal) + 2
                    ax.annotate(f"{gap:.1f}", xy=(idx, y_pos), xytext=(idx, y_pos + 5),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                                ha='center', va='bottom', color=color, fontweight='bold')

            # Set title and labels
            ax.set_title(f"{location} - {attribute} Distribution", fontsize=12)
            ax.set_ylabel('Percentage (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(attr_values, rotation=45, ha='right')
            ax.legend()

            # Add alignment score based on absolute gaps
            alignment_score = 10 - min(10, sum(abs(g) for g in gaps) / 10)
            ax.text(0.02, 0.98, f"Alignment Score: {alignment_score:.1f}/10",
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.suptitle("Portfolio Alignment Analysis: Kuwait vs. Jeju", fontsize=16, y=1.02)
    return fig


def create_product_shelf_visualization(kw_df, jj_df, attribute1='Thickness', attribute2='Length'):
    """
    Create a visual representation of product "shelf" showing distribution across two attributes.

    Args:
        kw_df (DataFrame): Kuwait product data
        jj_df (DataFrame): Jeju product data
        attribute1 (str): First attribute for categorization (x-axis)
        attribute2 (str): Second attribute for categorization (y-axis)

    Returns:
        matplotlib.figure.Figure: The figure containing the shelf visualization
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))

    # Define color scheme
    pmi_color = 'darkblue'
    comp_color = 'gray'
    pmi_cmap = plt.cm.Blues
    comp_cmap = plt.cm.Greys

    # Process each dataset
    for idx, (df, location, ax) in enumerate([(kw_df, 'Kuwait', axes[0]), (jj_df, 'Jeju', axes[1])]):
        # Check if required attributes exist
        if attribute1 not in df.columns or attribute2 not in df.columns:
            print(f"Required attributes not found in {location} data. Available columns: {df.columns.tolist()}")
            ax.text(0.5, 0.5, f"Missing attribute data for {location}", ha='center', va='center', fontsize=14)
            continue

        # Get unique values for each attribute
        attr1_values = sorted(df[attribute1].unique())
        attr2_values = sorted(df[attribute2].unique())

        # Create a grid for visualization
        grid_shape = (len(attr2_values), len(attr1_values))
        grid = np.zeros(grid_shape)
        pmi_grid = np.zeros(grid_shape)
        comp_grid = np.zeros(grid_shape)

        # Fill the grid with product counts and volumes
        for a1_idx, a1_val in enumerate(attr1_values):
            for a2_idx, a2_val in enumerate(attr2_values):
                # Filter data for this attribute combination
                mask = (df[attribute1] == a1_val) & (df[attribute2] == a2_val)
                products = df[mask]

                if not products.empty:
                    # Count PMI and competitor products
                    pmi_products = products[products['TMO'] == 'PMI']
                    comp_products = products[products['TMO'] != 'PMI']

                    # Update grids with volumes (using log scale for better visualization)
                    volume_col = 'DF_Vol' if 'DF_Vol' in products.columns else '$current_year Volume'

                    if volume_col in products.columns:
                        grid[a2_idx, a1_idx] = np.log1p(products[volume_col].sum())
                        if not pmi_products.empty:
                            pmi_grid[a2_idx, a1_idx] = np.log1p(pmi_products[volume_col].sum())
                        if not comp_products.empty:
                            comp_grid[a2_idx, a1_idx] = np.log1p(comp_products[volume_col].sum())
                    else:
                        # If no volume column, just use counts
                        grid[a2_idx, a1_idx] = len(products)
                        pmi_grid[a2_idx, a1_idx] = len(pmi_products)
                        comp_grid[a2_idx, a1_idx] = len(comp_products)

        # Create heatmap base
        im = ax.imshow(grid, cmap='YlOrBr', alpha=0.3)

        # Add grid lines
        for i in range(grid_shape[1] + 1):
            ax.axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        for i in range(grid_shape[0] + 1):
            ax.axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)

        # Add product indicators
        for a1_idx, a1_val in enumerate(attr1_values):
            for a2_idx, a2_val in enumerate(attr2_values):
                # Filter data for this attribute combination
                mask = (df[attribute1] == a1_val) & (df[attribute2] == a2_val)
                products = df[mask]

                if not products.empty:
                    # Get PMI and competitor products
                    pmi_products = products[products['TMO'] == 'PMI']
                    comp_products = products[products['TMO'] != 'PMI']

                    # Get total volume in this cell
                    volume_col = 'DF_Vol' if 'DF_Vol' in products.columns else '$current_year Volume'

                    if volume_col in products.columns:
                        total_vol = products[volume_col].sum()
                        pmi_vol = pmi_products[volume_col].sum() if not pmi_products.empty else 0
                        comp_vol = comp_products[volume_col].sum() if not comp_products.empty else 0
                    else:
                        # If no volume column, use counts
                        total_vol = len(products)
                        pmi_vol = len(pmi_products)
                        comp_vol = len(comp_products)


                    # Add PMI indicator (circles)
                    if not pmi_products.empty:
                        # Scale size by volume proportion
                        size = 2000 * (pmi_vol / total_vol) if total_vol > 0 else 0