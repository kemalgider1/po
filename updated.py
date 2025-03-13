import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import os
from pathlib import Path


def load_id_act_data(file_path):
    """
    Load ideal vs. actual data from ID_ACT files.

    Args:
        file_path (str): Path to ID_ACT_*.xlsx file

    Returns:
        DataFrame: Loaded data
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading ID_ACT data from {file_path}: {e}")
        return None


def load_product_data(file_path):
    """
    Load product data from Product_list files.

    Args:
        file_path (str): Path to Product_list_*.xlsx file

    Returns:
        DataFrame: Loaded product data
    """
    try:
        df = pd.read_excel(file_path)

        # Check required columns
        required_cols = ['TMO', 'Flavor', 'Taste', 'Thickness', 'Length']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Warning: Missing columns in {file_path}: {missing_cols}")

        return df
    except Exception as e:
        print(f"Error loading product data from {file_path}: {e}")
        return None


def load_distribution_data(file_path):
    """
    Load attribute distribution data.

    Args:
        file_path (str): Path to distribution_*.xlsx file

    Returns:
        DataFrame: Loaded distribution data
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading distribution data from {file_path}: {e}")
        return None


def create_attribute_grid_visualization(attribute, kw_id_act_df, jj_id_act_df):
    """
    Create a grid visualization comparing attribute distribution for Kuwait and Jeju.

    Args:
        attribute (str): Attribute to visualize (Flavor, Taste, Thickness, Length)
        kw_id_act_df (DataFrame): Kuwait ID_ACT data
        jj_id_act_df (DataFrame): Jeju ID_ACT data

    Returns:
        matplotlib.figure.Figure: The figure containing the grid visualization
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Process data for Kuwait and Jeju
    locations = ['Kuwait', 'Jeju']
    dfs = [kw_id_act_df, jj_id_act_df]

    # Define a diverging colormap for gaps
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    # Process each location
    for i, (loc, df) in enumerate(zip(locations, dfs)):
        if df is None:
            axes[i].text(0.5, 0.5, f"No data for {loc}", ha='center', va='center', fontsize=14)
            axes[i].axis('off')
            continue

        # Check first column for attribute values
        if df.empty or pd.isna(df.iloc[0, 0]):
            axes[i].text(0.5, 0.5, f"No attribute values found for {loc}", ha='center', va='center', fontsize=14)
            axes[i].axis('off')
            continue

        # Find column indices for actual, ideal, pmi and gap
        actual_col = next((i for i, col in enumerate(df.columns) if 'Actual' in str(col)), 1)
        ideal_col = next((i for i, col in enumerate(df.columns) if 'Ideal' in str(col)), 2)
        gap_col = next((i for i, col in enumerate(df.columns) if 'Gap' in str(col)), 3)
        pmi_col = next((i for i, col in enumerate(df.columns) if 'PMI' in str(col)), 4)

        # Extract data from each row
        attr_values = []
        actual_vals = []
        ideal_vals = []
        gaps = []
        pmi_vals = []

        for row_idx in range(len(df)):
            if pd.isna(df.iloc[row_idx, 0]):
                continue

            value = str(df.iloc[row_idx, 0]).strip()

            # Skip header rows or empty rows
            if not value or value == 'nan' or value.lower() == attribute.lower():
                continue

            # Extract values
            actual = float(df.iloc[row_idx, actual_col]) if pd.notna(df.iloc[row_idx, actual_col]) else 0
            ideal = float(df.iloc[row_idx, ideal_col]) if pd.notna(df.iloc[row_idx, ideal_col]) else 0
            gap = float(df.iloc[row_idx, gap_col]) if pd.notna(df.iloc[row_idx, gap_col]) else ideal - actual
            pmi = float(df.iloc[row_idx, pmi_col]) if pd.notna(df.iloc[row_idx, pmi_col]) else 0

            attr_values.append(value)
            actual_vals.append(actual)
            ideal_vals.append(ideal)
            gaps.append(gap)
            pmi_vals.append(pmi)

        # Set up the plot
        ax = axes[i]
        ax.set_title(f"{loc} - {attribute} Distribution", fontsize=14)
        ax.axis('off')

        # Calculate grid layout
        n_values = len(attr_values)
        if n_values == 0:
            ax.text(0.5, 0.5, f"No data extracted for {loc}", ha='center', va='center', fontsize=14)
            continue

        grid_cols = min(3, n_values)
        grid_rows = (n_values + grid_cols - 1) // grid_cols

        # Calculate cell size and spacing
        cell_width = 0.8 / grid_cols
        cell_height = 0.8 / grid_rows
        x_spacing = 0.05
        y_spacing = 0.05

        # Create color-coded grid cells
        for j, (value, actual, ideal, gap, pmi) in enumerate(zip(attr_values, actual_vals, ideal_vals, gaps, pmi_vals)):
            row = j // grid_cols
            col = j % grid_cols

            # Calculate cell position
            x = col * (cell_width + x_spacing) + 0.1
            y = 1 - (row * (cell_height + y_spacing) + 0.1 + cell_height)

            # Normalize gap for color mapping
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

        # Add market share
        market_share = 75.0 if loc == 'Kuwait' else 11.4
        ax.text(0.02, 0.02, f"Market Share: {market_share:.1f}%",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Add a colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('Gap: Underrepresented (Green) vs. Overrepresented (Red)')

    plt.tight_layout()
    plt.suptitle(f"Portfolio Alignment Analysis: {attribute}", fontsize=16, y=1.05)
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

        # Fill the grid with product counts
        for a1_idx, a1_val in enumerate(attr1_values):
            for a2_idx, a2_val in enumerate(attr2_values):
                # Filter data for this attribute combination
                mask = (df[attribute1] == a1_val) & (df[attribute2] == a2_val)
                products = df[mask]

                if not products.empty:
                    # Count PMI and competitor products
                    pmi_products = products[products['TMO'] == 'PMI']
                    comp_products = products[products['TMO'] != 'PMI']

                    # Update grid with product count
                    grid[a2_idx, a1_idx] = len(products)

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

                    # Calculate proportions
                    total_count = len(products)
                    pmi_count = len(pmi_products)
                    comp_count = len(comp_products)

                    # Add PMI indicator (circles)
                    if pmi_count > 0:
                        # Scale size by proportion
                        size = 2000 * (pmi_count / total_count)
                        ax.scatter(a1_idx, a2_idx, s=size, color=pmi_color, alpha=0.7,
                                   label='PMI' if a1_idx == 0 and a2_idx == 0 else "")

                    # Add competitor indicator (squares)
                    if comp_count > 0:
                        # Scale size by proportion
                        size = 2000 * (comp_count / total_count)
                        ax.scatter(a1_idx, a2_idx, s=size, marker='s', color=comp_color, alpha=0.7,
                                   label='Competitor' if a1_idx == 0 and a2_idx == 0 else "")

                    # Add text for product count
                    ax.text(a1_idx, a2_idx, f"{total_count}", ha='center', va='center',
                            color='black', fontsize=9, fontweight='bold')

        # Set title and labels
        ax.set_title(f"{location} - Product Distribution by {attribute1} and {attribute2}", fontsize=14)
        ax.set_xticks(range(len(attr1_values)))
        ax.set_xticklabels(attr1_values, rotation=45, ha='right')
        ax.set_yticks(range(len(attr2_values)))
        ax.set_yticklabels(attr2_values)
        ax.set_xlabel(attribute1)
        ax.set_ylabel(attribute2)

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    return fig


def create_radar_chart(attribute_scores, locations=['Kuwait', 'Jeju']):
    """
    Create a radar chart comparing attribute alignment scores across locations.

    Args:
        attribute_scores (dict): Dict of attribute scores by location
        locations (list): Locations to compare

    Returns:
        matplotlib.figure.Figure: The figure containing the radar chart
    """
    # Define attributes
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']

    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plot each location
    angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    colors = {'Kuwait': 'green', 'Jeju': 'red'}

    for loc in locations:
        if loc in attribute_scores:
            scores = [attribute_scores[loc].get(attr, 0) for attr in attributes]
            scores += scores[:1]  # Close the loop

            ax.plot(angles, scores, 'o-', linewidth=2, label=loc, color=colors.get(loc, 'blue'))
            ax.fill(angles, scores, alpha=0.25, color=colors.get(loc, 'blue'))

    # Set chart properties
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.grid(True)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title("Attribute Alignment by Location", fontsize=15)
    return fig


def calculate_attribute_scores(kw_id_act_files, jj_id_act_files):
    """
    Calculate alignment scores for each attribute and location.

    Args:
        kw_id_act_files (dict): Dict of Kuwait ID_ACT files by attribute
        jj_id_act_files (dict): Dict of Jeju ID_ACT files by attribute

    Returns:
        dict: Dict of alignment scores by location and attribute
    """
    attribute_scores = {'Kuwait': {}, 'Jeju': {}}

    # Calculate for Kuwait
    for attr, file_path in kw_id_act_files.items():
        df = load_id_act_data(file_path)
        if df is None or df.empty:
            continue

        # Find gap column
        gap_col = next((i for i, col in enumerate(df.columns) if 'Gap' in str(col)), 3)

        # Extract gap values
        gaps = []
        for row_idx in range(len(df)):
            if pd.isna(df.iloc[row_idx, 0]):
                continue

            value = str(df.iloc[row_idx, 0]).strip()

            # Skip header rows or empty rows
            if not value or value == 'nan' or value.lower() == attr.lower():
                continue

            gap = float(df.iloc[row_idx, gap_col]) if pd.notna(df.iloc[row_idx, gap_col]) else 0
            gaps.append(gap)

        # Calculate alignment score
        if gaps:
            alignment_score = 10 - min(10, sum(abs(g) for g in gaps) / 10)
            attribute_scores['Kuwait'][attr] = alignment_score

    # Calculate for Jeju
    for attr, file_path in jj_id_act_files.items():
        df = load_id_act_data(file_path)
        if df is None or df.empty:
            continue

        # Find gap column
        gap_col = next((i for i, col in enumerate(df.columns) if 'Gap' in str(col)), 3)

        # Extract gap values
        gaps = []
        for row_idx in range(len(df)):
            if pd.isna(df.iloc[row_idx, 0]):
                continue

            value = str(df.iloc[row_idx, 0]).strip()

            # Skip header rows or empty rows
            if not value or value == 'nan' or value.lower() == attr.lower():
                continue

            gap = float(df.iloc[row_idx, gap_col]) if pd.notna(df.iloc[row_idx, gap_col]) else 0
            gaps.append(gap)

        # Calculate alignment score
        if gaps:
            alignment_score = 10 - min(10, sum(abs(g) for g in gaps) / 10)
            attribute_scores['Jeju'][attr] = alignment_score

    return attribute_scores


def create_summary_dashboard(attribute_scores):
    """
    Create a summary dashboard comparing Kuwait and Jeju.

    Args:
        attribute_scores (dict): Dict of attribute scores by location

    Returns:
        matplotlib.figure.Figure: The figure containing the summary dashboard
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Kuwait summary (left)
    axes[0].axis('off')
    axes[0].set_title("Kuwait - Well-Aligned Portfolio", fontsize=16)

    # Add market share
    axes[0].text(0.5, 0.9, "Market Share: ~75%", fontsize=14, fontweight='bold',
                 ha='center', va='center')

    # Add alignment scores
    y_pos = 0.8
    for attr in attributes:
        score = attribute_scores['Kuwait'].get(attr, 0)
        color = 'green' if score >= 7 else ('orange' if score >= 5 else 'red')
        axes[0].text(0.5, y_pos, f"{attr} Alignment: {score:.1f}/10",
                     fontsize=12, color=color, ha='center', va='center')
        y_pos -= 0.05

    # Add key insights
    axes[0].text(0.5, 0.6, "Key Insights:", fontsize=12, fontweight='bold',
                 ha='center', va='center')

    insights_kw = [
        "Well-balanced portfolio across all attributes",
        "Strong alignment with passenger preferences",
        "Product mix matches market demand",
        "Few gaps between PMI and ideal distribution"
    ]

    y_pos = 0.55
    for insight in insights_kw:
        axes[0].text(0.5, y_pos, f"• {insight}", fontsize=11, ha='center', va='center')
        y_pos -= 0.05

    # Jeju summary (right)
    axes[1].axis('off')
    axes[1].set_title("Jeju - Misaligned Portfolio", fontsize=16)

    # Add market share
    axes[1].text(0.5, 0.9, "Market Share: ~12%", fontsize=14, fontweight='bold',
                 ha='center', va='center')

    # Add alignment scores
    y_pos = 0.8
    for attr in attributes:
        score = attribute_scores['Jeju'].get(attr, 0)
        color = 'green' if score >= 7 else ('orange' if score >= 5 else 'red')
        axes[1].text(0.5, y_pos, f"{attr} Alignment: {score:.1f}/10",
                     fontsize=12, color=color, ha='center', va='center')
        y_pos -= 0.05

    # Add key insights
    axes[1].text(0.5, 0.6, "Key Insights:", fontsize=12, fontweight='bold',
                 ha='center', va='center')

    insights_jj = [
        "Significant misalignment in taste and thickness",
        "Underrepresentation in key passenger preferences",
        "Overrepresentation in low-demand segments",
        "Optimization potential for market share growth"
    ]

    y_pos = 0.55
    for insight in insights_jj:
        axes[1].text(0.5, y_pos, f"• {insight}", fontsize=11, ha='center', va='center')
        y_pos -= 0.05

    # Add optimization recommendations for Jeju
    axes[1].text(0.5, 0.3, "Optimization Recommendations:", fontsize=12,
                 fontweight='bold', ha='center', va='center')

    recommendations = [
        "Increase SKUs in underrepresented segments",
        "Reduce SKUs in overrepresented segments",
        "Align portfolio with passenger mix",
        "Focus on high-demand attributes"
    ]

    y_pos = 0.25
    for rec in recommendations:
        axes[1].text(0.5, y_pos, f"• {rec}", fontsize=11, ha='center', va='center')
        y_pos -= 0.05

    plt.tight_layout()
    fig.suptitle("Portfolio Optimization Summary", fontsize=18, y=0.98)

    return fig


def load_validation_results(cat_c_validation_file):
    """
    Load Category C validation results.

    Args:
        cat_c_validation_file (str): Path to cat_c_validation.txt

    Returns:
        dict: Dict of validation results by location
    """
    validation_results = {}

    try:
        with open(cat_c_validation_file, 'r') as f:
            content = f.read()

        # Parse validation results
        location = None
        current_section = None

        for line in content.split('\n'):
            line = line.strip()

            if not line:
                continue

            if line.startswith('Location:'):
                location = line.split(':')[1].strip()
                validation_results[location] = {}
                current_section = 'location'
            elif line.startswith('Status:') and location:
                validation_results[location]['status'] = line.split(':')[1].strip()
            elif line.startswith('Data Points:') and location:
                validation_results[location]['data_points'] = int(line.split(':')[1].strip())
            elif line.startswith('Category C Score:') and location:
                validation_results[location]['cat_c_score'] = float(line.split(':')[1].strip())
            elif line.startswith('Correlation:') and location:
                validation_results[location]['correlation'] = float(line.split(':')[1].strip())
            elif line.startswith('R²:') and location:
                validation_results[location]['r_squared'] = float(line.split(':')[1].strip())
            elif line.startswith('Average Real Segment Share:') and location:
                validation_results[location]['avg_real_segment'] = float(line.split(':')[1].strip())
            elif line.startswith('Average Ideal Segment Share:') and location:
                validation_results[location]['avg_ideal_segment'] = float(line.split(':')[1].strip())
            elif line.startswith('Sum of Positive Gaps:') and location:
                validation_results[location]['positive_gaps_sum'] = float(line.split(':')[1].strip())
            elif line.startswith('Sum of Negative Gaps:') and location:
                validation_results[location]['negative_gaps_sum'] = float(line.split(':')[1].strip())
            elif line.startswith('Standard Deviation of Gaps:') and location:
                validation_results[location]['gap_std'] = float(line.split(':')[1].strip())
            elif line.startswith('2. Attribute Alignment Analysis'):
                current_section = 'attribute_alignment'

        return validation_results
    except Exception as e:
        print(f"Error loading validation results: {e}")
        return {}


def integrate_validation_results(attribute_scores, validation_results):
    """
    Integrate Category C validation results with attribute scores.

    Args:
        attribute_scores (dict): Dict of attribute scores by location
        validation_results (dict): Dict of validation results by location

    Returns:
        dict: Updated attribute scores
    """
    # Add Category C scores from validation results
    for location, results in validation_results.items():
        if 'cat_c_score' in results:
            # Update Category C specific scores
            if location in attribute_scores:
                attribute_scores[location]['Cat_C'] = results['cat_c_score']

    return attribute_scores


def generate_portfolio_optimization_dashboard(data_dir, output_dir=None, cat_c_validation_file=None):
    """
    Generate a comprehensive portfolio optimization dashboard.

    Args:
        data_dir (str): Directory containing data files
        output_dir (str, optional): Directory to save visualization outputs
        cat_c_validation_file (str, optional): Path to cat_c_validation.txt

    Returns:
        dict: Dictionary of visualization figures
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    kw_id_act_files = {
        'Flavor': os.path.join(data_dir, 'ID_ACT_flavor.xlsx'),
        'Taste': os.path.join(data_dir, 'ID_ACT_taste.xlsx'),
        'Thickness': os.path.join(data_dir, 'ID_ACT_thickness.xlsx'),
        'Length': os.path.join(data_dir, 'ID_ACT_length.xlsx')
    }

    jj_id_act_files = kw_id_act_files  # Same files, different columns for each location

    product_files = {
        'Kuwait': os.path.join(data_dir, 'Product_list_PMI_KW.xlsx'),
        'Jeju': os.path.join(data_dir, 'Product_list_PMI_JJ.xlsx')
    }

    distribution_files = {
        'Flavor': {
            'Kuwait': os.path.join(data_dir, 'distribution_flavor_KW.xlsx'),
            'Jeju': os.path.join(data_dir, 'distribution_flavor_JJ.xlsx')
        },
        'Taste': {
            'Kuwait': os.path.join(data_dir, 'distribution_taste_KW.xlsx'),
            'Jeju': os.path.join(data_dir, 'distribution_taste_JJ.xlsx')
        },
        'Thickness': {
            'Kuwait': os.path.join(data_dir, 'distribution_thickness_KW.xlsx'),
            'Jeju': os.path.join(data_dir, 'distribution_thickness_JJ.xlsx')
        },
        'Length': {
            'Kuwait': os.path.join(data_dir, 'distribution_length_KW.xlsx'),
            'Jeju': os.path.join(data_dir, 'distribution_length_JJ.xlsx')
        }
    }

    # Load data
    kw_products = load_product_data(product_files['Kuwait'])
    jj_products = load_product_data(product_files['Jeju'])

    # Load ID_ACT data
    kw_id_act_data = {}
    jj_id_act_data = {}

    for attr in ['Flavor', 'Taste', 'Thickness', 'Length']:
        kw_id_act_data[attr] = load_id_act_data(kw_id_act_files[attr])
        jj_id_act_data[attr] = kw_id_act_data[attr]  # Same file, different columns

    # Calculate attribute scores
    attribute_scores = calculate_attribute_scores(kw_id_act_files, jj_id_act_files)

    # Load Category C validation results if available
    if cat_c_validation_file and os.path.exists(cat_c_validation_file):
        validation_results = load_validation_results(cat_c_validation_file)
        attribute_scores = integrate_validation_results(attribute_scores, validation_results)

    # Generate visualizations
    visualizations = {}
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']

    # 1. Create shelf visualizations for each attribute
    print("Creating attribute shelf visualizations...")
    for attr in attributes:
        fig = create_attribute_grid_visualization(attr, kw_id_act_data[attr], jj_id_act_data[attr])
        visualizations[f"shelf_{attr.lower()}"] = fig

        if output_dir:
            fig.savefig(os.path.join(output_dir, f"shelf_{attr.lower()}.png"), dpi=300, bbox_inches='tight')

    # 2. Create product shelf visualization
    print("Creating product shelf visualization...")
    if kw_products is not None and jj_products is not None:
        shelf_fig = create_product_shelf_visualization(kw_products, jj_products)
        visualizations["product_shelf"] = shelf_fig

        if output_dir:
            shelf_fig.savefig(os.path.join(output_dir, "product_shelf.png"), dpi=300, bbox_inches='tight')

    # 3. Create radar chart comparing attribute alignment
    print("Creating radar chart...")
    radar_fig = create_radar_chart(attribute_scores)
    visualizations["radar_chart"] = radar_fig

    if output_dir:
        radar_fig.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')

    # 4. Create summary dashboard
    print("Creating summary dashboard...")
    summary_fig = create_summary_dashboard(attribute_scores)
    visualizations["summary_dashboard"] = summary_fig

    if output_dir:
        summary_fig.savefig(os.path.join(output_dir, "summary_dashboard.png"), dpi=300, bbox_inches='tight')

    print(f"Generated {len(visualizations)} visualizations")
    return visualizations


def process_distribution_data():
    """
    Process distribution data from distribution files to create integrated dataset.

    Returns:
        dict: Dictionary of processed distribution data by location and attribute
    """
    # This function would process and integrate data from the distribution_*.xlsx files
    # Not implemented in detail here since these files are used as supplementary data
    pass


def create_combined_attribute_dashboard(attribute_scores, validation_results=None):
    """
    Create a combined dashboard showing all attributes with Category C validation results.

    Args:
        attribute_scores (dict): Dict of attribute scores by location
        validation_results (dict, optional): Dict of validation results by location

    Returns:
        matplotlib.figure.Figure: The figure containing the combined dashboard
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Flatten axes
    axes = axes.flatten()

    # Plot each attribute
    for i, attr in enumerate(attributes):
        ax = axes[i]

        # Setup bar positions
        locations = ['Kuwait', 'Jeju']
        x = np.arange(len(locations))
        width = 0.35

        # Extract scores
        scores = [attribute_scores[loc].get(attr, 0) for loc in locations]

        # Create bars
        ax.bar(x, scores, width, color=['green', 'red'])

        # Add labels
        ax.set_title(f"{attr} Alignment", fontsize=14)
        ax.set_ylabel('Alignment Score (0-10)')
        ax.set_ylim(0, 10)
        ax.set_xticks(x)
        ax.set_xticklabels(locations)

        # Add score values
        for j, score in enumerate(scores):
            ax.text(j, score + 0.3, f"{score:.1f}", ha='center', fontsize=12)

    plt.tight_layout()
    fig.suptitle("Portfolio Alignment Scores by Attribute", fontsize=18, y=1.02)

    return fig


def load_comparison_data(comparison_file_path):
    """
    Load comparison data between Kuwait and Jeju.

    Args:
        comparison_file_path (str): Path to comparison file

    Returns:
        DataFrame: Comparison data
    """
    try:
        comp_df = pd.read_excel(comparison_file_path)
        return comp_df
    except Exception as e:
        print(f"Error loading comparison data: {e}")
        return None


def create_differential_heatmap(kw_df, jj_df, attribute1='Thickness', attribute2='Length'):
    """
    Create a heatmap showing the differential between Kuwait and Jeju distributions.

    Args:
        kw_df (DataFrame): Kuwait product data
        jj_df (DataFrame): Jeju product data
        attribute1 (str): First attribute for categorization
        attribute2 (str): Second attribute for categorization

    Returns:
        matplotlib.figure.Figure: The figure containing the differential heatmap
    """
    if attribute1 not in kw_df.columns or attribute2 not in kw_df.columns:
        print(f"Required attributes not found in Kuwait data")
        return None

    if attribute1 not in jj_df.columns or attribute2 not in jj_df.columns:
        print(f"Required attributes not found in Jeju data")
        return None

    # Get unique values
    attr1_values = sorted(set(list(kw_df[attribute1].unique()) + list(jj_df[attribute1].unique())))
    attr2_values = sorted(set(list(kw_df[attribute2].unique()) + list(jj_df[attribute2].unique())))

    # Create grids
    grid_shape = (len(attr2_values), len(attr1_values))
    kw_grid = np.zeros(grid_shape)
    jj_grid = np.zeros(grid_shape)

    # Fill Kuwait grid
    for a1_idx, a1_val in enumerate(attr1_values):
        for a2_idx, a2_val in enumerate(attr2_values):
            mask = (kw_df[attribute1] == a1_val) & (kw_df[attribute2] == a2_val) & (kw_df['TMO'] == 'PMI')
            kw_grid[a2_idx, a1_idx] = len(kw_df[mask])

    # Fill Jeju grid
    for a1_idx, a1_val in enumerate(attr1_values):
        for a2_idx, a2_val in enumerate(attr2_values):
            mask = (jj_df[attribute1] == a1_val) & (jj_df[attribute2] == a2_val) & (jj_df['TMO'] == 'PMI')
            jj_grid[a2_idx, a1_idx] = len(jj_df[mask])

    # Calculate differentials
    kw_total = kw_grid.sum()
    jj_total = jj_grid.sum()

    if kw_total == 0 or jj_total == 0:
        print("Error: One or both locations have zero products")
        return None

    # Normalize grids to percentages
    kw_grid_pct = kw_grid / kw_total * 100
    jj_grid_pct = jj_grid / jj_total * 100

    # Calculate differential
    diff_grid = jj_grid_pct - kw_grid_pct

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Kuwait distribution
    im1 = axes[0].imshow(kw_grid_pct, cmap='Blues')
    axes[0].set_title("Kuwait Distribution")
    axes[0].set_xticks(range(len(attr1_values)))
    axes[0].set_xticklabels(attr1_values, rotation=45, ha='right')
    axes[0].set_yticks(range(len(attr2_values)))
    axes[0].set_yticklabels(attr2_values)
    axes[0].set_xlabel(attribute1)
    axes[0].set_ylabel(attribute2)
    plt.colorbar(im1, ax=axes[0], label='% of Products')

    # Jeju distribution
    im2 = axes[1].imshow(jj_grid_pct, cmap='Reds')
    axes[1].set_title("Jeju Distribution")
    axes[1].set_xticks(range(len(attr1_values)))
    axes[1].set_xticklabels(attr1_values, rotation=45, ha='right')
    axes[1].set_yticks(range(len(attr2_values)))
    axes[1].set_yticklabels(attr2_values)
    axes[1].set_xlabel(attribute1)
    axes[1].set_ylabel(attribute2)
    plt.colorbar(im2, ax=axes[1], label='% of Products')

    # Differential
    im3 = axes[2].imshow(diff_grid, cmap='RdBu_r', norm=plt.Normalize(vmin=-10, vmax=10))
    axes[2].set_title("Differential (Jeju - Kuwait)")
    axes[2].set_xticks(range(len(attr1_values)))
    axes[2].set_xticklabels(attr1_values, rotation=45, ha='right')
    axes[2].set_yticks(range(len(attr2_values)))
    axes[2].set_yticklabels(attr2_values)
    axes[2].set_xlabel(attribute1)
    axes[2].set_ylabel(attribute2)
    plt.colorbar(im3, ax=axes[2], label='Difference in % Points')

    # Add text annotations
    for a1_idx, a1_val in enumerate(attr1_values):
        for a2_idx, a2_val in enumerate(attr2_values):
            kw_val = kw_grid_pct[a2_idx, a1_idx]
            jj_val = jj_grid_pct[a2_idx, a1_idx]
            diff_val = diff_grid[a2_idx, a1_idx]

            # Add text for Kuwait
            if kw_val > 1:
                axes[0].text(a1_idx, a2_idx, f"{kw_val:.1f}%", ha='center', va='center',
                             color='white' if kw_val > 5 else 'black', fontsize=8)

            # Add text for Jeju
            if jj_val > 1:
                axes[1].text(a1_idx, a2_idx, f"{jj_val:.1f}%", ha='center', va='center',
                             color='white' if jj_val > 5 else 'black', fontsize=8)

            # Add text for differential
            if abs(diff_val) > 1:
                axes[2].text(a1_idx, a2_idx, f"{diff_val:+.1f}", ha='center', va='center',
                             color='white' if abs(diff_val) > 5 else 'black', fontsize=8)

    plt.tight_layout()
    fig.suptitle(f"PMI Portfolio Distribution Comparison: {attribute1} vs {attribute2}", fontsize=16, y=1.05)

    return fig


# Main execution
if __name__ == "__main__":
    data_dir = "data"
    output_dir = "visualizations"
    cat_c_validation_file = os.path.join(data_dir, "cat_c_validation.txt")

    # Generate all visualizations
    visualizations = generate_portfolio_optimization_dashboard(
        data_dir,
        output_dir,
        cat_c_validation_file
    )

    print(f"All visualizations saved to {output_dir}")