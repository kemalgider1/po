import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pathlib import Path


def load_comparison_data(comparison_files):
    """
    Load and process the attribute comparison data between Kuwait and Jeju.

    Args:
        comparison_files (dict): Dictionary mapping attribute names to file paths

    Returns:
        dict: Processed comparison data by attribute
    """
    print("Loading comparison data...")

    comp_data = {}
    for attr, file_path in comparison_files.items():
        try:
            df = pd.read_csv(file_path)
            comp_data[attr] = df
            print(f"Loaded {attr} comparison data: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading {attr} comparison data: {e}")

    return comp_data


def load_location_data(location_data_files):
    """
    Load product and analysis data for specific locations.

    Args:
        location_data_files (dict): Dictionary mapping locations to their data files

    Returns:
        dict: Processed location data
    """
    print("Loading location-specific data...")

    location_data = {}

    for location, files in location_data_files.items():
        location_data[location] = {}

        for data_type, file_path in files.items():
            try:
                df = pd.read_csv(file_path)
                location_data[location][data_type] = df
                print(f"Loaded {location} {data_type} data: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"Error loading {location} {data_type} data: {e}")

    return location_data


def load_validation_data(validation_file):
    """
    Load category C validation data.

    Args:
        validation_file (str): Path to validation data file

    Returns:
        dict: Processed validation data
    """
    print("Loading validation data...")

    validation_data = {}

    try:
        # Since this is a text file, we'll parse it manually
        with open(validation_file, 'r') as f:
            content = f.read()

        # Parse sections for each location
        locations = ['Kuwait', 'Jeju']

        for location in locations:
            # Extract location section
            location_start = content.find(f"Location: {location}")
            if location_start == -1:
                continue

            # Find next location or end of file
            next_location = content.find("Location:", location_start + 1)
            if next_location == -1:
                location_content = content[location_start:]
            else:
                location_content = content[location_start:next_location]

            # Parse metrics
            validation_data[location] = {
                'cat_c_score': float(location_content.split("Category C Score: ")[1].split('\n')[
                                         0]) if "Category C Score:" in location_content else None,
                'correlation': float(location_content.split("Correlation: ")[1].split('\n')[
                                         0]) if "Correlation:" in location_content else None,
                'r_squared': float(
                    location_content.split("R²: ")[1].split('\n')[0]) if "R²:" in location_content else None
            }

            print(f"Loaded validation data for {location}")

    except Exception as e:
        print(f"Error loading validation data: {e}")

    return validation_data


def create_comparative_portfolio_grid(location_data, attributes=['Flavor', 'Taste', 'Thickness', 'Length']):
    """
    Create a grid visualization comparing current vs. ideal product distribution for Kuwait and Jeju.

    Args:
        location_data (dict): Dictionary containing location-specific data
        attributes (list): List of attributes to visualize

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    locations = ['Kuwait', 'Jeju']
    location_titles = {
        'Kuwait': 'Kuwait - Well-Aligned Portfolio (Market Share: ~75%)',
        'Jeju': 'Jeju - Misaligned Portfolio (Market Share: ~12%)'
    }

    # Custom diverging colormap for gaps
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    # Process each location and attribute
    for i, location in enumerate(locations):
        for j, attribute in enumerate(attributes[:2]):  # First row: Flavor, Taste
            attr_data = None
            for data_type in location_data[location]:
                if attribute.lower() in data_type.lower() and 'distribution' in data_type.lower():
                    attr_data = location_data[location][data_type]
                    break

            if attr_data is None:
                print(f"No {attribute} distribution data found for {location}")
                continue

            # Create subplot
            ax = plt.subplot(gs[i, j])

            # Sort by ideal percentage for better visualization
            if 'Ideal_Percentage' in attr_data.columns:
                sorted_data = attr_data.sort_values('Ideal_Percentage', ascending=False)
            else:
                sorted_data = attr_data

            # Extract relevant columns
            x = range(len(sorted_data))
            attr_values = sorted_data[attribute].tolist()

            # Get actual, PMI, and ideal percentages
            actual_pct = sorted_data['Volume_Percentage'] if 'Volume_Percentage' in sorted_data.columns else \
            sorted_data['Market_vs_Ideal_Gap'] + sorted_data['Ideal_Percentage']
            pmi_pct = sorted_data['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in sorted_data.columns else None
            ideal_pct = sorted_data['Ideal_Percentage'] if 'Ideal_Percentage' in sorted_data.columns else None

            # Set up bar positions
            width = 0.3

            # Plot bars
            if ideal_pct is not None:
                ax.bar(x, ideal_pct, width, label='Ideal Distribution', color='green', alpha=0.7)

            if actual_pct is not None:
                ax.bar([p + width for p in x], actual_pct, width, label='Current Market', color='blue', alpha=0.7)

            if pmi_pct is not None:
                ax.bar([p + width * 2 for p in x], pmi_pct, width, label='PMI Portfolio', color='red', alpha=0.7)

            # Calculate and visualize gaps
            if ideal_pct is not None and actual_pct is not None:
                for idx, (ideal, actual, attr_val) in enumerate(zip(ideal_pct, actual_pct, attr_values)):
                    gap = actual - ideal
                    gap_color = 'red' if gap < -5 else ('green' if gap > 5 else 'black')
                    ax.annotate(f"{gap:.1f}%",
                                xy=(idx + width, max(ideal, actual) + 2),
                                ha='center', color=gap_color, fontweight='bold')

            # Add titles and labels
            ax.set_title(f"{location} - {attribute} Distribution", fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_xticks([p + width for p in x])
            ax.set_xticklabels(attr_values, rotation=45, ha='right')
            ax.legend()

            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Add alignment score if available
            alignment_scores = {
                'Kuwait': {'Flavor': 9.64, 'Taste': 8.10, 'Thickness': 5.03, 'Length': 8.17},
                'Jeju': {'Flavor': 7.53, 'Taste': 4.37, 'Thickness': 5.82, 'Length': 6.38}
            }

            if location in alignment_scores and attribute in alignment_scores[location]:
                score = alignment_scores[location][attribute]
                ax.text(0.02, 0.98, f"Alignment Score: {score}/10",
                        transform=ax.transAxes, fontsize=12, fontweight='bold',
                        va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    for i, location in enumerate(locations):
        for j, attribute in enumerate(attributes[2:], 2):  # Second row: Thickness, Length
            j = j - 2  # Adjust index for second row

            attr_data = None
            for data_type in location_data[location]:
                if attribute.lower() in data_type.lower() and 'distribution' in data_type.lower():
                    attr_data = location_data[location][data_type]
                    break

            if attr_data is None:
                print(f"No {attribute} distribution data found for {location}")
                continue

            # Create subplot
            ax = plt.subplot(gs[i, j])

            # Sort by ideal percentage for better visualization
            if 'Ideal_Percentage' in attr_data.columns:
                sorted_data = attr_data.sort_values('Ideal_Percentage', ascending=False)
            else:
                sorted_data = attr_data

            # Extract relevant columns
            x = range(len(sorted_data))
            attr_values = sorted_data[attribute].tolist()

            # Get actual, PMI, and ideal percentages
            actual_pct = sorted_data['Volume_Percentage'] if 'Volume_Percentage' in sorted_data.columns else \
            sorted_data['Market_vs_Ideal_Gap'] + sorted_data['Ideal_Percentage']
            pmi_pct = sorted_data['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in sorted_data.columns else None
            ideal_pct = sorted_data['Ideal_Percentage'] if 'Ideal_Percentage' in sorted_data.columns else None

            # Set up bar positions
            width = 0.3

            # Plot bars
            if ideal_pct is not None:
                ax.bar(x, ideal_pct, width, label='Ideal Distribution', color='green', alpha=0.7)

            if actual_pct is not None:
                ax.bar([p + width for p in x], actual_pct, width, label='Current Market', color='blue', alpha=0.7)

            if pmi_pct is not None:
                ax.bar([p + width * 2 for p in x], pmi_pct, width, label='PMI Portfolio', color='red', alpha=0.7)

            # Calculate and visualize gaps
            if ideal_pct is not None and actual_pct is not None:
                for idx, (ideal, actual, attr_val) in enumerate(zip(ideal_pct, actual_pct, attr_values)):
                    gap = actual - ideal
                    gap_color = 'red' if gap < -5 else ('green' if gap > 5 else 'black')
                    ax.annotate(f"{gap:.1f}%",
                                xy=(idx + width, max(ideal, actual) + 2),
                                ha='center', color=gap_color, fontweight='bold')

            # Add titles and labels
            ax.set_title(f"{location} - {attribute} Distribution", fontsize=14)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_xticks([p + width for p in x])
            ax.set_xticklabels(attr_values, rotation=45, ha='right')
            ax.legend()

            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Add alignment score if available
            alignment_scores = {
                'Kuwait': {'Flavor': 9.64, 'Taste': 8.10, 'Thickness': 5.03, 'Length': 8.17},
                'Jeju': {'Flavor': 7.53, 'Taste': 4.37, 'Thickness': 5.82, 'Length': 6.38}
            }

            if location in alignment_scores and attribute in alignment_scores[location]:
                score = alignment_scores[location][attribute]
                ax.text(0.02, 0.98, f"Alignment Score: {score}/10",
                        transform=ax.transAxes, fontsize=12, fontweight='bold',
                        va='top', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    plt.suptitle("Portfolio Alignment Analysis: Kuwait (Well-Aligned) vs. Jeju (Misaligned)", fontsize=20, y=0.98)

    return fig


def create_product_shelf_heatmap(location_data, attributes=['Taste', 'Thickness']):
    """
    Create a heat map visualization showing product distribution across a grid of attributes.

    Args:
        location_data (dict): Dictionary containing location-specific data
        attributes (list): List of attributes to use for x and y axes

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    locations = ['Kuwait', 'Jeju']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for i, location in enumerate(locations):
        # Get product data
        top_products = None
        for data_type in location_data[location]:
            if 'top_90pct' in data_type.lower() or 'pmi_products' in data_type.lower():
                top_products = location_data[location][data_type]
                break

        if top_products is None:
            print(f"No product data found for {location}")
            continue

        # Get attribute distributions
        attr_distributions = {}
        for attr in attributes:
            for data_type in location_data[location]:
                if attr.lower() in data_type.lower() and 'distribution' in data_type.lower():
                    attr_distributions[attr] = location_data[location][data_type]
                    break

        # Check if we have both required attributes
        if len(attr_distributions) < 2:
            print(f"Missing attribute distribution data for {location}")
            continue

        # Create product distribution matrix
        attr_x, attr_y = attributes[:2]  # Use first two attributes for x and y axes

        # Get unique values for each attribute
        x_values = sorted(top_products[attr_x].unique())
        y_values = sorted(top_products[attr_y].unique())

        # Create matrices for PMI and market volumes
        pmi_matrix = np.zeros((len(y_values), len(x_values)))
        market_matrix = np.zeros((len(y_values), len(x_values)))
        ideal_matrix = np.zeros((len(y_values), len(x_values)))

        # Fill matrices
        for x_idx, x_val in enumerate(x_values):
            for y_idx, y_val in enumerate(y_values):
                # Get PMI volume
                pmi_vol = top_products[(top_products[attr_x] == x_val) &
                                       (top_products[attr_y] == y_val) &
                                       (top_products['TMO'] == 'PMI')][
                    'DF_Vol'].sum() if 'DF_Vol' in top_products.columns else 0

                # Get market volume
                market_vol = top_products[(top_products[attr_x] == x_val) &
                                          (top_products[attr_y] == y_val)][
                    'DF_Vol'].sum() if 'DF_Vol' in top_products.columns else 0

                # Calculate ideal volume based on distributions (this is an approximation)
                x_dist = attr_distributions[attr_x]
                y_dist = attr_distributions[attr_y]

                x_ideal_pct = x_dist[x_dist[attr_x] == x_val]['Ideal_Percentage'].values[0] if len(
                    x_dist[x_dist[attr_x] == x_val]) > 0 else 0
                y_ideal_pct = y_dist[y_dist[attr_y] == y_val]['Ideal_Percentage'].values[0] if len(
                    y_dist[y_dist[attr_y] == y_val]) > 0 else 0

                # Rough approximation of joint distribution
                ideal_pct = (x_ideal_pct * y_ideal_pct) / 100
                ideal_vol = ideal_pct * market_vol / 100 if market_vol > 0 else 0

                pmi_matrix[y_idx, x_idx] = pmi_vol
                market_matrix[y_idx, x_idx] = market_vol
                ideal_matrix[y_idx, x_idx] = ideal_vol

        # Normalize matrices
        pmi_total = pmi_matrix.sum()
        market_total = market_matrix.sum()
        ideal_total = ideal_matrix.sum()

        if pmi_total > 0:
            pmi_matrix = pmi_matrix / pmi_total * 100

        if market_total > 0:
            market_matrix = market_matrix / market_total * 100

        if ideal_total > 0:
            ideal_matrix = ideal_matrix / ideal_total * 100

        # Calculate gap matrix
        gap_matrix = market_matrix - ideal_matrix

        # Plot current market distribution
        im1 = axes[i, 0].imshow(market_matrix, cmap='Blues', interpolation='nearest')
        axes[i, 0].set_title(f"{location} - Current Market Distribution", fontsize=14)
        axes[i, 0].set_xlabel(attr_x, fontsize=12)
        axes[i, 0].set_ylabel(attr_y, fontsize=12)
        axes[i, 0].set_xticks(range(len(x_values)))
        axes[i, 0].set_yticks(range(len(y_values)))
        axes[i, 0].set_xticklabels(x_values)
        axes[i, 0].set_yticklabels(y_values)

        # Add text annotations for current market
        for y_idx in range(len(y_values)):
            for x_idx in range(len(x_values)):
                if market_matrix[y_idx, x_idx] > 0:
                    text_color = 'white' if market_matrix[y_idx, x_idx] > 10 else 'black'
                    axes[i, 0].text(x_idx, y_idx, f"{market_matrix[y_idx, x_idx]:.1f}%",
                                    ha="center", va="center", color=text_color, fontweight='bold')

        # Add colorbar
        plt.colorbar(im1, ax=axes[i, 0], label='% of Volume')

        # Plot ideal distribution
        im2 = axes[i, 1].imshow(ideal_matrix, cmap='Greens', interpolation='nearest')
        axes[i, 1].set_title(f"{location} - Ideal Distribution", fontsize=14)
        axes[i, 1].set_xlabel(attr_x, fontsize=12)
        axes[i, 1].set_ylabel(attr_y, fontsize=12)
        axes[i, 1].set_xticks(range(len(x_values)))
        axes[i, 1].set_yticks(range(len(y_values)))
        axes[i, 1].set_xticklabels(x_values)
        axes[i, 1].set_yticklabels(y_values)

        # Add text annotations for ideal distribution
        for y_idx in range(len(y_values)):
            for x_idx in range(len(x_values)):
                if ideal_matrix[y_idx, x_idx] > 0:
                    text_color = 'white' if ideal_matrix[y_idx, x_idx] > 10 else 'black'
                    axes[i, 1].text(x_idx, y_idx, f"{ideal_matrix[y_idx, x_idx]:.1f}%",
                                    ha="center", va="center", color=text_color, fontweight='bold')

                    # Add gap annotation
                    gap = market_matrix[y_idx, x_idx] - ideal_matrix[y_idx, x_idx]
                    if abs(gap) > 2:  # Only show significant gaps
                        gap_text = f"{gap:+.1f}%" if gap != 0 else ""
                        gap_color = 'red' if gap < 0 else ('green' if gap > 0 else 'black')
                        axes[i, 1].text(x_idx, y_idx + 0.3, gap_text,
                                        ha="center", va="bottom", color=gap_color, fontsize=9, fontweight='bold')

        # Add colorbar
        plt.colorbar(im2, ax=axes[i, 1], label='% of Volume')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    plt.suptitle("Product 'Shelf' Visualization: Current vs. Ideal Distribution", fontsize=20, y=0.98)

    return fig


def create_radar_chart(location_data, validation_data):
    """
    Create a radar chart comparing alignment scores across attributes.

    Args:
        location_data (dict): Dictionary containing location-specific data
        validation_data (dict): Dictionary containing category C validation data

    Returns:
        matplotlib.figure.Figure: The figure containing the radar chart
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)

    # Define attributes and categories
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    categories = ['A', 'B', 'C', 'D']

    # Define alignment scores (from cat_c_validation.txt or other sources)
    alignment_scores = {
        'Kuwait': {
            'Flavor': 9.64,
            'Taste': 8.10,
            'Thickness': 5.03,
            'Length': 8.17,
            'Overall': 7.73
        },
        'Jeju': {
            'Flavor': 7.53,
            'Taste': 4.37,
            'Thickness': 5.82,
            'Length': 6.38,
            'Overall': 6.02
        }
    }

    # Update with validation data if available
    for location in validation_data:
        if 'cat_c_score' in validation_data[location] and validation_data[location]['cat_c_score'] is not None:
            alignment_scores[location]['Category C'] = validation_data[location]['cat_c_score']

    # Prepare data for radar chart
    N = len(attributes)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Plot data for each location
    locations = ['Kuwait', 'Jeju']
    colors = ['blue', 'red']

    for i, location in enumerate(locations):
        values = [alignment_scores[location][attr] for attr in attributes]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=location)
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    # Set chart properties
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'])
    ax.set_ylim(0, 10)

    plt.legend(loc='upper right')
    plt.title('Portfolio Alignment Score by Attribute', size=16)

    return fig


def create_pmi_share_visualization(location_data):
    """
    Create visualization showing PMI market share differences between Kuwait and Jeju.

    Args:
        location_data (dict): Dictionary containing location-specific data

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Extract market share data
    market_shares = {
        'Kuwait': 0.75,  # Approximate as per requirements
        'Jeju': 0.12  # Approximate as per requirements
    }

    # Try to get more accurate data from location_data if available
    for location in location_data:
        summary_data = location_data[location].get('Summary', None)
        if summary_data is not None:
            for _, row in summary_data.iterrows():
                if 'Market Share' in row['Category'] and 'PMI Share' in row['Metric']:
                    try:
                        market_shares[location] = float(row['Value'].strip('%')) / 100
                    except:
                        pass

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bar chart
    locations = list(market_shares.keys())
    shares = [market_shares[loc] * 100 for loc in locations]  # Convert to percentage

    colors = ['green', 'red']  # Green for Kuwait (good), red for Jeju (needs improvement)

    bars = ax.bar(locations, shares, color=colors)

    # Add value labels
    for bar, share in zip(bars, shares):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{share:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add labels and title
    ax.set_ylabel('Market Share (%)', fontsize=14)
    ax.set_title('PMI Market Share Comparison', fontsize=16)

    # Add annotations explaining the differences
    ax.annotate('Well-aligned portfolio\nstrong market share', xy=(0, shares[0] / 2),
                xytext=(0.3, 50), ha='center', fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    ax.annotate('Misaligned portfolio\nlow market share', xy=(1, shares[1] / 2),
                xytext=(0.7, 50), ha='center', fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Set y-axis limit
    ax.set_ylim(0, 100)

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig


def create_opportunity_map(location_data, location='Jeju'):
    """
    Create an opportunity map visualization for SKU recommendations.

    Args:
        location_data (dict): Dictionary containing location-specific data
        location (str): Location to create recommendations for (default: Jeju)

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Get attribute distribution data
    attr_data = {}
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']

    for attr in attributes:
        for data_type in location_data[location]:
            if attr.lower() in data_type.lower() and 'distribution' in data_type.lower():
                attr_data[attr] = location_data[location][data_type]
                break

    # Check if we have data for at least two attributes
    if len(attr_data) < 2:
        print(f"Not enough attribute data for opportunity map for {location}")
        return None

    # Select two attributes with largest gaps (largest alignment opportunity)
    attr_gaps = {}
    for attr, df in attr_data.items():
        if 'PMI_vs_Ideal_Gap' in df.columns:
            # Calculate sum of absolute gaps
            total_gap = df['PMI_vs_Ideal_Gap'].abs().sum()
            attr_gaps[attr] = total_gap

    # Sort attributes by gap magnitude
    sorted_attrs = sorted(attr_gaps.items(), key=lambda x: x[1], reverse=True)
    top_attrs = [attr for attr, _ in sorted_attrs[:2]]

    if len(top_attrs) < 2:
        top_attrs = attributes[:2]  # Fallback to first two attributes

    # Create opportunity map
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get data for the two attributes with largest gaps
    attr_x, attr_y = top_attrs
    df_x = attr_data[attr_x]
    df_y = attr_data[attr_y]

    # Get most underrepresented values for each attribute
    x_gaps = df_x[['PMI_vs_Ideal_Gap', attr_x]].sort_values('PMI_vs_Ideal_Gap')
    y_gaps = df_y[['PMI_vs_Ideal_Gap', attr_y]].sort_values('PMI_vs_Ideal_Gap')

    x_under = x_gaps[x_gaps['PMI_vs_Ideal_Gap'] < -5][attr_x].tolist()
    y_under = y_gaps[y_gaps['PMI_vs_Ideal_Gap'] < -5][attr_y].tolist()

    if not x_under:
        x_under = x_gaps.iloc[:2][attr_x].tolist()

    if not y_under:
        y_under = y_gaps.iloc[:2][attr_y].tolist()

    # Create a scatter plot of opportunity bubbles
    x_vals = []
    y_vals = []
    sizes = []
    annotations = []

    # Generate opportunity points
    for x_val in x_under:
        for y_val in y_under:
            x_gap = float(x_gaps[x_gaps[attr_x] == x_val]['PMI_vs_Ideal_Gap'].values[0])
            y_gap = float(y_gaps[y_gaps[attr_y] == y_val]['PMI_vs_Ideal_Gap'].values[0])

            # Calculate opportunity size (bubble size)
            opportunity_size = abs(x_gap * y_gap) * 5  # Scale for visibility

            x_vals.append(x_val)
            y_vals.append(y_val)
            sizes.append(opportunity_size)
            annotations.append(f"{attr_x}: {x_val}\n{attr_y}: {y_val}\nGap: {x_gap:.1f}%, {y_gap:.1f}%")

    # Create categorical x and y axes
    x_categories = df_x[attr_x].unique()
    y_categories = df_y[attr_y].unique()

    x_pos = [list(x_categories).index(x) for x in x_vals]
    y_pos = [list(y_categories).index(y) for y in y_vals]

    scatter = ax.scatter(x_pos, y_pos, s=sizes, alpha=0.6, color='red', edgecolors='black')

    # Add annotations
    for i, (x, y, annotation) in enumerate(zip(x_pos, y_pos, annotations)):
        ax.annotate(annotation, (x, y), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    # Set custom tick locations and labels
    ax.set_xticks(range(len(x_categories)))
    ax.set_xticklabels(x_categories)
    ax.set_yticks(range(len(y_categories)))
    ax.set_yticklabels(y_categories)

    # Add labels and title
    ax.set_xlabel(attr_x, fontsize=14)
    ax.set_ylabel(attr_y, fontsize=14)
    ax.set_title(f'Opportunity Map for {location} - SKU Recommendations', fontsize=16)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig


def create_portfolio_optimization_dashboard(location_data_files, comparison_files, validation_file, output_dir=None):
    """
    Generate a comprehensive portfolio optimization dashboard.

    Args:
        location_data_files (dict): Dictionary mapping locations to their data files
        comparison_files (dict): Dictionary mapping attributes to comparison file paths
        validation_file (str): Path to the validation data file
        output_dir (str, optional): Directory to save outputs

    Returns:
        dict: Mapping of visualization names to figure objects
    """
    # Load data
    location_data = load_location_data(location_data_files)
    comparison_data = load_comparison_data(comparison_files)
    validation_data = load_validation_data(validation_file)

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    visualizations = {}

    # 1. Create comparative portfolio grid
    print("Creating comparative portfolio grid...")
    grid_vis = create_comparative_portfolio_grid(location_data)
    if grid_vis:
        visualizations["portfolio_grid"] = grid_vis
        if output_dir:
            grid_vis.savefig(os.path.join(output_dir, "portfolio_grid.png"), dpi=300, bbox_inches='tight')

    # 2. Create product shelf heatmap
    print("Creating product shelf heatmap...")
    shelf_vis = create_product_shelf_heatmap(location_data)
    if shelf_vis:
        visualizations["product_shelf"] = shelf_vis
        if output_dir:
            shelf_vis.savefig(os.path.join(output_dir, "product_shelf.png"), dpi=300, bbox_inches='tight')

    # 3. Create radar chart
    print("Creating radar chart...")
    radar_vis = create_radar_chart(location_data, validation_data)
    if radar_vis:
        visualizations["radar_chart"] = radar_vis
        if output_dir:
            radar_vis.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')

    # 4. Create market share visualization
    print("Creating market share visualization...")
    share_vis = create_pmi_share_visualization(location_data)
    if share_vis:
        visualizations["market_share"] = share_vis
        if output_dir:
            share_vis.savefig(os.path.join(output_dir, "market_share.png"), dpi=300, bbox_inches='tight')

    # 5. Create opportunity map for Jeju
    print("Creating opportunity map for Jeju...")
    opp_vis = create_opportunity_map(location_data, 'Jeju')
    if opp_vis:
        visualizations["opportunity_map"] = opp_vis
        if output_dir:
            opp_vis.savefig(os.path.join(output_dir, "opportunity_map.png"), dpi=300, bbox_inches='tight')

    print(f"Created {len(visualizations)} visualizations")
    return visualizations


def main():
    """Main function to run the portfolio optimization visualization script"""
    # Define data directories
    data_dir = Path('./locations_data')
    output_dir = Path('./visualization_results')

    # Set up file paths for Kuwait data
    kuwait_files = {
        'Flavor_Distribution': data_dir / 'Kuwait_product_analysis_Flavor_Distribution.csv',
        'Taste_Distribution': data_dir / 'Kuwait_product_analysis_Taste_Distribution.csv',
        'Thickness_Distribution': data_dir / 'Kuwait_product_analysis_Thickness_Distribution.csv',
        'Length_Distribution': data_dir / 'Kuwait_product_analysis_Length_Distribution.csv',
        'PMI_Products': data_dir / 'Kuwait_product_analysis_PMI_Products.csv',
        'Top_90pct_Products': data_dir / 'Kuwait_product_analysis_Top_90pct_Products.csv',
        'Summary': data_dir / 'Kuwait_product_analysis_Summary.csv',
        'Passenger_Distribution': data_dir / 'Kuwait_product_analysis_Passenger_Distribution.csv'
    }

    # Set up file paths for Jeju data
    jeju_files = {
        'Flavor_Distribution': data_dir / 'jeju_product_analysis_Flavor_Distribution.csv',
        'Taste_Distribution': data_dir / 'jeju_product_analysis_Taste_Distribution.csv',
        'Thickness_Distribution': data_dir / 'jeju_product_analysis_Thickness_Distribution.csv',
        'Length_Distribution': data_dir / 'jeju_product_analysis_Length_Distribution.csv',
        'PMI_Products': data_dir / 'jeju_product_analysis_PMI_Products.csv',
        'Top_90pct_Products': data_dir / 'jeju_product_analysis_Top_90pct_Products.csv',
        'Summary': data_dir / 'jeju_product_analysis_Summary.csv'
    }

    # Set up file paths for comparison data
    comparison_files = {
        'Flavor': data_dir / 'kuwait_jeju_attribute_analysis_Flavor_Distribution.csv',
        'Taste': data_dir / 'kuwait_jeju_attribute_analysis_Taste_Distribution.csv',
        'Thickness': data_dir / 'kuwait_jeju_attribute_analysis_Thickness_Distribution.csv',
        'Length': data_dir / 'kuwait_jeju_attribute_analysis_Length_Distribution.csv'
    }

    # Path to validation file
    validation_file = 'cat_c_validation.txt'

    # Organize data files
    location_data_files = {
        'Kuwait': kuwait_files,
        'Jeju': jeju_files
    }

    # Generate dashboard
    visualizations = create_portfolio_optimization_dashboard(
        location_data_files,
        comparison_files,
        validation_file,
        str(output_dir)
    )

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
