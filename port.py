import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import os


def load_data(data_dir):
    """
    Load data files from the specified directory

    Args:
        data_dir (str): Path to data directory

    Returns:
        dict: Dictionary containing loaded dataframes
    """
    data = {
        'kuwait': {},
        'jeju': {}
    }

    # Load Kuwait data
    kuwait_files = {
        'flavor': 'Kuwait_product_analysis_Flavor_Distribution.csv',
        'taste': 'Kuwait_product_analysis_Taste_Distribution.csv',
        'thickness': 'Kuwait_product_analysis_Thickness_Distribution.csv',
        'length': 'Kuwait_product_analysis_Length_Distribution.csv',
        'summary': 'Kuwait_product_analysis_Summary.csv',
        'products': 'Kuwait_product_analysis_PMI_Products.csv',
        'top_products': 'Kuwait_product_analysis_Top_90pct_Products.csv'
    }

    # Load Jeju data
    jeju_files = {
        'flavor': 'jeju_product_analysis_Flavor_Distribution.csv',
        'taste': 'jeju_product_analysis_Taste_Distribution.csv',
        'thickness': 'jeju_product_analysis_Thickness_Distribution.csv',
        'length': 'jeju_product_analysis_Length_Distribution.csv',
        'summary': 'jeju_product_analysis_Summary.csv',
        'products': 'jeju_product_analysis_PMI_Products.csv',
        'top_products': 'jeju_product_analysis_Top_90pct_Products.csv'
    }

    # Load comparison data
    comparison_files = {
        'flavor': 'kuwait_jeju_attribute_analysis_Flavor_Distribution.csv',
        'taste': 'kuwait_jeju_attribute_analysis_Taste_Distribution.csv',
        'thickness': 'kuwait_jeju_attribute_analysis_Thickness_Distribution.csv',
        'length': 'kuwait_jeju_attribute_analysis_Length_Distribution.csv',
        'gaps': {
            'kuwait': {
                'flavor': 'kuwait_jeju_attribute_analysis_Kuwait_Flavor_Gaps.csv',
                'taste': 'kuwait_jeju_attribute_analysis_Kuwait_Taste_Gaps.csv',
                'thickness': 'kuwait_jeju_attribute_analysis_Kuwait_Thickness_Gaps.csv',
                'length': 'kuwait_jeju_attribute_analysis_Kuwait_Length_Gaps.csv'
            },
            'jeju': {
                'flavor': 'kuwait_jeju_attribute_analysis_Jeju_Flavor_Gaps.csv',
                'taste': 'kuwait_jeju_attribute_analysis_Jeju_Taste_Gaps.csv',
                'thickness': 'kuwait_jeju_attribute_analysis_Jeju_Thickness_Gaps.csv',
                'length': 'kuwait_jeju_attribute_analysis_Jeju_Length_Gaps.csv'
            }
        }
    }

    # Load Kuwait data
    for key, filename in kuwait_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data['kuwait'][key] = pd.read_csv(filepath)

    # Load Jeju data
    for key, filename in jeju_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data['jeju'][key] = pd.read_csv(filepath)

    # Load comparison data
    data['comparison'] = {}
    for key, filename in comparison_files.items():
        if key != 'gaps':
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                data['comparison'][key] = pd.read_csv(filepath)

    # Load gap data
    data['gaps'] = {'kuwait': {}, 'jeju': {}}
    for location in ['kuwait', 'jeju']:
        for key, filename in comparison_files['gaps'][location].items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                data['gaps'][location][key] = pd.read_csv(filepath)

    # Load comparison summary
    summary_file = os.path.join(data_dir, 'kuwait_jeju_comparison_summary.csv')
    if os.path.exists(summary_file):
        data['comparison_summary'] = pd.read_csv(summary_file)

    return data


def create_attribute_matrix(location_data, attribute, location, pmi_only=False):
    """
    Create a matrix of product counts for a given attribute by TMO

    Args:
        location_data (dict): Dictionary containing location data
        attribute (str): Attribute to analyze (flavor, taste, thickness, length)
        location (str): Location to analyze (kuwait or jeju)
        pmi_only (bool): Whether to include only PMI products

    Returns:
        tuple: (matrix, row_labels, col_labels) for heatmap visualization
    """
    # Get product data
    if pmi_only:
        products_df = location_data[location]['products']
    else:
        products_df = location_data[location]['top_products']

    # Get attribute values
    attribute_df = location_data[location][attribute]

    # Create attribute matrix
    attribute_values = attribute_df[attribute].unique()

    if pmi_only:
        # PMI only matrix
        matrix = np.zeros(len(attribute_values))

        for i, attr_val in enumerate(attribute_values):
            count = len(products_df[products_df[attribute] == attr_val])
            matrix[i] = count

        return matrix, attribute_values, ['PMI']
    else:
        # TMO comparison matrix
        tmos = products_df['TMO'].unique()
        matrix = np.zeros((len(attribute_values), len(tmos)))

        for i, attr_val in enumerate(attribute_values):
            for j, tmo in enumerate(tmos):
                count = len(products_df[(products_df[attribute] == attr_val) &
                                        (products_df['TMO'] == tmo)])
                matrix[i, j] = count

        return matrix, attribute_values, tmos


def create_attribute_gap_matrix(location_data, attribute, location):
    """
    Create a matrix of gaps between actual and ideal distribution for an attribute

    Args:
        location_data (dict): Dictionary containing location data
        attribute (str): Attribute to analyze (flavor, taste, thickness, length)
        location (str): Location to analyze (kuwait or jeju)

    Returns:
        numpy.ndarray: Matrix of gaps
    """
    # Get gap data
    if location in location_data['gaps'] and attribute in location_data['gaps'][location]:
        gap_df = location_data['gaps'][location][attribute]
    else:
        # Fallback to attribute distribution data
        gap_df = location_data[location][attribute]

    # Extract gaps
    if 'Gap' in gap_df.columns:
        gaps = gap_df['Gap'].values
    elif 'Market_vs_Ideal_Gap' in gap_df.columns:
        gaps = gap_df['Market_vs_Ideal_Gap'].values
    else:
        # Compute gaps
        gaps = gap_df['Actual'].values - gap_df['Ideal'].values

    return gaps


def get_attribute_column(df, attribute):
    """
    Helper function to find the correct column name for an attribute
    """
    if attribute in df.columns:
        return attribute

    # Try capitalized version
    capitalized = attribute.capitalize()
    if capitalized in df.columns:
        return capitalized

    # Try other variations
    possible_columns = ['Category', 'Type', 'Name', 'Value']
    for col in possible_columns:
        if col in df.columns:
            return col

    # Return original as fallback
    return attribute


def create_attribute_grid(location_data, primary_attr, secondary_attr, location):
    """
    Create a grid visualization showing the distribution of products across two attributes
    """
    # Get product data
    products_df = location_data[location]['top_products']

    # Get attribute dataframes
    primary_df = location_data[location][primary_attr]
    secondary_df = location_data[location][secondary_attr]

    # Find correct column names
    primary_col = get_attribute_column(primary_df, primary_attr)
    secondary_col = get_attribute_column(secondary_df, secondary_attr)

    # Get attribute values
    primary_values = primary_df[primary_col].unique()
    secondary_values = secondary_df[secondary_col].unique()

    # Get correct column names in products dataframe
    products_primary_col = get_attribute_column(products_df, primary_attr)
    products_secondary_col = get_attribute_column(products_df, secondary_attr)

    # Create grid
    grid = np.zeros((len(primary_values), len(secondary_values)))

    # Fill grid with product counts
    for i, p_val in enumerate(primary_values):
        for j, s_val in enumerate(secondary_values):
            count = len(products_df[(products_df[products_primary_col] == p_val) &
                                    (products_df[products_secondary_col] == s_val)])
            grid[i, j] = count

    return grid, primary_values, secondary_values


def create_ideal_distribution_grid(location_data, primary_attr, secondary_attr, location):
    """
    Create a grid visualization showing the ideal distribution of products across two attributes

    Args:
        location_data (dict): Dictionary containing location data
        primary_attr (str): Primary attribute for rows
        secondary_attr (str): Secondary attribute for columns
        location (str): Location to analyze (kuwait or jeju)

    Returns:
        tuple: (matrix, row_labels, col_labels) for heatmap visualization
    """
    # Get attribute data
    primary_df = location_data[location][primary_attr]
    secondary_df = location_data[location][secondary_attr]

    # Find correct column names
    primary_col = get_attribute_column(primary_df, primary_attr)
    secondary_col = get_attribute_column(secondary_df, secondary_attr)

    # Get attribute values
    primary_values = primary_df[primary_col].unique()
    secondary_values = secondary_df[secondary_col].unique()

    # Create grid for ideal distribution
    grid = np.zeros((len(primary_values), len(secondary_values)))

    # This is a simplification - in reality this would need to be derived from passenger preference data
    # For now, we'll use a proportional distribution based on individual ideal percentages
    if 'Ideal_Percentage' in primary_df.columns and 'Ideal_Percentage' in secondary_df.columns:
        primary_ideal = primary_df.set_index(primary_col)['Ideal_Percentage'].to_dict()
        secondary_ideal = secondary_df.set_index(secondary_col)['Ideal_Percentage'].to_dict()

        for i, p_val in enumerate(primary_values):
            for j, s_val in enumerate(secondary_values):
                # Joint probability - simplified assumption of independence
                grid[i, j] = (primary_ideal.get(p_val, 0) * secondary_ideal.get(s_val, 0)) / 100

    return grid, primary_values, secondary_values


def create_gap_grid(actual_grid, ideal_grid):
    """
    Create a grid showing the gap between actual and ideal distribution

    Args:
        actual_grid (numpy.ndarray): Matrix of actual distribution
        ideal_grid (numpy.ndarray): Matrix of ideal distribution

    Returns:
        numpy.ndarray: Matrix of gaps
    """
    # Normalize grids for comparison if they have different scales
    if actual_grid.sum() > 0:
        actual_normalized = actual_grid / actual_grid.sum() * 100
    else:
        actual_normalized = actual_grid

    if ideal_grid.sum() > 0:
        ideal_normalized = ideal_grid / ideal_grid.sum() * 100
    else:
        ideal_normalized = ideal_grid

    # Calculate gaps
    gap_grid = actual_normalized - ideal_normalized

    return gap_grid


def plot_attribute_distribution(location_data, attribute, fig, ax, location, title=None):
    """
    Plot attribute distribution showing actual vs ideal percentages

    Args:
        location_data (dict): Dictionary containing location data
        attribute (str): Attribute to plot
        fig (matplotlib.figure.Figure): Figure to plot on
        ax (matplotlib.axes.Axes): Axes to plot on
        location (str): Location to analyze (kuwait or jeju)
        title (str, optional): Custom title
    """
    # Get attribute data
    attr_df = location_data[location][attribute]

    # Print debug info to check structure
    print(f"Columns available in {location}_{attribute}: {attr_df.columns.tolist()}")

    # Extract data - look for likely category column names
    category_column = attribute
    if attribute not in attr_df.columns:
        # Try to find a column that might contain categories
        possible_columns = ['Category', 'Type', attribute.capitalize(), 'Name', 'Value']
        for col in possible_columns:
            if col in attr_df.columns:
                category_column = col
                break

    # Extract values
    x = attr_df[category_column].values

    # Check for columns
    if 'Volume_Percentage' in attr_df.columns and 'Ideal_Percentage' in attr_df.columns:
        actual = attr_df['Volume_Percentage'].values
        ideal = attr_df['Ideal_Percentage'].values
        pmi = attr_df['PMI_Volume_Percentage'].values if 'PMI_Volume_Percentage' in attr_df.columns else None
    elif 'Actual' in attr_df.columns and 'Ideal' in attr_df.columns:
        actual = attr_df['Actual'].values
        ideal = attr_df['Ideal'].values
        pmi = None
    else:
        return

    # Set width for bars
    width = 0.3
    x_pos = np.arange(len(x))

    # Plot ideal distribution
    ax.bar(x_pos - width, ideal, width, color='green', alpha=0.7, label='Ideal')

    # Plot actual distribution
    ax.bar(x_pos, actual, width, color='blue', alpha=0.7, label='Market')

    # Plot PMI distribution if available
    if pmi is not None:
        ax.bar(x_pos + width, pmi, width, color='red', alpha=0.7, label='PMI')

    # Calculate and add gap annotations
    for i, (a, b, label) in enumerate(zip(actual, ideal, x)):
        gap = a - b
        color = 'red' if gap < -5 else ('green' if gap > 5 else 'black')
        ax.annotate(f"{gap:.1f}",
                    xy=(i, max(a, b) + 1),
                    ha='center', va='bottom',
                    color=color, fontweight='bold')

    # Add labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)')

    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{location.capitalize()} - {attribute} Distribution")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_portfolio_grid(location_data, primary_attr, secondary_attr, location, fig, ax, title=None):
    """
    Plot a heatmap grid showing product distribution across two attributes

    Args:
        location_data (dict): Dictionary containing location data
        primary_attr (str): Primary attribute for y-axis
        secondary_attr (str): Secondary attribute for x-axis
        location (str): Location to analyze (kuwait or jeju)
        fig (matplotlib.figure.Figure): Figure to plot on
        ax (matplotlib.axes.Axes): Axes to plot on
        title (str, optional): Custom title
    """
    # Get actual distribution grid
    actual_grid, primary_values, secondary_values = create_attribute_grid(
        location_data, primary_attr, secondary_attr, location)

    # Get ideal distribution grid
    ideal_grid, _, _ = create_ideal_distribution_grid(
        location_data, primary_attr, secondary_attr, location)

    # Calculate gap grid
    gap_grid = create_gap_grid(actual_grid, ideal_grid)

    # Create custom colormap - blue for negative gaps, red for positive gaps
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

    # Plot heatmap
    im = ax.imshow(gap_grid, cmap=cmap, vmin=-20, vmax=20)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Gap (Actual - Ideal %)')

    # Add value annotations
    for i in range(len(primary_values)):
        for j in range(len(secondary_values)):
            text = ax.text(j, i, f"{gap_grid[i, j]:.1f}\n({actual_grid[i, j]:.0f})",
                           ha="center", va="center", color="black",
                           fontsize=8, fontweight="bold",
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=1))

    # Set ticks and labels
    ax.set_xticks(np.arange(len(secondary_values)))
    ax.set_yticks(np.arange(len(primary_values)))
    ax.set_xticklabels(secondary_values)
    ax.set_yticklabels(primary_values)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{location.capitalize()} - {primary_attr} vs {secondary_attr} Portfolio")

    # Add axis labels
    ax.set_xlabel(secondary_attr)
    ax.set_ylabel(primary_attr)


def create_portfolio_visualization(data_dir, output_dir=None):
    """
    Create visualization for portfolio optimization

    Args:
        data_dir (str): Path to data directory
        output_dir (str, optional): Path to save output visualizations
    """
    # Load data
    data = load_data(data_dir)

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define locations
    locations = ['kuwait', 'jeju']

    # Create side-by-side attribute distribution visualizations
    attributes = ['flavor', 'taste', 'thickness', 'length']

    for attribute in attributes:
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Kuwait and Jeju distributions
        for i, location in enumerate(locations):
            if attribute in data[location]:
                plot_attribute_distribution(data, attribute, fig, axes[i], location)

        # Add overall title
        fig.suptitle(f"{attribute.capitalize()} Distribution: Kuwait vs Jeju", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure if output directory specified
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"{attribute}_distribution.png"), dpi=300, bbox_inches='tight')

    # Create portfolio grid visualizations
    attribute_pairs = [
        ('flavor', 'taste'),
        ('thickness', 'length')
    ]

    for primary_attr, secondary_attr in attribute_pairs:
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Kuwait and Jeju portfolio grids
        for i, location in enumerate(locations):
            if primary_attr in data[location] and secondary_attr in data[location]:
                plot_portfolio_grid(data, primary_attr, secondary_attr, location, fig, axes[i])

        # Add overall title
        fig.suptitle(f"Portfolio Grid: {primary_attr.capitalize()} vs {secondary_attr.capitalize()}", fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save figure if output directory specified
        if output_dir:
            fig.savefig(os.path.join(output_dir, f"{primary_attr}_{secondary_attr}_portfolio.png"), dpi=300,
                        bbox_inches='tight')

    # Create combined shelf visualization
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Define primary attributes for visualization
    primary_attrs = ['flavor', 'taste']
    secondary_attrs = ['thickness', 'length']

    # Create subplots
    for i, location in enumerate(locations):
        for j, (primary_attr, secondary_attr) in enumerate(zip(primary_attrs, secondary_attrs)):
            ax = plt.subplot(gs[i, j])

            if primary_attr in data[location] and secondary_attr in data[location]:
                plot_portfolio_grid(
                    data, primary_attr, secondary_attr, location, fig, ax,
                    title=f"{location.capitalize()} - {primary_attr.capitalize()} vs {secondary_attr.capitalize()}")

    # Add overall title
    fig.suptitle("Portfolio Alignment: Kuwait (Well-Aligned) vs Jeju (Misaligned)", fontsize=20)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if output directory specified
    if output_dir:
        fig.savefig(os.path.join(output_dir, "combined_portfolio_visualization.png"), dpi=300, bbox_inches='tight')

    # Create market share visualization
    if 'comparison_summary' in data:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract market share data (from summary or preset values)
        market_shares = {
            'kuwait': 0.75,  # Approximate as per documentation
            'jeju': 0.12  # Approximate as per documentation
        }

        # Try to extract from data if available
        if 'comparison_summary' in data:
            df = data['comparison_summary']
            if 'PMI Volume' in df.columns and 'Total Volume' in df.columns:
                for i, row in df.iterrows():
                    location = row['Location'].lower()
                    if location in market_shares:
                        market_shares[location] = row['PMI Volume'] / row['Total Volume']

        # Plot bar chart
        colors = ['green', 'red']
        bars = ax.bar(locations, [market_shares[loc] * 100 for loc in locations], color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Add annotations
        ax.annotate('Well-aligned portfolio\nStrong market share',
                    xy=(0, market_shares['kuwait'] * 50),
                    xytext=(0.3, 50), ha='center', fontsize=12,
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        ax.annotate('Misaligned portfolio\nLow market share',
                    xy=(1, market_shares['jeju'] * 50),
                    xytext=(0.7, 50), ha='center', fontsize=12,
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        # Add labels and title
        ax.set_ylabel('Market Share (%)')
        ax.set_title('PMI Market Share Comparison')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Save figure if output directory specified
        if output_dir:
            fig.savefig(os.path.join(output_dir, "market_share_comparison.png"), dpi=300, bbox_inches='tight')

    print(f"Visualizations created successfully{' and saved to ' + output_dir if output_dir else ''}")


def create_streamlit_app():
    """
    Create a Streamlit app for interactive visualization
    """
    import streamlit as st

    st.title("PMI Portfolio Optimization Visualization")

    # Set data directory
    data_dir = st.sidebar.text_input("Data Directory", value="./locations_data")

    # Load data
    data = load_data(data_dir)

    # Select visualization type
    viz_type = st.sidebar.selectbox(
        "Visualization Type",
        ["Attribute Distribution", "Portfolio Grid", "Market Share"]
    )

    if viz_type == "Attribute Distribution":
        # Select attribute
        attribute = st.sidebar.selectbox("Attribute", ["flavor", "taste", "thickness", "length"])

        # Create columns for Kuwait and Jeju
        col1, col2 = st.columns(2)

        # Create figures
        for i, (location, col) in enumerate(zip(['kuwait', 'jeju'], [col1, col2])):
            if attribute in data[location]:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_attribute_distribution(data, attribute, fig, ax, location)
                col.pyplot(fig)

                # Add market share information
                if 'summary' in data[location]:
                    summary = data[location]['summary']
                    if isinstance(summary, pd.DataFrame):
                        # Extract market share if available
                        market_share = None
                        for _, row in summary.iterrows():
                            if 'Market Share' in row['Category'] and 'PMI Share' in row['Metric']:
                                market_share = row['Value']
                                break

                        if market_share:
                            col.metric(f"{location.capitalize()} Market Share", market_share)

    elif viz_type == "Portfolio Grid":
        # Select attributes
        primary_attr = st.sidebar.selectbox("Primary Attribute", ["flavor", "taste"])
        secondary_attr = st.sidebar.selectbox("Secondary Attribute", ["thickness", "length"])

        # Create columns for Kuwait and Jeju
        col1, col2 = st.columns(2)

        # Create figures
        for i, (location, col) in enumerate(zip(['kuwait', 'jeju'], [col1, col2])):
            if primary_attr in data[location] and secondary_attr in data[location]:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_portfolio_grid(data, primary_attr, secondary_attr, location, fig, ax)
                col.pyplot(fig)

    elif viz_type == "Market Share":
        # Extract market share data
        market_shares = {
            'kuwait': 0.75,  # Approximate as per documentation
            'jeju': 0.12  # Approximate as per documentation
        }

        # Try to extract from data if available
        if 'comparison_summary' in data:
            df = data['comparison_summary']
            if 'PMI Volume' in df.columns and 'Total Volume' in df.columns:
                for i, row in df.iterrows():
                    location = row['Location'].lower()
                    if location in market_shares:
                        market_shares[location] = row['PMI Volume'] / row['Total Volume']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bar chart
        colors = ['green', 'red']
        bars = ax.bar(['kuwait', 'jeju'], [market_shares[loc] * 100 for loc in ['kuwait', 'jeju']], color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Add labels and title
        ax.set_ylabel('Market Share (%)')
        ax.set_title('PMI Market Share Comparison')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)

        # Add explanation
        st.markdown("""
        ### Market Share Analysis

        The visualization above shows a clear correlation between portfolio alignment and market share performance:

        - **Kuwait (75%)**: Well-aligned portfolio with strong market share
        - **Jeju (12%)**: Misaligned portfolio with low market share

        This demonstrates that better alignment of product attributes with consumer preferences leads to higher market penetration.
        """)


if __name__ == "__main__":
    # Define data directory
    data_dir = "./locations_data"

    # Define output directory
    output_dir = "./visualization_results"

    # Create visualizations
    create_portfolio_visualization(data_dir, output_dir)