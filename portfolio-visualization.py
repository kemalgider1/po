import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

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


def create_attribute_grid_visualization(comp_df, attribute, location1='Kuwait', location2='Jeju'):
    """
    Create a grid visualization comparing attribute distribution for two locations.

    Args:
        comp_df (DataFrame): Comparison data
        attribute (str): Attribute to visualize (Flavor, Taste, Thickness, Length)
        location1 (str): First location to compare (should be Kuwait - well aligned)
        location2 (str): Second location to compare (should be Jeju - misaligned)

    Returns:
        matplotlib.figure.Figure: The figure containing the grid visualization
    """
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Extract attribute values and metrics for each location
    locations = [location1, location2]
    location_data = {}

    # Process comparison data to extract metrics for each location and attribute value
    for loc in locations:
        loc_values = []
        actual_vals = []
        pmi_vals = []
        ideal_vals = []
        gaps = []

        # Find rows for this location in the comparison data
        loc_rows = comp_df[comp_df['Location'] == loc]

        # Find the section for this attribute
        attr_section = None
        for i, row in loc_rows.iterrows():
            if pd.notna(row[0]) and attribute in str(row[0]):
                attr_section = i
                break

        # If we found the section, extract data for the next several rows
        if attr_section is not None:
            idx = attr_section + 1
            while idx < len(comp_df) and pd.isna(comp_df.iloc[idx, 0]):
                row = comp_df.iloc[idx]

                # Extract attribute value (first column if not NaN)
                attr_value = str(row.iloc[0]) if pd.notna(row.iloc[0]) else None

                if attr_value and attr_value.strip():
                    # Extract metrics for this location
                    # Columns should be: [Value, Actual%, Ideal%, Gap%, VOL%, PMI%, Ideal%, Gap%]
                    loc_values.append(attr_value)
                    actual_vals.append(float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0)
                    ideal_vals.append(float(row.iloc[2]) if pd.notna(row.iloc[2]) else 0)
                    gaps.append(float(row.iloc[3]) if pd.notna(row.iloc[3]) else 0)
                    pmi_vals.append(float(row.iloc[5]) if pd.notna(row.iloc[5]) and len(row) > 5 else 0)

                idx += 1

        location_data[loc] = {
            'values': loc_values,
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

        loc_values = location_data[loc]['values']
        actual_vals = location_data[loc]['actual']
        ideal_vals = location_data[loc]['ideal']
        gaps = location_data[loc]['gaps']
        pmi_vals = location_data[loc]['pmi']

        if not loc_values:
            ax.text(0.5, 0.5, f"No data for {loc}", ha='center', va='center', fontsize=12)
            continue

        # Calculate grid layout
        n_values = len(loc_values)
        grid_cols = min(4, n_values)
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
            norm_gap = (gap + 100) / 200
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
            if pd.notna(attr_df.iloc[i, 0]) and any(attr in str(attr_df.iloc[i, 0]) for attr in ['Flavor', 'Taste', 'Thickness', 'Length']):
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
                
                actual_vals.append(float(row.iloc[actual_idx]) if actual_idx >= 0 and pd.notna(row.iloc[actual_idx]) else 0)
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
        hm = sns.heatmap(data, annot=True, fmt='.1f', cmap=cmap, 
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
                    if pd.notna(attr_df.iloc[row_idx, 0]) and any(attr in str(attr_df.iloc[row_idx, 0]) for attr in ['Flavor', 'Taste', 'Thickness', 'Length']):
                        break
                    
                    # Extract values
                    row = attr_df.iloc[row_idx]
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        # Find columns with actual, ideal, and gap values
                        actual_col = 1  # Assuming fixed positions based on observed patterns
                        ideal_col = 2
                        gap_col = 3
                        
                        if pd.notna(row.iloc[actual_col]) and pd.notna(row.iloc[ideal_col]):
                            attr_values.append(str(row.iloc[0]))
                            actual_vals.append(float(row.iloc[actual_col]))
                            ideal_vals.append(float(row.iloc[ideal_col]))
                            gaps.append(float(row.iloc[gap_col]) if pd.notna(row.iloc[gap_col]) else ideal_vals[-1] - actual_vals[-1])
            
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
            actual_bars = ax.bar(x - width/2, actual_vals, width, label='Actual', color=cmap_actual(0.6))
            
            # Plot ideal distribution
            ideal_bars = ax.bar(x + width/2, ideal_vals, width, label='Ideal', color=cmap_ideal(0.6))
            
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
                    grid[a2_idx, a1_idx] = np.log1p(products['DF_Vol'].sum()) if 'DF_Vol' in products.columns else len(products)
                    pmi_grid[a2_idx, a1_idx] = np.log1p(pmi_products['DF_Vol'].sum()) if 'DF_Vol' in pmi_products.columns and not pmi_products.empty else 0
                    comp_grid[a2_idx, a1_idx] = np.log1p(comp_products['DF_Vol'].sum()) if 'DF_Vol' in comp_products.columns and not comp_products.empty else 0
        
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
                    total_vol = products['DF_Vol'].sum() if 'DF_Vol' in products.columns else 0
                    pmi_vol = pmi_products['DF_Vol'].sum() if 'DF_Vol' in pmi_products.columns and not pmi_products.empty else 0
                    comp_vol = comp_products['DF_Vol'].sum() if 'DF_Vol' in comp_products.columns and not comp_products.empty else 0
                    
                    # Add PMI indicator (circles)
                    if not pmi_products.empty:
                        # Scale size by volume proportion
                        size = 2000 * (pmi_vol / total_vol) if total_vol > 0 else 0
                        ax.scatter(a1_idx, a2_idx, s=size, color=pmi_color, alpha=0.7, 
                                 edgecolors='black', linewidths=0.5, zorder=10)
                    
                    # Add competitor indicator (squares)
                    if not comp_products.empty:
                        # Scale size by volume proportion
                        size = 2000 * (comp_vol / total_vol) if total_vol > 0 else 0
                        if size > 0:
                            square_size = np.sqrt(size / 100)
                            rect = plt.Rectangle((a1_idx - square_size/2, a2_idx - square_size/2), 
                                                square_size, square_size, 
                                                color=comp_color, alpha=0.7, 
                                                edgecolor='black', linewidth=0.5, zorder=5)
                            ax.add_patch(rect)
                    
                    # Add product count and volume text
                    text = f"PMI: {len(pmi_products)}"
                    if 'DF_Vol' in pmi_products.columns and not pmi_products.empty:
                        text += f"\n({pmi_vol/1000:.1f}k)"
                    
                    text += f"\nComp: {len(comp_products)}"
                    if 'DF_Vol' in comp_products.columns and not comp_products.empty:
                        text += f"\n({comp_vol/1000:.1f}k)"
                    
                    ax.text(a1_idx, a2_idx, text, ha='center', va='center', fontsize=7,
                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(attr1_values)))
        ax.set_yticks(np.arange(len(attr2_values)))
        ax.set_xticklabels(attr1_values)
        ax.set_yticklabels(attr2_values)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title and labels
        ax.set_title(f"{location} - Product Portfolio Distribution", fontsize=14)
        ax.set_xlabel(attribute1)
        ax.set_ylabel(attribute2)
        
        # Add legend
        pmi_patch = mpatches.Patch(color=pmi_color, label='PMI Products')
        comp_patch = mpatches.Patch(color=comp_color, label='Competitor Products')
        ax.legend(handles=[pmi_patch, comp_patch], loc='upper right')
        
        # Add market share info if available
        if 'DF_Vol' in df.columns:
            pmi_share = df[df['TMO'] == 'PMI']['DF_Vol'].sum() / df['DF_Vol'].sum() * 100
            ax.text(0.02, 0.02, f"PMI Market Share: {pmi_share:.1f}%", 
                  transform=ax.transAxes, fontsize=12, fontweight='bold',
                  va='bottom', ha='left', 
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.suptitle(f"Product Portfolio 'Shelf' Visualization: {attribute1} × {attribute2}", fontsize=16, y=1.02)
    return fig

def create_radar_chart(kw_attr_df, jj_attr_df):
    """
    Create a radar chart comparing alignment scores across attributes.
    
    Args:
        kw_attr_df (DataFrame): Kuwait attribute analysis data
        jj_attr_df (DataFrame): Jeju attribute analysis data
    
    Returns:
        matplotlib.figure.Figure: The figure containing the radar chart
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    
    # Calculate alignment scores for each attribute and location
    alignment_scores = {
        'Kuwait': {},
        'Jeju': {}
    }
    
    for location, attr_df in [('Kuwait', kw_attr_df), ('Jeju', jj_attr_df)]:
        for attribute in attributes:
            # Find the section for this attribute
            found_section = False
            attr_values = []
            actual_vals = []
            ideal_vals = []
            gaps = []
            
            for row_idx in range(len(attr_df)):
                if pd.notna(attr_df.iloc[row_idx, 0]) and attribute in str(attr_df.iloc[row_idx, 0]):
                    found_section = True
                    continue
                
                if found_section:
                    # Stop when we hit another attribute section
                    if pd.notna(attr_df.iloc[row_idx, 0]) and any(attr in str(attr_df.iloc[row_idx, 0]) for attr in attributes):
                        break
                    
                    # Extract values
                    row = attr_df.iloc[row_idx]
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        # Find columns with actual, ideal, and gap values
                        actual_col = 1  # Assuming fixed positions based on observed patterns
                        ideal_col = 2
                        gap_col = 3
                        
                        if pd.notna(row.iloc[actual_col]) and pd.notna(row.iloc[ideal_col]):
                            attr_values.append(str(row.iloc[0]))
                            actual_vals.append(float(row.iloc[actual_col]))
                            ideal_vals.append(float(row.iloc[ideal_col]))
                            gaps.append(float(row.iloc[gap_col]) if pd.notna(row.iloc[gap_col]) else ideal_vals[-1] - actual_vals[-1])
            
            # Calculate alignment score based on gaps
            if gaps:
                # Alignment score is inversely proportional to sum of absolute gaps
                # Scale to 0-10 range
                alignment_score = 10 - min(10, sum(abs(g) for g in gaps) / 10)
                alignment_scores[location][attribute] = alignment_score
            else:
                alignment_scores[location][attribute] = 0
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Set number of attributes
    N = len(attributes)
    
    # Set angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    
    # Draw axis lines for each angle
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw labels at appropriate locations
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
    plt.ylim(0, 10)
    
    # Plot data
    for location, color in [('Kuwait', 'blue'), ('Jeju', 'red')]:
        values = [alignment_scores[location].get(attr, 0) for attr in attributes]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=location)
        ax.fill(angles, values, color=color, alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Portfolio Alignment Radar Chart: Kuwait vs. Jeju', size=15, y=1.1)
    
    return fig


def generate_portfolio_optimization_dashboard(kw_products_path, jj_products_path,
                                              kw_product_based_path, jj_product_based_path,
                                              comparison_file_path, output_dir=None):
    """
    Generate a comprehensive portfolio optimization dashboard.

    Args:
        kw_products_path (str): Path to Kuwait products data
        jj_products_path (str): Path to Jeju products data
        kw_product_based_path (str): Path to Kuwait product-based analysis
        jj_product_based_path (str): Path to Jeju product-based analysis
        comparison_file_path (str): Path to the comparison CSV file
        output_dir (str, optional): Directory to save outputs

    Returns:
        dict: Mapping of visualization names to figure objects
    """
    # Load data
    comp_df = load_comparison_data(comparison_file_path)
    kw_df, jj_df = load_product_data(kw_products_path, jj_products_path)
    kw_attr_df, jj_attr_df = load_attribute_analysis(kw_product_based_path, jj_product_based_path)

    if any(data is None for data in [comp_df, kw_df, jj_df, kw_attr_df, jj_attr_df]):
        print("Error: Failed to load one or more data files")
        return None

    # Create output directory if specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations
    visualizations = {}
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']

    # 1. Create attribute grid visualizations (shelf representation)
    print("Creating attribute grid visualizations...")
    for attribute in attributes:
        vis = create_attribute_grid_visualization(comp_df, attribute)
        if vis:
            name = f"{attribute.lower()}_grid"
            visualizations[name] = vis
            if output_dir:
                vis.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')

    # 2. Create portfolio alignment visualization comparing all attributes
    print("Creating portfolio alignment visualization...")
    alignment_vis = create_portfolio_alignment_visualization(kw_attr_df, jj_attr_df)
    if alignment_vis:
        visualizations["portfolio_alignment"] = alignment_vis
        if output_dir:
            alignment_vis.savefig(os.path.join(output_dir, "portfolio_alignment.png"), dpi=300, bbox_inches='tight')

    # 3. Create radar chart comparing attribute alignment scores
    print("Creating radar chart...")
    radar_vis = create_radar_chart(kw_attr_df, jj_attr_df)
    if radar_vis:
        visualizations["radar_chart"] = radar_vis
        if output_dir:
            radar_vis.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=300, bbox_inches='tight')

    # 4. Create product shelf visualization
    print("Creating product shelf visualization...")
    shelf_vis = create_product_shelf_visualization(kw_df, jj_df)
    if shelf_vis:
        visualizations["product_shelf"] = shelf_vis
        if output_dir:
            shelf_vis.savefig(os.path.join(output_dir, "product_shelf.png"), dpi=300, bbox_inches='tight')

    print(f"Created {len(visualizations)} visualizations")
    return visualizations
    # Example usage
if __name__ == "__main__":
    # This would be replaced with actual file paths
    kw_products_path = "KW_products.csv"
    jj_products_path = "JJ_products.csv"
    kw_product_based_path = "KW_product_based.csv"
    jj_product_based_path = "JJ_product_based.csv"
    comparison_file_path = "comparison_kw_jj.csv"
    output_dir = "visualization_results"
    
    visualizations = generate_portfolio_optimization_dashboard(
        kw_products_path, jj_products_path,
        kw_product_based_path, jj_product_based_path,
        comparison_file_path, output_dir
    )
