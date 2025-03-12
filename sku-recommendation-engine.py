import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def load_data_files(jj_products_path, jj_product_based_path, paris_output_path=None):
    """
    Load the necessary data files for generating SKU recommendations.
    
    Args:
        jj_products_path (str): Path to Jeju products data
        jj_product_based_path (str): Path to Jeju product-based analysis
        paris_output_path (str, optional): Path to PARIS_Output data
    
    Returns:
        tuple: (jj_df, jj_attr_df, paris_df) loaded dataframes
    """
    print("Loading data files for SKU recommendations...")
    
    try:
        # Load Jeju products data
        jj_df = pd.read_csv(jj_products_path)
        
        # Load Jeju attribute analysis
        jj_attr_df = pd.read_csv(jj_product_based_path)
        
        # Load PARIS output if provided
        paris_df = None
        if paris_output_path:
            paris_df = pd.read_csv(paris_output_path)
            # Filter for Jeju if location column exists
            if 'Location' in paris_df.columns:
                paris_df = paris_df[paris_df['Location'] == 'Jeju']
        
        return jj_df, jj_attr_df, paris_df
    
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None, None, None

def extract_attribute_gaps(jj_attr_df, paris_df=None):
    """
    Extract attribute gaps from Jeju attribute analysis or PARIS output.
    
    Args:
        jj_attr_df (DataFrame): Jeju attribute analysis data
        paris_df (DataFrame, optional): PARIS output data filtered for Jeju
    
    Returns:
        dict: Dictionary mapping attributes to their gaps data
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    attribute_gaps = {}
    
    # Extract from product-based analysis first
    for attribute in attributes:
        # Find the section for this attribute
        found_section = False
        attr_values = []
        actual_vals = []
        ideal_vals = []
        gaps = []
        
        for row_idx in range(len(jj_attr_df)):
            if pd.notna(jj_attr_df.iloc[row_idx, 0]) and attribute in str(jj_attr_df.iloc[row_idx, 0]):
                found_section = True
                continue
            
            if found_section:
                # Stop when we hit another attribute section
                if pd.notna(jj_attr_df.iloc[row_idx, 0]) and any(attr in str(jj_attr_df.iloc[row_idx, 0]) for attr in attributes):
                    break
                
                # Extract values
                row = jj_attr_df.iloc[row_idx]
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
        
        # Store extracted data
        if attr_values:
            gap_df = pd.DataFrame({
                attribute: attr_values,
                'Actual': actual_vals,
                'Ideal': ideal_vals,
                'Gap': gaps
            })
            attribute_gaps[attribute] = gap_df
    
    # If PARIS data is available, use it to supplement/validate
    if paris_df is not None:
        for attribute in attributes:
            if attribute in paris_df.columns:
                # Group by the attribute and aggregate
                attr_data = paris_df.groupby(attribute).agg({
                    'Real_So_Segment': 'sum',
                    'Ideal_So_Segment': 'sum',
                    'Delta_SoS': 'sum',
                    'DF_Vol': 'sum'
                }).reset_index()
                
                # If we don't already have data for this attribute, add it
                if attribute not in attribute_gaps:
                    gap_df = pd.DataFrame({
                        attribute: attr_data[attribute].values,
                        'Actual': attr_data['Real_So_Segment'].values * 100,  # Convert to percentage
                        'Ideal': attr_data['Ideal_So_Segment'].values * 100,  # Convert to percentage
                        'Gap': attr_data['Delta_SoS'].values * 100  # Convert to percentage
                    })
                    attribute_gaps[attribute] = gap_df
    
    return attribute_gaps

def identify_top_90_percent_products(jj_df):
    """
    Identify products that make up 90% of the market in Jeju.
    
    Args:
        jj_df (DataFrame): Jeju products data
    
    Returns:
        DataFrame: Top products that make up 90% of the market
    """
    # Verify necessary columns exist
    if 'DF_Vol' not in jj_df.columns:
        print("Error: DF_Vol column not found in products data")
        return None
    
    # Sort by volume
    sorted_df = jj_df.sort_values('DF_Vol', ascending=False).copy()
    
    # Calculate cumulative volume and percentage
    total_volume = sorted_df['DF_Vol'].sum()
    sorted_df['cum_vol'] = sorted_df['DF_Vol'].cumsum()
    sorted_df['cum_pct'] = sorted_df['cum_vol'] / total_volume * 100
    
    # Filter for 90%
    top_90_pct = sorted_df[sorted_df['cum_pct'] <= 90].copy()
    
    # Check if empty
    if top_90_pct.empty:
        print("Warning: No products make up 90% of the market")
        return None
    
    print(f"Identified {len(top_90_pct)} products making up 90% of the market")
    print(f"These products account for {top_90_pct['DF_Vol'].sum() / total_volume * 100:.2f}% of total volume")
    
    # Count PMI vs competitor products
    pmi_count = len(top_90_pct[top_90_pct['TMO'] == 'PMI'])
    comp_count = len(top_90_pct[top_90_pct['TMO'] != 'PMI'])
    print(f"PMI products in top 90%: {pmi_count} ({pmi_count / len(top_90_pct) * 100:.2f}%)")
    print(f"Competitor products in top 90%: {comp_count} ({comp_count / len(top_90_pct) * 100:.2f}%)")
    
    return top_90_pct

def analyze_attribute_distribution(jj_df, top_90_pct=None):
    """
    Analyze the attribute distribution of Jeju products.
    
    Args:
        jj_df (DataFrame): Jeju products data
        top_90_pct (DataFrame, optional): Top products that make up 90% of the market
    
    Returns:
        dict: Dictionary mapping attributes to their distribution data
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    
    # Validate that attributes exist in the data
    available_attrs = [attr for attr in attributes if attr in jj_df.columns]
    if not available_attrs:
        print("Error: No attribute columns found in products data")
        return None
    
    # Use either all products or top 90% products
    df_to_analyze = top_90_pct if top_90_pct is not None else jj_df
    
    attribute_distribution = {}
    
    for attribute in available_attrs:
        # Calculate overall distribution
        total_volume = df_to_analyze['DF_Vol'].sum()
        
        # Group by attribute and TMO
        attr_dist = df_to_analyze.groupby([attribute, 'TMO'])['DF_Vol'].sum().unstack(fill_value=0).reset_index()
        
        # Add total column
        if 'PMI' not in attr_dist.columns:
            attr_dist['PMI'] = 0
        if set(attr_dist.columns) - set(['PMI', attribute]):
            comp_cols = list(set(attr_dist.columns) - set(['PMI', attribute]))
            attr_dist['Comp'] = attr_dist[comp_cols].sum(axis=1)
        else:
            attr_dist['Comp'] = 0
        
        attr_dist['Total'] = attr_dist['PMI'] + attr_dist['Comp']
        
        # Calculate percentages
        attr_dist['Total_Pct'] = attr_dist['Total'] / total_volume * 100
        attr_dist['PMI_Pct'] = attr_dist['PMI'] / total_volume * 100
        attr_dist['Comp_Pct'] = attr_dist['Comp'] / total_volume * 100
        
        # Calculate PMI share within each attribute value
        attr_dist['PMI_Share'] = attr_dist['PMI'] / attr_dist['Total'] * 100
        
        # Store the distribution
        attribute_distribution[attribute] = attr_dist
    
    return attribute_distribution

def generate_sku_recommendations(jj_df, attribute_gaps, attribute_distribution, top_90_pct=None):
    """
    Generate SKU-level recommendations for Jeju.
    
    Args:
        jj_df (DataFrame): Jeju products data
        attribute_gaps (dict): Attribute gaps from extract_attribute_gaps function
        attribute_distribution (dict): Attribute distribution from analyze_attribute_distribution function
        top_90_pct (DataFrame, optional): Top products that make up 90% of the market
    
    Returns:
        dict: Dictionary containing recommendations for optimization
    """
    # Initialize recommendations
    recommendations = {
        'underrepresented_attributes': {},
        'overrepresented_attributes': {},
        'skus_to_add': [],
        'skus_to_remove': [],
        'skus_to_maintain': [],
        'attribute_combinations_needed': []
    }
    
    # Identify under and over-represented attributes
    for attribute, gap_df in attribute_gaps.items():
        # Sort by gap (descending)
        gap_df_sorted = gap_df.sort_values('Gap', ascending=False)
        
        # Identify underrepresented attributes (positive gap)
        underrepresented = gap_df_sorted[gap_df_sorted['Gap'] > 5].copy()
        if not underrepresented.empty:
            recommendations['underrepresented_attributes'][attribute] = underrepresented
        
        # Identify overrepresented attributes (negative gap)
        overrepresented = gap_df_sorted[gap_df_sorted['Gap'] < -5].sort_values('Gap').copy()
        if not overrepresented.empty:
            recommendations['overrepresented_attributes'][attribute] = overrepresented
    
    # Get current PMI products
    pmi_products = jj_df[jj_df['TMO'] == 'PMI'].copy()
    
    # Analyze SKUs to maintain (strong performers in well-represented or underrepresented segments)
    skus_to_maintain = []
    
    # Use top 90% if available, otherwise use all PMI products
    pmi_top_products = top_90_pct[top_90_pct['TMO'] == 'PMI'] if top_90_pct is not None else pmi_products
    
    for idx, product in pmi_top_products.iterrows():
        maintain = True
        attribute_match = 0
        
        # Count how many attributes are in underrepresented or well-represented segments
        for attribute in ['Flavor', 'Taste', 'Thickness', 'Length']:
            if attribute not in product or pd.isna(product[attribute]):
                continue
                
            # Check if this attribute value is underrepresented
            if attribute in recommendations['underrepresented_attributes']:
                underrep_values = recommendations['underrepresented_attributes'][attribute][attribute].tolist()
                if product[attribute] in underrep_values:
                    attribute_match += 1
            
            # Check if this attribute value is not overrepresented
            if attribute in recommendations['overrepresented_attributes']:
                overrep_values = recommendations['overrepresented_attributes'][attribute][attribute].tolist()
                if product[attribute] not in overrep_values:
                    attribute_match += 1
        
        # If product matches at least 2 attribute criteria, maintain it
        if attribute_match >= 2:
            skus_to_maintain.append({
                'SKU': product['SKU'] if 'SKU' in product else f'Brand ID: {product["CR_BrandId"]}',
                'CR_BrandId': product['CR_BrandId'],
                'Volume': product['DF_Vol'],
                'Attributes': {attr: product[attr] for attr in ['Flavor', 'Taste', 'Thickness', 'Length'] 
                              if attr in product and pd.notna(product[attr])},
                'Attribute_Match': attribute_match
            })
    
    # Sort by volume (descending)
    skus_to_maintain = sorted(skus_to_maintain, key=lambda x: x['Volume'], reverse=True)
    recommendations['skus_to_maintain'] = skus_to_maintain
    
    # Identify SKUs to remove (poor performers in overrepresented segments)
    skus_to_remove = []
    
    for idx, product in pmi_products.iterrows():
        # Skip products that are already in the maintain list
        if any(product['CR_BrandId'] == maintain_sku['CR_BrandId'] for maintain_sku in skus_to_maintain):
            continue
        
        remove = False
        attribute_mismatch = 0
        
        # Count how many attributes are in overrepresented segments
        for attribute in ['Flavor', 'Taste', 'Thickness', 'Length']:
            if attribute not in product or pd.isna(product[attribute]):
                continue
                
            # Check if this attribute value is overrepresented
            if attribute in recommendations['overrepresented_attributes']:
                overrep_values = recommendations['overrepresented_attributes'][attribute][attribute].tolist()
                if product[attribute] in overrep_values:
                    attribute_mismatch += 1
        
        # If product matches at least 2 overrepresented attributes, consider for removal
        if attribute_mismatch >= 2:
            # Check if it's a poor performer
            if top_90_pct is not None and product['CR_BrandId'] not in top_90_pct['CR_BrandId'].values:
                remove = True
            
            if remove:
                skus_to_remove.append({
                    'SKU': product['SKU'] if 'SKU' in product else f'Brand ID: {product["CR_BrandId"]}',
                    'CR_BrandId': product['CR_BrandId'],
                    'Volume': product['DF_Vol'],
                    'Attributes': {attr: product[attr] for attr in ['Flavor', 'Taste', 'Thickness', 'Length'] 
                                  if attr in product and pd.notna(product[attr])},
                    'Attribute_Mismatch': attribute_mismatch
                })
    
    # Sort by volume (ascending, so lowest volume first)
    skus_to_remove = sorted(skus_to_remove, key=lambda x: x['Volume'])
    recommendations['skus_to_remove'] = skus_to_remove
    
    # Identify attribute combinations for new SKUs
    attribute_combinations = []
    
    # Get top underrepresented attribute values for each attribute
    top_underrep = {}
    for attribute, underrep_df in recommendations['underrepresented_attributes'].items():
        # Take top 2 most underrepresented values
        top_values = underrep_df.sort_values('Gap', ascending=False).head(2)[attribute].tolist()
        top_underrep[attribute] = top_values
    
    # Generate combinations of underrepresented attributes
    if len(top_underrep) >= 2:
        # Start with the most underrepresented attributes
        priority_attributes = sorted(top_underrep.keys(), 
                                    key=lambda a: recommendations['underrepresented_attributes'][a]['Gap'].max(), 
                                    reverse=True)
        
        # Take the top 2 most underrepresented attributes
        primary_attrs = priority_attributes[:2]
        
        # Generate combinations
        for attr1 in primary_attrs:
            for val1 in top_underrep[attr1]:
                for attr2 in [a for a in priority_attributes if a != attr1]:
                    for val2 in top_underrep[attr2]:
                        # Check if this combination already exists in PMI portfolio
                        exists = False
                        for idx, product in pmi_products.iterrows():
                            if (attr1 in product and product[attr1] == val1 and 
                                attr2 in product and product[attr2] == val2):
                                exists = True
                                break
                        
                        if not exists:
                            # Check competitor products for this combination
                            comp_products = jj_df[(jj_df['TMO'] != 'PMI') & 
                                                (jj_df[attr1] == val1) & 
                                                (jj_df[attr2] == val2)]
                            
                            if not comp_products.empty:
                                # Calculate total volume for this combination in competitor portfolio
                                volume = comp_products['DF_Vol'].sum()
                                
                                attribute_combinations.append({
                                    'Combination': {attr1: val1, attr2: val2},
                                    'Gap1': float(recommendations['underrepresented_attributes'][attr1][recommendations['underrepresented_attributes'][attr1][attr1] == val1]['Gap'].iloc[0]),
                                    'Gap2': float(recommendations['underrepresented_attributes'][attr2][recommendations['underrepresented_attributes'][attr2][attr2] == val2]['Gap'].iloc[0]),
                                    'Total_Gap': float(recommendations['underrepresented_attributes'][attr1][recommendations['underrepresented_attributes'][attr1][attr1] == val1]['Gap'].iloc[0] + 
                                                    recommendations['underrepresented_attributes'][attr2][recommendations['underrepresented_attributes'][attr2][attr2] == val2]['Gap'].iloc[0]),
                                    'Comp_Volume': volume,
                                    'Comp_SKUs': len(comp_products),
                                    'Comp_Products': comp_products['SKU'].tolist() if 'SKU' in comp_products else comp_products['CR_BrandId'].tolist()
                                })
    
    # Sort by total gap and competitor volume
    if attribute_combinations:
        attribute_combinations = sorted(attribute_combinations, 
                                      key=lambda x: (x['Total_Gap'], x['Comp_Volume']), 
                                      reverse=True)
    
    recommendations['attribute_combinations_needed'] = attribute_combinations
    
    # Generate specific SKUs to add
    for combo in attribute_combinations[:5]:  # Take top 5 combinations
        # Get attribute values
        attrs = combo['Combination']
        
        # Generate SKU recommendation
        sku_rec = {
            'Attribute_Combination': attrs,
            'Total_Gap': combo['Total_Gap'],
            'Market_Potential': combo['Comp_Volume'],
            'Competitor_Products': combo['Comp_Products'][:3] if len(combo['Comp_Products']) > 3 else combo['Comp_Products']
        }
        
        # Add to recommendations
        recommendations['skus_to_add'].append(sku_rec)
    
    return recommendations

def visualize_sku_recommendations(recommendations, output_dir=None):
    """
    Create visualizations for SKU recommendations.
    
    Args:
        recommendations (dict): Recommendations from generate_sku_recommendations function
        output_dir (str, optional): Directory to save output visualizations
    
    Returns:
        dict: Dictionary of matplotlib figures
    """
    figures = {}
    
    # 1. Visualize underrepresented vs. overrepresented attributes
    attr_fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    attr_data = []
    
    for i, attribute in enumerate(attributes):
        # Get underrepresented values
        underrep_values = []
        underrep_gaps = []
        if attribute in recommendations['underrepresented_attributes']:
            underrep_df = recommendations['underrepresented_attributes'][attribute]
            underrep_values = underrep_df[attribute].tolist()
            underrep_gaps = underrep_df['Gap'].tolist()
        
        # Get overrepresented values
        overrep_values = []
        overrep_gaps = []
        if attribute in recommendations['overrepresented_attributes']:
            overrep_df = recommendations['overrepresented_attributes'][attribute]
            overrep_values = overrep_df[attribute].tolist()
            overrep_gaps = overrep_df['Gap'].tolist()
        
        # Combine data
        attr_values = underrep_values + overrep_values
        gaps = underrep_gaps + overrep_gaps
        colors = ['green'] * len(underrep_values) + ['red'] * len(overrep_values)
        
        # Sort by absolute gap value
        if attr_values:
            sorted_idx = np.argsort(np.abs(gaps))[::-1]  # Sort by absolute gap value (descending)
            attr_values = [attr_values[i] for i in sorted_idx]
            gaps = [gaps[i] for i in sorted_idx]
            colors = [colors[i] for i in sorted_idx]
            
            # Create horizontal bar chart
            ax = axes[i]
            bars = ax.barh(attr_values, gaps, color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x = width + 1 if width >= 0 else width - 5
                ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                       f"{width:.1f}%", va='center', 
                       color='green' if width >= 0 else 'red',
                       fontweight='bold')
            
            # Add dividing line at zero
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Set title and labels
            ax.set_title(f"{attribute} Gaps (Underrepresented vs. Overrepresented)")
            ax.set_xlabel("Gap (%)")
            
            # Track data for summary
            for val, gap in zip(attr_values, gaps):
                attr_data.append({
                    'Attribute': attribute,
                    'Value': val,
                    'Gap': gap,
                    'Status': 'Underrepresented' if gap > 0 else 'Overrepresented'
                })
    
    plt.tight_layout()
    figures['attribute_gaps'] = attr_fig
    
    # 2. Visualize SKUs to maintain vs. remove
    if recommendations['skus_to_maintain'] or recommendations['skus_to_remove']:
        sku_fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        maintain_skus = recommendations['skus_to_maintain']
        remove_skus = recommendations['skus_to_remove']
        
        # Create bar data
        sku_names = [sku['SKU'] if len(str(sku['SKU'])) < 30 else str(sku['SKU'])[:27] + '...' 
                    for sku in maintain_skus + remove_skus]
        volumes = [sku['Volume'] for sku in maintain_skus + remove_skus]
        colors = ['green'] * len(maintain_skus) + ['red'] * len(remove_skus)
        
        # Sort by volume (for better visualization)
        if sku_names:
            sorted_idx = np.argsort(volumes)[::-1]  # Sort by volume (descending)
            sku_names = [sku_names[i] for i in sorted_idx]
            volumes = [volumes[i] for i in sorted_idx]
            colors = [colors[i] for i in sorted_idx]
            
            # Create horizontal bar chart
            bars = ax.barh(sku_names, volumes, color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1 * max(volumes), bar.get_y() + bar.get_height()/2, 
                       f"{width:,.0f}", va='center')
            
            # Set title and labels
            ax.set_title("SKUs to Maintain (Green) vs. Remove (Red)")
            ax.set_xlabel("Volume")
            ax.set_ylabel("SKU")
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Maintain'),
                Patch(facecolor='red', label='Remove')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        figures['sku_recommendations'] = sku_fig
    
    # 3. Visualize attribute combinations for new SKUs
    if recommendations['attribute_combinations_needed']:
        combo_fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        combos = recommendations['attribute_combinations_needed'][:10]  # Top 10 combinations
        
        # Create bar data
        combo_names = [f"{', '.join([f'{k}: {v}' for k, v in combo['Combination'].items()])}" 
                      for combo in combos]
        gaps = [combo['Total_Gap'] for combo in combos]
        volumes = [combo['Comp_Volume'] for combo in combos]
        
        # Normalize volumes for bubble size
        if volumes:
            max_volume = max(volumes)
            norm_volumes = [v / max_volume * 1000 for v in volumes] if max_volume > 0 else [500] * len(volumes)
        else:
            norm_volumes = []
        
        # Create scatter plot with bubble size representing market potential
        if combo_names:
            # Create bar chart for gaps
            y_pos = np.arange(len(combo_names))
            ax.barh(y_pos, gaps, color='green', alpha=0.5)
            
            # Add volume bubbles
            for i, (gap, vol, norm_vol) in enumerate(zip(gaps, volumes, norm_volumes)):
                ax.scatter(gap, i, s=norm_vol, color='blue', alpha=0.7, edgecolors='black')
                
                # Add volume label
                ax.text(gap + 0.1 * max(gaps), i, f"Vol: {vol:,.0f}", va='center')
            
            # Set title and labels
            ax.set_title("Recommended Attribute Combinations for New SKUs")
            ax.set_xlabel("Total Gap (%)")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(combo_names)
            
            # Add a second axis for volume scale
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks([])
            
            # Add legend for bubble size
            from matplotlib.legend_handler import HandlerPatch
            import matplotlib.patches as mpatches
            
            class HandlerEllipse(HandlerPatch):
                def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                    center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                    p = mpatches.Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
                    self.update_prop(p, orig_handle, legend)
                    p.set_transform(trans)
                    return [p]
            
            size_labels = [int(max_volume * 0.25), int(max_volume * 0.5), int(max_volume)]
            size_bubbles = [250, 500, 1000]
            
            bubble_handles = [mpatches.Circle((0, 0), radius=np.sqrt(s) / 30) for s in size_bubbles]
            bubble_labels = [f"{v:,}" for v in size_labels]
            
            ax2.legend(bubble_handles, bubble_labels, 
                      title="Competitor Volume", frameon=True,
                      loc='upper right', handler_map={mpatches.Circle: HandlerEllipse()})
        
        plt.tight_layout()
        figures['attribute_combinations'] = combo_fig
    
    # 4. Create summary visualization
    summary_fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare summary data
    summary_data = []
    
    # Count underrepresented and overrepresented attributes
    for attr in attributes:
        underrep_count = len(recommendations['underrepresented_attributes'].get(attr, pd.DataFrame()).index)
        overrep_count = len(recommendations['overrepresented_attributes'].get(attr, pd.DataFrame()).index)
        
        summary_data.append({
            'Attribute': attr,
            'Underrepresented': underrep_count,
            'Overrepresented': overrep_count
        })
    
    # Create a DataFrame for easier plotting
    summary_df = pd.DataFrame(summary_data)
    
    # Set up the axes
    x = np.arange(len(attributes))
    width = 0.35
    
    # Create grouped bar chart
    ax.bar(x - width/2, summary_df['Underrepresented'], width, label='Underrepresented')
    ax.bar(x + width/2, summary_df['Overrepresented'], width, label='Overrepresented')
    
    # Add text at the top of the bars
    for i, v in enumerate(summary_df['Underrepresented']):
        ax.text(i - width/2, v + 0.1, str(v), ha='center')
    
    for i, v in enumerate(summary_df['Overrepresented']):
        ax.text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Customize the chart
    ax.set_title('Summary of Attribute Gaps')
    ax.set_xticks(x)
    ax.set_xticklabels(attributes)
    ax.legend()
    
    # Add recommendations summary
    text_str = f"""
    Recommendation Summary:
    
    • SKUs to maintain: {len(recommendations['skus_to_maintain'])}
    • SKUs to remove: {len(recommendations['skus_to_remove'])}
    • New SKU combinations: {len(recommendations['attribute_combinations_needed'])}
    
    Most underrepresented segments:
    """
    
    # Add top underrepresented segments
    top_gaps = []
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            top_val = df.iloc[0]
            top_gaps.append((attr, top_val[attr], top_val['Gap']))
    
    # Sort by gap
    top_gaps = sorted(top_gaps, key=lambda x: x[2], reverse=True)
    
    for attr, val, gap in top_gaps[:3]:  # Top 3 gaps
        text_str += f"• {attr}: {val} (Gap: {gap:.1f}%)\n"
    
    # Add the text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.97, 0.03, text_str, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    figures['summary'] = summary_fig
    
    # Save figures if output directory is specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
            print(f"Saved {name}.png to {output_dir}")
    
    return figures

def create_phased_implementation_plan(recommendations):
    """
    Create a phased implementation plan for the recommendations.
    
    Args:
        recommendations (dict): Recommendations from generate_sku_recommendations function
    
    Returns:
        dict: Phased implementation plan
    """
    implementation_plan = {
        'short_term': {
            'description': 'Actions that can be implemented immediately within existing constraints',
            'skus_to_maintain': [],
            'skus_to_remove': [],
            'attribute_focus': {}
        },
        'medium_term': {
            'description': 'Actions that require some planning and adjustment',
            'skus_to_add': [],
            'skus_to_remove': [],
            'attribute_focus': {}
        },
        'long_term': {
            'description': 'Strategic actions for comprehensive portfolio optimization',
            'skus_to_add': [],
            'strategy_shifts': [],
            'attribute_focus': {}
        }
    }
    
    # 1. Short-term actions (immediate)
    
    # SKUs to maintain (straightforward)
    implementation_plan['short_term']['skus_to_maintain'] = recommendations['skus_to_maintain'][:5]  # Top 5 SKUs
    
    # Low-volume SKUs to remove
    low_vol_remove = [sku for sku in recommendations['skus_to_remove'] 
                     if sku['Volume'] < np.percentile([s['Volume'] for s in recommendations['skus_to_remove']], 25) 
                     if recommendations['skus_to_remove']]
    implementation_plan['short_term']['skus_to_remove'] = low_vol_remove
    
    # Short-term attribute focus
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            # Get top underrepresented attribute values
            top_vals = df.sort_values('Gap', ascending=False).head(2)
            implementation_plan['short_term']['attribute_focus'][attr] = top_vals[attr].tolist()
    
    # 2. Medium-term actions (3-6 months)
    
    # SKUs to add (based on attribute combinations)
    if recommendations['attribute_combinations_needed']:
        implementation_plan['medium_term']['skus_to_add'] = recommendations['attribute_combinations_needed'][:3]  # Top 3 combinations
    
    # Additional SKUs to remove
    remaining_remove = [sku for sku in recommendations['skus_to_remove'] 
                       if sku not in low_vol_remove]
    implementation_plan['medium_term']['skus_to_remove'] = remaining_remove
    
    # Medium-term attribute focus (all significant gaps)
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            # Get all attribute values with gap > 10%
            sig_vals = df[df['Gap'] > 10]
            if not sig_vals.empty:
                implementation_plan['medium_term']['attribute_focus'][attr] = sig_vals[attr].tolist()
    
    # 3. Long-term actions (6+ months)
    
    # All recommended SKUs to add
    implementation_plan['long_term']['skus_to_add'] = recommendations['attribute_combinations_needed']
    
    # Strategic shifts
    
    # Identify major attribute categories to focus on
    major_gaps = {}
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            # Calculate total gap for this attribute
            total_gap = df['Gap'].sum()
            major_gaps[attr] = total_gap
    
    # Sort attributes by total gap
    sorted_attrs = sorted(major_gaps.items(), key=lambda x: x[1], reverse=True)
    
    # Add strategic shifts
    for attr, gap in sorted_attrs:
        implementation_plan['long_term']['strategy_shifts'].append({
            'attribute': attr,
            'total_gap': gap,
            'description': f"Shift portfolio towards underrepresented {attr} segments"
        })
    
    # Long-term attribute focus (comprehensive)
    for attr in ['Flavor', 'Taste', 'Thickness', 'Length']:
        if attr in recommendations['underrepresented_attributes']:
            implementation_plan['long_term']['attribute_focus'][attr] = recommendations['underrepresented_attributes'][attr][attr].tolist()
    
    return implementation_plan

def generate_optimization_report(recommendations, implementation_plan):
    """
    Generate a comprehensive report for the portfolio optimization.
    
    Args:
        recommendations (dict): Recommendations from generate_sku_recommendations function
        implementation_plan (dict): Phased implementation plan
    
    Returns:
        str: Formatted report text
    """
    report = "JEJU PORTFOLIO OPTIMIZATION REPORT\n"
    report += "=" * 50 + "\n\n"
    
    # 1. Executive Summary
    report += "EXECUTIVE SUMMARY\n"
    report += "-" * 30 + "\n\n"
    
    # Count recommendations
    maintain_count = len(recommendations['skus_to_maintain'])
    remove_count = len(recommendations['skus_to_remove'])
    add_count = len(recommendations['attribute_combinations_needed'])
    
    report += f"This analysis identifies significant portfolio gaps in the Jeju duty-free market, "
    report += f"with recommendations to maintain {maintain_count} SKUs, remove {remove_count} SKUs, "
    report += f"and add {add_count} new SKU combinations to better align with passenger preferences.\n\n"
    
    # Add key opportunity summary
    report += "Key Opportunities:\n"
    
    # Get top underrepresented segments across all attributes
    top_gaps = []
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            for _, row in df.head(2).iterrows():  # Top 2 per attribute
                top_gaps.append((attr, row[attr], row['Gap']))
    
    # Sort by gap size
    top_gaps = sorted(top_gaps, key=lambda x: x[2], reverse=True)
    
    for i, (attr, val, gap) in enumerate(top_gaps[:5]):  # Top 5 overall gaps
        report += f"{i+1}. {attr}: {val} (Gap: {gap:.1f}%)\n"
    
    report += "\n"
    
    # 2. Current Portfolio Assessment
    report += "CURRENT PORTFOLIO ASSESSMENT\n"
    report += "-" * 30 + "\n\n"
    
    # Summarize attribute alignment
    report += "Attribute Alignment Assessment:\n\n"
    
    for attr in ['Flavor', 'Taste', 'Thickness', 'Length']:
        report += f"{attr}:\n"
        
        # Underrepresented
        if attr in recommendations['underrepresented_attributes'] and not recommendations['underrepresented_attributes'][attr].empty:
            df = recommendations['underrepresented_attributes'][attr]
            report += "  Underrepresented segments:\n"
            for _, row in df.iterrows():
                report += f"    • {row[attr]}: Gap of {row['Gap']:.1f}% (Actual: {row['Actual']:.1f}%, Ideal: {row['Ideal']:.1f}%)\n"
        
        # Overrepresented
        if attr in recommendations['overrepresented_attributes'] and not recommendations['overrepresented_attributes'][attr].empty:
            df = recommendations['overrepresented_attributes'][attr]
            report += "  Overrepresented segments:\n"
            for _, row in df.iterrows():
                report += f"    • {row[attr]}: Gap of {row['Gap']:.1f}% (Actual: {row['Actual']:.1f}%, Ideal: {row['Ideal']:.1f}%)\n"
        
        report += "\n"
    
    # 3. SKU Recommendations
    report += "SKU RECOMMENDATIONS\n"
    report += "-" * 30 + "\n\n"
    
    # SKUs to maintain
    report += "SKUs to Maintain:\n"
    for i, sku in enumerate(recommendations['skus_to_maintain'][:10]):  # Top 10
        attr_str = ", ".join([f"{k}: {v}" for k, v in sku['Attributes'].items()])
        report += f"{i+1}. {sku['SKU']} (Volume: {sku['Volume']:,.0f}, Attributes: {attr_str})\n"
    
    report += "\n"
    
    # SKUs to remove
    report += "SKUs to Consider for Removal:\n"
    for i, sku in enumerate(recommendations['skus_to_remove'][:10]):  # Top 10
        attr_str = ", ".join([f"{k}: {v}" for k, v in sku['Attributes'].items()])
        report += f"{i+1}. {sku['SKU']} (Volume: {sku['Volume']:,.0f}, Attributes: {attr_str})\n"
    
    report += "\n"
    
    # New SKU recommendations
    report += "Recommended New Product Combinations:\n"
    for i, combo in enumerate(recommendations['skus_to_add'][:5]):  # Top 5
        attr_str = ", ".join([f"{k}: {v}" for k, v in combo['Attribute_Combination'].items()])
        report += f"{i+1}. {attr_str}\n"
        report += f"   - Total Gap: {combo['Total_Gap']:.1f}%\n"
        report += f"   - Market Potential: {combo['Market_Potential']:,.0f} units\n"
        if combo['Competitor_Products']:
            report += f"   - Competitor References: {', '.join(str(p) for p in combo['Competitor_Products'])}\n"
        report += "\n"
    
    # 4. Implementation Plan
    report += "IMPLEMENTATION PLAN\n"
    report += "-" * 30 + "\n\n"
    
    # Short-term
    report += "Short-Term Actions (0-3 months):\n"
    report += f"{implementation_plan['short_term']['description']}\n\n"
    
    report += "1. Maintain & Support Key SKUs:\n"
    for i, sku in enumerate(implementation_plan['short_term']['skus_to_maintain']):
        report += f"   • {sku['SKU']} (Volume: {sku['Volume']:,.0f})\n"
    
    report += "\n2. Remove Underperforming SKUs:\n"
    for i, sku in enumerate(implementation_plan['short_term']['skus_to_remove']):
        report += f"   • {sku['SKU']} (Volume: {sku['Volume']:,.0f})\n"
    
    report += "\n3. Focus on Key Attribute Segments:\n"
    for attr, values in implementation_plan['short_term']['attribute_focus'].items():
        report += f"   • {attr}: {', '.join(values)}\n"
    
    report += "\n"
    
    # Medium-term
    report += "Medium-Term Actions (3-6 months):\n"
    report += f"{implementation_plan['medium_term']['description']}\n\n"
    
    report += "1. Introduce New SKUs:\n"
    for i, combo in enumerate(implementation_plan['medium_term']['skus_to_add']):
        attr_str = ", ".join([f"{k}: {v}" for k, v in combo['Combination'].items()])
        report += f"   • {attr_str} (Market Potential: {combo['Comp_Volume']:,.0f} units)\n"
    
    report += "\n2. Continue Portfolio Rationalization:\n"
    for i, sku in enumerate(implementation_plan['medium_term']['skus_to_remove']):
        report += f"   • {sku['SKU']} (Volume: {sku['Volume']:,.0f})\n"
    
    report += "\n3. Expand Focus Attribute Segments:\n"
    for attr, values in implementation_plan['medium_term']['attribute_focus'].items():
        report += f"   • {attr}: {', '.join(values)}\n"
    
    report += "\n"
    
    # Long-term
    report += "Long-Term Strategy (6+ months):\n"
    report += f"{implementation_plan['long_term']['description']}\n\n"
    
    report += "1. Strategic Portfolio Shifts:\n"
    for shift in implementation_plan['long_term']['strategy_shifts']:
        report += f"   • {shift['description']} (Total Gap: {shift['total_gap']:.1f}%)\n"
    
    report += "\n2. Complete Attribute Optimization:\n"
    for attr, values in implementation_plan['long_term']['attribute_focus'].items():
        report += f"   • {attr}: {', '.join(values)}\n"
    
    report += "\n"
    
    # 5. Expected Impact
    report += "EXPECTED IMPACT\n"
    report += "-" * 30 + "\n\n"
    
    # Calculate potential market share gain based on gaps
    total_gap = 0
    for attr, df in recommendations['underrepresented_attributes'].items():
        if not df.empty:
            total_gap += df['Gap'].sum()
    
    # Estimate potential market share gain
    # This is a simplified estimate - assume we can capture half of the identified gaps
    potential_gain = total_gap * 0.5
    
    report += f"Potential Market Share Impact:\n"
    report += f"• Current estimated PMI market share: ~11.4%\n"
    report += f"• Total identified portfolio gaps: {total_gap:.1f}%\n"
    report += f"• Conservative market share gain potential: {potential_gain:.1f}%\n"
    report += f"• Potential future market share: {11.4 + potential_gain:.1f}%\n\n"
    
    report += "This represents a projected {:.1f}% increase in market share by addressing the identified portfolio gaps.".format((potential_gain / 11.4) * 100)
    
    return report

def run_sku_recommendation_engine(jj_products_path, jj_product_based_path, paris_output_path=None, output_dir=None):
    """
    Run the complete SKU recommendation engine process and output results.
    
    Args:
        jj_products_path (str): Path to Jeju products data
        jj_product_based_path (str): Path to Jeju product-based analysis
        paris_output_path (str, optional): Path to PARIS_Output data
        output_dir (str, optional): Directory to save outputs
    
    Returns:
        tuple: (recommendations, implementation_plan, figures)
    """
    # Load data
    jj_df, jj_attr_df, paris_df = load_data_files(jj_products_path, jj_product_based_path, paris_output_path)
    
    if jj_df is None or jj_attr_df is None:
        print("Failed to load necessary data files")
        return None, None, None
    
    # Extract attribute gaps
    attribute_gaps = extract_attribute_gaps(jj_attr_df, paris_df)
    
    # Identify top 90% products
    top_90_pct = identify_top_90_percent_products(jj_df)
    
    # Analyze attribute distribution
    attribute_distribution = analyze_attribute_distribution(jj_df, top_90_pct)
    
    # Generate SKU recommendations
    recommendations = generate_sku_recommendations(jj_df, attribute_gaps, attribute_distribution, top_90_pct)
    
    # Create implementation plan
    implementation_plan = create_phased_implementation_plan(recommendations)
    
    # Create visualizations
    figures = visualize_sku_recommendations(recommendations, output_dir)
    
    # Generate report
    report = generate_optimization_report(recommendations, implementation_plan)
    print(report)
    
    # Save report if output directory is specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'jeju_optimization_report.txt'), 'w') as f:
            f.write(report)
        
        print(f"Saved report to {os.path.join(output_dir, 'jeju_optimization_report.txt')}")
    
    return recommendations, implementation_plan, figures

# Example usage
if __name__ == "__main__":
    # This would be replaced with actual file paths
    jj_products_path = "JJ_products.csv"
    jj_product_based_path = "JJ_product_based.csv"
    paris_output_path = "PARIS_Output.csv"
    output_dir = "recommendation_results"
    
    recommendations, implementation_plan, figures = run_sku_recommendation_engine(
        jj_products_path, jj_product_based_path, paris_output_path, output_dir
    )
