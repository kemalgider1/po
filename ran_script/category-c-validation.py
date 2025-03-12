import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def load_paris_data(paris_output_path):
    """
    Load the PARIS Output data containing ideal vs. real segment distribution.
    
    Args:
        paris_output_path (str): Path to PARIS_Output data file
    
    Returns:
        DataFrame: Loaded PARIS data
    """
    print("Loading PARIS Output data...")
    try:
        paris_df = pd.read_csv(paris_output_path)
        return paris_df
    except Exception as e:
        print(f"Error loading PARIS data: {e}")
        return None

def validate_category_c_scoring(paris_df):
    """
    Validate Category C scoring calculations.
    
    Args:
        paris_df (DataFrame): PARIS Output data
    
    Returns:
        dict: Validation results with metrics by location
    """
    results = {}
    
    try:
        # Get unique locations
        locations = paris_df['Location'].unique()
        
        for location in locations:
            # Filter data for this location
            loc_data = paris_df[paris_df['Location'] == location]
            
            # Check if we have enough data points
            if len(loc_data) < 3:
                results[location] = {
                    'status': 'Insufficient data',
                    'data_points': len(loc_data)
                }
                continue
            
            # Validate Delta_SoS calculation (Ideal - Real)
            recalculated_delta = loc_data['Ideal_So_Segment'] - loc_data['Real_So_Segment']
            delta_error = (loc_data['Delta_SoS'] - recalculated_delta).abs().mean()
            
            # Calculate correlation between real and ideal segment shares
            correlation = pearsonr(loc_data['Real_So_Segment'], loc_data['Ideal_So_Segment'])[0]
            
            # Fit linear regression model to calculate R²
            X = loc_data[['Real_So_Segment']]
            y = loc_data[['Ideal_So_Segment']]
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            
            # Calculate average Real and Ideal segment shares
            avg_real = loc_data['Real_So_Segment'].mean()
            avg_ideal = loc_data['Ideal_So_Segment'].mean()
            
            # Calculate total positive and negative gaps
            positive_gaps = loc_data[loc_data['Delta_SoS'] > 0]['Delta_SoS'].sum()
            negative_gaps = loc_data[loc_data['Delta_SoS'] < 0]['Delta_SoS'].sum()
            
            # Calculate standard deviation of gaps
            gap_std = loc_data['Delta_SoS'].std()
            
            # Store metrics
            results[location] = {
                'status': 'Valid' if delta_error < 0.0001 else 'Error in Delta_SoS calculation',
                'data_points': len(loc_data),
                'delta_calculation_error': delta_error,
                'correlation': correlation,
                'r_squared': r_squared,
                'cat_c_score': r_squared * 10,  # This is the Category C score calculation
                'avg_real_segment': avg_real,
                'avg_ideal_segment': avg_ideal,
                'positive_gaps_sum': positive_gaps,
                'negative_gaps_sum': negative_gaps,
                'gap_std': gap_std
            }
            
    except Exception as e:
        print(f"Error validating Category C scoring: {e}")
        results['error'] = str(e)
    
    return results

def analyze_attribute_alignment(paris_df):
    """
    Analyze alignment between real and ideal distribution for each attribute.
    
    Args:
        paris_df (DataFrame): PARIS Output data
    
    Returns:
        dict: Alignment analysis by location and attribute
    """
    alignment_results = {}
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    
    try:
        # Get unique locations
        locations = paris_df['Location'].unique()
        
        for location in locations:
            # Filter data for this location
            loc_data = paris_df[paris_df['Location'] == location]
            
            alignment_results[location] = {}
            
            for attr in attributes:
                if attr in loc_data.columns:
                    # Group by the attribute and calculate average real and ideal shares
                    attr_data = loc_data.groupby(attr).agg({
                        'Real_So_Segment': 'sum',
                        'Ideal_So_Segment': 'sum',
                        'Delta_SoS': 'sum',
                        'DF_Vol': 'sum'
                    }).reset_index()
                    
                    # Calculate volume percentage for each attribute value
                    total_vol = attr_data['DF_Vol'].sum()
                    attr_data['Volume_Pct'] = attr_data['DF_Vol'] / total_vol * 100 if total_vol > 0 else 0
                    
                    # Calculate weighted alignment score
                    weighted_gaps = np.abs(attr_data['Delta_SoS']) * attr_data['Volume_Pct']
                    alignment_score = 10 - min(10, weighted_gaps.sum() / 10)
                    
                    # Identify most underrepresented attributes (where ideal > real)
                    underrepresented = attr_data[attr_data['Delta_SoS'] > 0].sort_values('Delta_SoS', ascending=False)
                    
                    # Identify most overrepresented attributes (where real > ideal)
                    overrepresented = attr_data[attr_data['Delta_SoS'] < 0].sort_values('Delta_SoS')
                    
                    alignment_results[location][attr] = {
                        'attribute_data': attr_data,
                        'alignment_score': alignment_score,
                        'underrepresented': underrepresented[attr].tolist() if not underrepresented.empty else [],
                        'overrepresented': overrepresented[attr].tolist() if not overrepresented.empty else [],
                        'max_gap': attr_data['Delta_SoS'].abs().max(),
                        'avg_gap': attr_data['Delta_SoS'].abs().mean()
                    }
                    
    except Exception as e:
        print(f"Error analyzing attribute alignment: {e}")
        alignment_results['error'] = str(e)
    
    return alignment_results

def visualize_category_c_validation(validation_results, locations=None):
    """
    Create visualizations for Category C validation results.
    
    Args:
        validation_results (dict): Results from validate_category_c_scoring function
        locations (list, optional): Specific locations to visualize, defaults to all
    
    Returns:
        matplotlib.figure.Figure: The figure containing the visualizations
    """
    if locations is None:
        # Use Kuwait and Jeju if available, otherwise use all locations
        if 'Kuwait' in validation_results and 'Jeju' in validation_results:
            locations = ['Kuwait', 'Jeju']
        else:
            locations = list(validation_results.keys())
            # Remove 'error' if present
            if 'error' in locations:
                locations.remove('error')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Extract metrics
    cat_c_scores = [validation_results[loc]['cat_c_score'] if loc in validation_results else 0 for loc in locations]
    correlations = [validation_results[loc]['correlation'] if loc in validation_results else 0 for loc in locations]
    r_squareds = [validation_results[loc]['r_squared'] if loc in validation_results else 0 for loc in locations]
    avg_gaps = [np.abs(validation_results[loc]['positive_gaps_sum'] + validation_results[loc]['negative_gaps_sum']) / validation_results[loc]['data_points'] if loc in validation_results else 0 for loc in locations]
    
    # Plot Category C scores
    axes[0].bar(locations, cat_c_scores, color=['green', 'red'] if len(locations) == 2 else 'blue')
    axes[0].set_title('Category C Scores')
    axes[0].set_ylabel('Score (0-10)')
    axes[0].set_ylim(0, 10)
    
    # Add value labels
    for i, v in enumerate(cat_c_scores):
        axes[0].text(i, v + 0.3, f"{v:.2f}", ha='center')
    
    # Plot correlations
    axes[1].bar(locations, correlations, color=['green', 'red'] if len(locations) == 2 else 'blue')
    axes[1].set_title('Real-Ideal Correlation')
    axes[1].set_ylabel('Correlation Coefficient')
    axes[1].set_ylim(-1, 1)
    
    # Add value labels
    for i, v in enumerate(correlations):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    # Plot R² values
    axes[2].bar(locations, r_squareds, color=['green', 'red'] if len(locations) == 2 else 'blue')
    axes[2].set_title('R² Values')
    axes[2].set_ylabel('R²')
    axes[2].set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(r_squareds):
        axes[2].text(i, v + 0.03, f"{v:.2f}", ha='center')
    
    # Plot average absolute gaps
    axes[3].bar(locations, avg_gaps, color=['green', 'red'] if len(locations) == 2 else 'blue')
    axes[3].set_title('Average Absolute Gaps')
    axes[3].set_ylabel('Average |Δ|')
    
    # Add value labels
    for i, v in enumerate(avg_gaps):
        axes[3].text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    return fig

def visualize_attribute_alignment(alignment_results, location, attributes=None):
    """
    Create visualizations for attribute alignment analysis.
    
    Args:
        alignment_results (dict): Results from analyze_attribute_alignment function
        location (str): Location to visualize
        attributes (list, optional): Specific attributes to visualize, defaults to all
    
    Returns:
        matplotlib.figure.Figure: The figure containing the visualizations
    """
    if location not in alignment_results:
        print(f"Location {location} not found in alignment results")
        return None
    
    loc_results = alignment_results[location]
    
    if attributes is None:
        attributes = list(loc_results.keys())
    
    # Create a figure with subplots for each attribute
    fig, axes = plt.subplots(len(attributes), 1, figsize=(12, 5 * len(attributes)))
    
    # Handle single attribute case
    if len(attributes) == 1:
        axes = [axes]
    
    for i, attr in enumerate(attributes):
        if attr not in loc_results:
            continue
        
        attr_data = loc_results[attr]['attribute_data']
        
        # Sort by ideal share for better visualization
        attr_data = attr_data.sort_values('Ideal_So_Segment', ascending=False)
        
        # Set up bar positions
        x = np.arange(len(attr_data))
        width = 0.35
        
        # Plot real shares
        axes[i].bar(x - width/2, attr_data['Real_So_Segment'], width, label='Actual Share', color='blue')
        
        # Plot ideal shares
        axes[i].bar(x + width/2, attr_data['Ideal_So_Segment'], width, label='Ideal Share', color='green')
        
        # Add a line for the gap
        for j, row in attr_data.iterrows():
            gap = row['Delta_SoS']
            color = 'red' if gap < 0 else 'green'
            y_pos = max(row['Real_So_Segment'], row['Ideal_So_Segment']) + 0.02
            axes[i].text(attr_data.index.get_loc(j), y_pos, f"{gap:.2f}", 
                         ha='center', color=color, fontweight='bold')
        
        # Set title and labels
        axes[i].set_title(f'{location} - {attr} Distribution (Alignment Score: {loc_results[attr]["alignment_score"]:.2f})')
        axes[i].set_ylabel('Segment Share')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(attr_data[attr], rotation=45, ha='right')
        axes[i].legend()
        
        # Add annotations for under/over-represented attributes
        underrep = ", ".join(loc_results[attr]['underrepresented'][:3])
        overrep = ", ".join(loc_results[attr]['overrepresented'][:3])
        
        if underrep:
            axes[i].text(0.01, 0.95, f"Underrepresented: {underrep}", transform=axes[i].transAxes,
                        fontsize=10, verticalalignment='top', color='green')
        if overrep:
            axes[i].text(0.01, 0.90, f"Overrepresented: {overrep}", transform=axes[i].transAxes,
                        fontsize=10, verticalalignment='top', color='red')
    
    plt.tight_layout()
    return fig

def generate_category_c_report(validation_results, alignment_results):
    """
    Generate a comprehensive report on Category C validation and attribute alignment.
    
    Args:
        validation_results (dict): Results from validate_category_c_scoring function
        alignment_results (dict): Results from analyze_attribute_alignment function
    
    Returns:
        str: Formatted text report
    """
    report = "Category C (PARIS) Validation Report\n"
    report += "=" * 40 + "\n\n"
    
    # First section: Scoring validation
    report += "1. Category C Scoring Validation\n"
    report += "-" * 30 + "\n\n"
    
    for location, metrics in validation_results.items():
        if location == 'error':
            continue
            
        report += f"Location: {location}\n"
        report += f"Status: {metrics['status']}\n"
        report += f"Data Points: {metrics['data_points']}\n"
        report += f"Category C Score: {metrics['cat_c_score']:.2f}\n"
        report += f"Correlation: {metrics['correlation']:.4f}\n"
        report += f"R²: {metrics['r_squared']:.4f}\n"
        report += f"Average Real Segment Share: {metrics['avg_real_segment']:.4f}\n"
        report += f"Average Ideal Segment Share: {metrics['avg_ideal_segment']:.4f}\n"
        report += f"Sum of Positive Gaps: {metrics['positive_gaps_sum']:.4f}\n"
        report += f"Sum of Negative Gaps: {metrics['negative_gaps_sum']:.4f}\n"
        report += f"Standard Deviation of Gaps: {metrics['gap_std']:.4f}\n\n"
    
    # Second section: Attribute alignment
    report += "2. Attribute Alignment Analysis\n"
    report += "-" * 30 + "\n\n"
    
    for location, attributes in alignment_results.items():
        if location == 'error':
            continue
            
        report += f"Location: {location}\n"
        
        # Calculate overall alignment score as average of attribute alignment scores
        alignment_scores = [attr_data['alignment_score'] for attr_name, attr_data in attributes.items()]
        overall_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        report += f"Overall Alignment Score: {overall_score:.2f}\n\n"
        
        for attr_name, attr_data in attributes.items():
            report += f"  {attr_name}:\n"
            report += f"    Alignment Score: {attr_data['alignment_score']:.2f}\n"
            report += f"    Max Gap: {attr_data['max_gap']:.4f}\n"
            report += f"    Average Gap: {attr_data['avg_gap']:.4f}\n"
            
            if attr_data['underrepresented']:
                report += f"    Underrepresented: {', '.join(attr_data['underrepresented'][:5])}\n"
            
            if attr_data['overrepresented']:
                report += f"    Overrepresented: {', '.join(attr_data['overrepresented'][:5])}\n"
            
            report += "\n"
    
    # Third section: Kuwait vs. Jeju comparison if both exist
    if 'Kuwait' in validation_results and 'Jeju' in validation_results:
        report += "3. Kuwait vs. Jeju Comparison\n"
        report += "-" * 30 + "\n\n"
        
        kw_score = validation_results['Kuwait']['cat_c_score']
        jj_score = validation_results['Jeju']['cat_c_score']
        score_diff = kw_score - jj_score
        
        report += f"Category C Score Difference (Kuwait - Jeju): {score_diff:.2f}\n"
        report += f"Kuwait score is {kw_score/jj_score:.2f}x Jeju score\n\n"
        
        # Compare attribute alignment
        if 'Kuwait' in alignment_results and 'Jeju' in alignment_results:
            kw_alignment = alignment_results['Kuwait']
            jj_alignment = alignment_results['Jeju']
            
            common_attrs = set(kw_alignment.keys()).intersection(set(jj_alignment.keys()))
            
            for attr in common_attrs:
                kw_score = kw_alignment[attr]['alignment_score']
                jj_score = jj_alignment[attr]['alignment_score']
                
                report += f"{attr} Alignment Comparison:\n"
                report += f"  Kuwait: {kw_score:.2f}\n"
                report += f"  Jeju: {jj_score:.2f}\n"
                report += f"  Difference: {kw_score - jj_score:.2f}\n\n"
    
    return report

def run_category_c_validation(paris_output_path, output_dir=None):
    """
    Run the complete Category C validation process and output results.
    
    Args:
        paris_output_path (str): Path to PARIS_Output data
        output_dir (str, optional): Directory to save outputs
    
    Returns:
        tuple: (validation_results, alignment_results)
    """
    # Load data
    paris_df = load_paris_data('/Users/kemalgider/Desktop/PORTFOLIO/All_tables/data/PARIS_Output.csv')
    
    if paris_df is None
        return None, None
    
    # Validate Category C scoring
    validation_results = validate_category_c_scoring(paris_df)
    
    # Analyze attribute alignment
    alignment_results = analyze_attribute_alignment(paris_df)
    
    # Generate report
    report = generate_category_c_report(validation_results, alignment_results)
    print(report)
    
    # Create visualizations
    if 'Kuwait' in validation_results and 'Jeju' in validation_results:
        validation_fig = visualize_category_c_validation(validation_results, ['Kuwait', 'Jeju'])
        
        # Create attribute alignment visualizations for Kuwait and Jeju
        kw_alignment_fig = visualize_attribute_alignment(alignment_results, 'Kuwait')
        jj_alignment_fig = visualize_attribute_alignment(alignment_results, 'Jeju')
    else:
        validation_fig = visualize_category_c_validation(validation_results)
        
        # Use the first location for attribute alignment visualization
        first_location = next(iter([loc for loc in validation_results.keys() if loc != 'error']), None)
        if first_location:
            first_loc_fig = visualize_attribute_alignment(alignment_results, first_location)
    
    # Save outputs if directory is specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save report
        with open(os.path.join(output_dir, 'category_c_validation_report.txt'), 'w') as f:
            f.write(report)
        
        # Save validation visualization
        if validation_fig:
            validation_fig.savefig(os.path.join(output_dir, 'category_c_validation.png'), dpi=300, bbox_inches='tight')
        
        # Save attribute alignment visualizations
        if 'Kuwait' in alignment_results and kw_alignment_fig:
            kw_alignment_fig.savefig(os.path.join(output_dir, 'kuwait_attribute_alignment.png'), dpi=300, bbox_inches='tight')
        
        if 'Jeju' in alignment_results and jj_alignment_fig:
            jj_alignment_fig.savefig(os.path.join(output_dir, 'jeju_attribute_alignment.png'), dpi=300, bbox_inches='tight')
    
    return validation_results, alignment_results

# Example usage
if __name__ == "__main__":
    # This would be replaced with actual file paths
    paris_output_path = "PARIS_Output.csv"
    output_dir = "validation_results"
    
    validation_results, alignment_results = run_category_c_validation(paris_output_path, output_dir)
