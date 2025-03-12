import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_product_data(kw_products_path, jj_products_path):
    """
    Load the product data for Kuwait and Jeju.
    Returns dataframes with product information.
    """
    print("Loading product data for Kuwait and Jeju...")
    
    # Reading Kuwait products data
    kw_df = pd.read_excel(kw_products_path)
    # Reading Jeju products data
    jj_df = pd.read_excel(jj_products_path)
    
    return kw_df, jj_df

def validate_market_share(kw_df, jj_df, comparison_file=None):
    """
    Calculate and validate market share for Kuwait and Jeju.
    """
    results = {
        'Kuwait': {
            'PMI_Volume': 0,
            'Total_Volume': 0,
            'Market_Share': 0,
            'PMI_SKU_Count': 0,
            'Total_SKU_Count': 0
        },
        'Jeju': {
            'PMI_Volume': 0,
            'Total_Volume': 0,
            'Market_Share': 0,
            'PMI_SKU_Count': 0,
            'Total_SKU_Count': 0
        }
    }

    # Calculate Kuwait market share
    try:
        # Check if required columns exist
        required_cols = ['TMO', 'DF_Vol', 'CR_BrandId']
        for df_name, df in [('Kuwait', kw_df), ('Jeju', jj_df)]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: {df_name} data is missing columns: {', '.join(missing_cols)}")
                continue

            # Process Kuwait data
            if df_name == 'Kuwait' and all(col in kw_df.columns for col in required_cols):
                kw_pmi_volume = kw_df[kw_df['TMO'] == 'PMI']['DF_Vol'].sum()
                kw_total_volume = kw_df['DF_Vol'].sum()
                kw_market_share = (kw_pmi_volume / kw_total_volume) * 100 if kw_total_volume > 0 else 0

                results['Kuwait'] = {
                    'PMI_Volume': kw_pmi_volume,
                    'Total_Volume': kw_total_volume,
                    'Market_Share': kw_market_share,
                    'PMI_SKU_Count': kw_df[kw_df['TMO'] == 'PMI']['CR_BrandId'].nunique(),
                    'Total_SKU_Count': kw_df['CR_BrandId'].nunique()
                }

            # Process Jeju data
            if df_name == 'Jeju' and all(col in jj_df.columns for col in required_cols):
                jj_pmi_volume = jj_df[jj_df['TMO'] == 'PMI']['DF_Vol'].sum()
                jj_total_volume = jj_df['DF_Vol'].sum()
                jj_market_share = (jj_pmi_volume / jj_total_volume) * 100 if jj_total_volume > 0 else 0

                results['Jeju'] = {
                    'PMI_Volume': jj_pmi_volume,
                    'Total_Volume': jj_total_volume,
                    'Market_Share': jj_market_share,
                    'PMI_SKU_Count': jj_df[jj_df['TMO'] == 'PMI']['CR_BrandId'].nunique(),
                    'Total_SKU_Count': jj_df['CR_BrandId'].nunique()
                }

        # Compare with reference data if provided
        if comparison_file:
            try:
                comp_df = pd.read_excel(comparison_file)

                for location in ['Kuwait', 'Jeju']:
                    ref_data = comp_df[comp_df['Location'] == location]
                    if not ref_data.empty:
                        ref_market_share = ref_data['Market_Share'].values[0]
                        calculated_share = results[location]['Market_Share']

                        # Calculate discrepancy
                        discrepancy = abs(calculated_share - ref_market_share)
                        results[location]['Reference_Market_Share'] = ref_market_share
                        results[location]['Discrepancy'] = discrepancy
                        results[location]['Validation_Status'] = 'Valid' if discrepancy < 1.0 else 'Warning'
            except Exception as e:
                print(f"Error comparing with reference data: {e}")
                results['error_reference'] = str(e)

    except Exception as e:
        print(f"Error calculating market share: {e}")
        results['error'] = str(e)

    return results

def visualize_market_shares(validation_results):
    """
    Create visualizations for market share comparison.
    
    Args:
        validation_results (dict): Results from validate_market_share function
    
    Returns:
        matplotlib.figure.Figure: The figure containing the visualizations
    """
    # Extract data
    locations = list(validation_results.keys())
    market_shares = [validation_results[loc]['Market_Share'] for loc in locations]
    pmi_volumes = [validation_results[loc]['PMI_Volume'] for loc in locations]
    total_volumes = [validation_results[loc]['Total_Volume'] for loc in locations]
    pmi_skus = [validation_results[loc]['PMI_SKU_Count'] for loc in locations]
    total_skus = [validation_results[loc]['Total_SKU_Count'] for loc in locations]
    
    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot market share comparison
    axes[0].bar(locations, market_shares, color=['green', 'red'])
    axes[0].set_title('Market Share Comparison')
    axes[0].set_ylabel('Market Share (%)')
    axes[0].set_ylim(0, 100)
    
    # Add value labels on bars
    for i, v in enumerate(market_shares):
        axes[0].text(i, v + 3, f"{v:.1f}%", ha='center')
    
    # Plot volume comparison
    x = np.arange(len(locations))
    width = 0.35
    axes[1].bar(x - width/2, pmi_volumes, width, label='PMI Volume')
    axes[1].bar(x + width/2, [total-pmi for total, pmi in zip(total_volumes, pmi_volumes)], width, label='Competitor Volume')
    axes[1].set_title('Volume Comparison')
    axes[1].set_ylabel('Volume')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(locations)
    axes[1].legend()
    
    # Plot SKU count comparison
    axes[2].bar(x - width/2, pmi_skus, width, label='PMI SKUs')
    axes[2].bar(x + width/2, [total-pmi for total, pmi in zip(total_skus, pmi_skus)], width, label='Competitor SKUs')
    axes[2].set_title('SKU Count Comparison')
    axes[2].set_ylabel('Number of SKUs')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(locations)
    axes[2].legend()
    
    plt.tight_layout()
    return fig
def generate_market_share_report(validation_results):
    """
    Generate a comprehensive report on market share validation.

    Args:
        validation_results (dict): Results from validate_market_share function

    Returns:
        str: Formatted text report
    """
    report = "Market Share Validation Report\n"
    report += "=" * 30 + "\n\n"

    for location, metrics in validation_results.items():
        if location in ['Kuwait', 'Jeju']:
            report += f"Location: {location}\n"
            report += "-" * 20 + "\n"
            report += f"PMI Volume: {metrics['PMI_Volume']:,.0f}\n"
            report += f"Total Market Volume: {metrics['Total_Volume']:,.0f}\n"
            report += f"Market Share: {metrics['Market_Share']:.2f}%\n"
            report += f"PMI SKU Count: {metrics['PMI_SKU_Count']}\n"
            report += f"Total SKU Count: {metrics['Total_SKU_Count']}\n"

            if 'Reference_Market_Share' in metrics:
                report += f"Reference Market Share: {metrics['Reference_Market_Share']:.2f}%\n"
                report += f"Discrepancy: {metrics['Discrepancy']:.2f}%\n"
                report += f"Validation Status: {metrics['Validation_Status']}\n"

            report += "\n"

    # Calculate key insights
    kw_share = validation_results['Kuwait']['Market_Share']
    jj_share = validation_results['Jeju']['Market_Share']
    share_diff = kw_share - jj_share

    report += "Key Insights:\n"
    report += "-" * 20 + "\n"
    report += f"Market Share Difference (Kuwait - Jeju): {share_diff:.2f}%\n"

    # Avoid division by zero
    if jj_share > 0:
        report += f"Kuwait has {share_diff/jj_share:.1f}x the market share of Jeju\n"
    else:
        report += "Cannot calculate ratio: Jeju market share is zero\n"

    # Calculate PMI SKU ratio to total SKUs - avoid division by zero
    kw_total = validation_results['Kuwait']['Total_SKU_Count']
    jj_total = validation_results['Jeju']['Total_SKU_Count']

    if kw_total > 0:
        kw_sku_ratio = validation_results['Kuwait']['PMI_SKU_Count'] / kw_total
        report += f"Kuwait PMI SKU ratio: {kw_sku_ratio:.2f} ({validation_results['Kuwait']['PMI_SKU_Count']} of {kw_total})\n"
    else:
        report += "Kuwait PMI SKU ratio: N/A (no SKUs found)\n"

    if jj_total > 0:
        jj_sku_ratio = validation_results['Jeju']['PMI_SKU_Count'] / jj_total
        report += f"Jeju PMI SKU ratio: {jj_sku_ratio:.2f} ({validation_results['Jeju']['PMI_SKU_Count']} of {jj_total})\n"
    else:
        report += "Jeju PMI SKU ratio: N/A (no SKUs found)\n"

    return report

# Main execution function
def run_market_share_validation(kw_products_path, jj_products_path, comparison_file=None, output_dir=None):
    """
    Run the complete market share validation process and output results.
    
    Args:
        kw_products_path (str): Path to Kuwait products data
        jj_products_path (str): Path to Jeju products data
        comparison_file (str, optional): Path to comparison reference file
        output_dir (str, optional): Directory to save outputs
    
    Returns:
        dict: Validation results
    """
    # Load data
    kw_df, jj_df = load_product_data(kw_products_path, jj_products_path)
    
    # Validate market share
    validation_results = validate_market_share(kw_df, jj_df, comparison_file)
    
    # Generate report
    report = generate_market_share_report(validation_results)
    print(report)
    
    # Create visualizations
    fig = visualize_market_shares(validation_results)
    
    # Save outputs if directory is specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save report
        with open(os.path.join(output_dir, 'market_share_validation_report.txt'), 'w') as f:
            f.write(report)
        
        # Save visualization
        fig.savefig(os.path.join(output_dir, 'market_share_comparison.png'), dpi=300, bbox_inches='tight')
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # This would be replaced with actual file paths
    kw_products_path = "KW_products.xlsx"
    jj_products_path = "JJ_products.xlsx"
    comparison_file = "comp_jj_kw.xlsx"
    output_dir = "validation_results"
    
    validation_results = run_market_share_validation(kw_products_path, jj_products_path, comparison_file, output_dir)
