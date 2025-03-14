"""
Portfolio Analysis and Visualization Testing 

This script provides testing functions to validate the portfolio visualization
implementation with the actual project data. It confirms that the visualization
accurately represents the product portfolio alignment between the well-aligned
Kuwait market and the misaligned Jeju market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
from unittest.mock import patch

# Add the main module path to import
sys.path.append('.')
from portfolio_visualization import (
    load_location_data,
    load_comparison_data,
    load_validation_data,
    create_comparative_portfolio_grid
)

def test_data_loading():
    """Test data loading functions with sample data files"""
    # Define data directories
    data_dir = Path('./locations_data')
    
    # Set up file paths for Kuwait data
    kuwait_files = {
        'Flavor_Distribution': data_dir / 'Kuwait_product_analysis_Flavor_Distribution.csv',
        'Taste_Distribution': data_dir / 'Kuwait_product_analysis_Taste_Distribution.csv',
        'PMI_Products': data_dir / 'Kuwait_product_analysis_PMI_Products.csv',
    }
    
    # Set up file paths for Jeju data
    jeju_files = {
        'Flavor_Distribution': data_dir / 'jeju_product_analysis_Flavor_Distribution.csv',
        'Taste_Distribution': data_dir / 'jeju_product_analysis_Taste_Distribution.csv',
        'PMI_Products': data_dir / 'jeju_product_analysis_PMI_Products.csv',
    }
    
    # Organize data files
    location_data_files = {
        'Kuwait': kuwait_files,
        'Jeju': jeju_files
    }
    
    # Test location data loading
    with patch('builtins.print'):  # Suppress print statements
        location_data = load_location_data(location_data_files)
    
    # Verify data loading
    assert 'Kuwait' in location_data
    assert 'Jeju' in location_data
    assert 'Flavor_Distribution' in location_data['Kuwait']
    assert 'Flavor_Distribution' in location_data['Jeju']
    
    # Check data content
    kuwait_flavor = location_data['Kuwait']['Flavor_Distribution']
    assert 'Flavor' in kuwait_flavor.columns
    assert 'Volume_Percentage' in kuwait_flavor.columns
    
    print("Data loading test passed!")


def test_comparative_grid_creation():
    """Test the creation of comparative portfolio grid"""
    # Define data directories
    data_dir = Path('./locations_data')
    
    # Set up minimal test data
    test_data = {
        'Kuwait': {
            'Flavor_Distribution': pd.DataFrame({
                'Flavor': ['Regular', 'Menthol'],
                'Volume_Percentage': [95.0, 5.0],
                'PMI_Volume_Percentage': [96.0, 4.0],
                'Ideal_Percentage': [92.0, 8.0],
                'Market_vs_Ideal_Gap': [3.0, -3.0]
            }),
            'Taste_Distribution': pd.DataFrame({
                'Taste': ['Full Flavor', 'Lights'],
                'Volume_Percentage': [80.0, 20.0],
                'PMI_Volume_Percentage': [82.0, 18.0],
                'Ideal_Percentage': [78.0, 22.0],
                'Market_vs_Ideal_Gap': [2.0, -2.0]
            })
        },
        'Jeju': {
            'Flavor_Distribution': pd.DataFrame({
                'Flavor': ['Regular', 'Menthol'],
                'Volume_Percentage': [90.0, 10.0],
                'PMI_Volume_Percentage': [95.0, 5.0],
                'Ideal_Percentage': [70.0, 30.0],
                'Market_vs_Ideal_Gap': [20.0, -20.0]
            }),
            'Taste_Distribution': pd.DataFrame({
                'Taste': ['Full Flavor', 'Lights'],
                'Volume_Percentage': [95.0, 5.0],
                'PMI_Volume_Percentage': [98.0, 2.0],
                'Ideal_Percentage': [70.0, 30.0],
                'Market_vs_Ideal_Gap': [25.0, -25.0]
            })
        }
    }
    
    # Test grid creation with minimal data
    with patch('builtins.print'):  # Suppress print statements
        fig = create_comparative_portfolio_grid(test_data, attributes=['Flavor', 'Taste'])
    
    # Verify figure creation
    assert fig is not None
    
    # Save test output
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / 'test_portfolio_grid.png', dpi=300, bbox_inches='tight')
    
    print("Comparative grid creation test passed!")


def validate_kuwait_alignment():
    """Validate Kuwait's portfolio alignment metrics"""
    # Define data directory
    data_dir = Path('./locations_data')
    
    # Load Kuwait data
    kuwait_flavor = pd.read_csv(data_dir / 'Kuwait_product_analysis_Flavor_Distribution.csv')
    kuwait_taste = pd.read_csv(data_dir / 'Kuwait_product_analysis_Taste_Distribution.csv')
    kuwait_thickness = pd.read_csv(data_dir / 'Kuwait_product_analysis_Thickness_Distribution.csv')
    kuwait_length = pd.read_csv(data_dir / 'Kuwait_product_analysis_Length_Distribution.csv')
    
    # Calculate alignment metrics
    alignment_scores = {}
    
    # Flavor alignment
    flavor_gaps = kuwait_flavor['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Flavor'] = 10 - min(flavor_gaps, 10)
    
    # Taste alignment
    taste_gaps = kuwait_taste['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Taste'] = 10 - min(taste_gaps, 10)
    
    # Thickness alignment
    thickness_gaps = kuwait_thickness['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Thickness'] = 10 - min(thickness_gaps, 10)
    
    # Length alignment
    length_gaps = kuwait_length['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Length'] = 10 - min(length_gaps, 10)
    
    # Output alignment scores
    expected_scores = {
        'Flavor': 9.64,
        'Taste': 8.10,
        'Thickness': 5.03,
        'Length': 8.17
    }
    
    print("Kuwait Alignment Validation:")
    print("============================")
    print(f"{'Attribute':<10} {'Calculated':<12} {'Expected':<12} {'Diff':<10}")
    print("-" * 44)
    
    for attr in alignment_scores:
        diff = abs(alignment_scores[attr] - expected_scores[attr])
        status = "OK" if diff < 1.0 else "CHECK"
        print(f"{attr:<10} {alignment_scores[attr]:<12.2f} {expected_scores[attr]:<12.2f} {diff:<10.2f} {status}")
    
    # Overall alignment score
    calculated_overall = sum(alignment_scores.values()) / len(alignment_scores)
    expected_overall = 7.73
    overall_diff = abs(calculated_overall - expected_overall)
    overall_status = "OK" if overall_diff < 0.5 else "CHECK"
    
    print("-" * 44)
    print(f"{'Overall':<10} {calculated_overall:<12.2f} {expected_overall:<12.2f} {overall_diff:<10.2f} {overall_status}")


def validate_jeju_alignment():
    """Validate Jeju's portfolio alignment metrics"""
    # Define data directory
    data_dir = Path('./locations_data')
    
    # Load Jeju data
    jeju_flavor = pd.read_csv(data_dir / 'jeju_product_analysis_Flavor_Distribution.csv')
    jeju_taste = pd.read_csv(data_dir / 'jeju_product_analysis_Taste_Distribution.csv')
    jeju_thickness = pd.read_csv(data_dir / 'jeju_product_analysis_Thickness_Distribution.csv')
    jeju_length = pd.read_csv(data_dir / 'jeju_product_analysis_Length_Distribution.csv')
    
    # Calculate alignment metrics
    alignment_scores = {}
    
    # Flavor alignment
    flavor_gaps = jeju_flavor['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Flavor'] = 10 - min(flavor_gaps, 10)
    
    # Taste alignment
    taste_gaps = jeju_taste['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Taste'] = 10 - min(taste_gaps, 10)
    
    # Thickness alignment
    thickness_gaps = jeju_thickness['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Thickness'] = 10 - min(thickness_gaps, 10)
    
    # Length alignment
    length_gaps = jeju_length['Market_vs_Ideal_Gap'].abs().mean()
    alignment_scores['Length'] = 10 - min(length_gaps, 10)
    
    # Output alignment scores
    expected_scores = {
        'Flavor': 7.53,
        'Taste': 4.37,
        'Thickness': 5.82,
        'Length': 6.38
    }
    
    print("\nJeju Alignment Validation:")
    print("==========================")
    print(f"{'Attribute':<10} {'Calculated':<12} {'Expected':<12} {'Diff':<10}")
    print("-" * 44)
    
    for attr in alignment_scores:
        diff = abs(alignment_scores[attr] - expected_scores[attr])
        status = "OK" if diff < 1.0 else "CHECK"
        print(f"{attr:<10} {alignment_scores[attr]:<12.2f} {expected_scores[attr]:<12.2f} {diff:<10.2f} {status}")
    
    # Overall alignment score
    calculated_overall = sum(alignment_scores.values()) / len(alignment_scores)
    expected_overall = 6.02
    overall_diff = abs(calculated_overall - expected_overall)
    overall_status = "OK" if overall_diff < 0.5 else "CHECK"
    
    print("-" * 44)
    print(f"{'Overall':<10} {calculated_overall:<12.2f} {expected_overall:<12.2f} {overall_diff:<10.2f} {overall_status}")


def analyze_key_misalignments(location):
    """Analyze key misalignments for a specific location"""
    # Define data directory
    data_dir = Path('./locations_data')
    
    # Load location data
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    misalignments = []
    
    for attr in attributes:
        try:
            if location == 'Kuwait':
                df = pd.read_csv(data_dir / f'Kuwait_product_analysis_{attr}_Distribution.csv')
            else:
                df = pd.read_csv(data_dir / f'jeju_product_analysis_{attr}_Distribution.csv')
                
            # Find key misalignments (largest gaps)
            if 'PMI_vs_Ideal_Gap' in df.columns:
                # Find negative gaps (underrepresented segments)
                under_rep = df[df['PMI_vs_Ideal_Gap'] < -5].sort_values('PMI_vs_Ideal_Gap')
                
                for _, row in under_rep.iterrows():
                    misalignments.append({
                        'Attribute': attr,
                        'Value': row[attr],
                        'PMI_Percentage': row['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in df.columns else 0,
                        'Ideal_Percentage': row['Ideal_Percentage'],
                        'Gap': row['PMI_vs_Ideal_Gap'],
                        'Type': 'Underrepresented'
                    })
                
                # Find positive gaps (overrepresented segments)
                over_rep = df[df['PMI_vs_Ideal_Gap'] > 5].sort_values('PMI_vs_Ideal_Gap', ascending=False)
                
                for _, row in over_rep.iterrows():
                    misalignments.append({
                        'Attribute': attr,
                        'Value': row[attr],
                        'PMI_Percentage': row['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in df.columns else 0,
                        'Ideal_Percentage': row['Ideal_Percentage'],
                        'Gap': row['PMI_vs_Ideal_Gap'],
                        'Type': 'Overrepresented'
                    })
        except Exception as e:
            print(f"Error analyzing {attr} for {location}: {e}")
    
    # Sort by absolute gap value
    misalignments = sorted(misalignments, key=lambda x: abs(x['Gap']), reverse=True)
    
    # Output key misalignments
    print(f"\nKey Portfolio Misalignments for {location}:")
    print("===========================================")
    print(f"{'Attribute':<10} {'Value':<15} {'PMI %':<10} {'Ideal %':<10} {'Gap':<10} {'Type':<15}")
    print("-" * 70)
    
    for m in misalignments[:10]:  # Show top 10 misalignments
        print(f"{m['Attribute']:<10} {str(m['Value']):<15} {m['PMI_Percentage']:<10.1f} {m['Ideal_Percentage']:<10.1f} {m['Gap']:<10.1f} {m['Type']:<15}")


def main():
    """Main function to run portfolio analysis tests"""
    print("Running Portfolio Analysis Tests...\n")
    
    # Test data loading
    test_data_loading()
    
    # Test comparative grid creation
    test_comparative_grid_creation()
    
    # Validate alignment metrics
    validate_kuwait_alignment()
    validate_jeju_alignment()
    
    # Analyze key misalignments
    analyze_key_misalignments('Kuwait')
    analyze_key_misalignments('Jeju')
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
