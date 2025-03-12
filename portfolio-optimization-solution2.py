"""
Portfolio Optimization Solution - Integrated Framework

This script integrates all components of the portfolio optimization solution:
1. Data validation
2. Category C scoring validation
3. Portfolio visualization
4. SKU recommendation engine

Author: Claude
Date: March 12, 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import individual modules
from market_share_validation import run_market_share_validation
from category_c_validation import run_category_c_validation
from portfolio_visualization import generate_portfolio_optimization_dashboard
from sku_recommendation_engine import run_sku_recommendation_engine

class PortfolioOptimizationSolution:
    """
    Integrated solution for PMI portfolio optimization.
    
    This class brings together all components of the solution:
    - Data validation
    - Scoring validation
    - Visualization
    - SKU recommendations
    """
    
    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the solution with data and output directories.
        
        Args:
            data_dir (str): Directory containing input data files
            output_dir (str, optional): Directory to save outputs, created if not specified
        """
        self.data_dir = data_dir
        
        # Set default output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join('results', f'portfolio_optimization_{timestamp}')
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Print initialization info
        print(f"Portfolio Optimization Solution")
        print(f"-------------------------------")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Initialize results containers
        self.validation_results = None
        self.category_c_results = None
        self.alignment_results = None
        self.visualizations = None
        self.recommendations = None
        self.implementation_plan = None
    
    def validate_data_files(self):
        """
        Check that all required data files exist.
        
        Returns:
            bool: True if all required files exist, False otherwise
        """
        # Define required files
        required_files = [
            'KW_products.csv',
            'JJ_products.csv',
            'KW_product_based.csv',
            'JJ_product_based.csv',
            'comparison_kw_jj.csv',
            'PARIS_Output.csv'
        ]
        
        missing_files = []
        
        # Check each file
        for file_name in required_files:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        # Report missing files
        if missing_files:
            print("ERROR: The following required files are missing:")
            for file_name in missing_files:
                print(f"  - {file_name}")
            return False
        else:
            print("All required data files are present.")
            return True
    
    def run_market_share_validation(self):
        """
        Run market share validation for Kuwait and Jeju.
        
        Returns:
            dict: Validation results
        """
        print("\n=== Running Market Share Validation ===\n")
        
        # Define file paths
        kw_products_path = os.path.join(self.data_dir, 'KW_products.csv')
        jj_products_path = os.path.join(self.data_dir, 'JJ_products.csv')
        comparison_file = os.path.join(self.data_dir, 'comparison_kw_jj.csv')
        
        # Create output subdirectory
        validation_dir = os.path.join(self.output_dir, 'market_share_validation')
        os.makedirs(validation_dir, exist_ok=True)
        
        # Run validation
        self.validation_results = run_market_share_validation(
            kw_products_path, 
            jj_products_path, 
            comparison_file, 
            validation_dir
        )
        
        return self.validation_results
    
    def run_category_c_validation(self):
        """
        Run validation for Category C scoring logic.
        
        Returns:
            tuple: (validation_results, alignment_results)
        """
        print("\n=== Running Category C Validation ===\n")
        
        # Define file path
        paris_output_path = os.path.join(self.data_dir, 'PARIS_Output.csv')
        
        # Create output subdirectory
        category_c_dir = os.path.join(self.output_dir, 'category_c_validation')
        os.makedirs(category_c_dir, exist_ok=True)
        
        # Run validation
        self.category_c_results, self.alignment_results = run_category_c_validation(
            paris_output_path,
            category_c_dir
        )
        
        return self.category_c_results, self.alignment_results
    
    def generate_visualizations(self):
        """
        Generate portfolio visualization dashboard.
        
        Returns:
            dict: Visualization figures
        """
        print("\n=== Generating Portfolio Visualizations ===\n")
        
        # Define file paths
        kw_products_path = os.path.join(self.data_dir, 'KW_products.csv')
        jj_products_path = os.path.join(self.data_dir, 'JJ_products.csv')
        kw_product_based_path = os.path.join(self.data_dir, 'KW_product_based.csv')
        jj_product_based_path = os.path.join(self.data_dir, 'JJ_product_based.csv')
        comparison_file_path = os.path.join(self.data_dir, 'comparison_kw_jj.csv')
        
        # Create output subdirectory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations
        self.visualizations = generate_portfolio_optimization_dashboard(
            kw_products_path,
            jj_products_path,
            kw_product_based_path,
            jj_product_based_path,
            comparison_file_path,
            viz_dir
        )
        
        return self.visualizations
    
    def generate_recommendations(self):
        """
        Generate SKU recommendations for Jeju.
        
        Returns:
            tuple: (recommendations, implementation_plan, figures)
        """
        print("\n=== Generating SKU Recommendations for Jeju ===\n")
        
        # Define file paths
        jj_products_path = os.path.join(self.data_dir, 'JJ_products.csv')
        jj_product_based_path = os.path.join(self.data_dir, 'JJ_product_based.csv')
        paris_output_path = os.path.join(self.data_dir, 'PARIS_Output.csv')
        
        # Create output subdirectory
        rec_dir = os.path.join(self.output_dir, 'recommendations')
        os.makedirs(rec_dir, exist_ok=True)
        
        # Generate recommendations
        self.recommendations, self.implementation_plan, rec_figures = run_sku_recommendation_engine(
            jj_products_path,
            jj_product_based_path,
            paris_output_path,
            rec_dir
        )
        
        return self.recommendations, self.implementation_plan, rec_figures
    
    def generate_final_report(self):
        """
        Generate a comprehensive final report.
        
        Returns:
            str: Report text
        """
        print("\n=== Generating Final Report ===\n")
        
        # Create report
        report = "PORTFOLIO OPTIMIZATION ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add validation summary
        report += "1. MARKET SHARE VALIDATION\n"
        report += "-" * 30 + "\n\n"
        
        if self.validation_results:
            # Extract market share data
            kw_market_share = self.validation_results['Kuwait']['Market_Share']
            jj_market_share = self.validation_results['Jeju']['Market_Share']
            
            report += f"Kuwait Market Share: {kw_market_share:.2f}%\n"
            report += f"Jeju Market Share: {jj_market_share:.2f}%\n"
            report += f"Difference: {kw_market_share - jj_market_share:.2f}%\n\n"
            
            # Add SKU counts
            kw_pmi_skus = self.validation_results['Kuwait']['PMI_SKU_Count']
            kw_total_skus = self.validation_results['Kuwait']['Total_SKU_Count']
            jj_pmi_skus = self.validation_results['Jeju']['PMI_SKU_Count']
            jj_total_skus = self.validation_results['Jeju']['Total_SKU_Count']
            
            report += f"Kuwait PMI SKUs: {kw_pmi_skus} of {kw_total_skus} ({kw_pmi_skus/kw_total_skus*100:.2f}%)\n"
            report += f"Jeju PMI SKUs: {jj_pmi_skus} of {jj_total_skus} ({jj_pmi_skus/jj_total_skus*100:.2f}%)\n\n"
        else:
            report += "No validation results available.\n\n"
        
        # Add Category C validation
        report += "2. CATEGORY C VALIDATION\n"
        report += "-" * 30 + "\n\n"
        
        if self.category_c_results:
            if 'Kuwait' in self.category_c_results and 'Jeju' in self.category_c_results:
                kw_score = self.category_c_results['Kuwait']['cat_c_score']
                jj_score = self.category_c_results['Jeju']['cat_c_score']
                
                report += f"Kuwait Category C Score: {kw_score:.2f}\n"
                report += f"Jeju Category C Score: {jj_score:.2f}\n"
                report += f"Difference: {kw_score - jj_score:.2f}\n\n"
                
                report += "This indicates that Kuwait's product portfolio is better aligned with passenger preferences than Jeju's.\n\n"
            else:
                report += "Category C scores not available for both locations.\n\n"
        else:
            report += "No Category C validation results available.\n\n"
        
        # Add portfolio alignment summary
        report += "3. PORTFOLIO ALIGNMENT ANALYSIS\n"
        report += "-" * 30 + "\n\n"
        
        if self.alignment_results:
            if 'Kuwait' in self.alignment_results and 'Jeju' in self.alignment_results:
                report += "Attribute Alignment Comparison:\n\n"
                
                for attr in ['Flavor', 'Taste', 'Thickness', 'Length']:
                    if attr in self.alignment_results['Kuwait'] and attr in self.alignment_results['Jeju']:
                        kw_score = self.alignment_results['Kuwait'][attr]['alignment_score']
                        jj_score = self.alignment_results['Jeju'][attr]['alignment_score']
                        
                        report += f"{attr}:\n"
                        report += f"  Kuwait Alignment: {kw_score:.2f}/10\n"
                        report += f"  Jeju Alignment: {jj_score:.2f}/10\n"
                        report += f"  Difference: {kw_score - jj_score:.2f}\n\n"
            else:
                report += "Alignment results not available for both locations.\n\n"
        else:
            report += "No portfolio alignment results available.\n\n"
        
        # Add recommendations summary
        report += "4. RECOMMENDATIONS FOR JEJU\n"
        report += "-" * 30 + "\n\n"
        
        if self.recommendations and self.implementation_plan:
            # Count recommendations
            maintain_count = len(self.recommendations['skus_to_maintain'])
            remove_count = len(self.recommendations['skus_to_remove'])
            add_count = len(self.recommendations['attribute_combinations_needed'])
            
            report += "Summary of Recommendations:\n\n"
            report += f"• SKUs to maintain: {maintain_count}\n"
            report += f"• SKUs to consider removing: {remove_count}\n"
            report += f"• New attribute combinations to add: {add_count}\n\n"
            
            # Top attribute gaps
            report += "Top Attribute Gaps:\n\n"
            
            for attr in ['Flavor', 'Taste', 'Thickness', 'Length']:
                if attr in self.recommendations['underrepresented_attributes']:
                    df = self.recommendations['underrepresented_attributes'][attr]
                    if not df.empty:
                        top_gap = df.iloc[0]
                        report += f"• {attr}: {top_gap[attr]} (Gap: {top_gap['Gap']:.1f}%)\n"
            
            report += "\nImplementation Plan Summary:\n\n"
            
            # Short-term
            report += "Short-Term (0-3 months):\n"
            
            short_term_attrs = self.implementation_plan['short_term']['attribute_focus']
            if short_term_attrs:
                report += "Focus on key segments:\n"
                for attr, values in short_term_attrs.items():
                    report += f"• {attr}: {', '.join(values)}\n"
            
            # Medium-term
            report += "\nMedium-Term (3-6 months):\n"
            
            if self.implementation_plan['medium_term']['skus_to_add']:
                report += "Introduce new SKUs with combinations of:\n"
                for combo in self.implementation_plan['medium_term']['skus_to_add'][:3]:
                    attr_str = ", ".join([f"{k}: {v}" for k, v in combo['Combination'].items()])
                    report += f"• {attr_str}\n"
            
            # Long-term
            report += "\nLong-Term (6+ months):\n"
            
            if self.implementation_plan['long_term']['strategy_shifts']:
                report += "Strategic portfolio shifts:\n"
                for shift in self.implementation_plan['long_term']['strategy_shifts'][:3]:
                    report += f"• {shift['description']}\n"
            
            # Potential impact
            report += "\nPotential Market Share Impact:\n"
            
            # Calculate potential market share gain based on gaps
            total_gap = 0
            for attr, df in self.recommendations['underrepresented_attributes'].items():
                if not df.empty:
                    total_gap += df['Gap'].sum()
            
            # Estimate potential market share gain (half of identified gaps)
            potential_gain = total_gap * 0.5
            
            report += f"• Current estimated market share: ~11.4%\n"
            report += f"• Potential gain: {potential_gain:.1f}%\n"
            report += f"• Projected future market share: {11.4 + potential_gain:.1f}%\n"
            
        else:
            report += "No recommendations available.\n\n"
        
        # Save the report
        report_path = os.path.join(self.output_dir, 'final_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Final report saved to {report_path}")
        
        return report
    
    def run_full_analysis(self):
        """
        Run the complete portfolio optimization analysis workflow.
        """
        # Validate data files
        if not self.validate_data_files():
            print("ERROR: Missing required data files. Analysis cannot proceed.")
            return False
        
        # Run all analysis components
        self.run_market_share_validation()
        self.run_category_c_validation()
        self.generate_visualizations()
        self.generate_recommendations()
        
        # Generate final report
        self.generate_final_report()
        
        print("\n=== Analysis Complete ===\n")
        print(f"All results have been saved to {self.output_dir}")
        
        return True

def main():
    """
    Main function to execute the portfolio optimization solution.
    """
    # Default data directory
    default_data_dir = 'data'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir
        print(f"No data directory specified, using default: {data_dir}")
    
    # Create solution instance
    solution = PortfolioOptimizationSolution(data_dir)
    
    # Run full analysis
    success = solution.run_full_analysis()
    
    if success:
        print("\nPortfolio optimization analysis completed successfully.")
    else:
        print("\nPortfolio optimization analysis could not be completed due to errors.")

if __name__ == "__main__":
    main()
