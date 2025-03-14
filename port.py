import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path
import json
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('portfolio_shelf_dash')

# Enable debug mode
DEBUG = True

def debug_print(message, data=None):
    """Print debug messages"""
    if DEBUG:
        logger.info(message)
        if data is not None:
            if isinstance(data, pd.DataFrame):
                logger.info(f"DataFrame shape: {data.shape}")
                logger.info(f"DataFrame columns: {data.columns.tolist()}")
                logger.info(f"DataFrame sample: \n{data.head(2)}")
            else:
                logger.info(f"Data: {data}")

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>PMI Portfolio Shelf Visualization</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: "Segoe UI", Arial, sans-serif;
                margin: 0;
                background-color: #f7f7f7;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 1rem;
                margin-bottom: 2rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .title {
                margin: 0;
                padding: 0;
                font-size: 24px;
                font-weight: bold;
            }
            .subtitle {
                margin: 5px 0 0 0;
                font-size: 16px;
                opacity: 0.9;
            }
            .dashboard-container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 0 20px;
            }
            .card {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .card-title {
                margin-top: 0;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
                font-size: 18px;
                color: #2c3e50;
            }
            .metric-card {
                text-align: center;
                padding: 15px 5px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            .metric-label {
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }
            .tab-content {
                padding: 20px;
                background-color: white;
                border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .info-box {
                background-color: #ebf5fb;
                border-left: 4px solid #3498db;
                padding: 10px 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .warning-box {
                background-color: #fef9e7;
                border-left: 4px solid #f1c40f;
                padding: 10px 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .error-box {
                background-color: #fdedec;
                border-left: 4px solid #e74c3c;
                padding: 10px 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .graph-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .debug-info {
                background-color: #eee;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                font-family: monospace;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="dashboard-container">
                <h1 class="title">PMI Portfolio Shelf Visualization</h1>
                <p class="subtitle">Interactive visualization of product portfolio alignment</p>
            </div>
        </div>
        <div class="dashboard-container">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Helper functions for data loading and processing
def load_location_data(data_dir="./locations_data"):
    """Load data from the locations_data directory with debugging info"""

    debug_print(f"Loading data from {data_dir}")

    data = {
        'Kuwait': {},
        'Jeju': {}
    }

    # Define file mappings with explicit column mapping
    file_mappings = {
        'Kuwait': {
            'flavor': {'file': 'Kuwait_product_analysis_Flavor_Distribution.csv', 'column_map': {'Flavor': 'flavor'}},
            'taste': {'file': 'Kuwait_product_analysis_Taste_Distribution.csv', 'column_map': {'Taste': 'taste'}},
            'thickness': {'file': 'Kuwait_product_analysis_Thickness_Distribution.csv',
                          'column_map': {'Thickness': 'thickness'}},
            'length': {'file': 'Kuwait_product_analysis_Length_Distribution.csv', 'column_map': {'Length': 'length'}},
            'pmi_products': {'file': 'Kuwait_product_analysis_PMI_Products.csv', 'column_map': {}},
            'top_90pct_products': {'file': 'Kuwait_product_analysis_Top_90pct_Products.csv', 'column_map': {}},
            'summary': {'file': 'Kuwait_product_analysis_Summary.csv', 'column_map': {}},
            'passenger': {'file': 'Kuwait_product_analysis_Passenger_Distribution.csv', 'column_map': {}}
        },
        'Jeju': {
            'flavor': {'file': 'jeju_product_analysis_Flavor_Distribution.csv', 'column_map': {'Flavor': 'flavor'}},
            'taste': {'file': 'jeju_product_analysis_Taste_Distribution.csv', 'column_map': {'Taste': 'taste'}},
            'thickness': {'file': 'jeju_product_analysis_Thickness_Distribution.csv',
                          'column_map': {'Thickness': 'thickness'}},
            'length': {'file': 'jeju_product_analysis_Length_Distribution.csv', 'column_map': {'Length': 'length'}},
            'pmi_products': {'file': 'jeju_product_analysis_PMI_Products.csv', 'column_map': {}},
            'top_90pct_products': {'file': 'jeju_product_analysis_Top_90pct_Products.csv', 'column_map': {}},
            'summary': {'file': 'jeju_product_analysis_Summary.csv', 'column_map': {}}
        }
    }

    # Add specific column mappings for product files
    for location in ['Kuwait', 'Jeju']:
        for product_key in ['pmi_products', 'top_90pct_products']:
            file_mappings[location][product_key]['column_map'] = {
                'Flavor': 'flavor',
                'Taste': 'taste',
                'Thickness': 'thickness',
                'Length': 'length'
            }

    # Load data for each location
    for location, file_map in file_mappings.items():
        for key, file_info in file_map.items():
            file_path = os.path.join(data_dir, file_info['file'])
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    debug_print(f"Loaded {location} {key} data", df)

                    # Apply column mapping
                    for original_col, target_col in file_info['column_map'].items():
                        if original_col in df.columns:
                            debug_print(f"Renaming column {original_col} to {target_col} in {location} {key}")
                            df[target_col] = df[original_col]

                    data[location][key] = df
                except Exception as e:
                    debug_print(f"Error loading {location} {key} data: {str(e)}")
                    debug_print(traceback.format_exc())

    # Load comparison data
    comparison_files = {
        'flavor': {'file': 'kuwait_jeju_attribute_analysis_Flavor_Distribution.csv',
                   'column_map': {'Flavor': 'flavor'}},
        'taste': {'file': 'kuwait_jeju_attribute_analysis_Taste_Distribution.csv', 'column_map': {'Taste': 'taste'}},
        'thickness': {'file': 'kuwait_jeju_attribute_analysis_Thickness_Distribution.csv',
                      'column_map': {'Thickness': 'thickness'}},
        'length': {'file': 'kuwait_jeju_attribute_analysis_Length_Distribution.csv', 'column_map': {'Length': 'length'}}
    }

    data['comparison'] = {}
    for key, file_info in comparison_files.items():
        file_path = os.path.join(data_dir, file_info['file'])
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                debug_print(f"Loaded comparison {key} data", df)

                # Apply column mapping
                for original_col, target_col in file_info['column_map'].items():
                    if original_col in df.columns:
                        debug_print(f"Renaming column {original_col} to {target_col} in comparison {key}")
                        df[target_col] = df[original_col]

                data['comparison'][key] = df
            except Exception as e:
                debug_print(f"Error loading comparison {key} data: {str(e)}")
                debug_print(traceback.format_exc())

    # Load gap data for each location
    for location in ['Kuwait', 'Jeju']:
        data[location]['gaps'] = {}
        gap_file_mappings = {
            'flavor': {'file': f'kuwait_jeju_attribute_analysis_{location}_Flavor_Gaps.csv',
                       'column_map': {'Flavor': 'flavor'}},
            'taste': {'file': f'kuwait_jeju_attribute_analysis_{location}_Taste_Gaps.csv',
                      'column_map': {'Taste': 'taste'}},
            'thickness': {'file': f'kuwait_jeju_attribute_analysis_{location}_Thickness_Gaps.csv',
                          'column_map': {'Thickness': 'thickness'}},
            'length': {'file': f'kuwait_jeju_attribute_analysis_{location}_Length_Gaps.csv',
                       'column_map': {'Length': 'length'}}
        }

        for key, file_info in gap_file_mappings.items():
            file_path = os.path.join(data_dir, file_info['file'])
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    debug_print(f"Loaded {location} {key} gap data", df)

                    # Apply column mapping
                    for original_col, target_col in file_info['column_map'].items():
                        if original_col in df.columns:
                            debug_print(f"Renaming column {original_col} to {target_col} in {location} {key} gaps")
                            df[target_col] = df[original_col]

                    data[location]['gaps'][key] = df
                except Exception as e:
                    debug_print(f"Error loading {location} {key} gap data: {str(e)}")
                    debug_print(traceback.format_exc())

    # Load comparison summary
    comparison_summary_file = os.path.join(data_dir, 'kuwait_jeju_comparison_summary.csv')
    if os.path.exists(comparison_summary_file):
        try:
            data['comparison_summary'] = pd.read_csv(comparison_summary_file)
            debug_print("Loaded comparison summary data", data['comparison_summary'])
        except Exception as e:
            debug_print(f"Error loading comparison summary data: {str(e)}")
            debug_print(traceback.format_exc())

    return data

def load_main_data(data_dir="./main_data"):
    """Load data from the main_data directory (SQL outputs)"""

    data = {}

    # Check if main_data directory exists
    if not os.path.exists(data_dir):
        debug_print(f"Main data directory {data_dir} not found")
        return data

    # Try to load JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    debug_print(f"Found JSON files: {json_files}")

    for file in json_files:
        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, 'r') as f:
                data[file] = json.load(f)
            debug_print(f"Loaded {file}")
        except Exception as e:
            debug_print(f"Could not load {file}: {str(e)}")
            debug_print(traceback.format_exc())

    return data

def load_validation_data(data_dir="./locations_data"):
    """Load Category C validation data"""

    data = {
        'Kuwait': {},
        'Jeju': {}
    }

    # Try to load validation text file
    validation_file = os.path.join(data_dir, 'cat_c_validation.txt')
    if not os.path.exists(validation_file):
        debug_print(f"Validation file {validation_file} not found")
        return data

    # Parse the validation file
    try:
        with open(validation_file, 'r') as f:
            content = f.read()
            debug_print("Validation file content length", len(content))

            # Extract Kuwait data
            kuwait_section = content.split("Location: Kuwait")[1].split("Location: Jeju")[
                0] if "Location: Kuwait" in content else ""
            if kuwait_section:
                debug_print("Found Kuwait validation section")
                if "Category C Score:" in kuwait_section:
                    data['Kuwait']['category_c_score'] = float(
                        kuwait_section.split("Category C Score: ")[1].split('\n')[0])
                if "Correlation:" in kuwait_section:
                    data['Kuwait']['correlation'] = float(kuwait_section.split("Correlation: ")[1].split('\n')[0])
                if "R²:" in kuwait_section:
                    data['Kuwait']['r_squared'] = float(kuwait_section.split("R²: ")[1].split('\n')[0])

            # Extract Jeju data
            jeju_section = content.split("Location: Jeju")[1] if "Location: Jeju" in content else ""
            if jeju_section:
                debug_print("Found Jeju validation section")
                if "Category C Score:" in jeju_section:
                    data['Jeju']['category_c_score'] = float(jeju_section.split("Category C Score: ")[1].split('\n')[0])
                if "Correlation:" in jeju_section:
                    data['Jeju']['correlation'] = float(jeju_section.split("Correlation: ")[1].split('\n')[0])
                if "R²:" in jeju_section:
                    data['Jeju']['r_squared'] = float(jeju_section.split("R²: ")[1].split('\n')[0])

            debug_print("Validation data", data)

    except Exception as e:
        debug_print(f"Could not load validation data: {str(e)}")
        debug_print(traceback.format_exc())

    return data

def get_attribute_colors():
    """Get color mappings for different attributes"""

    color_maps = {
        'flavor': {
            'Regular': '#8B4513',  # Brown
            'Menthol': '#00FF00',  # Green
            'Menthol Caps': '#00AA00',  # Dark Green
            'NTD': '#ADD8E6',  # Light Blue
            'NTD Caps': '#0000FF'  # Blue
        },
        'taste': {
            'Full Flavor': '#FF0000',  # Red
            'Lights': '#FFA500',  # Orange
            'Ultralights': '#FFFF00',  # Yellow
            '1mg': '#FFFFAA'  # Light Yellow
        },
        'thickness': {
            'STD': '#800080',  # Purple
            'SLI': '#FF00FF',  # Magenta
            'SSL': '#BA55D3',  # Medium Orchid
            'MAG': '#DA70D6'  # Orchid
        },
        'length': {
            'KS': '#2E8B57',  # Sea Green
            '100': '#3CB371',  # Medium Sea Green
            'LONGER THAN KS': '#66CDAA',  # Medium Aquamarine
            'LONG SIZE': '#8FBC8F',  # Dark Sea Green
            'REGULAR SIZE': '#90EE90'  # Light Green
        }
    }

    debug_print("Attribute color maps", color_maps)
    return color_maps

def create_attribute_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create a 2D heatmap data for two selected attributes"""

    debug_print(f"Creating attribute heatmap for {location}, {primary_attr} vs {secondary_attr}")

    # Check if location data exists
    if location not in location_data:
        debug_print(f"Location {location} not found in data")
        return pd.DataFrame()

    # Check if attribute data exists
    if primary_attr not in location_data[location] or secondary_attr not in location_data[location]:
        debug_print(f"Attributes {primary_attr} or {secondary_attr} not found for {location}")
        return pd.DataFrame()

    # Get attribute values from distribution data
    try:
        primary_values = location_data[location][primary_attr][primary_attr].unique()
        secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

        debug_print(f"Primary attribute values ({primary_attr})", primary_values)
        debug_print(f"Secondary attribute values ({secondary_attr})", secondary_values)
    except Exception as e:
        debug_print(f"Error getting attribute values: {str(e)}")
        debug_print(f"Primary attribute data columns: {location_data[location][primary_attr].columns.tolist()}")
        debug_print(f"Secondary attribute data columns: {location_data[location][secondary_attr].columns.tolist()}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get product data
    if 'top_90pct_products' not in location_data[location]:
        debug_print(f"top_90pct_products not found for {location}")
        return pd.DataFrame()

    products_df = location_data[location]['top_90pct_products']

    # Check if attributes exist in product data
    if primary_attr not in products_df.columns or secondary_attr not in products_df.columns:
        debug_print(f"Attributes {primary_attr} or {secondary_attr} not found in product data")
        debug_print(f"Product data columns: {products_df.columns.tolist()}")
        return pd.DataFrame()

    # Fill heatmap with volume data
    try:
        for p_val in primary_values:
            for s_val in secondary_values:
                volume = products_df[(products_df[primary_attr] == p_val) &
                                     (products_df[secondary_attr] == s_val)]['DF_Vol'].sum()
                heatmap_data.loc[p_val, s_val] = volume

        debug_print(f"Raw heatmap data", heatmap_data)
    except Exception as e:
        debug_print(f"Error filling heatmap data: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Normalize to percentage of total volume
    total_volume = heatmap_data.sum().sum()
    if total_volume > 0:
        heatmap_data = (heatmap_data / total_volume) * 100
        debug_print(f"Normalized heatmap data", heatmap_data)
    else:
        debug_print("Total volume is zero, cannot normalize")

    # Set index and column names
    heatmap_data.index.name = primary_attr
    heatmap_data.columns.name = secondary_attr

    return heatmap_data


def create_pmi_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create a 2D heatmap data for PMI products only"""

    debug_print(f"Creating PMI heatmap for {location}, {primary_attr} vs {secondary_attr}")

    # Check if location data exists
    if location not in location_data:
        debug_print(f"Location {location} not found in data")
        return pd.DataFrame()

    # Check if attribute data exists
    if primary_attr not in location_data[location] or secondary_attr not in location_data[location]:
        debug_print(f"Attributes {primary_attr} or {secondary_attr} not found for {location}")
        return pd.DataFrame()

    # Get attribute values from distribution data
    try:
        primary_values = location_data[location][primary_attr][primary_attr].unique()
        secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

        debug_print(f"Primary attribute values ({primary_attr})", primary_values)
        debug_print(f"Secondary attribute values ({secondary_attr})", secondary_values)
    except Exception as e:
        debug_print(f"Error getting attribute values: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get PMI product data
    if 'pmi_products' not in location_data[location]:
        debug_print(f"pmi_products not found for {location}")
        return pd.DataFrame()

    products_df = location_data[location]['pmi_products']

    # Check if attributes exist in product data
    if primary_attr not in products_df.columns or secondary_attr not in products_df.columns:
        debug_print(f"Attributes {primary_attr} or {secondary_attr} not found in PMI product data")
        debug_print(f"PMI product data columns: {products_df.columns.tolist()}")
        return pd.DataFrame()

    # Fill heatmap with volume data
    try:
        for p_val in primary_values:
            for s_val in secondary_values:
                volume = products_df[(products_df[primary_attr] == p_val) &
                                     (products_df[secondary_attr] == s_val)]['DF_Vol'].sum()
                heatmap_data.loc[p_val, s_val] = volume

        debug_print(f"Raw PMI heatmap data", heatmap_data)
    except Exception as e:
        debug_print(f"Error filling PMI heatmap data: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Normalize to percentage of total volume
    total_volume = heatmap_data.sum().sum()
    if total_volume > 0:
        heatmap_data = (heatmap_data / total_volume) * 100
        debug_print(f"Normalized PMI heatmap data", heatmap_data)
    else:
        debug_print("Total PMI volume is zero, cannot normalize")

    # Set index and column names
    heatmap_data.index.name = primary_attr
    heatmap_data.columns.name = secondary_attr

    return heatmap_data


def create_ideal_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create an ideal 2D heatmap based on passenger preferences (Category C)"""

    debug_print(f"Creating ideal heatmap for {location}, {primary_attr} vs {secondary_attr}")

    # Check if location data exists
    if location not in location_data:
        debug_print(f"Location {location} not found in data")
        return pd.DataFrame()

    # Check if attribute data exists
    if primary_attr not in location_data[location] or secondary_attr not in location_data[location]:
        debug_print(f"Attributes {primary_attr} or {secondary_attr} not found for {location}")
        return pd.DataFrame()

    # Get attribute values from distribution data
    try:
        primary_values = location_data[location][primary_attr][primary_attr].unique()
        secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

        debug_print(f"Primary attribute values ({primary_attr})", primary_values)
        debug_print(f"Secondary attribute values ({secondary_attr})", secondary_values)
    except Exception as e:
        debug_print(f"Error getting attribute values for ideal heatmap: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get attribute distributions
    primary_dist = location_data[location][primary_attr]
    secondary_dist = location_data[location][secondary_attr]

    # Check for Ideal_Percentage column
    if 'Ideal_Percentage' not in primary_dist.columns or 'Ideal_Percentage' not in secondary_dist.columns:
        debug_print(f"Ideal_Percentage column not found in attribute distribution data")
        debug_print(f"Primary distribution columns: {primary_dist.columns.tolist()}")
        debug_print(f"Secondary distribution columns: {secondary_dist.columns.tolist()}")
        return pd.DataFrame()

    # Create dictionaries for ideal percentages
    try:
        primary_ideal = {}
        for _, row in primary_dist.iterrows():
            primary_ideal[row[primary_attr]] = row['Ideal_Percentage']

        secondary_ideal = {}
        for _, row in secondary_dist.iterrows():
            secondary_ideal[row[secondary_attr]] = row['Ideal_Percentage']

        debug_print(f"Primary ideal percentages", primary_ideal)
        debug_print(f"Secondary ideal percentages", secondary_ideal)
    except Exception as e:
        debug_print(f"Error creating ideal percentage dictionaries: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Calculate joint probabilities (assuming independence)
    try:
        for p_val in primary_values:
            for s_val in secondary_values:
                # Joint probability = P(A) * P(B) / 100 (to normalize percentage)
                heatmap_data.loc[p_val, s_val] = (primary_ideal.get(p_val, 0) * secondary_ideal.get(s_val, 0)) / 100

        debug_print(f"Raw ideal heatmap data", heatmap_data)
    except Exception as e:
        debug_print(f"Error calculating joint probabilities: {str(e)}")
        debug_print(traceback.format_exc())
        return pd.DataFrame()

    # Normalize to ensure total is 100%
    total = heatmap_data.sum().sum()
    if total > 0:
        heatmap_data = (heatmap_data / total) * 100
        debug_print(f"Normalized ideal heatmap data", heatmap_data)
    else:
        debug_print("Total ideal percentage is zero, cannot normalize")

    # Set index and column names
    heatmap_data.index.name = primary_attr
    heatmap_data.columns.name = secondary_attr

    return heatmap_data


def create_gap_heatmap_data(actual_data, ideal_data):
    """Create a gap heatmap showing the difference between actual and ideal"""

    debug_print("Creating gap heatmap data")

    # Check if data exists
    if actual_data.empty or ideal_data.empty:
        debug_print("Actual or ideal data is empty")
        return pd.DataFrame()

    # Ensure indices and columns match
    if not all(actual_data.index.isin(ideal_data.index)) or not all(actual_data.columns.isin(ideal_data.columns)):
        debug_print("Indices or columns don't match between actual and ideal data")
        debug_print(f"Actual indices: {actual_data.index.tolist()}")
        debug_print(f"Ideal indices: {ideal_data.index.tolist()}")
        debug_print(f"Actual columns: {actual_data.columns.tolist()}")
        debug_print(f"Ideal columns: {ideal_data.columns.tolist()}")

        # Create a new DataFrame with common indices and columns
        common_indices = actual_data.index.intersection(ideal_data.index)
        common_columns = actual_data.columns.intersection(ideal_data.columns)

        actual_subset = actual_data.loc[common_indices, common_columns]
        ideal_subset = ideal_data.loc[common_indices, common_columns]

        # Calculate the gap
        gap_data = actual_subset - ideal_subset
        debug_print("Gap data with subset", gap_data)
        return gap_data

        # Calculate the gap
        gap_data = actual_data - ideal_data
        debug_print("Gap data", gap_data)
        return gap_data


def create_shelf_visualization(heatmap_data, color_attr_data, color_attr, primary_attr, secondary_attr,
                               attribute_colors, title, is_pmi=False):
    """
    Create a shelf visualization using Plotly

    Parameters:
    -----------
    heatmap_data : pandas DataFrame
        2D heatmap data with product distribution
    color_attr_data : pandas DataFrame
        Data for the attribute used for coloring
    color_attr : str
        The attribute to use for coloring
    primary_attr : str
        Primary attribute (y-axis)
    secondary_attr : str
        Secondary attribute (x-axis)
    attribute_colors : dict
        Dictionary with color mappings
    title : str
        Title for the visualization
    is_pmi : bool
        Whether this is for PMI products only

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    debug_print(f"Creating shelf visualization: {title}, {primary_attr} vs {secondary_attr}, colored by {color_attr}")

    # Initialize figure
    fig = go.Figure()

    # Check if heatmap data is valid
    if heatmap_data is None or heatmap_data.empty:
        debug_print("Heatmap data is empty, returning empty figure")
        fig.add_annotation(
            text="No data available for this visualization",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    # Add heatmap as background
    try:
        heatmap_z = heatmap_data.values
        debug_print("Heatmap z values shape", heatmap_z.shape)

        fig.add_trace(go.Heatmap(
            z=heatmap_z,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale='Blues',
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title="% Volume",
                titleside="right",
                titlefont=dict(size=14),
                tickfont=dict(size=12)
            ),
            hovertemplate='%{y} × %{x}: %{z:.2f}%<extra></extra>'
        ))
    except Exception as e:
        debug_print(f"Error creating heatmap: {str(e)}")
        debug_print(traceback.format_exc())
        fig.add_annotation(
            text=f"Error creating heatmap: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    # Get color mapping
    color_map = attribute_colors.get(color_attr, {})
    debug_print(f"Color map for {color_attr}", color_map)

    # Check if color attribute exists in data
    if color_attr not in color_attr_data.columns:
        debug_print(f"Color attribute {color_attr} not found in data columns: {color_attr_data.columns.tolist()}")
        color_values = []
    else:
        color_values = color_attr_data[color_attr].unique()
        debug_print(f"Color attribute values", color_values)

    # Add bubbles for each attribute combination
    try:
        for i, p_val in enumerate(heatmap_data.index):
            for j, s_val in enumerate(heatmap_data.columns):
                # Cell value (percentage)
                cell_pct = heatmap_data.loc[p_val, s_val]

                # Skip if percentage is too small
                if cell_pct < 0.5:
                    continue

                # For each color attribute value, add a bubble
                for k, c_val in enumerate(color_values):
                    # Calculate position with slight offset based on color value
                    # to avoid complete overlap
                    offset = (k - (len(color_values) - 1) / 2) * 0.2
                    x_pos = j + offset
                    y_pos = i

                    # Get color for this attribute value
                    color = color_map.get(c_val, '#CCCCCC')

                    # Calculate size - proportional to percentage but minimum size for visibility
                    size = max(20 * np.sqrt(cell_pct), 10)

                    # Add bubble
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers',
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=0.8,
                            line=dict(width=1, color='#333333')
                        ),
                        name=f"{c_val}",
                        text=f"{c_val}: {cell_pct:.1f}%",
                        hoverinfo='text',
                        showlegend=True
                    ))

        debug_print("Added bubbles to visualization")
    except Exception as e:
        debug_print(f"Error adding bubbles: {str(e)}")
        debug_print(traceback.format_exc())

    # Update layout
    try:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color="#333333")
            ),
            xaxis=dict(
                title=dict(text=secondary_attr.capitalize(), font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor='#eeeeee'
            ),
            yaxis=dict(
                title=dict(text=primary_attr.capitalize(), font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor='#eeeeee'
            ),
            legend=dict(
                title=dict(text=color_attr.capitalize(), font=dict(size=14)),
                font=dict(size=12)
            ),
            margin=dict(l=80, r=80, t=100, b=80),
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )

        # Group legend items by color attribute value
        fig.update_layout(
            legend_tracegroupgap=0
        )
    except Exception as e:
        debug_print(f"Error updating layout: {str(e)}")
        debug_print(traceback.format_exc())

    return fig


# Define layout
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3("Portfolio Controls", className="card-title"),

            html.Div([
                html.Label("Select Location"),
                dcc.Dropdown(
                    id="location-dropdown",
                    options=[
                        {"label": "Kuwait (Well-Aligned)", "value": "Kuwait"},
                        {"label": "Jeju (Misaligned)", "value": "Jeju"}
                    ],
                    value="Kuwait",
                    clearable=False
                )
            ], className="mb-4"),

            html.Div([
                html.H4("Shelf Attributes", className="mt-4 mb-2"),

                html.Div([
                    html.Label("Primary Attribute (Y-axis)"),
                    dcc.Dropdown(
                        id="primary-attr-dropdown",
                        options=[
                            {"label": "Flavor", "value": "flavor"},
                            {"label": "Taste", "value": "taste"},
                            {"label": "Thickness", "value": "thickness"},
                            {"label": "Length", "value": "length"}
                        ],
                        value="flavor",
                        clearable=False
                    )
                ], className="mb-3"),

                html.Div([
                    html.Label("Secondary Attribute (X-axis)"),
                    dcc.Dropdown(
                        id="secondary-attr-dropdown",
                        options=[
                            {"label": "Flavor", "value": "flavor"},
                            {"label": "Taste", "value": "taste"},
                            {"label": "Thickness", "value": "thickness"},
                            {"label": "Length", "value": "length"}
                        ],
                        value="taste",
                        clearable=False
                    )
                ], className="mb-3"),

                html.Div([
                    html.Label("Color Products By"),
                    dcc.Dropdown(
                        id="color-attr-dropdown",
                        options=[
                            {"label": "Flavor", "value": "flavor"},
                            {"label": "Taste", "value": "taste"},
                            {"label": "Thickness", "value": "thickness"},
                            {"label": "Length", "value": "length"}
                        ],
                        value="thickness",
                        clearable=False
                    )
                ], className="mb-3"),
            ]),

            # Debug Options
            html.Div([
                html.H4("Debug Options", className="mt-4 mb-2"),
                html.Div([
                    html.Label("Debug Mode"),
                    dcc.RadioItems(
                        id="debug-mode",
                        options=[
                            {"label": "On", "value": "on"},
                            {"label": "Off", "value": "off"}
                        ],
                        value="on",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    )
                ]),
                html.Div(id="debug-output", className="debug-info")
            ]),

            html.Div([
                html.H4("Validation Warnings", className="mt-4 mb-2"),
                html.Div(id="validation-warnings")
            ])

        ], className="card", style={"gridArea": "controls"}),

        html.Div([
            html.H3("Market Information", className="card-title"),
            html.Div(id="market-info"),
        ], className="card", style={"gridArea": "market"}),

        html.Div([
            html.H3("Category Scores", className="card-title"),
            html.Div(id="category-scores"),
        ], className="card", style={"gridArea": "scores"}),

        html.Div([
            html.H3("Portfolio Insights", className="card-title"),
            html.Div(id="portfolio-insights"),
        ], className="card", style={"gridArea": "insights"})
    ], style={
        "display": "grid",
        "gridTemplateAreas": "'controls market' 'controls scores' 'controls insights'",
        "gridTemplateColumns": "350px 1fr",
        "gridGap": "20px",
        "marginBottom": "20px"
    }),

    html.Div([
        dcc.Tabs(id='visualization-tabs', value='comparison', children=[
            dcc.Tab(label='Market vs PMI Comparison', value='comparison', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(label='PMI Portfolio Detail', value='pmi', className="custom-tab",
                    selected_className="custom-tab--selected"),
            dcc.Tab(label='Ideal Market Portfolio', value='ideal', className="custom-tab",
                    selected_className="custom-tab--selected"),
        ]),
        html.Div(id='tab-content', className="tab-content")
    ], className="card"),

    # Store data in hidden divs
    html.Div(id='stored-data', style={'display': 'none'}),
    html.Div(id='stored-validation-data', style={'display': 'none'}),
])


# Callback to load and store data
@app.callback(
    [Output('stored-data', 'children'),
     Output('stored-validation-data', 'children')],
    [Input('location-dropdown', 'value')]
)
def load_and_store_data(location):
    debug_print(f"Loading data for {location}")

    try:
        # Load data
        location_data = load_location_data()
        validation_data = load_validation_data()

        # Prepare data for storage (select only what we need)
        selected_data = {
            'Kuwait': {},
            'Jeju': {}
        }

        for loc in ['Kuwait', 'Jeju']:
            for key in ['flavor', 'taste', 'thickness', 'length', 'summary']:
                if key in location_data[loc]:
                    selected_data[loc][key] = location_data[loc][key].to_dict(orient='records')

            if 'pmi_products' in location_data[loc]:
                # Filter columns for product data to reduce data size
                cols = [col for col in
                        ['CR_BrandId', 'flavor', 'taste', 'thickness', 'length', 'DF_Vol', 'Brand Family']
                        if col in location_data[loc]['pmi_products'].columns]
                selected_data[loc]['pmi_products'] = location_data[loc]['pmi_products'][cols].to_dict(orient='records')

            if 'top_90pct_products' in location_data[loc]:
                cols = [col for col in ['CR_BrandId', 'flavor', 'taste', 'thickness', 'length', 'DF_Vol', 'TMO']
                        if col in location_data[loc]['top_90pct_products'].columns]
                selected_data[loc]['top_90pct_products'] = location_data[loc]['top_90pct_products'][cols].to_dict(
                    orient='records')

        debug_print("Data prepared for storage")

        # Convert to JSON
        return json.dumps(selected_data), json.dumps(validation_data)

    except Exception as e:
        debug_print(f"Error loading and storing data: {str(e)}")
        debug_print(traceback.format_exc())
        return json.dumps({}), json.dumps({})


# Callback to update debug output
@app.callback(
    Output('debug-output', 'children'),
    [Input('debug-mode', 'value'),
     Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value'),
     Input('color-attr-dropdown', 'value'),
     Input('stored-data', 'children')]
)
def update_debug_output(debug_mode, location, primary_attr, secondary_attr, color_attr, stored_data):
    if debug_mode != "on" or not stored_data:
        return ""

    try:
        # Parse stored data
        data_dict = json.loads(stored_data)

        debug_info = []

        # Check location data
        if location in data_dict:
            debug_info.append(html.P(f"Location data available for {location}"))

            # Check attribute data
            for attr in [primary_attr, secondary_attr, color_attr]:
                if attr in data_dict[location]:
                    attr_records = data_dict[location][attr]
                    attr_df = pd.DataFrame(attr_records)
                    debug_info.append(html.P(f"{attr.capitalize()} data: {len(attr_records)} records"))
                    debug_info.append(html.P(f"Columns: {list(attr_df.columns)}"))
                else:
                    debug_info.append(html.P(f"{attr.capitalize()} data: Not available", style={"color": "red"}))

            # Check product data
            for product_type in ['pmi_products', 'top_90pct_products']:
                if product_type in data_dict[location]:
                    product_records = data_dict[location][product_type]
                    product_df = pd.DataFrame(product_records)
                    debug_info.append(html.P(f"{product_type}: {len(product_records)} records"))
                    debug_info.append(html.P(f"Columns: {list(product_df.columns)}"))

                    # Check if attributes exist in product data
                    for attr in [primary_attr, secondary_attr, color_attr]:
                        if attr in product_df.columns:
                            debug_info.append(html.P(f"✓ {attr} found in {product_type}"))
                        else:
                            debug_info.append(html.P(f"✗ {attr} not found in {product_type}", style={"color": "red"}))
                else:
                    debug_info.append(html.P(f"{product_type}: Not available", style={"color": "red"}))
        else:
            debug_info.append(html.P(f"No data available for {location}", style={"color": "red"}))

        return debug_info

    except Exception as e:
        return html.P(f"Error in debug output: {str(e)}", style={"color": "red"})


# Callback to validate attribute selection
@app.callback(
    Output('validation-warnings', 'children'),
    [Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value'),
     Input('color-attr-dropdown', 'value')]
)
def validate_attributes(primary_attr, secondary_attr, color_attr):
    debug_print(f"Validating attributes: {primary_attr}, {secondary_attr}, {color_attr}")

    warnings = []

    if primary_attr == secondary_attr:
        warnings.append(html.Div([
            html.I(className="fas fa-exclamation-triangle", style={"marginRight": "10px"}),
            "Primary and secondary attributes should be different"
        ], className="warning-box"))

    if primary_attr == color_attr or secondary_attr == color_attr:
        warnings.append(html.Div([
            html.I(className="fas fa-info-circle", style={"marginRight": "10px"}),
            "For best results, color attribute should be different from shelf axes"
        ], className="info-box"))

    return warnings


# Callback to display market information
@app.callback(
    Output('market-info', 'children'),
    [Input('location-dropdown', 'value'),
     Input('stored-data', 'children')]
)
def display_market_info(location, stored_data):
    debug_print(f"Displaying market info for {location}")

    if not stored_data:
        return []

    try:
        # Parse stored data
        data = json.loads(stored_data)

        # Market share (default values)
        market_shares = {
            "Kuwait": 75,  # Approximate as per documentation
            "Jeju": 12  # Approximate as per documentation
        }

        # Try to extract more accurate market share from data
        if location in data and 'summary' in data[location]:
            summary = data[location]['summary']
            for row in summary:
                if 'Category' in row and 'Metric' in row and 'Value' in row:
                    if 'Market Share' in row['Category'] and 'PMI Share' in row['Metric']:
                        try:
                            share_str = row['Value']
                            if isinstance(share_str, str) and '%' in share_str:
                                market_shares[location] = float(share_str.strip('%'))
                        except Exception as e:
                            debug_print(f"Error extracting market share: {str(e)}")

        # Get SKU counts
        pmi_skus = len(data.get(location, {}).get('pmi_products', []))
        total_skus = len(data.get(location, {}).get('top_90pct_products', []))

        # Create metric cards
        market_share_card = html.Div([
            html.Div(f"{market_shares[location]}%", className="metric-value"),
            html.Div("PMI Market Share", className="metric-label")
        ], className="metric-card")

        sku_card = html.Div([
            html.Div([
                html.Span(f"{pmi_skus}", style={"fontWeight": "bold"}),
                html.Span(f" / {total_skus}")
            ], className="metric-value"),
            html.Div("PMI SKUs / Total SKUs", className="metric-label")
        ], className="metric-card")

        # Create status indicator
        status = "Well-aligned" if market_shares[location] > 40 else "Misaligned"
        status_color = "green" if status == "Well-aligned" else "red"

        status_card = html.Div([
            html.Div(status, className="metric-value", style={"color": status_color}),
            html.Div("Portfolio Status", className="metric-label")
        ], className="metric-card")

        return [
            html.Div([
                market_share_card,
                sku_card,
                status_card
            ], style={"display": "flex", "justifyContent": "space-between"})
        ]

    except Exception as e:
        debug_print(f"Error displaying market info: {str(e)}")
        debug_print(traceback.format_exc())
        return [
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                f"Error displaying market information: {str(e)}"
            ], className="error-box")
        ]


# Callback to display category scores
@app.callback(
    Output('category-scores', 'children'),
    [Input('location-dropdown', 'value'),
     Input('stored-validation-data', 'children')]
)
def display_category_scores(location, stored_validation_data):
    debug_print(f"Displaying category scores for {location}")

    if not stored_validation_data:
        return []

    try:
        # Parse stored data
        validation_data = json.loads(stored_validation_data)

        # Default category scores
        category_scores = {
            "Kuwait": {"A": 9.64, "B": 8.10, "C": 3.93, "D": 5.03},
            "Jeju": {"A": 7.53, "B": 4.37, "C": 6.93, "D": 5.82}
        }

        # Try to extract from validation data
        if location in validation_data and 'category_c_score' in validation_data[location]:
            category_scores[location]["C"] = validation_data[location]['category_c_score']

        # Create score cards
        score_cards = []

        for category, score in category_scores[location].items():
            color = "green" if score >= 7.5 else ("orange" if score >= 5 else "red")

            card = html.Div([
                html.Div(f"{score:.2f}", className="metric-value", style={"color": color}),
                html.Div(f"Category {category}", className="metric-label")
            ], className="metric-card")

            score_cards.append(card)

        # Calculate average score
        avg_score = sum(category_scores[location].values()) / len(category_scores[location])
        avg_color = "green" if avg_score >= 7.5 else ("orange" if avg_score >= 5 else "red")

        avg_card = html.Div([
            html.Div(f"{avg_score:.2f}", className="metric-value", style={"color": avg_color}),
            html.Div("Average Score", className="metric-label")
        ], className="metric-card")

        score_cards.append(avg_card)

        return [
            html.Div(score_cards, style={"display": "flex", "justifyContent": "space-between"})
        ]

    except Exception as e:
        debug_print(f"Error displaying category scores: {str(e)}")
        debug_print(traceback.format_exc())
        return [
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                f"Error displaying category scores: {str(e)}"
            ], className="error-box")
        ]


# Callback to display portfolio insights
@app.callback(
    Output('portfolio-insights', 'children'),
    [Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('stored-data', 'children')]
)
def display_portfolio_insights(location, primary_attr, stored_data):
    debug_print(f"Displaying portfolio insights for {location}, {primary_attr}")

    if not stored_data:
        return []

    try:
        # Parse stored data
        data = json.loads(stored_data)

        insights = []

        # Location-specific insights
        if location == "Kuwait":
            insights.append(html.Div([
                html.I(className="fas fa-check-circle", style={"marginRight": "10px", "color": "green"}),
                "Kuwait shows strong portfolio alignment with consumer preferences, particularly in Flavor and Taste attributes."
            ], className="info-box"))

            insights.append(html.Div([
                html.I(className="fas fa-chart-line", style={"marginRight": "10px", "color": "green"}),
                "This alignment contributes to the high market share of approximately 75%."
            ], className="info-box"))
        else:  # Jeju
            insights.append(html.Div([
                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px", "color": "red"}),
                "Jeju shows significant misalignment in its portfolio, especially in Taste and Length attributes."
            ], className="warning-box"))

            insights.append(html.Div([
                html.I(className="fas fa-chart-line", style={"marginRight": "10px", "color": "red"}),
                "This misalignment contributes to the lower market share of approximately 12%."
            ], className="warning-box"))

        # Primary attribute insights
        if location in data and primary_attr in data[location]:
            primary_data = data[location][primary_attr]

            # Convert list of dicts to pandas DataFrame
            primary_df = pd.DataFrame(primary_data)

            # Try to extract gaps
            if 'PMI_vs_Ideal_Gap' in primary_df.columns and primary_attr in primary_df.columns:
                # Find most underrepresented attributes
                underrep = primary_df.sort_values('PMI_vs_Ideal_Gap').head(2)

                if not underrep.empty:
                    underrep_values = [row[primary_attr] for _, row in underrep.iterrows()]
                    insights.append(html.Div([
                        html.I(className="fas fa-arrow-down", style={"marginRight": "10px", "color": "red"}),
                        f"Most underrepresented {primary_attr}: ",
                        html.B(", ".join(underrep_values))
                    ], className="info-box"))

                # Find most overrepresented attributes
                overrep = primary_df.sort_values('PMI_vs_Ideal_Gap', ascending=False).head(2)

                if not overrep.empty:
                    overrep_values = [row[primary_attr] for _, row in overrep.iterrows()]
                    insights.append(html.Div([
                        html.I(className="fas fa-arrow-up", style={"marginRight": "10px", "color": "blue"}),
                        f"Most overrepresented {primary_attr}: ",
                        html.B(", ".join(overrep_values))
                    ], className="info-box"))

        return insights

    except Exception as e:
        debug_print(f"Error displaying portfolio insights: {str(e)}")
        debug_print(traceback.format_exc())
        return [
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                f"Error displaying portfolio insights: {str(e)}"
            ], className="error-box")
        ]


# Helper function to display gap analysis
def display_gap_analysis(gap_data, primary_attr, secondary_attr):
    """Display gap analysis between actual and ideal data"""
    debug_print("Displaying gap analysis")

    # Handle empty data
    if gap_data is None or gap_data.empty:
        return html.Div("No gap data available", className="info-box")

    # Flatten the gap data for easier sorting
    gap_values = []

    for p_idx, p_val in enumerate(gap_data.index):
        for s_idx, s_val in enumerate(gap_data.columns):
            gap = gap_data.loc[p_val, s_val]
            if abs(gap) > 0.5:  # Only include non-trivial gaps
                gap_values.append({
                    'primary': p_val,
                    'secondary': s_val,
                    'gap': gap,
                    'abs_gap': abs(gap)
                })

    # Sort by absolute gap
    gap_values = sorted(gap_values, key=lambda x: x['abs_gap'], reverse=True)

    # Create gap list
    gap_items = []

    for i, gap_info in enumerate(gap_values[:5]):  # Show top 5 gaps
        direction = "deficit" if gap_info['gap'] < 0 else "excess"
        color = "red" if gap_info['gap'] < 0 else "blue"

        item = html.Div([
            html.I(className=f"fas fa-{'minus' if gap_info['gap'] < 0 else 'plus'}-circle",
                   style={"marginRight": "10px", "color": color}),
            f"{gap_info['primary']} × {gap_info['secondary']}: ",
            html.B(f"{abs(gap_info['gap']):.1f}% {direction}")
        ], className="info-box", style={"margin": "5px 0"})

        gap_items.append(item)

    if not gap_items:
        gap_items = [html.Div("No significant gaps found", className="info-box")]

    return html.Div(gap_items)


# Helper function to display product statistics
def display_product_statistics(product_data, attr):
    """Display product statistics for a given attribute"""
    debug_print(f"Displaying product statistics for {attr}")

    # Convert to DataFrame if it's a list of dicts
    if isinstance(product_data, list):
        product_data = pd.DataFrame(product_data)

    if attr not in product_data.columns:
        return html.Div("Attribute data not available", className="warning-box")

    # Group by attribute and calculate total volume
    if 'DF_Vol' in product_data.columns:
        distribution = product_data.groupby(attr)['DF_Vol'].sum()
        total_vol = distribution.sum()

        # Calculate percentages
        percentages = {}
        for attr_val, vol in distribution.items():
            if total_vol > 0:
                percentages[attr_val] = (vol / total_vol) * 100

        # Sort by percentage
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create list items
        items = []

        for attr_val, pct in sorted_items:
            items.append(html.Div([
                html.Span(f"{attr_val}: ", style={"fontWeight": "bold"}),
                html.Span(f"{pct:.1f}%")
            ], style={"margin": "5px 0"}))

        return html.Div(items)

    # If no volume data, just count SKUs
    else:
        counts = product_data[attr].value_counts()
        total = counts.sum()

        # Calculate percentages
        percentages = {}
        for attr_val, count in counts.items():
            if total > 0:
                percentages[attr_val] = (count / total) * 100

        # Sort by percentage
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create list items
        items = []

        for attr_val, pct in sorted_items:
            items.append(html.Div([
                html.Span(f"{attr_val}: ", style={"fontWeight": "bold"}),
                html.Span(f"{pct:.1f}% ({counts[attr_val]} SKUs)")
            ], style={"margin": "5px 0"}))

        return html.Div(items)


# Helper function to display brand statistics
def display_brand_statistics(product_data):
    """Display brand statistics"""
    debug_print("Displaying brand statistics")

    # Convert to DataFrame if it's a list of dicts
    if isinstance(product_data, list):
        product_data = pd.DataFrame(product_data)

    if 'Brand Family' not in product_data.columns:
        return html.Div("Brand data not available", className="warning-box")

    # Group by brand and calculate total volume
    if 'DF_Vol' in product_data.columns:
        distribution = product_data.groupby('Brand Family')['DF_Vol'].sum()
        total_vol = distribution.sum()

        # Calculate percentages
        percentages = {}
        for brand, vol in distribution.items():
            if total_vol > 0:
                percentages[brand] = (vol / total_vol) * 100

        # Sort by percentage
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create list items
        items = []

        for brand, pct in sorted_items:
            items.append(html.Div([
                html.Span(f"{brand}: ", style={"fontWeight": "bold"}),
                html.Span(f"{pct:.1f}% ({distribution[brand]:,.1f} units)")
            ], style={"margin": "5px 0"}))

        return html.Div(items)

    # If no volume data, just count SKUs
    else:
        counts = product_data['Brand Family'].value_counts()
        total = counts.sum()

        # Calculate percentages
        percentages = {}
        for brand, count in counts.items():
            if total > 0:
                percentages[brand] = (count / total) * 100

        # Sort by percentage
        sorted_items = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Create list items
        items = []

        for brand, pct in sorted_items:
            items.append(html.Div([
                html.Span(f"{brand}: ", style={"fontWeight": "bold"}),
                html.Span(f"{pct:.1f}% ({counts[brand]} SKUs)")
            ], style={"margin": "5px 0"}))

        return html.Div(items)


# Helper function to display ideal distribution
def display_ideal_distribution(attr_data, attr_name):
    """Display ideal distribution for an attribute"""
    debug_print(f"Displaying ideal distribution for {attr_name}")

    # Convert to DataFrame if it's a list of dicts
    if isinstance(attr_data, list):
        attr_data = pd.DataFrame(attr_data)

    if attr_name not in attr_data.columns or 'Ideal_Percentage' not in attr_data.columns:
        return html.Div("Ideal distribution data not available", className="warning-box")

    # Sort by ideal percentage
    sorted_data = attr_data.sort_values('Ideal_Percentage', ascending=False)

    # Create list items
    items = []

    for _, row in sorted_data.iterrows():
        items.append(html.Div([
            html.Span(f"{row[attr_name]}: ", style={"fontWeight": "bold"}),
            html.Span(f"{row['Ideal_Percentage']:.1f}%")
        ], style={"margin": "5px 0"}))

    return html.Div(items)


# Callback to display tab content
@app.callback(
    Output('tab-content', 'children'),
    [Input('visualization-tabs', 'value'),
     Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value'),
     Input('color-attr-dropdown', 'value'),
     Input('stored-data', 'children'),
     Input('debug-mode', 'value')]
)
def display_tab_content(tab, location, primary_attr, secondary_attr, color_attr, stored_data, debug_mode):
    debug_print(f"Displaying tab content: {tab}, {location}, {primary_attr}, {secondary_attr}, {color_attr}")

    if not stored_data:
        return [
            html.Div([
                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                "No data available. Please check data loading."
            ], className="error-box")
        ]

    try:
        # Parse stored data
        data_dict = json.loads(stored_data)

        # Convert dictionary back to DataFrames
        data = {
            'Kuwait': {},
            'Jeju': {}
        }

        for loc in ['Kuwait', 'Jeju']:
            for key in data_dict[loc]:
                data[loc][key] = pd.DataFrame(data_dict[loc][key])
                debug_print(f"Converted {loc} {key} to DataFrame", data[loc][key])

        # Get attribute colors
        attribute_colors = get_attribute_colors()

        # Debug information for tab content
        debug_info = []
        if debug_mode == "on":
            debug_info.append(html.Div([
                html.H4("Debug Information"),
                html.P(f"Tab: {tab}"),
                html.P(f"Location: {location}"),
                html.P(f"Primary attribute: {primary_attr}"),
                html.P(f"Secondary attribute: {secondary_attr}"),
                html.P(f"Color attribute: {color_attr}")
            ], className="debug-info"))

        if tab == 'comparison':
            # Create market, PMI, and ideal heatmaps
            try:
                debug_print("Creating heatmaps for comparison tab")

                market_heatmap = create_attribute_heatmap_data(data, location, primary_attr, secondary_attr)
                pmi_heatmap = create_pmi_heatmap_data(data, location, primary_attr, secondary_attr)
                ideal_heatmap = create_ideal_heatmap_data(data, location, primary_attr, secondary_attr)

                if market_heatmap.empty or pmi_heatmap.empty or ideal_heatmap.empty:
                    if debug_mode == "on":
                        return debug_info + [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create heatmaps. Please check debug information."
                            ], className="error-box")
                        ]
                    else:
                        return [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create heatmaps. Please enable debug mode for more information."
                            ], className="error-box")
                        ]

                # Create visualizations
                debug_print("Creating market visualization")
                market_fig = create_shelf_visualization(
                    market_heatmap, data[location][color_attr], color_attr,
                    primary_attr, secondary_attr, attribute_colors,
                    f"{location} - Market Portfolio", False
                )

                debug_print("Creating PMI visualization")
                pmi_fig = create_shelf_visualization(
                    pmi_heatmap, data[location][color_attr], color_attr,
                    primary_attr, secondary_attr, attribute_colors,
                    f"{location} - PMI Portfolio", True
                )

                # Calculate gap data
                market_gap = create_gap_heatmap_data(market_heatmap, ideal_heatmap)
                pmi_gap = create_gap_heatmap_data(pmi_heatmap, ideal_heatmap)

                # Create layout
                return debug_info + [
                    html.Div([
                        html.H3("Market vs PMI Portfolio Comparison", className="card-title"),
                        html.Div([
                            html.Div([
                                dcc.Graph(figure=market_fig, id='market-graph')
                            ], className="graph-container", style={"width": "48%"}),
                            html.Div([
                                dcc.Graph(figure=pmi_fig, id='pmi-graph')
                            ], className="graph-container", style={"width": "48%"})
                        ], style={"display": "flex", "justifyContent": "space-between"}),

                        html.Div([
                            html.H3("Gap Analysis", className="card-title"),
                            html.Div([
                                html.Div([
                                    html.H4("Market vs Ideal Portfolio Gaps"),
                                    display_gap_analysis(market_gap, primary_attr, secondary_attr)
                                ], className="card", style={"width": "48%"}),
                                html.Div([
                                    html.H4("PMI vs Ideal Portfolio Gaps"),
                                    display_gap_analysis(pmi_gap, primary_attr, secondary_attr)
                                ], className="card", style={"width": "48%"})
                            ], style={"display": "flex", "justifyContent": "space-between"})
                        ])
                    ])
                ]

            except Exception as e:
                debug_print(f"Error creating comparison visualization: {str(e)}")
                debug_print(traceback.format_exc())

                if debug_mode == "on":
                    return debug_info + [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box"),
                            html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap"})
                        ])
                    ]
                else:
                    return [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box")
                        ])
                    ]

        elif tab == 'pmi':
            # Create PMI heatmap
            try:
                debug_print("Creating heatmap for PMI tab")

                pmi_heatmap = create_pmi_heatmap_data(data, location, primary_attr, secondary_attr)

                if pmi_heatmap.empty:
                    if debug_mode == "on":
                        return debug_info + [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create PMI heatmap. Please check debug information."
                            ], className="error-box")
                        ]
                    else:
                        return [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create PMI heatmap. Please enable debug mode for more information."
                            ], className="error-box")
                        ]

                # Create visualization
                debug_print("Creating PMI detail visualization")
                pmi_fig = create_shelf_visualization(
                    pmi_heatmap, data[location][color_attr], color_attr,
                    primary_attr, secondary_attr, attribute_colors,
                    f"{location} - PMI Portfolio Detail", True
                )

                # Get PMI product data
                pmi_products = data[location]['pmi_products']

                # Create layout
                return debug_info + [
                    html.Div([
                        html.H3("PMI Portfolio Detail", className="card-title"),
                        dcc.Graph(figure=pmi_fig, id='pmi-detail-graph'),

                        html.Div([
                            html.H3("PMI Portfolio Statistics", className="card-title"),
                            html.Div([
                                html.Div([
                                    html.H4("Product Distribution"),
                                    display_product_statistics(pmi_products, primary_attr)
                                ], className="card", style={"width": "48%"}),
                                html.Div([
                                    html.H4("Brand Statistics"),
                                    display_brand_statistics(pmi_products)
                                ], className="card", style={"width": "48%"})
                            ], style={"display": "flex", "justifyContent": "space-between"})
                        ])
                    ])
                ]

            except Exception as e:
                debug_print(f"Error creating PMI detail visualization: {str(e)}")
                debug_print(traceback.format_exc())

                if debug_mode == "on":
                    return debug_info + [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box"),
                            html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap"})
                        ])
                    ]
                else:
                    return [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box")
                        ])
                    ]

        elif tab == 'ideal':
            # Create ideal heatmap
            try:
                debug_print("Creating heatmap for ideal tab")

                ideal_heatmap = create_ideal_heatmap_data(data, location, primary_attr, secondary_attr)

                if ideal_heatmap.empty:
                    if debug_mode == "on":
                        return debug_info + [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create ideal heatmap. Please check debug information."
                            ], className="error-box")
                        ]
                    else:
                        return [
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                "Could not create ideal heatmap. Please enable debug mode for more information."
                            ], className="error-box")
                        ]

                # Create visualization
                debug_print("Creating ideal market visualization")
                ideal_fig = create_shelf_visualization(
                    ideal_heatmap, data[location][color_attr], color_attr,
                    primary_attr, secondary_attr, attribute_colors,
                    f"{location} - Ideal Portfolio (Based on Category C)", False
                )

                # Create layout
                return debug_info + [
                    html.Div([
                        html.H3("Ideal Market Portfolio", className="card-title"),
                        dcc.Graph(figure=ideal_fig, id='ideal-graph'),

                        html.Div([
                            html.H3("Category C Information", className="card-title"),
                            html.Div([
                                html.P([
                                    "Category C measures how well the portfolio aligns with passenger nationality mix ",
                                    "and the corresponding consumption preferences. A higher score indicates better alignment ",
                                    "with passenger mix-based preferences."
                                ]),

                                html.H4(f"Ideal {primary_attr.capitalize()} Distribution (Based on Passenger Mix):"),
                                display_ideal_distribution(data[location][primary_attr], primary_attr)
                            ], className="info-box")
                        ])
                    ])
                ]

            except Exception as e:
                debug_print(f"Error creating ideal visualization: {str(e)}")
                debug_print(traceback.format_exc())

                if debug_mode == "on":
                    return debug_info + [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box"),
                            html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap"})
                        ])
                    ]
                else:
                    return [
                        html.Div([
                            html.H3("Error Creating Visualization", className="card-title"),
                            html.Div([
                                html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                                f"An error occurred: {str(e)}"
                            ], className="error-box")
                        ])
                    ]

        return []

    except Exception as e:
        debug_print(f"Error displaying tab content: {str(e)}")
        debug_print(traceback.format_exc())

        if debug_mode == "on":
            return [
                html.Div([
                    html.H3("Error", className="card-title"),
                    html.Div([
                        html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                        f"An error occurred: {str(e)}"
                    ], className="error-box"),
                    html.Pre(traceback.format_exc(), style={"whiteSpace": "pre-wrap"})
                ])
            ]
        else:
            return [
                html.Div([
                    html.H3("Error", className="card-title"),
                    html.Div([
                        html.I(className="fas fa-exclamation-circle", style={"marginRight": "10px"}),
                        f"An error occurred: {str(e)}"
                    ], className="error-box")
                ])
            ]


# Main entry point
if __name__ == '__main__':
    app.run_server(debug=True)