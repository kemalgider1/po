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
            .graph-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
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
    """Load data from the locations_data directory with improved error handling"""

    data = {
        'Kuwait': {},
        'Jeju': {}
    }

    # Define file mappings with potential alternative column names
    file_mappings = {
        'Kuwait': {
            'flavor': {'file': 'Kuwait_product_analysis_Flavor_Distribution.csv', 'alt_columns': ['Flavor', 'flavor']},
            'taste': {'file': 'Kuwait_product_analysis_Taste_Distribution.csv', 'alt_columns': ['Taste', 'taste']},
            'thickness': {'file': 'Kuwait_product_analysis_Thickness_Distribution.csv',
                          'alt_columns': ['Thickness', 'thickness']},
            'length': {'file': 'Kuwait_product_analysis_Length_Distribution.csv', 'alt_columns': ['Length', 'length']},
            'pmi_products': {'file': 'Kuwait_product_analysis_PMI_Products.csv', 'alt_columns': []},
            'top_90pct_products': {'file': 'Kuwait_product_analysis_Top_90pct_Products.csv', 'alt_columns': []},
            'summary': {'file': 'Kuwait_product_analysis_Summary.csv', 'alt_columns': []},
            'passenger': {'file': 'Kuwait_product_analysis_Passenger_Distribution.csv', 'alt_columns': []}
        },
        'Jeju': {
            'flavor': {'file': 'jeju_product_analysis_Flavor_Distribution.csv', 'alt_columns': ['Flavor', 'flavor']},
            'taste': {'file': 'jeju_product_analysis_Taste_Distribution.csv', 'alt_columns': ['Taste', 'taste']},
            'thickness': {'file': 'jeju_product_analysis_Thickness_Distribution.csv',
                          'alt_columns': ['Thickness', 'thickness']},
            'length': {'file': 'jeju_product_analysis_Length_Distribution.csv', 'alt_columns': ['Length', 'length']},
            'pmi_products': {'file': 'jeju_product_analysis_PMI_Products.csv', 'alt_columns': []},
            'top_90pct_products': {'file': 'jeju_product_analysis_Top_90pct_Products.csv', 'alt_columns': []},
            'summary': {'file': 'jeju_product_analysis_Summary.csv', 'alt_columns': []}
        }
    }

    # Load data for each location
    for location, file_map in file_mappings.items():
        for key, file_info in file_map.items():
            file_path = os.path.join(data_dir, file_info['file'])
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    # Handle potential column name variations
                    if key in ['flavor', 'taste', 'thickness', 'length'] and file_info['alt_columns']:
                        for col in file_info['alt_columns']:
                            if col in df.columns:
                                # Standardize column names if needed
                                df = df.rename(columns={col: key})
                                break

                    data[location][key] = df
                    print(f"Loaded {location} {key} data: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    print(f"Error loading {location} {key} data: {e}")

    # Load comparison data with similar robust handling
    comparison_files = {
        'flavor': {'file': 'kuwait_jeju_attribute_analysis_Flavor_Distribution.csv',
                   'alt_columns': ['Flavor', 'flavor']},
        'taste': {'file': 'kuwait_jeju_attribute_analysis_Taste_Distribution.csv', 'alt_columns': ['Taste', 'taste']},
        'thickness': {'file': 'kuwait_jeju_attribute_analysis_Thickness_Distribution.csv',
                      'alt_columns': ['Thickness', 'thickness']},
        'length': {'file': 'kuwait_jeju_attribute_analysis_Length_Distribution.csv',
                   'alt_columns': ['Length', 'length']}
    }

    data['comparison'] = {}
    for key, file_info in comparison_files.items():
        file_path = os.path.join(data_dir, file_info['file'])
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                # Handle potential column name variations
                if file_info['alt_columns']:
                    for col in file_info['alt_columns']:
                        if col in df.columns:
                            # Standardize column names if needed
                            df = df.rename(columns={col: key})
                            break

                data['comparison'][key] = df
            except Exception as e:
                print(f"Error loading comparison {key} data: {e}")

    # Similar robust handling for gap data
    for location in ['Kuwait', 'Jeju']:
        data[location]['gaps'] = {}
        gap_files = {
            'flavor': f'kuwait_jeju_attribute_analysis_{location}_Flavor_Gaps.csv',
            'taste': f'kuwait_jeju_attribute_analysis_{location}_Taste_Gaps.csv',
            'thickness': f'kuwait_jeju_attribute_analysis_{location}_Thickness_Gaps.csv',
            'length': f'kuwait_jeju_attribute_analysis_{location}_Length_Gaps.csv'
        }

        for key, filename in gap_files.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    data[location]['gaps'][key] = df
                except Exception as e:
                    print(f"Error loading {location} {key} gap data: {e}")

    return data

def load_main_data(data_dir="./main_data"):
    """Load data from the main_data directory (SQL outputs)"""

    data = {}

    # Check if main_data directory exists
    if not os.path.exists(data_dir):
        return data

    # Try to load JSON files
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    for file in json_files:
        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, 'r') as f:
                data[file] = json.load(f)
        except Exception as e:
            print(f"Could not load {file}: {e}")

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
        return data

    # Parse the validation file
    try:
        with open(validation_file, 'r') as f:
            content = f.read()

            # Extract Kuwait data
            kuwait_section = content.split("Location: Kuwait")[1].split("Location: Jeju")[
                0] if "Location: Kuwait" in content else ""
            if kuwait_section:
                data['Kuwait']['category_c_score'] = float(kuwait_section.split("Category C Score: ")[1].split('\n')[
                                                               0]) if "Category C Score:" in kuwait_section else None
                data['Kuwait']['correlation'] = float(kuwait_section.split("Correlation: ")[1].split('\n')[
                                                          0]) if "Correlation:" in kuwait_section else None
                data['Kuwait']['r_squared'] = float(
                    kuwait_section.split("R²: ")[1].split('\n')[0]) if "R²:" in kuwait_section else None

            # Extract Jeju data
            jeju_section = content.split("Location: Jeju")[1] if "Location: Jeju" in content else ""
            if jeju_section:
                data['Jeju']['category_c_score'] = float(jeju_section.split("Category C Score: ")[1].split('\n')[
                                                             0]) if "Category C Score:" in jeju_section else None
                data['Jeju']['correlation'] = float(
                    jeju_section.split("Correlation: ")[1].split('\n')[0]) if "Correlation:" in jeju_section else None
                data['Jeju']['r_squared'] = float(
                    jeju_section.split("R²: ")[1].split('\n')[0]) if "R²:" in jeju_section else None

    except Exception as e:
        print(f"Could not load validation data: {e}")

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

    return color_maps


def create_attribute_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create a 2D heatmap data for two selected attributes"""

    # Get attribute values
    primary_values = location_data[location][primary_attr][primary_attr].unique()
    secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get product data
    products_df = location_data[location]['top_90pct_products']

    # Fill heatmap with volume data
    for p_val in primary_values:
        for s_val in secondary_values:
            volume = products_df[(products_df[primary_attr] == p_val) &
                                 (products_df[secondary_attr] == s_val)]['DF_Vol'].sum()
            heatmap_data.loc[p_val, s_val] = volume

    # Normalize to percentage of total volume
    total_volume = heatmap_data.sum().sum()
    if total_volume > 0:
        heatmap_data = (heatmap_data / total_volume) * 100

    return heatmap_data


def create_pmi_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create a 2D heatmap data for PMI products only"""

    # Get attribute values
    primary_values = location_data[location][primary_attr][primary_attr].unique()
    secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get PMI product data
    products_df = location_data[location]['pmi_products']

    # Fill heatmap with volume data
    for p_val in primary_values:
        for s_val in secondary_values:
            volume = products_df[(products_df[primary_attr] == p_val) &
                                 (products_df[secondary_attr] == s_val)]['DF_Vol'].sum()
            heatmap_data.loc[p_val, s_val] = volume

    # Normalize to percentage of total volume
    total_volume = heatmap_data.sum().sum()
    if total_volume > 0:
        heatmap_data = (heatmap_data / total_volume) * 100

    return heatmap_data


def create_ideal_heatmap_data(location_data, location, primary_attr, secondary_attr):
    """Create an ideal 2D heatmap based on passenger preferences (Category C)"""

    # Get attribute values
    primary_values = location_data[location][primary_attr][primary_attr].unique()
    secondary_values = location_data[location][secondary_attr][secondary_attr].unique()

    # Initialize heatmap data
    heatmap_data = pd.DataFrame(0, index=primary_values, columns=secondary_values)

    # Get attribute distributions
    primary_dist = location_data[location][primary_attr]
    secondary_dist = location_data[location][secondary_attr]

    # Create dictionaries for ideal percentages
    primary_ideal = {}
    for _, row in primary_dist.iterrows():
        primary_ideal[row[primary_attr]] = row['Ideal_Percentage'] if 'Ideal_Percentage' in primary_dist.columns else 0

    secondary_ideal = {}
    for _, row in secondary_dist.iterrows():
        secondary_ideal[row[secondary_attr]] = row[
            'Ideal_Percentage'] if 'Ideal_Percentage' in secondary_dist.columns else 0

    # Calculate joint probabilities (assuming independence)
    for p_val in primary_values:
        for s_val in secondary_values:
            # Joint probability = P(A) * P(B) / 100 (to normalize percentage)
            heatmap_data.loc[p_val, s_val] = (primary_ideal.get(p_val, 0) * secondary_ideal.get(s_val, 0)) / 100

    # Normalize to ensure total is 100%
    total = heatmap_data.sum().sum()
    if total > 0:
        heatmap_data = (heatmap_data / total) * 100

    return heatmap_data


def create_gap_heatmap_data(actual_data, ideal_data):
    """Create a gap heatmap showing the difference between actual and ideal"""

    # Calculate the gap
    gap_data = actual_data - ideal_data

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
    # Initialize figure
    fig = go.Figure()

    # Add heatmap as background
    heatmap_z = heatmap_data.values

    # Create heatmap
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

    # Get color mapping
    color_map = attribute_colors[color_attr]

    # Add bubbles for each attribute combination
    for i, p_val in enumerate(heatmap_data.index):
        for j, s_val in enumerate(heatmap_data.columns):
            # Cell value (percentage)
            cell_pct = heatmap_data.loc[p_val, s_val]

            # Skip if percentage is too small
            if cell_pct < 0.5:
                continue

            # Get color attribute values for this cell position
            color_values = color_attr_data[color_attr].unique()

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

    # Update layout
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
            # For product data, just store CR_BrandId, attributes, and DF_Vol
            cols = ['CR_BrandId', 'flavor', 'taste', 'thickness', 'length', 'DF_Vol']
            available_cols = [col for col in cols if col in location_data[loc]['pmi_products'].columns]
            selected_data[loc]['pmi_products'] = location_data[loc]['pmi_products'][available_cols].to_dict(
                orient='records')

        if 'top_90pct_products' in location_data[loc]:
            # For product data, just store CR_BrandId, attributes, and DF_Vol
            cols = ['CR_BrandId', 'flavor', 'taste', 'thickness', 'length', 'DF_Vol']
            available_cols = [col for col in cols if col in location_data[loc]['top_90pct_products'].columns]
            selected_data[loc]['top_90pct_products'] = location_data[loc]['top_90pct_products'][available_cols].to_dict(
                orient='records')

    # Convert to JSON
    return json.dumps(selected_data), json.dumps(validation_data)


# Callback to validate attribute selection
@app.callback(
    Output('validation-warnings', 'children'),
    [Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value'),
     Input('color-attr-dropdown', 'value')]
)
def validate_attributes(primary_attr, secondary_attr, color_attr):
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
    if not stored_data:
        return []

    # Parse stored data
    data = json.loads(stored_data)

    # Market share (default values)
    market_shares = {
        "Kuwait": 75,  # Approximate as per documentation
        "Jeju": 12  # Approximate as per documentation
    }

    # Try to extract more accurate market share from data
    if 'summary' in data[location]:
        summary = data[location]['summary']
        for row in summary:
            if 'Category' in row and 'Metric' in row and 'Value' in row:
                if 'Market Share' in row['Category'] and 'PMI Share' in row['Metric']:
                    try:
                        share_str = row['Value']
                        if isinstance(share_str, str) and '%' in share_str:
                            market_shares[location] = float(share_str.strip('%'))
                    except:
                        pass

    # Create metric cards
    market_share_card = html.Div([
        html.Div(f"{market_shares[location]}%", className="metric-value"),
        html.Div("PMI Market Share", className="metric-label")
    ], className="metric-card")

    # Get SKU counts
    pmi_skus = len(data[location].get('pmi_products', []))
    total_skus = len(data[location].get('top_90pct_products', []))

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


# Callback to display category scores
@app.callback(
    Output('category-scores', 'children'),
    [Input('location-dropdown', 'value'),
     Input('stored-validation-data', 'children')]
)
def display_category_scores(location, stored_validation_data):
    if not stored_validation_data:
        return []

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


# Callback to display portfolio insights
@app.callback(
    Output('portfolio-insights', 'children'),
    [Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('stored-data', 'children')]
)
def display_portfolio_insights(location, primary_attr, stored_data):
    if not stored_data:
        return []

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
    if primary_attr in data[location]:
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


# Callback to display tab content
@app.callback(
    Output('tab-content', 'children'),
    [Input('visualization-tabs', 'value'),
     Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value'),
     Input('color-attr-dropdown', 'value'),
     Input('stored-data', 'children')]
)
def display_tab_content(tab, location, primary_attr, secondary_attr, color_attr, stored_data):
    if not stored_data:
        return []

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

    # Get attribute colors
    attribute_colors = get_attribute_colors()

    if tab == 'comparison':
        # Create market, PMI, and ideal heatmaps
        try:
            market_heatmap = create_attribute_heatmap_data(data, location, primary_attr, secondary_attr)
            pmi_heatmap = create_pmi_heatmap_data(data, location, primary_attr, secondary_attr)
            ideal_heatmap = create_ideal_heatmap_data(data, location, primary_attr, secondary_attr)

            # Create visualizations
            market_fig = create_shelf_visualization(
                market_heatmap, data[location][color_attr], color_attr,
                primary_attr, secondary_attr, attribute_colors,
                f"{location} - Market Portfolio", False
            )

            pmi_fig = create_shelf_visualization(
                pmi_heatmap, data[location][color_attr], color_attr,
                primary_attr, secondary_attr, attribute_colors,
                f"{location} - PMI Portfolio", True
            )

            # Create layout
            return [
                html.Div([
                    html.H3("Market vs PMI Portfolio Comparison", className="card-title"),
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=market_fig, id='market-graph')
                        ], className="graph-container", style={"width": "50%"}),
                        html.Div([
                            dcc.Graph(figure=pmi_fig, id='pmi-graph')
                        ], className="graph-container", style={"width": "50%"})
                    ], style={"display": "flex", "justifyContent": "space-between"}),

                    html.Div([
                        html.H3("Gap Analysis", className="card-title"),
                        html.Div([
                            html.Div([
                                html.H4("Market vs Ideal Portfolio Gaps"),
                                display_gap_analysis(market_heatmap, ideal_heatmap, primary_attr, secondary_attr)
                            ], className="card", style={"width": "48%"}),
                            html.Div([
                                html.H4("PMI vs Ideal Portfolio Gaps"),
                                display_gap_analysis(pmi_heatmap, ideal_heatmap, primary_attr, secondary_attr)
                            ], className="card", style={"width": "48%"})
                        ], style={"display": "flex", "justifyContent": "space-between"})
                    ])
                ])
            ]

        except Exception as e:
            return [
                html.Div([
                    html.H3("Error Creating Visualization", className="card-title"),
                    html.Div(f"An error occurred: {str(e)}", className="warning-box")
                ])
            ]

    elif tab == 'pmi':
        # Create PMI heatmap
        try:
            pmi_heatmap = create_pmi_heatmap_data(data, location, primary_attr, secondary_attr)

            # Create visualization
            pmi_fig = create_shelf_visualization(
                pmi_heatmap, data[location][color_attr], color_attr,
                primary_attr, secondary_attr, attribute_colors,
                f"{location} - PMI Portfolio Detail", True
            )

            # Get PMI product data
            pmi_products = data[location]['pmi_products']

            # Create layout
            return [
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
            return [
                html.Div([
                    html.H3("Error Creating Visualization", className="card-title"),
                    html.Div(f"An error occurred: {str(e)}", className="warning-box")
                ])
            ]

    elif tab == 'ideal':
        # Create ideal heatmap
        try:
            ideal_heatmap = create_ideal_heatmap_data(data, location, primary_attr, secondary_attr)

            # Create visualization
            ideal_fig = create_shelf_visualization(
                ideal_heatmap, data[location][color_attr], color_attr,
                primary_attr, secondary_attr, attribute_colors,
                f"{location} - Ideal Portfolio (Based on Category C)", False
            )

            # Create layout
            return [
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
            return [
                html.Div([
                    html.H3("Error Creating Visualization", className="card-title"),
                    html.Div(f"An error occurred: {str(e)}", className="warning-box")
                ])
            ]

    return []


def display_gap_analysis(actual_data, ideal_data, primary_attr, secondary_attr):
    """Display gap analysis between actual and ideal data"""

    # Calculate gap
    gap_data = create_gap_heatmap_data(actual_data, ideal_data)

    # Get top gaps (both positive and negative)
    gap_values = []

    for p_val in gap_data.index:
        for s_val in gap_data.columns:
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


def display_product_statistics(product_data, attr):
    """Display product statistics for a given attribute"""

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


def display_brand_statistics(product_data):
    """Display brand statistics"""

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


def display_ideal_distribution(attr_data, attr_name):
    """Display ideal distribution for an attribute"""

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


# Main entry point
if __name__ == '__main__':
    app.run_server(debug=True)