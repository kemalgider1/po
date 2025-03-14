import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path


# Helper function to handle column name capitalization
def standardize_column_names(df, column_map=None):
    """
    Standardize column names to lowercase
    If column_map is provided, use it to map specific columns
    """
    if column_map is None:
        column_map = {}

    # Create a mapping of all columns to lowercase
    all_column_map = {col: col.lower() for col in df.columns}

    # Update with any specific mappings provided
    all_column_map.update(column_map)

    # Apply the mapping only for columns that exist in the DataFrame
    valid_column_map = {k: v for k, v in all_column_map.items() if k in df.columns}

    return df.rename(columns=valid_column_map)


# Load data with proper column handling
def load_location_data(data_dir="./locations_data"):
    """Load data from the locations_data directory with robust column name handling"""

    data = {
        'Kuwait': {},
        'Jeju': {}
    }

    # Define expected column mappings for each file type
    column_mappings = {
        'flavor': {'Flavor': 'flavor'},
        'taste': {'Taste': 'taste'},
        'thickness': {'Thickness': 'thickness'},
        'length': {'Length': 'length'},
        'pmi_products': {},
        'top_90pct_products': {},
        'summary': {},
    }

    # Define file mappings
    file_mappings = {
        'Kuwait': {
            'flavor': 'Kuwait_product_analysis_Flavor_Distribution.csv',
            'taste': 'Kuwait_product_analysis_Taste_Distribution.csv',
            'thickness': 'Kuwait_product_analysis_Thickness_Distribution.csv',
            'length': 'Kuwait_product_analysis_Length_Distribution.csv',
            'pmi_products': 'Kuwait_product_analysis_PMI_Products.csv',
            'top_90pct_products': 'Kuwait_product_analysis_Top_90pct_Products.csv',
            'summary': 'Kuwait_product_analysis_Summary.csv'
        },
        'Jeju': {
            'flavor': 'jeju_product_analysis_Flavor_Distribution.csv',
            'taste': 'jeju_product_analysis_Taste_Distribution.csv',
            'thickness': 'jeju_product_analysis_Thickness_Distribution.csv',
            'length': 'jeju_product_analysis_Length_Distribution.csv',
            'pmi_products': 'jeju_product_analysis_PMI_Products.csv',
            'top_90pct_products': 'jeju_product_analysis_Top_90pct_Products.csv',
            'summary': 'jeju_product_analysis_Summary.csv'
        }
    }

    # Load data for each location
    for location, file_map in file_mappings.items():
        for key, filename in file_map.items():
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"Loaded {location} {key} data with columns: {list(df.columns)}")

                    # Standardize column names
                    df = standardize_column_names(df, column_mappings.get(key, {}))

                    # Store the data
                    data[location][key] = df
                    print(f"  After standardization: {list(df.columns)}")
                except Exception as e:
                    print(f"Error loading {location} {key} data: {e}")

    # Load comparison data
    data['comparison'] = {}
    comparison_files = {
        'flavor': 'kuwait_jeju_attribute_analysis_Flavor_Distribution.csv',
        'taste': 'kuwait_jeju_attribute_analysis_Taste_Distribution.csv',
        'thickness': 'kuwait_jeju_attribute_analysis_Thickness_Distribution.csv',
        'length': 'kuwait_jeju_attribute_analysis_Length_Distribution.csv'
    }

    for key, filename in comparison_files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Standardize column names
                df = standardize_column_names(df, column_mappings.get(key, {}))
                data['comparison'][key] = df
            except Exception as e:
                print(f"Error loading comparison {key} data: {e}")

    return data


def create_shelf_visualization(location_data, location, view_type='actual', primary_attr='flavor',
                               secondary_attr='taste'):
    """
    Create a shelf visualization for a given location and view type

    Parameters:
    -----------
    location_data : dict
        Dictionary containing the data for the location
    location : str
        The location to visualize (Kuwait or Jeju)
    view_type : str
        The type of view to create (actual or ideal)
    primary_attr : str
        The primary attribute to use for visualization
    secondary_attr : str
        The secondary attribute to use for visualization

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Define color mappings for attributes
    color_maps = {
        'flavor': {
            'regular': '#8B4513',  # Brown
            'menthol': '#00FF00',  # Green
            'menthol caps': '#00AA00',  # Dark Green
            'ntd': '#ADD8E6',  # Light Blue
            'ntd caps': '#0000FF'  # Blue
        },
        'taste': {
            'full flavor': '#FF0000',  # Red
            'lights': '#FFA500',  # Orange
            'ultralights': '#FFFF00',  # Yellow
            '1mg': '#FFFFAA'  # Light Yellow
        },
        'thickness': {
            'std': '#800080',  # Purple
            'sli': '#FF00FF',  # Magenta
            'ssl': '#BA55D3',  # Medium Orchid
            'mag': '#DA70D6'  # Orchid
        },
        'length': {
            'ks': '#2E8B57',  # Sea Green
            '100': '#3CB371',  # Medium Sea Green
            'longer than ks': '#66CDAA',  # Medium Aquamarine
            'long size': '#8FBC8F',  # Dark Sea Green
            'regular size': '#90EE90'  # Light Green
        }
    }

    # Get the data based on view type
    if view_type == 'actual':
        # Use PMI products for actual view
        products_df = location_data[location]['pmi_products'].copy()
    else:
        # Use market data for ideal view
        products_df = location_data[location]['top_90pct_products'].copy()

    # Get unique values for primary and secondary attributes
    # Convert column access to lowercase
    primary_attr_values = location_data[location][primary_attr][primary_attr].unique()
    secondary_attr_values = location_data[location][secondary_attr][secondary_attr].unique()

    # Create a figure
    fig = go.Figure()

    # Calculate the total volume for normalization
    total_volume = products_df['df_vol'].sum() if 'df_vol' in products_df.columns else 1.0

    # Create a grid layout
    n_rows = len(primary_attr_values)
    n_cols = len(secondary_attr_values)

    # Create a heatmap for the background
    if primary_attr in products_df.columns and secondary_attr in products_df.columns:
        # Create a grid of product volumes by attributes
        heatmap_data = np.zeros((n_rows, n_cols))

        for i, p_val in enumerate(primary_attr_values):
            for j, s_val in enumerate(secondary_attr_values):
                # Filter products by primary and secondary attributes
                mask = (products_df[primary_attr].str.lower() == p_val.lower()) & \
                       (products_df[secondary_attr].str.lower() == s_val.lower())

                # Calculate the volume for this cell
                volume = products_df.loc[mask, 'df_vol'].sum() if 'df_vol' in products_df.columns else 0.0

                # Store the normalized volume
                heatmap_data[i, j] = (volume / total_volume) * 100

        # Add heatmap to figure
        fig.add_trace(go.Heatmap(
            z=heatmap_data,
            x=secondary_attr_values,
            y=primary_attr_values,
            colorscale='Blues',
            colorbar=dict(title='% of Volume'),
            hoverinfo='text',
            text=[[f"{p_val} × {s_val}: {heatmap_data[i, j]:.1f}%"
                   for j, s_val in enumerate(secondary_attr_values)]
                  for i, p_val in enumerate(primary_attr_values)]
        ))

        # Add products as markers
        for i, p_val in enumerate(primary_attr_values):
            for j, s_val in enumerate(secondary_attr_values):
                # Filter products by primary and secondary attributes
                mask = (products_df[primary_attr].str.lower() == p_val.lower()) & \
                       (products_df[secondary_attr].str.lower() == s_val.lower())

                if mask.sum() > 0:
                    # Get the products for this cell
                    cell_products = products_df[mask].copy()

                    # Calculate the total volume for this cell
                    cell_volume = cell_products['df_vol'].sum() if 'df_vol' in cell_products.columns else 0.0

                    # Normalize the products' volumes within the cell
                    if cell_volume > 0:
                        cell_products['normalized_volume'] = cell_products['df_vol'] / cell_volume
                    else:
                        cell_products['normalized_volume'] = 0.0

                    # Add a marker for each product
                    for k, product in cell_products.iterrows():
                        # Get the product color based on a third attribute
                        color_attr = list(set(['flavor', 'taste', 'thickness', 'length']) -
                                          set([primary_attr, secondary_attr]))[0]

                        color_value = product[color_attr].lower() if color_attr in product else 'unknown'
                        color = color_maps[color_attr].get(color_value, '#CCCCCC')

                        # Calculate marker size based on volume
                        size = np.sqrt(product['normalized_volume']) * 50

                        # Add slight offsets to avoid complete overlap
                        x_offset = (np.random.random() - 0.5) * 0.2
                        y_offset = (np.random.random() - 0.5) * 0.2

                        # Add the marker
                        fig.add_trace(go.Scatter(
                            x=[j + x_offset],
                            y=[i + y_offset],
                            mode='markers',
                            marker=dict(
                                size=size,
                                color=color,
                                line=dict(color='white', width=2)
                            ),
                            name=f"{product['brand family'] if 'brand family' in product else ''} - {product[color_attr]}",
                            hoverinfo='text',
                            text=f"{product['brand family'] if 'brand family' in product else ''}<br>" +
                                 f"{product['sku'] if 'sku' in product else ''}<br>" +
                                 f"Volume: {product['df_vol']:.1f}<br>" +
                                 f"{primary_attr.capitalize()}: {product[primary_attr]}<br>" +
                                 f"{secondary_attr.capitalize()}: {product[secondary_attr]}<br>" +
                                 f"{color_attr.capitalize()}: {product[color_attr]}"
                        ))

    # Update layout
    fig.update_layout(
        title=f"{location} - {'PMI Portfolio' if view_type == 'actual' else 'Ideal Market Portfolio'}",
        xaxis=dict(
            title=secondary_attr.capitalize(),
            tickmode='array',
            tickvals=list(range(len(secondary_attr_values))),
            ticktext=secondary_attr_values
        ),
        yaxis=dict(
            title=primary_attr.capitalize(),
            tickmode='array',
            tickvals=list(range(len(primary_attr_values))),
            ticktext=primary_attr_values
        ),
        height=600,
        width=800
    )

    return fig


# Initialize the app
app = dash.Dash(__name__)

# Load data
data = load_location_data()

# Define app layout
app.layout = html.Div([
    html.H1("PMI Portfolio Shelf Visualization"),

    html.Div([
        html.Div([
            html.Label("Select Location:"),
            dcc.Dropdown(
                id='location-dropdown',
                options=[
                    {'label': 'Kuwait (Well-Aligned)', 'value': 'Kuwait'},
                    {'label': 'Jeju (Misaligned)', 'value': 'Jeju'}
                ],
                value='Kuwait'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Primary Attribute (Y-axis):"),
            dcc.Dropdown(
                id='primary-attr-dropdown',
                options=[
                    {'label': 'Flavor', 'value': 'flavor'},
                    {'label': 'Taste', 'value': 'taste'},
                    {'label': 'Thickness', 'value': 'thickness'},
                    {'label': 'Length', 'value': 'length'}
                ],
                value='flavor'
            )
        ], style={'width': '33%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Secondary Attribute (X-axis):"),
            dcc.Dropdown(
                id='secondary-attr-dropdown',
                options=[
                    {'label': 'Flavor', 'value': 'flavor'},
                    {'label': 'Taste', 'value': 'taste'},
                    {'label': 'Thickness', 'value': 'thickness'},
                    {'label': 'Length', 'value': 'length'}
                ],
                value='taste'
            )
        ], style={'width': '33%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.Div([
            html.H3("PMI Portfolio"),
            dcc.Graph(id='actual-shelf')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Ideal Market Portfolio"),
            dcc.Graph(id='ideal-shelf')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),

    html.Div([
        html.H3("Portfolio Summary"),
        html.Div(id='summary-stats')
    ])
])


# Define callbacks
@app.callback(
    [Output('actual-shelf', 'figure'),
     Output('ideal-shelf', 'figure'),
     Output('summary-stats', 'children')],
    [Input('location-dropdown', 'value'),
     Input('primary-attr-dropdown', 'value'),
     Input('secondary-attr-dropdown', 'value')]
)
def update_figures(location, primary_attr, secondary_attr):
    # Print for debugging
    print(f"Updating figures for {location} with {primary_attr} and {secondary_attr}")

    # Check that we have the necessary data
    if location not in data or primary_attr not in data[location] or secondary_attr not in data[location]:
        return go.Figure(), go.Figure(), html.Div("Data not available for the selected options")

    # Create figures
    actual_fig = create_shelf_visualization(data, location, 'actual', primary_attr, secondary_attr)
    ideal_fig = create_shelf_visualization(data, location, 'ideal', primary_attr, secondary_attr)

    # Create summary stats
    if 'summary' in data[location]:
        summary_df = data[location]['summary']

        # Extract market share
        try:
            market_share = summary_df[summary_df['metric'] == 'PMI Share in Top 90% Products']['value'].iloc[0]
            market_share = market_share.replace('%', '') if isinstance(market_share, str) else market_share
            market_share = float(market_share)
        except:
            market_share = "N/A"

        # Create summary component
        summary_stats = html.Div([
            html.H4(f"{location} Market Summary"),
            html.P(f"PMI Market Share: {market_share}%"),
            html.P(f"Category C Score: {data[location].get('category_c_score', 'N/A')}"),

            html.H4("Portfolio Alignment:"),
            html.Ul([
                html.Li(f"{primary_attr.capitalize()} Alignment: "
                        f"{'Good' if location == 'Kuwait' else 'Poor'}"),
                html.Li(f"{secondary_attr.capitalize()} Alignment: "
                        f"{'Good' if location == 'Kuwait' else 'Poor'}"),
            ]),

            html.H4("Recommendations:"),
            html.Ul([
                html.Li(f"{'Maintain current mix' if location == 'Kuwait' else 'Optimize product attributes'}")
            ])
        ])
    else:
        summary_stats = html.Div("Summary data not available")

    return actual_fig, ideal_fig, summary_stats


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)