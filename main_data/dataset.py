import pandas as pd
import os

def create_enhanced_portfolio_dataset(locations=['Kuwait', 'Jeju']):
    """Create enhanced product portfolio dataset with Category C metrics using base tables"""
    # Load base tables from main_data
    base_list = pd.read_csv('base_list.csv')
    df_volume = pd.read_csv('df_volume.csv')
    paris_output = pd.read_csv('PARIS_Output.csv')
    cat_c_scores = pd.read_csv('CategoryCScores.csv')
    selma_df_map = pd.read_csv('selma_df_map.csv')  # Add this line to load the attributes data

    # Print column names for debugging
    print("Paris output columns:", paris_output.columns.tolist())
    print("Base list columns:", base_list.columns.tolist())
    print("Selma df map columns:", selma_df_map.columns.tolist())

    # Filter for target locations
    base_location_products = base_list[base_list['Location'].isin(locations)]
    volumes_location = df_volume[df_volume['Location'].isin(locations)]
    selma_location = selma_df_map[selma_df_map['Location'].isin(locations)]  # Filter selma data

    # Join to get product data with volumes
    location_products = pd.merge(
        base_location_products,
        volumes_location,
        on=['CR_BrandId', 'Location', 'TMO'],
        how='inner'
    )

    # Join with selma_df_map to get product attributes
    location_products = pd.merge(
        location_products,
        selma_location[['Location', 'CR_BrandId', 'Flavor', 'Taste', 'Thickness', 'Length']],
        on=['Location', 'CR_BrandId'],
        how='left'
    )

    print("Location products columns:", location_products.columns.tolist())

    # Calculate market share for each product within its location
    location_products['Total_Location_Volume'] = location_products.groupby('Location')['DF_Vol'].transform('sum')
    location_products['Share'] = (location_products['DF_Vol'] / location_products['Total_Location_Volume']) * 100

    # Add Category C scores for each location
    location_products = pd.merge(
        location_products,
        cat_c_scores[['Location', 'Cat_C']],
        on='Location',
        how='left'
    )

    # Add delta values from PARIS_Output
    # Create mapping dictionaries for product attributes to their delta values
    delta_mapping = {}

    # Get the actual column names (case-insensitive) for paris_output
    location_col = next((col for col in paris_output.columns if col.lower() == 'location'), 'Location')
    flavor_col = next((col for col in paris_output.columns if col.lower() == 'flavor'), 'Flavor')
    taste_col = next((col for col in paris_output.columns if col.lower() == 'taste'), 'Taste')
    thickness_col = next((col for col in paris_output.columns if col.lower() == 'thickness'), 'Thickness')
    length_col = next((col for col in paris_output.columns if col.lower() == 'length'), 'Length')

    for _, row in paris_output.iterrows():
        key = (row[location_col], row[flavor_col], row[taste_col], row[thickness_col], row[length_col])
        delta_mapping[key] = {
            'Real_So_Segment': row['Real_So_Segment'],
            'Ideal_So_Segment': row['Ideal_So_Segment'],
            'Delta_SoS': row['Delta_SoS']
        }

    # Get column names for location_products - MOVED THIS OUTSIDE THE LOOP
    base_location_col = 'Location'  # This one is known to exist

    # Use more flexible matching for the attribute columns
    base_flavor_col = next((col for col in location_products.columns if 'flavor' in col.lower()), None)
    base_taste_col = next((col for col in location_products.columns if 'taste' in col.lower()), None)
    base_thickness_col = next((col for col in location_products.columns if 'thick' in col.lower()), None)
    base_length_col = next((col for col in location_products.columns if 'length' in col.lower()), None)

    # Check if we found all the required columns
    if not all([base_flavor_col, base_taste_col, base_thickness_col, base_length_col]):
        missing = []
        if not base_flavor_col: missing.append('flavor')
        if not base_taste_col: missing.append('taste')
        if not base_thickness_col: missing.append('thickness')
        if not base_length_col: missing.append('length')

        raise ValueError(f"Could not find columns in location_products: {', '.join(missing)}")

    # Add delta values to products
    for index, row in location_products.iterrows():
        key = (row[base_location_col], row[base_flavor_col], row[base_taste_col],
               row[base_thickness_col], row[base_length_col])
        if key in delta_mapping:
            location_products.at[index, 'Delta_SoS'] = delta_mapping[key]['Delta_SoS']
            location_products.at[index, 'Real_So_Segment'] = delta_mapping[key]['Real_So_Segment']
            location_products.at[index, 'Ideal_So_Segment'] = delta_mapping[key]['Ideal_So_Segment']
    # Add alignment indicator
    location_products['Alignment'] = location_products['Delta_SoS'].apply(
        lambda x: 'Well Aligned' if abs(x) < 0.05 else
        ('Under-Represented' if x > 0.05 else 'Over-Represented')
    )

    # Create visualization category field (Actual vs Ideal)
    # For each location, create both actual (PMI only) and ideal (all products) datasets
    result_datasets = {}

    for location in locations:
        location_data = location_products[location_products['Location'] == location].copy()

        # Create Actual dataset (PMI only)
        actual_pmi = location_data[location_data['TMO'] == 'PMI'].copy()
        actual_pmi['Category'] = f'Actual_{location}'

        # Create Ideal dataset (all products)
        ideal_market = location_data.copy()
        ideal_market['Category'] = f'Ideal_{location}'

        # Combine for this location
        combined = pd.concat([actual_pmi, ideal_market])
        result_datasets[location] = combined

    # Return as either a dict of location datasets or a combined dataframe
    all_combined = pd.concat(result_datasets.values())
    return all_combined


def create_attribute_gap_dataset(location):
    """Create dataset showing gaps by attribute"""
    # Load PARIS data
    paris_data = pd.read_csv('PARIS_Output.csv')
    paris_location = paris_data[paris_data['Location'] == location]

    # Create aggregated datasets for each attribute
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    attribute_gaps = {}

    for attr in attributes:
        # Group by the attribute and calculate aggregate values
        grouped = paris_location.groupby(attr).agg({
            'Real_So_Segment': 'sum',
            'Ideal_So_Segment': 'sum',
            'Delta_SoS': 'sum'
        }).reset_index()

        # Calculate percentage values
        grouped['Real_Percentage'] = grouped['Real_So_Segment'] * 100
        grouped['Ideal_Percentage'] = grouped['Ideal_So_Segment'] * 100
        grouped['Gap_Percentage'] = grouped['Delta_SoS'] * 100

        # Rank by absolute gap size
        grouped['Gap_Rank'] = grouped['Gap_Percentage'].abs().rank(ascending=False)

        attribute_gaps[attr] = grouped

    return attribute_gaps


def create_passenger_influence_dataset(location):
    """Create dataset showing passenger nationality influence on ideal mix"""
    # Load passenger data
    pax_data = pd.read_csv('pax_data.csv')
    location_pax = pax_data[pax_data['Market'] == location]

    # Get top nationalities
    nationality_dist = location_pax.groupby('Nationality')['Pax'].sum().reset_index()
    nationality_dist['Percentage'] = nationality_dist['Pax'] / nationality_dist['Pax'].sum() * 100
    nationality_dist = nationality_dist.sort_values('Percentage', ascending=False)

    # Get top 5 nationalities
    top_nationalities = nationality_dist.head(5)

    # This would ideally be joined with domestic preference data to show
    # how each nationality influences the ideal product mix

    return top_nationalities


def create_visualization_dataset(locations=None):
    """Create final datasets for visualization"""
    if locations is None:
        locations = ['Kuwait', 'Jeju']
    all_datasets = {}

    for location in locations:
        # Create core datasets
        portfolio_data = create_enhanced_portfolio_dataset([location])  # Pass as list, not string
        attribute_gaps = create_attribute_gap_dataset(location)
        passenger_data = create_passenger_influence_dataset(location)

        # Split into actual and ideal for visualization
        actual_pmi = portfolio_data[portfolio_data['TMO'] == 'PMI'].copy()
        actual_pmi['Category'] = f'Actual_{location}'

        ideal_market = portfolio_data.copy()  # All TMOs for ideal view
        ideal_market['Category'] = f'Ideal_{location}'

        # Combine for final visualization dataset
        combined = pd.concat([actual_pmi, ideal_market])

        # Add insights column that combines Category C and Delta insights
        combined['Insight'] = combined.apply(
            lambda row: f"Delta: {row['Delta_SoS']:.2f}, Alignment: {row['Alignment']}, Cat C: {row['Cat_C']:.2f}",
            axis=1
        )

        all_datasets[location] = {
            'visualization_data': combined,
            'attribute_gaps': attribute_gaps,
            'passenger_influence': passenger_data
        }

    return all_datasets

def export_visualization_datasets(all_datasets, output_folder='visualization_data'):
    """
    Export all visualization datasets to Excel files.

    Args:
        all_datasets: Dictionary output from create_visualization_dataset()
        output_folder: Directory path to save the Excel files
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for location, datasets in all_datasets.items():
        # Export the main visualization dataset
        vis_data = datasets['visualization_data']
        vis_data_path = f'{output_folder}/{location}_visualization_data.xlsx'
        vis_data.to_excel(vis_data_path, index=False)

        # Export attribute gaps datasets
        for attr, gap_data in datasets['attribute_gaps'].items():
            attr_path = f'{output_folder}/{location}_{attr}_gaps.xlsx'
            gap_data.to_excel(attr_path, index=False)

        # Export passenger influence data
        pax_path = f'{output_folder}/{location}_passenger_influence.xlsx'
        datasets['passenger_influence'].to_excel(pax_path, index=False)

        print(f"Exported {location} datasets to {output_folder}/")

    # Option: Create a combined Excel with multiple sheets
    with pd.ExcelWriter(f'{output_folder}/all_visualization_data.xlsx') as writer:
        for location, datasets in all_datasets.items():
            # Main sheet
            datasets['visualization_data'].to_excel(
                writer, sheet_name=f'{location}_Main', index=False)

            # Attribute gaps
            for attr, gap_data in datasets['attribute_gaps'].items():
                gap_data.to_excel(
                    writer, sheet_name=f'{location}_{attr[:3]}', index=False)

            # Passenger data
            datasets['passenger_influence'].to_excel(
                writer, sheet_name=f'{location}_Pax', index=False)

        print(f"Exported combined dataset to {output_folder}/all_visualization_data.xlsx")

def validate_column_names():
    """Validate column names in data files against expected names"""
    # Load each critical file and validate columns
    validation_results = {}

    # Check base_list.csv
    try:
        base_list = pd.read_csv('base_list.csv')
        expected_base_columns = ['SKU', 'Item per Bundle', 'CR_BrandId', 'DF_Market', 'Location', 'TMO']
        missing_base = [col for col in expected_base_columns if col not in base_list.columns]
        validation_results['base_list.csv'] = {'status': 'OK' if not missing_base else 'MISSING', 'missing': missing_base}
    except Exception as e:
        validation_results['base_list.csv'] = {'status': 'ERROR', 'message': str(e)}

    # Check df_volume.csv
    try:
        df_volume = pd.read_csv('df_volume.csv')
        expected_volume_columns = ['Year', 'Product Category', 'Location', 'DF_Market', 'TMO', 'Brand Family', 'CR_BrandId', 'DF_Vol']
        missing_volume = [col for col in expected_volume_columns if col not in df_volume.columns]
        validation_results['df_volume.csv'] = {'status': 'OK' if not missing_volume else 'MISSING', 'missing': missing_volume}
    except Exception as e:
        validation_results['df_volume.csv'] = {'status': 'ERROR', 'message': str(e)}

    # Check PARIS_Output.csv
    try:
        paris_output = pd.read_csv('PARIS_Output.csv')
        expected_paris_columns = ['Location', 'DF_Market', 'Flavor', 'Taste', 'Thickness', 'Length', 'DF_Vol', 'Real_So_Segment', 'Ideal_So_Segment', 'Delta_SoS']
        missing_paris = [col for col in expected_paris_columns if col not in paris_output.columns]
        validation_results['PARIS_Output.csv'] = {'status': 'OK' if not missing_paris else 'MISSING', 'missing': missing_paris}
    except Exception as e:
        validation_results['PARIS_Output.csv'] = {'status': 'ERROR', 'message': str(e)}

    # Check attribute columns in location_products
    # This is a key issue - df_volume doesn't have attribute columns according to column_names.py
    # We might need selma_df_map.csv for these attributes
    try:
        selma_df_map = pd.read_csv('selma_df_map.csv')
        expected_selma_columns = ['DF_Market', 'Product Category', 'Location', 'CR_BrandId', 'Flavor', 'Taste', 'Thickness', 'Length']
        missing_selma = [col for col in expected_selma_columns if col not in selma_df_map.columns]
        validation_results['selma_df_map.csv'] = {'status': 'OK' if not missing_selma else 'MISSING', 'missing': missing_selma}
    except Exception as e:
        validation_results['selma_df_map.csv'] = {'status': 'ERROR', 'message': str(e)}

    return validation_results

# Add to main
if __name__ == "__main__":
    # Validate column names first
    validation_results = validate_column_names()
    for file, result in validation_results.items():
        if result['status'] == 'OK':
            print(f"✅ {file}: All required columns present")
        elif result['status'] == 'MISSING':
            print(f"❌ {file}: Missing columns: {result['missing']}")
        else:
            print(f"❌ {file}: Error: {result['message']}")

    # If validation passes, continue with dataset creation
    if all(result['status'] == 'OK' for result in validation_results.values()):
        print("\nColumn validation passed, creating datasets...")
        datasets = create_visualization_dataset()
        export_visualization_datasets(datasets)
    else:
        print("\n⚠️ Column validation failed. Please fix the issues before proceeding.")