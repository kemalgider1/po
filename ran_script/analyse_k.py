import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the data directory
data_dir = Path('/Users/kemalgider/Desktop/PORTFOLIO/All_tables/data/')


# Load the relevant data files
def load_data():
    print("Loading data files...")
    files = {
        'cat_a_df_vols': pd.read_csv(data_dir / 'cat_a_df_vols.csv'),
        'df_volume': pd.read_csv(data_dir / 'df_volume.csv'),
        'market_mix': pd.read_csv(data_dir / 'Market_Mix.csv'),
        'paris_output': pd.read_csv(data_dir / 'PARIS_Output.csv'),
        'market_summary': pd.read_csv(data_dir / 'MarketSummary.csv'),
        'pax_data': pd.read_csv(data_dir / 'pax_data.csv'),
        'market_delta': pd.read_csv(data_dir / 'Market_Delta.csv'),
        'selma_df_map': pd.read_csv(data_dir / 'selma_df_map.csv')
    }

    # Print information about the loaded files
    for name, df in files.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
        if 'Location' in df.columns:
            locations = df['Location'].unique()
            print(f"  Locations: {', '.join(locations[:5])}{'...' if len(locations) > 5 else ''}")

    return files


# Load all data
dfs = load_data()


# Function to get product-level data for Kuwait
def get_Kuwait_product_data(dfs):
    print("\n===== Extracting Kuwait Product-Level Data =====")

    # Extract data specific to Kuwait from each dataset
    Kuwait_data = {}

    # Process df_volume data
    Kuwait_volume = dfs['df_volume'][dfs['df_volume']['Location'] == 'Kuwait'].copy()
    if not Kuwait_volume.empty:
        print(f"Found {len(Kuwait_volume)} products in df_volume for Kuwait")

        # Identify volume column
        volume_col = 'DF_Vol' if 'DF_Vol' in Kuwait_volume.columns else '$current_year Volume'

        # Split into PMI and competitor products
        pmi_volume = Kuwait_volume[Kuwait_volume['TMO'] == 'PMI'].copy()
        comp_volume = Kuwait_volume[Kuwait_volume['TMO'] != 'PMI'].copy()

        # Calculate total volumes
        total_volume = Kuwait_volume[volume_col].sum()
        pmi_volume_sum = pmi_volume[volume_col].sum() if not pmi_volume.empty else 0
        comp_volume_sum = comp_volume[volume_col].sum() if not comp_volume.empty else 0

        # Calculate market shares
        pmi_market_share = (pmi_volume_sum / total_volume) if total_volume > 0 else 0

        Kuwait_data['df_volume'] = {
            'all_products': Kuwait_volume,
            'pmi_products': pmi_volume,
            'comp_products': comp_volume,
            'volume_col': volume_col,
            'total_volume': total_volume,
            'pmi_volume': pmi_volume_sum,
            'comp_volume': comp_volume_sum,
            'pmi_share': pmi_market_share
        }

        print(f"  PMI market share in Kuwait: {pmi_market_share * 100:.2f}%")
        print(f"  Volume column identified: {volume_col}")
    else:
        print("No Kuwait data found in df_volume")

    # Process cat_a_df_vols data
    Kuwait_cat_a = dfs['cat_a_df_vols'][dfs['cat_a_df_vols']['Location'] == 'Kuwait'].copy()
    if not Kuwait_cat_a.empty:
        print(f"Found {len(Kuwait_cat_a)} products in cat_a_df_vols for Kuwait")

        # Identify volume column
        volume_cols = [col for col in Kuwait_cat_a.columns if 'volume' in col.lower() and '$current' in col.lower()]
        volume_col = volume_cols[0] if volume_cols else None

        if volume_col:
            # Split into PMI and competitor products
            pmi_cat_a = Kuwait_cat_a[Kuwait_cat_a['TMO'] == 'PMI'].copy()
            comp_cat_a = Kuwait_cat_a[Kuwait_cat_a['TMO'] != 'PMI'].copy()

            # Calculate total volumes
            total_volume = Kuwait_cat_a[volume_col].sum()
            pmi_volume_sum = pmi_cat_a[volume_col].sum() if not pmi_cat_a.empty else 0
            comp_volume_sum = comp_cat_a[volume_col].sum() if not comp_cat_a.empty else 0

            # Calculate market shares
            pmi_market_share = (pmi_volume_sum / total_volume) if total_volume > 0 else 0

            Kuwait_data['cat_a_df_vols'] = {
                'all_products': Kuwait_cat_a,
                'pmi_products': pmi_cat_a,
                'comp_products': comp_cat_a,
                'volume_col': volume_col,
                'total_volume': total_volume,
                'pmi_volume': pmi_volume_sum,
                'comp_volume': comp_volume_sum,
                'pmi_share': pmi_market_share
            }

            print(f"  PMI market share in Kuwait from cat_a_df_vols: {pmi_market_share * 100:.2f}%")
            print(f"  Volume column identified: {volume_col}")
        else:
            print("  No volume column found in cat_a_df_vols for Kuwait")
    else:
        print("No Kuwait data found in cat_a_df_vols")

    # Process market_mix data
    Kuwait_market_mix = dfs['market_mix'][dfs['market_mix']['Location'] == 'Kuwait'].copy()
    if not Kuwait_market_mix.empty:
        print(f"Found {len(Kuwait_market_mix)} products in market_mix for Kuwait")
        Kuwait_data['market_mix'] = Kuwait_market_mix
    else:
        print("No Kuwait data found in market_mix")

    # Process PARIS_Output data
    Kuwait_paris = dfs['paris_output'][dfs['paris_output']['Location'] == 'Kuwait'].copy()
    if not Kuwait_paris.empty:
        print(f"Found {len(Kuwait_paris)} entries in PARIS_Output for Kuwait")
        Kuwait_data['paris_output'] = Kuwait_paris
    else:
        print("No Kuwait data found in PARIS_Output")

    # Process passenger data
    Kuwait_pax = dfs['pax_data'][dfs['pax_data']['Market'] == 'Kuwait'].copy()
    if not Kuwait_pax.empty:
        print(f"Found {len(Kuwait_pax)} entries in pax_data for Kuwait")
        Kuwait_data['pax_data'] = Kuwait_pax
    else:
        print("No Kuwait data found in pax_data")

    return Kuwait_data


# Get product data for Kuwait
Kuwait_data = get_Kuwait_product_data(dfs)


# Merge attribute data with product data
def merge_product_attributes(Kuwait_data):
    print("\n===== Merging Product Attributes =====")

    # Choose the primary dataset based on availability
    primary_source = None
    primary_products = None
    volume_col = None

    if 'df_volume' in Kuwait_data:
        primary_source = 'df_volume'
        primary_products = Kuwait_data['df_volume']['all_products'].copy()
        volume_col = Kuwait_data['df_volume']['volume_col']
        print(f"Using df_volume as primary source with {len(primary_products)} products")
    elif 'cat_a_df_vols' in Kuwait_data:
        primary_source = 'cat_a_df_vols'
        primary_products = Kuwait_data['cat_a_df_vols']['all_products'].copy()
        volume_col = Kuwait_data['cat_a_df_vols']['volume_col']
        print(f"Using cat_a_df_vols as primary source with {len(primary_products)} products")
    else:
        print("No product data available for Kuwait")
        return None

    # Merge with attribute data from market_mix or selma_df_map
    product_data = primary_products.copy()

    if 'market_mix' in Kuwait_data and 'CR_BrandId' in product_data.columns:
        market_mix = Kuwait_data['market_mix']
        if 'CR_BrandId' in market_mix.columns:
            # Get attributes
            attr_cols = ['Flavor', 'Taste', 'Thickness', 'Length']
            available_attrs = [col for col in attr_cols if col in market_mix.columns]

            if available_attrs:
                print(f"Merging attributes from market_mix: {', '.join(available_attrs)}")
                # Merge on CR_BrandId
                product_data = pd.merge(
                    product_data,
                    market_mix[['CR_BrandId'] + available_attrs],
                    on='CR_BrandId',
                    how='left'
                )

    # Create the final merged dataset
    merged_data = {
        'products': product_data,
        'volume_col': volume_col,
        'primary_source': primary_source
    }

    # Add PMI and competitor splits
    if primary_source == 'df_volume':
        merged_data['pmi_products'] = product_data[product_data['TMO'] == 'PMI'].copy()
        merged_data['comp_products'] = product_data[product_data['TMO'] != 'PMI'].copy()
        merged_data['pmi_share'] = Kuwait_data['df_volume']['pmi_share']
    elif primary_source == 'cat_a_df_vols':
        merged_data['pmi_products'] = product_data[product_data['TMO'] == 'PMI'].copy()
        merged_data['comp_products'] = product_data[product_data['TMO'] != 'PMI'].copy()
        merged_data['pmi_share'] = Kuwait_data['cat_a_df_vols']['pmi_share']

    # Add PARIS_Output data if available
    if 'paris_output' in Kuwait_data:
        merged_data['paris_output'] = Kuwait_data['paris_output']

    return merged_data


# Merge product attributes
Kuwait_merged = merge_product_attributes(Kuwait_data)


# Function to identify top products making up 90% of the market
def identify_top_90_percent(merged_data):
    print("\n===== Identifying Products Making Up 90% of Market =====")

    if not merged_data:
        print("No merged data available for analysis")
        return None

    products = merged_data['products']
    volume_col = merged_data['volume_col']

    if not products.empty and volume_col in products.columns:
        # Calculate total volume
        total_volume = products[volume_col].sum()
        print(f"Total volume: {total_volume}")

        # Sort products by volume (descending)
        sorted_products = products.sort_values(by=volume_col, ascending=False).copy()

        # Calculate cumulative percentage
        sorted_products['cum_volume'] = sorted_products[volume_col].cumsum()
        sorted_products['cum_pct'] = sorted_products['cum_volume'] / total_volume * 100

        # Identify products making up 90%
        top_90_pct = sorted_products[sorted_products['cum_pct'] <= 90].copy()

        if not top_90_pct.empty:
            print(f"Identified {len(top_90_pct)} products making up 90% of the market")
            print(
                f"These products account for {top_90_pct[volume_col].sum() / total_volume * 100:.2f}% of the total volume")

            # Split into PMI and competitor products
            top_90_pct_pmi = top_90_pct[top_90_pct['TMO'] == 'PMI'].copy()
            top_90_pct_comp = top_90_pct[top_90_pct['TMO'] != 'PMI'].copy()

            print(
                f"PMI products in top 90%: {len(top_90_pct_pmi)} ({len(top_90_pct_pmi) / len(top_90_pct) * 100:.2f}%)")
            print(
                f"Competitor products in top 90%: {len(top_90_pct_comp)} ({len(top_90_pct_comp) / len(top_90_pct) * 100:.2f}%)")

            top_90_pct_data = {
                'all_products': top_90_pct,
                'pmi_products': top_90_pct_pmi,
                'comp_products': top_90_pct_comp,
                'volume_col': volume_col,
                'total_products': len(top_90_pct),
                'total_volume': top_90_pct[volume_col].sum(),
                'pmi_products_count': len(top_90_pct_pmi),
                'comp_products_count': len(top_90_pct_comp),
                'pmi_volume': top_90_pct_pmi[volume_col].sum() if not top_90_pct_pmi.empty else 0,
                'comp_volume': top_90_pct_comp[volume_col].sum() if not top_90_pct_comp.empty else 0
            }

            # Calculate PMI share in top 90%
            if top_90_pct_data['total_volume'] > 0:
                top_90_pct_data['pmi_share'] = top_90_pct_data['pmi_volume'] / top_90_pct_data['total_volume']
                print(f"PMI share in top 90% products: {top_90_pct_data['pmi_share'] * 100:.2f}%")

            return top_90_pct_data
        else:
            print("No products found in top 90%")
    else:
        print("No product or volume data available")

    return None


# Identify top products making up 90% of the market
Kuwait_top_90 = identify_top_90_percent(Kuwait_merged)


# Function to analyze attribute distribution of top products
def analyze_attribute_distribution(top_90_data, merged_data):
    print("\n===== Analyzing Attribute Distribution of Top Products =====")

    if not top_90_data or not merged_data:
        print("Insufficient data for attribute analysis")
        return None

    # Get the products making up 90% of the market
    top_products = top_90_data['all_products']
    volume_col = top_90_data['volume_col']

    # Define the attributes to analyze
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    available_attrs = [attr for attr in attributes if attr in top_products.columns]

    if not available_attrs:
        print("No attribute data available for analysis")
        return None

    print(f"Analyzing distributions for attributes: {', '.join(available_attrs)}")

    attribute_data = {}

    for attr in available_attrs:
        print(f"\n----- {attr} Distribution Analysis -----")

        # Calculate actual distribution (volume-based)
        actual_dist = top_products.groupby(attr)[volume_col].sum().reset_index()
        actual_dist['Volume_Percentage'] = actual_dist[volume_col] / actual_dist[volume_col].sum() * 100

        # Calculate PMI distribution
        pmi_products = top_90_data['pmi_products']
        if not pmi_products.empty:
            pmi_dist = pmi_products.groupby(attr)[volume_col].sum().reset_index()
            if not pmi_dist.empty and pmi_dist[volume_col].sum() > 0:
                pmi_dist['PMI_Volume_Percentage'] = pmi_dist[volume_col] / pmi_dist[volume_col].sum() * 100
            else:
                pmi_dist['PMI_Volume_Percentage'] = 0
        else:
            pmi_dist = pd.DataFrame(columns=[attr, volume_col, 'PMI_Volume_Percentage'])

        # Calculate ideal distribution (from PARIS_Output if available)
        if 'paris_output' in merged_data and attr in merged_data['paris_output'].columns:
            paris_data = merged_data['paris_output']

            # Get ideal distribution from PARIS_Output
            ideal_dist = paris_data.groupby(attr)['Ideal_So_Segment'].sum().reset_index()
            ideal_dist['Ideal_Percentage'] = ideal_dist['Ideal_So_Segment'] * 100
        else:
            ideal_dist = pd.DataFrame(columns=[attr, 'Ideal_Percentage'])

        # Combine actual, PMI, and ideal distributions
        combined_dist = pd.merge(actual_dist[[attr, 'Volume_Percentage']],
                                 pmi_dist[[attr, 'PMI_Volume_Percentage']],
                                 on=attr, how='outer').fillna(0)

        if not ideal_dist.empty:
            combined_dist = pd.merge(combined_dist,
                                     ideal_dist[[attr, 'Ideal_Percentage']],
                                     on=attr, how='outer').fillna(0)

            # Calculate gaps
            combined_dist['Market_vs_Ideal_Gap'] = combined_dist['Volume_Percentage'] - combined_dist[
                'Ideal_Percentage']
            combined_dist['PMI_vs_Ideal_Gap'] = combined_dist['PMI_Volume_Percentage'] - combined_dist[
                'Ideal_Percentage']
            combined_dist['PMI_vs_Market_Gap'] = combined_dist['PMI_Volume_Percentage'] - combined_dist[
                'Volume_Percentage']

        print(f"\n{attr} Distribution in Top 90% Products:")
        print(combined_dist.sort_values('Volume_Percentage', ascending=False))

        # Store the distribution data
        attribute_data[attr] = {
            'actual_dist': actual_dist,
            'pmi_dist': pmi_dist,
            'ideal_dist': ideal_dist,
            'combined_dist': combined_dist
        }

        # Find the top attribute values making up 90% of the market
        sorted_dist = actual_dist.sort_values('Volume_Percentage', ascending=False).copy()
        sorted_dist['cum_pct'] = sorted_dist['Volume_Percentage'].cumsum()
        top_attrs = sorted_dist[sorted_dist['cum_pct'] <= 90].copy()

        if not top_attrs.empty:
            print(f"\nTop {attr} values making up 90% of the market:")
            for idx, row in top_attrs.iterrows():
                print(f"  - {row[attr]}: {row['Volume_Percentage']:.2f}%")

            attribute_data[attr]['top_values'] = top_attrs

    return attribute_data


# Analyze attribute distribution
Kuwait_attr_dist = analyze_attribute_distribution(Kuwait_top_90, Kuwait_merged)


# Function to identify top products by attribute
def identify_top_products_by_attribute(top_90_data, attr_dist):
    print("\n===== Identifying Top Products by Attribute =====")

    if not top_90_data or not attr_dist:
        print("Insufficient data for product attribute analysis")
        return None

    top_products = top_90_data['all_products']
    volume_col = top_90_data['volume_col']

    top_products_by_attr = {}

    for attr, data in attr_dist.items():
        if 'top_values' in data:
            print(f"\n----- Top Products by {attr} -----")

            top_values = data['top_values'][attr].tolist()

            attr_top_products = {}

            for value in top_values:
                # Filter products for this attribute value
                segment_products = top_products[top_products[attr] == value].copy()

                if not segment_products.empty:
                    # Calculate total volume for this segment
                    segment_volume = segment_products[volume_col].sum()

                    # Get top products by volume
                    top_segment_products = segment_products.sort_values(volume_col, ascending=False).head(10)

                    print(
                        f"\n{attr}: {value} (Total Volume: {segment_volume}, {segment_volume / top_90_data['total_volume'] * 100:.2f}% of top 90% market)")
                    print(f"Top products in this segment:")

                    # Extract PMI and competitor products
                    segment_pmi = top_segment_products[top_segment_products['TMO'] == 'PMI'].copy()
                    segment_comp = top_segment_products[top_segment_products['TMO'] != 'PMI'].copy()

                    # Calculate PMI share in this segment
                    pmi_segment_share = segment_pmi[volume_col].sum() / segment_volume if segment_volume > 0 else 0

                    print(f"  PMI share in this segment: {pmi_segment_share * 100:.2f}%")

                    products_data = []

                    for idx, row in top_segment_products.iterrows():
                        # Determine product name based on available columns
                        if 'SKU' in row:
                            product_name = row['SKU']
                        elif 'Brand Family' in row and isinstance(row['Brand Family'], str):
                            product_name = f"{row['Brand Family']} (CR_BrandId: {row['CR_BrandId']})"
                        else:
                            product_name = f"CR_BrandId: {row['CR_BrandId']}"

                        # Calculate share of segment
                        share_of_segment = (row[volume_col] / segment_volume) * 100 if segment_volume > 0 else 0

                        # Print product details
                        print(
                            f"  - {product_name}: {row[volume_col]} units ({share_of_segment:.1f}% of segment) - {row['TMO']}")

                        # Add to products data
                        products_data.append({
                            'CR_BrandId': row['CR_BrandId'],
                            'Product_Name': product_name,
                            'TMO': row['TMO'],
                            'Volume': row[volume_col],
                            'Share_of_Segment': share_of_segment
                        })

                    attr_top_products[value] = {
                        'segment_volume': segment_volume,
                        'pmi_share': pmi_segment_share,
                        'top_products': products_data
                    }

            top_products_by_attr[attr] = attr_top_products

    return top_products_by_attr


# Identify top products by attribute
Kuwait_top_products = identify_top_products_by_attribute(Kuwait_top_90, Kuwait_attr_dist)


# Function to compare actual vs. ideal distribution and generate recommendations
def generate_portfolio_recommendations(top_90_data, attr_dist, top_products):
    print("\n===== Portfolio Optimization Recommendations =====")

    if not top_90_data or not attr_dist or not top_products:
        print("Insufficient data for recommendations")
        return None

    recommendations = {}

    # Overall market assessment
    pmi_share = top_90_data['pmi_share'] if 'pmi_share' in top_90_data else 0

    print(f"Current PMI market share in Kuwait: {pmi_share * 100:.2f}%")

    if pmi_share < 0.20:
        market_strategy = "Expansion"
        print("\nRecommended Overall Strategy: EXPANSION")
        print(
            "PMI has significant room for growth in the Kuwait market. Focus on introducing new SKUs in key segments and expanding distribution of existing SKUs.")
    elif pmi_share < 0.50:
        market_strategy = "Balanced Growth"
        print("\nRecommended Overall Strategy: BALANCED GROWTH")
        print(
            "PMI has a moderate presence in the Kuwait market. Focus on both expanding in key segments and optimizing the existing portfolio for efficiency.")
    else:
        market_strategy = "Optimization/Defense"
        print("\nRecommended Overall Strategy: OPTIMIZATION/DEFENSE")
        print(
            "PMI has a strong presence in the Kuwait market. Focus on defending market share, optimizing the portfolio, and rationalizing underperforming SKUs.")

    recommendations['overall_strategy'] = market_strategy

    # Attribute-specific recommendations
    attr_recommendations = {}

    for attr, data in attr_dist.items():
        if 'combined_dist' in data and 'PMI_vs_Ideal_Gap' in data['combined_dist'].columns:
            print(f"\n----- {attr} Optimization Recommendations -----")

            combined_dist = data['combined_dist']

            # Identify gaps for each attribute value
            under_represented = combined_dist[combined_dist['PMI_vs_Ideal_Gap'] < -5].sort_values('PMI_vs_Ideal_Gap')
            over_represented = combined_dist[combined_dist['PMI_vs_Ideal_Gap'] > 5].sort_values('PMI_vs_Ideal_Gap',
                                                                                                ascending=False)

            attr_gaps = {}

            # Generate recommendations for under-represented segments
            if not under_represented.empty:
                print(f"\n{attr} segments where PMI is under-represented vs. ideal:")

                for idx, row in under_represented.iterrows():
                    attr_value = row[attr]
                    pmi_pct = row['PMI_Volume_Percentage']
                    ideal_pct = row['Ideal_Percentage']
                    market_pct = row['Volume_Percentage']
                    gap = abs(row['PMI_vs_Ideal_Gap'])

                    print(f"  {attr_value}: PMI {pmi_pct:.1f}% vs Ideal {ideal_pct:.1f}% (Gap: {gap:.1f}%)")

                    # Check if this attribute value is among the top products
                    if attr in top_products and attr_value in top_products[attr]:
                        segment_data = top_products[attr][attr_value]

                        # Check if PMI has any products in this segment
                        pmi_products = [p for p in segment_data['top_products'] if p['TMO'] == 'PMI']

                        if pmi_products:
                            print(f"    PMI already has {len(pmi_products)} products in this segment")
                            print(
                                f"    Recommendation: Expand existing PMI SKUs in this segment to increase market share")

                            # List top PMI products in this segment
                            print(f"    Top existing PMI products:")
                            for p in pmi_products[:3]:
                                print(
                                    f"      - {p['Product_Name']}: {p['Volume']} units ({p['Share_of_Segment']:.1f}% of segment)")
                        else:
                            print(f"    PMI has no products in this segment")
                            print(f"    Recommendation: Introduce new PMI SKUs in this segment to address the gap")

                            # List top competitor products in this segment for reference
                            comp_products = [p for p in segment_data['top_products'] if p['TMO'] != 'PMI']
                            if comp_products:
                                print(f"    Top competitor products for reference:")
                                for p in comp_products[:3]:
                                    print(
                                        f"      - {p['Product_Name']}: {p['Volume']} units ({p['Share_of_Segment']:.1f}% of segment)")

                    # Add to gaps data
                    attr_gaps[attr_value] = {
                        'type': 'under_represented',
                        'pmi_percentage': pmi_pct,
                        'ideal_percentage': ideal_pct,
                        'market_percentage': market_pct,
                        'gap': row['PMI_vs_Ideal_Gap'],
                        'recommendation': 'Increase SKUs/volume in this segment'
                    }

            # Generate recommendations for over-represented segments
            if not over_represented.empty:
                print(f"\n{attr} segments where PMI is over-represented vs. ideal:")

                for idx, row in over_represented.iterrows():
                    attr_value = row[attr]
                    pmi_pct = row['PMI_Volume_Percentage']
                    ideal_pct = row['Ideal_Percentage']
                    market_pct = row['Volume_Percentage']
                    gap = row['PMI_vs_Ideal_Gap']

                    print(f"  {attr_value}: PMI {pmi_pct:.1f}% vs Ideal {ideal_pct:.1f}% (Gap: +{gap:.1f}%)")

                    # Check if this attribute value is among the top products
                    if attr in top_products and attr_value in top_products[attr]:
                        segment_data = top_products[attr][attr_value]

                        # Get PMI products in this segment
                        pmi_products = [p for p in segment_data['top_products'] if p['TMO'] == 'PMI']

                        if pmi_products:
                            # If there are many PMI products, consider rationalization
                            if len(pmi_products) > 3:
                                print(f"    PMI has {len(pmi_products)} products in this segment")
                                print(
                                    f"    Recommendation: Consider SKU rationalization to align with ideal distribution")

                                # List bottom PMI products in this segment
                                sorted_pmi = sorted(pmi_products, key=lambda p: p['Volume'])
                                print(f"    Bottom PMI products for potential rationalization:")
                                for p in sorted_pmi[:min(3, len(sorted_pmi))]:
                                    print(
                                        f"      - {p['Product_Name']}: {p['Volume']} units ({p['Share_of_Segment']:.1f}% of segment)")
                            else:
                                print(f"    PMI has {len(pmi_products)} products in this segment")
                                print(
                                    f"    Recommendation: Maintain current SKUs but consider rebalancing volumes across the portfolio")

                    # Add to gaps data
                    attr_gaps[attr_value] = {
                        'type': 'over_represented',
                        'pmi_percentage': pmi_pct,
                        'ideal_percentage': ideal_pct,
                        'market_percentage': market_pct,
                        'gap': row['PMI_vs_Ideal_Gap'],
                        'recommendation': 'Consider SKU rationalization or volume rebalancing'
                    }

            attr_recommendations[attr] = attr_gaps

    recommendations['attribute_recommendations'] = attr_recommendations

    # Generate summary of top product opportunities
    print("\n===== Top Product Opportunities =====")

    # Identify top 5 opportunities based on largest gaps and market size
    opportunities = []

    for attr, gaps in attr_recommendations.items():
        for attr_value, gap_data in gaps.items():
            if gap_data['type'] == 'under_represented' and gap_data['market_percentage'] > 5:
                opportunities.append({
                    'attribute': attr,
                    'value': attr_value,
                    'market_percentage': gap_data['market_percentage'],
                    'gap': abs(gap_data['gap']),
                    'score': gap_data['market_percentage'] * abs(gap_data['gap']) / 100  # Opportunity score
                })

    if opportunities:
        # Sort by opportunity score (descending)
        sorted_opps = sorted(opportunities, key=lambda x: x['score'], reverse=True)

        print("Top product opportunities ranked by impact:")
        for i, opp in enumerate(sorted_opps[:5], 1):
            print(f"{i}. {opp['attribute']}: {opp['value']}")
            print(f"   Market representation: {opp['market_percentage']:.1f}%")
            print(f"   PMI gap vs. ideal: {opp['gap']:.1f}%")
            print(f"   Opportunity score: {opp['score']:.2f}")

            # Check if this attribute value is among the top products
            if opp['attribute'] in top_products and opp['value'] in top_products[opp['attribute']]:
                segment_data = top_products[opp['attribute']][opp['value']]

                # List top competitor products in this segment for reference
                comp_products = [p for p in segment_data['top_products'] if p['TMO'] != 'PMI']
                if comp_products:
                    print(f"   Top competitor products for reference:")
                    for p in comp_products[:3]:
                        print(
                            f"     - {p['Product_Name']}: {p['Volume']} units ({p['Share_of_Segment']:.1f}% of segment)")
    else:
        print("No significant product opportunities identified.")

    # Summary of recommended actions
    print("\n===== Summary of Recommended Actions =====")

    print(f"1. Overall Portfolio Strategy for Kuwait: {market_strategy.upper()}")

    # Key action items based on strategy
    if market_strategy == "Expansion":
        print("   • Introduce new SKUs in underrepresented segments")
        print("   • Increase distribution of existing SKUs")
        print("   • Consider strategic acquisitions or partnerships to gain market share")
    elif market_strategy == "Balanced Growth":
        print("   • Selectively introduce new SKUs in high-opportunity segments")
        print("   • Optimize existing SKU performance")
        print("   • Balance portfolio across all key attribute segments")
    else:  # Optimization/Defense
        print("   • Rationalize underperforming SKUs")
        print("   • Defend high-share segments against competitor incursion")
        print("   • Focus on efficiency and profitability metrics")

    print("\n2. Critical Attribute Focus Areas:")
    for attr, gaps in attr_recommendations.items():
        under_rep = [v for k, v in gaps.items() if v['type'] == 'under_represented']
        over_rep = [v for k, v in gaps.items() if v['type'] == 'over_represented']

        if under_rep:
            print(f"   • {attr}: Increase representation in {len(under_rep)} segment(s)")
        if over_rep:
            print(f"   • {attr}: Consider rebalancing {len(over_rep)} segment(s)")

    return recommendations


# Generate portfolio recommendations
Kuwait_recommendations = generate_portfolio_recommendations(Kuwait_top_90, Kuwait_attr_dist, Kuwait_top_products)


# Function to analyze passenger mix and its influence on the ideal portfolio
def analyze_passenger_mix(Kuwait_data):
    print("\n===== Passenger Mix Analysis =====")

    if 'pax_data' not in Kuwait_data:
        print("No passenger data available for Kuwait")
        return None

    pax_data = Kuwait_data['pax_data']

    if not pax_data.empty:
        print(f"Found {len(pax_data)} passenger data entries for Kuwait")

        # Group by nationality to see passenger distribution
        if 'Nationality' in pax_data.columns and 'Pax' in pax_data.columns:
            nationality_dist = pax_data.groupby('Nationality')['Pax'].sum().reset_index()
            nationality_dist['Percentage'] = nationality_dist['Pax'] / nationality_dist['Pax'].sum() * 100

            # Sort by percentage (descending)
            nationality_dist = nationality_dist.sort_values('Percentage', ascending=False)

            print("\nPassenger Distribution by Nationality:")
            for idx, row in nationality_dist.head(10).iterrows():
                print(f"  {row['Nationality']}: {row['Percentage']:.2f}%")

            # Identify top nationalities making up 90% of passengers
            nationality_dist['cum_pct'] = nationality_dist['Percentage'].cumsum()
            top_nationalities = nationality_dist[nationality_dist['cum_pct'] <= 90].copy()

            if not top_nationalities.empty:
                print(f"\nTop nationalities making up 90% of passengers:")
                for idx, row in top_nationalities.iterrows():
                    print(f"  {row['Nationality']}: {row['Percentage']:.2f}%")

            return {
                'nationality_dist': nationality_dist,
                'top_nationalities': top_nationalities
            }
        else:
            print("Required columns not found in passenger data")
    else:
        print("No passenger data available for analysis")

    return None


# Analyze passenger mix
passenger_analysis = analyze_passenger_mix(Kuwait_data)


# Function to visualize the key findings
def create_visualizations(attr_dist, top_products, passenger_analysis):
    print("\n===== Creating Visualizations =====")

    # 1. Create attribute distribution visualizations
    if attr_dist:
        for attr, data in attr_dist.items():
            if 'combined_dist' in data:
                combined_dist = data['combined_dist']

                if 'PMI_Volume_Percentage' in combined_dist.columns and 'Ideal_Percentage' in combined_dist.columns:
                    try:
                        plt.figure(figsize=(12, 6))

                        # Sort by market volume percentage
                        sorted_dist = combined_dist.sort_values('Volume_Percentage', ascending=False)

                        # Get the attribute values
                        attr_values = sorted_dist[attr].tolist()
                        x = np.arange(len(attr_values))

                        # Plot the bars
                        bar_width = 0.25
                        plt.bar(x - bar_width, sorted_dist['Volume_Percentage'], bar_width, label='Market %',
                                color='steelblue')
                        plt.bar(x, sorted_dist['PMI_Volume_Percentage'], bar_width, label='PMI %', color='tomato')
                        plt.bar(x + bar_width, sorted_dist['Ideal_Percentage'], bar_width, label='Ideal %',
                                color='forestgreen')

                        # Add labels and title
                        plt.xlabel(attr, fontsize=12)
                        plt.ylabel('Percentage (%)', fontsize=12)
                        plt.title(f'Kuwait {attr} Distribution: Market vs. PMI vs. Ideal', fontsize=14)
                        plt.xticks(x, attr_values, rotation=45)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.legend()
                        plt.tight_layout()

                        # Save the figure
                        plt.savefig(f"Kuwait_{attr}_distribution.png")
                        print(f"Plot saved as Kuwait_{attr}_distribution.png")

                        # Create bar chart for the gaps
                        plt.figure(figsize=(12, 6))

                        # Sort by gap for clearer visualization
                        gap_sorted = combined_dist.sort_values('PMI_vs_Ideal_Gap')
                        attr_values = gap_sorted[attr].tolist()
                        gaps = gap_sorted['PMI_vs_Ideal_Gap'].tolist()

                        # Create color map (red for negative, green for positive)
                        colors = ['red' if g < 0 else 'green' for g in gaps]

                        # Plot the bars
                        plt.bar(attr_values, gaps, color=colors)

                        # Add labels and title
                        plt.xlabel(attr, fontsize=12)
                        plt.ylabel('Gap (PMI % - Ideal %)', fontsize=12)
                        plt.title(f'Kuwait {attr} Gaps: PMI vs. Ideal', fontsize=14)
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                        # Save the figure
                        plt.savefig(f"Kuwait_{attr}_gaps.png")
                        print(f"Plot saved as Kuwait_{attr}_gaps.png")
                    except Exception as e:
                        print(f"Error creating visualization for {attr}: {e}")

    # 2. Create passenger mix visualization
    if passenger_analysis and 'nationality_dist' in passenger_analysis:
        try:
            nationality_dist = passenger_analysis['nationality_dist']
            if len(nationality_dist) > 0:
                plt.figure(figsize=(12, 6))

                # Get top 10 nationalities
                top_10 = nationality_dist.head(10)

                # Plot the bars
                plt.bar(top_10['Nationality'], top_10['Percentage'], color='skyblue')

                # Add labels and title
                plt.xlabel('Nationality', fontsize=12)
                plt.ylabel('Percentage (%)', fontsize=12)
                plt.title('Kuwait Passenger Distribution by Top 10 Nationalities', fontsize=14)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save the figure
                plt.savefig("Kuwait_passenger_distribution.png")
                print("Plot saved as Kuwait_passenger_distribution.png")
        except Exception as e:
            print(f"Error creating passenger visualization: {e}")


# Create visualizations
create_visualizations(Kuwait_attr_dist, Kuwait_top_products, passenger_analysis)


# Export all findings to Excel for further analysis
def export_to_excel(Kuwait_top_90, attr_dist, top_products, passenger_analysis):
    print("\n===== Exporting Results to Excel =====")

    # Create Excel writer
    excel_path = data_dir / 'Kuwait_product_analysis.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        # 1. Export top 90% products
        if Kuwait_top_90 and 'all_products' in Kuwait_top_90:
            Kuwait_top_90['all_products'].to_excel(writer, sheet_name='Top_90pct_Products', index=False)
            print(f"Exported top 90% products ({len(Kuwait_top_90['all_products'])} rows)")

        # 2. Export PMI products in top 90%
        if Kuwait_top_90 and 'pmi_products' in Kuwait_top_90 and not Kuwait_top_90['pmi_products'].empty:
            Kuwait_top_90['pmi_products'].to_excel(writer, sheet_name='PMI_Products', index=False)
            print(f"Exported PMI products ({len(Kuwait_top_90['pmi_products'])} rows)")

        # 3. Export attribute distributions
        if attr_dist:
            for attr, data in attr_dist.items():
                if 'combined_dist' in data:
                    data['combined_dist'].to_excel(writer, sheet_name=f'{attr}_Distribution', index=False)
                    print(f"Exported {attr} distribution ({len(data['combined_dist'])} rows)")

        # 4. Export top products by attribute
        if top_products:
            for attr, attr_data in top_products.items():
                # Create a dataframe to hold all products for this attribute
                all_products_data = []

                for value, segment_data in attr_data.items():
                    for product in segment_data['top_products']:
                        product_row = product.copy()
                        product_row[attr] = value
                        all_products_data.append(product_row)

                if all_products_data:
                    attr_df = pd.DataFrame(all_products_data)
                    attr_df.to_excel(writer, sheet_name=f'{attr}_Top_Products', index=False)
                    print(f"Exported top products for {attr} ({len(attr_df)} rows)")

        # 5. Export passenger data if available
        if passenger_analysis and 'nationality_dist' in passenger_analysis:
            passenger_analysis['nationality_dist'].to_excel(writer, sheet_name='Passenger_Distribution', index=False)
            print(f"Exported passenger distribution ({len(passenger_analysis['nationality_dist'])} rows)")

        # 6. Create a summary sheet
        summary_data = []

        # Market share summary
        if Kuwait_top_90:
            summary_data.append({
                'Category': 'Market Share',
                'Metric': 'PMI Share in Top 90% Products',
                'Value': f"{Kuwait_top_90['pmi_share'] * 100:.2f}%"
            })
            summary_data.append({
                'Category': 'Market Share',
                'Metric': 'PMI Products Count',
                'Value': Kuwait_top_90['pmi_products_count']
            })
            summary_data.append({
                'Category': 'Market Share',
                'Metric': 'Competitor Products Count',
                'Value': Kuwait_top_90['comp_products_count']
            })

        # Attribute distribution summary
        if attr_dist:
            for attr, data in attr_dist.items():
                if 'combined_dist' in data:
                    under_rep = data['combined_dist'][data['combined_dist']['PMI_vs_Ideal_Gap'] < -5]
                    over_rep = data['combined_dist'][data['combined_dist']['PMI_vs_Ideal_Gap'] > 5]

                    summary_data.append({
                        'Category': f'{attr} Distribution',
                        'Metric': 'Under-represented Segments',
                        'Value': len(under_rep)
                    })
                    summary_data.append({
                        'Category': f'{attr} Distribution',
                        'Metric': 'Over-represented Segments',
                        'Value': len(over_rep)
                    })

        # Passenger mix summary
        if passenger_analysis and 'top_nationalities' in passenger_analysis:
            summary_data.append({
                'Category': 'Passenger Mix',
                'Metric': 'Top Nationalities (90% of passengers)',
                'Value': len(passenger_analysis['top_nationalities'])
            })

        # Create and export summary dataframe
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print("Exported summary sheet")

    print(f"Results exported to {excel_path}")


# Export results to Excel
export_to_excel(Kuwait_top_90, Kuwait_attr_dist, Kuwait_top_products, passenger_analysis)

print("\nAnalysis completed. All results have been exported to Excel.")