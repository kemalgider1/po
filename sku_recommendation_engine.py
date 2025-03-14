"""
Portfolio Optimization Recommendations Engine

This script analyzes the product distribution data and generates specific
SKU-level recommendations for portfolio optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_location_data(location, data_dir='./locations_data'):
    """
    Load all available data files for a specific location.

    Args:
        location (str): Location name (e.g., 'Kuwait', 'Jeju')
        data_dir (str): Directory containing location data files

    Returns:
        dict: Dictionary containing all loaded dataframes
    """
    data_dir = Path(data_dir)
    data = {}

    # Try to load common file types
    file_types = [
        'Flavor_Distribution',
        'Taste_Distribution',
        'Thickness_Distribution',
        'Length_Distribution',
        'PMI_Products',
        'Top_90pct_Products',
        'Flavor_Top_Products',
        'Taste_Top_Products',
        'Thickness_Top_Products',
        'Length_Top_Products',
        'Summary',
        'Passenger_Distribution'
    ]

    for file_type in file_types:
        try:
            # Try different naming patterns
            file_patterns = [
                f"{location}_product_analysis_{file_type}.csv",
                f"{location.lower()}_product_analysis_{file_type}.csv"
            ]

            for pattern in file_patterns:
                file_path = data_dir / pattern
                if file_path.exists():
                    data[file_type] = pd.read_csv(file_path)
                    print(f"Loaded {file_type} data for {location}")
                    break

        except Exception as e:
            print(f"Error loading {file_type} data for {location}: {e}")

    return data


def analyze_portfolio_gaps(location_data):
    """
    Analyze portfolio gaps across all attributes.

    Args:
        location_data (dict): Dictionary containing location data

    Returns:
        dict: Portfolio gap analysis
    """
    attributes = ['Flavor', 'Taste', 'Thickness', 'Length']
    gap_analysis = {
        'underrepresented': [],
        'overrepresented': []
    }

    for attr in attributes:
        dist_key = f"{attr}_Distribution"
        if dist_key in location_data:
            df = location_data[dist_key]

            # Check for PMI vs Ideal gaps
            if 'PMI_vs_Ideal_Gap' in df.columns:
                # Identify underrepresented segments
                under_rep = df[df['PMI_vs_Ideal_Gap'] < -5].sort_values('PMI_vs_Ideal_Gap')
                for _, row in under_rep.iterrows():
                    gap_analysis['underrepresented'].append({
                        'attribute': attr,
                        'value': row[attr],
                        'pmi_pct': row['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in df.columns else 0,
                        'ideal_pct': row['Ideal_Percentage'],
                        'gap': row['PMI_vs_Ideal_Gap'],
                        'priority': abs(row['PMI_vs_Ideal_Gap']) * (row['Ideal_Percentage'] / 100)  # Weighted priority
                    })

                # Identify overrepresented segments
                over_rep = df[df['PMI_vs_Ideal_Gap'] > 5].sort_values('PMI_vs_Ideal_Gap', ascending=False)
                for _, row in over_rep.iterrows():
                    gap_analysis['overrepresented'].append({
                        'attribute': attr,
                        'value': row[attr],
                        'pmi_pct': row['PMI_Volume_Percentage'] if 'PMI_Volume_Percentage' in df.columns else 0,
                        'ideal_pct': row['Ideal_Percentage'],
                        'gap': row['PMI_vs_Ideal_Gap'],
                        'priority': row['PMI_vs_Ideal_Gap'] * (row['PMI_Volume_Percentage'] / 100)  # Weighted priority
                    })

    # Sort by priority
    gap_analysis['underrepresented'] = sorted(gap_analysis['underrepresented'],
                                              key=lambda x: x['priority'], reverse=True)
    gap_analysis['overrepresented'] = sorted(gap_analysis['overrepresented'],
                                             key=lambda x: x['priority'], reverse=True)

    return gap_analysis


def identify_competitor_benchmark_products(location_data, gap_analysis):
    """
    Identify top competitor products in underrepresented segments.

    Args:
        location_data (dict): Dictionary containing location data
        gap_analysis (dict): Portfolio gap analysis

    Returns:
        dict: Competitor benchmark products
    """
    benchmarks = {}

    for gap in gap_analysis['underrepresented']:
        attr = gap['attribute']
        value = gap['value']

        # Look for top product data
        top_key = f"{attr}_Top_Products"
        if top_key in location_data:
            top_products = location_data[top_key]

            # Filter for the specific attribute value and competitor products
            competitors = top_products[(top_products[attr] == value) &
                                       (top_products['TMO'] != 'PMI')]

            if not competitors.empty:
                # Sort by volume/share and get top competitors
                if 'Volume' in competitors.columns:
                    competitors = competitors.sort_values('Volume', ascending=False)
                elif 'Share_of_Segment' in competitors.columns:
                    competitors = competitors.sort_values('Share_of_Segment', ascending=False)

                # Store top competitors
                benchmarks[f"{attr}_{value}"] = competitors.head(3).to_dict('records')

    return benchmarks


def generate_sku_recommendations(location, gap_analysis, benchmarks, top_products_df):
    """
    Generate specific SKU recommendations.

    Args:
        location (str): Location name
        gap_analysis (dict): Portfolio gap analysis
        benchmarks (dict): Competitor benchmark products
        top_products_df (DataFrame): Top products data

    Returns:
        dict: SKU recommendations
    """
    recommendations = {
        'location': location,
        'add': [],
        'remove': [],
        'adjust': []
    }

    # Process underrepresented segments (add recommendations)
    for gap in gap_analysis['underrepresented'][:5]:  # Focus on top 5 underrepresented segments
        attr = gap['attribute']
        value = gap['value']

        benchmark_key = f"{attr}_{value}"

        # Add new SKU recommendation
        rec = {
            'action': 'add',
            'attribute': attr,
            'value': value,
            'gap': gap['gap'],
            'priority': gap['priority'],
            'rationale': f"Significantly underrepresented segment with {abs(gap['gap']):.1f}% gap",
            'description': f"Add new SKU with {attr}: {value}"
        }

        # Add benchmark products if available
        if benchmark_key in benchmarks and benchmarks[benchmark_key]:
            top_competitor = benchmarks[benchmark_key][0]
            rec['benchmark'] = {
                'product_name': top_competitor.get('Product_Name', ''),
                'tmo': top_competitor.get('TMO', ''),
                'volume': top_competitor.get('Volume', 0),
                'share': top_competitor.get('Share_of_Segment', 0)
            }
            rec[
                'detailed_recommendation'] = f"Consider adding product similar to competitor benchmark: {top_competitor.get('Product_Name', '')}"

        recommendations['add'].append(rec)

    # Process overrepresented segments (remove recommendations)
    for gap in gap_analysis['overrepresented'][:3]:  # Focus on top 3 overrepresented segments
        attr = gap['attribute']
        value = gap['value']

        # Find PMI products in this segment
        pmi_products = top_products_df[
            (top_products_df[attr] == value) &
            (top_products_df['TMO'] == 'PMI')
            ]

        if not pmi_products.empty:
            # Sort by volume (ascending) to identify potential removals
            if 'DF_Vol' in pmi_products.columns:
                pmi_products = pmi_products.sort_values('DF_Vol')

            # Get lowest volume products
            for _, product in pmi_products.head(2).iterrows():
                product_name = product.get('SKU', product.get('CR_BrandId', ''))
                volume = product.get('DF_Vol', 0)

                rec = {
                    'action': 'remove',
                    'attribute': attr,
                    'value': value,
                    'gap': gap['gap'],
                    'priority': gap['priority'],
                    'product_id': product.get('CR_BrandId', ''),
                    'product_name': product_name,
                    'volume': volume,
                    'rationale': f"Overrepresented segment with {gap['gap']:.1f}% gap",
                    'description': f"Consider removing {product_name} ({attr}: {value})"
                }

                recommendations['remove'].append(rec)

    # Generate adjustment recommendations for partial misalignments
    for gap in gap_analysis['underrepresented'][5:8]:  # Focus on moderate underrepresented segments
        attr = gap['attribute']
        value = gap['value']

        # Find existing PMI products in complementary segments
        top_pmi_products = top_products_df[top_products_df['TMO'] == 'PMI']

        if not top_pmi_products.empty:
            # Get highest volume PMI products
            if 'DF_Vol' in top_pmi_products.columns:
                top_pmi_products = top_pmi_products.sort_values('DF_Vol', ascending=False)

            # Recommend product adjustments
            for _, product in top_pmi_products.head(2).iterrows():
                product_name = product.get('SKU', product.get('CR_BrandId', ''))
                current_value = product.get(attr, '')

                if current_value != value:  # Only suggest changing to the underrepresented value
                    rec = {
                        'action': 'adjust',
                        'attribute': attr,
                        'current_value': current_value,
                        'target_value': value,
                        'gap': gap['gap'],
                        'priority': gap['priority'] * 0.8,  # Lower priority than new additions
                        'product_id': product.get('CR_BrandId', ''),
                        'product_name': product_name,
                        'volume': product.get('DF_Vol', 0),
                        'rationale': f"Opportunity to address {abs(gap['gap']):.1f}% gap through product adjustment",
                        'description': f"Adjust {product_name} from {attr}: {current_value} to {value}"
                    }

                    recommendations['adjust'].append(rec)

    # Sort recommendations by priority
    recommendations['add'] = sorted(recommendations['add'], key=lambda x: x['priority'], reverse=True)
    recommendations['remove'] = sorted(recommendations['remove'], key=lambda x: x['priority'], reverse=True)
    recommendations['adjust'] = sorted(recommendations['adjust'], key=lambda x: x['priority'], reverse=True)

    return recommendations


def create_recommendation_visualization(recommendations):
    """
    Create visualization for SKU recommendations.

    Args:
        recommendations (dict): SKU recommendations

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # 1. Create "Add" recommendations visualization
    if recommendations['add']:
        # Extract data for visualization
        attrs = [rec['attribute'] for rec in recommendations['add']]
        values = [rec['value'] for rec in recommendations['add']]
        gaps = [abs(rec['gap']) for rec in recommendations['add']]
        priorities = [rec['priority'] * 20 for rec in recommendations['add']]  # Scale for visibility

        # Create scatter plot
        axes[0].scatter(range(len(attrs)), gaps, s=priorities, alpha=0.7, color='green')

        # Add text annotations
        for i, (attr, value, gap) in enumerate(zip(attrs, values, gaps)):
            axes[0].annotate(f"{attr}: {value}\nGap: {gap:.1f}%",
                             xy=(i, gap),
                             xytext=(5, 5),
                             textcoords="offset points",
                             fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

        # Set labels and title
        axes[0].set_title("Add Recommendations (New SKUs)", fontsize=14)
        axes[0].set_xticks(range(len(attrs)))
        axes[0].set_xticklabels([f"{a}: {v}" for a, v in zip(attrs, values)], rotation=45, ha='right')
        axes[0].set_ylabel("Gap Magnitude (%)", fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[0].text(0.5, 0.5, "No Add Recommendations", ha='center', va='center', fontsize=14)
        axes[0].set_title("Add Recommendations (New SKUs)", fontsize=14)

    # 2. Create "Remove" recommendations visualization
    if recommendations['remove']:
        # Extract data for visualization
        products = [rec['product_name'] for rec in recommendations['remove']]
        attrs = [f"{rec['attribute']}: {rec['value']}" for rec in recommendations['remove']]
        gaps = [rec['gap'] for rec in recommendations['remove']]
        volumes = [rec['volume'] / 1000 for rec in recommendations['remove']]  # Convert to thousands

        # Create bar chart
        bars = axes[1].barh(range(len(products)), volumes, color='red', alpha=0.7)

        # Add text annotations
        for i, (product, attr, gap, volume) in enumerate(zip(products, attrs, gaps, volumes)):
            axes[1].annotate(f"{product}\n{attr}\nGap: +{gap:.1f}%",
                             xy=(volume / 2, i),
                             ha='center', va='center',
                             color='white', fontweight='bold',
                             fontsize=9)

        # Set labels and title
        axes[1].set_title("Remove Recommendations (Overrepresented SKUs)", fontsize=14)
        axes[1].set_yticks(range(len(products)))
        axes[1].set_yticklabels(products)
        axes[1].set_xlabel("Volume (thousands)", fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No Remove Recommendations", ha='center', va='center', fontsize=14)
        axes[1].set_title("Remove Recommendations (Overrepresented SKUs)", fontsize=14)

    # 3. Create "Adjust" recommendations visualization
    if recommendations['adjust']:
        # Extract data for visualization
        products = [rec['product_name'] for rec in recommendations['adjust']]
        adjustments = [f"{rec['attribute']}: {rec['current_value']} → {rec['target_value']}" for rec in
                       recommendations['adjust']]
        gaps = [abs(rec['gap']) for rec in recommendations['adjust']]

        # Create horizontal bar chart
        y_pos = range(len(products))
        bars = axes[2].barh(y_pos, gaps, color='orange', alpha=0.7)

        # Add text annotations
        for i, (product, adjustment, gap) in enumerate(zip(products, adjustments, gaps)):
            axes[2].annotate(f"{product}\n{adjustment}",
                             xy=(gap / 2, i),
                             ha='center', va='center',
                             color='black', fontweight='bold',
                             fontsize=9)

        # Set labels and title
        axes[2].set_title("Adjust Recommendations (Existing SKUs)", fontsize=14)
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(products)
        axes[2].set_xlabel("Gap Magnitude (%)", fontsize=12)
        axes[2].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[2].text(0.5, 0.5, "No Adjust Recommendations", ha='center', va='center', fontsize=14)
        axes[2].set_title("Adjust Recommendations (Existing SKUs)", fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f"Portfolio Optimization Recommendations for {recommendations['location']}", fontsize=16, y=0.98)

    return fig


def generate_implementation_plan(recommendations):
    """
    Generate phased implementation plan for recommendations.

    Args:
        recommendations (dict): SKU recommendations

    Returns:
        dict: Implementation plan with short, medium, and long-term actions
    """
    implementation_plan = {
        'location': recommendations['location'],
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }

    # Prioritize and assign recommendations to different phases

    # Short-term: Quick adjustments with immediate impact
    # - High priority removals of overrepresented SKUs
    # - Minor adjustments to existing products
    for rec in recommendations['remove'][:2]:  # Top 2 removals
        implementation_plan['short_term'].append({
            'action': 'remove',
            'description': rec['description'],
            'product_name': rec['product_name'],
            'impact': f"Reduce overrepresentation of {rec['attribute']}: {rec['value']}",
            'priority': 'High',
            'timeline': '1-3 months'
        })

    for rec in recommendations['adjust'][:2]:  # Top 2 adjustments
        implementation_plan['short_term'].append({
            'action': 'adjust',
            'description': rec['description'],
            'product_name': rec['product_name'],
            'impact': f"Address gap of {abs(rec['gap']):.1f}% for {rec['attribute']}: {rec['target_value']}",
            'priority': 'Medium',
            'timeline': '2-4 months'
        })

    # Medium-term: New product introductions based on existing capabilities
    for rec in recommendations['add'][:3]:  # Top 3 additions
        implementation_plan['medium_term'].append({
            'action': 'add',
            'description': rec['description'],
            'attribute': f"{rec['attribute']}: {rec['value']}",
            'impact': f"Address gap of {abs(rec['gap']):.1f}%",
            'priority': 'High',
            'timeline': '4-8 months',
            'benchmark': rec.get('benchmark', {}).get('product_name', 'N/A')
        })

    for rec in recommendations['adjust'][2:]:  # Remaining adjustments
        implementation_plan['medium_term'].append({
            'action': 'adjust',
            'description': rec['description'],
            'product_name': rec['product_name'],
            'impact': f"Address gap of {abs(rec['gap']):.1f}% for {rec['attribute']}: {rec['target_value']}",
            'priority': 'Medium',
            'timeline': '4-6 months'
        })

    # Long-term: Strategic portfolio alignment
    for rec in recommendations['add'][3:]:  # Remaining additions
        implementation_plan['long_term'].append({
            'action': 'add',
            'description': rec['description'],
            'attribute': f"{rec['attribute']}: {rec['value']}",
            'impact': f"Address gap of {abs(rec['gap']):.1f}%",
            'priority': 'Medium',
            'timeline': '8-12 months',
            'benchmark': rec.get('benchmark', {}).get('product_name', 'N/A')
        })

    # Add strategic initiative for long-term portfolio alignment
    implementation_plan['long_term'].append({
        'action': 'strategic',
        'description': f"Conduct comprehensive portfolio review for {recommendations['location']}",
        'impact': "Holistic alignment of entire portfolio with passenger preferences",
        'priority': 'High',
        'timeline': '12-18 months'
    })

    return implementation_plan


def create_implementation_plan_visualization(implementation_plan):
    """
    Create visualization for implementation plan.

    Args:
        implementation_plan (dict): Implementation plan

    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Count actions in each phase
    short_term_count = len(implementation_plan['short_term'])
    medium_term_count = len(implementation_plan['medium_term'])
    long_term_count = len(implementation_plan['long_term'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up timeline positions
    phases = ['Short-term\n(1-4 months)', 'Medium-term\n(4-8 months)', 'Long-term\n(8+ months)']
    phase_positions = [1, 2, 3]

    # Count action types in each phase
    action_types = ['add', 'remove', 'adjust', 'strategic']
    action_colors = ['green', 'red', 'orange', 'blue']

    phase_data = []
    for phase, actions in zip(phases, [implementation_plan['short_term'],
                                       implementation_plan['medium_term'],
                                       implementation_plan['long_term']]):
        type_counts = {action_type: 0 for action_type in action_types}
        for action in actions:
            if action['action'] in type_counts:
                type_counts[action['action']] += 1

        phase_data.append(type_counts)

    # Create stacked bar chart
    bottom = np.zeros(3)
    for action_type, color in zip(action_types, action_colors):
        values = [phase[action_type] for phase in phase_data]
        ax.bar(phase_positions, values, bottom=bottom, label=action_type.capitalize(), color=color, alpha=0.7)
        bottom += values

    # Add text annotations for each action
    current_heights = np.zeros(3)
    for phase_idx, (phase, actions) in enumerate(zip(phases, [implementation_plan['short_term'],
                                                              implementation_plan['medium_term'],
                                                              implementation_plan['long_term']])):
        for action in actions:
            action_type = action['action']
            action_idx = action_types.index(action_type)

            # Get position for annotation
            y_pos = current_heights[phase_idx] + 0.5
            current_heights[phase_idx] += 1

            # Add annotation
            if 'product_name' in action:
                label = f"{action['product_name']}"
            elif 'description' in action and len(action['description']) < 30:
                label = action['description']
            else:
                label = f"{action_type.capitalize()}"

            ax.annotate(label,
                        xy=(phase_positions[phase_idx], y_pos),
                        xytext=(10, 0),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=action_colors[action_idx], alpha=0.7))

    # Set labels and title
    ax.set_title(f"Implementation Plan for {implementation_plan['location']} Portfolio Optimization", fontsize=16)
    ax.set_xticks(phase_positions)
    ax.set_xticklabels(phases, fontsize=12)
    ax.set_ylabel("Number of Actions", fontsize=12)
    ax.legend(title="Action Type")

    # Add text with expected outcomes
    expected_outcomes = {
        'Short-term': "Quick wins to reduce overrepresentation",
        'Medium-term': "New SKUs to address key gaps",
        'Long-term': "Strategic alignment of full portfolio"
    }

    outcome_text = "\nExpected Outcomes:\n"
    for phase, outcome in expected_outcomes.items():
        outcome_text += f"• {phase}: {outcome}\n"

    plt.figtext(0.5, 0.01, outcome_text, ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    return fig


def run_portfolio_optimization(location, data_dir='./locations_data', output_dir='./optimization_results'):
    """
    Run the full portfolio optimization process for a specific location.

    Args:
        location (str): Location name (e.g., 'Kuwait', 'Jeju')
        data_dir (str): Directory containing location data files
        output_dir (str): Directory to save output files and visualizations

    Returns:
        dict: Complete optimization results
    """
    print(f"Running portfolio optimization analysis for {location}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load location data
    location_data = load_location_data(location, data_dir)

    # 2. Analyze portfolio gaps
    gap_analysis = analyze_portfolio_gaps(location_data)

    # 3. Identify competitor benchmark products
    benchmarks = identify_competitor_benchmark_products(location_data, gap_analysis)

    # 4. Generate SKU recommendations
    top_products_df = None
    if 'Top_90pct_Products' in location_data:
        top_products_df = location_data['Top_90pct_Products']
    elif 'PMI_Products' in location_data:
        top_products_df = location_data['PMI_Products']

    if top_products_df is not None:
        recommendations = generate_sku_recommendations(location, gap_analysis, benchmarks, top_products_df)

        # 5. Create recommendation visualization
        rec_vis = create_recommendation_visualization(recommendations)
        if rec_vis:
            rec_vis.savefig(os.path.join(output_dir, f"{location}_recommendations.png"), dpi=300, bbox_inches='tight')

        # 6. Generate implementation plan
        implementation_plan = generate_implementation_plan(recommendations)

        # 7. Create implementation plan visualization
        plan_vis = create_implementation_plan_visualization(implementation_plan)
        if plan_vis:
            plan_vis.savefig(os.path.join(output_dir, f"{location}_implementation_plan.png"), dpi=300,
                             bbox_inches='tight')

        # 8. Save results to CSV files
        add_recs_df = pd.DataFrame(recommendations['add'])
        remove_recs_df = pd.DataFrame(recommendations['remove'])
        adjust_recs_df = pd.DataFrame(recommendations['adjust'])

        add_recs_df.to_csv(os.path.join(output_dir, f"{location}_add_recommendations.csv"), index=False)
        remove_recs_df.to_csv(os.path.join(output_dir, f"{location}_remove_recommendations.csv"), index=False)
        adjust_recs_df.to_csv(os.path.join(output_dir, f"{location}_adjust_recommendations.csv"), index=False)

        # Combine results
        results = {
            'location': location,
            'gap_analysis': gap_analysis,
            'benchmarks': benchmarks,
            'recommendations': recommendations,
            'implementation_plan': implementation_plan
        }

        print(f"Portfolio optimization analysis for {location} completed!")
        return results
    else:
        print(f"Error: No product data found for {location}. Cannot generate recommendations.")
        return None


def main():
    """Main function to run portfolio optimization for multiple locations"""
    locations = ['Jeju', 'Kuwait']
    data_dir = './locations_data'
    output_dir = './optimization_results'

    results = {}

    for location in locations:
        result = run_portfolio_optimization(location, data_dir, output_dir)
        if result:
            results[location] = result

    print("Portfolio optimization completed for all locations!")


if __name__ == "__main__":
    main()
