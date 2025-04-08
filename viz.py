import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe


def create_portfolio_bubble_chart(df, market, category, output_filename):
    """
    Creates an enhanced bubble chart visualization for portfolio optimization.

    Parameters:
    df (DataFrame): The complete dataset
    market (str): Market name (e.g., 'Jeju' or 'Kuwait')
    category (str): Category type (e.g., 'Actual' or 'Ideal')
    output_filename (str): Filename for saving the visualization
    """
    # Set up styling for dark theme
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.facecolor'] = '#1a1a1a'  # Dark gray background
    plt.rcParams['figure.facecolor'] = '#000000'  # Black background
    plt.rcParams['axes.edgecolor'] = '#333333'  # Slightly lighter gray for edges
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#333333'  # Dark grid lines
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['text.color'] = '#ffffff'  # White text
    plt.rcParams['axes.labelcolor'] = '#ffffff'  # White axis labels
    plt.rcParams['xtick.color'] = '#ffffff'  # White tick labels
    plt.rcParams['ytick.color'] = '#ffffff'  # White tick labels

    # Filter data for specified market and category
    filter_cat = f"{category}_{market}"
    df_filtered = df[(df['Location'] == market) & (df['Category'].str.contains(filter_cat, na=False))].copy()

    # Ensure key columns are numeric
    for col in ['Delta_SoS', 'Share', 'DF_Vol']:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

    # Drop rows with missing key values
    df_filtered = df_filtered.dropna(
        subset=['Delta_SoS', 'Share', 'DF_Vol', 'Flavor', 'Taste', 'Thickness', 'Alignment'])

    # Ensure SKU column exists, if not create a placeholder
    if 'SKU' not in df_filtered.columns:
        df_filtered['SKU'] = df_filtered.index.astype(str)

    # Map unique flavors to discrete positions on the y-axis
    unique_flavors = sorted(df_filtered['Flavor'].unique())
    flavor_to_y = {flavor: i for i, flavor in enumerate(unique_flavors)}

    # Calculate vertical offset within each flavor category based on DF_Vol
    result_list = []
    for flavor, group in df_filtered.groupby('Flavor'):
        min_vol = group['DF_Vol'].min()
        max_vol = group['DF_Vol'].max()

        # Create a copy of the group data
        group_copy = group.copy()

        # Calculate vertical offset
        if min_vol == max_vol:
            group_copy['vert_offset'] = 0.5  # center if only one product
        else:
            # Normalize DF_Vol within the flavor group
            # Modified spacing to create more vertical separation
            group_copy['vert_offset'] = (group_copy['DF_Vol'] - min_vol) / (max_vol - min_vol) * 0.7 + 0.15

        result_list.append(group_copy)

    # Combine the results
    df_filtered = pd.concat(result_list)

    # Calculate final y-coordinate: base flavor position + vertical offset
    df_filtered['y_coord'] = df_filtered['Flavor'].map(flavor_to_y) + df_filtered['vert_offset']

    # Enhanced color mapping for Taste with higher contrast - brightened for dark theme
    taste_color_map = {
        'Full Flavor': '#8a60fd',  # Brightened purple
        'Lights': '#5a9eff',  # Brightened blue
        'Ultralights': '#40e0ff',  # Brightened cyan
    }

    # Define transparency mapping for Thickness (more distinct values)
    thickness_alpha_map = {
        'STD': 1.0,
        'SLI': 0.8,
        'SSL': 0.6,
    }

    # Define border color mapping for Alignment with higher contrast
    align_color_map = {
        'Over-Represented': '#ff6464',  # Brightened red
        'Under-Represented': '#42f57e',  # Brightened green
        'Well Aligned': '#ffd340'  # Brightened yellow
    }

    # Adjust scale factor for bubble size to reduce overlap
    size_scale = 2400

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    # Create gradient background
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    gradient_cmap = LinearSegmentedColormap.from_list('', ['#1a1a1a', '#000000'])
    ax.imshow(gradient, cmap=gradient_cmap, aspect='auto', extent=[-0.30, 0.40, -0.5, len(unique_flavors) - 0.5],
              alpha=0.7)

    # Set axis labels and title with enhanced styling for dark theme
    title_font = {'fontsize': 16, 'fontweight': 'bold', 'fontfamily': 'sans-serif', 'color': '#ffffff'}
    ax.set_xlabel('Delta_SoS (Share of Segment Difference)',
                  fontdict={'fontsize': 12, 'fontweight': 'medium', 'color': '#ffffff'})
    ax.set_ylabel('Flavor', fontdict={'fontsize': 12, 'fontweight': 'medium', 'color': '#ffffff'})

    # Add subtitle that explains the comparison
    subtitle = "Comparison of Actual vs. Ideal Performance" if category == "Actual" else "Ideal Performance Targets"
    title = f"{market} {category}: Product Performance"
    ax.set_title(f"{title}\n{subtitle}", fontdict=title_font, pad=20)

    # Set x-axis limits and fine interval ticks
    ax.set_xlim(-0.30, 0.40)
    ax.set_xticks(np.arange(-0.30, 0.41, 0.05))
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#333333')

    # Set y-axis ticks to flavor categories
    ax.set_yticks(ticks=[y + 0.45 for y in flavor_to_y.values()])
    ax.set_yticklabels(labels=flavor_to_y.keys(), fontdict={'fontsize': 11, 'color': '#ffffff'})
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#333333')

    # Dictionaries to track legend handles
    taste_handles = {}
    thickness_handles = {}
    align_handles = {}

    # Sort data by Share to plot smaller bubbles last (on top)
    df_filtered = df_filtered.sort_values('Share', ascending=False)

    # Plot each product as a bubble with enhanced styling
    for idx, row in df_filtered.iterrows():
        x = row['Delta_SoS']
        y = row['y_coord']
        size = row['Share'] * size_scale

        # All markers are circles
        marker = 'o'

        # Color based on Taste
        facecolor = taste_color_map.get(row['Taste'], '#888888')

        # Transparency based on Thickness
        alpha_val = thickness_alpha_map.get(row['Thickness'], 0.8)

        # Border color based on Alignment
        edgecolor = align_color_map.get(row['Alignment'], '#ffffff')  # White default for dark theme

        # Add subtle glow effect for dark theme
        shadow = ax.scatter(x, y, s=size * 1.05, marker=marker, color=facecolor,
                            alpha=0.3, zorder=5)

        # Plot the bubble with enhanced styling and thinner border lines (reduced linewidth)
        bubble = ax.scatter(x, y, s=size, marker=marker, color=facecolor, edgecolor=edgecolor,
                            linewidth=1.25, alpha=alpha_val, zorder=10)  # Reduced from 2.5 to 1.25

        # Build legend handles with enhanced styling
        if row['Taste'] not in taste_handles:
            taste_handles[row['Taste']] = mpatches.Patch(color=facecolor,
                                                         label=f'Taste: {row["Taste"]}',
                                                         alpha=0.9)

        if row['Thickness'] not in thickness_handles:
            base_color = np.array([0.7, 0.7, 0.7, alpha_val])  # Lighter gray for dark theme
            thickness_handles[row['Thickness']] = mpatches.Patch(color=base_color,
                                                                 label=f'Thickness: {row["Thickness"]}')

        if row['Alignment'] not in align_handles:
            # Reduced line width for alignment indicators in legend
            align_handles[row['Alignment']] = mlines.Line2D(
                [], [], color=edgecolor, marker='o',
                markerfacecolor='none', linestyle='None',
                markersize=10, markeredgewidth=1.25,  # Reduced from 2.5 to 1.25
                label=f'Alignment: {row["Alignment"]}')

    # Create a styled frame around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')  # Lighter gray for spines
        spine.set_linewidth(1.5)

    # Add gradient background to the legend area
    legend_bg = mpatches.Rectangle((1.02, 0), 0.3, 1, transform=ax.transAxes,
                                   facecolor='#1a1a1a', edgecolor='#444444',
                                   alpha=0.8, zorder=0, linewidth=1)
    ax.add_patch(legend_bg)

    # Ensure all possible taste values are represented in the legend, even if not present in the data
    all_tastes = ['Full Flavor', 'Lights', 'Ultralights']
    all_thicknesses = ['STD', 'SLI', 'SSL']
    all_alignments = ['Under-Represented', 'Well Aligned', 'Over-Represented']  # Standardized order

    # Add any missing taste handles
    for taste in all_tastes:
        if taste not in taste_handles:
            color = taste_color_map.get(taste, '#888888')
            taste_handles[taste] = mpatches.Patch(color=color, alpha=0.9, label=f'Taste: {taste}')

    # Create a more intuitive thickness visualization using a gradient
    thickness_gradient = []
    for i, thickness in enumerate(all_thicknesses):
        alpha = thickness_alpha_map.get(thickness, 0.8)
        thickness_gradient.append(mpatches.Patch(
            color=np.array([0.7, 0.7, 0.7, alpha]),  # Lighter gray for dark theme
            label=f'Thickness: {thickness}'
        ))

    # Ensure all alignment types are represented in standardized order
    for alignment in all_alignments:
        if alignment not in align_handles:
            color = align_color_map.get(alignment, '#ffffff')
            align_handles[alignment] = mlines.Line2D(
                [], [], color=color, marker='o',
                markerfacecolor='none', linestyle='None',
                markersize=10, markeredgewidth=1.25,  # Reduced from 2.5 to 1.25
                label=f'Alignment: {alignment}'
            )

    # Group legend items by category with spacing between groups
    taste_section = list(taste_handles.values())
    thickness_section = thickness_gradient

    # Use standardized order for alignment section
    alignment_section = [align_handles.get(align, None) for align in all_alignments]
    alignment_section = [handle for handle in alignment_section if handle is not None]

    # Create section headers/spacers for better legend organization
    blank_spacer = mpatches.Patch(color='none', label=" ")  # Blank spacer

    # Create bubble size representation
    bubble_size = mpatches.Patch(color='none', label="Bubble Size = Share of Market")

    # Combine legend items with section headers for spacing
    legend_handles = taste_section + [blank_spacer] + thickness_section + [blank_spacer] + alignment_section + [
        blank_spacer] + [bubble_size]

    # Create legend with improved spacing
    legend = ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True, title='Legend',
                       title_fontsize=12, fontsize=10)
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor('#444444')
    legend.get_frame().set_facecolor('#1a1a1a')

    # Set legend text color to white
    for text in legend.get_texts():
        text.set_color('#ffffff')

    # Set legend title color to white
    legend.get_title().set_color('#ffffff')

    # Add explanation text for Delta_SoS below the legend instead of at the bottom
    explanation_text = (
        "Delta_SoS = Difference between actual and ideal share of segment\n"
        "Positive values suggest under-representation, negative values suggest over-representation"
    )

    # Add the explanation in a textbox below the legend
    props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor='#444444', alpha=0.9)
    legend_explanation = fig.text(0.105, 0.5, explanation_text,
                                  transform=ax.transAxes, fontsize=8,
                                  verticalalignment='center', horizontalalignment='center',
                                  color='#ffffff', bbox=props)

    # Add the color/transparency/border subtitle below the chart
    fig.text(0.5, 0.01, "(Color = Taste, Transparency = Thickness, Border = Alignment)",
             ha='center', va='bottom', fontsize=10, fontstyle='italic', color='#aaaaaa')

    # Add watermark
    fig.text(0.95, 0.02, 'Portfolio Analysis', fontsize=8, color='#444444',
             ha='right', va='bottom', alpha=0.7)

    # Optimized label placement approach to avoid overlaps
    # For SKU labels we'll use a smarter positioning algorithm

    # Add SKU labels to products with dynamic placement to avoid overlaps
    # Sort by importance (share size) to prioritize labeling important products
    products_to_label = df_filtered.nlargest(5, 'Share').copy()

    # Create an empty list to track occupied positions
    occupied_positions = []

    # Function to find nearest grid line for label positioning
    def snap_to_grid(value, grid_spacing=0.05):
        return round(value / grid_spacing) * grid_spacing

    # Function to check if a new position would overlap with existing ones
    def position_overlaps(pos, occupied, threshold=0.08):
        for occ in occupied:
            dist = np.sqrt((pos[0] - occ[0])**2 + (pos[1] - occ[1])**2)
            if dist < threshold:
                return True
        return False

    # Calculate base offsets for annotations - aligned with grid
    base_offsets = [
        (0.05, 0.05),   # slightly right and up
        (-0.05, 0.05),  # slightly left and up
        (0.05, -0.05),  # slightly right and down
        (-0.05, -0.05), # slightly left and down
        (0.10, 0),      # directly right
        (-0.10, 0),     # directly left
        (0, 0.10),      # directly up
        (0, -0.10)      # directly down
    ]

    # Function to find best position for a label aligned to grid
    def find_best_position(base_x, base_y, occupied_positions):
        # Try all offsets and find the first non-overlapping one
        for x_offset, y_offset in base_offsets:
            # Snap position to nearest grid line
            new_x = snap_to_grid(base_x + x_offset)
            new_y = snap_to_grid(base_y + y_offset)
            new_pos = (new_x, new_y)

            if not position_overlaps(new_pos, occupied_positions):
                return new_pos

        # If all positions overlap, try with larger offsets
        larger_offsets = [(x*1.5, y*1.5) for x, y in base_offsets]
        for x_offset, y_offset in larger_offsets:
            # Snap position to nearest grid line
            new_x = snap_to_grid(base_x + x_offset)
            new_y = snap_to_grid(base_y + y_offset)
            new_pos = (new_x, new_y)

            if not position_overlaps(new_pos, occupied_positions):
                return new_pos

        # If still all overlap, return a default position aligned to grid
        return (snap_to_grid(base_x + 0.15), snap_to_grid(base_y + 0.15))

    # Add annotations for each product with improved positioning
    for _, product in products_to_label.iterrows():
        base_x, base_y = product['Delta_SoS'], product['y_coord']

        # Find the best position for this label
        text_x, text_y = find_best_position(base_x, base_y, occupied_positions)

        # Add this position to occupied positions
        occupied_positions.append((text_x, text_y))

        # Create label with a white outline for better readability
        # Smaller font size and thinner border as requested
        label = ax.annotate(f"{product['SKU']}",  # Removed "SKU:" prefix as requested
                    xy=(base_x, base_y),
                    xytext=(text_x, text_y),
                    arrowprops=dict(arrowstyle='->', color='#aaaaaa', alpha=0.7, linewidth=0.75),  # Thinner arrow
                    color='#ffffff', fontsize=8,  # Smaller font size
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a1a", ec="#444444", alpha=0.9,
                             linewidth=1.25),  # Thinner border
                    zorder=20)

        # Add path effects for better text visibility against dark background
        label.set_path_effects([
            pe.withStroke(linewidth=2, foreground='#000000')
        ])

    # Add comparison text for actual charts
    if category == 'Actual':
        comparison_text = (
            f"This chart shows current product performance in {market}.\n"
            "Compare with the 'Ideal' chart to identify optimization opportunities."
        )

        # Add the explanation text aligned to axis
        props = dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor='#444444', alpha=0.9)
        ax.text(0.02, -0.08, comparison_text, transform=ax.transAxes, fontsize=9, color='#ffffff',
                verticalalignment='top', horizontalalignment='left', bbox=props)

    # Save with higher DPI and optimized layout
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='#000000')
    plt.close()

    print(f"Enhanced dark theme bubble chart for {market} {category} saved as {output_filename}")


def main():
    try:
        # Load data
        df = pd.read_csv('ALL.csv')

        # Generate the four enhanced dark theme visualizations
        create_portfolio_bubble_chart(df, 'Kuwait', 'Ideal', 'kuwait_ideal_dark_visualization.png')
        create_portfolio_bubble_chart(df, 'Kuwait', 'Actual', 'kuwait_actual_dark_visualization.png')
        create_portfolio_bubble_chart(df, 'Jeju', 'Ideal', 'jeju_ideal_dark_visualization.png')
        create_portfolio_bubble_chart(df, 'Jeju', 'Actual', 'jeju_actual_dark_visualization.png')

        print("All enhanced dark theme visualizations completed successfully.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
