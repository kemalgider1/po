import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create output directory if it doesn't exist
os.makedirs('./output', exist_ok=True)

def create_3d_portfolio_visualization():
    """
    Create 4 3D visualizations for Kuwait and Jeju portfolios
    """
    # Key metrics - hardcoded data based on analysis
    # Format: [x, y, z, size, color, label]
    # x = flavor position (0:Regular, 1:Menthol, 2:Menthol Caps, 3:NTD, 4:NTD Caps)
    # y = thickness position (0:STD, 1:SSL, 2:SLI)
    # z = taste position (0:Full Flavor, 1:Lights, 2:Ultralights, 3:Medium)
    # size = market share (scaled)
    # color = taste color code
    # label = brand name

    # Colors for taste
    colors = {
        'Full Flavor': '#D62728',  # Red
        'Lights': '#1F77B4',      # Blue
        'Ultralights': '#17BECF', # Light Blue
        'Medium': '#FF7F0E'       # Orange
    }

    # Kuwait Ideal Market Portfolio
    kuwait_ideal = [
        # Format: [flavor, thickness, taste, size, color, label, market_share]
        [0, 0, 0, 2000, colors['Full Flavor'], 'DUNHILL', 74.1],  # STD+Regular+Full Flavor
        [0, 2, 0, 800, colors['Full Flavor'], 'DAVIDOFF', 22.1],  # SLI+Regular+Full Flavor
        [0, 1, 0, 300, colors['Full Flavor'], '', 8.2],           # SSL+Regular+Full Flavor
        [4, 0, 0, 150, colors['Full Flavor'], '', 4.1],           # STD+NTD Caps+Full Flavor
    ]

    # Kuwait Actual PMI Portfolio
    kuwait_actual = [
        # Format: [flavor, thickness, taste, size, color, label, market_share, gap]
        [0, 0, 0, 1800, colors['Full Flavor'], 'L&M', 67.2, -6.9],         # STD+Regular+Full Flavor
        [0, 0, 2, 600, colors['Ultralights'], 'PHILIP M', 12.8, 0],        # STD+Regular+Ultralights
        [0, 2, 0, 1000, colors['Full Flavor'], 'MARLBORO', 32.5, +10.4],   # SLI+Regular+Full Flavor
        [0, 1, 0, 100, colors['Full Flavor'], '', 4.6, -3.6],              # SSL+Regular+Full Flavor
        [4, 0, 0, 200, colors['Full Flavor'], '', 5.2, +1.1],              # STD+NTD Caps+Full Flavor
    ]

    # Jeju Ideal Market Portfolio
    jeju_ideal = [
        # Format: [flavor, thickness, taste, size, color, label, market_share]
        [0, 0, 0, 1200, colors['Full Flavor'], 'DUNHILL', 35.5],       # STD+Regular+Full Flavor
        [4, 1, 0, 900, colors['Full Flavor'], 'BOHEM', 24.8],          # SSL+NTD Caps+Full Flavor
        [0, 1, 0, 800, colors['Full Flavor'], 'CLOUD 9', 25.1],        # SSL+Regular+Full Flavor
        [0, 1, 1, 400, colors['Lights'], 'ESSE', 8.3],                 # SSL+Regular+Lights
        [4, 0, 0, 300, colors['Full Flavor'], 'BOHEM', 8.0],           # STD+NTD Caps+Full Flavor
        [2, 0, 0, 200, colors['Full Flavor'], '', 6.5],                # STD+Menthol Caps+Full Flavor
        [2, 1, 0, 180, colors['Full Flavor'], 'ESSE', 5.2],            # SSL+Menthol Caps+Full Flavor
        [1, 0, 0, 150, colors['Full Flavor'], '', 4.3],                # STD+Menthol+Full Flavor
        [0, 2, 0, 100, colors['Full Flavor'], '', 3.1],                # SLI+Regular+Full Flavor
        [0, 2, 1, 90, colors['Lights'], '', 2.8],                      # SLI+Regular+Lights
    ]

    # Jeju Actual PMI Portfolio
    jeju_actual = [
        # Format: [flavor, thickness, taste, size, color, label, market_share, gap]
        [0, 0, 0, 2500, colors['Full Flavor'], 'MARLBORO', 66.9, +31.4],     # STD+Regular+Full Flavor - MAJOR OVERINDEX
        [4, 0, 0, 700, colors['Full Flavor'], '', 17.7, +9.7],               # STD+NTD Caps+Full Flavor
        [2, 0, 0, 500, colors['Full Flavor'], '', 12.5, +6.0],               # STD+Menthol Caps+Full Flavor
        [0, 2, 1, 300, colors['Lights'], 'VIRGINIA S', 8.7, +5.6],           # SLI+Regular+Lights
        # Missing areas (add small transparent markers to highlight gaps)
        [4, 1, 0, 50, 'none', 'MISSING -24.8%', -24.8],                      # SSL+NTD Caps+Full Flavor - MAJOR GAP
        [0, 1, 0, 50, 'none', 'MISSING -25.1%', -25.1],                      # SSL+Regular+Full Flavor - MAJOR GAP
    ]

    # Create the 4 plots
    create_3d_plot(kuwait_ideal, "Kuwait - Ideal Market Portfolio", "./output/kuwait_market_3d.png")
    create_3d_plot(kuwait_actual, "Kuwait - Actual PMI Portfolio", "./output/kuwait_pmi_3d.png", show_gaps=True)
    create_3d_plot(jeju_ideal, "Jeju - Ideal Market Portfolio", "./output/jeju_market_3d.png")
    create_3d_plot(jeju_actual, "Jeju - Actual PMI Portfolio", "./output/jeju_pmi_3d.png", show_gaps=True)
    
    print("3D visualizations generated successfully:")
    print("1. Kuwait - Ideal Market Portfolio (kuwait_market_3d.png)")
    print("2. Kuwait - Actual PMI Portfolio (kuwait_pmi_3d.png)")
    print("3. Jeju - Ideal Market Portfolio (jeju_market_3d.png)")
    print("4. Jeju - Actual PMI Portfolio (jeju_pmi_3d.png)")


def create_3d_plot(data, title, output_path, show_gaps=False):
    """Create a 3D plot visualization"""
    
    # Define colors for taste (same as in main function)
    colors = {
        'Full Flavor': '#D62728',  # Red
        'Lights': '#1F77B4',      # Blue
        'Ultralights': '#17BECF', # Light Blue
        'Medium': '#FF7F0E'       # Orange
    }
    
    # Create figure with larger size for better visibility
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set background color to light beige for shelf appearance
    ax.set_facecolor((0.95, 0.93, 0.85))
    fig.patch.set_facecolor('white')
    
    # Create translucent planes for each thickness level with different colors
    thickness_colors = [(0.85, 0.8, 0.7, 0.3), (0.8, 0.75, 0.65, 0.3), (0.75, 0.7, 0.6, 0.3)]
    for y in range(3):  # 3 thickness levels
        # Create grid/plane for each thickness level
        xs = np.linspace(0, 4, 5)
        zs = np.linspace(0, 3, 4)
        X, Z = np.meshgrid(xs, zs)
        Y = np.full_like(X, y)
        
        # Add the plane with translucent color
        ax.plot_surface(X, Y, Z, color=thickness_colors[y], alpha=0.4, edgecolor='none')
    
    # Add grid lines for better orientation
    for x in range(5):  # 5 flavor types
        ax.plot([x, x], [0, 2], [0, 0], 'gray', alpha=0.3, linestyle='--')
        ax.plot([x, x], [0, 2], [3, 3], 'gray', alpha=0.3, linestyle='--')
    
    for z in range(4):  # 4 taste types
        ax.plot([0, 0], [0, 2], [z, z], 'gray', alpha=0.3, linestyle='--')
        ax.plot([4, 4], [0, 2], [z, z], 'gray', alpha=0.3, linestyle='--')
    
    # Plot spheres for each product
    for item in data:
        x, y, z, size, color, label, share = item[0], item[1], item[2], item[3], item[4], item[5], item[6]
        
        # Scale size for better visualization
        scaled_size = size / 100
        
        # Draw spheres for products
        if 'none' not in color:
            # Actual products
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            xs = scaled_size/30 * np.cos(u) * np.sin(v) + x
            ys = scaled_size/30 * np.sin(u) * np.sin(v) + y
            zs = scaled_size/30 * np.cos(v) + z
            ax.plot_surface(xs, ys, zs, color=color, alpha=0.7)
            
            # Add percentage labels for major products
            if scaled_size > 5:
                ax.text(x, y+0.1, z+0.1, f"{label}\n{share:.1f}%", fontsize=9, 
                       color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Show gaps for actual portfolios if requested
        if show_gaps and len(item) > 7:
            gap = item[7]
            if gap < -5:  # Only show significant gaps
                # Draw "missing" marker - red X
                ax.text(x, y+0.15, z, f"GAP: {gap:.1f}%", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
                ax.plot([x-0.1, x+0.1], [y, y], [z-0.1, z+0.1], color='red', linewidth=2)
                ax.plot([x-0.1, x+0.1], [y, y], [z+0.1, z-0.1], color='red', linewidth=2)
            elif gap > 5:  # Only show significant overindexing
                ax.text(x, y-0.15, z, f"OVER: +{gap:.1f}%", color='blue', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Set axis labels and title
    ax.set_xlabel('Flavor', fontsize=12, labelpad=10)
    ax.set_ylabel('Thickness', fontsize=12, labelpad=10)
    ax.set_zlabel('Taste', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Set tick labels for each axis
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['Regular', 'Menthol', 'Menthol\nCaps', 'NTD', 'NTD\nCaps'], fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['STD', 'SSL', 'SLI'], fontsize=10)
    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels(['Full Flavor', 'Lights', 'Ultralights', 'Medium'], fontsize=9)
    
    # Add a legend for taste colors
    for i, (taste, color) in enumerate(zip(['Full Flavor', 'Lights', 'Ultralights', 'Medium'], 
                                          [colors['Full Flavor'], colors['Lights'], 
                                           colors['Ultralights'], colors['Medium']])):
        ax.plot([0], [0], [0], 'o', markersize=10, color=color, label=taste)
    
    # Add a legend for size reference
    size_ref = [10, 25, 50]
    size_labels = ['Small (0-10%)', 'Medium (10-30%)', 'Large (30%+)']
    for i, (size, label) in enumerate(zip(size_ref, size_labels)):
        ax.plot([0], [0], [0], 'o', markersize=size/10, color='gray', alpha=0.5, label=label)
    
    # Position the legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # Set the viewing angle for better 3D visualization
    ax.view_init(elev=25, azim=45)
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    
    # Add explanation text at the bottom
    if "Kuwait" in title:
        if "Ideal" in title:
            info_text = "Kuwait's ideal market has major demand in STD+Regular+Full Flavor (74.1%)\nand SLI+Regular+Full Flavor (22.1%) segments."
        else:
            info_text = "Kuwait's PMI portfolio closely matches market demand with\nminor over/under indexing. Good market alignment."
    else:  # Jeju
        if "Ideal" in title:
            info_text = "Jeju's ideal market has balanced demand across STD+Regular (35.5%),\nSSL+Regular (25.1%) and SSL+NTD Caps (24.8%) segments."
        else:
            info_text = "Jeju's PMI portfolio shows significant misalignment with market demand.\nMissing from 50% of market (SSL segments) but overindexed in STD+Regular (+31.4%)."
    
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    create_3d_portfolio_visualization()