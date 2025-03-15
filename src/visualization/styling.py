"""
Styling configuration for hydraulic visualizations.

This module provides consistent styling and color schemes for all
visualization components, ensuring a professional and cohesive appearance.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define standard color palettes
WATER_COLORS = {
    'surface': '#4287f5',  # Medium blue for water surface
    'body': '#6badf7',     # Lighter blue for water body
    'deep': '#1a56b7',     # Darker blue for deep water
    'shallow': '#94c6ff',  # Very light blue for shallow water
    'bed': '#8B7355',      # Brown for channel bed
    'dam': '#4F4F4F',      # Dark gray for dam body
}

REGIME_COLORS = {
    'subcritical': '#3498db',  # Blue for subcritical flow
    'critical': '#27ae60',     # Green for critical flow
    'supercritical': '#e74c3c', # Red for supercritical flow
    'jump': '#9b59b6',        # Purple for hydraulic jump
    'dry': '#f1c40f',         # Yellow for dry bed
}

PARAMETER_COLORS = {
    'velocity': 'viridis',    # Colormap for velocity
    'froude': 'coolwarm',     # Colormap for Froude number
    'depth': 'Blues',         # Colormap for water depth
    'energy': 'plasma',       # Colormap for energy
    'shear': 'YlOrRd',        # Colormap for shear stress
}

# Custom colormap for water depth
WATER_DEPTH_CMAP = LinearSegmentedColormap.from_list(
    'WaterDepth', 
    [(0, WATER_COLORS['shallow']), 
     (0.5, WATER_COLORS['surface']), 
     (1, WATER_COLORS['deep'])],
    N=256
)

# Custom colormap for flow regimes
FLOW_REGIME_CMAP = LinearSegmentedColormap.from_list(
    'FlowRegime', 
    [(0, REGIME_COLORS['subcritical']), 
     (0.5, REGIME_COLORS['critical']), 
     (1, REGIME_COLORS['supercritical'])],
    N=256
)

# Theme configurations
THEMES = {
    'default': {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'text.color': 'black',
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    },
    'dark': {
        'figure.facecolor': '#1f1f1f',
        'axes.facecolor': '#2a2a2a',
        'axes.edgecolor': '#cccccc',
        'axes.labelcolor': '#dddddd',
        'axes.titlecolor': 'white',
        'xtick.color': '#cccccc',
        'ytick.color': '#cccccc',
        'grid.color': '#555555',
        'grid.alpha': 0.5,
        'text.color': '#dddddd',
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    },
    'paper': {
        'figure.facecolor': 'white',
        'axes.facecolor': '#fcfcfc',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'gray',
        'grid.alpha': 0.2,
        'text.color': 'black',
        'font.family': 'serif',
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    },
    'presentation': {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'text.color': 'black',
        'font.family': 'sans-serif',
        'font.weight': 'bold',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
    }
}

def apply_theme(theme='default'):
    """
    Apply a predefined theme to all matplotlib plots.
    
    Parameters:
        theme (str): Name of the theme to apply
    """
    if theme not in THEMES:
        print(f"Unknown theme: {theme}. Using default.")
        theme = 'default'
    
    # Apply theme settings
    for key, value in THEMES[theme].items():
        plt.rcParams[key] = value
    
    # Set other common parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Return color settings specific to the theme
    if theme == 'dark':
        return {
            'water_colors': {
                'surface': '#5d9cf5',  # Lighter blue for visibility
                'body': '#4287f5',
                'deep': '#1e59c5',
                'shallow': '#94c6ff',
                'bed': '#a58665',      # Lighter brown for visibility
                'dam': '#707070',      # Lighter gray for visibility
            },
            'regime_colors': {
                'subcritical': '#3498db',
                'critical': '#27ae60',
                'supercritical': '#e74c3c',
                'jump': '#9b59b6',
                'dry': '#f1c40f',
            }
        }
    else:
        return {
            'water_colors': WATER_COLORS,
            'regime_colors': REGIME_COLORS
        }


def style_axis_for_hydraulic_profile(ax, grid=True, show_legend=True):
    """
    Apply consistent styling to a hydraulic profile plot.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to style
        grid (bool): Whether to show the grid
        show_legend (bool): Whether to show the legend
    """
    # Set grid style
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    
    # Style the legend if requested
    if show_legend:
        legend = ax.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
        
        # Adjust legend text sizes
        for text in legend.get_texts():
            text.set_fontsize(10)
    
    # Add a subtle box around the axis
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_alpha(0.2)
    ax.spines['right'].set_alpha(0.2)


def add_colorbar(fig, ax, mappable, label, orientation='vertical', pad=0.05):
    """
    Add a nicely styled colorbar to a plot.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure
        ax (matplotlib.axes.Axes): The axes with the mappable
        mappable: The mappable to which the colorbar is mapped
        label (str): Label for the colorbar
        orientation (str): 'vertical' or 'horizontal'
        pad (float): Padding between plot and colorbar
        
    Returns:
        matplotlib.colorbar.Colorbar: The colorbar
    """
    # Create colorbar
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=pad)
    
    # Set label
    cbar.set_label(label, fontsize=11)
    
    # Style ticks
    cbar.ax.tick_params(labelsize=9)
    
    return cbar


def create_water_surface_segments(x, wse, bed, cmap_name='Blues', norm=None):
    """
    Create a multi-colored water surface line based on specified parameters.
    
    Parameters:
        x (array): x-coordinates
        wse (array): water surface elevations
        bed (array): bed elevations 
        cmap_name (str): colormap name
        norm: normalization for coloring
        
    Returns:
        LineCollection: Colored line segments for water surface
    """
    from matplotlib.collections import LineCollection
    import matplotlib.colors as colors
    
    # Create a colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Create segments for the water surface
    points = np.array([x, wse]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
    
    # Create an array of depths to color by
    depths = wse - bed
    
    # Set array based on depths (interpolate for the segments)
    lc.set_array((depths[:-1] + depths[1:]) / 2)
    
    return lc


def plot_dam_with_shading(ax, x, profile, base_elevation):
    """
    Plot a dam with enhanced shading for a 3D effect.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        x (array): x-coordinates
        profile (dict): Dam profile
        base_elevation (float): Base elevation
    """
    from matplotlib.patches import Polygon
    
    # Create dam outline
    ax.plot(x, profile['z'], 'k-', linewidth=1.5)
    
    # Create dam body
    dam_poly = np.column_stack([
        np.concatenate([x, x[::-1]]),
        np.concatenate([profile['z'], np.full_like(x, base_elevation)[::-1]])
    ])
    
    # Add the dam body with gradient shading
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom gray gradient
    dam_cmap = LinearSegmentedColormap.from_list(
        'DamGradient', 
        [(0, '#707070'), (0.5, '#505050'), (1, '#303030')],
        N=256
    )
    
    # Add shading based on height
    z_normalized = (profile['z'] - base_elevation) / (np.max(profile['z']) - base_elevation)
    colors = dam_cmap(z_normalized)
    
    # Create a polygon collection for better shading
    dam_patches = []
    
    # Create multiple patches for shading segments
    for i in range(len(x) - 1):
        patch = Polygon([
            [x[i], profile['z'][i]],
            [x[i+1], profile['z'][i+1]],
            [x[i+1], base_elevation],
            [x[i], base_elevation]
        ])
        dam_patches.append(patch)
    
    # Create patch collection
    from matplotlib.collections import PatchCollection
    dam_collection = PatchCollection(dam_patches, cmap=dam_cmap)
    
    # Set colors based on height
    dam_collection.set_array(z_normalized)
    
    # Add to axis
    ax.add_collection(dam_collection)
    
    # Add a subtle highlight line on top of dam
    ax.plot(x, profile['z'], 'w-', linewidth=0.5, alpha=0.5)


def plot_enhancement_effects(ax, water_surface, bed, froude, x):
    """
    Add enhancement effects to water visualization based on flow conditions.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        water_surface (array): Water surface elevations
        bed (array): Bed elevations
        froude (array): Froude numbers
        x (array): x-coordinates
    """
    # Add wave patterns based on Froude number
    for i in range(len(x) - 5):
        # Skip if no water
        if water_surface[i] <= bed[i]:
            continue
            
        # Current depth and Froude number
        depth = max(0.1, water_surface[i] - bed[i])
        fr = froude[i]
        
        # Skip some points for cleaner visualization
        if i % 10 != 0:
            continue
            
        # Create different wave patterns based on flow regime
        if fr < 0.8:  # Subcritical - smooth surface
            n_points = 8
            wave_x = np.linspace(x[i] - depth, x[i] + depth, n_points)
            wave_amp = 0.01 * depth
            wave_y = water_surface[i] + wave_amp * np.sin(np.linspace(0, 2*np.pi, n_points))
            
            # Add subtle wave line
            ax.plot(wave_x, wave_y, 'white', linewidth=0.7, alpha=0.3)
            
        elif fr > 1.2:  # Supercritical - rough surface
            n_points = 12
            wave_x = np.linspace(x[i] - depth, x[i] + depth, n_points)
            wave_amp = 0.04 * depth * min(fr, 3) / 3  # Scale with Froude
            wave_y = water_surface[i] + wave_amp * np.sin(np.linspace(0, 4*np.pi, n_points))
            
            # Add more pronounced waves
            ax.plot(wave_x, wave_y, 'white', linewidth=1, alpha=0.4)
            
        else:  # Near critical - transitional
            # Just add a highlight at the surface
            ax.plot([x[i] - depth/2, x[i] + depth/2], 
                  [water_surface[i], water_surface[i]], 
                  'white', linewidth=1.2, alpha=0.5)


"""
Fix for styling.py module to ensure compatibility with different matplotlib versions.

The issue is with the Legend object's frame alpha setting method, which varies between
matplotlib versions.
"""

def style_for_publication(fig, tight=True, dpi=300):
    """
    Apply styling suitable for publication to a figure.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to style
        tight (bool): Whether to apply tight layout
        dpi (int): DPI for the figure
    """
    # Apply paper theme
    apply_theme('paper')
    
    # Set figure size to typical publication dimensions (in inches)
    fig.set_size_inches(7, 5)
    
    # Improve axis styling for all subplots
    for ax in fig.get_axes():
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        
        # Adjust tick parameters
        ax.tick_params(width=0.75, length=3, labelsize=9)
        
        # Set grid style
        ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5)
        
        # Ensure axis labels are set with proper font size
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=11, fontweight='bold')
    
    # Style the legends - updated to be compatible with different matplotlib versions
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            # Use a try/except block for compatibility
            try:
                # Modern matplotlib versions
                legend.set_frame_on(True)
                legend.set_framealpha(0.9)
            except AttributeError:
                # Alternative approach for older matplotlib versions
                frame = legend.get_frame()
                if frame:
                    frame.set_alpha(0.9)
                    frame.set_edgecolor('gray')
            
            # Set text sizes - compatible with all versions
            for text in legend.get_texts():
                text.set_fontsize(8)
    
    # Apply tight layout if requested
    if tight:
        fig.tight_layout()
    
    # Set DPI
    fig.set_dpi(dpi)
    
    return fig


def style_for_presentation(fig):
    """
    Apply styling suitable for presentations to a figure.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to style
    """
    # Apply presentation theme
    apply_theme('presentation')
    
    # Set figure size to typical presentation dimensions (in inches)
    fig.set_size_inches(12, 8)
    
    # Improve axis styling for all subplots
    for ax in fig.get_axes():
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Adjust tick parameters
        ax.tick_params(width=1.5, length=6, labelsize=12)
        
        # Set grid style
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Ensure axis labels are set with proper font size
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=16, fontweight='bold')
    
    # Style the legends - updated to be compatible with different matplotlib versions
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend:
            # Use a try/except block for compatibility
            try:
                # Modern matplotlib versions
                legend.set_frame_on(True)
                legend.set_framealpha(0.9)
            except AttributeError:
                # Alternative approach for older matplotlib versions
                frame = legend.get_frame()
                if frame:
                    frame.set_alpha(0.9)
                    frame.set_edgecolor('gray')
            
            # Set text sizes - compatible with all versions
            for text in legend.get_texts():
                text.set_fontsize(12)
    
    # Apply tight layout
    fig.tight_layout()
    
    return fig