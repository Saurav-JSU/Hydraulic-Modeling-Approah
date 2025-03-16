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
        
    Returns:
        dict: Color settings specific to the theme
    """
    if theme not in THEMES:
        print(f"Unknown theme: {theme}. Using default.")
        theme = 'default'
    
    # Apply theme settings - with error handling
    for key, value in THEMES[theme].items():
        try:
            plt.rcParams[key] = value
        except KeyError:
            print(f"Warning: rcParam '{key}' not found. Skipping.")
    
    # Set other common parameters - with error handling
    try:
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
    except Exception as e:
        print(f"Warning: Could not set some rcParams: {str(e)}")
    
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
    if ax is None:
        print("Warning: No axes provided for styling.")
        return
    
    # Set grid style
    if grid:
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    
    # Style the legend if requested and if a legend exists
    if show_legend:
        legend = ax.get_legend()
        if legend is not None:
            # Use framealpha if available, otherwise set_alpha
            try:
                legend.set_framealpha(0.9)
            except (AttributeError, TypeError):
                # For older matplotlib versions
                frame = legend.get_frame()
                if frame is not None:
                    frame.set_alpha(0.9)
            
            # Set text sizes
            for text in legend.get_texts():
                text.set_fontsize(10)
    
    # Add a subtle box around the axis
    try:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_alpha(0.2)
        ax.spines['right'].set_alpha(0.2)
    except (AttributeError, KeyError):
        # Some matplotlib versions might not have spines attribute
        pass


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
        matplotlib.colorbar.Colorbar or None: The colorbar or None if creation failed
    """
    if fig is None or ax is None or mappable is None:
        print("Warning: Required parameters missing for colorbar creation.")
        return None
    
    try:
        # Create colorbar
        cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=pad)
        
        # Set label
        cbar.set_label(label, fontsize=11)
        
        # Style ticks
        cbar.ax.tick_params(labelsize=9)
        
        return cbar
    except Exception as e:
        print(f"Warning: Could not create colorbar: {str(e)}")
        return None


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
        LineCollection or None: Colored line segments for water surface
    """
    # Validate inputs
    if x is None or wse is None or bed is None:
        print("Warning: Input arrays cannot be None for water surface segments.")
        return None
    
    # Ensure arrays have the same length and are not empty
    if len(x) != len(wse) or len(x) != len(bed) or len(x) < 2:
        print("Warning: Input arrays must have same length (at least 2) for water surface segments.")
        return None
    
    try:
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
        if len(depths) > 1:
            lc.set_array((depths[:-1] + depths[1:]) / 2)
        
        return lc
    except Exception as e:
        print(f"Warning: Error creating water surface segments: {str(e)}")
        return None


def plot_dam_with_shading(ax, x, profile, base_elevation):
    """
    Plot a dam with enhanced shading for a 3D effect.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        x (array): x-coordinates
        profile (dict): Dam profile
        base_elevation (float): Base elevation
    """
    if ax is None or x is None or profile is None:
        print("Warning: Required parameters missing for dam plotting.")
        return
    
    # Check if profile contains z values
    if not isinstance(profile, dict) or 'z' not in profile or len(profile['z']) != len(x):
        print("Warning: Dam profile must contain 'z' key with array of same length as x.")
        # Draw a simple dam as fallback
        try:
            center_x = np.mean(x) if len(x) > 0 else 0
            dam_height = 10.0  # Arbitrary height if not provided
            ax.plot([center_x, center_x], [base_elevation, base_elevation + dam_height], 'k-', linewidth=1.5)
            ax.fill_betweenx([base_elevation, base_elevation + dam_height], 
                           [center_x - 0.5, center_x - 0.5], 
                           [center_x + 0.5, center_x + 0.5],
                           color='#4F4F4F')
            return
        except Exception as e:
            print(f"Warning: Could not draw simple dam: {str(e)}")
            return
    
    try:
        from matplotlib.patches import Polygon
        
        # Create dam outline
        ax.plot(x, profile['z'], 'k-', linewidth=1.5)
        
        # Ensure z values are valid for gradient calculation
        z_max = np.max(profile['z'])
        z_min = base_elevation
        
        # Check for division by zero
        if z_max <= z_min:
            # Use a simple color instead of gradient
            dam_poly = np.column_stack([
                np.concatenate([x, x[::-1]]),
                np.concatenate([profile['z'], np.full_like(x, base_elevation)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, color='#4F4F4F')
            ax.add_patch(dam_patch)
            return
        
        # Create a custom gray gradient
        from matplotlib.colors import LinearSegmentedColormap
        dam_cmap = LinearSegmentedColormap.from_list(
            'DamGradient', 
            [(0, '#707070'), (0.5, '#505050'), (1, '#303030')],
            N=256
        )
        
        # Add shading based on height with safe computation
        z_normalized = np.clip((profile['z'] - z_min) / (z_max - z_min), 0, 1)
        
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
    
    except Exception as e:
        print(f"Warning: Error plotting dam with shading: {str(e)}")
        # Attempt a simpler dam visualization
        try:
            dam_poly = np.column_stack([
                np.concatenate([x, x[::-1]]),
                np.concatenate([profile['z'], np.full_like(x, base_elevation)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, color='#4F4F4F')
            ax.add_patch(dam_patch)
        except Exception as e2:
            print(f"Warning: Could not draw simple dam: {str(e2)}")


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
    if ax is None:
        print("Warning: No axes provided for enhancement effects.")
        return
    
    # Validate input arrays
    if water_surface is None or bed is None or froude is None or x is None:
        print("Warning: Input arrays cannot be None for enhancement effects.")
        return
    
    # Ensure arrays have the same length
    min_length = min(len(water_surface), len(bed), len(froude), len(x))
    if min_length < 5:  # Need at least a few points for effects
        print("Warning: Input arrays too short for enhancement effects.")
        return
    
    try:
        # Create local copies trimmed to the same length for safety
        water = water_surface[:min_length]
        bed_elev = bed[:min_length]
        fr = froude[:min_length]
        x_coords = x[:min_length]
        
        # Add wave patterns based on Froude number
        for i in range(min_length - 5):
            # Skip if no water
            if water[i] <= bed_elev[i]:
                continue
                
            # Current depth and Froude number
            depth = max(0.1, water[i] - bed_elev[i])
            fr_val = fr[i]
            
            # Skip some points for cleaner visualization
            if i % 10 != 0:
                continue
                
            # Create different wave patterns based on flow regime
            if fr_val < 0.8:  # Subcritical - smooth surface
                n_points = 8
                wave_x = np.linspace(x_coords[i] - depth, x_coords[i] + depth, n_points)
                wave_amp = 0.01 * depth
                wave_y = water[i] + wave_amp * np.sin(np.linspace(0, 2*np.pi, n_points))
                
                # Add subtle wave line
                ax.plot(wave_x, wave_y, 'white', linewidth=0.7, alpha=0.3)
                
            elif fr_val > 1.2:  # Supercritical - rough surface
                n_points = 12
                wave_x = np.linspace(x_coords[i] - depth, x_coords[i] + depth, n_points)
                wave_amp = 0.04 * depth * min(fr_val, 3) / 3  # Scale with Froude
                wave_y = water[i] + wave_amp * np.sin(np.linspace(0, 4*np.pi, n_points))
                
                # Add more pronounced waves
                ax.plot(wave_x, wave_y, 'white', linewidth=1, alpha=0.4)
                
            else:  # Near critical - transitional
                # Just add a highlight at the surface
                ax.plot([x_coords[i] - depth/2, x_coords[i] + depth/2], 
                      [water[i], water[i]], 
                      'white', linewidth=1.2, alpha=0.5)
    
    except Exception as e:
        print(f"Warning: Error plotting enhancement effects: {str(e)}")


def style_for_publication(fig, tight=True, dpi=300):
    """
    Apply styling suitable for publication to a figure.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to style
        tight (bool): Whether to apply tight layout
        dpi (int): DPI for the figure
        
    Returns:
        matplotlib.figure.Figure: The styled figure
    """
    if fig is None:
        print("Warning: No figure provided for publication styling.")
        return None
    
    try:
        # Apply paper theme
        apply_theme('paper')
        
        # Set figure size to typical publication dimensions (in inches)
        fig.set_size_inches(7, 5)
        
        # Improve axis styling for all subplots
        for ax in fig.get_axes():
            # Thicker axis lines
            try:
                for spine in ax.spines.values():
                    spine.set_linewidth(0.75)
            except (AttributeError, KeyError):
                pass
            
            # Adjust tick parameters
            try:
                ax.tick_params(width=0.75, length=3, labelsize=9)
            except AttributeError:
                pass
            
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
            try:
                fig.tight_layout()
            except Exception as e:
                print(f"Warning: Could not apply tight layout: {str(e)}")
        
        # Set DPI
        try:
            fig.set_dpi(dpi)
        except AttributeError:
            print("Warning: Could not set figure DPI.")
        
        return fig
    
    except Exception as e:
        print(f"Warning: Error applying publication style: {str(e)}")
        return fig


def style_for_presentation(fig):
    """
    Apply styling suitable for presentations to a figure.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to style
        
    Returns:
        matplotlib.figure.Figure: The styled figure
    """
    if fig is None:
        print("Warning: No figure provided for presentation styling.")
        return None
    
    try:
        # Apply presentation theme
        apply_theme('presentation')
        
        # Set figure size to typical presentation dimensions (in inches)
        fig.set_size_inches(12, 8)
        
        # Improve axis styling for all subplots
        for ax in fig.get_axes():
            # Thicker axis lines
            try:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
            except (AttributeError, KeyError):
                pass
            
            # Adjust tick parameters
            try:
                ax.tick_params(width=1.5, length=6, labelsize=12)
            except AttributeError:
                pass
            
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
        try:
            fig.tight_layout()
        except Exception as e:
            print(f"Warning: Could not apply tight layout: {str(e)}")
        
        return fig
    
    except Exception as e:
        print(f"Warning: Error applying presentation style: {str(e)}")
        return fig