"""
Enhanced cross-section visualizations for hydraulic modeling.

This module provides functions for generating and visualizing cross-sections
of channels and flow conditions from 1D hydraulic model results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Arc
import matplotlib.patheffects as PathEffects
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

def plot_channel_cross_section(ax, channel_type, channel_params, water_depth, 
                              highlight_param=None, annotate=True):
    """
    Plot a detailed channel cross-section with water.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        channel_type (str): Type of channel ('rectangular', 'trapezoidal', etc.)
        channel_params (dict): Channel parameters (width, slope, etc.)
        water_depth (float): Water depth (m)
        highlight_param (dict, optional): Parameter to highlight (e.g., velocity, shear)
        annotate (bool): Whether to add detailed annotations
        
    Returns:
        matplotlib.axes.Axes: The updated axes
    """
    # Extract channel parameters
    if channel_type.lower() == 'rectangular':
        bottom_width = channel_params.get('bottom_width', 5.0)
        # For rectangular, the side slope is infinity (vertical walls)
        side_slope = 0
    elif channel_type.lower() == 'trapezoidal':
        bottom_width = channel_params.get('bottom_width', 5.0)
        side_slope = channel_params.get('side_slope', 1.5)
    else:
        # Default to rectangular if unknown
        bottom_width = channel_params.get('bottom_width', 5.0)
        side_slope = 0
    
    # Calculate dimensions
    half_bottom = bottom_width / 2
    if water_depth > 0:
        water_width = bottom_width + 2 * side_slope * water_depth
        half_water = water_width / 2
    else:
        half_water = half_bottom
    
    # Plot the channel bed
    bed_color = '#8B7355'  # Brown color for bed
    
    # Create channel boundary coordinates
    if side_slope > 0:  # Trapezoidal
        channel_x = [-half_bottom - side_slope * 1.5, -half_bottom, half_bottom, 
                   half_bottom + side_slope * 1.5]
        channel_y = [-1.5, 0, 0, -1.5]
    else:  # Rectangular
        channel_x = [-half_bottom - 0.2, -half_bottom, half_bottom, half_bottom + 0.2]
        channel_y = [-1.5, 0, 0, -1.5]
    
    # Plot channel bed
    ax.fill(channel_x, channel_y, color=bed_color, alpha=0.7, hatch='///')
    
    # Plot channel walls
    if side_slope > 0:  # Trapezoidal
        left_wall_x = [-half_bottom - side_slope * 3, -half_bottom]
        right_wall_x = [half_bottom, half_bottom + side_slope * 3]
        
        left_wall_y = [-3, 0]
        right_wall_y = [0, -3]
        
        # Add subtle shading to walls
        ax.fill_between(left_wall_x, left_wall_y, -3, color='gray', alpha=0.3)
        ax.fill_between(right_wall_x, right_wall_y, -3, color='gray', alpha=0.3)
    else:  # Rectangular
        # Add vertical walls extending below
        ax.fill_between([-half_bottom-0.1, -half_bottom+0.1], [3, 3], [-3, -3], 
                      color='gray', alpha=0.5)
        ax.fill_between([half_bottom-0.1, half_bottom+0.1], [3, 3], [-3, -3], 
                      color='gray', alpha=0.5)
    
    # Draw channel profile with proper banks
    if side_slope > 0:  # Trapezoidal
        # Left bank
        ax.plot([-half_bottom - side_slope * 3, -half_bottom], [-3, 0], 'k-', linewidth=1.5)
        # Bottom
        ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
        # Right bank
        ax.plot([half_bottom, half_bottom + side_slope * 3], [0, -3], 'k-', linewidth=1.5)
    else:  # Rectangular
        # Vertical walls
        ax.plot([-half_bottom, -half_bottom], [-3, 3], 'k-', linewidth=1.5)
        ax.plot([half_bottom, half_bottom], [-3, 3], 'k-', linewidth=1.5)
        # Bottom
        ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
    
    # Add water if depth > 0
    if water_depth > 0:
        # Create water polygon
        if side_slope > 0:  # Trapezoidal
            water_x = [-half_water, -half_bottom, half_bottom, half_water]
            water_y = [water_depth, 0, 0, water_depth]
        else:  # Rectangular
            water_x = [-half_bottom, -half_bottom, half_bottom, half_bottom]
            water_y = [water_depth, 0, 0, water_depth]
        
        # Determine water color based on highlight parameter
        if highlight_param is not None:
            param_type = highlight_param.get('type', 'velocity')
            param_value = highlight_param.get('value', 0)
            
            if param_type == 'velocity':
                # Color by velocity
                velocity = param_value
                vmax = highlight_param.get('max_value', 5.0)
                
                # Use viridis colormap for velocity
                cmap = plt.cm.viridis
                norm = colors.Normalize(vmin=0, vmax=vmax)
                rgba_color = cmap(norm(velocity))
                
                # Add velocity vector indicators if velocity is significant
                if velocity > 0.1:
                    # Add flow direction indicators
                    arrow_spacing = max(0.5, bottom_width / 6)
                    arrow_x_positions = np.arange(-half_bottom + arrow_spacing/2, 
                                                half_bottom, arrow_spacing)
                    
                    for x_pos in arrow_x_positions:
                        # Scale arrow size by velocity
                        arrow_size = 0.1 + 0.1 * (velocity / vmax)
                        
                        # Create arrow at different heights
                        heights = np.linspace(0.2 * water_depth, 0.8 * water_depth, 3)
                        
                        for h in heights:
                            # Only add if within water
                            if h < water_depth:
                                # Calculate width at this height
                                if side_slope > 0:
                                    width_at_height = bottom_width + 2 * side_slope * h
                                    half_width = width_at_height / 2
                                else:
                                    half_width = half_bottom
                                
                                # Skip if outside water width at this height
                                if abs(x_pos) > half_width:
                                    continue
                                
                                # Draw arrow
                                ax.arrow(x_pos, h, arrow_size, 0, head_width=arrow_size*0.8, 
                                       head_length=arrow_size*1.5, fc='white', ec='white', 
                                       alpha=0.7)
                
            elif param_type == 'froude':
                # Color by Froude number
                froude = param_value
                
                # Use coolwarm colormap with center at Fr=1
                cmap = plt.cm.coolwarm
                norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
                rgba_color = cmap(norm(froude))
                
                # Add flow regime indicators
                if froude < 0.8:  # Subcritical
                    # Smooth surface
                    surface_x = np.linspace(-half_water, half_water, 50)
                    wave_amp = 0.01 * water_depth
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 4*np.pi, 50))
                    ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1)
                    
                elif froude > 1.2:  # Supercritical
                    # Rougher surface
                    surface_x = np.linspace(-half_water, half_water, 100)
                    wave_amp = 0.05 * water_depth
                    # Add higher frequency waves
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 12*np.pi, 100))
                    ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1.5)
                
                else:  # Near critical - transitional
                    # Intermediate waves
                    surface_x = np.linspace(-half_water, half_water, 75)
                    wave_amp = 0.03 * water_depth
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 8*np.pi, 75))
                    ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1.2)
                    
            elif param_type == 'shear':
                # Color by shear stress
                shear = param_value
                shear_max = highlight_param.get('max_value', 100.0)
                
                # Use autumn colormap for shear stress
                cmap = plt.cm.YlOrRd
                norm = colors.Normalize(vmin=0, vmax=shear_max)
                rgba_color = cmap(norm(shear))
                
                # Add shear visualization along the boundary
                if shear > 1:
                    # Higher shear means more intensity
                    intensity = min(1.0, shear / shear_max)
                    
                    # Create markers along the boundary to indicate shear
                    if side_slope > 0:  # Trapezoidal
                        # Left bank
                        left_x = np.linspace(-half_bottom, -half_water, 10)
                        left_y = np.linspace(0, water_depth, 10)
                        
                        # Right bank
                        right_x = np.linspace(half_bottom, half_water, 10)
                        right_y = np.linspace(0, water_depth, 10)
                        
                        # Bottom
                        bottom_x = np.linspace(-half_bottom, half_bottom, 20)
                        bottom_y = np.zeros_like(bottom_x)
                        
                        # Add small lines perpendicular to boundary to indicate shear
                        for i in range(len(left_x)):
                            # Calculate perpendicular direction
                            dx = left_x[i] - (-half_bottom)
                            dy = left_y[i] - 0
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                nx = -dy / length  # Perpendicular x component
                                ny = dx / length   # Perpendicular y component
                                # Length proportional to shear
                                line_length = 0.05 + 0.1 * intensity
                                ax.plot([left_x[i], left_x[i] + nx * line_length], 
                                      [left_y[i], left_y[i] + ny * line_length],
                                      'r-', alpha=0.7, linewidth=1)
                        
                        for i in range(len(right_x)):
                            # Calculate perpendicular direction
                            dx = right_x[i] - half_bottom
                            dy = right_y[i] - 0
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                nx = -dy / length
                                ny = dx / length
                                line_length = 0.05 + 0.1 * intensity
                                ax.plot([right_x[i], right_x[i] + nx * line_length], 
                                      [right_y[i], right_y[i] + ny * line_length],
                                      'r-', alpha=0.7, linewidth=1)
                        
                        for i in range(len(bottom_x)):
                            # For bottom, perpendicular is straight up
                            line_length = 0.05 + 0.1 * intensity
                            ax.plot([bottom_x[i], bottom_x[i]], 
                                  [bottom_y[i], bottom_y[i] + line_length],
                                  'r-', alpha=0.7, linewidth=1)
                    
                    else:  # Rectangular
                        # Left wall
                        left_x = -half_bottom * np.ones(10)
                        left_y = np.linspace(0, water_depth, 10)
                        
                        # Right wall
                        right_x = half_bottom * np.ones(10)
                        right_y = np.linspace(0, water_depth, 10)
                        
                        # Bottom
                        bottom_x = np.linspace(-half_bottom, half_bottom, 20)
                        bottom_y = np.zeros_like(bottom_x)
                        
                        # Add small lines perpendicular to boundary
                        for i in range(len(left_y)):
                            line_length = 0.05 + 0.1 * intensity
                            ax.plot([left_x[i], left_x[i] + line_length], 
                                  [left_y[i], left_y[i]],
                                  'r-', alpha=0.7, linewidth=1)
                        
                        for i in range(len(right_y)):
                            line_length = 0.05 + 0.1 * intensity
                            ax.plot([right_x[i], right_x[i] - line_length], 
                                  [right_y[i], right_y[i]],
                                  'r-', alpha=0.7, linewidth=1)
                        
                        for i in range(len(bottom_x)):
                            line_length = 0.05 + 0.1 * intensity
                            ax.plot([bottom_x[i], bottom_x[i]], 
                                  [bottom_y[i], bottom_y[i] + line_length],
                                  'r-', alpha=0.7, linewidth=1)
            else:
                # Default color for water
                rgba_color = (0.5, 0.7, 1.0, 0.7)  # Light blue, semi-transparent
        else:
            # Default color for water
            rgba_color = (0.5, 0.7, 1.0, 0.7)  # Light blue, semi-transparent
        
        # Plot water polygon
        water_poly = Polygon(np.column_stack([water_x, water_y]), 
                           facecolor=rgba_color, edgecolor='blue', linewidth=1)
        ax.add_patch(water_poly)
        
        # Add a subtle water surface line
        ax.plot([-half_water, half_water], [water_depth, water_depth], 
              'blue', linewidth=1.5, alpha=0.8)
    
    # Add annotations if requested
    if annotate:
        # Annotate bottom width
        ax.annotate(f"Bottom Width = {bottom_width:.2f}m", 
                   xy=(0, -0.2), 
                   xytext=(0, -0.5),
                   ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if water_depth > 0:
            # Annotate water depth
            ax.annotate(f"Depth = {water_depth:.2f}m", 
                       xy=(half_water/2, water_depth/2), 
                       xytext=(half_water/2 + 1, water_depth/2),
                       ha='left', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            if water_width > bottom_width:
                # Annotate water surface width
                ax.annotate(f"Top Width = {water_width:.2f}m", 
                           xy=(0, water_depth + 0.1), 
                           xytext=(0, water_depth + 0.5),
                           ha='center', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if side_slope > 0:
            # Annotate side slope
            slope_text = f"Side Slope = {side_slope}:1"
            # Position on the right bank
            slope_x = half_bottom + side_slope * water_depth / 2
            slope_y = water_depth / 2
            
            ax.annotate(slope_text,
                       xy=(slope_x, slope_y),
                       xytext=(slope_x + 1, slope_y),
                       ha='left', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits with padding
    max_width = max(bottom_width, water_width if water_depth > 0 else bottom_width)
    ax.set_xlim(-max_width/2 - 3, max_width/2 + 3)
    
    max_height = max(3, water_depth * 1.5)
    ax.set_ylim(-1.5, max_height)
    
    return ax


def create_cross_section_dashboard(scenario, results, locations=None, figsize=(14, 10)):
    """
    Create a comprehensive dashboard of cross-sections at multiple locations
    along the channel with hydraulic parameters.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        locations (list): List of x locations for cross-sections
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    if locations is None:
        # Choose default locations: upstream, at dam, just downstream,
        # hydraulic jump (if present), and far downstream
        locations = [-10, 0, 20, 100]
        
        # Add hydraulic jump location if present
        jump = results['hydraulic_jump']
        if jump.get('jump_possible', False):
            locations.insert(3, jump['location'])
    
    # Number of locations
    n_locations = len(locations)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Create one row for the profile and one row for cross-sections
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, n_locations, height_ratios=[1, 2], figure=fig)
    
    # Main profile plot will span the top row
    ax_profile = fig.add_subplot(gs[0, :])
    
    # Cross-section plots in the bottom row
    axes_sections = []
    for i in range(n_locations):
        axes_sections.append(fig.add_subplot(gs[1, i]))
    
    # Plot the main profile
    from .enhanced_profiles import plot_enhanced_profile
    plot_enhanced_profile(scenario, results, ax=ax_profile)
    
    # Add location markers to the profile
    for i, loc in enumerate(locations):
        # Add vertical line at location
        ax_profile.axvline(x=loc, color='gray', linestyle='--', alpha=0.5)
        
        # Add marker with letter label
        marker_y = ax_profile.get_ylim()[1] - 0.05 * (ax_profile.get_ylim()[1] - ax_profile.get_ylim()[0])
        ax_profile.annotate(f"{chr(65+i)}", xy=(loc, marker_y), ha='center', va='center',
                           bbox=dict(boxstyle="circle", fc="white", ec="black"))
    
    # Extract channel parameters
    channel_type = 'trapezoidal'  # Default type
    channel_params = {
        'bottom_width': scenario['channel_bottom_width'],
        'side_slope': scenario['channel_side_slope']
    }
    
    # Extract data for cross-sections
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    dam_crest = scenario['dam_crest_elevation']
    tailwater = results['tailwater']
    
    # Plot cross-sections
    for i, loc in enumerate(locations):
        ax = axes_sections[i]
        
        # Determine water depth and bed elevation at this location
        if loc <= 0:  # Upstream of dam
            water_elevation = results['upstream_level']
            bed_elevation = dam_base
            water_depth = max(0, water_elevation - bed_elevation)
            
            # Estimate velocity
            if water_depth > 0 and results['discharge'] > 0:
                area = water_depth * scenario['channel_width_at_dam']
                velocity = results['discharge'] / area
                froude = velocity / np.sqrt(9.81 * water_depth)
            else:
                velocity = 0
                froude = 0
                
            # Estimate shear stress (simplified τ = ρgRS)
            if water_depth > 0:
                rho = 1000  # Water density
                g = 9.81     # Gravity
                R = water_depth  # Simplified hydraulic radius
                S = scenario['channel_slope']
                shear = rho * g * R * S
            else:
                shear = 0
                
        else:  # Downstream of dam
            # Find closest point in tailwater results
            idx = np.argmin(np.abs(tailwater['x'] - loc))
            if idx < len(tailwater['wse']):
                water_elevation = tailwater['wse'][idx]
                bed_elevation = tailwater['z'][idx]
                water_depth = max(0, water_elevation - bed_elevation)
                
                # Get velocity and Froude number if available
                velocity = tailwater['v'][idx] if idx < len(tailwater['v']) else 0
                froude = tailwater['fr'][idx] if idx < len(tailwater['fr']) else 0
                
                # Estimate shear stress
                if water_depth > 0:
                    rho = 1000
                    g = 9.81
                    R = water_depth  # Simplified
                    S = scenario['downstream_slope']
                    shear = rho * g * R * S
                else:
                    shear = 0
            else:
                water_depth = 0
                velocity = 0
                froude = 0
                shear = 0
        
        # Determine parameter to highlight
        # Choose only one parameter to highlight - prioritize based on location
        if loc <= 0:  # Upstream
            # Default to velocity for upstream
            highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
        elif abs(loc) < 0.1:  # At dam
            # Default to velocity for dam
            highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
        else:  # Downstream
            # Use Froude for downstream to show flow regime
            highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
            
            # Check if near hydraulic jump
            jump = results['hydraulic_jump']
            if jump.get('jump_possible', False):
                jump_loc = jump['location']
                if abs(loc - jump_loc) < 20:
                    # Highlight Froude number near jump
                    highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
                elif loc > jump_loc + 20:
                    # Highlight shear stress far downstream
                    highlight_param = {'type': 'shear', 'value': shear, 'max_value': 100.0}
        
        # Plot enhanced cross-section
        plot_channel_cross_section(ax, channel_type, channel_params, water_depth, 
                                  highlight_param=highlight_param, annotate=True)
        
        # Set title with location and key parameters
        if loc <= 0:
            title = f"A{i+1}: Upstream (x={loc}m)"
        elif abs(loc) < 0.1:
            title = f"A{i+1}: At Dam (x=0m)"
        else:
            title = f"A{i+1}: Downstream (x={loc}m)"
            
        # Add key hydraulic parameters to title
        if water_depth > 0:
            title += f"\nDepth={water_depth:.2f}m, V={velocity:.2f}m/s, Fr={froude:.2f}"
        
        ax.set_title(title)
        
        # Add corresponding label
        ax.annotate(f"{chr(65+i)}", xy=(0.05, 0.95), xycoords='axes fraction',
                   ha='left', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="circle", fc="white", ec="black"))
        
        # Set labels (only on left edge and bottom row)
        if i == 0:
            ax.set_ylabel('Elevation (m)')
        
    # Add overall title
    plt.suptitle('Hydraulic Cross-Sections Along Channel', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, (ax_profile, axes_sections)


def create_animated_cross_section(scenario, results_list, location, fps=5, dpi=100):
    """
    Create an animation of a channel cross-section changing over time.
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results at different time steps
        location (float): x location for the cross-section
        fps (int): Frames per second for the animation
        dpi (int): Resolution for the animation
        
    Returns:
        tuple: (anim, fig, ax) Animation object, figure and axis
    """
    import matplotlib.animation as animation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract channel parameters
    channel_type = 'trapezoidal'  # Default type
    channel_params = {
        'bottom_width': scenario['channel_bottom_width'],
        'side_slope': scenario['channel_side_slope']
    }
    
    # Calculate max depth across all results for consistent scaling
    max_depth = 0
    
    for result in results_list:
        # Determine water depth at this location
        if location <= 0:  # Upstream of dam
            water_elevation = result['upstream_level']
            bed_elevation = scenario['dam_base_elevation']
            water_depth = max(0, water_elevation - bed_elevation)
        else:  # Downstream of dam
            tailwater = result['tailwater']
            # Find closest point in tailwater results
            idx = np.argmin(np.abs(tailwater['x'] - location))
            if idx < len(tailwater['wse']):
                water_elevation = tailwater['wse'][idx]
                bed_elevation = tailwater['z'][idx]
                water_depth = max(0, water_elevation - bed_elevation)
            else:
                water_depth = 0
        
        max_depth = max(max_depth, water_depth)
    
    # Add some padding to max depth
    max_depth = max(max_depth * 1.2, 3)
    
    # Calculate width limits for consistent scaling
    bottom_width = channel_params['bottom_width']
    side_slope = channel_params['side_slope']
    max_width = bottom_width + 2 * side_slope * max_depth
    
    # Set fixed axis limits for a stable animation
    ax.set_xlim(-max_width/2 - 3, max_width/2 + 3)
    ax.set_ylim(-1.5, max_depth + 1)
    
    # Static elements
    # Set title based on location
    if location <= 0:
        ax.set_title(f"Cross-Section at Upstream Location (x={location}m)")
    elif abs(location) < 0.1:
        ax.set_title(f"Cross-Section at Dam (x=0m)")
    else:
        ax.set_title(f"Cross-Section at Downstream Location (x={location}m)")
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set labels
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Elevation (m)')
    
    # Add equal aspect ratio for better visualization
    ax.set_aspect('equal')
    
    # Create progress indicator at bottom
    progress_text = ax.text(0.5, -0.1, "", transform=ax.transAxes, 
                          ha='center', va='top', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Create parameter display
    param_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                       ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    
    # List to track dynamic elements
    dynamic_elements = []
    
    # Function to update the animation for each frame
    def update(frame_idx):
        # Clear previous dynamic elements
        for element in dynamic_elements:
            element.remove()
        dynamic_elements.clear()
        
        # Get the result for this frame
        result = results_list[frame_idx]
        
        # Determine water depth and parameters at this location
        if location <= 0:  # Upstream of dam
            water_elevation = result['upstream_level']
            bed_elevation = scenario['dam_base_elevation']
            water_depth = max(0, water_elevation - bed_elevation)
            
            # Estimate velocity
            if water_depth > 0 and result['discharge'] > 0:
                area = water_depth * scenario['channel_width_at_dam']
                velocity = result['discharge'] / area
                froude = velocity / np.sqrt(9.81 * water_depth)
            else:
                velocity = 0
                froude = 0
            
            # Parameters to highlight
            highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
            
        else:  # Downstream of dam
            tailwater = result['tailwater']
            # Find closest point in tailwater results
            idx = np.argmin(np.abs(tailwater['x'] - location))
            if idx < len(tailwater['wse']):
                water_elevation = tailwater['wse'][idx]
                bed_elevation = tailwater['z'][idx]
                water_depth = max(0, water_elevation - bed_elevation)
                
                # Get velocity and Froude number if available
                velocity = tailwater['v'][idx] if idx < len(tailwater['v']) else 0
                froude = tailwater['fr'][idx] if idx < len(tailwater['fr']) else 0
            else:
                water_depth = 0
                velocity = 0
                froude = 0
            
            # Check if near a hydraulic jump
            jump = result['hydraulic_jump']
            if jump.get('jump_possible', False):
                jump_loc = jump['location']
                if abs(location - jump_loc) < 10:
                    # Use Froude highlighting near a jump
                    highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
                else:
                    # Default to velocity for other locations
                    highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
            else:
                # Default to velocity
                highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
        
        # Calculate cross-section geometry
        half_bottom = bottom_width / 2
        if water_depth > 0:
            water_width = bottom_width + 2 * side_slope * water_depth
            half_water = water_width / 2
        else:
            water_width = bottom_width
            half_water = half_bottom
        
        # Draw static channel bed and banks
        if frame_idx == 0:
            # Only draw once on first frame
            if side_slope > 0:  # Trapezoidal
                # Left bank
                ax.plot([-half_bottom - side_slope * 3, -half_bottom], [-3, 0], 'k-', linewidth=1.5)
                # Bottom
                ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
                # Right bank
                ax.plot([half_bottom, half_bottom + side_slope * 3], [0, -3], 'k-', linewidth=1.5)
                
                # Fill channel bed
                bed_patch = Polygon([
                    [-half_bottom - side_slope * 3, -3],
                    [-half_bottom, 0],
                    [half_bottom, 0],
                    [half_bottom + side_slope * 3, -3],
                    [-half_bottom - side_slope * 3, -3]
                ], facecolor='#8B7355', alpha=0.7, hatch='///')
                ax.add_patch(bed_patch)
                
            else:  # Rectangular
                # Vertical walls
                ax.plot([-half_bottom, -half_bottom], [-3, 3], 'k-', linewidth=1.5)
                ax.plot([half_bottom, half_bottom], [-3, 3], 'k-', linewidth=1.5)
                # Bottom
                ax.plot([-half_bottom, half_bottom], [0, 0], 'k-', linewidth=1.5)
                
                # Fill channel bed
                bed_patch = Polygon([
                    [-half_bottom-1, -3],
                    [-half_bottom-1, 0],
                    [half_bottom+1, 0],
                    [half_bottom+1, -3],
                    [-half_bottom-1, -3]
                ], facecolor='#8B7355', alpha=0.7, hatch='///')
                ax.add_patch(bed_patch)
        
        # Add water if depth > 0
        if water_depth > 0:
            # Create water polygon
            if side_slope > 0:  # Trapezoidal
                water_x = [-half_water, -half_bottom, half_bottom, half_water]
                water_y = [water_depth, 0, 0, water_depth]
            else:  # Rectangular
                water_x = [-half_bottom, -half_bottom, half_bottom, half_bottom]
                water_y = [water_depth, 0, 0, water_depth]
            
            # Determine water color based on highlight parameter
            if highlight_param['type'] == 'velocity':
                # Color by velocity
                velocity = highlight_param['value']
                vmax = highlight_param['max_value']
                
                # Use viridis colormap for velocity
                cmap = plt.cm.viridis
                norm = colors.Normalize(vmin=0, vmax=vmax)
                rgba_color = cmap(norm(velocity))
                
            elif highlight_param['type'] == 'froude':
                # Color by Froude number
                froude = highlight_param['value']
                
                # Use coolwarm colormap with center at Fr=1
                cmap = plt.cm.coolwarm
                norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
                rgba_color = cmap(norm(froude))
                
            else:
                # Default color for water
                rgba_color = (0.5, 0.7, 1.0, 0.7)  # Light blue, semi-transparent
            
            # Plot water polygon
            water_poly = Polygon(np.column_stack([water_x, water_y]), 
                               facecolor=rgba_color, edgecolor='blue', linewidth=1)
            ax.add_patch(water_poly)
            dynamic_elements.append(water_poly)
            
            # Add a subtle water surface line
            water_line, = ax.plot([-half_water, half_water], [water_depth, water_depth], 
                                'blue', linewidth=1.5, alpha=0.8)
            dynamic_elements.append(water_line)
            
            # Add flow visualization based on parameters
            if highlight_param['type'] == 'velocity' and velocity > 0.1:
                # Add velocity vectors
                arrow_spacing = max(0.5, bottom_width / 6)
                arrow_x_positions = np.arange(-half_bottom + arrow_spacing/2, 
                                            half_bottom, arrow_spacing)
                
                for x_pos in arrow_x_positions:
                    # Scale arrow size with velocity
                    arrow_size = 0.1 + 0.1 * (velocity / vmax)
                    
                    # Create arrows at different heights
                    heights = np.linspace(0.2 * water_depth, 0.8 * water_depth, 3)
                    
                    for h in heights:
                        # Only add if within water
                        if h < water_depth:
                            # Calculate width at this height
                            if side_slope > 0:
                                width_at_height = bottom_width + 2 * side_slope * h
                                half_width = width_at_height / 2
                            else:
                                half_width = half_bottom
                            
                            # Skip if outside water width at this height
                            if abs(x_pos) > half_width:
                                continue
                            
                            # Draw arrow
                            arrow = ax.arrow(x_pos, h, arrow_size, 0, 
                                           head_width=arrow_size*0.8, 
                                           head_length=arrow_size*1.5, 
                                           fc='white', ec='white', alpha=0.7)
                            dynamic_elements.append(arrow)
                
            elif highlight_param['type'] == 'froude':
                # Add surface features based on Froude number
                froude = highlight_param['value']
                
                if froude < 0.8:  # Subcritical
                    # Smooth surface
                    surface_x = np.linspace(-half_water, half_water, 50)
                    wave_amp = 0.01 * water_depth
                    # Animate waves
                    phase = frame_idx * 0.2
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 4*np.pi, 50) + phase)
                    surf_line, = ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1)
                    dynamic_elements.append(surf_line)
                    
                elif froude > 1.2:  # Supercritical
                    # Rougher surface
                    surface_x = np.linspace(-half_water, half_water, 100)
                    wave_amp = 0.05 * water_depth
                    # Animate waves with higher frequency
                    phase = frame_idx * 0.4
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 12*np.pi, 100) + phase)
                    surf_line, = ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1.5)
                    dynamic_elements.append(surf_line)
                
                else:  # Near critical - transitional
                    # Intermediate waves
                    surface_x = np.linspace(-half_water, half_water, 75)
                    wave_amp = 0.03 * water_depth
                    # Animate waves
                    phase = frame_idx * 0.3
                    surface_y = water_depth + wave_amp * np.sin(np.linspace(0, 8*np.pi, 75) + phase)
                    surf_line, = ax.plot(surface_x, surface_y, 'white', alpha=0.7, linewidth=1.2)
                    dynamic_elements.append(surf_line)
        
        # Update progress text
        progress_text.set_text(f"Time Step: {frame_idx+1}/{len(results_list)}")
        
        # Update parameter display
        if water_depth > 0:
            discharge = result['discharge']
            param_str = f"Depth: {water_depth:.2f}m\nVelocity: {velocity:.2f}m/s\n" \
                         f"Froude: {froude:.2f}\nDischarge: {discharge:.2f}m³/s"
            flow_regime = "Subcritical" if froude < 1 else "Supercritical" if froude > 1 else "Critical"
            param_str += f"\nFlow Regime: {flow_regime}"
        else:
            param_str = "No flow"
        
        param_text.set_text(param_str)
        
        return dynamic_elements + [progress_text, param_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                 interval=1000/fps, blit=True)
    
    plt.tight_layout()
    
    return anim, fig, ax