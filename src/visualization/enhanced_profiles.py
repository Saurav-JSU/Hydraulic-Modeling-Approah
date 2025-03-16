"""
Enhanced profile visualization for hydraulic modeling.

This module extends the basic visualization capabilities with advanced
coloring, shading, and annotation features for 1D hydraulic profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as PathEffects

def plot_enhanced_profile(scenario, results, display_range=None, 
                          color_by='froude', show_annotations=True, ax=None):
    """
    Create an enhanced visualization of the hydraulic profile.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        display_range (tuple): The x-range to display (min, max)
        color_by (str): Parameter to color the water surface by 
                       ('froude', 'velocity', 'depth', 'energy')
        show_annotations (bool): Whether to show detailed annotations
        ax (matplotlib.axes.Axes, optional): The axes to plot on
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure
    
    # Extract data with safe dictionary access
    dam = scenario.get('dam')
    if dam is None:
        raise ValueError("Scenario dictionary must contain 'dam' key")
    
    dam_base = scenario.get('dam_base_elevation', 0)
    tailwater = results.get('tailwater', {})
    
    # Validate required data is present
    if not tailwater or 'x' not in tailwater:
        raise ValueError("Results dictionary must contain valid 'tailwater' data with 'x' coordinates")
    
    # Create x and y ranges for full domain
    x_dam = np.linspace(-20, 5, 50)  # For dam
    x_values = tailwater.get('x', [])  # Downstream profile
    
    # Ensure we have data to work with
    if len(x_values) == 0:
        raise ValueError("Tailwater 'x' coordinates cannot be empty")
    
    # Combine ranges and sort
    combined_x = np.concatenate([x_dam, x_values])
    idx = np.argsort(combined_x)
    combined_x = combined_x[idx]
    
    # Get dam profile - handle if get_profile method doesn't exist
    try:
        dam_profile = dam.get_profile(x_dam)
    except (AttributeError, TypeError):
        # Create a simple dam profile if get_profile doesn't exist
        dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
        dam_profile = {'z': dam_base + (dam_crest - dam_base) * np.maximum(0, np.minimum(1, (x_dam + 20) / 20))}
    
    # Create bed profile along full domain
    bed = np.zeros_like(combined_x)
    
    # Fill dam portion
    for i, x in enumerate(combined_x):
        if x <= 0:  # Upstream of dam
            bed[i] = dam_base
        elif x <= 5:  # Dam body
            # Interpolate dam profile
            idx = np.argmin(np.abs(x_dam - x))
            if 'z' in dam_profile and idx < len(dam_profile['z']):
                bed[i] = dam_profile['z'][idx]
            else:
                # Fallback if no z values in dam profile
                bed[i] = dam_base
        else:  # Downstream of dam
            # Interpolate tailwater bed
            idx = np.argmin(np.abs(x_values - x))
            if idx < len(tailwater.get('z', [])):
                bed[i] = tailwater['z'][idx]
            else:
                # Fallback if index out of range
                bed[i] = dam_base
    
    # Create water surface profile
    water_surface = np.zeros_like(combined_x)
    
    for i, x in enumerate(combined_x):
        if x <= 0:  # Upstream of dam
            water_surface[i] = results.get('upstream_level', dam_base)
        else:  # Downstream of dam
            # Interpolate tailwater surface
            idx = np.argmin(np.abs(x_values - x))
            if idx < len(tailwater.get('wse', [])):
                water_surface[i] = tailwater['wse'][idx]
            else:
                # Fallback if index out of range
                water_surface[i] = bed[i]  # Assume dry bed
    
    # Plot enhanced bed profile with detailed shading
    ax.plot(combined_x, bed, 'k-', linewidth=1.5, label='Channel Bed')
    
    # Add bed shading for visual depth
    min_bed = np.min(bed) if len(bed) > 0 else 0
    bed_shading = np.full_like(combined_x, min_bed - 0.5)
    ax.fill_between(combined_x, bed, bed_shading, 
                   color='#8B7355', alpha=0.7, 
                   hatch='///', linewidth=0)
    
    # Create water depth and other parameters arrays for coloring
    water_depth = np.zeros_like(combined_x)
    froude_numbers = np.zeros_like(combined_x)
    velocities = np.zeros_like(combined_x)
    
    for i, x in enumerate(combined_x):
        # Calculate water depth (ensure non-negative)
        water_depth[i] = max(0, water_surface[i] - bed[i])
        
        if x > 0:  # Downstream of dam
            # Interpolate values from tailwater
            idx = np.argmin(np.abs(x_values - x))
            
            # Check array bounds before accessing
            if idx < len(tailwater.get('fr', [])):
                froude_numbers[i] = tailwater['fr'][idx]
            
            if idx < len(tailwater.get('v', [])):
                velocities[i] = tailwater['v'][idx]
            
            # If arrays don't exist or index is out of bounds, calculate approximate values
            if froude_numbers[i] == 0 and water_depth[i] > 0 and velocities[i] > 0:
                # Approximate Froude number from velocity and depth
                froude_numbers[i] = velocities[i] / np.sqrt(9.81 * max(water_depth[i], 0.001))
            
            if velocities[i] == 0 and water_depth[i] > 0 and results.get('discharge', 0) > 0:
                # Approximate velocity from discharge and cross-sectional area
                # For trapezoidal channel
                bottom_width = scenario.get('channel_bottom_width', 5.0)
                side_slope = scenario.get('channel_side_slope', 0)
                
                # Calculate top width
                top_width = bottom_width + 2 * side_slope * water_depth[i]
                
                # Calculate area
                area = 0.5 * (bottom_width + top_width) * water_depth[i]
                
                # Ensure non-zero area to avoid division by zero
                area = max(area, 0.001)
                
                velocities[i] = results.get('discharge', 0) / area
                
                # Update Froude number with new velocity
                if froude_numbers[i] == 0:
                    # Use hydraulic depth (A/T) for Froude number calculation
                    hydraulic_depth = area / top_width if top_width > 0 else water_depth[i]
                    froude_numbers[i] = velocities[i] / np.sqrt(9.81 * max(hydraulic_depth, 0.001))
                
        elif x <= 0 and water_depth[i] > 0:  # Upstream of dam with water
            # Approximate upstream values using physically correct calculations
            # Get channel parameters
            channel_width = scenario.get('channel_width_at_dam', 5.0)
            side_slope = scenario.get('channel_side_slope', 0)
            
            # Calculate top width for trapezoidal section
            top_width = channel_width + 2 * side_slope * water_depth[i]
            
            # Calculate cross-sectional area
            area = 0.5 * (channel_width + top_width) * water_depth[i]
            
            # Ensure non-zero area to avoid division by zero
            area = max(area, 0.001)
            
            # Calculate velocity
            velocities[i] = results.get('discharge', 0) / area
            
            # Calculate Froude number using hydraulic depth
            hydraulic_depth = area / top_width if top_width > 0 else water_depth[i]
            froude_numbers[i] = velocities[i] / np.sqrt(9.81 * max(hydraulic_depth, 0.001))
    
    # Determine coloring parameter based on user choice
    if color_by.lower() == 'froude':
        color_param = froude_numbers
        cmap = plt.cm.coolwarm
        # Ensure reasonable values for TwoSlopeNorm
        fr_max = max(2, np.max(froude_numbers)) if len(froude_numbers) > 0 else 2
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=fr_max)
        cbar_label = 'Froude Number'
    elif color_by.lower() == 'velocity':
        color_param = velocities
        cmap = plt.cm.viridis
        # Ensure reasonable values for Normalize
        v_max = max(1, np.max(velocities)) if len(velocities) > 0 else 1
        norm = colors.Normalize(vmin=0, vmax=v_max)
        cbar_label = 'Velocity (m/s)'
    elif color_by.lower() == 'depth':
        color_param = water_depth
        cmap = plt.cm.Blues
        # Ensure reasonable values for Normalize
        d_max = max(1, np.max(water_depth)) if len(water_depth) > 0 else 1
        norm = colors.Normalize(vmin=0, vmax=d_max)
        cbar_label = 'Water Depth (m)'
    else:  # Default to Froude
        color_param = froude_numbers
        cmap = plt.cm.coolwarm
        # Ensure reasonable values for TwoSlopeNorm
        fr_max = max(2, np.max(froude_numbers)) if len(froude_numbers) > 0 else 2
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=fr_max)
        cbar_label = 'Froude Number'
    
    # Check if we have enough points to create a LineCollection
    if len(combined_x) > 1:
        # Plot enhanced water surface with coloring
        points = np.array([combined_x, water_surface]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a LineCollection with the specified colormap
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
        
        if len(color_param) > 0 and len(color_param) >= len(segments):
            lc.set_array(color_param[:-1])
            line = ax.add_collection(lc)
            
            # Add water body with gradient coloring
            for i in range(len(combined_x)-1):
                # Skip if both points are at or below bed
                if water_surface[i] <= bed[i] and water_surface[i+1] <= bed[i+1]:
                    continue
                
                # Create polygon for each segment
                x_vals = [combined_x[i], combined_x[i+1], combined_x[i+1], combined_x[i]]
                
                # Ensure water surface is at or above bed level at each point
                y_surface_i = max(water_surface[i], bed[i])
                y_surface_i_plus_1 = max(water_surface[i+1], bed[i+1])
                
                y_vals = [y_surface_i, y_surface_i_plus_1, bed[i+1], bed[i]]
                
                # Skip invalid polygons - need at least 3 distinct points to form polygon
                if len(set(zip(x_vals, y_vals))) < 3:
                    continue
                
                # Ensure safe indexing for color_param
                if i < len(color_param):
                    color_val = color_param[i]
                    rgba_color = cmap(norm(color_val))
                else:
                    # Use default color if index out of range
                    rgba_color = cmap(norm(0))
                
                # Add transparency based on depth for visual effect
                max_depth = np.max(water_depth) if len(water_depth) > 0 else 1
                depth_factor = min(1, water_depth[i] / max(0.1, max_depth))
                rgba_color = list(rgba_color)
                rgba_color[3] = 0.3 + 0.4 * depth_factor  # Alpha between 0.3 and 0.7
                
                # Create and add polygon
                poly = Polygon(np.column_stack([x_vals, y_vals]), 
                               facecolor=rgba_color, edgecolor=None)
                ax.add_patch(poly)
            
            # Add color bar for the water surface coloring
            cbar = plt.colorbar(line, ax=ax)
            cbar.set_label(cbar_label)
            
            # If coloring by Froude number, add regime labels
            if color_by.lower() == 'froude':
                # Add regime labels to the color bar
                cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                            rotation=90, transform=cbar.ax.transAxes, color='white')
                cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                            rotation=90, transform=cbar.ax.transAxes, color='white')
                cbar.ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Fill the dam body with enhanced visualization
    try:
        # Create dam polygon safely
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            
            # Create a dam visualization with shading effects
            dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                               edgecolor='black', linewidth=0.5)
            
            # Add dam patch to axis
            ax.add_patch(dam_patch)
    except Exception as e:
        # If dam visualization fails, at least draw a simple dam shape
        dam_x = [0, 0]
        dam_y = [dam_base, results.get('dam_crest_elevation', dam_base + 10)]
        ax.plot(dam_x, dam_y, 'k-', linewidth=2)
    
    # Plot reference lines with enhanced styling
    yn = results.get('normal_depth', 0)
    yc = results.get('critical_depth', 0)
    
    # Only draw reference lines if values are positive
    if yn > 0:
        # Normal depth line - use base elevation from tailwater if available
        if len(tailwater.get('z', [])) > 0:
            base_elevation = tailwater['z'][0]
        else:
            base_elevation = dam_base
            
        ax.axhline(y=base_elevation + yn, color='green', linestyle='--', 
                   alpha=0.7, linewidth=1.5, label=f'Normal Depth ({yn:.2f} m)')
    
    if yc > 0:
        # Critical depth line
        ax.axhline(y=base_elevation + yc, color='red', linestyle=':', 
                   alpha=0.7, linewidth=1.5, label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_idx = -1
        min_dist = float('inf')
        for i, x in enumerate(x_values):
            dist = abs(x - jump_loc)
            if dist < min_dist:
                min_dist = dist
                jump_idx = i
        
        if jump_idx >= 0 and jump_idx < len(tailwater.get('z', [])):
            jump_z = tailwater['z'][jump_idx]
            
            # Plot the jump with enhanced styling
            ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                      alpha=0.8, linewidth=2, label='Hydraulic Jump')
            
            # Get hydraulic jump parameters - handle missing keys gracefully
            y1 = jump.get('initial_depth', 0.1)
            y2 = jump.get('sequent_depth', y1 * 2)
            
            # Ensure physically realistic jump (y2 > y1)
            if y2 <= y1:
                # Calculate sequent depth using momentum conservation if not provided correctly
                # Sequent depth formula: y2/y1 = 0.5 * (-1 + sqrt(1 + 8*Fr₁²))
                if 'initial_froude' in jump and jump['initial_froude'] > 1:
                    fr1 = jump['initial_froude']
                    depth_ratio = 0.5 * (-1 + np.sqrt(1 + 8 * fr1**2))
                    y2 = y1 * depth_ratio
                else:
                    # Fallback: assume transitional jump with y2 = 2*y1
                    y2 = y1 * 2
            
            # Create more physically accurate hydraulic jump visualization
            x_jump = np.linspace(jump_loc-2, jump_loc+8, 20)
            y_jump = np.zeros_like(x_jump)
            
            for i, x in enumerate(x_jump):
                rel_pos = (x - (jump_loc-2)) / 10  # 0 to 1 across the jump
                
                # Create more physically accurate water surface profile through jump
                if rel_pos < 0.3:
                    # Pre-jump (supercritical)
                    y_jump[i] = y1
                elif rel_pos < 0.5:
                    # Initial roller - water level rises rapidly
                    t = (rel_pos - 0.3) / 0.2  # 0 to 1 in transition
                    # Non-linear transition for more realism
                    y_jump[i] = y1 + (y2 - y1) * (1 - np.cos(t * np.pi/2))
                elif rel_pos < 0.7:
                    # Turbulent zone - fluctuations around sequent depth
                    t = (rel_pos - 0.5) / 0.2  # 0 to 1 in transition
                    # Add small oscillations for turbulence visualization
                    oscillation = 0.1 * y2 * np.sin(t * 3 * np.pi)
                    y_jump[i] = y2 + oscillation
                else:
                    # Post-jump (subcritical)
                    y_jump[i] = y2
            
            # Add jump shading with enhanced colors
            for i in range(len(x_jump)-1):
                # Create polygon for turbulent water
                x_vals = [x_jump[i], x_jump[i+1], x_jump[i+1], x_jump[i]]
                y_vals = [jump_z + y_jump[i], jump_z + y_jump[i+1], jump_z, jump_z]
                
                # Distance from jump center affects color intensity
                dist_from_jump = abs((x_jump[i] + x_jump[i+1])/2 - jump_loc)
                intensity = max(0, 1 - dist_from_jump/8)
                
                # Create polygon
                poly = Polygon(np.column_stack([x_vals, y_vals]), 
                              facecolor='magenta', alpha=0.2 + 0.4 * intensity)
                ax.add_patch(poly)
            
            # Add textual annotation for jump details
            if show_annotations and 'initial_froude' in jump and 'energy_loss' in jump:
                # Enhanced annotation with background and outline for better visibility
                txt = ax.annotate(
                    f"Hydraulic Jump\n"
                    f"Type: {jump.get('jump_type', 'Unknown')}\n"
                    f"Fr₁ = {jump.get('initial_froude', 0):.2f}\n"
                    f"y₁ = {y1:.2f} m → y₂ = {y2:.2f} m\n"
                    f"E.Loss = {jump.get('energy_loss', 0):.2f} m",
                    xy=(jump_loc, jump_z + y2),
                    xytext=(jump_loc+20, jump_z + y2 + 1),
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8,
                             edgecolor='gray', linewidth=1),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3",
                                   color='gray'),
                    fontsize=9
                )
                # Add outline to text for better readability
                import matplotlib.patheffects as PathEffects
                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
    
    # Add discharge information with enhanced styling
    if show_annotations:
        # Add a structured information box in the upper left
        discharge = results.get('discharge', 0)
        head = results.get('head', 0)
        velocity = results.get('velocity', 0)
        
        # Get energy dissipation safely
        energy_results = results.get('energy_results', {})
        dissipation_ratio = energy_results.get('dissipation_ratio', 0) * 100 if isinstance(energy_results, dict) else 0
        
        info_text = (
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Head: {head:.2f} m\n"
            f"Velocity at dam: {velocity:.2f} m/s\n"
            f"Energy dissipation: {dissipation_ratio:.1f}%"
        )
        
        # Position the info box based on the view range - safely handle empty arrays
        info_x = combined_x[0] + 10 if len(combined_x) > 0 else 0
        info_y = np.max(water_surface) - 1 if len(water_surface) > 0 else 0
        
        ax.text(info_x, info_y, info_text,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8,
                        edgecolor='blue', linewidth=1),
               fontsize=10, color='black')
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range - safely handle empty arrays
        if len(x_values) > 0:
            ax.set_xlim(-20, min(200, np.max(x_values)))
        else:
            ax.set_xlim(-20, 200)
    
    # Calculate y limits to include important elements with some padding
    if len(bed) > 0 and len(water_surface) > 0:
        y_min = min(np.min(bed) - 1, np.min(water_surface) - 1)
        y_max = max(np.max(bed) + 2, np.max(water_surface) + 2)
    else:
        y_min = dam_base - 1
        y_max = dam_base + 10
    
    ax.set_ylim(y_min, y_max)
    
    # Add enhanced title and labels
    ax.set_title('Enhanced Hydraulic Profile Visualization', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Dam (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    
    # Add grid with enhanced styling
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend with better styling
    legend = ax.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig, ax


def create_profile_with_parameter_plots(scenario, results, parameters=None, figsize=(14, 10)):
    """
    Create a multi-panel figure with the main profile and additional parameter plots.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        parameters (list): List of parameters to plot 
                          ('velocity', 'froude', 'depth', 'energy', 'shear')
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axs) The figure and axes objects
    """
    if parameters is None:
        parameters = ['velocity', 'froude']
    
    # Determine number of panels
    n_panels = 1 + len(parameters)
    
    # Create figure with subplots
    fig, axs = plt.subplots(n_panels, 1, figsize=figsize, sharex=True, 
                           gridspec_kw={'height_ratios': [3] + [1] * len(parameters)})
    
    # If only one parameter, ensure axs is a list
    if n_panels == 2:
        axs = [axs[0], axs[1]]
    
    # Plot main profile in the first panel
    plot_enhanced_profile(scenario, results, ax=axs[0])
    
    # Extract x-coordinates from results for parameter plots
    tailwater = results.get('tailwater', {})
    x_values = tailwater.get('x', [])
    
    # Check if we have data
    if len(x_values) == 0:
        # Not enough data, add a warning and return with just the profile
        for i in range(1, n_panels):
            axs[i].text(0.5, 0.5, "Insufficient data for parameter plots", 
                       ha='center', va='center', transform=axs[i].transAxes)
            axs[i].set_ylabel("N/A")
        
        # Set common x-axis label
        axs[-1].set_xlabel('Distance from Dam (m)')
        plt.tight_layout()
        return fig, axs
    
    # Plot additional parameters
    panel_idx = 1
    for param in parameters:
        if param.lower() == 'velocity':
            # Plot velocity profile if data exists
            if 'v' in tailwater and len(tailwater['v']) > 0:
                # Ensure velocities array matches x_values length
                if len(tailwater['v']) == len(x_values):
                    axs[panel_idx].plot(x_values, tailwater['v'], 'b-', linewidth=2)
                else:
                    # Create subset for plotting if arrays don't match
                    plot_length = min(len(x_values), len(tailwater['v']))
                    axs[panel_idx].plot(x_values[:plot_length], tailwater['v'][:plot_length], 'b-', linewidth=2)
            else:
                # No velocity data, create empty plot
                axs[panel_idx].text(0.5, 0.5, "No velocity data available", 
                                   ha='center', va='center', transform=axs[panel_idx].transAxes)
            
            axs[panel_idx].set_ylabel('Velocity (m/s)')
            
            # Add dam velocity reference if available
            if 'velocity' in results:
                axs[panel_idx].axhline(y=results['velocity'], color='r', linestyle='--', 
                                      label=f'Dam Crest Velocity ({results.get("velocity", 0):.2f} m/s)')
            
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'froude':
            # Plot Froude number profile if data exists
            if 'fr' in tailwater and len(tailwater['fr']) > 0:
                # Ensure Fr array matches x_values length
                if len(tailwater['fr']) == len(x_values):
                    fr = tailwater['fr']
                    axs[panel_idx].plot(x_values, fr, 'r-', linewidth=2)
                    
                    # Color regions based on flow regime
                    axs[panel_idx].fill_between(x_values, fr, 0, where=(fr<1), color='blue', alpha=0.2, label='Subcritical')
                    axs[panel_idx].fill_between(x_values, fr, 1, where=(fr>1), color='red', alpha=0.2, label='Supercritical')
                else:
                    # Create subset for plotting if arrays don't match
                    plot_length = min(len(x_values), len(tailwater['fr']))
                    fr = tailwater['fr'][:plot_length]
                    subset_x = x_values[:plot_length]
                    
                    axs[panel_idx].plot(subset_x, fr, 'r-', linewidth=2)
                    axs[panel_idx].fill_between(subset_x, fr, 0, where=(fr<1), color='blue', alpha=0.2, label='Subcritical')
                    axs[panel_idx].fill_between(subset_x, fr, 1, where=(fr>1), color='red', alpha=0.2, label='Supercritical')
            else:
                # No Froude data, create empty plot
                axs[panel_idx].text(0.5, 0.5, "No Froude number data available", 
                                   ha='center', va='center', transform=axs[panel_idx].transAxes)
            
            axs[panel_idx].set_ylabel('Froude Number')
            axs[panel_idx].axhline(y=1, color='k', linestyle='--', label='Critical Flow (Fr=1)')
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'depth':
            # Plot water depth profile if data exists
            if 'y' in tailwater and len(tailwater['y']) > 0:
                # Ensure depth array matches x_values length
                if len(tailwater['y']) == len(x_values):
                    depth = tailwater['y']
                    axs[panel_idx].plot(x_values, depth, 'g-', linewidth=2)
                else:
                    # Create subset for plotting if arrays don't match
                    plot_length = min(len(x_values), len(tailwater['y']))
                    depth = tailwater['y'][:plot_length]
                    subset_x = x_values[:plot_length]
                    axs[panel_idx].plot(subset_x, depth, 'g-', linewidth=2)
            else:
                # No depth data, create empty plot
                axs[panel_idx].text(0.5, 0.5, "No depth data available", 
                                   ha='center', va='center', transform=axs[panel_idx].transAxes)
            
            axs[panel_idx].set_ylabel('Water Depth (m)')
            
            # Add normal and critical depth references if available
            if 'normal_depth' in results and results['normal_depth'] > 0:
                axs[panel_idx].axhline(y=results['normal_depth'], color='g', linestyle='--', 
                                     label=f'Normal Depth ({results.get("normal_depth", 0):.2f} m)')
            
            if 'critical_depth' in results and results['critical_depth'] > 0:
                axs[panel_idx].axhline(y=results['critical_depth'], color='r', linestyle=':', 
                                     label=f'Critical Depth ({results.get("critical_depth", 0):.2f} m)')
            
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'energy':
            # Plot energy grade line if data exists
            if 'energy' in tailwater and 'z' in tailwater and len(tailwater['energy']) > 0 and len(tailwater['z']) > 0:
                # Ensure arrays match x_values length
                if len(tailwater['energy']) == len(x_values) and len(tailwater['z']) == len(x_values):
                    energy = tailwater['energy']
                    bed = tailwater['z']
                    axs[panel_idx].plot(x_values, energy - bed, 'purple', linewidth=2, label='Specific Energy')
                else:
                    # Create subset for plotting if arrays don't match
                    plot_length = min(len(x_values), len(tailwater['energy']), len(tailwater['z']))
                    energy = tailwater['energy'][:plot_length]
                    bed = tailwater['z'][:plot_length]
                    subset_x = x_values[:plot_length]
                    axs[panel_idx].plot(subset_x, energy - bed, 'purple', linewidth=2, label='Specific Energy')
            else:
                # No energy data, create empty plot
                axs[panel_idx].text(0.5, 0.5, "No energy data available", 
                                   ha='center', va='center', transform=axs[panel_idx].transAxes)
            
            axs[panel_idx].set_ylabel('Specific Energy (m)')
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
        
        elif param.lower() == 'shear':
            # Calculate and plot shear stress using hydraulic radius for more accuracy
            if 'y' in tailwater and 'z' in tailwater and len(tailwater['y']) > 0:
                # Get required data
                depth = tailwater.get('y', [])
                channel_bottom_width = scenario.get('channel_bottom_width', 5.0)
                channel_side_slope = scenario.get('channel_side_slope', 0)
                
                # Get energy slope either from data or use bed slope as approximation
                if 's' in tailwater and len(tailwater['s']) > 0:
                    slope = tailwater['s']
                else:
                    slope = np.full_like(depth, scenario.get('downstream_slope', 0.001))
                
                # Physical constants
                rho = 1000  # Water density (kg/m³)
                g = 9.81    # Gravity (m/s²)
                
                # Calculate hydraulic radius for each point
                hydraulic_radius = np.zeros_like(depth)
                shear_stress = np.zeros_like(depth)
                
                # Calculate for each point
                for i, d in enumerate(depth):
                    if d > 0:
                        # Calculate wetted perimeter and area
                        if channel_side_slope > 0:
                            # Trapezoidal channel
                            # Top width = bottom width + 2 * side_slope * depth
                            top_width = channel_bottom_width + 2 * channel_side_slope * d
                            
                            # Area = (bottom width + top width) * depth / 2
                            area = (channel_bottom_width + top_width) * d / 2
                            
                            # Wetted perimeter = bottom width + 2 * sloped sides
                            sloped_sides = d * np.sqrt(1 + channel_side_slope**2)
                            wetted_perimeter = channel_bottom_width + 2 * sloped_sides
                        else:
                            # Rectangular channel
                            area = channel_bottom_width * d
                            wetted_perimeter = channel_bottom_width + 2 * d
                        
                        # Ensure non-zero wetted perimeter
                        wetted_perimeter = max(wetted_perimeter, 0.001)
                        
                        # Calculate hydraulic radius
                        hydraulic_radius[i] = area / wetted_perimeter
                        
                        # Calculate shear stress using τ = ρgRS where R is hydraulic radius
                        # Use correct index for slope if available
                        s = slope[i] if i < len(slope) else scenario.get('downstream_slope', 0.001)
                        shear_stress[i] = rho * g * hydraulic_radius[i] * s
                
                # Plot shear stress
                # Ensure arrays match x_values length
                if len(shear_stress) == len(x_values):
                    axs[panel_idx].plot(x_values, shear_stress, 'brown', linewidth=2)
                else:
                    # Create subset for plotting if arrays don't match
                    plot_length = min(len(x_values), len(shear_stress))
                    axs[panel_idx].plot(x_values[:plot_length], shear_stress[:plot_length], 'brown', linewidth=2)
            else:
                # No data for shear calculation, create empty plot
                axs[panel_idx].text(0.5, 0.5, "Insufficient data for shear stress calculation", 
                                   ha='center', va='center', transform=axs[panel_idx].transAxes)
            
            axs[panel_idx].set_ylabel('Shear Stress (N/m²)')
            axs[panel_idx].grid(True, alpha=0.3)
        
        # Increment panel index
        panel_idx += 1
    
    # Set common x-axis label
    axs[-1].set_xlabel('Distance from Dam (m)')
    
    # Mark hydraulic jump location across all panels if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        for ax in axs:
            ax.axvline(x=jump_loc, color='m', linestyle='-.', alpha=0.7)
    
    # Add title to the figure
    fig.suptitle('Hydraulic Profile with Parameter Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig, axs


def plot_enhanced_profile_with_cross_sections(scenario, results, x_locations=None):
    """
    Create a figure with the main profile and multiple cross-section views.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        x_locations (list): List of x locations for cross-sections
        
    Returns:
        tuple: (fig, axs) The figure and axes objects
    """
    # If no x_locations provided, choose some reasonable defaults
    if x_locations is None:
        # Choose upstream, at dam, just downstream, hydraulic jump (if present), and far downstream
        x_locations = [-10, 0, 20]
        
        # Add hydraulic jump location if present
        jump = results.get('hydraulic_jump', {})
        if jump.get('jump_possible', False) and 'location' in jump:
            x_locations.append(jump['location'])
        
        # Get maximum x if available
        tailwater = results.get('tailwater', {})
        if 'x' in tailwater and len(tailwater['x']) > 0:
            max_x = min(200, np.max(tailwater['x']))
            x_locations.append(max_x)
        else:
            # Default far downstream location
            x_locations.append(100)
    
    # Determine how many cross-sections we'll show
    n_sections = len(x_locations)
    
    # Create the figure with a main profile plot and cross-section subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Define grid spec to organize the layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, n_sections, height_ratios=[2, 1], figure=fig)
    
    # Create main profile axis
    ax_profile = fig.add_subplot(gs[0, :])
    
    # Plot the main enhanced profile
    plot_enhanced_profile(scenario, results, ax=ax_profile)
    
    # Create cross-section axes
    ax_sections = []
    for i in range(n_sections):
        ax_sections.append(fig.add_subplot(gs[1, i]))
    
    # Extract necessary data from scenario and results
    dam_base = scenario.get('dam_base_elevation', 0)
    dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
    tailwater = results.get('tailwater', {})
    channel_side_slope = scenario.get('channel_side_slope', 1.5)
    channel_bottom_width = scenario.get('channel_bottom_width', 5.0)
    
    # Plot cross-sections
    for i, x_loc in enumerate(x_locations):
        ax = ax_sections[i]
        
        # Determine water depth and bed elevation at this location
        if x_loc <= 0:  # Upstream of dam
            water_elevation = results.get('upstream_level', dam_base)
            bed_elevation = dam_base
        else:  # Downstream of dam
            # Check if tailwater data exists
            if 'x' in tailwater and 'wse' in tailwater and 'z' in tailwater:
                # Find closest point in tailwater results
                x_values = np.array(tailwater['x'])
                if len(x_values) > 0:
                    idx = np.argmin(np.abs(x_values - x_loc))
                    if idx < len(tailwater['wse']) and idx < len(tailwater['z']):
                        water_elevation = tailwater['wse'][idx]
                        bed_elevation = tailwater['z'][idx]
                    else:
                        water_elevation = dam_base
                        bed_elevation = dam_base
                else:
                    water_elevation = dam_base
                    bed_elevation = dam_base
            else:
                water_elevation = dam_base
                bed_elevation = dam_base
        
        water_depth = max(0, water_elevation - bed_elevation)
        
        # Calculate cross-section geometry (trapezoidal channel)
        half_bottom = channel_bottom_width / 2
        if water_depth > 0:
            water_width = channel_bottom_width + 2 * channel_side_slope * water_depth
            half_water = water_width / 2
        else:
            water_width = channel_bottom_width
            half_water = half_bottom
        
        # Plot channel cross-section
        # For trapezoidal channel, the bottom is horizontal and sides slope upward
        bottom_y = np.array([0, 0])
        bottom_x = np.array([-half_bottom, half_bottom])
        
        left_bank_x = np.array([-half_bottom, -half_water])
        right_bank_x = np.array([half_bottom, half_water])
        
        bank_y = np.array([0, water_depth])
        
        # Plot channel bed (bottom)
        ax.plot(bottom_x, bottom_y, 'k-', linewidth=1.5)
        
        # Plot banks
        if water_depth > 0:
            ax.plot(left_bank_x, bank_y, 'k-', linewidth=1.5)
            ax.plot(right_bank_x, bank_y, 'k-', linewidth=1.5)
            
            # Plot water surface
            ax.plot([-half_water, half_water], [water_depth, water_depth], 'b-', linewidth=1.5)
        
        # Fill the channel bed
        ax.fill_between([-half_bottom-1, half_bottom+1], 0, -0.5, color='#8B7355', hatch='///')
        
        # Fill water area if there's water
        if water_depth > 0:
            # For trapezoidal channel
            water_x = np.array([-half_water, -half_bottom, half_bottom, half_water])
            water_y = np.array([water_depth, 0, 0, water_depth])
            
            # Create and add polygon
            water_poly = Polygon(np.column_stack([water_x, water_y]), 
                               facecolor='skyblue', alpha=0.5)
            ax.add_patch(water_poly)
        
        # Add annotations for dimensions
        ax.annotate(f"{channel_bottom_width:.1f}m", 
                   xy=(0, -0.05), 
                   xytext=(0, -0.2),
                   ha='center', va='top',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if water_depth > 0:
            # Annotate water depth
            ax.annotate(f"{water_depth:.2f}m", 
                       xy=(half_water/2, water_depth/2), 
                       xytext=(half_water/2 + 0.5, water_depth/2),
                       ha='left', va='center',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Annotate water width
            ax.annotate(f"{water_width:.1f}m", 
                       xy=(0, water_depth + 0.05), 
                       xytext=(0, water_depth + 0.2),
                       ha='center', va='bottom',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Set title with location
        if x_loc <= 0:
            title = f"Upstream (x={x_loc}m)"
        elif abs(x_loc) < 0.1:
            title = "At Dam (x=0m)"
        else:
            title = f"Downstream (x={x_loc}m)"
            
        # Add Froude number to title if downstream
        if x_loc > 0 and 'x' in tailwater and 'fr' in tailwater:
            x_values = np.array(tailwater['x'])
            if len(x_values) > 0:
                idx = np.argmin(np.abs(x_values - x_loc))
                if idx < len(tailwater['fr']):
                    fr = tailwater['fr'][idx]
                    regime = "Subcritical" if fr < 1 else "Supercritical" if fr > 1 else "Critical"
                    title += f"\n{regime} (Fr={fr:.2f})"
        
        ax.set_title(title)
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set y-axis label only for the first cross-section
        if i == 0:
            ax.set_ylabel('Elevation (m)')
        
        # Set reasonable y-limits with some padding
        max_height = max(water_depth * 1.3, 1)
        ax.set_ylim(-0.5, max_height)
        
        # Make sure x-axis is centered and has reasonable limits
        max_width = max(half_water, half_bottom) * 1.3
        ax.set_xlim(-max_width, max_width)
        
        # Add marker line in the main profile view
        ax_profile.axvline(x=x_loc, color='gray', linestyle='--', alpha=0.5)
        
        # Add text marker
        marker_y = ax_profile.get_ylim()[1] - 0.05 * (ax_profile.get_ylim()[1] - ax_profile.get_ylim()[0])
        ax_profile.annotate(f"{chr(65+i)}", xy=(x_loc, marker_y), ha='center', va='center',
                           bbox=dict(boxstyle="circle", fc="white", ec="black"))
        
        # Add corresponding label to the cross-section
        ax.annotate(f"{chr(65+i)}", xy=(0.05, 0.95), xycoords='axes fraction',
                   ha='left', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="circle", fc="white", ec="black"))
    
    # Add overall title
    fig.suptitle('Hydraulic Profile with Cross-Sections', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig, (ax_profile, ax_sections)