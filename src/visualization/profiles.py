"""
Visualization utilities for dam and channel flow.

This module provides functions for creating high-quality visualizations
of dam and channel flow characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

def plot_full_profile(scenario, results, display_range=None, show_jump=True, show_annotations=True):
    """
    Create a comprehensive plot of the water surface profile.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        display_range (tuple): The x-range to display (min, max)
        show_jump (bool): Whether to highlight the hydraulic jump
        show_annotations (bool): Whether to show detailed annotations
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
    
    # Plot bed and water surface
    ax.plot(combined_x, bed, 'k-', linewidth=2, label='Channel Bed')
    ax.plot(combined_x, water_surface, 'b-', linewidth=2, label='Water Surface')
    
    # Fill the water body with light blue
    # Make sure to handle cases where arrays might have different shapes
    try:
        water_poly = np.column_stack([
            np.concatenate([combined_x, combined_x[::-1]]),
            np.concatenate([water_surface, bed[::-1]])
        ])
        water_patch = Polygon(water_poly, closed=True, alpha=0.3, color='skyblue')
        ax.add_patch(water_patch)
    except Exception as e:
        # If polygon creation fails, skip it and log the error
        print(f"Warning: Could not create water polygon: {str(e)}")
    
    # Fill the dam body with dark gray
    try:
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, alpha=1.0, color='dimgray')
            ax.add_patch(dam_patch)
    except Exception as e:
        # If dam polygon creation fails, draw a simple dam shape
        print(f"Warning: Could not create dam polygon: {str(e)}")
        ax.fill_between([0, 0], [dam_base, scenario.get('dam_crest_elevation', dam_base + 10)], 
                       color='dimgray', alpha=1.0)
    
    # Plot reference lines - only if values are positive
    yn = results.get('normal_depth', 0)
    yc = results.get('critical_depth', 0)
    
    # Get base elevation from tailwater if available
    if len(tailwater.get('z', [])) > 0:
        base_elevation = tailwater['z'][0]
    else:
        base_elevation = dam_base
    
    # Normal depth line - only if positive
    if yn > 0:
        ax.axhline(y=base_elevation + yn, color='g', linestyle='--', 
                 label=f'Normal Depth ({yn:.2f} m)')
    
    # Critical depth line - only if positive
    if yc > 0:
        ax.axhline(y=base_elevation + yc, color='r', linestyle=':', 
                 label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present and requested
    jump = results.get('hydraulic_jump', {})
    if show_jump and jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = -1
        min_dist = float('inf')
        for i, x in enumerate(x_values):
            dist = abs(x - jump_loc)
            if dist < min_dist:
                min_dist = dist
                jump_index = i
        
        if jump_index >= 0 and jump_index < len(tailwater.get('z', [])):
            jump_z = tailwater['z'][jump_index]
            
            # Plot the jump
            ax.axvline(x=jump_loc, color='m', linestyle='-.',
                     label='Hydraulic Jump')
            
            # Create a polygon for the jump
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
            
            # Create jump polygon with more realistic shape
            try:
                jump_poly = np.array([
                    [jump_loc-5, jump_z],
                    [jump_loc-5, jump_z + y1],
                    [jump_loc, jump_z + y2],
                    [jump_loc+5, jump_z + y2],
                    [jump_loc+5, jump_z]
                ])
                jump_patch = Polygon(jump_poly, closed=True, alpha=0.5, color='magenta')
                ax.add_patch(jump_patch)
            except Exception as e:
                # If polygon creation fails, use a simple rectangle
                print(f"Warning: Could not create jump polygon: {str(e)}")
                ax.axvspan(jump_loc-5, jump_loc+5, ymin=jump_z, ymax=jump_z+y2, 
                          alpha=0.3, color='magenta')
            
            # Add annotation for jump details
            if show_annotations and 'initial_froude' in jump:
                jump_type = jump.get('jump_type', 'Unknown')
                net_force = jump.get('net_force', 0)
                
                # Create annotation text with safe values
                annotation_text = f"Hydraulic Jump\nType: {jump_type}\nFr₁ = {jump.get('initial_froude', 0):.2f}"
                
                # Only add depth info if we have valid depths
                if y1 > 0 and y2 > 0:
                    annotation_text += f"\ny₁ = {y1:.2f} m\ny₂ = {y2:.2f} m"
                
                # Only add force if provided and non-zero
                if net_force != 0:
                    annotation_text += f"\nForce = {net_force:.2f} N/m"
                
                ax.annotate(
                    annotation_text,
                    xy=(jump_loc, jump_z + y2),
                    xytext=(jump_loc+30, jump_z + y2 + 0.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
                )
    
    # Add discharge information
    if show_annotations:
        # Get values safely
        discharge = results.get('discharge', 0)
        head = results.get('head', 0)
        velocity = results.get('velocity', 0)
        
        # Only create annotation if we have a valid upstream level
        upstream_level = results.get('upstream_level', dam_base)
        
        ax.annotate(
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Head: {head:.2f} m\n"
            f"Velocity: {velocity:.2f} m/s",
            xy=(combined_x[0] if len(combined_x) > 0 else -20, upstream_level),
            xytext=(combined_x[0] + 20 if len(combined_x) > 0 else 0, upstream_level + 1),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
        
        # Add energy information - safely access nested dictionary
        energy = results.get('energy_results', {})
        if isinstance(energy, dict):
            energy_loss = energy.get('energy_loss', 0)
            dissipation_ratio = energy.get('dissipation_ratio', 0)
            power = energy.get('power', 0)
            
            ax.annotate(
                f"Energy Dissipation\n"
                f"Loss: {energy_loss:.2f} m\n"
                f"Ratio: {dissipation_ratio*100:.1f}%\n"
                f"Power: {power/1000:.2f} kW",
                xy=(5, dam_base + 3),
                xytext=(10, dam_base + 6),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
            )
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range - handle case of empty x_values
        if len(x_values) > 0:
            ax.set_xlim(-20, min(200, np.max(x_values)))
        else:
            ax.set_xlim(-20, 200)
    
    # Calculate y limits to include important elements
    if len(bed) > 0 and len(water_surface) > 0:
        y_min = min(np.min(bed) - 1, np.min(water_surface) - 1)
        y_max = max(np.max(bed) + 2, np.max(water_surface) + 2)
    else:
        # Default values if arrays are empty
        y_min = dam_base - 1
        y_max = dam_base + 10
    
    ax.set_ylim(y_min, y_max)
    
    # Add title and labels
    ax.set_title('Dam Flow Analysis: Water Surface Profile')
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Elevation (m)')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig, ax

def plot_velocity_profile(scenario, results):
    """
    Create a plot showing velocity distribution in the system.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data safely
    tailwater = results.get('tailwater', {})
    discharge = results.get('discharge', 0)
    
    # Get x and velocity values from tailwater
    x_values = tailwater.get('x', [])
    velocities = tailwater.get('v', [])
    
    # Check if we have data to plot
    if len(x_values) == 0 or len(velocities) == 0:
        ax.text(0.5, 0.5, "No velocity data available", 
               ha='center', va='center', transform=ax.transAxes)
        
        # Set basic axes properties
        ax.set_title('Flow Velocity Distribution')
        ax.set_xlabel('Distance from Dam (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        return fig, ax
    
    # Ensure arrays have matching length
    if len(x_values) != len(velocities):
        # Truncate to the shorter length
        plot_length = min(len(x_values), len(velocities))
        x_values = x_values[:plot_length]
        velocities = velocities[:plot_length]
    
    # Plot velocity profile
    ax.plot(x_values, velocities, 'r-', linewidth=2, label='Flow Velocity')
    
    # Add critical velocity line - handle potential division by zero
    critical_depth = results.get('critical_depth', 0)
    channel_width = scenario.get('channel_width_at_dam', 1.0)  # Default to 1.0 to avoid division by zero
    
    if critical_depth > 0 and channel_width > 0:
        critical_velocity = discharge / (critical_depth * channel_width)
        ax.axhline(y=critical_velocity, color='k', linestyle=':', 
                 label=f'Critical Velocity ({critical_velocity:.2f} m/s)')
    
    # Mark velocity at the dam
    dam_velocity = results.get('velocity', 0)
    if dam_velocity > 0:
        ax.plot(0, dam_velocity, 'ro', markersize=8,
              label=f'Dam Crest Velocity ({dam_velocity:.2f} m/s)')
    
    # Mark hydraulic jump if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        
        # Get jump parameters safely
        y1 = jump.get('initial_depth', 0.1)
        y2 = jump.get('sequent_depth', y1 * 2)
        
        # Ensure physically realistic jump (y2 > y1)
        if y2 <= y1 and 'initial_froude' in jump and jump['initial_froude'] > 1:
            # Calculate sequent depth using momentum conservation
            fr1 = jump['initial_froude']
            depth_ratio = 0.5 * (-1 + np.sqrt(1 + 8 * fr1**2))
            y2 = y1 * depth_ratio
        
        # Calculate velocities safely
        if y1 > 0 and y2 > 0 and channel_width > 0:
            v1 = discharge / (y1 * channel_width)
            v2 = discharge / (y2 * channel_width)
            
            # Mark jump on velocity profile
            ax.plot([jump_loc, jump_loc], [v1, v2], 'm-', linewidth=2)
            ax.plot(jump_loc, v1, 'mo', label=f'Before Jump ({v1:.2f} m/s)')
            ax.plot(jump_loc, v2, 'ms', label=f'After Jump ({v2:.2f} m/s)')
            
            # Annotate the velocity change
            velocity_change = v1 - v2
            percent_change = (velocity_change / v1 * 100) if v1 > 0 else 0
            
            ax.annotate(
                f"Velocity Reduction: {velocity_change:.2f} m/s\n"
                f"({percent_change:.1f}%)",
                xy=(jump_loc, (v1+v2)/2),
                xytext=(jump_loc+20, (v1+v2)/2),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
            )
    
    # Add title and labels
    ax.set_title('Flow Velocity Distribution')
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Velocity (m/s)')
    
    # Set axis limits
    if len(x_values) > 0:
        ax.set_xlim(0, min(200, np.max(x_values)))
    else:
        ax.set_xlim(0, 200)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig, ax

def plot_froude_profile(scenario, results):
    """
    Create a plot showing Froude number distribution in the system.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data safely
    tailwater = results.get('tailwater', {})
    
    # Get x and froude values from tailwater
    x_values = tailwater.get('x', [])
    froude_numbers = tailwater.get('fr', [])
    
    # Check if we have data to plot
    if len(x_values) == 0 or len(froude_numbers) == 0:
        ax.text(0.5, 0.5, "No Froude number data available", 
               ha='center', va='center', transform=ax.transAxes)
        
        # Set basic axes properties
        ax.set_title('Froude Number Distribution')
        ax.set_xlabel('Distance from Dam (m)')
        ax.set_ylabel('Froude Number')
        ax.set_xlim(0, 200)
        ax.grid(True, alpha=0.3)
        return fig, ax
    
    # Ensure arrays have matching length
    if len(x_values) != len(froude_numbers):
        # Truncate to the shorter length
        plot_length = min(len(x_values), len(froude_numbers))
        x_values = x_values[:plot_length]
        froude_numbers = froude_numbers[:plot_length]
    
    # Create a colormap for the flow regime
    cmap = plt.cm.coolwarm
    
    # Calculate max Froude safely
    if len(froude_numbers) > 0:
        max_fr = max(3, np.max(froude_numbers))  # Ensure vmax is at least 3
    else:
        max_fr = 3
    
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max_fr)
    
    # Plot Froude number profile with colored points
    sc = ax.scatter(x_values, froude_numbers, c=froude_numbers, 
                   cmap=cmap, norm=norm, s=30, edgecolor='k', linewidth=0.5)
    
    # Connect the points with a line
    ax.plot(x_values, froude_numbers, 'k-', alpha=0.3)
    
    # Add a color bar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Froude Number')
    
    # Add regime labels to the color bar
    cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                rotation=90, transform=cbar.ax.transAxes, color='k')
    cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                rotation=90, transform=cbar.ax.transAxes, color='k')
    
    # Add Fr=1 line
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Critical Flow (Fr=1)')
    
    # Mark hydraulic jump if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump and 'initial_froude' in jump:
        jump_loc = jump['location']
        initial_froude = jump['initial_froude']
        
        # Mark jump on Froude profile
        ax.plot(jump_loc, initial_froude, 'ro', markersize=8,
               label=f'Before Jump (Fr={initial_froude:.2f})')
        ax.axvline(x=jump_loc, color='m', linestyle='-.', alpha=0.7, label='Hydraulic Jump')
        
        # Annotate the jump
        jump_type = jump.get('jump_type', 'Unknown')
        
        ax.annotate(
            f"Hydraulic Jump\n"
            f"Fr₁ = {initial_froude:.2f}\n"
            f"Type: {jump_type}",
            xy=(jump_loc, initial_froude),
            xytext=(jump_loc+20, initial_froude+0.5),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
    
    # Add title and labels
    ax.set_title('Froude Number Distribution')
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Froude Number')
    
    # Set axis limits
    if len(x_values) > 0:
        ax.set_xlim(0, min(200, np.max(x_values)))
    else:
        ax.set_xlim(0, 200)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    return fig, ax

def create_flood_animation(scenario, water_levels, frames=50, interval=100):
    """
    Create an animation of increasing water levels during a flood.
    
    Parameters:
        scenario (dict): The scenario parameters
        water_levels (list): List of analysis results at different water levels
        frames (int): Number of frames in the animation
        interval (int): Interval between frames in milliseconds
        
    Returns:
        animation.FuncAnimation: The animation object
    """
    # Input validation
    if not water_levels:
        raise ValueError("water_levels list cannot be empty")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data safely
    dam = scenario.get('dam')
    if dam is None:
        raise ValueError("Scenario dictionary must contain 'dam' key")
        
    dam_base = scenario.get('dam_base_elevation', 0)
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile - handle if get_profile method doesn't exist
    try:
        dam_profile = dam.get_profile(x_dam)
    except (AttributeError, TypeError):
        # Create a simple dam profile if get_profile doesn't exist
        dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
        dam_profile = {'z': dam_base + (dam_crest - dam_base) * np.maximum(0, np.minimum(1, (x_dam + 20) / 20))}
    
    # Plot dam profile (static)
    if 'z' in dam_profile:
        dam_line, = ax.plot(x_dam, dam_profile['z'], 'k-', linewidth=2, label='Dam')
    else:
        # Fallback if no z values in dam profile
        dam_line, = ax.plot([0, 0], [dam_base, scenario.get('dam_crest_elevation', dam_base + 10)], 
                          'k-', linewidth=2, label='Dam')
    
    # Plot the dam body (static)
    try:
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, alpha=1.0, color='dimgray')
            ax.add_patch(dam_patch)
    except Exception as e:
        # If dam polygon creation fails, draw a simple dam shape
        print(f"Warning: Could not create dam polygon for animation: {str(e)}")
        ax.fill_between([0, 0], [dam_base, scenario.get('dam_crest_elevation', dam_base + 10)], 
                       [dam_base, dam_base], color='dimgray', alpha=1.0)
    
    # Create placeholder for dynamic elements
    water_line, = ax.plot([], [], 'b-', linewidth=2, label='Water Surface')
    flow_line, = ax.plot([], [], 'c-', linewidth=1.5, alpha=0.7, label='Flow')
    
    # Add text annotations
    level_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       bbox=dict(facecolor='white', alpha=0.8))
    
    discharge_text = ax.text(0.02, 0.87, '', transform=ax.transAxes,
                          bbox=dict(facecolor='white', alpha=0.8))
    
    # Set up the plot
    ax.set_xlim(-20, 200)
    
    # Calculate y limits
    y_min = dam_base - 1
    
    # Safely extract upstream levels
    upstream_levels = []
    for r in water_levels:
        if 'upstream_level' in r:
            upstream_levels.append(r['upstream_level'])
    
    if upstream_levels:
        y_max = max(upstream_levels) + 2
    else:
        # Default if no levels available
        y_max = dam_base + 15
    
    ax.set_ylim(y_min, y_max)
    
    # Add title and labels
    ax.set_title('Dam Flow During Flood Event')
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Elevation (m)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Initialize hydraulic jump marker
    jump_line = ax.axvline(x=0, color='m', linestyle='-.', alpha=0.0)
    jump_marker = ax.scatter([], [], color='m', s=50, marker='o', alpha=0.0)
    
    def init():
        """Initialize animation elements."""
        water_line.set_data([], [])
        flow_line.set_data([], [])
        level_text.set_text('')
        discharge_text.set_text('')
        jump_line.set_alpha(0.0)
        jump_marker.set_alpha(0.0)
        return water_line, flow_line, level_text, discharge_text, jump_line, jump_marker
    
    def update(frame):
        """Update animation elements for each frame."""
        # Ensure valid frame index
        frame_idx = min(frame, len(water_levels) - 1)
        
        # Get the result for this frame
        result = water_levels[frame_idx]
        
        # Extract tailwater data
        tailwater = result.get('tailwater', {})
        x_values = tailwater.get('x', [])
        
        # Skip if no x values
        if len(x_values) == 0:
            return water_line, flow_line, level_text, discharge_text, jump_line, jump_marker
        
        # Create bed profile
        bed = tailwater.get('z', np.zeros_like(x_values))
        
        # Create water surface profile
        upstream_level = result.get('upstream_level', dam_base)
        
        # Check if we have valid downstream water surface elevations
        if 'wse' in tailwater and len(tailwater['wse']) == len(x_values):
            # Combine with upstream level
            water_surface = np.concatenate([
                np.full_like(x_dam, upstream_level),
                tailwater['wse']
            ])
            
            # Combine x values
            combined_x = np.concatenate([x_dam, x_values])
            
            # Update water line
            water_line.set_data(combined_x, water_surface)
            
            # Update flow line (simplified representation)
            # Only show flow when water is over the crest
            dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
            
            if upstream_level > dam_crest:
                # Create a simplified flow path from upstream to downstream
                flow_x = np.linspace(-5, 50, 100)
                
                # Flow path approximation
                flow_y = np.zeros_like(flow_x)
                
                # Calculate flow with safer indexing
                for i, x in enumerate(flow_x):
                    if x < 0:  # Upstream of dam
                        flow_y[i] = upstream_level - 0.5  # Slightly below water surface
                    elif x < 5:  # Over dam
                        # Approximate nappe trajectory
                        t = (x / 5)  # Normalized distance along dam
                        h = upstream_level - dam_crest  # Head
                        height_above_crest = h * (1 - t**2)  # Quadratic approximation
                        flow_y[i] = dam_crest + height_above_crest
                    else:  # Downstream of dam
                        # Find closest point in tailwater
                        if len(x_values) > 0 and len(tailwater.get('wse', [])) > 0:
                            closest_idx = np.argmin(np.abs(np.array(x_values) - x))
                            # Check bounds before accessing
                            if closest_idx < len(tailwater['wse']):
                                flow_y[i] = tailwater['wse'][closest_idx]
                            else:
                                # Default to last value if out of bounds
                                flow_y[i] = tailwater['wse'][-1] if len(tailwater['wse']) > 0 else dam_base
                        else:
                            # Default if no tailwater data
                            flow_y[i] = dam_base
                
                flow_line.set_data(flow_x, flow_y)
                flow_line.set_alpha(0.7)
            else:
                flow_line.set_data([], [])
                flow_line.set_alpha(0.0)
            
            # Update text
            level_text.set_text(f'Water Level: {upstream_level:.2f} m')
            discharge_text.set_text(f'Discharge: {result.get("discharge", 0):.2f} m³/s')
            
            # Update hydraulic jump if present
            jump = result.get('hydraulic_jump', {})
            if jump.get('jump_possible', False) and 'location' in jump:
                jump_loc = jump['location']
                jump_line.set_xdata([jump_loc])
                jump_line.set_alpha(0.7)
                
                # Find the bed elevation at jump location
                if len(x_values) > 0 and len(bed) > 0:
                    closest_idx = np.argmin(np.abs(np.array(x_values) - jump_loc))
                    if closest_idx < len(bed):
                        jump_z = bed[closest_idx]
                        
                        # Get initial depth
                        y1 = jump.get('initial_depth', 0.1)
                        
                        # Update jump marker
                        jump_marker.set_offsets([[jump_loc, jump_z + y1]])
                        jump_marker.set_alpha(1.0)
            else:
                jump_line.set_alpha(0.0)
                jump_marker.set_alpha(0.0)
        else:
            # No valid water surface data
            water_line.set_data([], [])
            flow_line.set_data([], [])
            level_text.set_text(f'Water Level: {upstream_level:.2f} m')
            discharge_text.set_text(f'Discharge: {result.get("discharge", 0):.2f} m³/s')
            jump_line.set_alpha(0.0)
            jump_marker.set_alpha(0.0)
        
        return water_line, flow_line, level_text, discharge_text, jump_line, jump_marker
    
    # Determine number of frames to use
    actual_frames = min(frames, len(water_levels))
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=actual_frames,
                                 init_func=init, blit=True, interval=interval)
    
    return anim, fig, ax