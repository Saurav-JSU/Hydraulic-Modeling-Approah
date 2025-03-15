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
    
    # Extract data
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    tailwater = results['tailwater']
    
    # Create x and y ranges for full domain
    x_dam = np.linspace(-20, 5, 50)  # For dam
    x_values = tailwater['x']  # Downstream profile
    
    # Combine ranges and sort
    combined_x = np.concatenate([x_dam, x_values])
    idx = np.argsort(combined_x)
    combined_x = combined_x[idx]
    
    # Get dam profile
    dam_profile = dam.get_profile(x_dam)
    
    # Create bed profile along full domain
    bed = np.zeros_like(combined_x)
    
    # Fill dam portion
    for i, x in enumerate(combined_x):
        if x <= 0:  # Upstream of dam
            bed[i] = dam_base
        elif x <= 5:  # Dam body
            # Interpolate dam profile
            idx = np.argmin(np.abs(x_dam - x))
            bed[i] = dam_profile['z'][idx]
        else:  # Downstream of dam
            # Interpolate tailwater bed
            idx = np.argmin(np.abs(x_values - x))
            bed[i] = tailwater['z'][idx]
    
    # Create water surface profile
    water_surface = np.zeros_like(combined_x)
    
    for i, x in enumerate(combined_x):
        if x <= 0:  # Upstream of dam
            water_surface[i] = results['upstream_level']
        else:  # Downstream of dam
            # Interpolate tailwater surface
            idx = np.argmin(np.abs(x_values - x))
            water_surface[i] = tailwater['wse'][idx]
    
    # Plot bed and water surface
    ax.plot(combined_x, bed, 'k-', linewidth=2, label='Channel Bed')
    ax.plot(combined_x, water_surface, 'b-', linewidth=2, label='Water Surface')
    
    # Fill the water body with light blue
    water_poly = np.column_stack([
        np.concatenate([combined_x, combined_x[::-1]]),
        np.concatenate([water_surface, bed[::-1]])
    ])
    water_patch = Polygon(water_poly, closed=True, alpha=0.3, color='skyblue')
    ax.add_patch(water_patch)
    
    # Fill the dam body with dark gray
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    dam_patch = Polygon(dam_poly, closed=True, alpha=1.0, color='dimgray')
    ax.add_patch(dam_patch)
    
    # Plot reference lines
    yn = results['normal_depth']
    yc = results['critical_depth']
    
    # Normal depth line
    base_elevation = tailwater['z'][0]
    ax.axhline(y=base_elevation + yn, color='g', linestyle='--', 
               label=f'Normal Depth ({yn:.2f} m)')
    
    # Critical depth line
    ax.axhline(y=base_elevation + yc, color='r', linestyle=':', 
               label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present and requested
    jump = results['hydraulic_jump']
    if show_jump and jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        jump_z = tailwater['z'][jump_index]
        
        # Plot the jump
        ax.axvline(x=jump_loc, color='m', linestyle='-.',
                  label=f'Hydraulic Jump')
        
        # Create a polygon for the jump
        y1 = jump['initial_depth']
        y2 = jump['sequent_depth']
        
        jump_poly = np.array([
            [jump_loc-5, jump_z],
            [jump_loc-5, jump_z + y1],
            [jump_loc, jump_z + y2],
            [jump_loc+5, jump_z + y2],
            [jump_loc+5, jump_z]
        ])
        jump_patch = Polygon(jump_poly, closed=True, alpha=0.5, color='magenta')
        ax.add_patch(jump_patch)
        
        # Add annotation for jump details
        if show_annotations:
            ax.annotate(
                f"Hydraulic Jump\n"
                f"Type: {jump['jump_type']}\n"
                f"Fr₁ = {jump['initial_froude']:.2f}\n"
                f"y₁ = {y1:.2f} m\n"
                f"y₂ = {y2:.2f} m\n"
                f"Force = {jump.get('net_force', 0):.2f} N/m",
                xy=(jump_loc, jump_z + y2),
                xytext=(jump_loc+30, jump_z + y2 + 0.5),
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
            )
    
    # Add discharge information
    if show_annotations:
        ax.annotate(
            f"Discharge: {results['discharge']:.2f} m³/s\n"
            f"Head: {results['head']:.2f} m\n"
            f"Velocity: {results['velocity']:.2f} m/s",
            xy=(combined_x[0], results['upstream_level']),
            xytext=(combined_x[0] + 20, results['upstream_level'] + 1),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
        
        # Add energy information
        energy = results['energy_results']
        ax.annotate(
            f"Energy Dissipation\n"
            f"Loss: {energy['energy_loss']:.2f} m\n"
            f"Ratio: {energy['dissipation_ratio']*100:.1f}%\n"
            f"Power: {energy['power']/1000:.2f} kW",
            xy=(5, dam_base + 3),
            xytext=(10, dam_base + 6),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range
        ax.set_xlim(-20, min(200, np.max(x_values)))
    
    # Calculate y limits to include important elements
    y_min = min(bed) - 1
    y_max = max(water_surface) + 2
    
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
    
    # Extract data
    tailwater = results['tailwater']
    
    # Get x and velocity values from tailwater
    x_values = tailwater['x']
    
    # Calculate velocities from discharge
    discharge = results['discharge']
    velocities = tailwater['v']
    
    # Plot velocity profile
    ax.plot(x_values, velocities, 'r-', linewidth=2, label='Flow Velocity')
    
    # Add critical velocity line
    critical_velocity = discharge / (results['critical_depth'] * scenario['channel_width_at_dam'])
    ax.axhline(y=critical_velocity, color='k', linestyle=':', 
              label=f'Critical Velocity ({critical_velocity:.2f} m/s)')
    
    # Mark velocity at the dam
    dam_velocity = results['velocity']
    ax.plot(0, dam_velocity, 'ro', markersize=8,
           label=f'Dam Crest Velocity ({dam_velocity:.2f} m/s)')
    
    # Mark hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        
        # Get velocities before and after jump
        v1 = discharge / (jump['initial_depth'] * scenario['channel_width_at_dam'])
        v2 = discharge / (jump['sequent_depth'] * scenario['channel_width_at_dam'])
        
        # Mark jump on velocity profile
        ax.plot([jump_loc, jump_loc], [v1, v2], 'm-', linewidth=2)
        ax.plot(jump_loc, v1, 'mo', label=f'Before Jump ({v1:.2f} m/s)')
        ax.plot(jump_loc, v2, 'ms', label=f'After Jump ({v2:.2f} m/s)')
        
        # Annotate the velocity change
        ax.annotate(
            f"Velocity Reduction: {v1-v2:.2f} m/s\n"
            f"({((v1-v2)/v1*100):.1f}%)",
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
    ax.set_xlim(0, min(200, np.max(x_values)))
    
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
    
    # Extract data
    tailwater = results['tailwater']
    
    # Get x and froude values from tailwater
    x_values = tailwater['x']
    froude_numbers = tailwater['fr']
    
    # Create a colormap for the flow regime
    cmap = plt.cm.coolwarm
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    
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
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Mark jump on Froude profile
        ax.plot(jump_loc, jump['initial_froude'], 'ro', markersize=8,
               label=f'Before Jump (Fr={jump["initial_froude"]:.2f})')
        ax.axvline(x=jump_loc, color='m', linestyle='-.', alpha=0.7, label='Hydraulic Jump')
        
        # Annotate the jump
        ax.annotate(
            f"Hydraulic Jump\n"
            f"Fr₁ = {jump['initial_froude']:.2f}\n"
            f"Type: {jump['jump_type']}",
            xy=(jump_loc, jump['initial_froude']),
            xytext=(jump_loc+20, jump['initial_froude']+0.5),
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3")
        )
    
    # Add title and labels
    ax.set_title('Froude Number Distribution')
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Froude Number')
    
    # Set axis limits
    ax.set_xlim(0, min(200, np.max(x_values)))
    
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
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile
    dam_profile = dam.get_profile(x_dam)
    
    # Plot dam profile (static)
    dam_line, = ax.plot(x_dam, dam_profile['z'], 'k-', linewidth=2, label='Dam')
    
    # Plot the dam body (static)
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    dam_patch = Polygon(dam_poly, closed=True, alpha=1.0, color='dimgray')
    ax.add_patch(dam_patch)
    
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
    y_max = max([r['upstream_level'] for r in water_levels]) + 2
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
        water_line.set_data([], [])
        flow_line.set_data([], [])
        level_text.set_text('')
        discharge_text.set_text('')
        jump_line.set_alpha(0.0)
        jump_marker.set_alpha(0.0)
        return water_line, flow_line, level_text, discharge_text, jump_line, jump_marker
    
    def update(frame):
        # Get the result for this frame
        result = water_levels[frame]
        
        # Extract tailwater data
        tailwater = result['tailwater']
        x_values = tailwater['x']
        
        # Create bed profile
        bed = tailwater['z']
        
        # Create water surface profile
        upstream_level = result['upstream_level']
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
        if upstream_level > scenario['dam_crest_elevation']:
            # Create a simplified flow path from upstream to downstream
            flow_x = np.linspace(-5, 50, 100)
            
            # Flow path approximation
            flow_y = np.zeros_like(flow_x)
            
            for i, x in enumerate(flow_x):
                if x < 0:  # Upstream of dam
                    flow_y[i] = upstream_level - 0.5  # Slightly below water surface
                elif x < 5:  # Over dam
                    # Approximate nappe trajectory
                    t = (x / 5)  # Normalized distance along dam
                    h = upstream_level - scenario['dam_crest_elevation']  # Head
                    height_above_crest = h * (1 - t**2)  # Quadratic approximation
                    flow_y[i] = scenario['dam_crest_elevation'] + height_above_crest
                else:  # Downstream of dam
                    # Linear approximation to tailwater level
                    idx = np.argmin(np.abs(x_values - x))
                    flow_y[i] = tailwater['wse'][idx]
            
            flow_line.set_data(flow_x, flow_y)
            flow_line.set_alpha(0.7)
        else:
            flow_line.set_data([], [])
            flow_line.set_alpha(0.0)
        
        # Update text
        level_text.set_text(f'Water Level: {upstream_level:.2f} m')
        discharge_text.set_text(f'Discharge: {result["discharge"]:.2f} m³/s')
        
        # Update hydraulic jump if present
        jump = result['hydraulic_jump']
        if jump.get('jump_possible', False):
            jump_loc = jump['location']
            jump_line.set_xdata([jump_loc])
            jump_line.set_alpha(0.7)
            
            # Find the bed elevation at jump location
            jump_index = np.argmin(np.abs(x_values - jump_loc))
            jump_z = tailwater['z'][jump_index]
            
            # Update jump marker
            jump_marker.set_offsets([[jump_loc, jump_z + jump['initial_depth']]])
            jump_marker.set_alpha(1.0)
        else:
            jump_line.set_alpha(0.0)
            jump_marker.set_alpha(0.0)
        
        return water_line, flow_line, level_text, discharge_text, jump_line, jump_marker
    
    anim = animation.FuncAnimation(fig, update, frames=len(water_levels),
                                 init_func=init, blit=True, interval=interval)
    
    return anim, fig, ax