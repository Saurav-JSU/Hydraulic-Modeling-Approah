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

"""
Fix for the enhanced_profiles.py module.

The main issue is that plot_enhanced_profile() needs to accept an 'ax' parameter
for proper integration with multi-panel visualizations.
"""

# Here's the updated function definition for plot_enhanced_profile:

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
    
    # Rest of the function remains the same...
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
    
    # Plot enhanced bed profile with detailed shading
    ax.plot(combined_x, bed, 'k-', linewidth=1.5, label='Channel Bed')
    
    # Add bed shading for visual depth
    bed_shading = np.full_like(combined_x, min(bed) - 0.5)
    ax.fill_between(combined_x, bed, bed_shading, 
                   color='#8B7355', alpha=0.7, 
                   hatch='///', linewidth=0)
    
    # Create water depth and other parameters arrays for coloring
    water_depth = np.zeros_like(combined_x)
    froude_numbers = np.zeros_like(combined_x)
    velocities = np.zeros_like(combined_x)
    
    for i, x in enumerate(combined_x):
        water_depth[i] = max(0, water_surface[i] - bed[i])
        
        if x > 0:  # Downstream of dam
            # Interpolate values from tailwater
            idx = np.argmin(np.abs(x_values - x))
            if idx < len(tailwater['fr']):
                froude_numbers[i] = tailwater['fr'][idx]
            if idx < len(tailwater['v']):
                velocities[i] = tailwater['v'][idx]
        elif x <= 0 and water_depth[i] > 0:  # Upstream of dam with water
            # Approximate upstream values
            area = water_depth[i] * scenario['channel_width_at_dam']
            velocities[i] = results['discharge'] / area if area > 0 else 0
            froude_numbers[i] = velocities[i] / np.sqrt(9.81 * water_depth[i]) if water_depth[i] > 0 else 0
    
    # Determine coloring parameter based on user choice
    if color_by.lower() == 'froude':
        color_param = froude_numbers
        cmap = plt.cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2, np.max(froude_numbers)))
        cbar_label = 'Froude Number'
    elif color_by.lower() == 'velocity':
        color_param = velocities
        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=0, vmax=max(1, np.max(velocities)))
        cbar_label = 'Velocity (m/s)'
    elif color_by.lower() == 'depth':
        color_param = water_depth
        cmap = plt.cm.Blues
        norm = colors.Normalize(vmin=0, vmax=max(1, np.max(water_depth)))
        cbar_label = 'Water Depth (m)'
    else:  # Default to Froude
        color_param = froude_numbers
        cmap = plt.cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2, np.max(froude_numbers)))
        cbar_label = 'Froude Number'
    
    # Plot enhanced water surface with coloring
    points = np.array([combined_x, water_surface]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a LineCollection with the specified colormap
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
    lc.set_array(color_param[:-1])
    line = ax.add_collection(lc)
    
    # Add water body with gradient coloring
    for i in range(len(combined_x)-1):
        # Skip if both points are at or below bed
        if water_surface[i] <= bed[i] and water_surface[i+1] <= bed[i+1]:
            continue
            
        # Create polygon for each segment
        x_vals = [combined_x[i], combined_x[i+1], combined_x[i+1], combined_x[i]]
        y_vals = [water_surface[i], water_surface[i+1], 
                  max(bed[i+1], water_surface[i+1]), max(bed[i], water_surface[i])]
        
        # Skip invalid polygons
        if len(set(y_vals)) <= 2:  # Not enough distinct y-values
            continue
            
        color_val = color_param[i]
        rgba_color = cmap(norm(color_val))
        
        # Add transparency based on depth for visual effect
        depth_factor = min(1, water_depth[i] / max(1, np.max(water_depth)))
        rgba_color = list(rgba_color)
        rgba_color[3] = 0.3 + 0.4 * depth_factor  # Alpha between 0.3 and 0.7
        
        poly = Polygon(np.column_stack([x_vals, y_vals]), 
                       facecolor=rgba_color, edgecolor=None)
        ax.add_patch(poly)
    
    # Fill the dam body with enhanced visualization
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    
    # Create a dam visualization with shading effects
    dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                       edgecolor='black', linewidth=0.5)
    
    # Add dam patch to axis
    ax.add_patch(dam_patch)
    
    # Add shading to make the dam body look three-dimensional
    dam_z_min = np.min(dam_profile['z'])
    dam_z_max = np.max(dam_profile['z'])
    
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
    
    # Plot reference lines with enhanced styling
    yn = results['normal_depth']
    yc = results['critical_depth']
    
    # Normal depth line
    base_elevation = tailwater['z'][0]
    ax.axhline(y=base_elevation + yn, color='green', linestyle='--', 
               alpha=0.7, linewidth=1.5, label=f'Normal Depth ({yn:.2f} m)')
    
    # Critical depth line
    ax.axhline(y=base_elevation + yc, color='red', linestyle=':', 
               alpha=0.7, linewidth=1.5, label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        jump_z = tailwater['z'][jump_index]
        
        # Plot the jump with enhanced styling
        ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                  alpha=0.8, linewidth=2, label=f'Hydraulic Jump')
        
        # Create a gradient polygon for the jump
        y1 = jump['initial_depth']
        y2 = jump['sequent_depth']
        
        # Gradient for hydraulic jump visualization
        x_jump = np.linspace(jump_loc-2, jump_loc+8, 20)
        y_jump = np.zeros_like(x_jump)
        
        for i, x in enumerate(x_jump):
            rel_pos = (x - (jump_loc-2)) / 10  # 0 to 1 across the jump
            
            # Create smooth transition from y1 to y2
            if rel_pos < 0.5:
                # Pre-jump (shallow depth)
                y_jump[i] = y1
            elif rel_pos < 0.7:
                # Transition zone with turbulence
                t = (rel_pos - 0.5) / 0.2  # 0 to 1 in transition
                y_jump[i] = y1 + (y2 - y1) * t
            else:
                # Post-jump (deeper depth)
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
        if show_annotations:
            # Enhanced annotation with background and outline for better visibility
            txt = ax.annotate(
                f"Hydraulic Jump\n"
                f"Type: {jump['jump_type']}\n"
                f"Fr₁ = {jump['initial_froude']:.2f}\n"
                f"y₁ = {y1:.2f} m → y₂ = {y2:.2f} m\n"
                f"E.Loss = {jump['energy_loss']:.2f} m",
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
        info_text = (
            f"Discharge: {results['discharge']:.2f} m³/s\n"
            f"Head: {results['head']:.2f} m\n"
            f"Velocity at dam: {results['velocity']:.2f} m/s\n"
            f"Energy dissipation: {results['energy_results']['dissipation_ratio']*100:.1f}%"
        )
        
        # Position the info box based on the view range
        info_x = combined_x[0] + 10
        info_y = np.max(water_surface) - 1
        
        ax.text(info_x, info_y, info_text,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8,
                        edgecolor='blue', linewidth=1),
               fontsize=10, color='black')
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range
        ax.set_xlim(-20, min(200, np.max(x_values)))
    
    # Calculate y limits to include important elements with some padding
    y_min = min(bed) - 1
    y_max = max(water_surface) + 2
    
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
    x_values = results['tailwater']['x']
    
    # Plot additional parameters
    panel_idx = 1
    for param in parameters:
        if param.lower() == 'velocity':
            # Plot velocity profile
            axs[panel_idx].plot(x_values, results['tailwater']['v'], 'b-', linewidth=2)
            axs[panel_idx].set_ylabel('Velocity (m/s)')
            axs[panel_idx].axhline(y=results['velocity'], color='r', linestyle='--', 
                                  label=f'Dam Crest Velocity ({results["velocity"]:.2f} m/s)')
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'froude':
            # Plot Froude number profile
            fr = results['tailwater']['fr']
            axs[panel_idx].plot(x_values, fr, 'r-', linewidth=2)
            axs[panel_idx].set_ylabel('Froude Number')
            axs[panel_idx].axhline(y=1, color='k', linestyle='--', label='Critical Flow (Fr=1)')
            
            # Color regions based on flow regime
            axs[panel_idx].fill_between(x_values, fr, 0, where=(fr<1), color='blue', alpha=0.2, label='Subcritical')
            axs[panel_idx].fill_between(x_values, fr, 1, where=(fr>1), color='red', alpha=0.2, label='Supercritical')
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'depth':
            # Plot water depth profile
            depth = results['tailwater']['y']
            axs[panel_idx].plot(x_values, depth, 'g-', linewidth=2)
            axs[panel_idx].set_ylabel('Water Depth (m)')
            
            # Add normal and critical depth references
            axs[panel_idx].axhline(y=results['normal_depth'], color='g', linestyle='--', 
                                 label=f'Normal Depth ({results["normal_depth"]:.2f} m)')
            axs[panel_idx].axhline(y=results['critical_depth'], color='r', linestyle=':', 
                                 label=f'Critical Depth ({results["critical_depth"]:.2f} m)')
            axs[panel_idx].grid(True, alpha=0.3)
            axs[panel_idx].legend(loc='upper right')
            
        elif param.lower() == 'energy':
            # Plot energy grade line
            if 'energy' in results['tailwater']:
                energy = results['tailwater']['energy']
                bed = results['tailwater']['z']
                axs[panel_idx].plot(x_values, energy - bed, 'purple', linewidth=2, label='Specific Energy')
                axs[panel_idx].set_ylabel('Specific Energy (m)')
                axs[panel_idx].grid(True, alpha=0.3)
                axs[panel_idx].legend(loc='upper right')
        
        elif param.lower() == 'shear':
            # Calculate and plot shear stress (approximation)
            depth = results['tailwater']['y']
            slope = scenario['downstream_slope']
            rho = 1000  # Water density
            g = 9.81     # Gravity
            
            # Simple approximation: τ = ρ*g*y*S
            shear_stress = rho * g * depth * slope
            
            axs[panel_idx].plot(x_values, shear_stress, 'brown', linewidth=2)
            axs[panel_idx].set_ylabel('Shear Stress (N/m²)')
            axs[panel_idx].grid(True, alpha=0.3)
        
        # Increment panel index
        panel_idx += 1
    
    # Set common x-axis label
    axs[-1].set_xlabel('Distance from Dam (m)')
    
    # Mark hydraulic jump location across all panels if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
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
        jump = results['hydraulic_jump']
        if jump.get('jump_possible', False):
            x_locations.append(jump['location'])
        
        # Add a far downstream location
        x_locations.append(min(200, np.max(results['tailwater']['x'])))
    
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
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    dam_crest = scenario['dam_crest_elevation']
    tailwater = results['tailwater']
    channel_side_slope = scenario['channel_side_slope']
    channel_bottom_width = scenario['channel_bottom_width']
    
    # Plot cross-sections
    for i, x_loc in enumerate(x_locations):
        ax = ax_sections[i]
        
        # Determine water depth and bed elevation at this location
        if x_loc <= 0:  # Upstream of dam
            water_elevation = results['upstream_level']
            bed_elevation = dam_base
        else:  # Downstream of dam
            # Find closest point in tailwater results
            idx = np.argmin(np.abs(tailwater['x'] - x_loc))
            water_elevation = tailwater['wse'][idx]
            bed_elevation = tailwater['z'][idx]
        
        water_depth = max(0, water_elevation - bed_elevation)
        
        # Calculate cross-section geometry (trapezoidal channel)
        half_bottom = channel_bottom_width / 2
        if water_depth > 0:
            water_width = channel_bottom_width + 2 * channel_side_slope * water_depth
            half_water = water_width / 2
        else:
            half_water = half_bottom
        
        # Plot channel cross-section
        channel_y = np.array([0, water_depth, water_depth, 0])
        channel_x = np.array([-half_bottom, -half_water, half_water, half_bottom])
        
        ax.plot(channel_x, channel_y, 'k-', linewidth=1.5)
        
        # Fill the channel bed
        ax.fill_between([-half_bottom, half_bottom], 0, -0.5, color='#8B7355', hatch='///')
        
        # Fill water area if there's water
        if water_depth > 0:
            ax.fill_between([-half_water, half_water], 0, water_depth, color='skyblue', alpha=0.5)
        
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
        if x_loc > 0:
            idx = np.argmin(np.abs(tailwater['x'] - x_loc))
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