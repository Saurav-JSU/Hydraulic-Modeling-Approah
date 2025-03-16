"""
Time series visualization for hydraulic modeling.

This module provides enhanced animations for time-varying hydraulic processes,
including flood waves, dam overflow transitions, and backwater effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects

def create_enhanced_flood_animation(scenario, results_list, fps=5, colormap='coolwarm', 
                                   color_by='froude', dpi=100):
    """
    Create an enhanced animation of a flood event with dynamic coloring and effects.
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results at different time steps/water levels
        fps (int): Frames per second for the animation
        colormap (str): Matplotlib colormap for water surface coloring
        color_by (str): Parameter to color by ('froude', 'velocity', 'depth')
        dpi (int): Resolution for the animation
        
    Returns:
        tuple: (anim, fig, ax) Animation object and figure/axis
    """
    # Input validation
    if not results_list:
        raise ValueError("results_list cannot be empty")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data for setup with safe dictionary access
    dam = scenario.get('dam')
    if dam is None:
        raise ValueError("Scenario dictionary must contain 'dam' key")
        
    dam_base = scenario.get('dam_base_elevation', 0)
    dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile - handle if get_profile method doesn't exist
    try:
        dam_profile = dam.get_profile(x_dam)
    except (AttributeError, TypeError) as e:
        # Create a simple dam profile if get_profile doesn't exist
        dam_profile = {'z': dam_base + (dam_crest - dam_base) * np.maximum(0, np.minimum(1, (x_dam + 20) / 20))}
    
    # Set up plot limits based on all results
    max_x = 0
    max_y = dam_crest + 5  # Some space above crest
    min_y = dam_base - 1   # Some space below base
    
    for result in results_list:
        tailwater = result.get('tailwater', {})
        x_values = tailwater.get('x', [])
        wse = tailwater.get('wse', [])
        
        if len(x_values) > 0:
            max_x = max(max_x, np.max(x_values))
        
        # Get upstream level safely
        upstream_level = result.get('upstream_level', dam_base)
        max_y = max(max_y, upstream_level)
        
        if len(wse) > 0:
            max_y = max(max_y, np.max(wse))
    
    # Set axis limits with some padding
    ax.set_xlim(-20, min(200, max_x))
    ax.set_ylim(min_y, max_y + 1)
    
    # Plot static elements
    # Dam outline (will be filled dynamically to show proper water overlay)
    if 'z' in dam_profile:
        dam_line, = ax.plot(x_dam, dam_profile['z'], 'k-', linewidth=1.5, label='Dam')
    else:
        # Default simple dam line if profile has no 'z' values
        dam_line, = ax.plot([0, 0], [dam_base, dam_crest], 'k-', linewidth=1.5, label='Dam')
    
    # Reference lines for normal and critical depths
    # These will be updated for each frame
    normal_line = ax.axhline(y=0, color='g', linestyle='--', alpha=0.7, 
                           label='Normal Depth', visible=False)
    critical_line = ax.axhline(y=0, color='r', linestyle=':', alpha=0.7, 
                             label='Critical Depth', visible=False)
    
    # Create color map
    try:
        cmap = plt.cm.get_cmap(colormap)
    except ValueError:
        print(f"Warning: Unknown colormap '{colormap}'. Using 'coolwarm' instead.")
        cmap = plt.cm.get_cmap('coolwarm')
    
    # Create variable containers for dynamic elements
    water_patches = []  # Will hold water polygons
    jump_marker = ax.axvline(x=0, color='m', linestyle='-.', alpha=0.0, label='Hydraulic Jump')
    jump_annotation = ax.annotate("", xy=(0, 0), xytext=(30, 30), 
                               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
                               visible=False)
    
    # Information text for time step, discharge, etc.
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                       fontsize=10)
    
    # Add progress bar
    progress_height = 0.02
    progress_y = 0.01
    progress_bg = plt.Rectangle((0.1, progress_y), 0.8, progress_height, 
                              transform=fig.transFigure, facecolor='lightgray',
                              edgecolor='gray', clip_on=False)
    progress_bar = plt.Rectangle((0.1, progress_y), 0.0, progress_height, 
                               transform=fig.transFigure, facecolor='green',
                               edgecolor='darkgreen', clip_on=False)
    
    try:
        fig.patches.extend([progress_bg, progress_bar])
    except AttributeError:
        # If fig.patches.extend doesn't work (older matplotlib)
        fig.patches.append(progress_bg)
        fig.patches.append(progress_bar)
    
    # Add colorbar for the water coloring
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])  # Empty array, will be updated in animation
    
    if color_by.lower() == 'froude':
        # For Froude number, use TwoSlopeNorm with center at 1
        sm.set_norm(colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=2))
        cbar_label = 'Froude Number'
    elif color_by.lower() == 'velocity':
        sm.set_norm(colors.Normalize(vmin=0, vmax=5))
        cbar_label = 'Velocity (m/s)'
    elif color_by.lower() == 'depth':
        sm.set_norm(colors.Normalize(vmin=0, vmax=5))
        cbar_label = 'Water Depth (m)'
    else:
        # Default to Froude
        sm.set_norm(colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=2))
        cbar_label = 'Froude Number'
        
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)
    
    # If coloring by Froude, add regime indicators
    if color_by.lower() == 'froude':
        cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                    rotation=90, transform=cbar.ax.transAxes, color='black')
        cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                    rotation=90, transform=cbar.ax.transAxes, color='black')
        cbar.ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set up title and labels
    ax.set_title('Dam Flow During Flood Event', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Dam (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add dam body (will be static throughout animation)
    try:
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                            edgecolor='black', linewidth=0.5)
            ax.add_patch(dam_patch)
        else:
            # Simple dam rectangle as fallback
            ax.fill_between([0, 0], [dam_base, dam_crest], color='#4F4F4F')
    except Exception as e:
        # If polygon creation fails, draw a simple dam
        print(f"Warning: Could not create dam polygon: {str(e)}")
        ax.fill_between([0, 0], [dam_base, dam_crest], color='#4F4F4F')
    
    # Function to update the animation for each frame
    def update(frame_idx):
        # Clear previous water patches
        for patch in water_patches:
            patch.remove()
        water_patches.clear()
        
        # Get the result for this frame
        result = results_list[min(frame_idx, len(results_list) - 1)]
        
        # Extract data safely
        tailwater = result.get('tailwater', {})
        upstream_level = result.get('upstream_level', dam_base)
        discharge = result.get('discharge', 0)
        
        # Extract or compute coloring parameter
        if color_by.lower() == 'froude':
            color_param = tailwater.get('fr', [])
            # Set norm again to adjust for current frame data
            vmax = max(2, np.max(color_param)) if len(color_param) > 0 else 2
            sm.set_norm(colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax))
        elif color_by.lower() == 'velocity':
            color_param = tailwater.get('v', [])
            vmax = max(1, np.max(color_param)) if len(color_param) > 0 else 1
            sm.set_norm(colors.Normalize(vmin=0, vmax=vmax))
        elif color_by.lower() == 'depth':
            color_param = tailwater.get('y', [])
            vmax = max(1, np.max(color_param)) if len(color_param) > 0 else 1
            sm.set_norm(colors.Normalize(vmin=0, vmax=vmax))
        else:
            # Default in case of unknown parameter
            color_param = tailwater.get('fr', [])
            vmax = max(2, np.max(color_param)) if len(color_param) > 0 else 2
            sm.set_norm(colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax))
        
        # Update reference lines for normal and critical depth
        yn = result.get('normal_depth', 0)
        yc = result.get('critical_depth', 0)
        
        # Get base elevation - safely handle empty z array
        z_array = tailwater.get('z', [])
        base_elevation = z_array[0] if len(z_array) > 0 else dam_base
        
        if yn > 0:
            normal_line.set_ydata(base_elevation + yn)
            normal_line.set_visible(True)
        else:
            normal_line.set_visible(False)
            
        if yc > 0:
            critical_line.set_ydata(base_elevation + yc)
            critical_line.set_visible(True)
        else:
            critical_line.set_visible(False)
        
        # Create water surface and bed profiles safely
        x_values = tailwater.get('x', [])
        z = tailwater.get('z', [])
        wse = tailwater.get('wse', [])
        
        # Only proceed if we have valid tailwater data
        if len(x_values) > 0 and len(z) > 0 and len(wse) > 0:
            # Include upstream portion
            x_upstream = np.linspace(-20, 0, 20)
            z_upstream = np.full_like(x_upstream, dam_base)
            wse_upstream = np.full_like(x_upstream, upstream_level)
            
            # Combine arrays for complete profile
            x_combined = np.concatenate([x_upstream, x_values])
            
            # Ensure z and wse arrays are compatible with x_values
            if len(z) == len(x_values) and len(wse) == len(x_values):
                z_combined = np.concatenate([z_upstream, z])
                wse_combined = np.concatenate([wse_upstream, wse])
                
                # Create coloring parameter for full domain
                full_color_param = np.zeros_like(x_combined)
                
                # Upstream values based on discharge
                upstream_depth = max(0, upstream_level - dam_base)  # Ensure non-negative depth
                channel_width = scenario.get('channel_width_at_dam', 5.0)
                
                if upstream_depth > 0 and discharge > 0 and channel_width > 0:
                    # Get channel shape parameters for trapezoidal section
                    side_slope = scenario.get('channel_side_slope', 0)
                    
                    # Calculate top width for trapezoidal section
                    top_width = channel_width + 2 * side_slope * upstream_depth
                    
                    # Calculate cross-sectional area
                    area = 0.5 * (channel_width + top_width) * upstream_depth
                    
                    # Ensure non-zero area to avoid division by zero
                    area = max(area, 0.001)
                    
                    # Calculate velocity
                    upstream_velocity = discharge / area
                    
                    # Calculate Froude number using hydraulic depth
                    hydraulic_depth = area / top_width if top_width > 0 else upstream_depth
                    hydraulic_depth = max(hydraulic_depth, 0.001)  # Avoid division by zero
                    upstream_froude = upstream_velocity / np.sqrt(9.81 * hydraulic_depth)
                    
                    # Set appropriate upstream color parameter
                    if color_by.lower() == 'froude':
                        upstream_color = upstream_froude
                    elif color_by.lower() == 'velocity':
                        upstream_color = upstream_velocity
                    else:  # depth
                        upstream_color = upstream_depth
                    
                    # Fill upstream portion of parameter array
                    full_color_param[:len(x_upstream)] = upstream_color
                
                # Downstream values from tailwater - handle length mismatches
                if len(color_param) > 0:
                    downstream_length = min(len(full_color_param) - len(x_upstream), len(color_param))
                    full_color_param[len(x_upstream):len(x_upstream) + downstream_length] = color_param[:downstream_length]
                
                # Create water polygons for areas with depth > 0
                for i in range(len(x_combined)-1):
                    # Skip if both points are at or below bed
                    if wse_combined[i] <= z_combined[i] and wse_combined[i+1] <= z_combined[i+1]:
                        continue
                        
                    # Create polygon for each segment
                    x_vals = [x_combined[i], x_combined[i+1], x_combined[i+1], x_combined[i]]
                    
                    # Ensure water surface is at or above bed
                    y_surface_i = max(wse_combined[i], z_combined[i])
                    y_surface_i_plus_1 = max(wse_combined[i+1], z_combined[i+1])
                    
                    y_vals = [y_surface_i, y_surface_i_plus_1, z_combined[i+1], z_combined[i]]
                    
                    # Skip invalid polygons - need at least 3 distinct points
                    if len(set(zip(x_vals, y_vals))) < 3:
                        continue
                        
                    # Get color based on parameter value - handle index safety
                    if i < len(full_color_param):
                        color_val = full_color_param[i]
                    else:
                        color_val = 0
                    
                    # Create different color effects based on parameter
                    if color_by.lower() == 'froude':
                        # Use colormap with transparency effect for Froude
                        rgba_color = cmap(sm.norm(color_val))
                        depth = max(0, wse_combined[i] - z_combined[i])
                        
                        # Find maximum depth for scaling
                        max_depth = max(1, np.max(tailwater.get('y', [1])) if len(tailwater.get('y', [])) > 0 else 1)
                        
                        # Add flow regime visual effects
                        if color_val < 0.8:  # Subcritical - calm water
                            alpha = 0.5 + 0.3 * (depth / max_depth)
                            rgba_color = list(rgba_color)
                            rgba_color[3] = min(0.8, alpha)  # Transparency adjustment
                        elif color_val > 1.2:  # Supercritical - turbulent
                            alpha = 0.7
                            rgba_color = list(rgba_color)
                            rgba_color[3] = alpha
                        else:  # Near critical - transitional
                            alpha = 0.6
                            rgba_color = list(rgba_color)
                            rgba_color[3] = alpha
                    else:
                        # Standard coloring for other parameters
                        rgba_color = cmap(sm.norm(color_val))
                        rgba_color = list(rgba_color)
                        rgba_color[3] = 0.6  # Fixed transparency
                    
                    # Create and add the water polygon
                    try:
                        poly = Polygon(np.column_stack([x_vals, y_vals]), 
                                    facecolor=rgba_color, edgecolor=None)
                        water_patches.append(poly)
                        ax.add_patch(poly)
                    except Exception as e:
                        print(f"Warning: Could not create water polygon segment at i={i}: {str(e)}")
                
                # Handle hydraulic jump if present
                jump = result.get('hydraulic_jump', {})
                if jump.get('jump_possible', False) and 'location' in jump:
                    jump_loc = jump['location']
                    jump_marker.set_xdata([jump_loc])
                    jump_marker.set_alpha(0.7)
                    
                    # Find the bed elevation at jump location
                    if len(x_values) > 0 and len(z) > 0:
                        # Find closest point in tailwater results
                        jump_index = np.argmin(np.abs(np.array(x_values) - jump_loc))
                        
                        # Check bounds before accessing
                        if jump_index < len(z):
                            jump_z = z[jump_index]
                        else:
                            jump_z = dam_base
                    else:
                        jump_z = dam_base
                    
                    # Get initial and sequent depths
                    y1 = jump.get('initial_depth', 0.1)
                    y2 = jump.get('sequent_depth', y1 * 2)
                    
                    # Ensure physically realistic jump (y2 > y1)
                    if y2 <= y1:
                        # Calculate sequent depth using momentum equation if not provided correctly
                        if 'initial_froude' in jump and jump['initial_froude'] > 1:
                            fr1 = jump['initial_froude']
                            depth_ratio = 0.5 * (-1 + np.sqrt(1 + 8 * fr1**2))
                            y2 = y1 * depth_ratio
                        else:
                            # Fallback: assume transitional jump with y2 = 2*y1
                            y2 = y1 * 2
                    
                    # Add annotation for jump
                    jump_annotation.xy = (jump_loc, jump_z + y2)
                    jump_annotation.xytext = (jump_loc + 30, jump_z + y2 + 1)
                    
                    # Create annotation text with safe values
                    jump_type = jump.get('jump_type', 'Unknown')
                    initial_froude = jump.get('initial_froude', 0)
                    
                    jump_annotation.set_text(
                        f"Hydraulic Jump\n"
                        f"Type: {jump_type}\n"
                        f"Fr₁ = {initial_froude:.2f}\n"
                        f"y₁ = {y1:.2f} m → y₂ = {y2:.2f} m"
                    )
                    jump_annotation.set_visible(True)
                    
                    # Create hydraulic jump turbulence effect
                    try:
                        # Add some turbulence visualization around jump
                        jump_x = np.linspace(jump_loc-5, jump_loc+10, 30)
                        for i in range(len(jump_x)-1):
                            # Only add turbulence near the jump center
                            dist = abs(jump_x[i] - jump_loc)
                            if dist > 8:
                                continue
                                
                            # Create dynamic "waves" that vary with each frame for animation effect
                            wave_height = 0.15 * y2 * (1 - dist/10) * (0.8 + 0.4 * np.sin(frame_idx * 0.4 + i))
                            
                            # Create polygon for wave/turbulence
                            x_vals = [jump_x[i], jump_x[i+1], jump_x[i+1], jump_x[i]]
                            
                            # Base height determined by sequent depth with turbulence on top
                            if jump_x[i] >= jump_loc:
                                base_y = jump_z + y2
                            else:
                                # Transition from y1 to y2 over a distance of 5m
                                transition_dist = 5.0
                                transition = max(0, min(1, (jump_x[i] - (jump_loc-transition_dist))/transition_dist))
                                base_y = jump_z + y1 + transition * (y2-y1)
                            
                            y_vals = [base_y + wave_height, 
                                    base_y + wave_height * (0.8 + 0.4 * np.sin(frame_idx * 0.4 + i + 1)),
                                    base_y, base_y]
                            
                            # Color intensity varies with turbulence
                            intensity = max(0, 1 - dist/8)
                            alpha = 0.2 + 0.3 * intensity
                            
                            # Create wavy surface on top of the jump
                            try:
                                poly = Polygon(np.column_stack([x_vals, y_vals]), 
                                            facecolor='white', alpha=alpha)
                                water_patches.append(poly)
                                ax.add_patch(poly)
                            except Exception as e:
                                # Skip this turbulence element if it fails
                                pass
                    except Exception as e:
                        print(f"Warning: Could not create jump turbulence: {str(e)}")
                else:
                    jump_marker.set_alpha(0.0)
                    jump_annotation.set_visible(False)
        else:
            # No valid tailwater data
            jump_marker.set_alpha(0.0)
            jump_annotation.set_visible(False)
        
        # Update info text
        time_position = frame_idx / max(1, len(results_list) - 1)  # Avoid division by zero
        
        water_level_text = f"{upstream_level:.2f}m" if upstream_level > dam_crest else \
                        f"{upstream_level:.2f}m (below crest)"
        
        info_text.set_text(
            f"Time Step: {frame_idx+1}/{len(results_list)}\n"
            f"Water Level: {water_level_text}\n"
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Head: {result.get('head', 0):.2f}m"
        )
        
        # Update progress bar
        progress_bar.set_width(time_position * 0.8)
        
        return water_patches + [jump_marker, jump_annotation, info_text, 
                              normal_line, critical_line, progress_bar]
    
    # Create animation with correct number of frames
    try:
        anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                    interval=1000/fps, blit=True)
    except Exception as e:
        print(f"Warning: Could not create animation: {str(e)}")
        # Try a simpler version as fallback
        anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                    interval=1000/fps, blit=False)
    
    return anim, fig, ax


def create_flow_regime_transition_animation(scenario, results_list, fps=5, dpi=100):
    """
    Create an animation specifically focused on flow regime transitions,
    with enhanced visual indicators of subcritical, critical, and supercritical flow.
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results at different time steps/water levels
        fps (int): Frames per second for the animation
        dpi (int): Resolution for the animation
        
    Returns:
        tuple: (anim, fig, ax) Animation object and figure/axis objects
    """
    # Input validation
    if not results_list:
        raise ValueError("results_list cannot be empty")
    
    # Create figure with two subplots - profile and Froude number
    fig, (ax_profile, ax_froude) = plt.subplots(2, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract data for setup - safely access dictionary values
    dam = scenario.get('dam')
    if dam is None:
        raise ValueError("Scenario dictionary must contain 'dam' key")
        
    dam_base = scenario.get('dam_base_elevation', 0)
    dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile - handle if get_profile method doesn't exist
    try:
        dam_profile = dam.get_profile(x_dam)
    except (AttributeError, TypeError) as e:
        # Create a simple dam profile if get_profile doesn't exist
        dam_profile = {'z': dam_base + (dam_crest - dam_base) * np.maximum(0, np.minimum(1, (x_dam + 20) / 20))}
    
    # Set up plot limits based on all results
    max_x = 0
    max_y = dam_crest + 5  # Some space above crest
    min_y = dam_base - 1   # Some space below base
    max_fr = 0
    
    for result in results_list:
        tailwater = result.get('tailwater', {})
        x_values = tailwater.get('x', [])
        wse = tailwater.get('wse', [])
        fr = tailwater.get('fr', [])
        
        if len(x_values) > 0:
            max_x = max(max_x, np.max(x_values))
        
        # Get upstream level safely
        upstream_level = result.get('upstream_level', dam_base)
        max_y = max(max_y, upstream_level)
        
        if len(wse) > 0:
            max_y = max(max_y, np.max(wse))
            
        if len(fr) > 0:
            max_fr = max(max_fr, np.max(fr))
    
    # Ensure reasonable Froude axis maximum
    max_fr = max(max_fr, 2.0) * 1.1  # Add 10% padding
    
    # Set axis limits with some padding
    ax_profile.set_xlim(-20, min(200, max_x))
    ax_profile.set_ylim(min_y, max_y + 1)
    ax_froude.set_xlim(-20, min(200, max_x))
    ax_froude.set_ylim(0, max_fr)
    
    # Plot static elements in profile view
    # Dam outline
    if 'z' in dam_profile:
        dam_line, = ax_profile.plot(x_dam, dam_profile['z'], 'k-', linewidth=1.5, label='Dam')
    else:
        # Default simple dam line if profile has no 'z' values
        dam_line, = ax_profile.plot([0, 0], [dam_base, dam_crest], 'k-', linewidth=1.5, label='Dam')
    
    # Add dam body
    try:
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                            edgecolor='black', linewidth=0.5)
            ax_profile.add_patch(dam_patch)
        else:
            # Simple dam rectangle as fallback
            ax_profile.fill_between([0, 0], [dam_base, dam_crest], color='#4F4F4F')
    except Exception as e:
        # If polygon creation fails, draw a simple dam
        print(f"Warning: Could not create dam polygon: {str(e)}")
        ax_profile.fill_between([0, 0], [dam_base, dam_crest], color='#4F4F4F')
    
    # Reference lines for normal and critical depths
    normal_line = ax_profile.axhline(y=0, color='g', linestyle='--', alpha=0.7, 
                                   label='Normal Depth', visible=False)
    critical_line = ax_profile.axhline(y=0, color='r', linestyle=':', alpha=0.7, 
                                     label='Critical Depth', visible=False)
    
    # Critical Froude line in Froude plot
    ax_froude.axhline(y=1.0, color='r', linestyle='--', 
                     label='Critical Flow (Fr=1)', linewidth=1.5)
    
    # Fill regions for flow regimes in Froude plot - get axis limits safely
    froude_xlim = ax_froude.get_xlim()
    ax_froude.fill_between([froude_xlim[0], froude_xlim[1]], 
                          0, 1, color='blue', alpha=0.2, label='Subcritical')
    ax_froude.fill_between([froude_xlim[0], froude_xlim[1]], 
                          1, max_fr, color='red', alpha=0.2, label='Supercritical')
    
    # Create variable containers for dynamic elements
    water_patches = []  # Will hold water polygons
    regime_patches = []  # Will hold flow regime patches
    
    # Lines for Froude plot
    froude_line, = ax_froude.plot([], [], 'k-', linewidth=2)
    
    # Jump marker
    jump_marker_profile = ax_profile.axvline(x=0, color='m', linestyle='-.', 
                                          alpha=0.0, label='Hydraulic Jump')
    jump_marker_froude = ax_froude.axvline(x=0, color='m', linestyle='-.', alpha=0.0)
    
    # Annotations
    jump_annotation = ax_profile.annotate("", xy=(0, 0), xytext=(30, 30), 
                                       bbox=dict(boxstyle="round,pad=0.5", 
                                               fc="white", alpha=0.8),
                                       arrowprops=dict(arrowstyle="->", 
                                                     connectionstyle="arc3,rad=0.3"),
                                       visible=False)
    
    # Information text
    info_text = ax_profile.text(0.02, 0.95, "", transform=ax_profile.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                               fontsize=10)
    
    # Add progress bar
    progress_height = 0.01
    progress_y = 0.01
    progress_bg = plt.Rectangle((0.1, progress_y), 0.8, progress_height, 
                              transform=fig.transFigure, facecolor='lightgray',
                              edgecolor='gray', clip_on=False)
    progress_bar = plt.Rectangle((0.1, progress_y), 0.0, progress_height, 
                               transform=fig.transFigure, facecolor='green',
                               edgecolor='darkgreen', clip_on=False)
    
    try:
        fig.patches.extend([progress_bg, progress_bar])
    except AttributeError:
        # If fig.patches.extend doesn't work (older matplotlib)
        fig.patches.append(progress_bg)
        fig.patches.append(progress_bar)
    
    # Set up titles and labels
    ax_profile.set_title('Water Surface Profile During Flood Event', 
                        fontsize=14, fontweight='bold')
    ax_profile.set_ylabel('Elevation (m)', fontsize=12)
    
    ax_froude.set_title('Froude Number Profile', fontsize=14)
    ax_froude.set_xlabel('Distance from Dam (m)', fontsize=12)
    ax_froude.set_ylabel('Froude Number', fontsize=12)
    
    # Add grids
    ax_profile.grid(True, alpha=0.3)
    ax_froude.grid(True, alpha=0.3)
    
    # Add legends
    ax_profile.legend(loc='upper right')
    ax_froude.legend(loc='upper right')
    
    # Function to update the animation for each frame
    def update(frame_idx):
        # Clear previous dynamic elements
        for patch in water_patches:
            patch.remove()
        water_patches.clear()
        
        for patch in regime_patches:
            patch.remove()
        regime_patches.clear()
        
        # Get the result for this frame - ensure valid index
        frame_idx = min(frame_idx, len(results_list) - 1)
        result = results_list[frame_idx]
        
        # Extract data safely
        tailwater = result.get('tailwater', {})
        upstream_level = result.get('upstream_level', dam_base)
        discharge = result.get('discharge', 0)
        
        # Update reference lines for normal and critical depth
        yn = result.get('normal_depth', 0)
        yc = result.get('critical_depth', 0)
        
        # Get base elevation - safely handle empty z array
        z_array = tailwater.get('z', [])
        base_elevation = z_array[0] if len(z_array) > 0 else dam_base
        
        if yn > 0:
            normal_line.set_ydata(base_elevation + yn)
            normal_line.set_visible(True)
        else:
            normal_line.set_visible(False)
            
        if yc > 0:
            critical_line.set_ydata(base_elevation + yc)
            critical_line.set_visible(True)
        else:
            critical_line.set_visible(False)
        
        # Create water surface and bed profiles
        x_values = tailwater.get('x', [])
        z = tailwater.get('z', [])
        wse = tailwater.get('wse', [])
        fr = tailwater.get('fr', [])
        
        # Ensure we have sufficient data to continue
        if len(x_values) > 0 and len(z) > 0 and len(wse) > 0:
            # Include upstream portion
            x_upstream = np.linspace(-20, 0, 20)
            z_upstream = np.full_like(x_upstream, dam_base)
            wse_upstream = np.full_like(x_upstream, upstream_level)
            
            # Combine arrays for complete profile - ensure compatible lengths
            x_combined = np.concatenate([x_upstream, x_values])
            z_combined = np.concatenate([z_upstream, z[:len(x_values)]])
            wse_combined = np.concatenate([wse_upstream, wse[:len(x_values)]])
            
            # Create and color water polygons based on flow regime
            froude_combined = np.zeros_like(x_combined)
            
            # Calculate upstream Froude number
            upstream_depth = max(0, upstream_level - dam_base)
            channel_width = scenario.get('channel_width_at_dam', 5.0)
            
            if upstream_depth > 0 and discharge > 0 and channel_width > 0:
                # Get channel shape parameters for trapezoidal section
                side_slope = scenario.get('channel_side_slope', 0)
                
                # Calculate top width for trapezoidal section
                top_width = channel_width + 2 * side_slope * upstream_depth
                
                # Calculate cross-sectional area
                area = 0.5 * (channel_width + top_width) * upstream_depth
                
                # Ensure non-zero area
                area = max(area, 0.001)
                
                # Calculate velocity
                upstream_velocity = discharge / area
                
                # Calculate Froude number using hydraulic depth for accuracy
                hydraulic_depth = area / top_width if top_width > 0 else upstream_depth
                hydraulic_depth = max(hydraulic_depth, 0.001)  # Avoid division by zero
                upstream_froude = upstream_velocity / np.sqrt(9.81 * hydraulic_depth)
                
                # Fill upstream portion
                froude_combined[:len(x_upstream)] = upstream_froude
            
            # Ensure froude array is compatible with x_values
            if len(fr) > 0:
                downstream_length = min(len(froude_combined) - len(x_upstream), len(fr))
                froude_combined[len(x_upstream):len(x_upstream) + downstream_length] = fr[:downstream_length]
            
            # Update Froude plot line
            froude_line.set_data(x_combined, froude_combined)
            
            # Create water polygons
            for i in range(len(x_combined)-1):
                # Skip if both points are at or below bed
                if wse_combined[i] <= z_combined[i] and wse_combined[i+1] <= z_combined[i+1]:
                    continue
                    
                # Create polygon for each segment
                x_vals = [x_combined[i], x_combined[i+1], x_combined[i+1], x_combined[i]]
                
                # Ensure water surface is at or above bed
                y_surface_i = max(wse_combined[i], z_combined[i])
                y_surface_i_plus_1 = max(wse_combined[i+1], z_combined[i+1])
                
                y_vals = [y_surface_i, y_surface_i_plus_1, z_combined[i+1], z_combined[i]]
                
                # Skip invalid polygons - need at least 3 distinct points
                if len(set(zip(x_vals, y_vals))) < 3:
                    continue
                    
                # Get Froude value safely
                fr_val = froude_combined[i] if i < len(froude_combined) else 0
                
                # Color based on flow regime
                if fr_val < 0.8:  # Subcritical
                    color = 'blue'
                    alpha = 0.5
                elif fr_val > 1.2:  # Supercritical
                    color = 'red'
                    alpha = 0.5
                else:  # Near critical
                    color = 'green'
                    alpha = 0.5
                
                # Create and add the water polygon
                try:
                    poly = Polygon(np.column_stack([x_vals, y_vals]), 
                                facecolor=color, alpha=alpha, edgecolor=None)
                    water_patches.append(poly)
                    ax_profile.add_patch(poly)
                except Exception as e:
                    # Skip this water polygon if creation fails
                    print(f"Warning: Could not create water polygon at i={i}: {str(e)}")
                
                # Add dynamic flow regime indicator patches
                # Only add for some segments to avoid clutter
                try:
                    if i % 10 == 0 and wse_combined[i] > z_combined[i]:
                        depth = wse_combined[i] - z_combined[i]
                        
                        if fr_val < 0.8:  # Subcritical - smooth surface
                            # Just add a subtle highlight
                            x_regime = np.linspace(x_combined[i]-2, x_combined[i]+2, 5)
                            y_regime = wse_combined[i] + 0.02 * depth * np.sin(np.linspace(0, np.pi, 5))
                            
                            regime_poly = Polygon(np.column_stack([x_regime, y_regime]), 
                                                facecolor='white', alpha=0.3, closed=False)
                            regime_patches.append(regime_poly)
                            ax_profile.add_patch(regime_poly)
                            
                        elif fr_val > 1.2:  # Supercritical - rough surface
                            # Add more pronounced waves
                            wave_amp = 0.1 * depth
                            wave_freq = 2 * np.pi / 2  # wavelength = 2m
                            
                            x_regime = np.linspace(x_combined[i]-3, x_combined[i]+3, 20)
                            y_regime = wse_combined[i] + wave_amp * np.sin(wave_freq * 
                                                                        np.linspace(0, 2*np.pi, 20) + 
                                                                        frame_idx * 0.2)
                            
                            regime_poly = Polygon(np.column_stack([x_regime, y_regime]), 
                                                facecolor='white', alpha=0.4, closed=False)
                            regime_patches.append(regime_poly)
                            ax_profile.add_patch(regime_poly)
                except Exception as e:
                    # Skip regime indicators if they fail
                    pass
            
            # Handle hydraulic jump
            jump = result.get('hydraulic_jump', {})
            if jump.get('jump_possible', False) and 'location' in jump:
                jump_loc = jump['location']
                
                # Update jump markers
                jump_marker_profile.set_xdata([jump_loc])
                jump_marker_profile.set_alpha(0.7)
                
                jump_marker_froude.set_xdata([jump_loc])
                jump_marker_froude.set_alpha(0.7)
                
                # Find the bed elevation at jump location
                if len(x_values) > 0 and len(z) > 0:
                    # Find closest point in tailwater results
                    jump_index = np.argmin(np.abs(np.array(x_values) - jump_loc))
                    
                    # Check bounds before accessing
                    if jump_index < len(z):
                        jump_z = z[jump_index]
                    else:
                        jump_z = dam_base
                else:
                    jump_z = dam_base
                
                # Get initial and sequent depths
                y1 = jump.get('initial_depth', 0.1)
                y2 = jump.get('sequent_depth', y1 * 2)
                
                # Ensure physically realistic jump (y2 > y1)
                if y2 <= y1:
                    # Calculate sequent depth using momentum equation if not provided correctly
                    if 'initial_froude' in jump and jump['initial_froude'] > 1:
                        fr1 = jump['initial_froude']
                        depth_ratio = 0.5 * (-1 + np.sqrt(1 + 8 * fr1**2))
                        y2 = y1 * depth_ratio
                    else:
                        # Fallback: assume transitional jump with y2 = 2*y1
                        y2 = y1 * 2
                
                # Add annotation for jump
                jump_annotation.xy = (jump_loc, jump_z + y2)
                jump_annotation.xytext = (jump_loc + 30, jump_z + y2 + 1)
                
                # Get jump values safely
                jump_type = jump.get('jump_type', 'Unknown')
                initial_froude = jump.get('initial_froude', 0)
                
                jump_annotation.set_text(
                    f"Hydraulic Jump\n"
                    f"Type: {jump_type}\n"
                    f"Fr₁ = {initial_froude:.2f} → Fr₂ ≈ {0.3:.2f}\n"
                    f"y₁ = {y1:.2f}m → y₂ = {y2:.2f}m"
                )
                jump_annotation.set_visible(True)
                
                # Add hydraulic jump visualization
                try:
                    # Create water surface turbulence at jump
                    turbulence_x = np.linspace(jump_loc-2, jump_loc+10, 30)
                    turbulence_y = np.zeros_like(turbulence_x)
                    
                    # Generate a realistic turbulent surface
                    for i in range(len(turbulence_x)):
                        x_rel = (turbulence_x[i] - jump_loc) / 12
                        
                        if x_rel < 0:  # Approaching jump
                            turb_height = 0.05 * y1
                            base_height = jump_z + y1
                        elif x_rel < 0.3:  # Maximum turbulence
                            # Random-looking turbulence at the jump center
                            turb_height = 0.3 * y2 * (1 - x_rel/0.3) * (0.8 + 0.4 * np.sin(frame_idx + i * 0.5))
                            # Transition from y1 to y2
                            transition = min(1, x_rel / 0.2)
                            base_height = jump_z + y1 + transition * (y2 - y1)
                        else:  # After jump
                            turb_height = 0.1 * y2 * np.exp(-(x_rel-0.3)*5) * np.sin(x_rel*20 + frame_idx*0.3)
                            base_height = jump_z + y2
                        
                        # Add some time-varying randomness for animation
                        turb_height *= (0.8 + 0.4 * np.sin(frame_idx * 0.4 + i * 0.7))
                        
                        turbulence_y[i] = base_height + turb_height
                    
                    # Add the turbulence as a line
                    turb_line, = ax_profile.plot(turbulence_x, turbulence_y, 'white', alpha=0.7, linewidth=1.5)
                    regime_patches.append(turb_line)
                    
                    # Add white water effect
                    for i in range(1, len(turbulence_x)-1, 2):
                        # Create small white water droplets/foam
                        foam_size = 0.05 * y2 * (0.8 + 0.4 * np.sin(frame_idx * 0.7 + i))
                        
                        if i % 4 == 0 and turbulence_x[i] > jump_loc:
                            # Only add some droplets after the jump
                            foam_x = [turbulence_x[i] - foam_size, turbulence_x[i], 
                                    turbulence_x[i] + foam_size]
                            foam_y = [turbulence_y[i], turbulence_y[i] + foam_size, 
                                    turbulence_y[i]]
                            
                            foam_poly = Polygon(np.column_stack([foam_x, foam_y]), 
                                            facecolor='white', alpha=0.6, edgecolor=None)
                            regime_patches.append(foam_poly)
                            ax_profile.add_patch(foam_poly)
                except Exception as e:
                    # Skip jump visualization if it fails
                    print(f"Warning: Could not create jump turbulence: {str(e)}")
            else:
                jump_marker_profile.set_alpha(0.0)
                jump_marker_froude.set_alpha(0.0)
                jump_annotation.set_visible(False)
            
            # Update info text
            time_position = frame_idx / max(1, len(results_list) - 1)  # Avoid division by zero
            
            water_level_text = f"{upstream_level:.2f}m" if upstream_level > dam_crest else \
                            f"{upstream_level:.2f}m (below crest)"
            
            # Determine overall flow regime
            if len(froude_combined) > 0:
                # Only include points with positive Froude values
                fr_positive = froude_combined[froude_combined > 0]
                if len(fr_positive) > 0:
                    avg_froude = np.mean(fr_positive)
                    if avg_froude < 0.8:
                        regime_text = "Primarily Subcritical"
                    elif avg_froude > 1.2:
                        regime_text = "Primarily Supercritical"
                    else:
                        regime_text = "Mixed Flow Regime"
                else:
                    regime_text = "No Flow"
            else:
                regime_text = "No Flow"
                
            info_text.set_text(
                f"Time Step: {frame_idx+1}/{len(results_list)}\n"
                f"Water Level: {water_level_text}\n"
                f"Discharge: {discharge:.2f} m³/s\n"
                f"Flow Regime: {regime_text}"
            )
        else:
            # No valid tailwater data
            # Reset lines and markers
            froude_line.set_data([], [])
            jump_marker_profile.set_alpha(0.0)
            jump_marker_froude.set_alpha(0.0)
            jump_annotation.set_visible(False)
            
            # Update info text
            info_text.set_text(
                f"Time Step: {frame_idx+1}/{len(results_list)}\n"
                f"Water Level: {upstream_level:.2f}m\n"
                f"Discharge: {discharge:.2f} m³/s\n"
                f"Flow Regime: No data available"
            )
        
        # Update progress bar
        progress_bar.set_width(time_position * 0.8)
        
        # Return all dynamic elements
        return [froude_line, jump_marker_profile, jump_marker_froude, jump_annotation, 
                info_text, normal_line, critical_line, progress_bar] + water_patches + regime_patches
    
    # Create the animation with proper error handling
    try:
        anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                    interval=1000/fps, blit=True)
    except Exception as e:
        print(f"Warning: Could not create animation with blit=True: {str(e)}")
        # Try a simpler version as fallback
        anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                    interval=1000/fps, blit=False)
    
    plt.tight_layout()
    
    return anim, fig, (ax_profile, ax_froude)