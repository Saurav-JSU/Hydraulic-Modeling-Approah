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
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data for setup
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    dam_crest = scenario['dam_crest_elevation']
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile
    dam_profile = dam.get_profile(x_dam)
    
    # Set up plot limits based on all results
    max_x = 0
    max_y = dam_crest + 5  # Some space above crest
    min_y = dam_base - 1   # Some space below base
    
    for result in results_list:
        tailwater = result['tailwater']
        max_x = max(max_x, np.max(tailwater['x']))
        max_y = max(max_y, np.max(result['upstream_level']), np.max(tailwater['wse']))
    
    # Set axis limits with some padding
    ax.set_xlim(-20, min(200, max_x))
    ax.set_ylim(min_y, max_y + 1)
    
    # Plot static elements
    # Dam outline (will be filled dynamically to show proper water overlay)
    dam_line, = ax.plot(x_dam, dam_profile['z'], 'k-', linewidth=1.5, label='Dam')
    
    # Reference lines for normal and critical depths
    # These will be updated for each frame
    normal_line = ax.axhline(y=0, color='g', linestyle='--', alpha=0.7, 
                           label='Normal Depth', visible=False)
    critical_line = ax.axhline(y=0, color='r', linestyle=':', alpha=0.7, 
                             label='Critical Depth', visible=False)
    
    # Create color map
    cmap = plt.cm.get_cmap(colormap)
    
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
    fig.patches.extend([progress_bg, progress_bar])
    
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
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                       edgecolor='black', linewidth=0.5)
    ax.add_patch(dam_patch)
    
    # Function to update the animation for each frame
    def update(frame_idx):
        # Clear previous water patches
        for patch in water_patches:
            patch.remove()
        water_patches.clear()
        
        # Get the result for this frame
        result = results_list[frame_idx]
        
        # Extract data
        tailwater = result['tailwater']
        upstream_level = result['upstream_level']
        discharge = result['discharge']
        
        # Extract or compute coloring parameter
        if color_by.lower() == 'froude':
            color_param = tailwater['fr']
            # Set norm again to adjust for current frame data
            vmax = max(2, np.max(color_param)) if len(color_param) > 0 else 2
            sm.set_norm(colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax))
        elif color_by.lower() == 'velocity':
            color_param = tailwater['v']
            vmax = max(1, np.max(color_param)) if len(color_param) > 0 else 1
            sm.set_norm(colors.Normalize(vmin=0, vmax=vmax))
        elif color_by.lower() == 'depth':
            color_param = tailwater['y']
            vmax = max(1, np.max(color_param)) if len(color_param) > 0 else 1
            sm.set_norm(colors.Normalize(vmin=0, vmax=vmax))
        
        # Update reference lines for normal and critical depth
        yn = result['normal_depth']
        yc = result['critical_depth']
        
        if yn > 0:
            base_elevation = tailwater['z'][0] if len(tailwater['z']) > 0 else dam_base
            normal_line.set_ydata(base_elevation + yn)
            normal_line.set_visible(True)
        else:
            normal_line.set_visible(False)
            
        if yc > 0:
            base_elevation = tailwater['z'][0] if len(tailwater['z']) > 0 else dam_base
            critical_line.set_ydata(base_elevation + yc)
            critical_line.set_visible(True)
        else:
            critical_line.set_visible(False)
        
        # Create water surface and bed profiles
        x_values = tailwater['x']
        z = tailwater['z']
        wse = tailwater['wse']
        
        # Include upstream portion
        x_upstream = np.linspace(-20, 0, 20)
        z_upstream = np.full_like(x_upstream, dam_base)
        wse_upstream = np.full_like(x_upstream, upstream_level)
        
        # Combine arrays for complete profile
        x_combined = np.concatenate([x_upstream, x_values])
        z_combined = np.concatenate([z_upstream, z])
        wse_combined = np.concatenate([wse_upstream, wse])
        
        # Create coloring parameter for full domain
        full_color_param = np.zeros_like(x_combined)
        
        # Upstream values based on discharge
        upstream_depth = upstream_level - dam_base
        if upstream_depth > 0 and discharge > 0:
            # Approximate upstream values from continuity
            upstream_velocity = discharge / (upstream_depth * scenario['channel_width_at_dam'])
            upstream_froude = upstream_velocity / np.sqrt(9.81 * upstream_depth)
            upstream_color = upstream_froude if color_by.lower() == 'froude' else \
                             upstream_velocity if color_by.lower() == 'velocity' else \
                             upstream_depth
            full_color_param[:len(x_upstream)] = upstream_color
        
        # Downstream values from tailwater
        if len(color_param) > 0:
            full_color_param[len(x_upstream):] = color_param
        
        # Create water polygons for areas with depth > 0
        for i in range(len(x_combined)-1):
            # Skip if both points are at or below bed
            if wse_combined[i] <= z_combined[i] and wse_combined[i+1] <= z_combined[i+1]:
                continue
                
            # Create polygon for each segment
            x_vals = [x_combined[i], x_combined[i+1], x_combined[i+1], x_combined[i]]
            y_vals = [wse_combined[i], wse_combined[i+1], 
                      max(z_combined[i+1], wse_combined[i+1]), max(z_combined[i], wse_combined[i])]
            
            # Skip invalid polygons
            if len(set(y_vals)) <= 2:  # Not enough distinct y-values
                continue
                
            # Get color based on parameter value
            color_val = full_color_param[i]
            
            # Create different color effects based on parameter
            if color_by.lower() == 'froude':
                # Use colormap with transparency effect for Froude
                rgba_color = cmap(sm.norm(color_val))
                depth = wse_combined[i] - z_combined[i]
                
                # Add flow regime visual effects
                if color_val < 0.8:  # Subcritical - calm water
                    alpha = 0.5 + 0.3 * (depth / max(1, np.max(tailwater['y'] if len(tailwater['y']) > 0 else [1])))
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
            poly = Polygon(np.column_stack([x_vals, y_vals]), 
                          facecolor=rgba_color, edgecolor=None)
            water_patches.append(poly)
            ax.add_patch(poly)
        
        # Handle hydraulic jump if present
        jump = result['hydraulic_jump']
        if jump.get('jump_possible', False):
            jump_loc = jump['location']
            jump_marker.set_xdata([jump_loc])
            jump_marker.set_alpha(0.7)
            
            # Find the bed elevation at jump location
            jump_index = np.argmin(np.abs(x_values - jump_loc))
            jump_z = z[jump_index] if jump_index < len(z) else dam_base
            
            # Add annotation for jump
            y1 = jump['initial_depth']
            y2 = jump['sequent_depth']
            
            jump_annotation.xy = (jump_loc, jump_z + y2)
            jump_annotation.xytext = (jump_loc + 30, jump_z + y2 + 1)
            jump_annotation.set_text(
                f"Hydraulic Jump\n"
                f"Type: {jump['jump_type']}\n"
                f"Fr₁ = {jump['initial_froude']:.2f}\n"
                f"y₁ = {y1:.2f} m → y₂ = {y2:.2f} m"
            )
            jump_annotation.set_visible(True)
            
            # Create hydraulic jump turbulence effect
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
                base_y = jump_z + y2 if jump_x[i] >= jump_loc else jump_z + y1 + (y2-y1) * max(0, (jump_x[i] - (jump_loc-5))/5)
                
                y_vals = [base_y + wave_height, 
                         base_y + wave_height * (0.8 + 0.4 * np.sin(frame_idx * 0.4 + i + 1)),
                         base_y, base_y]
                
                # Color intensity varies with turbulence
                intensity = max(0, 1 - dist/8)
                alpha = 0.2 + 0.3 * intensity
                
                # Create wavy surface on top of the jump
                poly = Polygon(np.column_stack([x_vals, y_vals]), 
                              facecolor='white', alpha=alpha)
                water_patches.append(poly)
                ax.add_patch(poly)
        else:
            jump_marker.set_alpha(0.0)
            jump_annotation.set_visible(False)
        
        # Update info text
        time_position = frame_idx / (len(results_list) - 1)
        water_level_text = f"{upstream_level:.2f}m" if upstream_level > dam_crest else \
                          f"{upstream_level:.2f}m (below crest)"
        
        info_text.set_text(
            f"Time Step: {frame_idx+1}/{len(results_list)}\n"
            f"Water Level: {water_level_text}\n"
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Head: {result['head']:.2f}m"
        )
        
        # Update progress bar
        progress_bar.set_width(time_position * 0.8)
        
        return water_patches + [jump_marker, jump_annotation, info_text, 
                               normal_line, critical_line, progress_bar]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                 interval=1000/fps, blit=True)
    
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
    # Create figure with two subplots - profile and Froude number
    fig, (ax_profile, ax_froude) = plt.subplots(2, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract data for setup
    dam = scenario['dam']
    dam_base = scenario['dam_base_elevation']
    dam_crest = scenario['dam_crest_elevation']
    
    # Create x range for dam
    x_dam = np.linspace(-20, 5, 50)
    
    # Get dam profile
    dam_profile = dam.get_profile(x_dam)
    
    # Set up plot limits based on all results
    max_x = 0
    max_y = dam_crest + 5  # Some space above crest
    min_y = dam_base - 1   # Some space below base
    max_fr = 0
    
    for result in results_list:
        tailwater = result['tailwater']
        max_x = max(max_x, np.max(tailwater['x']))
        max_y = max(max_y, np.max(result['upstream_level']), np.max(tailwater['wse']))
        if len(tailwater['fr']) > 0:
            max_fr = max(max_fr, np.max(tailwater['fr']))
    
    # Ensure reasonable Froude axis maximum
    max_fr = max(max_fr, 2.0) * 1.1  # Add 10% padding
    
    # Set axis limits with some padding
    ax_profile.set_xlim(-20, min(200, max_x))
    ax_profile.set_ylim(min_y, max_y + 1)
    ax_froude.set_xlim(-20, min(200, max_x))
    ax_froude.set_ylim(0, max_fr)
    
    # Plot static elements in profile view
    # Dam outline
    dam_line, = ax_profile.plot(x_dam, dam_profile['z'], 'k-', linewidth=1.5, label='Dam')
    
    # Add dam body
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                       edgecolor='black', linewidth=0.5)
    ax_profile.add_patch(dam_patch)
    
    # Reference lines for normal and critical depths
    normal_line = ax_profile.axhline(y=0, color='g', linestyle='--', alpha=0.7, 
                                   label='Normal Depth', visible=False)
    critical_line = ax_profile.axhline(y=0, color='r', linestyle=':', alpha=0.7, 
                                     label='Critical Depth', visible=False)
    
    # Critical Froude line in Froude plot
    ax_froude.axhline(y=1.0, color='r', linestyle='--', 
                     label='Critical Flow (Fr=1)', linewidth=1.5)
    
    # Fill regions for flow regimes in Froude plot
    ax_froude.fill_between([ax_froude.get_xlim()[0], ax_froude.get_xlim()[1]], 
                          0, 1, color='blue', alpha=0.2, label='Subcritical')
    ax_froude.fill_between([ax_froude.get_xlim()[0], ax_froude.get_xlim()[1]], 
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
    fig.patches.extend([progress_bg, progress_bar])
    
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
        
        # Get the result for this frame
        result = results_list[frame_idx]
        
        # Extract data
        tailwater = result['tailwater']
        upstream_level = result['upstream_level']
        discharge = result['discharge']
        
        # Update reference lines for normal and critical depth
        yn = result['normal_depth']
        yc = result['critical_depth']
        
        if yn > 0:
            base_elevation = tailwater['z'][0] if len(tailwater['z']) > 0 else dam_base
            normal_line.set_ydata(base_elevation + yn)
            normal_line.set_visible(True)
        else:
            normal_line.set_visible(False)
            
        if yc > 0:
            base_elevation = tailwater['z'][0] if len(tailwater['z']) > 0 else dam_base
            critical_line.set_ydata(base_elevation + yc)
            critical_line.set_visible(True)
        else:
            critical_line.set_visible(False)
        
        # Create water surface and bed profiles
        x_values = tailwater['x']
        z = tailwater['z']
        wse = tailwater['wse']
        
        # Include upstream portion
        x_upstream = np.linspace(-20, 0, 20)
        z_upstream = np.full_like(x_upstream, dam_base)
        wse_upstream = np.full_like(x_upstream, upstream_level)
        
        # Combine arrays for complete profile
        x_combined = np.concatenate([x_upstream, x_values])
        z_combined = np.concatenate([z_upstream, z])
        wse_combined = np.concatenate([wse_upstream, wse])
        
        # Create and color water polygons based on flow regime
        froude_combined = np.zeros_like(x_combined)
        
        # Upstream values
        upstream_depth = upstream_level - dam_base
        if upstream_depth > 0 and discharge > 0:
            upstream_velocity = discharge / (upstream_depth * scenario['channel_width_at_dam'])
            upstream_froude = upstream_velocity / np.sqrt(9.81 * upstream_depth)
            froude_combined[:len(x_upstream)] = upstream_froude
        
        # Downstream values
        if len(tailwater['fr']) > 0:
            froude_combined[len(x_upstream):] = tailwater['fr']
        
        # Create water polygons
        for i in range(len(x_combined)-1):
            # Skip if both points are at or below bed
            if wse_combined[i] <= z_combined[i] and wse_combined[i+1] <= z_combined[i+1]:
                continue
                
            # Create polygon for each segment
            x_vals = [x_combined[i], x_combined[i+1], x_combined[i+1], x_combined[i]]
            y_vals = [wse_combined[i], wse_combined[i+1], 
                      max(z_combined[i+1], wse_combined[i+1]), max(z_combined[i], wse_combined[i])]
            
            # Skip invalid polygons
            if len(set(y_vals)) <= 2:
                continue
                
            # Color based on flow regime
            fr_val = froude_combined[i]
            
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
            poly = Polygon(np.column_stack([x_vals, y_vals]), 
                          facecolor=color, alpha=alpha, edgecolor=None)
            water_patches.append(poly)
            ax_profile.add_patch(poly)
            
            # Add dynamic flow regime indicator patches
            # Only add for some segments to avoid clutter
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
        
        # Update Froude number plot
        froude_line.set_data(x_combined, froude_combined)
        
        # Handle hydraulic jump
        jump = result['hydraulic_jump']
        if jump.get('jump_possible', False):
            jump_loc = jump['location']
            
            # Update jump markers
            jump_marker_profile.set_xdata([jump_loc])
            jump_marker_profile.set_alpha(0.7)
            
            jump_marker_froude.set_xdata([jump_loc])
            jump_marker_froude.set_alpha(0.7)
            
            # Find the bed elevation at jump location
            jump_index = np.argmin(np.abs(x_values - jump_loc))
            jump_z = z[jump_index] if jump_index < len(z) else dam_base
            
            # Add annotation for jump
            y1 = jump['initial_depth']
            y2 = jump['sequent_depth']
            
            jump_annotation.xy = (jump_loc, jump_z + y2)
            jump_annotation.xytext = (jump_loc + 30, jump_z + y2 + 1)
            jump_annotation.set_text(
                f"Hydraulic Jump\n"
                f"Type: {jump['jump_type']}\n"
                f"Fr₁ = {jump['initial_froude']:.2f} → Fr₂ ≈ {0.3:.2f}\n"
                f"y₁ = {y1:.2f}m → y₂ = {y2:.2f}m"
            )
            jump_annotation.set_visible(True)
            
            # Add hydraulic jump visualization
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
            
            # Create polygon for jump turbulence
            turb_poly = np.column_stack([
                turbulence_x,
                turbulence_y
            ])
            
            # Add the turbulence as a line collection
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
        else:
            jump_marker_profile.set_alpha(0.0)
            jump_marker_froude.set_alpha(0.0)
            jump_annotation.set_visible(False)
        
        # Update info text
        time_position = frame_idx / (len(results_list) - 1)
        water_level_text = f"{upstream_level:.2f}m" if upstream_level > dam_crest else \
                          f"{upstream_level:.2f}m (below crest)"
        
        # Determine overall flow regime
        if len(froude_combined) > 0:
            avg_froude = np.mean(froude_combined[froude_combined > 0])
            if avg_froude < 0.8:
                regime_text = "Primarily Subcritical"
            elif avg_froude > 1.2:
                regime_text = "Primarily Supercritical"
            else:
                regime_text = "Mixed Flow Regime"
        else:
            regime_text = "No Flow"
            
        info_text.set_text(
            f"Time Step: {frame_idx+1}/{len(results_list)}\n"
            f"Water Level: {water_level_text}\n"
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Flow Regime: {regime_text}"
        )
        
        # Update progress bar
        progress_bar.set_width(time_position * 0.8)
        
        # Return all dynamic elements
        return [froude_line, jump_marker_profile, jump_marker_froude, jump_annotation, 
                info_text, normal_line, critical_line, progress_bar] + water_patches + regime_patches
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(results_list),
                                 interval=1000/fps, blit=True)
    
    plt.tight_layout()
    
    return anim, fig, (ax_profile, ax_froude)