"""
Flow regime visualizations for hydraulic modeling.

This module provides specialized visualizations for different flow regimes,
including subcritical, supercritical, critical flow, and transitions between regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patheffects as PathEffects

def plot_flow_regime_profile(scenario, results, ax=None, display_range=None, 
                            show_annotations=True, detailed_regimes=True):
    """
    Create a visualization highlighting flow regimes along the channel.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        ax (matplotlib.axes.Axes, optional): The axes to plot on
        display_range (tuple): The x-range to display (min, max)
        show_annotations (bool): Whether to show detailed annotations
        detailed_regimes (bool): Whether to show detailed regime transitions
        
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
    
    # Plot bed profile
    ax.plot(combined_x, bed, 'k-', linewidth=1.5, label='Channel Bed')
    
    # Fill the dam body
    try:
        # Create dam polygon safely
        if 'z' in dam_profile and len(dam_profile['z']) == len(x_dam):
            dam_poly = np.column_stack([
                np.concatenate([x_dam, x_dam[::-1]]),
                np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
            ])
            dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                               edgecolor='black', linewidth=0.5)
            ax.add_patch(dam_patch)
    except Exception as e:
        # If dam visualization fails, at least draw a simple dam shape
        dam_x = [0, 0]
        dam_y = [dam_base, results.get('dam_crest_elevation', dam_base + 10)]
        ax.plot(dam_x, dam_y, 'k-', linewidth=2)
    
    # Create Froude number array for the entire domain
    froude = np.zeros_like(combined_x)
    
    # Fill upstream portion
    upstream_level = results.get('upstream_level', dam_base)
    upstream_depth = max(0, upstream_level - dam_base)  # Ensure non-negative depth
    discharge = results.get('discharge', 0)
    
    if upstream_depth > 0 and discharge > 0:
        # Get channel parameters
        channel_width = scenario.get('channel_width_at_dam', 5.0)
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
        
        # Avoid division by zero in Froude calculation
        hydraulic_depth = max(hydraulic_depth, 0.001)
        upstream_froude = upstream_velocity / np.sqrt(9.81 * hydraulic_depth)
        
        # Fill upstream portion of Froude array
        froude[combined_x <= 0] = upstream_froude
    
    # Fill downstream portion
    for i, x in enumerate(combined_x):
        if x <= 0:
            continue  # Already filled upstream portion
            
        # Find closest point in tailwater results
        idx = np.argmin(np.abs(x_values - x))
        if idx < len(tailwater.get('fr', [])):
            froude[i] = tailwater['fr'][idx]
    
    # Create an array to store flow regime categories
    # 0: No flow, 1: Subcritical, 2: Critical, 3: Supercritical
    flow_regime = np.zeros_like(froude, dtype=int)
    
    # Classify flow regimes
    flow_regime[(froude > 0) & (froude < 0.8)] = 1  # Subcritical
    flow_regime[(froude >= 0.8) & (froude <= 1.2)] = 2  # Near critical
    flow_regime[froude > 1.2] = 3  # Supercritical
    
    # Colors for each flow regime
    regime_colors = ['white', 'blue', 'green', 'red']
    regime_alphas = [0.0, 0.5, 0.6, 0.5]
    
    # Color water based on flow regime
    for i in range(1, len(combined_x)):
        # Skip if both points are at or below bed
        if water_surface[i-1] <= bed[i-1] and water_surface[i] <= bed[i]:
            continue
            
        # Create polygon for this segment
        x_vals = [combined_x[i-1], combined_x[i], combined_x[i], combined_x[i-1]]
        
        # Ensure water surface is at or above bed level at each point
        y_surface_i_minus_1 = max(water_surface[i-1], bed[i-1])
        y_surface_i = max(water_surface[i], bed[i])
        
        y_vals = [y_surface_i_minus_1, y_surface_i, bed[i], bed[i-1]]
        
        # Skip invalid polygons - need at least 3 distinct points to form polygon
        if len(set(zip(x_vals, y_vals))) < 3:
            continue
            
        # Get regime for this segment - ensure valid index
        regime = flow_regime[i] if i < len(flow_regime) else 0
        
        if regime > 0 and regime < len(regime_colors):
            # Create polygon with regime color
            poly = Polygon(np.column_stack([x_vals, y_vals]), 
                          facecolor=regime_colors[regime], 
                          alpha=regime_alphas[regime], 
                          edgecolor=None)
            ax.add_patch(poly)
    
    # Plot water surface with enhanced styling
    ax.plot(combined_x, water_surface, 'b-', linewidth=2, label='Water Surface')
    
    # Add regime classification zones
    if detailed_regimes:
        # Find segments of consistent flow regime
        regime_segments = []
        current_regime = flow_regime[0]
        start_idx = 0
        
        for i in range(1, len(flow_regime)):
            if flow_regime[i] != current_regime:
                # Regime change detected
                if current_regime > 0:  # Skip no-flow segments
                    regime_segments.append({
                        'regime': current_regime,
                        'start_x': combined_x[start_idx],
                        'end_x': combined_x[i-1],
                        'start_wse': water_surface[start_idx],
                        'end_wse': water_surface[i-1]
                    })
                # Start new segment
                current_regime = flow_regime[i]
                start_idx = i
        
        # Add the last segment
        if current_regime > 0:
            regime_segments.append({
                'regime': current_regime,
                'start_x': combined_x[start_idx],
                'end_x': combined_x[-1],
                'start_wse': water_surface[start_idx],
                'end_wse': water_surface[-1]
            })
        
        # Add regime annotations
        for segment in regime_segments:
            # Skip very short segments
            if abs(segment['end_x'] - segment['start_x']) < 5:
                continue
                
            # Get regime properties
            regime = segment['regime']
            mid_x = (segment['start_x'] + segment['end_x']) / 2
            mid_wse = (segment['start_wse'] + segment['end_wse']) / 2
            
            # Determine label based on regime
            if regime == 1:
                label = "Subcritical Flow"
                color = 'blue'
            elif regime == 2:
                label = "Critical Flow"
                color = 'green'
            elif regime == 3:
                label = "Supercritical Flow"
                color = 'red'
            else:
                continue  # Skip unknown regimes
            
            # Mark regime zone
            if show_annotations:
                ax.annotate(label, xy=(mid_x, mid_wse + 0.5), 
                          ha='center', va='bottom', color=color,
                          bbox=dict(boxstyle="round,pad=0.3", 
                                   fc='white', ec=color, alpha=0.8))
    
    # Add normal and critical depth lines
    yn = results.get('normal_depth', 0)
    yc = results.get('critical_depth', 0)
    
    # Only show if we have flow and positive depths
    if results.get('discharge', 0) > 0 and yn > 0 and yc > 0:
        # Get base elevation from tailwater if available
        if len(tailwater.get('z', [])) > 0:
            base_elevation = tailwater['z'][0]
        else:
            base_elevation = dam_base
            
        # Normal depth line
        normal_line = ax.axhline(y=base_elevation + yn, color='green', linestyle='--', 
                               alpha=0.7, linewidth=1.5, 
                               label=f'Normal Depth ({yn:.2f} m)')
        
        # Critical depth line
        critical_line = ax.axhline(y=base_elevation + yc, color='red', linestyle=':', 
                                 alpha=0.7, linewidth=1.5, 
                                 label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
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
            
            # Mark the jump
            ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                      alpha=0.7, linewidth=2, label='Hydraulic Jump')
            
            # Add enhanced jump visualization
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
            
            # Create a region showing the jump
            jump_width = 15  # Width of jump visualization
            
            # Create jump patch
            jump_patch = Rectangle((jump_loc, jump_z), jump_width, y2,
                                  facecolor='magenta', alpha=0.3)
            ax.add_patch(jump_patch)
            
            # Add wavy pattern to represent turbulence
            n_waves = 30
            wave_x = np.linspace(jump_loc, jump_loc + jump_width, n_waves)
            wave_amp = 0.15 * y2
            wave_y = jump_z + y2 + wave_amp * np.sin(np.linspace(0, 4*np.pi, n_waves))
            
            ax.plot(wave_x, wave_y, 'white', linewidth=1.5, alpha=0.7)
            
            # Add a fancy arrow showing the jump
            arrow = FancyArrowPatch((jump_loc - 5, jump_z + y1), 
                                  (jump_loc + 5, jump_z + y2),
                                  connectionstyle="arc3,rad=0.3",
                                  arrowstyle="-|>", color='magenta',
                                  linewidth=2, alpha=0.9)
            ax.add_patch(arrow)
            
            # Add annotation
            if show_annotations and 'initial_froude' in jump:
                ax.annotate(
                    f"Hydraulic Jump\n"
                    f"Fr₁ = {jump.get('initial_froude', 0):.2f} → Fr₂ < 1\n"
                    f"y₁ = {y1:.2f}m → y₂ = {y2:.2f}m\n"
                    f"Type: {jump.get('jump_type', 'Unknown')}",
                    xy=(jump_loc + jump_width/2, jump_z + y2 + wave_amp),
                    xytext=(jump_loc + jump_width + 10, jump_z + y2 + wave_amp + 0.5),
                    ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                    fontsize=9
                )
    
    # Add Froude number legend
    # Create a custom legend for flow regimes
    subcritical_patch = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.5)
    critical_patch = Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.6)
    supercritical_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5)
    
    # Only add legend elements if we have flow
    if results.get('discharge', 0) > 0:
        legend_elements = [
            subcritical_patch,
            critical_patch,
            supercritical_patch
        ]
        
        legend_labels = [
            f"Subcritical Flow (Fr < 0.8)",
            f"Near Critical Flow (0.8 ≤ Fr ≤ 1.2)",
            f"Supercritical Flow (Fr > 1.2)"
        ]
        
        # Create a separate legend for flow regimes
        regime_legend = ax.legend(legend_elements, legend_labels,
                                loc='upper left', framealpha=0.9)
        
        # Add the regime legend manually
        ax.add_artist(regime_legend)
        
        # Create standard legend for other elements
        ax.legend(loc='upper right', framealpha=0.9)
    else:
        ax.legend(loc='upper right')
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range
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
    
    # Add title and labels
    ax.set_title('Flow Regime Visualization', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Dam (m)', fontsize=12)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add info about discharge and other key parameters
    if show_annotations:
        # Position info text in upper left corner
        discharge = results.get('discharge', 0)
        upstream_level = results.get('upstream_level', dam_base)
        dam_crest = scenario.get('dam_crest_elevation', dam_base + 10)
        head = max(0, upstream_level - dam_crest)
        
        info_text = (
            f"Discharge: {discharge:.2f} m³/s\n"
            f"Head over crest: {head:.2f} m\n"
            f"Normal depth: {yn:.2f} m\n"
            f"Critical depth: {yc:.2f} m"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
              ha='left', va='top', fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    return fig, ax


def plot_froude_profile(scenario, results, ax=None, display_range=None):
    """
    Create a dedicated visualization of Froude number along the channel.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        ax (matplotlib.axes.Axes, optional): The axes to plot on
        display_range (tuple): The x-range to display (min, max)
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure
    
    # Extract data safely
    tailwater = results.get('tailwater', {})
    x_values = tailwater.get('x', [])
    
    # Check if we have data to work with
    if len(x_values) == 0:
        ax.text(0.5, 0.5, "No x coordinates available in tailwater data", 
              ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Include upstream portion
    x_upstream = np.linspace(-20, 0, 20)
    
    # Estimate upstream Froude number
    upstream_level = results.get('upstream_level', 0)
    dam_base = scenario.get('dam_base_elevation', 0)
    upstream_depth = max(0, upstream_level - dam_base)  # Ensure non-negative depth
    discharge = results.get('discharge', 0)
    
    # Initialize with zeros
    fr_upstream = np.zeros_like(x_upstream)
    
    if upstream_depth > 0 and discharge > 0:
        # Get channel parameters
        channel_width = scenario.get('channel_width_at_dam', 5.0)
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
        
        # Avoid division by zero in Froude calculation
        hydraulic_depth = max(hydraulic_depth, 0.001)
        upstream_froude = upstream_velocity / np.sqrt(9.81 * hydraulic_depth)
        
        # Set upstream Froude values
        fr_upstream.fill(upstream_froude)
    
    # Combine arrays for complete profile
    x_combined = np.concatenate([x_upstream, x_values])
    
    # Create Froude number array
    fr_combined = np.zeros_like(x_combined)
    fr_combined[:len(x_upstream)] = fr_upstream
    
    # Fill downstream portion from tailwater data
    if 'fr' in tailwater and len(tailwater['fr']) > 0:
        # Handle case where arrays might have different lengths
        n_values = min(len(tailwater['fr']), len(x_values))
        fr_combined[len(x_upstream):len(x_upstream) + n_values] = tailwater['fr'][:n_values]
    
    # Create a colormap for the Froude number
    cmap = plt.cm.coolwarm
    
    # Calculate max Froude safely
    max_fr = np.max(fr_combined) if len(fr_combined) > 0 else 2
    max_fr = max(2, max_fr)  # Ensure reasonable vmax
    
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max_fr)
    
    # Plot Froude profile with colored line
    points = np.array([x_combined, fr_combined]).T.reshape(-1, 1, 2)
    
    # Ensure we have at least 2 points for segments
    if len(points) >= 2:
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(fr_combined[:-1])
        lc.set_linewidth(3)
        line = ax.add_collection(lc)
        
        # Add colorbar
        cbar = fig.colorbar(line, ax=ax)
        cbar.set_label('Froude Number')
        
        # Add regime labels to the colorbar
        cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                    rotation=90, transform=cbar.ax.transAxes, color='black')
        cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                    rotation=90, transform=cbar.ax.transAxes, color='black')
    else:
        # Not enough points for LineCollection
        ax.text(0.5, 0.5, "Insufficient data for Froude profile", 
              ha='center', va='center', transform=ax.transAxes)
    
    # Add reference line at Fr = 1 (critical flow)
    ax.axhline(y=1.0, color='black', linestyle='--', 
              label='Critical Flow (Fr = 1)', linewidth=1.5)
    
    # Add shaded regions for flow regimes
    ax.fill_between(x_combined, 0, 0.8, color='blue', alpha=0.2, 
                   label='Subcritical (Fr < 0.8)')
    ax.fill_between(x_combined, 0.8, 1.2, color='green', alpha=0.2, 
                   label='Near Critical (0.8 ≤ Fr ≤ 1.2)')
    ax.fill_between(x_combined, 1.2, max_fr + 0.5, 
                   color='red', alpha=0.2, label='Supercritical (Fr > 1.2)')
    
    # Mark hydraulic jump if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        # Mark with vertical line
        ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                  linewidth=2, label='Hydraulic Jump')
        
        # Get initial Froude number
        fr1 = jump.get('initial_froude', 0)
        
        # Only add arrow if Fr1 is valid
        if fr1 > 0:
            # Add a curve arrow from Fr1 to subcritical
            arrow = FancyArrowPatch((jump_loc - 3, fr1), (jump_loc + 3, 0.5),
                                  connectionstyle="arc3,rad=0.3",
                                  arrowstyle="-|>", color='magenta',
                                  linewidth=2, alpha=0.9)
            ax.add_patch(arrow)
            
            # Add annotation
            ax.annotate(
                f"Hydraulic Jump\n"
                f"Fr₁ = {fr1:.2f} → Fr₂ < 1",
                xy=(jump_loc, fr1),
                xytext=(jump_loc + 15, fr1 + 0.5),
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2")
            )
    
    # Set display range
    if display_range:
        ax.set_xlim(display_range)
    else:
        # Set a good default range
        if len(x_values) > 0:
            ax.set_xlim(-20, min(200, np.max(x_values)))
        else:
            ax.set_xlim(-20, 200)
    
    # Set y limit with some padding
    max_fr_with_padding = max_fr * 1.2
    ax.set_ylim(0, max(2, max_fr_with_padding))
    
    # Add title and labels
    ax.set_title('Froude Number Profile', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance from Dam (m)', fontsize=12)
    ax.set_ylabel('Froude Number', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right')
    
    return fig, ax


def create_flow_regime_dashboard(scenario, results, figsize=(14, 10)):
    """
    Create a comprehensive dashboard showing flow regimes with multiple
    visualizations of different aspects of the flow.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], figure=fig)
    
    # Create axes
    ax_profile = fig.add_subplot(gs[0, :])  # Main profile - spans both columns
    ax_froude = fig.add_subplot(gs[1, 0])   # Froude profile
    ax_velocity = fig.add_subplot(gs[1, 1])  # Velocity profile
    ax_energy = fig.add_subplot(gs[2, 0])    # Energy profile
    ax_shear = fig.add_subplot(gs[2, 1])     # Shear stress profile
    
    # Plot main profile with flow regimes
    plot_flow_regime_profile(scenario, results, ax=ax_profile)
    
    # Plot Froude profile
    plot_froude_profile(scenario, results, ax=ax_froude)
    
    # Extract data for other plots
    tailwater = results.get('tailwater', {})
    x_values = tailwater.get('x', [])
    
    # Check if we have sufficient data
    if len(x_values) == 0:
        # Not enough data, add a warning in all remaining panels
        for ax in [ax_velocity, ax_energy, ax_shear]:
            ax.text(0.5, 0.5, "Insufficient data for plot", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add titles to all axes
        ax_velocity.set_title('Velocity Profile')
        ax_energy.set_title('Specific Energy Profile')
        ax_shear.set_title('Bed Shear Stress Profile')
        
        # Add overall title
        fig.suptitle('Flow Regime Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        return fig, (ax_profile, ax_froude, ax_velocity, ax_energy, ax_shear)
    
    # Velocity profile
    if 'v' in tailwater and len(tailwater['v']) > 0:
        # Ensure velocities array matches x_values length
        velocities = tailwater['v']
        if len(velocities) == len(x_values):
            ax_velocity.plot(x_values, velocities, 'g-', linewidth=2)
        else:
            # Create subset for plotting if arrays don't match
            plot_length = min(len(x_values), len(velocities))
            ax_velocity.plot(x_values[:plot_length], velocities[:plot_length], 'g-', linewidth=2)
    else:
        # No velocity data, create empty plot
        ax_velocity.text(0.5, 0.5, "No velocity data available", 
                       ha='center', va='center', transform=ax_velocity.transAxes)
    
    ax_velocity.set_title('Velocity Profile')
    ax_velocity.set_xlabel('Distance from Dam (m)')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in velocity plot if present
    jump = results.get('hydraulic_jump', {})
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        ax_velocity.axvline(x=jump_loc, color='magenta', linestyle='-.', 
                          linewidth=1.5, alpha=0.7)
        
        # Check if velocities are available
        if 'v' in tailwater and len(tailwater['v']) > 0:
            # Find velocity values before and after jump safely
            jump_index = -1
            min_dist = float('inf')
            for i, x in enumerate(x_values):
                dist = abs(x - jump_loc)
                if dist < min_dist:
                    min_dist = dist
                    jump_index = i
            
            if jump_index >= 0 and jump_index < len(tailwater['v']):
                v1 = tailwater['v'][jump_index]
                
                # Find a suitable index for the post-jump velocity
                post_jump_index = -1
                for i, x in enumerate(x_values):
                    if x > jump_loc + 3 and i < len(tailwater['v']):
                        post_jump_index = i
                        break
                
                # If we found a valid post-jump index, add an arrow
                if post_jump_index >= 0 and post_jump_index < len(tailwater['v']):
                    v2 = tailwater['v'][post_jump_index]
                    
                    # Add arrow showing velocity reduction
                    arrow = FancyArrowPatch((jump_loc - 3, v1), (jump_loc + 3, v2),
                                          connectionstyle="arc3,rad=0.3",
                                          arrowstyle="-|>", color='magenta',
                                          linewidth=2, alpha=0.8)
                    ax_velocity.add_patch(arrow)
    
    # Energy profile
    if 'energy' in tailwater and 'z' in tailwater and len(tailwater['energy']) > 0 and len(tailwater['z']) > 0:
        # Ensure arrays match x_values length
        energy = tailwater['energy']
        bed = tailwater['z']
        
        if len(energy) == len(x_values) and len(bed) == len(x_values):
            # Calculate specific energy
            specific_energy = energy - bed
            ax_energy.plot(x_values, specific_energy, 'purple', linewidth=2)
        else:
            # Create subset for plotting if arrays don't match
            plot_length = min(len(x_values), len(energy), len(bed))
            specific_energy = energy[:plot_length] - bed[:plot_length]
            ax_energy.plot(x_values[:plot_length], specific_energy, 'purple', linewidth=2)
    else:
        # Energy not directly available, try to estimate from depth and velocity
        if 'y' in tailwater and 'v' in tailwater and len(tailwater['y']) > 0 and len(tailwater['v']) > 0:
            # Ensure arrays match x_values length
            depths = tailwater['y']
            velocities = tailwater['v']
            
            plot_length = min(len(x_values), len(depths), len(velocities))
            
            if plot_length > 0:
                # Estimate specific energy as depth + velocity head
                specific_energy = np.zeros(plot_length)
                for i in range(plot_length):
                    depth = depths[i]
                    velocity = velocities[i]
                    # Handle potential zeros in depth
                    if depth > 0:
                        # E = y + v²/2g
                        specific_energy[i] = depth + (velocity**2) / (2 * 9.81)
                    else:
                        specific_energy[i] = 0
                
                ax_energy.plot(x_values[:plot_length], specific_energy, 'purple', linewidth=2)
            else:
                ax_energy.text(0.5, 0.5, "Insufficient data for energy calculation", 
                             ha='center', va='center', transform=ax_energy.transAxes)
        else:
            ax_energy.text(0.5, 0.5, "No energy data available", 
                         ha='center', va='center', transform=ax_energy.transAxes)
    
    ax_energy.set_title('Specific Energy Profile')
    ax_energy.set_xlabel('Distance from Dam (m)')
    ax_energy.set_ylabel('Specific Energy (m)')
    ax_energy.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in energy plot if present
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        ax_energy.axvline(x=jump_loc, color='magenta', linestyle='-.', 
                        linewidth=1.5, alpha=0.7)
        
        # Add annotation for energy loss
        jump_index = -1
        min_dist = float('inf')
        for i, x in enumerate(x_values):
            dist = abs(x - jump_loc)
            if dist < min_dist:
                min_dist = dist
                jump_index = i
        
        # Check if we have energy values before and after jump
        if 'energy' in tailwater and jump_index >= 0 and jump_index < len(tailwater.get('energy', [])):
            # Find a suitable index for post-jump energy
            post_jump_index = -1
            for i, x in enumerate(x_values):
                if x > jump_loc + 5 and i < len(tailwater.get('energy', [])):
                    post_jump_index = i
                    break
            
            if post_jump_index >= 0 and post_jump_index < len(tailwater.get('energy', [])) and \
               jump_index < len(tailwater.get('z', [])) and post_jump_index < len(tailwater.get('z', [])):
                
                # Calculate energy loss across jump
                e1 = tailwater['energy'][jump_index] - tailwater['z'][jump_index]
                e2 = tailwater['energy'][post_jump_index] - tailwater['z'][post_jump_index]
                energy_loss = max(0, e1 - e2)  # Ensure non-negative loss
                
                # Check if we have a meaningful loss
                if energy_loss > 0.01:  # Only annotate if loss is significant
                    ax_energy.annotate(
                        f"Energy Loss\n{energy_loss:.2f} m",
                        xy=(jump_loc, e1),
                        xytext=(jump_loc + 15, e1 + 0.2),
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                        fontsize=9
                    )
    
    # Shear stress profile (improved calculation)
    if 'y' in tailwater and len(tailwater['y']) > 0:
        # Get required data
        depths = tailwater['y']
        plot_length = min(len(x_values), len(depths))
        
        if plot_length > 0:
            # Get channel parameters
            channel_bottom_width = scenario.get('channel_bottom_width', 5.0)
            channel_side_slope = scenario.get('channel_side_slope', 0)
            
            # Get energy slope if available, or use bed slope as approximation
            if 's' in tailwater and len(tailwater['s']) == len(x_values):
                slope = tailwater['s']
            else:
                slope = np.full(plot_length, scenario.get('downstream_slope', 0.001))
            
            # Physical constants
            rho = 1000  # Water density (kg/m³)
            g = 9.81    # Gravity (m/s²)
            
            # Calculate hydraulic radius and shear stress for each point
            hydraulic_radius = np.zeros(plot_length)
            shear_stress = np.zeros(plot_length)
            
            for i in range(plot_length):
                depth = depths[i]
                if depth > 0:
                    # Calculate wetted perimeter and area
                    if channel_side_slope > 0:
                        # Trapezoidal channel
                        # Top width = bottom width + 2 * side_slope * depth
                        top_width = channel_bottom_width + 2 * channel_side_slope * depth
                        
                        # Area = (bottom width + top width) * depth / 2
                        area = (channel_bottom_width + top_width) * depth / 2
                        
                        # Wetted perimeter = bottom width + 2 * sloped sides
                        sloped_sides = depth * np.sqrt(1 + channel_side_slope**2)
                        wetted_perimeter = channel_bottom_width + 2 * sloped_sides
                    else:
                        # Rectangular channel
                        area = channel_bottom_width * depth
                        wetted_perimeter = channel_bottom_width + 2 * depth
                    
                    # Ensure non-zero wetted perimeter
                    wetted_perimeter = max(wetted_perimeter, 0.001)
                    
                    # Calculate hydraulic radius
                    hydraulic_radius[i] = area / wetted_perimeter
                    
                    # Get slope for this point
                    s = slope[i] if i < len(slope) else scenario.get('downstream_slope', 0.001)
                    
                    # Calculate shear stress using τ = ρgRS where R is hydraulic radius
                    shear_stress[i] = rho * g * hydraulic_radius[i] * s
            
            # Plot shear stress
            ax_shear.plot(x_values[:plot_length], shear_stress, 'brown', linewidth=2)
        else:
            ax_shear.text(0.5, 0.5, "Insufficient data for shear stress calculation", 
                         ha='center', va='center', transform=ax_shear.transAxes)
    else:
        ax_shear.text(0.5, 0.5, "No depth data available for shear stress calculation", 
                     ha='center', va='center', transform=ax_shear.transAxes)
    
    ax_shear.set_title('Bed Shear Stress Profile')
    ax_shear.set_xlabel('Distance from Dam (m)')
    ax_shear.set_ylabel('Shear Stress (N/m²)')
    ax_shear.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in shear stress plot if present
    if jump.get('jump_possible', False) and 'location' in jump:
        jump_loc = jump['location']
        ax_shear.axvline(x=jump_loc, color='magenta', linestyle='-.', 
                       linewidth=1.5, alpha=0.7)
    
    # Add overall title
    fig.suptitle('Flow Regime Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    return fig, (ax_profile, ax_froude, ax_velocity, ax_energy, ax_shear)


def plot_regime_map(scenarios, dam_heights, discharges, figsize=(10, 8)):
    """
    Create a flow regime map showing how flow characteristics change
    with different dam heights and discharges.
    
    Parameters:
        scenarios (list): List of scenario results
        dam_heights (array): Array of dam heights used
        discharges (array): Array of discharges used
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Validate inputs
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        ax.text(0.5, 0.5, "No scenario data provided", 
              ha='center', va='center', transform=ax.transAxes)
        return fig, ax
        
    if not isinstance(dam_heights, (list, np.ndarray)) or len(dam_heights) == 0:
        ax.text(0.5, 0.5, "No dam height data provided", 
              ha='center', va='center', transform=ax.transAxes)
        return fig, ax
        
    if not isinstance(discharges, (list, np.ndarray)) or len(discharges) == 0:
        ax.text(0.5, 0.5, "No discharge data provided", 
              ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Convert to numpy arrays if they aren't already
    dam_heights = np.array(dam_heights)
    discharges = np.array(discharges)
    
    # Calculate number of scenarios
    n_heights = len(dam_heights)
    n_discharges = len(discharges)
    
    # Ensure we have a valid grid of scenarios
    expected_scenarios = n_heights * n_discharges
    if len(scenarios) < expected_scenarios:
        # Warn user but proceed with available data
        ax.text(0.5, 0.05, f"Warning: Expected {expected_scenarios} scenarios but only {len(scenarios)} provided.",
              ha='center', va='bottom', transform=ax.transAxes, color='red')
    
    # Create arrays to store flow characteristics
    has_jump = np.zeros((n_discharges, n_heights), dtype=bool)
    max_froude = np.zeros((n_discharges, n_heights))
    
    # Populate arrays from scenarios
    for i, discharge in enumerate(discharges):
        for j, height in enumerate(dam_heights):
            # Calculate index and check bounds
            idx = i * n_heights + j
            if idx < len(scenarios):
                scenario = scenarios[idx]
                
                # Check if scenario has results
                if 'results' not in scenario:
                    continue
                    
                # Check for hydraulic jump
                jump = scenario['results'].get('hydraulic_jump', {})
                has_jump[i, j] = jump.get('jump_possible', False)
                
                # Find maximum Froude number
                tailwater = scenario['results'].get('tailwater', {})
                if 'fr' in tailwater and len(tailwater['fr']) > 0:
                    max_froude[i, j] = np.max(tailwater['fr'])
    
    # Create regime map
    # Use colors to indicate flow characteristics
    cmap = plt.cm.coolwarm
    
    # Set reasonable vmax
    fr_max = np.max(max_froude) if np.any(max_froude) else 3
    fr_max = max(3, fr_max)  # Ensure reasonable vmax
    
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=fr_max)
    
    # Plot regime map as pcolormesh - ensure inputs are valid
    if n_heights > 1 and n_discharges > 1:
        try:
            c = ax.pcolormesh(dam_heights, discharges, max_froude, cmap=cmap, norm=norm)
            
            # Add colorbar
            cbar = fig.colorbar(c, ax=ax)
            cbar.set_label('Maximum Froude Number')
            
            # Add regime labels to the colorbar
            cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                        rotation=90, transform=cbar.ax.transAxes)
            cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                        rotation=90, transform=cbar.ax.transAxes)
            cbar.ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
        except Exception as e:
            # Handle failure gracefully
            ax.text(0.5, 0.5, f"Error creating regime map: {str(e)}", 
                  ha='center', va='center', transform=ax.transAxes)
            return fig, ax
    else:
        ax.text(0.5, 0.5, "Insufficient data for regime map (need multiple heights and discharges)", 
              ha='center', va='center', transform=ax.transAxes)
        return fig, ax
    
    # Add markers for scenarios with hydraulic jumps
    jump_heights = []
    jump_discharges = []
    
    # Collect points where jumps occur
    for i in range(n_discharges):
        for j in range(n_heights):
            if has_jump[i, j]:
                jump_heights.append(dam_heights[j])
                jump_discharges.append(discharges[i])
    
    # Plot jump markers if any exist
    if jump_heights:
        ax.scatter(jump_heights, jump_discharges, marker='o', s=100, 
                 edgecolor='black', facecolor='none', linewidth=2,
                 label='Hydraulic Jump Present')
    
    # Add title and labels
    ax.set_title('Flow Regime Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dam Height (m)', fontsize=12)
    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if needed
    if jump_heights:
        ax.legend(loc='upper right')
    
    return fig, ax