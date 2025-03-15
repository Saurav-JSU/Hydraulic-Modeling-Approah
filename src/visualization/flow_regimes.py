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
    
    # Plot bed profile
    ax.plot(combined_x, bed, 'k-', linewidth=1.5, label='Channel Bed')
    
    # Fill the dam body
    dam_poly = np.column_stack([
        np.concatenate([x_dam, x_dam[::-1]]),
        np.concatenate([dam_profile['z'], np.full_like(x_dam, dam_base)[::-1]])
    ])
    dam_patch = Polygon(dam_poly, closed=True, facecolor='#4F4F4F',
                       edgecolor='black', linewidth=0.5)
    ax.add_patch(dam_patch)
    
    # Create Froude number array for the entire domain
    froude = np.zeros_like(combined_x)
    
    # Fill upstream portion
    upstream_level = results['upstream_level']
    upstream_depth = upstream_level - dam_base
    
    if upstream_depth > 0 and results['discharge'] > 0:
        # Estimate upstream velocity
        upstream_area = upstream_depth * scenario['channel_width_at_dam']
        upstream_velocity = results['discharge'] / upstream_area
        upstream_froude = upstream_velocity / np.sqrt(9.81 * upstream_depth)
        froude[combined_x <= 0] = upstream_froude
    
    # Fill downstream portion
    for i, x in enumerate(combined_x):
        if x <= 0:
            continue  # Already filled upstream portion
            
        # Find closest point in tailwater results
        idx = np.argmin(np.abs(x_values - x))
        if idx < len(tailwater['fr']):
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
        y_vals = [water_surface[i-1], water_surface[i], 
                  max(bed[i], water_surface[i]), max(bed[i-1], water_surface[i-1])]
        
        # Skip invalid polygons
        if len(set(y_vals)) <= 2:
            continue
            
        # Get regime for this segment
        regime = flow_regime[i]
        
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
    yn = results['normal_depth']
    yc = results['critical_depth']
    
    # Only show if we have flow
    if results['discharge'] > 0:
        # Normal depth line
        base_elevation = tailwater['z'][0]
        normal_line = ax.axhline(y=base_elevation + yn, color='green', linestyle='--', 
                               alpha=0.7, linewidth=1.5, 
                               label=f'Normal Depth ({yn:.2f} m)')
        
        # Critical depth line
        critical_line = ax.axhline(y=base_elevation + yc, color='red', linestyle=':', 
                                 alpha=0.7, linewidth=1.5, 
                                 label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        if jump_index < len(tailwater['z']):
            jump_z = tailwater['z'][jump_index]
            
            # Mark the jump
            ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                      alpha=0.7, linewidth=2, label=f'Hydraulic Jump')
            
            # Add enhanced jump visualization
            y1 = jump['initial_depth']
            y2 = jump['sequent_depth']
            
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
            if show_annotations:
                ax.annotate(
                    f"Hydraulic Jump\n"
                    f"Fr₁ = {jump['initial_froude']:.2f} → Fr₂ < 1\n"
                    f"y₁ = {y1:.2f}m → y₂ = {y2:.2f}m\n"
                    f"Type: {jump['jump_type']}",
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
    if results['discharge'] > 0:
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
        ax.set_xlim(-20, min(200, np.max(x_values)))
    
    # Calculate y limits to include important elements with some padding
    y_min = min(bed) - 1
    y_max = max(water_surface) + 2
    
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
        discharge = results['discharge']
        upstream_level = results['upstream_level']
        dam_crest = scenario['dam_crest_elevation']
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
    
    # Extract data
    tailwater = results['tailwater']
    x_values = tailwater['x']
    
    # Include upstream portion
    x_upstream = np.linspace(-20, 0, 20)
    
    # Estimate upstream Froude number
    upstream_level = results['upstream_level']
    dam_base = scenario['dam_base_elevation']
    upstream_depth = upstream_level - dam_base
    
    if upstream_depth > 0 and results['discharge'] > 0:
        # Calculate upstream Froude number
        upstream_area = upstream_depth * scenario['channel_width_at_dam']
        upstream_velocity = results['discharge'] / upstream_area
        upstream_froude = upstream_velocity / np.sqrt(9.81 * upstream_depth)
        fr_upstream = np.full_like(x_upstream, upstream_froude)
    else:
        fr_upstream = np.zeros_like(x_upstream)
    
    # Combine arrays for complete profile
    x_combined = np.concatenate([x_upstream, x_values])
    
    # Create Froude number array
    fr_combined = np.zeros_like(x_combined)
    fr_combined[:len(x_upstream)] = fr_upstream
    
    if len(tailwater['fr']) > 0:
        fr_combined[len(x_upstream):len(x_upstream) + len(tailwater['fr'])] = tailwater['fr']
    
    # Create a colormap for the Froude number
    cmap = plt.cm.coolwarm
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2, np.max(fr_combined)))
    
    # Plot Froude profile with colored line
    points = np.array([x_combined, fr_combined]).T.reshape(-1, 1, 2)
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
    
    # Add reference line at Fr = 1 (critical flow)
    ax.axhline(y=1.0, color='black', linestyle='--', 
              label='Critical Flow (Fr = 1)', linewidth=1.5)
    
    # Add shaded regions for flow regimes
    ax.fill_between(x_combined, 0, 0.8, color='blue', alpha=0.2, 
                   label='Subcritical (Fr < 0.8)')
    ax.fill_between(x_combined, 0.8, 1.2, color='green', alpha=0.2, 
                   label='Near Critical (0.8 ≤ Fr ≤ 1.2)')
    ax.fill_between(x_combined, 1.2, max(2, np.max(fr_combined)) + 0.5, 
                   color='red', alpha=0.2, label='Supercritical (Fr > 1.2)')
    
    # Mark hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        # Mark with vertical line
        ax.axvline(x=jump_loc, color='magenta', linestyle='-.',
                  linewidth=2, label=f'Hydraulic Jump')
        
        # Add arrow showing the Froude transition
        fr1 = jump['initial_froude']
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
        ax.set_xlim(-20, min(200, np.max(x_values)))
    
    # Set y limit with some padding
    ax.set_ylim(0, max(2, np.max(fr_combined) * 1.2))
    
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
    tailwater = results['tailwater']
    x_values = tailwater['x']
    
    # Velocity profile
    velocities = tailwater['v']
    ax_velocity.plot(x_values, velocities, 'g-', linewidth=2)
    ax_velocity.set_title('Velocity Profile')
    ax_velocity.set_xlabel('Distance from Dam (m)')
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in velocity plot if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        ax_velocity.axvline(x=jump_loc, color='magenta', linestyle='-.', 
                          linewidth=1.5, alpha=0.7)
        
        # Find velocity values before and after jump
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        if jump_index < len(velocities):
            v1 = velocities[jump_index]
            if jump_index + 5 < len(velocities):
                v2 = velocities[jump_index + 5]  # A bit after the jump
            else:
                v2 = velocities[-1]  # Last available value
            
            # Add arrow showing velocity reduction
            arrow = FancyArrowPatch((jump_loc - 3, v1), (jump_loc + 3, v2),
                                  connectionstyle="arc3,rad=0.3",
                                  arrowstyle="-|>", color='magenta',
                                  linewidth=2, alpha=0.8)
            ax_velocity.add_patch(arrow)
    
    # Energy profile
    if 'energy' in tailwater:
        # Plot specific energy
        specific_energy = tailwater['energy'] - tailwater['z']
        ax_energy.plot(x_values, specific_energy, 'purple', linewidth=2)
    else:
        # Estimate specific energy as depth + velocity head
        depths = tailwater['y']
        velocities = tailwater['v']
        specific_energy = depths + velocities**2 / (2 * 9.81)
        ax_energy.plot(x_values, specific_energy, 'purple', linewidth=2)
    
    ax_energy.set_title('Specific Energy Profile')
    ax_energy.set_xlabel('Distance from Dam (m)')
    ax_energy.set_ylabel('Specific Energy (m)')
    ax_energy.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in energy plot if present
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        ax_energy.axvline(x=jump_loc, color='magenta', linestyle='-.', 
                        linewidth=1.5, alpha=0.7)
        
        # Add annotation for energy loss
        jump_index = np.argmin(np.abs(x_values - jump_loc))
        if jump_index < len(specific_energy):
            if jump_index + 5 < len(specific_energy):
                energy_loss = specific_energy[jump_index] - specific_energy[jump_index + 5]
                ax_energy.annotate(
                    f"Energy Loss\n{energy_loss:.2f} m",
                    xy=(jump_loc, specific_energy[jump_index]),
                    xytext=(jump_loc + 15, specific_energy[jump_index] + 0.2),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                    fontsize=9
                )
    
    # Shear stress profile (estimated)
    depths = tailwater['y']
    shear_stress = np.zeros_like(depths)
    
    # Calculate shear stress using τ = ρgRS
    rho = 1000  # Water density (kg/m³)
    g = 9.81    # Gravitational acceleration (m/s²)
    S = scenario['downstream_slope']
    
    for i, depth in enumerate(depths):
        if depth > 0:
            # Use hydraulic radius = depth for wide rectangular channel (simplified)
            R = depth
            shear_stress[i] = rho * g * R * S
    
    ax_shear.plot(x_values, shear_stress, 'brown', linewidth=2)
    ax_shear.set_title('Bed Shear Stress Profile')
    ax_shear.set_xlabel('Distance from Dam (m)')
    ax_shear.set_ylabel('Shear Stress (N/m²)')
    ax_shear.grid(True, alpha=0.3)
    
    # Mark hydraulic jump in shear stress plot if present
    if jump.get('jump_possible', False):
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
    
    # Calculate number of scenarios
    n_heights = len(dam_heights)
    n_discharges = len(discharges)
    
    # Create arrays to store flow characteristics
    has_jump = np.zeros((n_discharges, n_heights), dtype=bool)
    max_froude = np.zeros((n_discharges, n_heights))
    
    # Populate arrays from scenarios
    for i, discharge in enumerate(discharges):
        for j, height in enumerate(dam_heights):
            # Find the corresponding scenario
            idx = i * n_heights + j
            if idx < len(scenarios):
                scenario = scenarios[idx]
                
                # Check for hydraulic jump
                jump = scenario['results']['hydraulic_jump']
                has_jump[i, j] = jump.get('jump_possible', False)
                
                # Find maximum Froude number
                tailwater = scenario['results']['tailwater']
                if len(tailwater['fr']) > 0:
                    max_froude[i, j] = np.max(tailwater['fr'])
    
    # Create regime map
    # Use colors to indicate flow characteristics
    cmap = plt.cm.coolwarm
    norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)
    
    # Plot regime map
    c = ax.pcolormesh(dam_heights, discharges, max_froude, cmap=cmap, norm=norm)
    
    # Add markers for scenarios with hydraulic jumps
    jump_heights = []
    jump_discharges = []
    
    for i in range(n_discharges):
        for j in range(n_heights):
            if has_jump[i, j]:
                jump_heights.append(dam_heights[j])
                jump_discharges.append(discharges[i])
    
    if jump_heights:
        ax.scatter(jump_heights, jump_discharges, marker='o', s=100, 
                 edgecolor='black', facecolor='none', linewidth=2,
                 label='Hydraulic Jump Present')
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Maximum Froude Number')
    
    # Add regime labels to the colorbar
    cbar.ax.text(0.5, 0.25, 'Subcritical', ha='center', va='center', 
                rotation=90, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, 0.75, 'Supercritical', ha='center', va='center', 
                rotation=90, transform=cbar.ax.transAxes)
    cbar.ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    
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