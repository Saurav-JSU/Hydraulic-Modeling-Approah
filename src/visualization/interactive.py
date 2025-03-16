"""
Interactive visualizations for hydraulic modeling.

This module provides interactive plots with user controls for exploring
hydraulic modeling results in more detail.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

def create_interactive_profile(scenario, results_list, figsize=(14, 8)):
    """
    Create an interactive visualization with sliders to control the display.
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results at different water levels
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Input validation
    if not results_list:
        raise ValueError("results_list cannot be empty")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust subplot position to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
    
    # Initial display - use the first result
    from .enhanced_profiles import plot_enhanced_profile
    try:
        plot_enhanced_profile(scenario, results_list[0], ax=ax)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting profile: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Find max and min ranges for all results
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    for result in results_list:
        tailwater = result.get('tailwater', {})
        
        # Skip if tailwater data is missing or incomplete
        if 'x' not in tailwater or len(tailwater.get('x', [])) == 0:
            continue
            
        x_min = min(x_min, -20)  # Include upstream portion
        x_max = max(x_max, np.max(tailwater['x']))
        
        # Find min/max elevations safely
        dam_base = scenario.get('dam_base_elevation', 0)
        
        # Ensure we don't operate on empty arrays
        if len(tailwater.get('z', [])) > 0:
            bed_min = min(dam_base, np.min(tailwater['z']))
        else:
            bed_min = dam_base
        
        # Get water elevation
        water_max = result.get('upstream_level', dam_base)
        
        # Check if we have water surface elevations downstream
        if len(tailwater.get('wse', [])) > 0:
            water_max = max(water_max, np.max(tailwater['wse']))
        
        y_min = min(y_min, bed_min)
        y_max = max(y_max, water_max)
    
    # Handle case where no valid data was found
    if x_min == float('inf') or x_max == float('-inf'):
        x_min, x_max = -20, 200  # Default range
    if y_min == float('inf') or y_max == float('-inf'):
        dam_base = scenario.get('dam_base_elevation', 0)
        dam_height = scenario.get('dam_crest_elevation', 10) - dam_base
        y_min, y_max = dam_base - 1, dam_base + dam_height + 3  # Default range
    
    # Add some padding
    x_min -= 5
    x_max += 5
    y_min -= 1
    y_max += 1
    
    # Set fixed axis limits for stable display
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Create sliders
    ax_water_level = plt.axes([0.1, 0.15, 0.65, 0.03])
    ax_x_range = plt.axes([0.1, 0.10, 0.65, 0.03])
    ax_y_range = plt.axes([0.1, 0.05, 0.65, 0.03])
    
    # Get water level range from results
    water_levels = [result.get('upstream_level', 0) for result in results_list]
    min_level = min(water_levels) if water_levels else 0
    max_level = max(water_levels) if water_levels else 10
    
    # Ensure we have a valid range
    if min_level == max_level:
        max_level = min_level + 1  # Add 1m to create a valid range
    
    # Initial water level
    initial_level = results_list[0].get('upstream_level', min_level) if results_list else min_level
    
    # Create sliders
    water_level_slider = Slider(
        ax=ax_water_level,
        label='Water Level (m)',
        valmin=min_level,
        valmax=max_level,
        valinit=initial_level
    )
    
    x_range_slider = Slider(
        ax=ax_x_range,
        label='X Range',
        valmin=0,
        valmax=1,
        valinit=0.5
    )
    
    y_range_slider = Slider(
        ax=ax_y_range,
        label='Y Range',
        valmin=0,
        valmax=1,
        valinit=0.5
    )
    
    # Create radio buttons for coloring options
    ax_color_option = plt.axes([0.8, 0.05, 0.15, 0.15])
    color_options = ('Froude', 'Velocity', 'Depth', 'None')
    color_radio = RadioButtons(ax_color_option, color_options, active=0)
    
    # Function to update the plot when sliders change
    def update(_):
        # Clear the existing plot
        ax.clear()
        
        # Find the closest result to the selected water level
        target_level = water_level_slider.val
        closest_index = min(range(len(results_list)), 
                          key=lambda i: abs(results_list[i].get('upstream_level', 0) - target_level))
        
        # Get the corresponding result
        result = results_list[closest_index]
        
        # Determine coloring option
        color_option = color_radio.value_selected
        color_by = color_option.lower() if color_option != 'None' else None
        
        # Plot the profile with the selected result
        try:
            plot_enhanced_profile(scenario, result, ax=ax, color_by=color_by)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting profile: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            # Set basic axes properties to maintain display
            ax.set_xlabel('Distance from Dam (m)')
            ax.set_ylabel('Elevation (m)')
        
        # Adjust x and y ranges based on sliders
        x_center = (x_max + x_min) / 2
        x_range = (x_max - x_min) * (1 - 0.9 * x_range_slider.val)
        ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        
        y_center = (y_max + y_min) / 2
        y_range = (y_max - y_min) * (1 - 0.9 * y_range_slider.val)
        ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Add title with current water level and discharge (safely)
        discharge = result.get('discharge', 0)
        title = f'Water Surface Profile (Level: {result.get("upstream_level", 0):.2f} m, ' \
                f'Discharge: {discharge:.2f} m³/s)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Redraw canvas
        fig.canvas.draw_idle()
    
    # Register the update function with the sliders
    water_level_slider.on_changed(update)
    x_range_slider.on_changed(update)
    y_range_slider.on_changed(update)
    color_radio.on_clicked(update)
    
    # Create reset button
    ax_reset = plt.axes([0.8, 0.2, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    def reset(_):
        water_level_slider.reset()
        x_range_slider.reset()
        y_range_slider.reset()
        # Reset radio buttons (set to first option)
        color_radio.set_active(0)
    
    reset_button.on_clicked(reset)
    
    # Add instructions text
    fig.text(0.5, 0.01, 'Use sliders to adjust view and water level. '
            'Radio buttons change coloring method.', 
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                              fc="white", alpha=0.8))
    
    return fig, ax


def create_interactive_cross_section_viewer(scenario, results, figsize=(14, 10)):
    """
    Create an interactive viewer that allows exploration of cross-sections
    at any point along the channel.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    # Create figure with two subplots - profile and cross-section
    fig, (ax_profile, ax_cross) = plt.subplots(2, 1, figsize=figsize, 
                                             gridspec_kw={'height_ratios': [1, 2]})
    
    # Adjust subplot position to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.95, hspace=0.3)
    
    # Plot the main profile
    from .enhanced_profiles import plot_enhanced_profile
    try:
        plot_enhanced_profile(scenario, results, ax=ax_profile)
    except Exception as e:
        ax_profile.text(0.5, 0.5, f"Error plotting profile: {str(e)}", 
                      ha='center', va='center', transform=ax_profile.transAxes)
        ax_profile.set_xlabel('Distance from Dam (m)')
        ax_profile.set_ylabel('Elevation (m)')
    
    # Extract data safely
    tailwater = results.get('tailwater', {})
    x_values = tailwater.get('x', [])
    
    # Ensure we have data to work with
    if not x_values:
        ax_profile.text(0.5, 0.3, "No x-coordinate data available", 
                      ha='center', va='center', transform=ax_profile.transAxes)
        ax_cross.text(0.5, 0.5, "No cross-section data available", 
                     ha='center', va='center', transform=ax_cross.transAxes)
        return fig, (ax_profile, ax_cross)
    
    # Create a location marker (vertical line) in the profile
    location_line = ax_profile.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Create sliders
    ax_location = plt.axes([0.1, 0.1, 0.65, 0.03])
    
    # Determine x range
    x_min = -20  # Include upstream portion
    x_max = np.max(x_values) if len(x_values) > 0 else 200
    
    # Create location slider
    location_slider = Slider(
        ax=ax_location,
        label='Cross-Section Location (m)',
        valmin=x_min,
        valmax=x_max,
        valinit=0  # Start at the dam
    )
    
    # Create radio buttons for parameter highlighting
    ax_param_option = plt.axes([0.8, 0.05, 0.15, 0.15])
    param_options = ('Froude', 'Velocity', 'Shear', 'None')
    param_radio = RadioButtons(ax_param_option, param_options, active=0)
    
    # Function to update the cross-section view
    def update(_):
        # Get selected location
        loc = location_slider.val
        
        # Update the location marker
        location_line.set_xdata([loc, loc])
        
        # Clear the cross-section plot
        ax_cross.clear()
        
        # Get parameter highlighting option
        param_option = param_radio.value_selected
        
        # Determine water depth and parameters at this location
        if loc <= 0:  # Upstream of dam
            water_elevation = results.get('upstream_level', 0)
            bed_elevation = scenario.get('dam_base_elevation', 0)
            water_depth = max(0, water_elevation - bed_elevation)
            
            # Get channel parameters
            channel_width = scenario.get('channel_width_at_dam', 5.0)
            side_slope = scenario.get('channel_side_slope', 0)
            
            # Calculate top width for trapezoidal section
            top_width = channel_width + 2 * side_slope * water_depth
            
            # Calculate cross-sectional area
            area = 0.5 * (channel_width + top_width) * water_depth
            
            # Ensure non-zero area to avoid division by zero
            area = max(area, 0.001)
            
            # Estimate velocity
            discharge = results.get('discharge', 0)
            if water_depth > 0 and discharge > 0:
                velocity = discharge / area
                
                # Calculate Froude number using hydraulic depth
                hydraulic_depth = area / top_width if top_width > 0 else water_depth
                hydraulic_depth = max(hydraulic_depth, 0.001)  # Avoid division by zero
                froude = velocity / np.sqrt(9.81 * hydraulic_depth)
            else:
                velocity = 0
                froude = 0
                
            # Estimate shear stress using proper hydraulic radius
            if water_depth > 0:
                rho = 1000  # Water density (kg/m³)
                g = 9.81    # Gravity (m/s²)
                
                # Calculate wetted perimeter
                if side_slope > 0:
                    # Trapezoidal channel
                    sloped_sides = water_depth * np.sqrt(1 + side_slope**2)
                    wetted_perimeter = channel_width + 2 * sloped_sides
                else:
                    # Rectangular channel
                    wetted_perimeter = channel_width + 2 * water_depth
                
                # Ensure non-zero wetted perimeter
                wetted_perimeter = max(wetted_perimeter, 0.001)
                
                # Calculate hydraulic radius
                R = area / wetted_perimeter
                
                # Get slope
                S = scenario.get('channel_slope', 0.001)
                
                # Calculate shear stress
                shear = rho * g * R * S
            else:
                shear = 0
                
        else:  # Downstream of dam
            # Find closest point in tailwater results
            if len(x_values) > 0:
                idx = np.argmin(np.abs(np.array(x_values) - loc))
                
                # Check array bounds before accessing
                if idx < len(tailwater.get('wse', [])) and idx < len(tailwater.get('z', [])):
                    water_elevation = tailwater['wse'][idx]
                    bed_elevation = tailwater['z'][idx]
                    water_depth = max(0, water_elevation - bed_elevation)
                    
                    # Get velocity and Froude number if available
                    velocity = tailwater.get('v', [0])[idx] if idx < len(tailwater.get('v', [])) else 0
                    froude = tailwater.get('fr', [0])[idx] if idx < len(tailwater.get('fr', [])) else 0
                    
                    # Get channel parameters
                    channel_width = scenario.get('channel_bottom_width', 5.0)
                    side_slope = scenario.get('channel_side_slope', 0)
                    
                    # Estimate shear stress with proper hydraulic radius
                    if water_depth > 0:
                        rho = 1000  # Water density (kg/m³)
                        g = 9.81    # Gravity (m/s²)
                        
                        # Calculate top width for trapezoidal section
                        top_width = channel_width + 2 * side_slope * water_depth
                        
                        # Calculate cross-sectional area
                        area = 0.5 * (channel_width + top_width) * water_depth
                        
                        # Calculate wetted perimeter
                        if side_slope > 0:
                            # Trapezoidal channel
                            sloped_sides = water_depth * np.sqrt(1 + side_slope**2)
                            wetted_perimeter = channel_width + 2 * sloped_sides
                        else:
                            # Rectangular channel
                            wetted_perimeter = channel_width + 2 * water_depth
                        
                        # Ensure non-zero wetted perimeter
                        wetted_perimeter = max(wetted_perimeter, 0.001)
                        
                        # Calculate hydraulic radius
                        R = area / wetted_perimeter
                        
                        # Get slope
                        S = scenario.get('downstream_slope', 0.001)
                        
                        # Calculate shear stress
                        shear = rho * g * R * S
                    else:
                        shear = 0
                else:
                    water_depth = 0
                    velocity = 0
                    froude = 0
                    shear = 0
            else:
                water_depth = 0
                velocity = 0
                froude = 0
                shear = 0
        
        # Set up highlight parameter based on selection
        if param_option == 'Froude' and froude > 0:
            highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
        elif param_option == 'Velocity' and velocity > 0:
            highlight_param = {'type': 'velocity', 'value': velocity, 'max_value': 5.0}
        elif param_option == 'Shear' and shear > 0:
            highlight_param = {'type': 'shear', 'value': shear, 'max_value': 100.0}
        else:
            highlight_param = None
        
        # Plot the cross-section
        from .cross_sections import plot_channel_cross_section
        channel_type = 'trapezoidal'  # Default type
        channel_params = {
            'bottom_width': scenario.get('channel_bottom_width', 5.0),
            'side_slope': scenario.get('channel_side_slope', 0)
        }
        
        try:
            plot_channel_cross_section(ax_cross, channel_type, channel_params, water_depth, 
                                     highlight_param=highlight_param, annotate=True)
        except Exception as e:
            ax_cross.text(0.5, 0.5, f"Error plotting cross-section: {str(e)}", 
                        ha='center', va='center', transform=ax_cross.transAxes)
            # Set basic axes properties
            ax_cross.set_aspect('equal')
            ax_cross.grid(True, alpha=0.3)
        
        # Set cross-section title
        if loc <= 0:
            title = f"Cross-Section at Upstream Location (x={loc:.2f}m)"
        elif abs(loc) < 0.1:
            title = f"Cross-Section at Dam (x=0m)"
        else:
            title = f"Cross-Section at Downstream Location (x={loc:.2f}m)"
            
        # Add key hydraulic parameters to title
        if water_depth > 0:
            if param_option == 'Froude':
                title += f"\nDepth={water_depth:.2f}m, Fr={froude:.2f}"
                
                # Add flow regime description
                if froude < 0.8:
                    regime = "Subcritical Flow"
                elif froude > 1.2:
                    regime = "Supercritical Flow"
                else:
                    regime = "Near Critical Flow"
                    
                title += f" - {regime}"
                
            elif param_option == 'Velocity':
                title += f"\nDepth={water_depth:.2f}m, V={velocity:.2f}m/s"
            elif param_option == 'Shear':
                title += f"\nDepth={water_depth:.2f}m, τ={shear:.2f}N/m²"
            else:
                title += f"\nDepth={water_depth:.2f}m"
        
        ax_cross.set_title(title)
        
        # Redraw canvas
        fig.canvas.draw_idle()
    
    # Register the update function with the slider and radio buttons
    location_slider.on_changed(update)
    param_radio.on_clicked(update)
    
    # Create reset button
    ax_reset = plt.axes([0.8, 0.2, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    def reset(_):
        location_slider.reset()
        # Reset radio buttons (set to first option)
        param_radio.set_active(0)
    
    reset_button.on_clicked(reset)
    
    # Initialize cross-section view
    update(None)
    
    # Add instructions text
    fig.text(0.5, 0.01, 'Use slider to adjust cross-section location. '
            'Radio buttons change parameter highlighting.', 
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                              fc="white", alpha=0.8))
    
    return fig, (ax_profile, ax_cross)


def create_interactive_flood_explorer(scenario, results_list, figsize=(14, 10)):
    """
    Create an interactive explorer for analyzing flood progression with
    multiple synchronized views.
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results for different water levels
        figsize (tuple): Figure size
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    # Input validation
    if not results_list:
        raise ValueError("results_list cannot be empty")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[3, 1, 1], figure=fig)
    
    # Create main axes
    ax_profile = fig.add_subplot(gs[0, :])  # Profile spans all columns in first row
    ax_discharge = fig.add_subplot(gs[1, 0])  # Discharge vs water level
    ax_froude = fig.add_subplot(gs[1, 1:])  # Froude number
    ax_cross = fig.add_subplot(gs[2, 0])  # Cross-section
    ax_velocity = fig.add_subplot(gs[2, 1])  # Velocity
    ax_energy = fig.add_subplot(gs[2, 2])  # Energy
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.3)
    
    # Create slider for water level selection
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    
    # Get water level range from results
    water_levels = [result.get('upstream_level', 0) for result in results_list]
    min_level = min(water_levels) if water_levels else 0
    max_level = max(water_levels) if water_levels else 10
    
    # Ensure we have a valid range
    if min_level == max_level:
        max_level = min_level + 1  # Add 1m to create a valid range
    
    # Initial water level
    initial_level = results_list[0].get('upstream_level', min_level)
    
    # Create slider
    level_slider = Slider(
        ax=ax_slider,
        label='Water Level (m)',
        valmin=min_level,
        valmax=max_level,
        valinit=initial_level
    )
    
    # Create check buttons for display options
    ax_check = plt.axes([0.8, 0.03, 0.15, 0.15])
    display_options = ('Show Cross-Sections', 'Show Flow Regimes', 'Show Annotations')
    check_buttons = CheckButtons(ax_check, display_options, [True, True, True])
    
    # Plot discharge vs water level curve (static)
    discharges = [result.get('discharge', 0) for result in results_list]
    
    # Only plot if we have data
    if water_levels and discharges:
        ax_discharge.plot(water_levels, discharges, 'bo-', linewidth=2)
        ax_discharge.set_xlabel('Water Level (m)')
        ax_discharge.set_ylabel('Discharge (m³/s)')
        ax_discharge.set_title('Rating Curve')
        ax_discharge.grid(True, alpha=0.3)
        
        # Add dam crest line
        dam_crest = scenario.get('dam_crest_elevation', 0)
        ax_discharge.axvline(x=dam_crest, color='r', linestyle='--', label=f'Dam Crest ({dam_crest:.1f}m)')
        ax_discharge.legend(loc='upper left')
        
        # Create marker for current water level on discharge plot
        water_marker, = ax_discharge.plot([water_levels[0]], [discharges[0]], 'ro', ms=10)
    else:
        ax_discharge.text(0.5, 0.5, "Insufficient data for rating curve",
                        ha='center', va='center', transform=ax_discharge.transAxes)
        water_marker = None
    
    # Function to update all plots
    def update(_):
        # Get selected water level
        target_level = level_slider.val
        
        # Find the closest result to the selected water level
        closest_index = min(range(len(results_list)), 
                          key=lambda i: abs(results_list[i].get('upstream_level', 0) - target_level))
        
        # Get the selected result
        result = results_list[closest_index]
        
        # Update marker on discharge plot if it exists
        if water_marker is not None:
            water_marker.set_data([result.get('upstream_level', 0)], [result.get('discharge', 0)])
        
        # Get display options
        show_cross = check_buttons.get_status()[0]
        show_regimes = check_buttons.get_status()[1]
        show_annotations = check_buttons.get_status()[2]
        
        # Clear previous elements
        ax_profile.clear()
        ax_froude.clear()
        ax_cross.clear()
        ax_velocity.clear()
        ax_energy.clear()
        
        # Plot main profile
        if show_regimes:
            try:
                from .flow_regimes import plot_flow_regime_profile
                plot_flow_regime_profile(scenario, result, ax=ax_profile, 
                                      show_annotations=show_annotations)
            except Exception as e:
                ax_profile.text(0.5, 0.5, f"Error plotting flow regime profile: {str(e)}", 
                              ha='center', va='center', transform=ax_profile.transAxes)
                ax_profile.set_xlabel('Distance from Dam (m)')
                ax_profile.set_ylabel('Elevation (m)')
        else:
            try:
                from .enhanced_profiles import plot_enhanced_profile
                plot_enhanced_profile(scenario, result, ax=ax_profile, 
                                   show_annotations=show_annotations)
            except Exception as e:
                ax_profile.text(0.5, 0.5, f"Error plotting enhanced profile: {str(e)}", 
                              ha='center', va='center', transform=ax_profile.transAxes)
                ax_profile.set_xlabel('Distance from Dam (m)')
                ax_profile.set_ylabel('Elevation (m)')
        
        # Extract data for other plots
        tailwater = result.get('tailwater', {})
        x_values = tailwater.get('x', [])
        
        # Plot Froude number profile
        if 'fr' in tailwater and len(tailwater['fr']) > 0 and len(x_values) == len(tailwater['fr']):
            ax_froude.plot(x_values, tailwater['fr'], 'r-', linewidth=2)
            ax_froude.axhline(y=1, color='k', linestyle='--', 
                           label='Critical Flow (Fr=1)')
            
            # Add regime zones
            ax_froude.fill_between(x_values, 0, np.ones_like(x_values),
                               color='blue', alpha=0.2, label='Subcritical')
            ax_froude.fill_between(x_values, np.ones_like(x_values), 2,
                               color='red', alpha=0.2, label='Supercritical')
        else:
            ax_froude.text(0.5, 0.5, "No Froude number data available", 
                         ha='center', va='center', transform=ax_froude.transAxes)
        
        # Set Froude axis properties
        try:
            ax_froude.set_xlim(ax_profile.get_xlim())
        except:
            # Default limits if profile limits can't be obtained
            ax_froude.set_xlim(-20, 200)
            
        ax_froude.set_xlabel('Distance (m)')
        ax_froude.set_ylabel('Froude Number')
        ax_froude.set_title('Froude Number Profile')
        ax_froude.grid(True, alpha=0.3)
        ax_froude.legend(loc='upper right', fontsize=8)
        
        # Plot cross-section (if enabled)
        if show_cross and len(x_values) > 0:
            # Choose a location for the cross-section
            # Default to a hydraulic jump if present, otherwise middle of channel
            jump = result.get('hydraulic_jump', {})
            if jump.get('jump_possible', False) and 'location' in jump:
                loc = jump['location']
            else:
                loc = x_values[len(x_values)//2] if len(x_values) > 0 else 50
            
            # Draw a marker on the profile view
            ax_profile.axvline(x=loc, color='m', linestyle='--', linewidth=1, alpha=0.7)
            
            # Plot cross-section
            from .cross_sections import plot_channel_cross_section
            channel_type = 'trapezoidal'
            channel_params = {
                'bottom_width': scenario.get('channel_bottom_width', 5.0),
                'side_slope': scenario.get('channel_side_slope', 0)
            }
            
            # Determine water depth at this location
            if loc <= 0:  # Upstream of dam
                water_elevation = result.get('upstream_level', 0)
                bed_elevation = scenario.get('dam_base_elevation', 0)
                water_depth = max(0, water_elevation - bed_elevation)
                
                # Get channel parameters for calculations
                channel_width = scenario.get('channel_width_at_dam', 5.0)
                side_slope = scenario.get('channel_side_slope', 0)
                
                # Calculate top width for trapezoidal section
                top_width = channel_width + 2 * side_slope * water_depth
                
                # Calculate cross-sectional area
                area = 0.5 * (channel_width + top_width) * water_depth
                
                # Ensure non-zero area
                area = max(area, 0.001)
                
                # Estimate velocity and Froude number
                discharge = result.get('discharge', 0)
                if water_depth > 0 and discharge > 0:
                    velocity = discharge / area
                    
                    # Calculate hydraulic depth for Froude number
                    hydraulic_depth = area / top_width if top_width > 0 else water_depth
                    hydraulic_depth = max(hydraulic_depth, 0.001)  # Avoid division by zero
                    froude = velocity / np.sqrt(9.81 * hydraulic_depth)
                else:
                    velocity = 0
                    froude = 0
            else:  # Downstream of dam
                # Find closest point in tailwater results
                if len(x_values) > 0:
                    idx = np.argmin(np.abs(np.array(x_values) - loc))
                    
                    # Check array bounds before accessing
                    if idx < len(tailwater.get('wse', [])) and idx < len(tailwater.get('z', [])):
                        water_elevation = tailwater['wse'][idx]
                        bed_elevation = tailwater['z'][idx]
                        water_depth = max(0, water_elevation - bed_elevation)
                        
                        # Get velocity and Froude number if available
                        velocity = tailwater.get('v', [0])[idx] if idx < len(tailwater.get('v', [])) else 0
                        froude = tailwater.get('fr', [0])[idx] if idx < len(tailwater.get('fr', [])) else 0
                    else:
                        water_depth = 0
                        velocity = 0
                        froude = 0
                else:
                    water_depth = 0
                    velocity = 0
                    froude = 0
            
            # Select parameter to highlight
            if froude > 0:
                highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
            else:
                highlight_param = None
            
            try:
                plot_channel_cross_section(ax_cross, channel_type, channel_params, water_depth,
                                         highlight_param=highlight_param, annotate=show_annotations)
            except Exception as e:
                ax_cross.text(0.5, 0.5, f"Error plotting cross-section: {str(e)}", 
                            ha='center', va='center', transform=ax_cross.transAxes)
                # Set basic axes properties
                ax_cross.set_aspect('equal')
                ax_cross.grid(True, alpha=0.3)
            
            # Set title with location
            if loc <= 0:
                title = f"Cross-Section at x={loc:.1f}m (Upstream)"
            else:
                title = f"Cross-Section at x={loc:.1f}m (Downstream)"
                
            ax_cross.set_title(title)
        else:
            ax_cross.set_title("Cross-Section View (Disabled)")
            ax_cross.text(0.5, 0.5, "Cross-section display disabled or no data available", 
                        ha='center', va='center', transform=ax_cross.transAxes)
            ax_cross.set_aspect('equal')
            ax_cross.grid(True, alpha=0.3)
        
        # Plot velocity profile
        if 'v' in tailwater and len(tailwater['v']) > 0 and len(x_values) > 0:
            # Ensure arrays have matching lengths
            plot_length = min(len(x_values), len(tailwater['v']))
            if plot_length > 0:
                ax_velocity.plot(x_values[:plot_length], tailwater['v'][:plot_length], 'g-', linewidth=2)
            else:
                ax_velocity.text(0.5, 0.5, "Velocity data available but mismatched lengths", 
                               ha='center', va='center', transform=ax_velocity.transAxes)
        else:
            ax_velocity.text(0.5, 0.5, "No velocity data available", 
                           ha='center', va='center', transform=ax_velocity.transAxes)
        
        # Set velocity axis properties
        try:
            ax_velocity.set_xlim(ax_profile.get_xlim())
        except:
            ax_velocity.set_xlim(-20, 200)
            
        ax_velocity.set_xlabel('Distance (m)')
        ax_velocity.set_ylabel('Velocity (m/s)')
        ax_velocity.set_title('Velocity Profile')
        ax_velocity.grid(True, alpha=0.3)
        
        # Plot energy profile
        if 'energy' in tailwater and 'z' in tailwater and len(tailwater['energy']) > 0 and len(tailwater['z']) > 0 and len(x_values) > 0:
            # Ensure arrays have matching lengths
            plot_length = min(len(x_values), len(tailwater['energy']), len(tailwater['z']))
            if plot_length > 0:
                # Calculate specific energy
                specific_energy = np.zeros(plot_length)
                for i in range(plot_length):
                    specific_energy[i] = tailwater['energy'][i] - tailwater['z'][i]
                
                ax_energy.plot(x_values[:plot_length], specific_energy, 'purple', linewidth=2)
            else:
                ax_energy.text(0.5, 0.5, "Energy data available but mismatched lengths", 
                             ha='center', va='center', transform=ax_energy.transAxes)
        else:
            # Try to estimate from depth and velocity
            if 'y' in tailwater and 'v' in tailwater and len(tailwater['y']) > 0 and len(tailwater['v']) > 0 and len(x_values) > 0:
                plot_length = min(len(x_values), len(tailwater['y']), len(tailwater['v']))
                if plot_length > 0:
                    # Estimate specific energy as depth + velocity head
                    specific_energy = np.zeros(plot_length)
                    for i in range(plot_length):
                        depth = tailwater['y'][i]
                        velocity = tailwater['v'][i]
                        # Calculate E = y + v²/2g
                        specific_energy[i] = depth + (velocity**2) / (2 * 9.81)
                    
                    ax_energy.plot(x_values[:plot_length], specific_energy, 'purple', linewidth=2)
                else:
                    ax_energy.text(0.5, 0.5, "Cannot calculate energy - mismatched data lengths", 
                                 ha='center', va='center', transform=ax_energy.transAxes)
            else:
                ax_energy.text(0.5, 0.5, "No energy data available", 
                             ha='center', va='center', transform=ax_energy.transAxes)
        
        # Set energy axis properties
        try:
            ax_energy.set_xlim(ax_profile.get_xlim())
        except:
            ax_energy.set_xlim(-20, 200)
            
        ax_energy.set_xlabel('Distance (m)')
        ax_energy.set_ylabel('Specific Energy (m)')
        ax_energy.set_title('Energy Profile')
        ax_energy.grid(True, alpha=0.3)
        
        # Update the figure title
        fig.suptitle(f'Water Level: {result.get("upstream_level", 0):.2f}m, '
                   f'Discharge: {result.get("discharge", 0):.2f}m³/s', 
                   fontsize=16, fontweight='bold', y=0.98)
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Register the update function with the slider and check buttons
    level_slider.on_changed(update)
    check_buttons.on_clicked(update)
    
    # Create reset button
    ax_reset = plt.axes([0.8, 0.2, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    def reset(_):
        level_slider.reset()
        # Reset check buttons (all True)
        for i in range(len(display_options)):
            if not check_buttons.get_status()[i]:
                check_buttons.set_active(i)
        update(None)
    
    reset_button.on_clicked(reset)
    
    # Initial update
    update(None)
    
    # Add instructions text
    fig.text(0.5, 0.01, 'Use slider to adjust water level. '
            'Use checkboxes to toggle display options.', 
            ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                              fc="white", alpha=0.8))
    
    return fig, {'profile': ax_profile, 'discharge': ax_discharge, 
                'froude': ax_froude, 'cross': ax_cross, 
                'velocity': ax_velocity, 'energy': ax_energy}