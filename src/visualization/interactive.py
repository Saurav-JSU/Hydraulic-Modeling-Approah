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
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Adjust subplot position to make room for sliders
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
    
    # Initial display - use the first result
    from .enhanced_profiles import plot_enhanced_profile
    plot_enhanced_profile(scenario, results_list[0], ax=ax)
    
    # Find max and min ranges for all results
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    for result in results_list:
        tailwater = result['tailwater']
        x_min = min(x_min, -20)  # Include upstream portion
        x_max = max(x_max, np.max(tailwater['x']))
        
        # Find min/max elevations
        dam_base = scenario['dam_base_elevation']
        bed_min = min(dam_base, np.min(tailwater['z']))
        
        water_max = max(result['upstream_level'], np.max(tailwater['wse']))
        
        y_min = min(y_min, bed_min)
        y_max = max(y_max, water_max)
    
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
    water_levels = [result['upstream_level'] for result in results_list]
    min_level = min(water_levels)
    max_level = max(water_levels)
    
    # Create sliders
    water_level_slider = Slider(
        ax=ax_water_level,
        label='Water Level (m)',
        valmin=min_level,
        valmax=max_level,
        valinit=results_list[0]['upstream_level']
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
                          key=lambda i: abs(results_list[i]['upstream_level'] - target_level))
        
        # Get the corresponding result
        result = results_list[closest_index]
        
        # Determine coloring option
        color_option = color_radio.value_selected
        color_by = color_option.lower() if color_option != 'None' else None
        
        # Plot the profile with the selected result
        plot_enhanced_profile(scenario, result, ax=ax, color_by=color_by)
        
        # Adjust x and y ranges based on sliders
        x_center = (x_max + x_min) / 2
        x_range = (x_max - x_min) * (1 - 0.9 * x_range_slider.val)
        ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        
        y_center = (y_max + y_min) / 2
        y_range = (y_max - y_min) * (1 - 0.9 * y_range_slider.val)
        ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Add title with current water level
        ax.set_title(f'Water Surface Profile (Level: {result["upstream_level"]:.2f} m, '
                    f'Discharge: {result["discharge"]:.2f} m³/s)', 
                    fontsize=14, fontweight='bold')
        
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
    plot_enhanced_profile(scenario, results, ax=ax_profile)
    
    # Extract data
    tailwater = results['tailwater']
    x_values = tailwater['x']
    
    # Create a location marker (vertical line) in the profile
    location_line = ax_profile.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Create sliders
    ax_location = plt.axes([0.1, 0.1, 0.65, 0.03])
    
    # Determine x range
    x_min = -20  # Include upstream portion
    x_max = np.max(x_values)
    
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
            water_elevation = results['upstream_level']
            bed_elevation = scenario['dam_base_elevation']
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
            idx = np.argmin(np.abs(x_values - loc))
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
            'bottom_width': scenario['channel_bottom_width'],
            'side_slope': scenario['channel_side_slope']
        }
        
        plot_channel_cross_section(ax_cross, channel_type, channel_params, water_depth, 
                                  highlight_param=highlight_param, annotate=True)
        
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
    water_levels = [result['upstream_level'] for result in results_list]
    min_level = min(water_levels)
    max_level = max(water_levels)
    
    # Create slider
    level_slider = Slider(
        ax=ax_slider,
        label='Water Level (m)',
        valmin=min_level,
        valmax=max_level,
        valinit=water_levels[0]
    )
    
    # Create check buttons for display options
    ax_check = plt.axes([0.8, 0.03, 0.15, 0.15])
    display_options = ('Show Cross-Sections', 'Show Flow Regimes', 'Show Annotations')
    check_buttons = CheckButtons(ax_check, display_options, [True, True, True])
    
    # Plot discharge vs water level curve (static)
    discharges = [result['discharge'] for result in results_list]
    ax_discharge.plot(water_levels, discharges, 'bo-', linewidth=2)
    ax_discharge.set_xlabel('Water Level (m)')
    ax_discharge.set_ylabel('Discharge (m³/s)')
    ax_discharge.set_title('Rating Curve')
    ax_discharge.grid(True, alpha=0.3)
    
    # Add dam crest line
    dam_crest = scenario['dam_crest_elevation']
    ax_discharge.axvline(x=dam_crest, color='r', linestyle='--', label=f'Dam Crest ({dam_crest:.1f}m)')
    ax_discharge.legend(loc='upper left')
    
    # Create marker for current water level on discharge plot
    water_marker, = ax_discharge.plot([water_levels[0]], [discharges[0]], 'ro', ms=10)
    
    # Create dictionaries to store dynamic plot elements
    profile_elements = []
    froude_elements = []
    cross_elements = []
    velocity_elements = []
    energy_elements = []
    
    # Function to update all plots
    def update(_):
        # Get selected water level
        target_level = level_slider.val
        
        # Find the closest result to the selected water level
        closest_index = min(range(len(results_list)), 
                          key=lambda i: abs(results_list[i]['upstream_level'] - target_level))
        
        # Get the selected result
        result = results_list[closest_index]
        
        # Update marker on discharge plot
        water_marker.set_data([result['upstream_level']], [result['discharge']])
        
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
            from .flow_regimes import plot_flow_regime_profile
            plot_flow_regime_profile(scenario, result, ax=ax_profile, 
                                   show_annotations=show_annotations)
        else:
            from .enhanced_profiles import plot_enhanced_profile
            plot_enhanced_profile(scenario, result, ax=ax_profile, 
                                show_annotations=show_annotations)
        
        # Extract data for other plots
        tailwater = result['tailwater']
        x_values = tailwater['x']
        
        # Plot Froude number profile
        if len(tailwater['fr']) > 0:
            ax_froude.plot(x_values, tailwater['fr'], 'r-', linewidth=2)
            ax_froude.axhline(y=1, color='k', linestyle='--', 
                           label='Critical Flow (Fr=1)')
            
            # Add regime zones
            ax_froude.fill_between(x_values, 0, np.ones_like(x_values),
                               color='blue', alpha=0.2, label='Subcritical')
            ax_froude.fill_between(x_values, np.ones_like(x_values), 2,
                               color='red', alpha=0.2, label='Supercritical')
        
        ax_froude.set_xlim(ax_profile.get_xlim())
        ax_froude.set_xlabel('Distance (m)')
        ax_froude.set_ylabel('Froude Number')
        ax_froude.set_title('Froude Number Profile')
        ax_froude.grid(True, alpha=0.3)
        ax_froude.legend(loc='upper right', fontsize=8)
        
        # Plot cross-section (if enabled)
        if show_cross:
            # Choose a location for the cross-section
            # Default to a hydraulic jump if present, otherwise middle of channel
            jump = result['hydraulic_jump']
            if jump.get('jump_possible', False):
                loc = jump['location']
            else:
                loc = x_values[len(x_values)//2] if len(x_values) > 0 else 50
            
            # Draw a marker on the profile view
            ax_profile.axvline(x=loc, color='m', linestyle='--', linewidth=1, alpha=0.7)
            
            # Plot cross-section
            from .cross_sections import plot_channel_cross_section
            channel_type = 'trapezoidal'
            channel_params = {
                'bottom_width': scenario['channel_bottom_width'],
                'side_slope': scenario['channel_side_slope']
            }
            
            # Determine water depth at this location
            if loc <= 0:  # Upstream of dam
                water_elevation = result['upstream_level']
                bed_elevation = scenario['dam_base_elevation']
                water_depth = max(0, water_elevation - bed_elevation)
                
                # Estimate velocity and Froude number
                if water_depth > 0 and result['discharge'] > 0:
                    area = water_depth * scenario['channel_width_at_dam']
                    velocity = result['discharge'] / area
                    froude = velocity / np.sqrt(9.81 * water_depth)
                else:
                    velocity = 0
                    froude = 0
            else:  # Downstream of dam
                # Find closest point in tailwater results
                idx = np.argmin(np.abs(x_values - loc))
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
            
            # Select parameter to highlight
            if froude > 0:
                highlight_param = {'type': 'froude', 'value': froude, 'max_value': 2.0}
            else:
                highlight_param = None
            
            plot_channel_cross_section(ax_cross, channel_type, channel_params, water_depth,
                                     highlight_param=highlight_param, annotate=show_annotations)
            
            # Set title with location
            if loc <= 0:
                title = f"Cross-Section at x={loc:.1f}m (Upstream)"
            else:
                title = f"Cross-Section at x={loc:.1f}m (Downstream)"
                
            ax_cross.set_title(title)
        else:
            ax_cross.set_title("Cross-Section View (Disabled)")
        
        # Plot velocity profile
        if len(tailwater['v']) > 0:
            ax_velocity.plot(x_values, tailwater['v'], 'g-', linewidth=2)
        
        ax_velocity.set_xlim(ax_profile.get_xlim())
        ax_velocity.set_xlabel('Distance (m)')
        ax_velocity.set_ylabel('Velocity (m/s)')
        ax_velocity.set_title('Velocity Profile')
        ax_velocity.grid(True, alpha=0.3)
        
        # Plot energy profile
        if 'energy' in tailwater:
            # Plot specific energy
            specific_energy = tailwater['energy'] - tailwater['z']
            ax_energy.plot(x_values, specific_energy, 'purple', linewidth=2)
        else:
            # Estimate specific energy as depth + velocity head
            depths = tailwater['y']
            velocities = tailwater['v']
            if len(depths) > 0 and len(velocities) > 0:
                specific_energy = depths + velocities**2 / (2 * 9.81)
                ax_energy.plot(x_values, specific_energy, 'purple', linewidth=2)
        
        ax_energy.set_xlim(ax_profile.get_xlim())
        ax_energy.set_xlabel('Distance (m)')
        ax_energy.set_ylabel('Specific Energy (m)')
        ax_energy.set_title('Energy Profile')
        ax_energy.grid(True, alpha=0.3)
        
        # Update the figure title
        fig.suptitle(f'Water Level: {result["upstream_level"]:.2f}m, '
                   f'Discharge: {result["discharge"]:.2f}m³/s', 
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