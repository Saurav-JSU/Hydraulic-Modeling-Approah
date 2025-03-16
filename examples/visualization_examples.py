"""
Example script demonstrating the enhanced visualization capabilities.

This script shows how to use the various visualization tools to create
informative and visually appealing hydraulic visualizations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the src directory to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from src.dam.geometry import OgeeWeir, BroadCrestedWeir
from src.channel.geometry import TrapezoidalChannel
from examples.scenario_setup import create_scenario
from examples.steady_analysis import analyze_steady_state

# Import enhanced visualization modules
from src.visualization import (
    # Basic visualizations (original)
    plot_full_profile,
    plot_velocity_profile,
    
    # Enhanced single-result visualizations
    plot_enhanced_profile,
    create_profile_with_parameter_plots,
    plot_enhanced_profile_with_cross_sections,
    create_cross_section_dashboard,
    plot_flow_regime_profile,
    create_flow_regime_dashboard,
    
    # Time series visualizations
    create_enhanced_flood_animation,
    create_flow_regime_transition_animation,
    create_animated_cross_section,
    
    # Interactive visualizations
    create_interactive_profile,
    create_interactive_cross_section_viewer,
    create_interactive_flood_explorer,
    
    # Helper functions
    create_single_result_visualization,
    create_multi_result_visualization,
    visualize_hydraulic_jump
)

# Import styling module
from src.visualization.styling import (
    apply_theme, 
    style_for_publication,
    style_for_presentation
)

def run_basic_visualization_examples():
    """
    Run examples of basic, enhanced single-result visualizations.
    """
    print("\n=== Running Basic Visualization Examples ===")
    
    # Create scenario and run analysis
    scenario = create_scenario()
    results = analyze_steady_state(scenario)
    
    # Apply default theme for consistent styling
    apply_theme('default')
    
    # Example 1: Enhanced profile visualization
    print("Creating enhanced profile visualization...")
    fig, ax = plot_enhanced_profile(scenario, results, color_by='froude')
    plt.savefig('examples/output/enhanced_profile.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 2: Profile with parameter plots
    print("Creating profile with parameter plots...")
    fig, axs = create_profile_with_parameter_plots(
        scenario, results, 
        parameters=['velocity', 'froude', 'depth']
    )
    plt.savefig('examples/output/profile_with_parameters.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 3: Profile with cross-sections
    print("Creating profile with cross-sections...")
    fig, axs = plot_enhanced_profile_with_cross_sections(scenario, results)
    plt.savefig('examples/output/profile_with_cross_sections.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 4: Comprehensive cross-section dashboard
    print("Creating cross-section dashboard...")
    fig, axs = create_cross_section_dashboard(scenario, results)
    plt.savefig('examples/output/cross_section_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 5: Flow regime visualization
    print("Creating flow regime visualization...")
    fig, ax = plot_flow_regime_profile(scenario, results)
    plt.savefig('examples/output/flow_regime_profile.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 6: Comprehensive flow regime dashboard
    print("Creating flow regime dashboard...")
    fig, axs = create_flow_regime_dashboard(scenario, results)
    plt.savefig('examples/output/flow_regime_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 7: Publication-quality figure
    print("Creating publication-quality figure...")
    fig, ax = plot_enhanced_profile(scenario, results, color_by='froude')
    style_for_publication(fig)
    plt.savefig('examples/output/publication_quality.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # Example 8: Presentation-quality figure
    print("Creating presentation-quality figure...")
    fig, axs = create_flow_regime_dashboard(scenario, results)
    style_for_presentation(fig)
    plt.savefig('examples/output/presentation_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Basic visualization examples completed. Output saved to examples/output/ directory.")


def run_flood_visualization_examples():
    """
    Run examples of flood/time-series visualizations.
    """
    print("\n=== Running Flood Visualization Examples ===")
    
    # Create scenario
    scenario = create_scenario()
    
    # Create more water levels for smoother flood simulation (increased from 8 to 20)
    initial_level = scenario['initial_water_level']
    flood_level = scenario['flood_water_level']
    water_levels = np.linspace(initial_level, flood_level, 20)
    
    # Generate results for each water level
    results_list = []
    for level in water_levels:
        results = analyze_steady_state(scenario, level)
        results_list.append(results)
    
    # Add "pause frames" at critical points to emphasize important states
    # Identify hydraulic jump formation point
    jump_frame_index = None
    for i, result in enumerate(results_list):
        if result['hydraulic_jump'].get('jump_possible', False):
            jump_frame_index = i
            break
    
    # If a jump is found, duplicate those frames to create a pause effect
    enhanced_results_list = list(results_list)
    if jump_frame_index is not None:
        # Duplicate the jump frame 3 times to create a pause
        jump_result = results_list[jump_frame_index]
        enhanced_results_list.insert(jump_frame_index, jump_result)
        enhanced_results_list.insert(jump_frame_index, jump_result)
        enhanced_results_list.insert(jump_frame_index, jump_result)
        print(f"Hydraulic jump identified at frame {jump_frame_index}, adding pause frames")
    
    # Example 1: Enhanced flood animation with improved visualization
    print("Creating enhanced flood animation...")
    anim, fig, ax = create_enhanced_flood_animation(
        scenario, enhanced_results_list, 
        fps=2,  # Reduced from 5 to 2 for slower animation
        color_by='froude',
        highlight_jump=True,  # Add a parameter to highlight the jump
        show_annotations=True  # Add annotations for clarity
    )
    
    # Save animation with slower frame rate
    try:
        anim.save('examples/output/enhanced_flood_animation.gif', writer='pillow', fps=2)
        print("Animation saved as enhanced_flood_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Save a preview frame instead
        plt.savefig('examples/output/enhanced_flood_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 2: Flow regime transition animation (also slowed down)
    print("Creating flow regime transition animation...")
    anim, fig, axs = create_flow_regime_transition_animation(
        scenario, enhanced_results_list, 
        fps=2  # Reduced from 5 to 2
    )
    
    # Save animation
    try:
        anim.save('examples/output/flow_regime_animation.gif', writer='pillow', fps=2)
        print("Animation saved as flow_regime_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Save a preview frame instead
        plt.savefig('examples/output/flow_regime_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 3: Animated cross-section (slowed down)
    print("Creating animated cross-section...")
    
    # If a jump exists, use that location for the cross-section
    cross_section_location = 50  # default
    if jump_frame_index is not None:
        jump_result = results_list[jump_frame_index]
        if 'location' in jump_result['hydraulic_jump']:
            cross_section_location = jump_result['hydraulic_jump']['location']
            print(f"Using hydraulic jump location for cross-section: {cross_section_location}m")
    
    anim, fig, ax = create_animated_cross_section(
        scenario, enhanced_results_list, 
        location=cross_section_location,
        fps=2,  # Reduced from 5 to 2
        show_water_depth=True,  # Add water depth labels
        show_velocity=True,     # Add velocity labels
        show_froude=True        # Add Froude number labels
    )
    
    # Save animation
    try:
        anim.save('examples/output/animated_cross_section.gif', writer='pillow', fps=2)
        print("Animation saved as animated_cross_section.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Save a preview frame instead
        plt.savefig('examples/output/animated_cross_section_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 4: Multi-result visualization (static comparison)
    print("Creating multi-result visualization...")
    
    fig, axs = create_multi_result_visualization(
        scenario, results_list, 
        filename='examples/output/multi_result_comparison.png',
        animation=False, interactive=False
    )
    plt.close(fig)
    
    # Example 5: Hydraulic jump visualization for result with jump
    print("Creating hydraulic jump visualization...")
    # Find a result with a hydraulic jump
    jump_result = None
    for result in results_list:
        if result['hydraulic_jump'].get('jump_possible', False):
            jump_result = result
            break
    
    if jump_result:
        fig, axs = visualize_hydraulic_jump(
            scenario, jump_result,
            filename='examples/output/hydraulic_jump_analysis.png'
        )
        plt.close(fig)
    else:
        print("No hydraulic jump found in results")
    
    print("Flood visualization examples completed. Output saved to examples/output/ directory.")


def create_enhanced_flood_animation(scenario, results_list, fps=2, color_by='froude', 
                                   highlight_jump=True, show_annotations=True):
    """
    Create an enhanced animation of flood conditions with better visualization
    of hydraulic jumps and flow regimes.
    
    Parameters:
    -----------
    scenario : dict
        Scenario dictionary with all geometry information
    results_list : list
        List of result dictionaries for different water levels
    fps : int, optional
        Frames per second for the animation
    color_by : str, optional
        Parameter to color the water surface by ('froude', 'velocity', 'depth')
    highlight_jump : bool, optional
        Whether to highlight hydraulic jumps with markers
    show_annotations : bool, optional
        Whether to show annotations explaining flow regimes
        
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Initialize with first frame
    line_wse, = ax.plot([], [], 'b-', lw=2, label='Water Surface')
    line_bed, = ax.plot([], [], 'brown', lw=2, label='Channel Bed')
    
    # Dam representation
    dam_x = []
    dam_y = []
    line_dam, = ax.plot(dam_x, dam_y, 'k-', lw=3, label='Dam')
    
    # Add jump marker (initially empty)
    jump_marker, = ax.plot([], [], 'ro', ms=10, label='Hydraulic Jump')
    
    # Text annotations
    flow_regime_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                              fontsize=12, fontweight='bold', va='top')
    
    water_level_text = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                              fontsize=12, ha='right', va='top')
    
    # Set up the plot
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Flood Progression and Hydraulic Behavior', fontsize=16)
    ax.grid(alpha=0.3)
    
    # Add colorbar for Froude number or other parameter
    sm = plt.cm.ScalarMappable(cmap='viridis')
    if color_by == 'froude':
        sm.set_array(np.linspace(0, 2, 100))  # Froude range 0-2
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Froude Number')
    elif color_by == 'velocity':
        sm.set_array(np.linspace(0, 10, 100))  # Velocity range 0-10 m/s
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Velocity (m/s)')
    else:  # depth
        sm.set_array(np.linspace(0, 5, 100))  # Depth range 0-5 m
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Water Depth (m)')
    
    # Get x-values for the bed profile
    if 'tailwater' in results_list[0] and 'x' in results_list[0]['tailwater']:
        x_values = results_list[0]['tailwater']['x']
        # Include negative x for upstream reservoir
        x_min = min(0, min(x_values)) - 10
        x_max = max(x_values) + 10
    else:
        # Default x range if not available
        x_min = -50
        x_max = 150
    
    # Get bed profile
    if 'tailwater' in results_list[0] and 'z' in results_list[0]['tailwater']:
        bed_profile = results_list[0]['tailwater']['z']
    else:
        # Default bed profile if not available
        bed_profile = np.ones_like(x_values) * scenario.get('dam_base_elevation', 90)
    
    # Setup for upstream reservoir (before dam at x=0)
    if x_min < 0:
        # Extend x values to include upstream reservoir
        upstream_x = np.linspace(x_min, 0, 10)
        x_full = np.concatenate([upstream_x, x_values])
        
        # Extend bed profile for upstream
        upstream_bed = np.ones_like(upstream_x) * scenario.get('dam_base_elevation', 90)
        bed_full = np.concatenate([upstream_bed, bed_profile])
    else:
        x_full = x_values
        bed_full = bed_profile
    
    # Set up the dam geometry
    dam = scenario.get('dam', None)
    if dam:
        # Extract dam profile or create simple triangular dam
        dam_base_elev = scenario.get('dam_base_elevation', 90)
        dam_height = scenario.get('dam_height', 10)
        dam_crest_width = getattr(dam, 'crest_width', 1)
        
        # Simple dam profile
        dam_x = [-2, -1, 0, 1, 2]
        dam_y = [dam_base_elev, dam_base_elev + dam_height, 
                dam_base_elev + dam_height, 
                dam_base_elev + dam_height, dam_base_elev]
        
        # Update dam line
        line_dam.set_data(dam_x, dam_y)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    bed_min = min(bed_full) - 2
    crest_elev = scenario.get('dam_crest_elevation', 100)
    max_water_level = max(r.get('upstream_level', crest_elev) for r in results_list)
    ax.set_ylim(bed_min, max_water_level + 2)
    
    # Legend
    ax.legend(loc='upper right')
    
    # Create common colormaps for the animation
    def get_colors(values, param='froude'):
        if param == 'froude':
            # Blue to red colormap for Froude number
            # Subcritical (Fr<1) is blue, Supercritical (Fr>1) is red
            norm = plt.Normalize(0, 2)
            return plt.cm.coolwarm(norm(values))
        elif param == 'velocity':
            # Blue to green to red for velocity
            norm = plt.Normalize(0, 10)
            return plt.cm.viridis(norm(values))
        else:  # depth
            norm = plt.Normalize(0, 5)
            return plt.cm.Blues(norm(values))
    
    # Initialization function for animation
    def init():
        line_wse.set_data([], [])
        line_bed.set_data(x_full, bed_full)
        jump_marker.set_data([], [])
        flow_regime_text.set_text('')
        water_level_text.set_text('')
        return line_wse, line_bed, line_dam, jump_marker, flow_regime_text, water_level_text
    
    # Animation function
    def animate(i):
        result = results_list[i]
        
        # Update water level text
        water_level = result.get('upstream_level', 0)
        water_level_text.set_text(f'Water Level: {water_level:.2f} m')
        
        # Get tailwater profile
        tailwater = result.get('tailwater', {})
        if 'x' in tailwater and 'wse' in tailwater:
            x_tw = tailwater['x']
            wse_tw = tailwater['wse']
            
            # Get upstream reservoir level
            upstream_level = result.get('upstream_level', crest_elev)
            
            # Create full water surface profile
            if x_min < 0:
                # Add upstream reservoir
                upstream_x = np.linspace(x_min, 0, 10)
                upstream_wse = np.ones_like(upstream_x) * upstream_level
                
                # Full water surface profile
                x_full = np.concatenate([upstream_x, x_tw])
                wse_full = np.concatenate([upstream_wse, wse_tw])
            else:
                x_full = x_tw
                wse_full = wse_tw
            
            # Color by parameter
            if color_by in ['froude', 'velocity', 'depth'] and color_by in tailwater:
                param_values = tailwater[color_by]
                
                # Pad with zeros for upstream reservoir
                if x_min < 0:
                    param_values = np.concatenate([np.zeros(len(upstream_x)), param_values])
                
                # Get colors
                colors = get_colors(param_values, color_by)
                
                # Create colored segments
                for j in range(len(x_full)-1):
                    ax.plot(x_full[j:j+2], wse_full[j:j+2], color=colors[j], lw=2)
            
            # Update water surface line
            line_wse.set_data(x_full, wse_full)
            
            # Check for hydraulic jump
            jump = result.get('hydraulic_jump', {})
            if highlight_jump and jump.get('jump_possible', False):
                jump_loc = jump.get('location', 0)
                jump_marker.set_data([jump_loc], [wse_tw[np.argmin(np.abs(x_tw - jump_loc))]])
                
                if show_annotations:
                    # Add annotation about the jump
                    flow_regime_text.set_text(
                        f"Hydraulic Jump at x={jump_loc:.1f}m\n"
                        f"Fr1={jump.get('fr1', 0):.2f} → Fr2={jump.get('fr2', 0):.2f}\n"
                        f"y1={jump.get('y1', 0):.2f}m → y2={jump.get('y2', 0):.2f}m"
                    )
            else:
                jump_marker.set_data([], [])
                
                if show_annotations:
                    # Flow regime information
                    if 'fr' in tailwater and len(tailwater['fr']) > 0:
                        max_fr = max(tailwater['fr'])
                        min_fr = min(tailwater['fr'])
                        if max_fr > 1.0:
                            regime = "Mixed Flow (Sub & Supercritical)"
                        elif max_fr > 0.7:
                            regime = "Subcritical Flow (approaching critical)"
                        else:
                            regime = "Subcritical Flow"
                        
                        flow_regime_text.set_text(
                            f"Flow Regime: {regime}\n"
                            f"Max Fr: {max_fr:.2f}, Min Fr: {min_fr:.2f}"
                        )
        
        return line_wse, line_bed, line_dam, jump_marker, flow_regime_text, water_level_text
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(results_list), interval=1000/fps, blit=True
    )
    
    return anim, fig, ax


def create_animated_cross_section(scenario, results_list, location=50, fps=2,
                                 show_water_depth=True, show_velocity=True, 
                                 show_froude=True):
    """
    Create animation of a cross-section at a specific location as water levels change.
    
    Parameters:
    -----------
    scenario : dict
        Scenario dictionary with all geometry information
    results_list : list
        List of result dictionaries for different water levels
    location : float
        Location along the channel for the cross-section (distance from dam in meters)
    fps : int
        Frames per second for the animation
    show_water_depth : bool
        Whether to show water depth labels
    show_velocity : bool
        Whether to show velocity labels
    show_froude : bool
        Whether to show Froude number labels
        
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get channel parameters
    channel_type = 'trapezoidal'  # Assuming trapezoidal channel
    channel_params = {
        'bottom_width': scenario.get('channel_bottom_width', 10),
        'side_slope': scenario.get('channel_side_slope', 1.5)
    }
    
    # Get cross-section width for plotting
    bottom_width = channel_params['bottom_width']
    max_depth = 5  # Assume maximum depth for plotting
    side_slope = channel_params['side_slope']
    top_width = bottom_width + 2 * side_slope * max_depth
    
    # Set up plot limits
    ax.set_xlim(-top_width/2, top_width/2)
    ax.set_ylim(0, max_depth + 0.5)
    
    # Plot channel cross-section
    left_bank_x = -bottom_width/2 - side_slope * max_depth
    right_bank_x = bottom_width/2 + side_slope * max_depth
    
    # Channel bed
    ax.plot([-bottom_width/2, bottom_width/2], [0, 0], 'brown', lw=2)
    # Left bank
    ax.plot([-bottom_width/2, left_bank_x], [0, max_depth], 'brown', lw=2)
    # Right bank
    ax.plot([bottom_width/2, right_bank_x], [0, max_depth], 'brown', lw=2)
    
    # Water surface (initially empty)
    water_line, = ax.plot([], [], 'b-', lw=2)
    water_fill = ax.fill_between([-bottom_width/2, bottom_width/2], 0, 0, 
                                color='skyblue', alpha=0.5)
    
    # Add information text
    info_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.7))
    
    # Title and labels
    ax.set_title(f'Cross-Section at x={location}m', fontsize=14)
    ax.set_xlabel('Width (m)')
    ax.set_ylabel('Depth (m)')
    ax.grid(alpha=0.3)
    
    # Initialize animation
    def init():
        water_line.set_data([], [])
        info_text.set_text('')
        return water_line, info_text
    
    # Animation function
    def animate(i):
        result = results_list[i]
        
        # Get water depth at this location
        water_depth = 0
        velocity = 0
        froude = 0
        
        # Check if location is upstream or downstream of dam
        if location <= 0:  # Upstream of dam
            water_elevation = result.get('upstream_level', 0)
            bed_elevation = scenario.get('dam_base_elevation', 0)
            water_depth = max(0, water_elevation - bed_elevation)
            
            # Estimate velocity and Froude
            if water_depth > 0 and result.get('discharge', 0) > 0:
                area = water_depth * scenario.get('channel_width_at_dam', bottom_width)
                velocity = result.get('discharge', 0) / area
                froude = velocity / np.sqrt(9.81 * water_depth)
        else:  # Downstream of dam
            # Find in tailwater results
            tailwater = result.get('tailwater', {})
            if 'x' in tailwater and 'wse' in tailwater and 'z' in tailwater:
                x_values = tailwater['x']
                wse = tailwater['wse']
                bed_z = tailwater['z']
                
                # Find closest point to location
                idx = np.argmin(np.abs(x_values - location))
                
                if idx < len(wse) and idx < len(bed_z):
                    water_elevation = wse[idx]
                    bed_elevation = bed_z[idx]
                    water_depth = max(0, water_elevation - bed_elevation)
                    
                    # Get velocity and Froude if available
                    if 'v' in tailwater and idx < len(tailwater['v']):
                        velocity = tailwater['v'][idx]
                    if 'fr' in tailwater and idx < len(tailwater['fr']):
                        froude = tailwater['fr'][idx]
        
        # Calculate top width for this water depth
        water_top_width = bottom_width + 2 * side_slope * water_depth
        
        # Update water surface line
        x_water = [-water_top_width/2, water_top_width/2]
        y_water = [water_depth, water_depth]
        water_line.set_data(x_water, y_water)
        
        # Update water fill area
        # Create polygon vertices for water area
        x_fill = [-bottom_width/2]
        y_fill = [0]
        
        if water_depth > 0:
            # Left bank underwater portion
            if water_depth > 0:
                x_left = -bottom_width/2 - side_slope * water_depth
                x_fill.append(x_left)
                y_fill.append(water_depth)
            
            # Water surface
            x_fill.extend([-water_top_width/2, water_top_width/2])
            y_fill.extend([water_depth, water_depth])
            
            # Right bank underwater portion
            if water_depth > 0:
                x_right = bottom_width/2 + side_slope * water_depth
                x_fill.append(x_right)
                y_fill.append(water_depth)
            
            # Complete the polygon
            x_fill.append(bottom_width/2)
            y_fill.append(0)
        
        # Update water fill
        ax.collections.clear()  # Clear previous fill
        ax.fill_between(x_fill, 0, y_fill, color='skyblue', alpha=0.5)
        
        # Calculate wetted perimeter and hydraulic radius
        if water_depth > 0:
            wetted_perimeter = bottom_width + 2 * water_depth * np.sqrt(1 + side_slope**2)
            area = water_depth * (bottom_width + side_slope * water_depth)
            hydraulic_radius = area / wetted_perimeter
        else:
            wetted_perimeter = 0
            area = 0
            hydraulic_radius = 0
        
        # Update info text
        info_lines = []
        if show_water_depth:
            info_lines.append(f"Water Depth: {water_depth:.2f} m")
        if show_velocity:
            info_lines.append(f"Velocity: {velocity:.2f} m/s")
        if show_froude:
            info_lines.append(f"Froude Number: {froude:.2f}")
            # Add flow regime description
            if froude < 0.01:
                info_lines.append("Flow Regime: None/Dry")
            elif froude < 1:
                info_lines.append("Flow Regime: Subcritical")
            else:
                info_lines.append("Flow Regime: Supercritical")
        
        # Add hydraulic parameters
        info_lines.append(f"Area: {area:.2f} m²")
        info_lines.append(f"Wetted Perimeter: {wetted_perimeter:.2f} m")
        info_lines.append(f"Hydraulic Radius: {hydraulic_radius:.2f} m")
        
        # Update water level info
        upstream_level = result.get('upstream_level', 0)
        info_lines.append(f"Upstream Water Level: {upstream_level:.2f} m")
        
        # Format info text
        info_text.set_text('\n'.join(info_lines))
        
        # Add a border around the text for better visibility
        info_text.set_bbox(dict(facecolor='white', alpha=0.7, 
                               edgecolor='black', boxstyle='round'))
        
        return water_line, info_text
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(results_list), interval=1000/fps, blit=True
    )
    
    return anim, fig, ax


def run_interactive_examples():
    """
    Run examples of interactive visualizations.
    
    Note: These will display in a matplotlib window and won't save automatically.
    You'll need to interact with them and close manually.
    """
    print("\n=== Running Interactive Visualization Examples ===")
    print("Note: Interactive examples will open in matplotlib windows.")
    print("Close each window to proceed to the next example.")
    
    # Check if we're in interactive mode
    if not plt.isinteractive():
        plt.ion()  # Turn on interactive mode
    
    # Create scenario
    scenario = create_scenario()
    results = analyze_steady_state(scenario)
    
    # Create multiple water levels for flood simulation
    initial_level = scenario['initial_water_level']
    flood_level = scenario['flood_water_level']
    water_levels = np.linspace(initial_level, flood_level, 8)
    
    # Generate results for each water level
    results_list = []
    for level in water_levels:
        results = analyze_steady_state(scenario, level)
        results_list.append(results)
    
    # Example 1: Interactive cross-section viewer
    print("\nOpening interactive cross-section viewer...")
    fig, axs = create_interactive_cross_section_viewer(scenario, results)
    plt.show()
    input("Press Enter to close this example and continue...")
    plt.close(fig)
    
    # Example 2: Interactive flood explorer
    print("\nOpening interactive flood explorer...")
    fig, axs = create_interactive_flood_explorer(scenario, results_list)
    plt.show()
    input("Press Enter to close this example and continue...")
    plt.close(fig)
    
    # Example 3: Interactive profile with controls
    print("\nOpening interactive profile with controls...")
    fig, ax = create_interactive_profile(scenario, results_list)
    plt.show()
    input("Press Enter to close this example and continue...")
    plt.close(fig)
    
    print("Interactive examples completed.")


def run_alternative_dam_types():
    """
    Generate visualizations for different dam types for comparison.
    """
    print("\n=== Running Alternative Dam Types Examples ===")
    
    # Base scenario
    base_scenario = create_scenario()
    
    # Create three different dam types with similar parameters
    dam_height = 10.0
    crest_elevation = 100.0
    dam_types = [
        # Ogee weir (already in base scenario)
        {
            'type': 'ogee',
            'name': 'Ogee Spillway',
            'dam': OgeeWeir(
                height=dam_height,
                crest_elevation=crest_elevation,
                design_head=2.0
            )
        },
        # Broad-crested weir
        {
            'type': 'broad_crested',
            'name': 'Broad-Crested Weir',
            'dam': BroadCrestedWeir(
                height=dam_height,
                crest_elevation=crest_elevation,
                crest_width=3.0
            )
        },
        # Sharp-crested weir
        {
            'type': 'sharp_crested',
            'name': 'Sharp-Crested Weir',
            'dam': base_scenario['dam'].__class__(  # Just using OgeeWeir for now as placeholder
                height=dam_height,
                crest_elevation=crest_elevation,
                design_head=0.5  # Smaller design head for sharper crest
            )
        }
    ]
    
    # Create scenarios and generate results for each dam type
    for dam_type in dam_types:
        # Create new scenario based on base scenario
        scenario = dict(base_scenario)
        scenario['dam'] = dam_type['dam']
        
        # Analyze flow
        results = analyze_steady_state(scenario)
        
        # Create flow regime visualization
        print(f"Creating flow regime visualization for {dam_type['name']}...")
        fig, ax = plot_flow_regime_profile(
            scenario, results,
            show_annotations=True
        )
        plt.title(f"Flow Regime Profile - {dam_type['name']}")
        plt.savefig(f"examples/output/flow_regime_{dam_type['type']}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create cross-section dashboard
        print(f"Creating cross-section dashboard for {dam_type['name']}...")
        fig, axs = create_cross_section_dashboard(scenario, results)
        plt.suptitle(f"Cross-Section Analysis - {dam_type['name']}", 
                    fontsize=16, fontweight='bold')
        plt.savefig(f"examples/output/cross_sections_{dam_type['type']}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print("Alternative dam types examples completed. Output saved to examples/output/ directory.")


def run_custom_visualization_example():
    """
    Show how to create custom visualizations by combining 
    and extending the existing components.
    """
    print("\n=== Running Custom Visualization Example ===")
    
    # Create scenario and analyze
    scenario = create_scenario()
    results = analyze_steady_state(scenario)
    
    # Create a custom multi-panel visualization
    print("Creating custom 4-panel visualization...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], figure=fig)
    
    # Top row spans both columns - main profile
    ax_profile = fig.add_subplot(gs[0, :])
    
    # Middle row - Froude number and velocity
    ax_froude = fig.add_subplot(gs[1, 0])
    ax_velocity = fig.add_subplot(gs[1, 1])
    
    # Bottom row - cross-section and energy
    ax_cross = fig.add_subplot(gs[2, 0])
    ax_energy = fig.add_subplot(gs[2, 1])
    
    # Plot main profile with flow regimes
    plot_flow_regime_profile(scenario, results, ax=ax_profile)
    
    # Extract necessary data
    tailwater = results['tailwater']
    x_values = tailwater['x']
    
    # Plot Froude number
    ax_froude.plot(x_values, tailwater['fr'], 'r-', linewidth=2)
    ax_froude.axhline(y=1, color='black', linestyle='--', label='Critical (Fr=1)')
    ax_froude.set_xlim(ax_profile.get_xlim())
    ax_froude.set_ylabel('Froude Number')
    ax_froude.set_title('Froude Number Profile')
    ax_froude.grid(True, alpha=0.3)
    ax_froude.legend()
    
    # Plot velocity
    ax_velocity.plot(x_values, tailwater['v'], 'g-', linewidth=2)
    ax_velocity.set_xlim(ax_profile.get_xlim())
    ax_velocity.set_ylabel('Velocity (m/s)')
    ax_velocity.set_title('Velocity Profile')
    ax_velocity.grid(True, alpha=0.3)
    
    # Plot cross-section
    from src.visualization.cross_sections import plot_channel_cross_section
    
    # Choose a location - hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        loc = jump['location']
    else:
        # Default to location with highest Froude number
        if len(tailwater['fr']) > 0:
            idx = np.argmax(tailwater['fr'])
            loc = x_values[idx]
        else:
            loc = 50  # Default
    
    # Determine water depth and parameters at this location
    if loc <= 0:  # Upstream of dam
        water_elevation = results['upstream_level']
        bed_elevation = scenario['dam_base_elevation']
        water_depth = max(0, water_elevation - bed_elevation)
        
        # Estimate velocity and Froude
        if water_depth > 0 and results['discharge'] > 0:
            area = water_depth * scenario['channel_width_at_dam']
            velocity = results['discharge'] / area
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
            velocity = tailwater['v'][idx] if idx < len(tailwater['v']) else 0
            froude = tailwater['fr'][idx] if idx < len(tailwater['fr']) else 0
        else:
            water_depth = 0
            velocity = 0
            froude = 0
    
    # Plot cross-section
    channel_type = 'trapezoidal'
    channel_params = {
        'bottom_width': scenario['channel_bottom_width'],
        'side_slope': scenario['channel_side_slope']
    }
    
    plot_channel_cross_section(
        ax_cross, channel_type, channel_params, water_depth,
        highlight_param={'type': 'froude', 'value': froude, 'max_value': 2.0}
    )
    
    ax_cross.set_title(f'Cross-Section at x={loc:.1f}m')
    
    # Plot energy profile
    # Estimate specific energy
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
    
    # Mark jump location on all plots if present
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        ax_profile.axvline(x=jump_loc, color='magenta', linestyle='-.', alpha=0.7)
        ax_froude.axvline(x=jump_loc, color='magenta', linestyle='-.', alpha=0.7)
        ax_velocity.axvline(x=jump_loc, color='magenta', linestyle='-.', alpha=0.7)
        ax_energy.axvline(x=jump_loc, color='magenta', linestyle='-.', alpha=0.7)
    
    # Customize for presentation
    style_for_presentation(fig)
    
    # Add overall title
    fig.suptitle('Custom Hydraulic Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Set common x-axis label for bottom row only
    for ax in [ax_froude, ax_velocity]:
        ax.set_xlabel('')
    
    for ax in [ax_cross, ax_energy]:
        ax.set_xlabel('Distance (m)')
    
    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig('examples/output/custom_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Custom visualization example completed. Output saved to examples/output/custom_visualization.png")


def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs('examples/output', exist_ok=True)


if __name__ == "__main__":
    # Create output directory
    ensure_output_directory()
    
    # Run examples
    run_basic_visualization_examples()
    run_flood_visualization_examples()
    run_alternative_dam_types()
    run_custom_visualization_example()
    
    # Run interactive examples if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_examples()
    else:
        print("\nTo run interactive examples, use: python visualization_examples.py --interactive")
    
    print("\nAll examples completed. Results saved to examples/output/ directory.")