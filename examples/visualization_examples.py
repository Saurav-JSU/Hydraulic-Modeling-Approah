"""
Example script demonstrating the enhanced visualization capabilities.

This script shows how to use the various visualization tools to create
informative and visually appealing hydraulic visualizations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Create multiple water levels for flood simulation
    initial_level = scenario['initial_water_level']
    flood_level = scenario['flood_water_level']
    water_levels = np.linspace(initial_level, flood_level, 8)
    
    # Generate results for each water level
    results_list = []
    for level in water_levels:
        results = analyze_steady_state(scenario, level)
        results_list.append(results)
    
    # Example 1: Enhanced flood animation
    print("Creating enhanced flood animation...")
    anim, fig, ax = create_enhanced_flood_animation(
        scenario, results_list, fps=5, color_by='froude'
    )
    
    # Save animation
    try:
        anim.save('examples/output/enhanced_flood_animation.gif', writer='pillow', fps=5)
        print("Animation saved as enhanced_flood_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Save a preview frame instead
        plt.savefig('examples/output/enhanced_flood_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 2: Flow regime transition animation
    print("Creating flow regime transition animation...")
    anim, fig, axs = create_flow_regime_transition_animation(
        scenario, results_list, fps=5
    )
    
    # Save animation
    try:
        anim.save('examples/output/flow_regime_animation.gif', writer='pillow', fps=5)
        print("Animation saved as flow_regime_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")
        # Save a preview frame instead
        plt.savefig('examples/output/flow_regime_preview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Example 3: Animated cross-section
    print("Creating animated cross-section...")
    anim, fig, ax = create_animated_cross_section(
        scenario, results_list, location=50, fps=5
    )
    
    # Save animation
    try:
        anim.save('examples/output/animated_cross_section.gif', writer='pillow', fps=5)
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