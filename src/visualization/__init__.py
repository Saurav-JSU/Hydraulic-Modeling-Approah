"""
Enhanced visualization package for hydraulic modeling.

This package provides comprehensive visualization tools for hydraulic
calculations, including profile plots, animations, cross-sections,
flow regime analysis, and interactive exploration.
"""

# Import from basic visualization module
from .profiles import (
    plot_full_profile,
    plot_velocity_profile,
    plot_froude_profile,
    create_flood_animation
)

# Import from enhanced visualization modules
from .enhanced_profiles import (
    plot_enhanced_profile,
    create_profile_with_parameter_plots,
    plot_enhanced_profile_with_cross_sections
)

from .time_series import (
    create_enhanced_flood_animation,
    create_flow_regime_transition_animation
)

from .cross_sections import (
    plot_channel_cross_section,
    create_cross_section_dashboard,
    create_animated_cross_section
)

from .flow_regimes import (
    plot_flow_regime_profile,
    plot_froude_profile as plot_detailed_froude_profile,
    create_flow_regime_dashboard,
    plot_regime_map
)

from .interactive import (
    create_interactive_profile,
    create_interactive_cross_section_viewer,
    create_interactive_flood_explorer
)

# Package metadata
__version__ = '0.2.0'


# ============= Helper Functions for Common Visualization Tasks =============

def create_single_result_visualization(scenario, results, filename=None, 
                                     include_regimes=True, include_cross_sections=True):
    """
    Create a comprehensive visualization of a single hydraulic result.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        filename (str, optional): Filename to save the figure
        include_regimes (bool): Whether to include flow regime visualization
        include_cross_sections (bool): Whether to include cross-section views
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    import matplotlib.pyplot as plt
    
    # Determine what to include based on parameters
    if include_regimes and include_cross_sections:
        # Create flow regime dashboard with cross-sections
        fig, axes = create_flow_regime_dashboard(scenario, results)
    elif include_regimes:
        # Create flow regime visualization without cross-sections
        fig, ax = plot_flow_regime_profile(scenario, results)
        axes = ax
    elif include_cross_sections:
        # Create profile with cross-sections but without regime highlighting
        fig, axes = plot_enhanced_profile_with_cross_sections(scenario, results)
    else:
        # Create basic enhanced profile
        fig, ax = plot_enhanced_profile(scenario, results)
        axes = ax
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as {filename}")
    
    return fig, axes


def create_multi_result_visualization(scenario, results_list, filename=None,
                                    animation=True, interactive=True):
    """
    Create a visualization for multiple hydraulic results (e.g., flood progression).
    
    Parameters:
        scenario (dict): The scenario parameters
        results_list (list): List of analysis results
        filename (str, optional): Filename to save the figure/animation
        animation (bool): Whether to create an animation
        interactive (bool): Whether to create interactive controls
        
    Returns:
        If animation: tuple of (anim, fig, axes)
        If interactive: tuple of (fig, axes)
        Otherwise: tuple of (fig, axes) for static plot
    """
    import matplotlib.pyplot as plt
    
    if interactive and plt.isinteractive():
        # Create interactive flood explorer
        fig, axes = create_interactive_flood_explorer(scenario, results_list)
        
        # Save static version if filename provided
        if filename:
            plt.savefig(f"{filename.split('.')[0]}_preview.png", dpi=300, bbox_inches='tight')
            print(f"Preview saved as {filename.split('.')[0]}_preview.png")
        
        return fig, axes
        
    elif animation:
        # Create enhanced animation
        anim, fig, ax = create_enhanced_flood_animation(scenario, results_list)
        
        # Save animation if filename provided
        if filename:
            # Try different writers based on filename extension
            if filename.endswith('.mp4'):
                try:
                    from matplotlib.animation import FFMpegWriter
                    anim.save(filename, writer=FFMpegWriter(fps=5), dpi=200)
                    print(f"Animation saved as {filename}")
                except Exception as e:
                    print(f"Could not save MP4 animation: {e}")
                    # Fallback to GIF
                    try:
                        anim.save(f"{filename.split('.')[0]}.gif", writer='pillow', fps=5)
                        print(f"Animation saved as {filename.split('.')[0]}.gif")
                    except:
                        print("Could not save animation")
            elif filename.endswith('.gif'):
                try:
                    anim.save(filename, writer='pillow', fps=5)
                    print(f"Animation saved as {filename}")
                except Exception as e:
                    print(f"Could not save GIF animation: {e}")
            else:
                print("Unsupported animation format. Use .mp4 or .gif extension.")
        
        return anim, fig, ax
        
    else:
        # Create static comparison of first, middle, and last result
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Choose three representative results
        indices = [0, len(results_list)//2, -1]
        titles = ['Initial Condition', 'Intermediate Stage', 'Peak Condition']
        
        axes = []
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs[i, 0])
            plot_enhanced_profile(scenario, results_list[idx], ax=ax)
            ax.set_title(f"{titles[i]}: Water Level = {results_list[idx]['upstream_level']:.2f}m, "
                       f"Discharge = {results_list[idx]['discharge']:.2f}m³/s")
            axes.append(ax)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as {filename}")
        
        return fig, axes


def visualize_hydraulic_jump(scenario, results, filename=None, detailed=True):
    """
    Create specialized visualization focusing on hydraulic jump characteristics.
    
    Parameters:
        scenario (dict): The scenario parameters
        results (dict): The analysis results
        filename (str, optional): Filename to save the figure
        detailed (bool): Whether to show detailed jump characteristics
        
    Returns:
        tuple: (fig, axes) The figure and axes objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Check if there's a hydraulic jump
    jump = results['hydraulic_jump']
    if not jump.get('jump_possible', False):
        print("No hydraulic jump detected in these results")
        # Create a basic profile visualization instead
        return plot_enhanced_profile(scenario, results)
    
    # Extract jump data
    jump_loc = jump['location']
    initial_depth = jump['initial_depth']
    sequent_depth = jump['sequent_depth']
    froude1 = jump['initial_froude']
    jump_type = jump['jump_type']
    energy_loss = jump['energy_loss']
    
    # Create figure with multiple views of the jump
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, height_ratios=[2, 1], figure=fig)
    
    # Main profile view spanning top row
    ax_profile = fig.add_subplot(gs[0, :])
    
    # Detail views in bottom row
    ax_froude = fig.add_subplot(gs[1, 0])
    ax_cross = fig.add_subplot(gs[1, 1])
    ax_energy = fig.add_subplot(gs[1, 2])
    
    # Plot main profile with flow regimes
    plot_flow_regime_profile(scenario, results, ax=ax_profile)
    
    # Focus the view around the jump
    view_range = 100  # View range in meters
    ax_profile.set_xlim(jump_loc - view_range/2, jump_loc + view_range/2)
    
    # Add a rectangle highlighting the jump region
    from matplotlib.patches import Rectangle
    tailwater = results['tailwater']
    idx = np.argmin(np.abs(tailwater['x'] - jump_loc))
    if idx < len(tailwater['z']):
        bed_elevation = tailwater['z'][idx]
        highlight = Rectangle((jump_loc - 10, bed_elevation), 20, sequent_depth * 1.5,
                            facecolor='yellow', alpha=0.2, edgecolor='orange',
                            linewidth=2, linestyle='--')
        ax_profile.add_patch(highlight)
    
    # Plot Froude number
    x_values = tailwater['x']
    if len(tailwater['fr']) > 0:
        ax_froude.plot(x_values, tailwater['fr'], 'r-', linewidth=2)
        ax_froude.axhline(y=1, color='k', linestyle='--', label='Critical (Fr=1)')
        
        # Highlight jump region in Froude plot
        ax_froude.axvline(x=jump_loc, color='magenta', linestyle='-.', linewidth=2)
        
        # Add arrow showing Froude transition
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch((jump_loc - 5, froude1), (jump_loc + 5, 0.5),
                              connectionstyle="arc3,rad=0.3",
                              arrowstyle="-|>", color='magenta',
                              linewidth=2, alpha=0.9)
        ax_froude.add_patch(arrow)
        
        # Focus the view around the jump
        ax_froude.set_xlim(jump_loc - view_range/2, jump_loc + view_range/2)
    
    ax_froude.set_title('Froude Number Profile')
    ax_froude.set_xlabel('Distance from Dam (m)')
    ax_froude.set_ylabel('Froude Number')
    ax_froude.grid(True, alpha=0.3)
    
    # Plot jump cross-section
    from .cross_sections import plot_channel_cross_section
    channel_type = 'trapezoidal'
    channel_params = {
        'bottom_width': scenario['channel_bottom_width'],
        'side_slope': scenario['channel_side_slope']
    }
    
    # Create two cross-sections - before and after jump
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_cross = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.1)
    
    ax_cross.remove()  # Remove the original axis
    ax_cross1 = fig.add_subplot(gs_cross[0, 0])
    ax_cross2 = fig.add_subplot(gs_cross[0, 1])
    
    # Plot before jump (supercritical)
    highlight_before = {'type': 'froude', 'value': froude1, 'max_value': 3.0}
    plot_channel_cross_section(ax_cross1, channel_type, channel_params, initial_depth,
                              highlight_param=highlight_before, annotate=True)
    ax_cross1.set_title(f"Before Jump\nDepth={initial_depth:.2f}m, Fr={froude1:.2f}")
    
    # Plot after jump (subcritical)
    highlight_after = {'type': 'froude', 'value': 0.5, 'max_value': 3.0}  # Estimated Fr2
    plot_channel_cross_section(ax_cross2, channel_type, channel_params, sequent_depth,
                              highlight_param=highlight_after, annotate=True)
    ax_cross2.set_title(f"After Jump\nDepth={sequent_depth:.2f}m, Fr≈0.5")
    
    # Plot energy profile
    if 'energy' in tailwater:
        # Use energy profile from results
        energy_profile = tailwater['energy'] - tailwater['z']
        ax_energy.plot(x_values, energy_profile, 'purple', linewidth=2)
    else:
        # Estimate energy profile
        depths = tailwater['y']
        velocities = tailwater['v']
        if len(depths) > 0 and len(velocities) > 0:
            energy_profile = depths + velocities**2 / (2 * 9.81)
            ax_energy.plot(x_values, energy_profile, 'purple', linewidth=2)
    
    # Highlight jump region in energy plot
    ax_energy.axvline(x=jump_loc, color='magenta', linestyle='-.', linewidth=2)
    
    # Find energy before and after jump
    jump_index = np.argmin(np.abs(x_values - jump_loc))
    if jump_index < len(energy_profile):
        energy_before = energy_profile[jump_index]
        if jump_index + 5 < len(energy_profile):
            energy_after = energy_profile[jump_index + 5]  # A bit after the jump
        else:
            energy_after = energy_profile[-1]  # Last available value
        
        # Add rectangle showing energy loss
        rect = Rectangle((jump_loc - 5, energy_after), 10, energy_before - energy_after,
                        facecolor='red', alpha=0.3, edgecolor='red',
                        linewidth=1, linestyle='-')
        ax_energy.add_patch(rect)
        
        # Add annotation for energy loss
        ax_energy.annotate(
            f"Energy Loss\n{energy_loss:.2f} m",
            xy=(jump_loc, (energy_before + energy_after)/2),
            xytext=(jump_loc + 15, (energy_before + energy_after)/2),
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
            fontsize=9
        )
    
    # Focus the view around the jump
    ax_energy.set_xlim(jump_loc - view_range/2, jump_loc + view_range/2)
    ax_energy.set_title('Energy Profile')
    ax_energy.set_xlabel('Distance from Dam (m)')
    ax_energy.set_ylabel('Specific Energy (m)')
    ax_energy.grid(True, alpha=0.3)
    
    # Add overall title with jump information
    fig.suptitle(f'Hydraulic Jump Analysis\n'
               f'Type: {jump_type}, Location: {jump_loc:.2f}m downstream\n'
               f'Fr₁ = {froude1:.2f}, y₁ = {initial_depth:.2f}m → y₂ = {sequent_depth:.2f}m',
               fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Hydraulic jump visualization saved as {filename}")
    
    # Return figure and axes dictionary
    axes = {
        'profile': ax_profile,
        'froude': ax_froude,
        'cross1': ax_cross1,
        'cross2': ax_cross2,
        'energy': ax_energy
    }
    
    return fig, axes