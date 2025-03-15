"""
Unsteady flow analysis for dam and channel system.

This script performs unsteady flow analysis by simulating flood conditions
with increasing water levels and analyzing the corresponding changes.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steady_analysis import analyze_steady_state
from scenario_setup import create_scenario
from visualization import create_flood_animation, plot_full_profile

def simulate_flood_event(scenario, num_levels=10, save_animation=True):
    """
    Simulate a flood event by analyzing different water levels.
    
    Parameters:
        scenario (dict): Scenario parameters and objects
        num_levels (int): Number of water levels to analyze
        save_animation (bool): Whether to save the animation
        
    Returns:
        list: List of results for each water level
    """
    # Get initial and flood water levels
    initial_level = scenario['initial_water_level']
    flood_level = scenario['flood_water_level']
    dam_crest = scenario['dam_crest_elevation']
    
    # Create range of water levels from initial to flood
    water_levels = np.linspace(initial_level, flood_level, num_levels)
    
    # Analyze each water level
    results_list = []
    
    print("\nSimulating Flood Event")
    print("=" * 60)
    print(f"Initial water level: {initial_level:.2f} m")
    print(f"Flood peak water level: {flood_level:.2f} m")
    print(f"Dam crest elevation: {dam_crest:.2f} m")
    print("=" * 60)
    
    for i, level in enumerate(water_levels):
        print(f"\nSimulating level {i+1}/{num_levels}: {level:.2f} m "
              f"({max(0, level - dam_crest):.2f} m above crest)")
        
        # Skip analysis for levels well below dam crest - it would result in zero discharge
        if level < dam_crest - 0.1:
            print(f"  Water level below dam crest - no flow over dam")
            # Create a minimal result with no flow
            result = {
                'upstream_level': level,
                'discharge': 0,
                'head': 0,
                'velocity': 0,
                'energy_results': {'energy_loss': 0, 'dissipation_ratio': 0, 'power': 0},
                'tailwater': {'x': np.array([0, 100]), 'z': np.array([scenario['dam_base_elevation'], scenario['dam_base_elevation'] - 5]), 
                              'wse': np.array([scenario['dam_base_elevation'], scenario['dam_base_elevation'] - 5]), 
                              'y': np.array([0, 0]), 'v': np.array([0, 0]), 'fr': np.array([0, 0])},
                'normal_depth': 0,
                'critical_depth': 0,
                'hydraulic_jump': {'jump_possible': False, 'reason': 'No flow over dam'}
            }
        else:
            # Analyze this water level
            result = analyze_steady_state(scenario, level)
            
            # Print key results
            print(f"  Discharge: {result['discharge']:.2f} m³/s")
            
            jump = result['hydraulic_jump']
            if jump.get('jump_possible', False):
                print(f"  Hydraulic jump at {jump['location']:.2f} m downstream")
                print(f"  Jump type: {jump['jump_type']}")
        
        results_list.append(result)
    
    # Create animation if requested
    if save_animation:
        print("\nCreating flood animation...")
        anim, fig, ax = create_flood_animation(scenario, results_list)
        
        # Save the animation - try different methods
        try:
            # Check if ffmpeg is available
            try:
                from matplotlib.animation import FFMpegWriter
                print("Using ffmpeg for animation...")
                anim.save('flood_animation.mp4', writer='ffmpeg', fps=5, 
                         dpi=200, extra_args=['-vcodec', 'libx264'])
                print("Animation saved as 'flood_animation.mp4'")
            except (ImportError, ValueError):
                # If ffmpeg not available, try Pillow writer for GIF
                print("ffmpeg not found, using Pillow for GIF animation...")
                anim.save('flood_animation.gif', writer='pillow', fps=5, dpi=100)
                print("Animation saved as 'flood_animation.gif'")
        except Exception as e:
            print(f"Could not save animation: {e}")
            print("Saving individual frames instead...")
            
            # Save key frames as individual images
            for i, result in enumerate([results_list[0], results_list[len(results_list)//2], results_list[-1]]):
                fig, ax = plot_full_profile(scenario, result)
                plt.savefig(f'flood_stage_{i+1}.png', dpi=200, bbox_inches='tight')
                plt.close()
            print("Saved key frames as individual images.")
    
    return results_list

def plot_discharge_rating_curve(scenario, results_list):
    """
    Plot the discharge rating curve for the flood event.
    
    Parameters:
        scenario (dict): Scenario parameters and objects
        results_list (list): List of results for each water level
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Extract water levels and discharges
    water_levels = [r['upstream_level'] for r in results_list]
    discharges = [r['discharge'] for r in results_list]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot rating curve
    ax.plot(water_levels, discharges, 'bo-', linewidth=2, markersize=6)
    
    # Mark dam crest elevation
    dam_crest = scenario['dam_crest_elevation']
    ax.axvline(x=dam_crest, color='r', linestyle='--', 
              label=f'Dam Crest ({dam_crest:.2f} m)')
    
    # Add reference lines at specific depths above crest
    for h in [1, 2, 3]:
        level = dam_crest + h
        if min(water_levels) <= level <= max(water_levels):
            ax.axvline(x=level, color='k', linestyle=':', alpha=0.5,
                      label=f'{h} m Above Crest')
    
    # Add title and labels
    ax.set_title('Dam Discharge Rating Curve')
    ax.set_xlabel('Upstream Water Level (m)')
    ax.set_ylabel('Discharge (m³/s)')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    return fig, ax

def plot_jump_characteristics(results_list):
    """
    Plot hydraulic jump characteristics for different discharges.
    
    Parameters:
        results_list (list): List of results for each water level
        
    Returns:
        tuple: (fig, ax) The figure and axis objects
    """
    # Extract discharges and jump characteristics
    discharges = []
    locations = []
    froude_numbers = []
    energy_losses = []
    jump_types = []
    
    for result in results_list:
        discharge = result['discharge']
        jump = result['hydraulic_jump']
        
        if jump.get('jump_possible', False):
            discharges.append(discharge)
            locations.append(jump['location'])
            froude_numbers.append(jump['initial_froude'])
            energy_losses.append(jump['energy_loss'])
            jump_types.append(jump['jump_type'])
    
    # If no jumps or only one jump, return None
    if len(discharges) <= 1:
        print("Not enough hydraulic jumps detected for meaningful plots.")
        return None, None
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot jump location vs discharge
    axs[0, 0].plot(discharges, locations, 'bo-')
    axs[0, 0].set_xlabel('Discharge (m³/s)')
    axs[0, 0].set_ylabel('Jump Location (m downstream)')
    axs[0, 0].set_title('Hydraulic Jump Location')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot Froude number vs discharge
    axs[0, 1].plot(discharges, froude_numbers, 'ro-')
    axs[0, 1].set_xlabel('Discharge (m³/s)')
    axs[0, 1].set_ylabel('Initial Froude Number')
    axs[0, 1].set_title('Jump Intensity')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot energy loss vs discharge
    axs[1, 0].plot(discharges, energy_losses, 'go-')
    axs[1, 0].set_xlabel('Discharge (m³/s)')
    axs[1, 0].set_ylabel('Energy Loss (m)')
    axs[1, 0].set_title('Energy Dissipation')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot jump types (as categories)
    unique_types = list(set(jump_types))
    type_indices = [unique_types.index(t) for t in jump_types]
    
    axs[1, 1].plot(discharges, type_indices, 'mo-')
    axs[1, 1].set_xlabel('Discharge (m³/s)')
    axs[1, 1].set_ylabel('Jump Type')
    axs[1, 1].set_yticks(range(len(unique_types)))
    axs[1, 1].set_yticklabels(unique_types)
    axs[1, 1].set_title('Jump Classification')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axs

def analyze_peak_conditions(scenario):
    """
    Perform detailed analysis of peak flood conditions.
    
    Parameters:
        scenario (dict): Scenario parameters and objects
        
    Returns:
        dict: Results for peak conditions
    """
    # Get flood peak water level
    flood_level = scenario['flood_water_level']
    
    # Analyze peak conditions
    results = analyze_steady_state(scenario, flood_level)
    
    # Print detailed results
    print("\nPeak Flood Conditions")
    print("=" * 60)
    print(f"Water level: {flood_level:.2f} m")
    print(f"Head above crest: {results['head']:.2f} m")
    print(f"Discharge: {results['discharge']:.2f} m³/s")
    print(f"Velocity at crest: {results['velocity']:.2f} m/s")
    
    # Energy dissipation
    energy = results['energy_results']
    print("\nEnergy Dissipation:")
    print(f"  Energy loss: {energy['energy_loss']:.2f} m")
    print(f"  Dissipation ratio: {energy['dissipation_ratio']*100:.1f}%")
    print(f"  Power: {energy['power']/1000:.2f} kW")
    
    # Hydraulic jump
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        print("\nHydraulic Jump:")
        print(f"  Location: {jump['location']:.2f} m downstream")
        print(f"  Initial depth: {jump['initial_depth']:.2f} m")
        print(f"  Sequent depth: {jump['sequent_depth']:.2f} m")
        print(f"  Froude number: {jump['initial_froude']:.2f}")
        print(f"  Jump type: {jump['jump_type']}")
        print(f"  Energy loss in jump: {jump['energy_loss']:.2f} m")
        
        # Calculate forces if available
        if 'net_force' in jump:
            print(f"  Net force: {jump['net_force']:.2f} N/m")
            print(f"  Total force (per meter width): {jump['net_force'] * scenario['channel_width_at_dam']:.2f} N")
    
    # Create detailed visualization
    fig, ax = plot_full_profile(scenario, results)
    plt.savefig('peak_conditions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPeak conditions plotted and saved as 'peak_conditions.png'")
    
    return results

if __name__ == "__main__":
    # Create the default scenario
    scenario = create_scenario()
    
    # Simulate the flood event
    results_list = simulate_flood_event(scenario, num_levels=8)
    
    # Plot the discharge rating curve
    fig, ax = plot_discharge_rating_curve(scenario, results_list)
    plt.savefig('discharge_rating_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nDischarge rating curve saved as 'discharge_rating_curve.png'")
    
    # Plot hydraulic jump characteristics
    fig, ax = plot_jump_characteristics(results_list)
    if fig:
        plt.savefig('jump_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nJump characteristics saved as 'jump_characteristics.png'")
    
    # Analyze peak conditions
    peak_results = analyze_peak_conditions(scenario)