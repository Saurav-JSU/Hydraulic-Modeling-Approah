"""
Steady-state analysis of the dam and channel system.

This script performs steady-state hydraulic analysis of the dam and
channel system for different upstream water levels.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dam import calculate_flow_over_dam, hydraulic_jump_location
from src.dam import estimate_tailwater_profile, calculate_energy_dissipation
from src.channel import normal_depth, critical_depth
from src.hydraulics.energy import specific_force

from scenario_setup import create_scenario

def analyze_steady_state(scenario, upstream_level=None):
    """
    Perform steady-state analysis for a given upstream water level.
    
    Parameters:
        scenario (dict): Scenario parameters and objects
        upstream_level (float, optional): Upstream water level (m).
            If None, uses the flood peak level from the scenario.
    
    Returns:
        dict: Results of the steady-state analysis
    """
    # Get scenario parameters
    dam = scenario['dam']
    downstream_channel = scenario['downstream_channel']
    channel_width = scenario['channel_width_at_dam']
    downstream_slope = scenario['downstream_slope']
    
    # Use provided upstream level or default to flood peak
    if upstream_level is None:
        upstream_level = scenario['flood_water_level']
    
    # Check if water level is above crest
    if upstream_level <= scenario['dam_crest_elevation']:
        # No flow over dam - return minimal results
        return {
            'upstream_level': upstream_level,
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
    
    # Calculate flow over dam
    # Set minimum downstream water level to avoid division by zero
    minimum_depth = 0.1  # m, minimum depth to avoid division by zero
    downstream_level = scenario['dam_base_elevation'] + minimum_depth  # Small depth for initial calculation
    
    flow_results = calculate_flow_over_dam(
        dam, upstream_level, downstream_level, channel_width
    )
    
    discharge = flow_results['discharge']
    head = flow_results['head']
    velocity = flow_results['velocity']
    
    # Calculate downstream normal depth
    yn = normal_depth(downstream_channel, discharge, downstream_slope)
    yc = critical_depth(downstream_channel, discharge)
    
    # Calculate tailwater profile
    tailwater = estimate_tailwater_profile(
        dam, discharge, channel_width, downstream_slope,
        downstream_channel.roughness, 500, 100
    )
    
    # Update downstream level to use the first calculated depth from tailwater
    # This provides a more realistic value for energy calculations
    if len(tailwater['y']) > 0:
        realistic_downstream_depth = tailwater['y'][0]
        downstream_level = scenario['dam_base_elevation'] + max(realistic_downstream_depth, minimum_depth)
    
    # Calculate energy dissipation with the updated downstream level
    energy_results = calculate_energy_dissipation(
        dam, upstream_level, downstream_level, channel_width
    )
    
    # Check for hydraulic jump
    # Use depth at 20m downstream as reference tailwater depth
    index_20m = min(len(tailwater['x']) - 1, np.argmin(np.abs(tailwater['x'] - 20)))
    tailwater_depth = tailwater['y'][index_20m]
    
    jump = hydraulic_jump_location(
        dam, discharge, channel_width, downstream_slope,
        downstream_channel.roughness, tailwater_depth
    )
    
    # Calculate forces if jump exists
    if jump.get('jump_possible', False):
        # Before jump
        force_before = specific_force(
            jump['initial_depth'], discharge, channel_width
        )
        
        # After jump
        force_after = specific_force(
            jump['sequent_depth'], discharge, channel_width
        )
        
        # Net force
        net_force = force_after - force_before
        jump['net_force'] = net_force
    
    # Return all results
    return {
        'upstream_level': upstream_level,
        'discharge': discharge,
        'head': head,
        'velocity': velocity,
        'energy_results': energy_results,
        'tailwater': tailwater,
        'normal_depth': yn,
        'critical_depth': yc,
        'hydraulic_jump': jump
    }

def print_analysis_results(results):
    """Print a summary of the analysis results."""
    print("\nSteady-State Analysis Results")
    print("=" * 60)
    
    print(f"\nUpstream Water Level: {results['upstream_level']:.2f} m")
    print(f"Discharge: {results['discharge']:.2f} m³/s")
    print(f"Head Over Crest: {results['head']:.2f} m")
    print(f"Velocity at Crest: {results['velocity']:.2f} m/s")
    
    energy = results['energy_results']
    print(f"\nEnergy Dissipation:")
    print(f"  Energy Loss: {energy['energy_loss']:.2f} m")
    print(f"  Dissipation Ratio: {energy['dissipation_ratio']*100:.1f}%")
    print(f"  Power Dissipated: {energy['power']/1000:.2f} kW")
    
    print(f"\nDownstream Channel:")
    print(f"  Normal Depth: {results['normal_depth']:.2f} m")
    print(f"  Critical Depth: {results['critical_depth']:.2f} m")
    
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        print(f"\nHydraulic Jump:")
        print(f"  Location: {jump['location']:.2f} m downstream")
        print(f"  Initial Depth: {jump['initial_depth']:.2f} m")
        print(f"  Sequent Depth: {jump['sequent_depth']:.2f} m")
        print(f"  Initial Froude Number: {jump['initial_froude']:.2f}")
        print(f"  Depth Ratio: {jump['depth_ratio']:.2f}")
        print(f"  Jump Type: {jump['jump_type']}")
        print(f"  Energy Loss in Jump: {jump['energy_loss']:.2f} m")
        print(f"  Net Force at Jump: {jump.get('net_force', 0):.2f} N/m")
    else:
        reason = jump.get('reason', 'Unknown reason')
        print(f"\nNo Hydraulic Jump: {reason}")
    
    print("=" * 60)

def plot_results(scenario, results):
    """Plot the water surface profile and other results."""
    # Create a new figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get data
    dam = scenario['dam']
    tailwater = results['tailwater']
    
    # Plot dam profile (simplified for clarity)
    dam_stations = np.linspace(-10, 10, 50)
    dam_profile = dam.get_profile(dam_stations)
    ax.plot(dam_stations, dam_profile['z'], 'k-', linewidth=2, label='Dam')
    
    # Plot water surface profile
    # Account for dam location in x-coordinates
    x_adjusted = tailwater['x']
    z_adjusted = tailwater['z']
    wse_adjusted = tailwater['wse']
    
    # Plot channel bed and water surface
    ax.plot(x_adjusted, z_adjusted, 'k-', label='Channel Bed')
    ax.plot(x_adjusted, wse_adjusted, 'b-', label='Water Surface')
    
    # Plot upstream water level
    ax.axhline(y=results['upstream_level'], 
              xmin=0, xmax=0.45, 
              color='b', linestyle='-')
    
    # Mark normal and critical depths
    yn = results['normal_depth']
    yc = results['critical_depth']
    
    ax.axhline(y=z_adjusted[0] + yn, color='g', linestyle='--', 
              label=f'Normal Depth ({yn:.2f} m)')
    ax.axhline(y=z_adjusted[0] + yc, color='r', linestyle=':', 
              label=f'Critical Depth ({yc:.2f} m)')
    
    # Mark hydraulic jump if present
    jump = results['hydraulic_jump']
    if jump.get('jump_possible', False):
        jump_loc = jump['location']
        
        # Find the closest x-coordinate to the jump location
        jump_index = np.argmin(np.abs(x_adjusted - jump_loc))
        jump_z = z_adjusted[jump_index]
        
        # Plot the jump
        ax.axvline(x=jump_loc, color='m', linestyle='-.',
                  label=f'Hydraulic Jump ({jump_loc:.2f} m)')
        
        # Add annotation for jump depths
        ax.annotate(f"y1={jump['initial_depth']:.2f}m\ny2={jump['sequent_depth']:.2f}m",
                   xy=(jump_loc, jump_z + jump['initial_depth']),
                   xytext=(jump_loc+30, jump_z + jump['sequent_depth']),
                   arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Set labels and title
    ax.set_xlabel('Distance from Dam (m)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Water Surface Profile with Dam and Hydraulic Jump')
    
    # Add discharge and other info to the title
    ax.set_title(f'Water Surface Profile for Q = {results["discharge"]:.2f} m³/s\n'
                f'Head = {results["head"]:.2f} m, Velocity = {results["velocity"]:.2f} m/s')
    
    # Set axis limits
    ax.set_xlim(-20, 200)  # Adjust as needed
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True)
    
    # Return the figure and axis for further customization
    return fig, ax

if __name__ == "__main__":
    # Create the default scenario
    scenario = create_scenario()
    
    # Run the steady-state analysis
    results = analyze_steady_state(scenario)
    
    # Print the results
    print_analysis_results(results)
    
    # Plot the results
    fig, ax = plot_results(scenario, results)
    
    # Save the figure
    plt.savefig('steady_state_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResults plotted and saved as 'steady_state_analysis.png'")