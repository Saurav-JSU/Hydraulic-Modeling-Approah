"""
Example script demonstrating the use of the dam module.

This script shows how to create and analyze different dam types,
calculate flow characteristics, and visualize results.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
# This is for demonstration only - in a real project, you'd install the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel import RectangularChannel, normal_depth
from src.dam import (
    BroadCrestedWeir, SharpCrestedWeir, OgeeWeir, create_dam,
    calculate_flow_over_dam, create_rating_curve,
    calculate_backwater_curve, estimate_water_surface_profile,
    determine_control_point, hydraulic_jump_location
)

def analyze_different_dam_types():
    """Compare flow characteristics of different dam types."""
    print("\nComparing Different Dam Types")
    print("-" * 50)
    
    # Create different dam types at the same height
    crest_elevation = 10.0  # m
    height = 5.0  # m
    
    broad_crested = BroadCrestedWeir(height, crest_elevation, crest_width=2.0)
    sharp_crested = SharpCrestedWeir(height, crest_elevation)
    ogee = OgeeWeir(height, crest_elevation, design_head=1.5)
    
    # Flow conditions
    upstream_elevation = 11.5  # m
    downstream_elevation = 5.5  # m
    width = 10.0  # m
    
    # Calculate flow for each dam type
    broad_flow = calculate_flow_over_dam(broad_crested, upstream_elevation, downstream_elevation, width)
    sharp_flow = calculate_flow_over_dam(sharp_crested, upstream_elevation, downstream_elevation, width)
    ogee_flow = calculate_flow_over_dam(ogee, upstream_elevation, downstream_elevation, width)
    
    # Print results
    print("\nBroad-Crested Weir:")
    print(f"  Discharge: {broad_flow['discharge']:.2f} m³/s")
    print(f"  Head: {broad_flow['head']:.2f} m")
    print(f"  Overflow velocity: {broad_flow['velocity']:.2f} m/s")
    print(f"  Flow type: {broad_flow['flow_type']}")
    
    print("\nSharp-Crested Weir:")
    print(f"  Discharge: {sharp_flow['discharge']:.2f} m³/s")
    print(f"  Head: {sharp_flow['head']:.2f} m")
    print(f"  Overflow velocity: {sharp_flow['velocity']:.2f} m/s")
    print(f"  Flow type: {sharp_flow['flow_type']}")
    
    print("\nOgee Spillway:")
    print(f"  Discharge: {ogee_flow['discharge']:.2f} m³/s")
    print(f"  Head: {ogee_flow['head']:.2f} m")
    print(f"  Overflow velocity: {ogee_flow['velocity']:.2f} m/s")
    print(f"  Flow type: {ogee_flow['flow_type']}")
    
    # Create and plot rating curves
    elevation_range = (crest_elevation, crest_elevation + 3.0)
    
    broad_rating = create_rating_curve(broad_crested, width, elevation_range, 20)
    sharp_rating = create_rating_curve(sharp_crested, width, elevation_range, 20)
    ogee_rating = create_rating_curve(ogee, width, elevation_range, 20)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(broad_rating['discharges'], broad_rating['elevations'], 'b-', label='Broad-Crested Weir')
    plt.plot(sharp_rating['discharges'], sharp_rating['elevations'], 'r-', label='Sharp-Crested Weir')
    plt.plot(ogee_rating['discharges'], ogee_rating['elevations'], 'g-', label='Ogee Spillway')
    
    plt.xlabel('Discharge (m³/s)')
    plt.ylabel('Water Surface Elevation (m)')
    plt.title('Rating Curves for Different Dam Types')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('dam_rating_curves.png')
    plt.close()
    
    print("\nRating curves saved as 'dam_rating_curves.png'")
    
    # Plot dam profiles
    stations = np.linspace(-5, 10, 100)
    
    broad_profile = broad_crested.get_profile(stations)
    sharp_profile = sharp_crested.get_profile(stations)
    ogee_profile = ogee.get_profile(stations)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(broad_profile['x'], broad_profile['z'], 'b-', label='Broad-Crested Weir')
    plt.plot(sharp_profile['x'], sharp_profile['z'], 'r-', label='Sharp-Crested Weir')
    plt.plot(ogee_profile['x'], ogee_profile['z'], 'g-', label='Ogee Spillway')
    
    # Add water levels
    plt.axhline(y=upstream_elevation, color='c', linestyle='--', label='Upstream Water Level')
    plt.axhline(y=downstream_elevation, color='m', linestyle='--', label='Downstream Water Level')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Dam Profiles')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('dam_profiles.png')
    plt.close()
    
    print("Dam profiles saved as 'dam_profiles.png'")

def backwater_analysis():
    """Analyze backwater effects of a dam."""
    print("\nBackwater Analysis")
    print("-" * 50)
    
    # Create a channel
    channel = RectangularChannel(bottom_width=15.0, roughness=0.015)
    
    # Channel characteristics
    slope = 0.001  # m/m
    discharge = 50.0  # m³/s
    
    # Create a dam
    dam_height = 4.0  # m
    crest_elevation = 10.0  # m
    dam = BroadCrestedWeir(dam_height, crest_elevation, crest_width=2.0)
    
    # Calculate normal depth
    yn = normal_depth(channel, discharge, slope)
    print(f"Normal depth: {yn:.3f} m")
    
    # Calculate backwater curve
    backwater = calculate_backwater_curve(dam, channel, discharge, slope, 3000, 100)
    
    # Calculate control point
    control = determine_control_point(dam, channel, discharge, slope, channel.bottom_width)
    
    print(f"\nUpstream depth at dam: {control['upstream_depth']:.3f} m")
    print(f"Control type: {control['control_type']}")
    print(f"Control location: {control['control_location']}")
    
    # Find where the water surface approaches normal depth
    wse_diff = np.abs(backwater['y'] - yn)
    threshold = 0.05 * yn  # 5% difference
    normal_indices = np.where(wse_diff < threshold)[0]
    
    if len(normal_indices) > 0:
        affected_distance = backwater['x'][normal_indices[0]]
        print(f"Backwater effect extends approximately {affected_distance:.2f} m upstream.")
    else:
        print(f"Backwater effect extends beyond the calculation range ({backwater['x'][-1]:.2f} m).")
    
    # Plot backwater curve
    plt.figure(figsize=(12, 6))
    
    plt.plot(backwater['x'], backwater['z'], 'k-', label='Channel bed')
    plt.plot(backwater['x'], backwater['wse'], 'b-', label='Water surface')
    plt.axhline(y=backwater['z'][0] + yn, color='g', linestyle='--', label=f'Normal depth ({yn:.3f} m)')
    
    # Indicate the dam
    plt.plot([0, 0], [dam.base_elevation, dam.crest_elevation], 'r-', linewidth=2, label='Dam')
    
    # Mark the extent of backwater effect if found
    if len(normal_indices) > 0:
        plt.axvline(x=affected_distance, color='m', linestyle=':', 
                   label=f'Normal depth reached ({affected_distance:.2f} m)')
    
    plt.xlabel('Distance Upstream from Dam (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Dam Backwater Effect')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('dam_backwater_analysis.png')
    plt.close()
    
    print("\nBackwater analysis plot saved as 'dam_backwater_analysis.png'")

def complete_water_surface_profile():
    """Calculate and plot complete water surface profile through a dam."""
    print("\nComplete Water Surface Profile")
    print("-" * 50)
    
    # Create a channel
    channel = RectangularChannel(bottom_width=15.0, roughness=0.015)
    
    # Channel characteristics
    slope = 0.001  # m/m
    discharge = 50.0  # m³/s
    
    # Create a dam (ogee spillway for this example)
    dam_height = 4.0  # m
    crest_elevation = 102.0  # m
    dam = OgeeWeir(dam_height, crest_elevation, design_head=1.5)
    
    # Calculate complete profile
    profile = estimate_water_surface_profile(
        dam, channel, discharge, slope, 2000, 1000, None, 150
    )
    
    # Calculate normal depth for reference
    yn = normal_depth(channel, discharge, slope)
    print(f"Normal depth: {yn:.3f} m")
    
    # Check for hydraulic jump
    jump = hydraulic_jump_location(dam, discharge, channel.bottom_width, slope, channel.roughness, yn)
    
    if jump.get('jump_possible', False):
        print("\nHydraulic jump possible:")
        print(f"  Location: {jump['location']:.2f} m downstream of dam")
        print(f"  Initial depth: {jump['initial_depth']:.3f} m")
        print(f"  Sequent depth: {jump['sequent_depth']:.3f} m")
        print(f"  Jump type: {jump['jump_type']}")
    else:
        reason = jump.get('reason', 'Unknown reason')
        print(f"\nHydraulic jump not possible: {reason}")
    
    # Plot complete profile
    plt.figure(figsize=(15, 8))
    
    # Plot bed profile
    plt.plot(profile['x'], profile['z'], 'k-', label='Channel bed')
    
    # Plot water surface
    plt.plot(profile['x'], profile['wse'], 'b-', label='Water surface')
    
    # Add horizontal line for normal depth upstream and downstream
    plt.axhline(y=dam.base_elevation + yn, color='g', linestyle='--', label=f'Normal depth ({yn:.3f} m)')
    
    # Add vertical line for dam
    plt.axvline(x=0, color='r', linestyle=':', label='Dam location')
    
    # Mark hydraulic jump if present
    if jump.get('jump_possible', False):
        jump_location = jump['location']
        plt.axvline(x=jump_location, color='m', linestyle=':', 
                   label=f'Hydraulic jump ({jump_location:.2f} m)')
    
    plt.xlabel('Distance (m) - Upstream (negative), Downstream (positive)')
    plt.ylabel('Elevation (m)')
    plt.title('Complete Water Surface Profile Through Dam')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis limits for better visualization
    plt.xlim(-1000, 500)
    
    plt.savefig('complete_water_surface_profile.png')
    plt.close()
    
    print("\nComplete water surface profile saved as 'complete_water_surface_profile.png'")

def dam_flow_comparison():
    """Compare flow over dam at different upstream water levels."""
    print("\nDam Flow Comparison")
    print("-" * 50)
    
    # Create a dam (broad-crested weir for this example)
    height = 5.0  # m
    crest_elevation = 100.0  # m
    dam = BroadCrestedWeir(height, crest_elevation, crest_width=2.0)
    
    # Downstream conditions
    downstream_elevation = 96.0  # m
    width = 10.0  # m
    
    # Different upstream water levels
    upstream_elevations = [100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
    
    # Calculate flow for each upstream elevation
    discharges = []
    heads = []
    velocities = []
    
    for elev in upstream_elevations:
        flow = calculate_flow_over_dam(dam, elev, downstream_elevation, width)
        discharges.append(flow['discharge'])
        heads.append(flow['head'])
        velocities.append(flow['velocity'])
    
    # Print results
    print("\nFlow comparison at different upstream water levels:")
    print("    Elevation (m) | Head (m) | Discharge (m³/s) | Velocity (m/s)")
    print("   " + "-" * 60)
    
    for i, elev in enumerate(upstream_elevations):
        print(f"      {elev:6.2f}     | {heads[i]:6.2f}  |     {discharges[i]:8.2f}    |    {velocities[i]:6.2f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot discharge vs. head
    ax1.plot(heads, discharges, 'bo-')
    ax1.set_xlabel('Head (m)')
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title('Discharge vs. Head')
    ax1.grid(True)
    
    # Plot velocity vs. head
    ax2.plot(heads, velocities, 'ro-')
    ax2.set_xlabel('Head (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs. Head')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dam_flow_comparison.png')
    plt.close()
    
    print("\nFlow comparison plot saved as 'dam_flow_comparison.png'")

if __name__ == "__main__":
    # Run example analyses
    analyze_different_dam_types()
    backwater_analysis()
    complete_water_surface_profile()
    dam_flow_comparison()
    
    print("\nExamples completed successfully.")