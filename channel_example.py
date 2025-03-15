"""
Example script demonstrating the use of the channel module.

This script shows how to create channel objects and perform
both normal flow and gradually varied flow calculations.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
# This is for demonstration only - in a real project, you'd install the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel import (
    RectangularChannel, TrapezoidalChannel, create_channel,
    normal_depth, critical_depth, critical_slope,
    calculate_normal_flow_profile, classify_channel_slope,
    classify_flow_profile, direct_step_method, backwater_calculation
)

def analyze_channel_flow():
    """Analyze flow in different channel cross-sections."""
    print("\nChannel Flow Analysis")
    print("-" * 50)
    
    # Parameters
    discharge = 20.0  # m³/s
    slope = 0.001     # m/m
    
    # Create different channel types
    rect_channel = RectangularChannel(bottom_width=5.0, roughness=0.015)
    trap_channel = TrapezoidalChannel(bottom_width=4.0, side_slope=2.0, roughness=0.015)
    
    # Calculate normal and critical depths
    print("\nRectangular Channel:")
    yn_rect = normal_depth(rect_channel, discharge, slope)
    yc_rect = critical_depth(rect_channel, discharge)
    
    print(f"  Normal depth: {yn_rect:.3f} m")
    print(f"  Critical depth: {yc_rect:.3f} m")
    print(f"  Channel slope classification: {classify_channel_slope(rect_channel, discharge, slope)}")
    
    print("\nTrapezoidal Channel:")
    yn_trap = normal_depth(trap_channel, discharge, slope)
    yc_trap = critical_depth(trap_channel, discharge)
    
    print(f"  Normal depth: {yn_trap:.3f} m")
    print(f"  Critical depth: {yc_trap:.3f} m")
    print(f"  Channel slope classification: {classify_channel_slope(trap_channel, discharge, slope)}")

def plot_water_surface_profiles():
    """Calculate and plot water surface profiles for different scenarios."""
    print("\nCalculating Water Surface Profiles")
    print("-" * 50)
    
    # Create a rectangular channel
    channel = RectangularChannel(bottom_width=5.0, roughness=0.015)
    
    # Flow parameters
    discharge = 20.0  # m³/s
    slope = 0.001     # m/m
    
    # Calculate normal and critical depths
    yn = normal_depth(channel, discharge, slope)
    yc = critical_depth(channel, discharge)
    
    print(f"Normal depth: {yn:.3f} m")
    print(f"Critical depth: {yc:.3f} m")
    
    # Calculate M1 profile (backwater curve from a downstream control)
    # Downstream depth > Normal depth
    downstream_depth = 1.5 * yn
    
    print(f"\nCalculating M1 profile from downstream depth = {downstream_depth:.3f} m")
    m1_profile = backwater_calculation(
        channel, discharge, slope, downstream_depth, 
        control_location='downstream', distance=1000, num_points=50
    )
    
    # Calculate M2 profile using direct step method
    # From normal depth to near critical depth
    print("\nCalculating M2 profile from normal to critical depth")
    m2_profile = direct_step_method(
        channel, discharge, slope, yn, 1.05*yc,
        max_distance=1000, distance_step=10
    )
    
    # Visualize the profiles
    plt.figure(figsize=(12, 8))
    
    # Plot M1 profile
    plt.subplot(2, 1, 1)
    plt.plot(m1_profile['x'], m1_profile['z'], 'k-', label='Channel bed')
    plt.plot(m1_profile['x'], m1_profile['wse'], 'b-', label='Water surface (M1)')
    plt.axhline(y=m1_profile['z'][0] + yn, color='g', linestyle='--', label=f'Normal depth ({yn:.3f} m)')
    plt.axhline(y=m1_profile['z'][0] + yc, color='r', linestyle='--', label=f'Critical depth ({yc:.3f} m)')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('M1 Water Surface Profile (Mild Slope, Backwater Curve)')
    plt.grid(True)
    plt.legend()
    
    # Plot M2 profile
    plt.subplot(2, 1, 2)
    plt.plot(m2_profile['x'], m2_profile['z'], 'k-', label='Channel bed')
    plt.plot(m2_profile['x'], m2_profile['wse'], 'b-', label='Water surface (M2)')
    plt.axhline(y=m2_profile['z'][0] + yn, color='g', linestyle='--', label=f'Normal depth ({yn:.3f} m)')
    plt.axhline(y=m2_profile['z'][0] + yc, color='r', linestyle='--', label=f'Critical depth ({yc:.3f} m)')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('M2 Water Surface Profile (Mild Slope, Drawdown Curve)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('water_surface_profiles.png')
    plt.close()
    
    print("\nPlot saved as 'water_surface_profiles.png'")

def dam_backwater_effect():
    """Simulate the backwater effect of a dam."""
    print("\nDam Backwater Effect Simulation")
    print("-" * 50)
    
    # Create a rectangular channel
    channel = RectangularChannel(bottom_width=5.0, roughness=0.015)
    
    # Flow parameters
    discharge = 20.0  # m³/s
    slope = 0.001     # m/m
    
    # Normal depth
    yn = normal_depth(channel, discharge, slope)
    print(f"Normal depth: {yn:.3f} m")
    
    # Dam height (above channel bed)
    dam_height = 2.5  # m
    
    # Simulate backwater curve from dam
    backwater = backwater_calculation(
        channel, discharge, slope, dam_height, 
        control_location='downstream', distance=5000, num_points=100
    )
    
    # Find where the water surface approaches normal depth
    # Use a threshold of 5% difference
    threshold = 0.05 * yn
    normal_indices = np.where(np.abs(backwater['y'] - yn) < threshold)[0]
    
    if len(normal_indices) > 0:
        # Take the first point where normal depth is reached
        normal_depth_index = normal_indices[0]
        affected_distance = backwater['x'][normal_depth_index]
        print(f"Dam height: {dam_height:.2f} m")
        print(f"Backwater effect extends approximately {affected_distance:.2f} m upstream.")
    else:
        # Backwater effect extends beyond calculation range
        affected_distance = backwater['x'][-1]
        print(f"Dam height: {dam_height:.2f} m")
        print(f"Backwater effect extends beyond {affected_distance:.2f} m upstream.")
    
    # Plot backwater curve
    plt.figure(figsize=(12, 6))
    
    plt.plot(backwater['x'], backwater['z'], 'k-', label='Channel bed')
    plt.plot(backwater['x'], backwater['wse'], 'b-', label='Water surface')
    plt.axhline(y=backwater['z'][0] + yn, color='g', linestyle='--', label=f'Normal depth ({yn:.3f} m)')
    
    # Indicate the dam
    plt.plot([0, 0], [backwater['z'][0], backwater['z'][0] + dam_height], 'r-', linewidth=2, label='Dam')
    
    # Mark the extent of backwater effect
    if len(normal_indices) > 0:
        plt.axvline(x=affected_distance, color='m', linestyle=':', 
                   label=f'Normal depth reached ({affected_distance:.2f} m)')
    
    plt.xlabel('Distance Upstream from Dam (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Dam Backwater Effect')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('dam_backwater.png')
    plt.close()
    
    print("Plot saved as 'dam_backwater.png'")

def flow_regime_comparison():
    """Compare flow regimes for different slopes."""
    print("\nFlow Regime Comparison")
    print("-" * 50)
    
    # Create a rectangular channel
    width = 5.0
    roughness = 0.015
    
    # Flow parameters
    discharge = 20.0  # m³/s
    
    # Different slopes
    mild_slope = 0.001
    steep_slope = 0.01
    
    # Create channels
    channel_mild = RectangularChannel(bottom_width=width, roughness=roughness)
    channel_steep = RectangularChannel(bottom_width=width, roughness=roughness)
    
    # Calculate normal and critical depths
    yn_mild = normal_depth(channel_mild, discharge, mild_slope)
    yc_mild = critical_depth(channel_mild, discharge)
    
    yn_steep = normal_depth(channel_steep, discharge, steep_slope)
    yc_steep = critical_depth(channel_steep, discharge)
    
    print(f"Mild Slope (S = {mild_slope}):")
    print(f"  Normal depth: {yn_mild:.3f} m")
    print(f"  Critical depth: {yc_mild:.3f} m")
    print(f"  Classification: {classify_channel_slope(channel_mild, discharge, mild_slope)}")
    print(f"  Flow regime: {'Subcritical' if yn_mild > yc_mild else 'Supercritical'}")
    
    print(f"\nSteep Slope (S = {steep_slope}):")
    print(f"  Normal depth: {yn_steep:.3f} m")
    print(f"  Critical depth: {yc_steep:.3f} m")
    print(f"  Classification: {classify_channel_slope(channel_steep, discharge, steep_slope)}")
    print(f"  Flow regime: {'Subcritical' if yn_steep > yc_steep else 'Supercritical'}")
    
    # Calculate normal flow profiles
    profile_mild = calculate_normal_flow_profile(
        channel_mild, discharge, mild_slope, (0, 1000), 100
    )
    
    profile_steep = calculate_normal_flow_profile(
        channel_steep, discharge, steep_slope, (0, 1000), 100
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Mild slope profile
    plt.subplot(2, 1, 1)
    plt.plot(profile_mild['x'], profile_mild['z'], 'k-', label='Channel bed')
    plt.plot(profile_mild['x'], profile_mild['wse'], 'b-', label='Water surface')
    
    # Adjust origin for better visualization
    z_offset = profile_mild['z'][0]
    plt.plot(profile_mild['x'], profile_mild['z'] - z_offset, 'k-')
    plt.plot(profile_mild['x'], profile_mild['wse'] - z_offset, 'b-')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Relative Elevation (m)')
    plt.title(f'Mild Slope (S = {mild_slope}) - Subcritical Flow')
    plt.text(50, yn_mild*1.2, f'Fr = {profile_mild["fr"][0]:.2f}')
    plt.grid(True)
    plt.legend()
    
    # Steep slope profile
    plt.subplot(2, 1, 2)
    plt.plot(profile_steep['x'], profile_steep['z'], 'k-', label='Channel bed')
    plt.plot(profile_steep['x'], profile_steep['wse'], 'b-', label='Water surface')
    
    # Adjust origin for better visualization
    z_offset = profile_steep['z'][0]
    plt.plot(profile_steep['x'], profile_steep['z'] - z_offset, 'k-')
    plt.plot(profile_steep['x'], profile_steep['wse'] - z_offset, 'b-')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Relative Elevation (m)')
    plt.title(f'Steep Slope (S = {steep_slope}) - Supercritical Flow')
    plt.text(50, yn_steep*1.2, f'Fr = {profile_steep["fr"][0]:.2f}')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('flow_regime_comparison.png')
    plt.close()
    
    print("\nPlot saved as 'flow_regime_comparison.png'")

if __name__ == "__main__":
    # Run example analyses
    analyze_channel_flow()
    plot_water_surface_profiles()
    dam_backwater_effect()
    flow_regime_comparison()
    
    print("\nExamples completed successfully.")