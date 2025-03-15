"""
Simple example demonstrating the use of the hydraulics module.

This script shows how to use the core hydraulic functions to analyze
a basic rectangular channel with steady, uniform flow.
"""

import sys
import os
import math

# Add the project root directory to the Python path
# This is for demonstration only - in a real project, you'd install the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hydraulics import basic, manning, energy

def analyze_rectangular_channel(discharge, width, slope, roughness):
    """
    Analyze flow in a rectangular channel.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient
    """
    print(f"\nAnalyzing rectangular channel with:")
    print(f"  Discharge: {discharge} m³/s")
    print(f"  Width: {width} m")
    print(f"  Slope: {slope} m/m")
    print(f"  Manning's n: {roughness}")
    print("\n" + "-" * 50)
    
    # Calculate normal depth
    yn = manning.normal_depth(discharge, width, slope, roughness)
    print(f"\nNormal depth: {yn:.3f} m")
    
    # Calculate critical depth
    yc = basic.critical_depth(discharge, width)
    print(f"Critical depth: {yc:.3f} m")
    
    # Calculate section properties at normal depth
    props = basic.section_properties(yn, width)
    print(f"\nSection properties at normal depth:")
    print(f"  Area: {props['area']:.3f} m²")
    print(f"  Wetted perimeter: {props['wetted_perimeter']:.3f} m")
    print(f"  Hydraulic radius: {props['hydraulic_radius']:.3f} m")
    
    # Calculate velocity at normal depth
    velocity = discharge / props['area']
    print(f"\nVelocity at normal depth: {velocity:.3f} m/s")
    
    # Calculate Froude number
    fr = basic.froude_number(velocity, yn)
    print(f"Froude number: {fr:.3f}")
    print(f"Flow regime: {basic.flow_classification(fr)}")
    
    # Calculate specific energy
    energy_value = energy.specific_energy(yn, velocity)
    print(f"\nSpecific energy: {energy_value:.3f} m")
    
    # Calculate minimum specific energy
    min_energy = energy.critical_specific_energy(discharge, width)
    print(f"Minimum specific energy: {min_energy:.3f} m")
    
    # Calculate shear stress
    tau = manning.shear_stress(props['hydraulic_radius'], slope)
    print(f"\nBoundary shear stress: {tau:.3f} N/m²")
    
    print("\n" + "-" * 50)

def compare_channel_conditions(discharge_values, width, slope, roughness):
    """
    Compare channel conditions for different discharge values.
    
    Parameters:
        discharge_values (list): List of flow rates to analyze (m³/s)
        width (float): Channel width (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient
    """
    print("\nComparison of channel conditions for different discharges:")
    print("\n" + "-" * 80)
    print(f"{'Discharge (m³/s)':^15} | {'Normal Depth (m)':^15} | {'Velocity (m/s)':^15} | {'Froude Number':^15} | {'Flow Regime':^15}")
    print("-" * 80)
    
    for q in discharge_values:
        # Calculate normal depth
        yn = manning.normal_depth(q, width, slope, roughness)
        
        # Calculate section properties
        props = basic.section_properties(yn, width)
        
        # Calculate velocity
        velocity = q / props['area']
        
        # Calculate Froude number
        fr = basic.froude_number(velocity, yn)
        
        # Determine flow regime
        regime = basic.flow_classification(fr)
        
        print(f"{q:^15.2f} | {yn:^15.3f} | {velocity:^15.3f} | {fr:^15.3f} | {regime:^15}")
    
    print("-" * 80)

def hydraulic_jump_analysis(discharge, width, slope, roughness, upstream_depth=None):
    """
    Analyze hydraulic jump in a rectangular channel.
    
    Parameters:
        discharge (float): Flow rate (m³/s)
        width (float): Channel width (m)
        slope (float): Channel bed slope (m/m)
        roughness (float): Manning's roughness coefficient
        upstream_depth (float, optional): Upstream depth before jump.
            If None, uses 0.8 * critical depth.
    """
    print("\nHydraulic Jump Analysis:")
    print("-" * 50)
    
    # Calculate critical depth
    yc = basic.critical_depth(discharge, width)
    print(f"Critical depth: {yc:.3f} m")
    
    # Calculate normal depth
    yn = manning.normal_depth(discharge, width, slope, roughness)
    print(f"Normal depth: {yn:.3f} m")
    
    # Check if hydraulic jump is possible
    if yn > yc:
        print("\nHydraulic jump is not possible in this channel.")
        print("Normal depth > Critical depth (S < Sc, mild slope)")
        return
    
    # Set upstream depth if not provided
    if upstream_depth is None:
        y1 = 0.8 * yc  # Arbitrary supercritical depth
    else:
        y1 = upstream_depth
    
    # Calculate upstream velocity and Froude number
    v1 = discharge / (width * y1)
    fr1 = basic.froude_number(v1, y1)
    
    if fr1 <= 1:
        print(f"\nUpstream flow is not supercritical (Fr = {fr1:.3f}).")
        print("Hydraulic jump will not occur.")
        return
    
    # Calculate sequent depth (downstream depth)
    y2 = energy.sequent_depth(y1, fr1)
    
    # Calculate downstream velocity and Froude number
    v2 = discharge / (width * y2)
    fr2 = basic.froude_number(v2, y2)
    
    # Calculate energy loss in the jump
    E1 = energy.specific_energy(y1, v1)
    E2 = energy.specific_energy(y2, v2)
    energy_loss = E1 - E2
    
    # Print results
    print(f"\nUpstream conditions (before jump):")
    print(f"  Depth: {y1:.3f} m")
    print(f"  Velocity: {v1:.3f} m/s")
    print(f"  Froude number: {fr1:.3f}")
    print(f"  Specific energy: {E1:.3f} m")
    
    print(f"\nDownstream conditions (after jump):")
    print(f"  Depth: {y2:.3f} m")
    print(f"  Velocity: {v2:.3f} m/s")
    print(f"  Froude number: {fr2:.3f}")
    print(f"  Specific energy: {E2:.3f} m")
    
    print(f"\nJump characteristics:")
    print(f"  Depth ratio (y2/y1): {y2/y1:.3f}")
    print(f"  Energy loss: {energy_loss:.3f} m ({energy_loss/E1*100:.1f}% of upstream energy)")
    
    # Classify jump based on Froude number
    if fr1 < 1.7:
        jump_type = "Undular jump"
    elif fr1 < 2.5:
        jump_type = "Weak jump"
    elif fr1 < 4.5:
        jump_type = "Oscillating jump"
    elif fr1 < 9.0:
        jump_type = "Steady jump"
    else:
        jump_type = "Strong jump"
    
    print(f"  Jump classification: {jump_type}")

if __name__ == "__main__":
    # Example parameters
    width = 5.0  # Channel width (m)
    slope = 0.001  # Channel bed slope (m/m)
    roughness = 0.015  # Manning's roughness coefficient
    
    # Analyze a single discharge
    discharge = 10.0  # Flow rate (m³/s)
    analyze_rectangular_channel(discharge, width, slope, roughness)
    
    # Compare multiple discharge values
    discharge_values = [5.0, 10.0, 15.0, 20.0, 25.0]
    compare_channel_conditions(discharge_values, width, slope, roughness)
    
    # For hydraulic jump, use steeper slope
    steep_slope = 0.01
    hydraulic_jump_analysis(15.0, width, steep_slope, roughness)