"""
Setup script for the dam overflow scenario with a flood event.

This script creates and configures the dam and channel objects
for analyzing a flood scenario with an ogee spillway dam.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.channel import TrapezoidalChannel
from src.dam import OgeeWeir

def create_scenario(
    dam_height=10.0,                # Dam height (m)
    dam_crest_elevation=100.0,      # Dam crest elevation (m)
    design_head=2.0,                # Design head for ogee (m)
    channel_bottom_width=5.0,       # Channel bottom width (m)
    channel_side_slope=1.5,         # Channel side slope (H:V)
    channel_roughness=0.015,        # Manning's roughness
    channel_slope=0.001,            # Channel bed slope (m/m)
    downstream_slope=0.005,         # Downstream channel slope (m/m)
    initial_water_level=95.0,       # Initial upstream water level (m)
    flood_water_level=102.0,        # Flood peak water level (m)
    channel_width_at_dam=20.0       # Width at dam face (m)
):
    """
    Create and configure the dam and channel objects for the flood scenario.
    
    Returns:
        dict: Dictionary containing all scenario parameters and objects
    """
    # Create the ogee spillway dam
    dam = OgeeWeir(
        height=dam_height,
        crest_elevation=dam_crest_elevation,
        design_head=design_head
    )
    
    # Create the upstream and downstream channels
    upstream_channel = TrapezoidalChannel(
        bottom_width=channel_bottom_width,
        side_slope=channel_side_slope,
        roughness=channel_roughness
    )
    
    downstream_channel = TrapezoidalChannel(
        bottom_width=channel_bottom_width,
        side_slope=channel_side_slope,
        roughness=channel_roughness
    )
    
    # Calculate derived parameters
    dam_base_elevation = dam_crest_elevation - dam_height
    
    # Channel length parameters
    upstream_length = 2000.0  # m
    downstream_length = 1000.0  # m
    
    # Return the complete scenario
    return {
        # Objects
        'dam': dam,
        'upstream_channel': upstream_channel,
        'downstream_channel': downstream_channel,
        
        # Dam parameters
        'dam_height': dam_height,
        'dam_crest_elevation': dam_crest_elevation,
        'dam_base_elevation': dam_base_elevation,
        'design_head': design_head,
        
        # Channel parameters
        'channel_bottom_width': channel_bottom_width,
        'channel_side_slope': channel_side_slope,
        'channel_roughness': channel_roughness,
        'channel_slope': channel_slope,
        'downstream_slope': downstream_slope,
        'upstream_length': upstream_length,
        'downstream_length': downstream_length,
        'channel_width_at_dam': channel_width_at_dam,
        
        # Water levels
        'initial_water_level': initial_water_level,
        'flood_water_level': flood_water_level
    }

def print_scenario_summary(scenario):
    """Print a summary of the scenario parameters."""
    print("\nFlood Scenario Summary")
    print("=" * 60)
    
    print("\nDam Parameters:")
    print(f"  Type: Ogee Spillway")
    print(f"  Height: {scenario['dam_height']:.1f} m")
    print(f"  Crest Elevation: {scenario['dam_crest_elevation']:.1f} m")
    print(f"  Base Elevation: {scenario['dam_base_elevation']:.1f} m")
    print(f"  Design Head: {scenario['design_head']:.1f} m")
    
    print("\nChannel Parameters:")
    print(f"  Type: Trapezoidal")
    print(f"  Bottom Width: {scenario['channel_bottom_width']:.1f} m")
    print(f"  Side Slope: {scenario['channel_side_slope']:.1f} H:1V")
    print(f"  Manning's Roughness: {scenario['channel_roughness']:.3f}")
    print(f"  Upstream Slope: {scenario['channel_slope']:.4f} m/m")
    print(f"  Downstream Slope: {scenario['downstream_slope']:.4f} m/m")
    
    print("\nWater Levels:")
    print(f"  Initial Water Level: {scenario['initial_water_level']:.1f} m")
    print(f"  Flood Peak Water Level: {scenario['flood_water_level']:.1f} m")
    print(f"  Initial Head Over Crest: {max(0, scenario['initial_water_level'] - scenario['dam_crest_elevation']):.2f} m")
    print(f"  Flood Peak Head Over Crest: {max(0, scenario['flood_water_level'] - scenario['dam_crest_elevation']):.2f} m")
    
    print("\nDimensions:")
    print(f"  Upstream Channel Length: {scenario['upstream_length']:.1f} m")
    print(f"  Downstream Channel Length: {scenario['downstream_length']:.1f} m")
    print(f"  Channel Width at Dam: {scenario['channel_width_at_dam']:.1f} m")
    
    print("=" * 60)

if __name__ == "__main__":
    # Create the default scenario
    scenario = create_scenario()
    
    # Print the scenario summary
    print_scenario_summary(scenario)