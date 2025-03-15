# Enhanced Hydraulic Visualization System

This documentation covers the enhanced visualization capabilities implemented for the hydraulic modeling system. The visualization modules provide high-quality, informative visualizations for 1D hydraulic models, including dam and channel systems.

## Overview

The enhanced visualization system includes:

1. **Improved Profile Visualizations**
   - Dynamic coloring based on hydraulic parameters (Froude number, velocity, depth)
   - Enhanced water body representation with shading and transparency
   - Detailed annotations and reference lines

2. **Multi-Parameter Visualizations**
   - Combined plots showing multiple hydraulic parameters
   - Cross-section views at key locations
   - Flow regime visualization with color-coded regions

3. **Time-Series Animations**
   - Flood progression animations with dynamic coloring
   - Flow regime transition visualization
   - Animated cross-sections

4. **Interactive Visualizations**
   - Parameter sliders for exploring different scenarios
   - Toggle controls for display options
   - Interactive cross-section location selection

5. **Publication and Presentation Styling**
   - Consistent styling system
   - Templates for publication-quality figures
   - Presentation-friendly visualizations

## Module Structure

```
src/visualization/
  __init__.py               # Package initialization
  profiles.py               # Original basic visualizations (unchanged)
  enhanced_profiles.py      # Enhanced profile visualizations
  time_series.py            # Time-series animations
  cross_sections.py         # Cross-section visualizations
  flow_regimes.py           # Flow regime analysis visualizations
  interactive.py            # Interactive controls and dashboards
  styling.py                # Styling and color schemes
```

## Key Functions

### Enhanced Profile Visualizations

- `plot_enhanced_profile(scenario, results, color_by='froude')`: Creates an enhanced visualization of the hydraulic profile with color mapping.
- `create_profile_with_parameter_plots(scenario, results, parameters)`: Creates a multi-panel figure with the main profile and additional parameter plots.
- `plot_enhanced_profile_with_cross_sections(scenario, results, x_locations)`: Creates a figure with the main profile and multiple cross-section views.

### Cross-Section Visualizations

- `plot_channel_cross_section(ax, channel_type, channel_params, water_depth)`: Plots a detailed channel cross-section with water.
- `create_cross_section_dashboard(scenario, results, locations)`: Creates a comprehensive dashboard of cross-sections at multiple locations.
- `create_animated_cross_section(scenario, results_list, location)`: Creates an animation of a channel cross-section changing over time.

### Flow Regime Visualizations

- `plot_flow_regime_profile(scenario, results)`: Creates a visualization highlighting flow regimes along the channel.
- `create_flow_regime_dashboard(scenario, results)`: Creates a comprehensive dashboard showing flow regimes with multiple visualizations.
- `plot_regime_map(scenarios, dam_heights, discharges)`: Creates a flow regime map showing how flow characteristics change with different parameters.

### Time-Series Animations

- `create_enhanced_flood_animation(scenario, results_list, color_by='froude')`: Creates an enhanced animation of a flood event with dynamic coloring.
- `create_flow_regime_transition_animation(scenario, results_list)`: Creates an animation specifically focused on flow regime transitions.

### Interactive Visualizations

- `create_interactive_profile(scenario, results_list)`: Creates an interactive visualization with sliders to control the display.
- `create_interactive_cross_section_viewer(scenario, results)`: Creates an interactive viewer that allows exploration of cross-sections.
- `create_interactive_flood_explorer(scenario, results_list)`: Creates an interactive explorer for analyzing flood progression.

### Styling and Helper Functions

- `apply_theme(theme='default')`: Applies a predefined theme to all matplotlib plots.
- `style_for_publication(fig)`: Applies styling suitable for publication to a figure.
- `style_for_presentation(fig)`: Applies styling suitable for presentations to a figure.
- `create_single_result_visualization(scenario, results)`: Creates a comprehensive visualization of a single hydraulic result.
- `create_multi_result_visualization(scenario, results_list)`: Creates a visualization for multiple hydraulic results.
- `visualize_hydraulic_jump(scenario, results)`: Creates specialized visualization focusing on hydraulic jump characteristics.

## Usage Examples

Here's a quick example of creating a simple enhanced visualization:

```python
from src.visualization import plot_enhanced_profile

# Create an enhanced profile visualization
fig, ax = plot_enhanced_profile(scenario, results, color_by='froude')
plt.savefig('enhanced_profile.png', dpi=300, bbox_inches='tight')
```

For more complex visualization with multiple parameters:

```python
from src.visualization import create_profile_with_parameter_plots

# Create a profile with velocity and Froude number plots
fig, axs = create_profile_with_parameter_plots(
    scenario, results, 
    parameters=['velocity', 'froude']
)
plt.savefig('profile_with_parameters.png', dpi=300, bbox_inches='tight')
```

For flow regime visualization:

```python
from src.visualization import plot_flow_regime_profile

# Create a flow regime visualization
fig, ax = plot_flow_regime_profile(scenario, results)
plt.savefig('flow_regime_profile.png', dpi=300, bbox_inches='tight')
```

Creating animations:

```python
from src.visualization import create_enhanced_flood_animation

# Create a flood animation
anim, fig, ax = create_enhanced_flood_animation(
    scenario, results_list, fps=5, color_by='froude'
)
anim.save('flood_animation.gif', writer='pillow', fps=5)
```

For detailed examples, see the `visualization_examples.py` script.

## Tips for Getting the Best Results

1. **Color Mapping**: Choose appropriate color mapping based on what you want to highlight:
   - `'froude'`: Best for visualizing flow regimes (subcritical vs. supercritical)
   - `'velocity'`: Useful for showing speed variations
   - `'depth'`: Shows water depth variations

2. **Cross-Sections**: Place cross-sections at hydraulically significant locations:
   - Near the dam
   - At hydraulic jumps
   - At locations with maximum or minimum Froude numbers
   - At locations with significant changes in bed slope

3. **Animation Settings**: 
   - Use lower fps (3-5) for clearer visualization of changes
   - Increase the number of frames for smoother transitions
   - Use `color_by='froude'` to highlight regime transitions

4. **Publication Quality**:
   - Use `style_for_publication(fig)` for journal-quality figures
   - Increase dpi to 600 for high-resolution output
   - Use minimal annotations to avoid clutter

5. **Presentation Quality**:
   - Use `style_for_presentation(fig)` for clear, bold figures
   - Add comprehensive annotations
   - Use animated GIFs to demonstrate dynamic processes

## Extending the Visualization System

The visualization system is designed to be extensible. You can create custom visualizations by:

1. Combining existing visualization functions
2. Creating new visualization functions that leverage the styling system
3. Extending the interactive controls

Example of creating a custom visualization function:

```python
def my_custom_visualization(scenario, results):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use existing visualization functions
    from src.visualization.enhanced_profiles import plot_enhanced_profile
    from src.visualization.styling import apply_theme
    
    # Apply styling
    apply_theme('default')
    
    # Create the plot
    plot_enhanced_profile(scenario, results, ax=ax, color_by='velocity')
    
    # Add custom elements
    # ...
    
    return fig, ax
```

## Notes on Integration with Existing Code

The enhanced visualization system is designed to work seamlessly with the existing hydraulic calculation code. It uses the same data structures and formats for scenario parameters and results.

Key integration points:

1. **Scenario Format**: The visualization functions expect the scenario format used by `create_scenario()` in `scenario_setup.py`.
2. **Results Format**: The visualization functions expect the results format returned by `analyze_steady_state()` in `steady_analysis.py`.
3. **Existing Visualizations**: The original visualization functions in `profiles.py` are still available and unchanged.

## Future Improvements

Potential areas for future enhancement:

1. Support for additional channel and dam types
2. Interactive web-based visualizations (using Plotly or Bokeh)
3. 3D visualizations of water surfaces
4. Export of animations to video formats
5. Real-time visualization during hydraulic calculations