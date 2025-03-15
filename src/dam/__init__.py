"""
Dam module for hydraulic analysis of dams.

This module provides classes and functions for analyzing dams,
flow over dams, and the interaction with channels.
"""

# Import key components from submodules
from .geometry import (
    Dam,
    BroadCrestedWeir,
    SharpCrestedWeir,
    OgeeWeir,
    create_dam
)

from .flow import (
    calculate_flow_over_dam,
    create_rating_curve,
    calculate_energy_dissipation,
    estimate_tailwater_profile,
    hydraulic_jump_location
)

from .backwater import (
    calculate_backwater_curve,
    calculate_reservoir_characteristics,
    estimate_reservoir_residence_time,
    estimate_water_surface_profile,
    determine_control_point
)

# Package metadata
__version__ = '0.1.0'