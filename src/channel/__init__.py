"""
Channel module for open channel flow analysis.

This module provides classes and functions for analyzing flow in
open channels, including normal flow and gradually varied flow calculations.
"""

# Import key components from submodules
from .geometry import (
    Channel,
    RectangularChannel,
    TrapezoidalChannel,
    TriangularChannel,
    CircularChannel,
    CompoundChannel,
    create_channel
)

from .normal_flow import (
    normal_depth,
    critical_depth,
    critical_slope,
    calculate_normal_flow_profile,
    design_channel
)

from .gvf import (
    classify_channel_slope,
    classify_flow_profile,
    compute_froude_number,
    direct_step_method,
    standard_step_method,
    backwater_calculation
)

# Package metadata
__version__ = '0.1.0'