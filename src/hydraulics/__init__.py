"""
Hydraulics module for open channel flow analysis.

This module provides fundamental hydraulic calculations required for 
analyzing open channel flows and dam effects.
"""

from . import basic
from . import manning
from . import energy

# Expose common functions at module level
from .basic import (
    hydraulic_radius,
    section_properties,
    critical_depth,
    froude_number,
    flow_classification
)

from .manning import (
    manning_velocity,
    discharge,
    normal_depth,
    shear_stress
)

from .energy import (
    specific_energy,
    energy_grade_line,
    momentum_function
)

# Package metadata
__version__ = '0.1.0'