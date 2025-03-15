"""
Numerical module for solving differential equations in hydraulics.

This module provides numerical methods for solving the differential
equations that arise in hydraulic calculations.
"""

# Import key components from submodules
from .finite_diff import (
    direct_step_method,
    standard_step_method,
    implicit_euler_method,
    crank_nicolson_method,
    forward_time_centered_space,
    lax_wendroff_method,
    preissmann_scheme
)

from .runge_kutta import (
    runge_kutta_4,
    adaptive_rk4,
    runge_kutta_fehlberg,
    runge_kutta_system,
    adaptive_runge_kutta_system,
    solve_boundary_value_problem,
    solve_unsteady_flow
)

# Package metadata
__version__ = '0.1.0'