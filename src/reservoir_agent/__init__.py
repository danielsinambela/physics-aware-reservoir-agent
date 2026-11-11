"""Physics-aware tools for PDE-based reservoir simulation."""

from .darcy_solver import DarcyResult, harmonic_mean, solve_darcy_2d, transmissibilities
from .physics_validation import CheckResult, ValidationReport, validate_solution

__all__ = [
    "CheckResult",
    "DarcyResult",
    "ValidationReport",
    "harmonic_mean",
    "solve_darcy_2d",
    "transmissibilities",
    "validate_solution",
]

__version__ = "0.1.0"
