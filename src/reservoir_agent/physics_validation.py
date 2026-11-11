"""Physics checks for solutions produced by :mod:`reservoir_agent.darcy_solver`.

Every check is formulated at the discrete level, so it is consistent with
the two-point flux finite-volume scheme. Checks that pass therefore hold
to round-off, not merely to truncation error:

* **Local mass conservation**: net outflow of every cell, computed from the
  solver's face fluxes, is zero.
* **Global balance**: left inflow equals right outflow, and the top and bottom
  boundary fluxes are exactly zero.
* **Discrete maximum principle**: pressures stay within ``[min(pL, pR), max(pL, pR)]``.
* **Exact linear solution** for constant permeability: the scheme reproduces
  ``p(x) = pL + (pR - pL) x / Lx`` exactly at cell centres.
* **Layered media**: for permeability varying only in ``x`` (layers in series)
  the total flow equals ``Ly (pL - pR) / (mu * sum(dx / k_i))``, which only holds
  with harmonic face averaging. For permeability varying only in ``y``
  (layers in parallel) the flow equals ``sum(k_j dy) (pL - pR) / (mu Lx)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .darcy_solver import DarcyResult

__all__ = [
    "CheckResult",
    "ValidationReport",
    "cell_mass_residual",
    "check_global_balance",
    "check_layered_flow",
    "check_linear_solution",
    "check_mass_conservation",
    "check_maximum_principle",
    "linear_pressure",
    "parallel_layer_flow",
    "series_layer_flow",
    "validate_solution",
]

DEFAULT_RTOL = 1e-9


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one physics check. ``error`` is a relative error compared to ``tolerance``."""

    name: str
    passed: bool
    error: float
    tolerance: float
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        text = f"[{status}] {self.name}: error={self.error:.3e} (tol {self.tolerance:.1e})"
        return f"{text} - {self.detail}" if self.detail else text


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [str(c) for c in self.checks]
        n_pass = sum(c.passed for c in self.checks)
        lines.append(f"{n_pass}/{len(self.checks)} checks passed")
        return "\n".join(lines)


def _check_tolerance(rtol: float) -> float:
    rtol = float(rtol)
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be a positive finite number, got {rtol}")
    return rtol


def _relative(error: float, scale: float) -> float:
    if scale > 0.0:
        return error / scale
    return 0.0 if error == 0.0 else float("inf")


def _flux_scale(result: DarcyResult) -> float:
    return float(max(np.abs(result.flux_x).max(), np.abs(result.flux_y).max()))


def _pressure_scale(result: DarcyResult) -> float:
    return float(max(abs(result.p_left), abs(result.p_right), abs(result.p_left - result.p_right)))


# ---------------------------------------------------------------------------
# Reference solutions
# ---------------------------------------------------------------------------

def linear_pressure(x, length_x: float, p_left: float, p_right: float) -> np.ndarray:
    """Exact pressure for constant permeability: linear between the two boundaries."""
    return p_left + (p_right - p_left) * np.asarray(x, dtype=float) / length_x


def series_layer_flow(result: DarcyResult) -> float:
    """Exact discrete total flow when permeability varies only in ``x``."""
    k_column = result.permeability[0, :]
    resistance = np.sum(result.dx / k_column)
    return result.length_y * (result.p_left - result.p_right) / (result.viscosity * resistance)


def parallel_layer_flow(result: DarcyResult) -> float:
    """Exact discrete total flow when permeability varies only in ``y``."""
    k_row = result.permeability[:, 0]
    return np.sum(k_row * result.dy) * (result.p_left - result.p_right) / (
        result.viscosity * result.length_x
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def cell_mass_residual(result: DarcyResult) -> np.ndarray:
    """Net outflow of each cell, shape (ny, nx). Zero for a conservative solution."""
    return (
        result.flux_x[:, 1:] - result.flux_x[:, :-1]
        + result.flux_y[1:, :] - result.flux_y[:-1, :]
    )


def check_mass_conservation(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> CheckResult:
    rtol = _check_tolerance(rtol)
    residual = float(np.abs(cell_mass_residual(result)).max())
    error = _relative(residual, _flux_scale(result))
    return CheckResult(
        "local mass conservation", error <= rtol, error, rtol,
        f"max |cell net outflow| = {residual:.3e}",
    )


def check_global_balance(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> CheckResult:
    rtol = _check_tolerance(rtol)
    imbalance = abs(result.inflow - result.outflow)
    boundary_leak = float(np.abs(result.flux_y[0, :]).sum() + np.abs(result.flux_y[-1, :]).sum())
    error = _relative(imbalance + boundary_leak, _flux_scale(result))
    return CheckResult(
        "global balance", error <= rtol, error, rtol,
        f"inflow={result.inflow:.6e}, outflow={result.outflow:.6e}, "
        f"no-flow leak={boundary_leak:.1e}",
    )


def check_maximum_principle(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> CheckResult:
    rtol = _check_tolerance(rtol)
    lower = min(result.p_left, result.p_right)
    upper = max(result.p_left, result.p_right)
    overshoot = float(max(lower - result.pressure.min(), result.pressure.max() - upper, 0.0))
    error = _relative(overshoot, _pressure_scale(result))
    return CheckResult(
        "maximum principle", error <= rtol, error, rtol,
        f"p in [{result.pressure.min():.6g}, {result.pressure.max():.6g}], "
        f"bounds [{lower:.6g}, {upper:.6g}]",
    )


def check_linear_solution(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> CheckResult:
    """Compare with the exact linear solution. Only valid for constant permeability."""
    rtol = _check_tolerance(rtol)
    if np.ptp(result.permeability) != 0.0:
        raise ValueError("the linear exact solution only applies to constant permeability")
    exact = linear_pressure(result.x, result.length_x, result.p_left, result.p_right)
    error_abs = float(np.abs(result.pressure - exact[np.newaxis, :]).max())
    error = _relative(error_abs, _pressure_scale(result))
    return CheckResult(
        "exact linear solution", error <= rtol, error, rtol,
        f"max |p - p_exact| = {error_abs:.3e}",
    )


def _layering(k: np.ndarray) -> str | None:
    varies_in_y = np.ptp(k, axis=0).max() != 0.0
    varies_in_x = np.ptp(k, axis=1).max() != 0.0
    if not varies_in_y:
        return "series"
    if not varies_in_x:
        return "parallel"
    return None


def check_layered_flow(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> CheckResult:
    """Compare total flow with the exact series or parallel formula for layered media."""
    rtol = _check_tolerance(rtol)
    layering = _layering(result.permeability)
    if layering is None:
        raise ValueError("permeability is not layered in x or y")
    expected = series_layer_flow(result) if layering == "series" else parallel_layer_flow(result)
    error = _relative(abs(result.inflow - expected), abs(expected))
    return CheckResult(
        f"{layering} layered flow", error <= rtol, error, rtol,
        f"Q={result.inflow:.6e}, exact={expected:.6e}",
    )


def validate_solution(result: DarcyResult, rtol: float = DEFAULT_RTOL) -> ValidationReport:
    """Run every check that applies to ``result``."""
    rtol = _check_tolerance(rtol)
    report = ValidationReport([
        check_mass_conservation(result, rtol),
        check_global_balance(result, rtol),
        check_maximum_principle(result, rtol),
    ])
    if np.ptp(result.permeability) == 0.0:
        report.checks.append(check_linear_solution(result, rtol))
    if _layering(result.permeability) is not None:
        report.checks.append(check_layered_flow(result, rtol))
    return report
