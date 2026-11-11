from dataclasses import replace

import numpy as np
import pytest

from reservoir_agent.darcy_solver import solve_darcy_2d
from reservoir_agent.physics_validation import (
    check_global_balance,
    check_layered_flow,
    check_linear_solution,
    check_mass_conservation,
    check_maximum_principle,
    validate_solution,
)


@pytest.fixture
def constant_result():
    return solve_darcy_2d(20, 10, 100.0, 50.0, 1e-13, 2.0e7, 1.0e7, viscosity=1e-3)


@pytest.fixture
def heterogeneous_result():
    rng = np.random.default_rng(7)
    k = 1e-13 * np.exp(2.0 * rng.standard_normal((15, 30)))
    return solve_darcy_2d(30, 15, 300.0, 150.0, k, 2.0e7, 1.0e7, viscosity=1e-3)


def test_constant_case_passes_all_checks(constant_result):
    report = validate_solution(constant_result)
    names = {c.name for c in report.checks}
    assert {"local mass conservation", "global balance", "maximum principle", "exact linear solution"} <= names
    assert report.passed, report.summary()


def test_heterogeneous_case_passes_applicable_checks(heterogeneous_result):
    report = validate_solution(heterogeneous_result)
    names = {c.name for c in report.checks}
    assert "exact linear solution" not in names
    assert not any("layered" in n for n in names)
    assert report.passed, report.summary()


def test_series_and_parallel_layers_pass():
    k_series = np.tile(np.array([1.0, 0.01, 5.0, 0.2]), (3, 1))
    k_parallel = k_series.T.copy()
    series = solve_darcy_2d(4, 3, 4.0, 3.0, k_series, 1.0, 0.0)
    parallel = solve_darcy_2d(3, 4, 3.0, 4.0, k_parallel, 1.0, 0.0)
    assert check_layered_flow(series).passed
    assert check_layered_flow(series).name == "series layered flow"
    assert check_layered_flow(parallel).passed
    assert check_layered_flow(parallel).name == "parallel layered flow"


# --- each check must detect a corrupted solution ---------------------------

def test_mass_conservation_detects_flux_error(heterogeneous_result):
    flux_x = heterogeneous_result.flux_x.copy()
    flux_x[7, 12] *= 1.01
    bad = replace(heterogeneous_result, flux_x=flux_x)
    assert not check_mass_conservation(bad).passed


def test_global_balance_detects_leak_through_no_flow_boundary(constant_result):
    flux_y = constant_result.flux_y.copy()
    flux_y[-1, 3] = 1e-3 * np.abs(constant_result.flux_x).max()
    bad = replace(constant_result, flux_y=flux_y)
    assert not check_global_balance(bad).passed


def test_maximum_principle_detects_overshoot(constant_result):
    pressure = constant_result.pressure.copy()
    pressure[2, 0] = 2.0e7 * 1.001
    bad = replace(constant_result, pressure=pressure)
    assert not check_maximum_principle(bad).passed


def test_linear_solution_detects_perturbation(constant_result):
    pressure = constant_result.pressure.copy()
    pressure[4, 9] += 10.0
    bad = replace(constant_result, pressure=pressure)
    assert not check_linear_solution(bad).passed


def test_layered_flow_detects_wrong_total_flow():
    k = np.tile(np.array([1.0, 0.01, 5.0]), (2, 1))
    result = solve_darcy_2d(3, 2, 3.0, 2.0, k, 1.0, 0.0)
    bad = replace(result, flux_x=result.flux_x * 1.001)
    assert not check_layered_flow(bad).passed


def test_failed_check_fails_report(constant_result):
    flux_x = constant_result.flux_x.copy()
    flux_x[0, 5] *= 2.0
    report = validate_solution(replace(constant_result, flux_x=flux_x))
    assert not report.passed
    assert "FAIL" in report.summary()


# --- argument validation ---------------------------------------------------

def test_linear_check_rejects_heterogeneous_field(heterogeneous_result):
    with pytest.raises(ValueError):
        check_linear_solution(heterogeneous_result)


def test_layered_check_rejects_general_field(heterogeneous_result):
    with pytest.raises(ValueError):
        check_layered_flow(heterogeneous_result)


@pytest.mark.parametrize("rtol", [0.0, -1e-9, np.nan, np.inf])
def test_invalid_tolerance_raises(constant_result, rtol):
    with pytest.raises(ValueError):
        validate_solution(constant_result, rtol=rtol)
