import numpy as np
import pytest

from reservoir_agent.darcy_solver import harmonic_mean, solve_darcy_2d, transmissibilities
from reservoir_agent.physics_validation import cell_mass_residual, linear_pressure

P_LEFT, P_RIGHT = 2.0e6, 1.0e6


def random_field(ny, nx, seed=0, sigma=2.0):
    rng = np.random.default_rng(seed)
    return np.exp(sigma * rng.standard_normal((ny, nx)))


# --- harmonic averaging ----------------------------------------------------

def test_harmonic_mean_values():
    np.testing.assert_allclose(harmonic_mean(1.0, 1.0), 1.0)
    np.testing.assert_allclose(harmonic_mean(1.0, 3.0), 1.5)
    # Dominated by the smaller value: a near-impermeable cell blocks the face.
    assert harmonic_mean(1.0, 1e-6) < 2.1e-6


def test_transmissibilities_shapes_and_boundaries():
    k = np.array([[1.0, 4.0, 2.0], [3.0, 3.0, 3.0]])
    dx, dy = 0.5, 2.0
    tx, ty = transmissibilities(k, dx, dy)
    assert tx.shape == (2, 4) and ty.shape == (3, 3)
    np.testing.assert_allclose(tx[0, 1], dy / dx * 2 * 1.0 * 4.0 / 5.0)
    np.testing.assert_allclose(tx[:, 0], 2 * k[:, 0] * dy / dx)    # half-cell Dirichlet
    np.testing.assert_allclose(tx[:, -1], 2 * k[:, -1] * dy / dx)
    np.testing.assert_allclose(ty[1, 0], dx / dy * 2 * 1.0 * 3.0 / 4.0)
    assert np.all(ty[0] == 0) and np.all(ty[-1] == 0)                 # no flow


# --- exact solutions -------------------------------------------------------

@pytest.mark.parametrize("nx, ny, lx, ly", [(1, 1, 1.0, 1.0), (10, 5, 100.0, 20.0), (37, 13, 3.0, 7.5)])
def test_constant_permeability_reproduces_linear_solution(nx, ny, lx, ly):
    result = solve_darcy_2d(nx, ny, lx, ly, 2.5e-13, P_LEFT, P_RIGHT, viscosity=1e-3)
    exact = linear_pressure(result.x, lx, P_LEFT, P_RIGHT)
    np.testing.assert_allclose(result.pressure, np.broadcast_to(exact, (ny, nx)), rtol=1e-12)


def test_constant_permeability_flux_matches_darcy_law():
    k, mu, lx, ly = 3.0e-13, 2.0e-3, 50.0, 10.0
    result = solve_darcy_2d(20, 8, lx, ly, k, P_LEFT, P_RIGHT, viscosity=mu)
    face_flux = k / mu * (P_LEFT - P_RIGHT) / lx * result.dy   # uniform u = -(k/mu) dp/dx
    np.testing.assert_allclose(result.flux_x, face_flux, rtol=1e-11)
    np.testing.assert_allclose(result.flux_y, 0.0, atol=1e-12 * abs(face_flux))
    np.testing.assert_allclose(result.inflow, k / mu * (P_LEFT - P_RIGHT) / lx * ly, rtol=1e-11)


def test_series_layers_use_harmonic_averaging():
    nx, ny, lx, ly, mu = 6, 3, 6.0, 2.0, 1.0
    k_cols = np.array([1.0, 1.0, 1e-2, 1e-2, 10.0, 10.0])
    k = np.tile(k_cols, (ny, 1))
    result = solve_darcy_2d(nx, ny, lx, ly, k, 1.0, 0.0, viscosity=mu)

    dx = lx / nx
    expected = ly * 1.0 / (mu * np.sum(dx / k_cols))           # resistances in series
    np.testing.assert_allclose(result.flux_x, expected / ny, rtol=1e-12)

    arithmetic = ly * 1.0 / (mu * lx / k_cols.mean())          # wrong averaging for comparison
    assert arithmetic > 10 * expected


def test_series_layers_interface_pressure():
    # Two cells, k = (1, 3): flux continuity gives p_interface = (k1 pL + k2 pR)/(k1 + k2)
    # at the material interface; cell-centre pressures are midway on each linear piece.
    result = solve_darcy_2d(2, 1, 2.0, 1.0, np.array([[1.0, 3.0]]), 1.0, 0.0)
    p_interface = (1.0 * 1.0 + 3.0 * 0.0) / 4.0
    np.testing.assert_allclose(result.pressure[0], [(1.0 + p_interface) / 2, p_interface / 2], rtol=1e-13)


def test_parallel_layers_have_no_crossflow():
    nx, ny, lx, ly = 12, 4, 10.0, 4.0
    k_rows = np.array([1.0, 50.0, 0.1, 5.0])
    k = np.tile(k_rows[:, None], (1, nx))
    result = solve_darcy_2d(nx, ny, lx, ly, k, 3.0, 1.0)
    dy = ly / ny
    np.testing.assert_allclose(result.flux_x, (k_rows * dy * 2.0 / lx)[:, None] * np.ones((1, nx + 1)), rtol=1e-11)
    np.testing.assert_allclose(result.flux_y, 0.0, atol=1e-12)
    exact = linear_pressure(result.x, lx, 3.0, 1.0)
    np.testing.assert_allclose(result.pressure, np.broadcast_to(exact, (ny, nx)), rtol=1e-12)


# --- conservation and symmetry ---------------------------------------------

def test_heterogeneous_field_is_locally_conservative():
    k = random_field(25, 40, seed=1, sigma=2.5)   # permeability contrast ~1e8
    result = solve_darcy_2d(40, 25, 400.0, 250.0, k, P_LEFT, P_RIGHT)
    scale = np.abs(result.flux_x).max()
    assert np.abs(cell_mass_residual(result)).max() <= 1e-10 * scale
    np.testing.assert_allclose(result.inflow, result.outflow, rtol=1e-10)
    assert np.all(result.flux_y[[0, -1], :] == 0.0)
    assert result.inflow > 0.0


def test_fluxes_are_consistent_with_discretization():
    k = random_field(6, 7, seed=2)
    result = solve_darcy_2d(7, 6, 7.0, 3.0, k, 1.0, 0.0, viscosity=0.5)
    tx, ty = transmissibilities(k, result.dx, result.dy)
    p = result.pressure
    np.testing.assert_allclose(result.flux_x[:, 1:-1], tx[:, 1:-1] * (p[:, :-1] - p[:, 1:]) / 0.5)
    np.testing.assert_allclose(result.flux_y[1:-1], ty[1:-1] * (p[:-1] - p[1:]) / 0.5)


def test_reversing_boundary_pressures_reverses_flow():
    k = random_field(8, 10, seed=3)
    forward = solve_darcy_2d(10, 8, 1.0, 1.0, k, 1.0, 0.0)
    backward = solve_darcy_2d(10, 8, 1.0, 1.0, k, 0.0, 1.0)
    np.testing.assert_allclose(forward.pressure + backward.pressure, 1.0, rtol=1e-12)
    np.testing.assert_allclose(forward.flux_x, -backward.flux_x, rtol=1e-10, atol=1e-14)


def test_viscosity_scales_flux_not_pressure():
    k = random_field(5, 6, seed=4)
    a = solve_darcy_2d(6, 5, 1.0, 1.0, k, 1.0, 0.0, viscosity=1.0)
    b = solve_darcy_2d(6, 5, 1.0, 1.0, k, 1.0, 0.0, viscosity=4.0)
    np.testing.assert_allclose(a.pressure, b.pressure, rtol=1e-13)
    np.testing.assert_allclose(a.flux_x, 4.0 * b.flux_x, rtol=1e-12)


def test_equal_boundary_pressures_give_no_flow():
    result = solve_darcy_2d(5, 4, 1.0, 1.0, random_field(4, 5), 7.0, 7.0)
    np.testing.assert_allclose(result.pressure, 7.0, rtol=1e-14)
    np.testing.assert_allclose(result.flux_x, 0.0, atol=1e-12)


# --- parameter validation --------------------------------------------------

GOOD = dict(nx=4, ny=3, length_x=1.0, length_y=1.0, permeability=1.0, p_left=1.0, p_right=0.0, viscosity=1.0)


@pytest.mark.parametrize(
    "name, value",
    [
        ("nx", 0), ("nx", -2), ("nx", 2.5), ("nx", True), ("ny", 0), ("ny", "3"),
        ("length_x", 0.0), ("length_x", -1.0), ("length_y", np.inf), ("length_y", np.nan),
        ("p_left", np.nan), ("p_right", np.inf), ("p_left", "1"),
        ("viscosity", 0.0), ("viscosity", -1e-3),
        ("permeability", 0.0), ("permeability", -1.0), ("permeability", np.nan),
        ("permeability", np.ones((4, 3))),                      # transposed shape
        ("permeability", np.array([[1.0, 1.0, 1.0, -1.0]] * 3)),
        ("permeability", np.array([[1.0, 1.0, np.inf, 1.0]] * 3)),
        ("permeability", "abc"),
    ],
)
def test_invalid_parameters_raise(name, value):
    with pytest.raises(ValueError):
        solve_darcy_2d(**{**GOOD, name: value})


def test_input_permeability_array_is_not_modified():
    k = np.ones((3, 4))
    result = solve_darcy_2d(**{**GOOD, "permeability": k})
    assert result.permeability is not k
    np.testing.assert_array_equal(k, 1.0)
