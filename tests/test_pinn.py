import pytest

import numpy as np

torch = pytest.importorskip("torch")

from reservoir_agent.pinn import (
    _HeterogeneousPressureNetwork,
    _PressureNetwork,
    train_heterogeneous_pinn,
    train_homogeneous_pinn,
)


def test_network_ansatz_enforces_all_boundary_conditions():
    model = _PressureNetwork(width=8, depth=1)
    coordinates = torch.tensor(
        [[0.0, 0.25], [1.0, 0.75], [0.35, 0.0], [0.72, 1.0]],
        requires_grad=True,
    )
    pressure = model(coordinates)
    gradient = torch.autograd.grad(
        pressure, coordinates, grad_outputs=torch.ones_like(pressure)
    )[0]

    assert pressure[0].item() == pytest.approx(1.0)
    assert pressure[1].item() == pytest.approx(0.0)
    assert gradient[2, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert gradient[3, 1].item() == pytest.approx(0.0, abs=1e-7)


def test_heterogeneous_network_ansatz_enforces_boundaries():
    model = _HeterogeneousPressureNetwork(width=8, depth=1)
    coordinates = torch.tensor(
        [[0.0, 0.25], [1.0, 0.75], [0.35, 0.0], [0.72, 1.0]],
        requires_grad=True,
    )
    pressure = model(coordinates)
    gradient = torch.autograd.grad(
        pressure, coordinates, grad_outputs=torch.ones_like(pressure)
    )[0]

    assert pressure[0].item() == pytest.approx(1.0)
    assert pressure[1].item() == pytest.approx(0.0)
    assert gradient[2, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert gradient[3, 1].item() == pytest.approx(0.0, abs=1e-7)


def test_pinn_training_reduces_residual_and_matches_linear_solution():
    result = train_homogeneous_pinn(
        nx=12,
        ny=6,
        length_x=12.0,
        length_y=6.0,
        steps=700,
        collocation_points=256,
        width=24,
        depth=2,
        seed=3,
    )

    assert result.pressure.shape == (6, 12)
    assert result.x.shape == (12,)
    assert result.y.shape == (6,)
    assert len(result.loss_history) == 701
    assert result.final_loss < result.initial_loss
    assert result.max_pressure_error / 1.0e7 < 1.0e-3
    assert result.relative_l2_error < 1.0e-3
    assert result.flow_rate == pytest.approx(result.exact_flow_rate, rel=1.0e-3)


def test_heterogeneous_pinn_matches_series_layer_reference():
    nx, ny = 16, 4
    column_permeability = np.repeat(
        np.array([1.0e-13, 1.0e-15, 5.0e-13, 2.0e-14]), nx // 4
    )
    permeability = np.tile(column_permeability, (ny, 1))
    result = train_heterogeneous_pinn(
        permeability,
        length_x=16.0,
        length_y=4.0,
        steps=1000,
        quadrature_order=16,
        width=64,
        depth=3,
        seed=3,
    )

    assert result.pressure.shape == (ny, nx)
    assert result.final_loss < result.initial_loss
    assert result.relative_l2_error < 5.0e-3
    assert result.relative_flow_error < 3.0e-2
    assert result.boundary_balance_error < 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nx": 0},
        {"steps": 0},
        {"collocation_points": 0},
        {"length_x": float("inf")},
        {"permeability": -1.0},
        {"viscosity": 0.0},
        {"learning_rate": 0.0},
        {"p_left": 1.0, "p_right": 1.0},
    ],
)
def test_pinn_rejects_invalid_settings(kwargs):
    settings = {"steps": 1, "collocation_points": 4, "width": 4, "depth": 1}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        train_homogeneous_pinn(**settings)
