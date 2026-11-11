"""Physics-informed neural network for homogeneous two-dimensional Darcy flow.

PyTorch is an optional dependency. Install it with ``pip install -e '.[pinn]'``.
This first PINN implementation supports constant permeability on a rectangular
domain, matching the analytical baseline used by the finite-volume solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    raise ImportError(
        "The PINN requires PyTorch. Install it with `pip install -e '.[pinn]'`."
    ) from exc


class _PressureNetwork(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        xi, eta = coordinates[:, 0:1], coordinates[:, 1:2]
        correction = xi * (1.0 - xi) * eta.square() * (1.0 - eta).square()
        return 1.0 - xi + correction * self.layers(coordinates)


@dataclass(frozen=True)
class PinnResult:
    """PINN pressure prediction and training diagnostics."""

    pressure: np.ndarray
    x: np.ndarray
    y: np.ndarray
    loss_history: tuple[float, ...]
    initial_loss: float
    final_loss: float
    flow_rate: float
    exact_flow_rate: float
    max_pressure_error: float
    relative_l2_error: float


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def train_homogeneous_pinn(
    *,
    nx: int = 100,
    ny: int = 20,
    length_x: float = 1000.0,
    length_y: float = 200.0,
    permeability: float = 1.0e-13,
    p_left: float = 2.0e7,
    p_right: float = 1.0e7,
    viscosity: float = 1.0e-3,
    steps: int = 3000,
    learning_rate: float = 1.0e-3,
    collocation_points: int = 2048,
    width: int = 64,
    depth: int = 3,
    seed: int = 0,
) -> PinnResult:
    """Train a PINN for steady Darcy flow with constant permeability.

    Coordinates and pressure are normalized to ``(xi, eta) in [0, 1]^2`` and
    ``P = (p - p_right) / (p_left - p_right)``. A boundary-constrained output
    ansatz enforces both fixed-pressure boundaries and top/bottom no-flow
    conditions exactly. Training minimizes the interior PDE residual.

    ``loss_history`` contains the initial residual loss and then one value per
    Adam step. Flow is estimated by midpoint quadrature of the learned left
    boundary pressure gradient. The result is compared to both the exact
    linear solution and the finite-volume solver at the requested cell centres.
    """
    nx = _positive_int("nx", nx)
    ny = _positive_int("ny", ny)
    steps = _positive_int("steps", steps)
    collocation_points = _positive_int("collocation_points", collocation_points)
    width = _positive_int("width", width)
    depth = _positive_int("depth", depth)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)

    length_x = _positive_finite("length_x", length_x)
    length_y = _positive_finite("length_y", length_y)
    permeability = _positive_finite("permeability", permeability)
    viscosity = _positive_finite("viscosity", viscosity)
    learning_rate = _positive_finite("learning_rate", learning_rate)
    if isinstance(p_left, bool) or not isinstance(p_left, Real) or not math.isfinite(float(p_left)):
        raise ValueError("p_left must be a finite real number")
    if isinstance(p_right, bool) or not isinstance(p_right, Real) or not math.isfinite(float(p_right)):
        raise ValueError("p_right must be a finite real number")
    p_left, p_right = float(p_left), float(p_right)
    if p_left == p_right:
        raise ValueError("p_left and p_right must differ to define normalized pressure")

    # Keep model initialization deterministic without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _PressureNetwork(width=width, depth=depth)

    sampler = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    coordinates = sampler.draw(collocation_points).requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    aspect_squared = (length_x / length_y) ** 2
    loss_history: list[float] = []

    for _ in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        normalized_pressure = model(coordinates)
        gradient = torch.autograd.grad(
            normalized_pressure,
            coordinates,
            grad_outputs=torch.ones_like(normalized_pressure),
            create_graph=True,
        )[0]
        p_xi = gradient[:, 0:1]
        p_eta = gradient[:, 1:2]
        p_xixi = torch.autograd.grad(
            p_xi,
            coordinates,
            grad_outputs=torch.ones_like(p_xi),
            create_graph=True,
        )[0][:, 0:1]
        p_etaeta = torch.autograd.grad(
            p_eta,
            coordinates,
            grad_outputs=torch.ones_like(p_eta),
            create_graph=True,
        )[0][:, 1:2]
        residual = p_xixi + aspect_squared * p_etaeta
        loss = torch.mean(residual.square())
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError("PINN training produced a non-finite PDE residual")
        loss_history.append(loss_value)
        if len(loss_history) > steps:
            break
        loss.backward()
        optimizer.step()

    model.eval()
    x = (np.arange(nx, dtype=np.float64) + 0.5) * length_x / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * length_y / ny
    xx, yy = np.meshgrid(x / length_x, y / length_y)
    cell_coordinates = torch.tensor(
        np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32
    )
    with torch.no_grad():
        normalized = model(cell_coordinates).cpu().numpy().reshape(ny, nx)
    pressure = p_right + (p_left - p_right) * normalized

    left_coordinates = torch.tensor(
        np.column_stack((np.zeros(ny), y / length_y)), dtype=torch.float32,
        requires_grad=True,
    )
    boundary_pressure = model(left_coordinates)
    boundary_gradient = torch.autograd.grad(
        boundary_pressure,
        left_coordinates,
        grad_outputs=torch.ones_like(boundary_pressure),
    )[0][:, 0].cpu().numpy()
    dp_dx = (p_left - p_right) * boundary_gradient / length_x
    flow_rate = float(np.sum((-permeability / viscosity * dp_dx) * (length_y / ny)))
    exact_flow_rate = (
        permeability / viscosity * (p_left - p_right) / length_x * length_y
    )

    exact_pressure = p_left + (p_right - p_left) * (x / length_x)
    max_pressure_error = float(np.max(np.abs(pressure - exact_pressure[None, :])))

    from .darcy_solver import solve_darcy_2d

    finite_volume = solve_darcy_2d(
        nx, ny, length_x, length_y, permeability, p_left, p_right, viscosity
    )
    pressure_scale = abs(p_left - p_right)
    relative_l2_error = float(
        np.linalg.norm(pressure - finite_volume.pressure)
        / (math.sqrt(pressure.size) * pressure_scale)
    )

    return PinnResult(
        pressure=pressure,
        x=x,
        y=y,
        loss_history=tuple(loss_history),
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        flow_rate=flow_rate,
        exact_flow_rate=exact_flow_rate,
        max_pressure_error=max_pressure_error,
        relative_l2_error=relative_l2_error,
    )
