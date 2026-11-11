"""Physics-informed neural networks for two-dimensional Darcy flow.

PyTorch is an optional dependency. Install it with ``pip install -e '.[pinn]'``.
The homogeneous solver uses a strong-form PDE residual. The heterogeneous
solver uses the Darcy energy functional, whose weak formulation handles
piecewise-constant permeability and enforces interface flux continuity.
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


class _HeterogeneousPressureNetwork(nn.Module):
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
        nn.init.zeros_(self.layers[-1].weight)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        xi, eta = coordinates[:, 0:1], coordinates[:, 1:2]
        # This smoothstep coordinate has zero derivative at both y boundaries.
        eta_no_flow = 3.0 * eta.square() - 2.0 * eta.pow(3)
        transformed = torch.cat((xi, eta_no_flow), dim=1)
        correction = xi * (1.0 - xi)
        return 1.0 - xi + correction * self.layers(transformed)


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


@dataclass(frozen=True)
class HeterogeneousPinnResult:
    """PINN prediction and TPFA-recovered boundary-flow comparison."""

    pressure: np.ndarray
    x: np.ndarray
    y: np.ndarray
    permeability: np.ndarray
    loss_history: tuple[float, ...]
    initial_loss: float
    final_loss: float
    inflow: float
    outflow: float
    finite_volume_inflow: float
    relative_l2_error: float
    relative_flow_error: float
    boundary_balance_error: float


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


def train_heterogeneous_pinn(
    permeability,
    *,
    length_x: float = 1000.0,
    length_y: float = 500.0,
    p_left: float = 2.0e7,
    p_right: float = 1.0e7,
    viscosity: float = 1.0e-3,
    steps: int = 500,
    learning_rate: float = 0.8,
    quadrature_order: int = 8,
    width: int = 64,
    depth: int = 3,
    seed: int = 0,
) -> HeterogeneousPinnResult:
    """Train a weak-form (Deep Ritz) PINN for cellwise-constant permeability.

    The objective is the Darcy energy ``integral(k * |grad(p)|**2)``. Its
    stationary point satisfies ``-div(k grad(p)) = 0`` in the weak sense. A
    single continuous neural pressure field therefore imposes pressure
    continuity, while the weak form gives continuity of normal Darcy flux
    across permeability interfaces. A smooth coordinate map enforces zero
    normal pressure gradient at the top and bottom; the output ansatz enforces
    the prescribed left/right pressures exactly.

    Gauss-Legendre quadrature is applied separately inside every permeability
    cell, so no quadrature stencil straddles a discontinuity in ``k``. The
    result is compared with the existing finite-volume solver on the same
    field. ``steps`` is the maximum number of L-BFGS iterations.
    """
    try:
        k = np.asarray(permeability, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("permeability must be a finite positive 2D array") from exc
    if k.ndim != 2 or min(k.shape) < 1:
        raise ValueError("permeability must have shape (ny, nx) with positive dimensions")
    if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("permeability must be finite and strictly positive everywhere")
    k = np.array(k, copy=True)
    ny, nx = k.shape

    steps = _positive_int("steps", steps)
    quadrature_order = _positive_int("quadrature_order", quadrature_order)
    width = _positive_int("width", width)
    depth = _positive_int("depth", depth)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)

    length_x = _positive_finite("length_x", length_x)
    length_y = _positive_finite("length_y", length_y)
    viscosity = _positive_finite("viscosity", viscosity)
    learning_rate = _positive_finite("learning_rate", learning_rate)
    if isinstance(p_left, bool) or not isinstance(p_left, Real) or not math.isfinite(float(p_left)):
        raise ValueError("p_left must be a finite real number")
    if isinstance(p_right, bool) or not isinstance(p_right, Real) or not math.isfinite(float(p_right)):
        raise ValueError("p_right must be a finite real number")
    p_left, p_right = float(p_left), float(p_right)
    if p_left == p_right:
        raise ValueError("p_left and p_right must differ to define normalized pressure")

    # Normalize coefficients before converting to float32 for PyTorch.
    k_normalized = k / float(np.max(k))
    if float(np.min(k_normalized)) < torch.finfo(torch.float32).tiny:
        raise ValueError(
            "permeability contrast exceeds float32 range; rescale the field or use "
            "a less extreme contrast"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _HeterogeneousPressureNetwork(width=width, depth=depth)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(quadrature_order)
    gauss_nodes = (gauss_nodes + 1.0) / 2.0
    gauss_weights = gauss_weights / 2.0
    cell_j, cell_i = np.indices((ny, nx))
    cell_j, cell_i = cell_j.ravel(), cell_i.ravel()
    quadrature_coordinates = []
    quadrature_k = []
    quadrature_weights = []
    for xi, wx in zip(gauss_nodes, gauss_weights):
        for eta, wy in zip(gauss_nodes, gauss_weights):
            quadrature_coordinates.append(
                np.column_stack(((cell_i + xi) / nx, (cell_j + eta) / ny))
            )
            quadrature_k.append(k_normalized.ravel())
            quadrature_weights.append(np.full(nx * ny, wx * wy))
    coordinates = torch.tensor(
        np.concatenate(quadrature_coordinates), dtype=torch.float32,
        requires_grad=True,
    )
    sampled_k = torch.tensor(
        np.concatenate(quadrature_k), dtype=torch.float32
    )[:, None]
    weights = torch.tensor(
        np.concatenate(quadrature_weights), dtype=torch.float32
    )[:, None]
    cell_count = nx * ny
    energy_x_weight = (length_y / length_x) ** 2

    def energy_loss() -> torch.Tensor:
        normalized_pressure = model(coordinates)
        gradient = torch.autograd.grad(
            normalized_pressure,
            coordinates,
            grad_outputs=torch.ones_like(normalized_pressure),
            create_graph=True,
        )[0]
        density = sampled_k * (
            energy_x_weight * gradient[:, 0:1].square()
            + gradient[:, 1:2].square()
        )
        return torch.sum(weights * density) / cell_count

    with torch.enable_grad():
        initial_loss = float(energy_loss().detach())
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        max_iter=steps,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = energy_loss()
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError("PINN training produced a non-finite energy")
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = float(energy_loss().detach())
    if not math.isfinite(final_loss):
        raise RuntimeError("PINN training produced a non-finite final energy")
    loss_history = (initial_loss, final_loss)

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

    from .darcy_solver import solve_darcy_2d

    finite_volume = solve_darcy_2d(
        nx, ny, length_x, length_y, k, p_left, p_right, viscosity
    )
    tx_left = 2.0 * k[:, 0] * (length_y / ny) / (length_x / nx)
    tx_right = 2.0 * k[:, -1] * (length_y / ny) / (length_x / nx)
    inflow = float(np.sum(tx_left * (p_left - pressure[:, 0]) / viscosity))
    outflow = float(np.sum(tx_right * (pressure[:, -1] - p_right) / viscosity))
    pressure_scale = abs(p_left - p_right)
    relative_l2_error = float(
        np.linalg.norm(pressure - finite_volume.pressure)
        / (math.sqrt(pressure.size) * pressure_scale)
    )
    flow_scale = max(abs(finite_volume.inflow), np.finfo(float).tiny)
    relative_flow_error = float(abs(inflow - finite_volume.inflow) / flow_scale)
    boundary_scale = max(abs(inflow), abs(outflow), np.finfo(float).tiny)
    boundary_balance_error = float(abs(inflow - outflow) / boundary_scale)

    return HeterogeneousPinnResult(
        pressure=pressure,
        x=x,
        y=y,
        permeability=k,
        loss_history=tuple(loss_history),
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        inflow=inflow,
        outflow=outflow,
        finite_volume_inflow=finite_volume.inflow,
        relative_l2_error=relative_l2_error,
        relative_flow_error=relative_flow_error,
        boundary_balance_error=boundary_balance_error,
    )
