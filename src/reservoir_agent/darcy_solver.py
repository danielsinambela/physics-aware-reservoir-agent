"""Two-point flux finite-volume solver for steady single-phase Darcy flow.

Problem
-------
On the rectangle ``[0, Lx] x [0, Ly]`` solve

    -div( k(x, y) grad p ) = 0

with

* ``p = p_left``  on ``x = 0``   (Dirichlet),
* ``p = p_right`` on ``x = Lx``  (Dirichlet),
* ``k dp/dn = 0`` on ``y = 0`` and ``y = Ly`` (no flow).

Discretization
--------------
Cell-centred finite volumes on a uniform ``ny x nx`` grid with a two-point
flux approximation (TPFA). Permeability is constant within each cell, and
``permeability[j, i]`` is the value in the cell centred at ``(x[i], y[j])``.
Row ``j = 0`` is the bottom of the domain.

The flux through the face between cells ``L`` and ``R`` (distance ``h`` apart,
face area ``A``) is

    F = T (p_L - p_R) / mu,    T = A * k_face / h,
    k_face = 2 k_L k_R / (k_L + k_R)   (harmonic mean).

The harmonic mean comes from requiring the flux to be continuous across a face
where ``k`` jumps: the two half-cell resistances ``(h/2)/k_L`` and
``(h/2)/k_R`` are in series. On a Dirichlet boundary the face is ``h/2`` from
the cell centre, so ``T = A * k / (h/2)``. No-flow faces have ``T = 0``.

All quantities are per unit depth, in any consistent unit system (for example
SI: m, m^2, Pa, Pa*s, giving fluxes in m^2/s).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

__all__ = [
    "DarcyResult",
    "harmonic_mean",
    "solve_darcy_2d",
    "transmissibilities",
]


def harmonic_mean(a, b):
    """Return the elementwise harmonic mean ``2ab / (a + b)`` of positive arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 2.0 * a * b / (a + b)


@dataclass(frozen=True)
class DarcyResult:
    """Solution of :func:`solve_darcy_2d`.

    Attributes
    ----------
    pressure : (ny, nx) array
        Cell-centred pressure.
    flux_x : (ny, nx + 1) array
        Volumetric flux (per unit depth) through vertical faces, positive in
        ``+x``. Column 0 is the left boundary and column ``nx`` the right one.
    flux_y : (ny + 1, nx) array
        Flux through horizontal faces, positive in ``+y``. Rows 0 and ``ny``
        are the no-flow boundaries and are exactly zero.
    x, y : 1-D arrays
        Cell-centre coordinates.
    """

    pressure: np.ndarray
    flux_x: np.ndarray
    flux_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    permeability: np.ndarray
    length_x: float
    length_y: float
    p_left: float
    p_right: float
    viscosity: float

    @property
    def nx(self) -> int:
        return self.pressure.shape[1]

    @property
    def ny(self) -> int:
        return self.pressure.shape[0]

    @property
    def dx(self) -> float:
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        return self.length_y / self.ny

    @property
    def inflow(self) -> float:
        """Total flux entering through the left boundary (positive for ``+x`` flow)."""
        return float(self.flux_x[:, 0].sum())

    @property
    def outflow(self) -> float:
        """Total flux leaving through the right boundary."""
        return float(self.flux_x[:, -1].sum())

    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Darcy velocity ``(u, v)`` at cell centres, averaged from face fluxes."""
        u = 0.5 * (self.flux_x[:, :-1] + self.flux_x[:, 1:]) / self.dy
        v = 0.5 * (self.flux_y[:-1, :] + self.flux_y[1:, :]) / self.dx
        return u, v


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _finite_real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _positive_real(name: str, value) -> float:
    value = _finite_real(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _permeability_field(permeability, nx: int, ny: int) -> np.ndarray:
    try:
        k = np.asarray(permeability, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("permeability must be numeric") from exc
    if k.ndim == 0:
        k = np.full((ny, nx), float(k))
    elif k.shape != (ny, nx):
        raise ValueError(
            f"permeability must be a scalar or have shape (ny, nx) = ({ny}, {nx}), "
            f"got {k.shape}"
        )
    else:
        k = k.copy()
    if not np.all(np.isfinite(k)):
        raise ValueError("permeability must be finite everywhere")
    if np.any(k <= 0.0):
        raise ValueError("permeability must be strictly positive everywhere")
    return k


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

def transmissibilities(k: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Face transmissibilities ``(Tx, Ty)`` for a permeability field ``k`` of shape (ny, nx).

    ``Tx`` has shape (ny, nx + 1) and includes the Dirichlet half-cell
    transmissibilities on the left and right boundaries. ``Ty`` has shape
    (ny + 1, nx) and is zero on the no-flow top and bottom boundaries.
    """
    ny, nx = k.shape
    tx = np.empty((ny, nx + 1))
    tx[:, 1:-1] = (dy / dx) * harmonic_mean(k[:, :-1], k[:, 1:])
    tx[:, 0] = 2.0 * k[:, 0] * dy / dx
    tx[:, -1] = 2.0 * k[:, -1] * dy / dx

    ty = np.zeros((ny + 1, nx))
    ty[1:-1, :] = (dx / dy) * harmonic_mean(k[:-1, :], k[1:, :])
    return tx, ty


def _assemble(tx: np.ndarray, ty: np.ndarray, p_left: float, p_right: float):
    """Assemble the symmetric positive-definite TPFA system ``A p = b``."""
    ny, nx = tx.shape[0], tx.shape[1] - 1
    n = nx * ny
    index = np.arange(n).reshape(ny, nx)

    diagonal = tx[:, :-1] + tx[:, 1:] + ty[:-1, :] + ty[1:, :]

    rows = [index.ravel()]
    cols = [index.ravel()]
    vals = [diagonal.ravel()]

    # Interior x-faces couple (j, i) and (j, i + 1).
    left, right, t = index[:, :-1].ravel(), index[:, 1:].ravel(), tx[:, 1:-1].ravel()
    rows += [left, right]
    cols += [right, left]
    vals += [-t, -t]

    # Interior y-faces couple (j, i) and (j + 1, i).
    below, above, t = index[:-1, :].ravel(), index[1:, :].ravel(), ty[1:-1, :].ravel()
    rows += [below, above]
    cols += [above, below]
    vals += [-t, -t]

    matrix = sparse.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )

    rhs = np.zeros((ny, nx))
    rhs[:, 0] += tx[:, 0] * p_left
    rhs[:, -1] += tx[:, -1] * p_right
    return matrix, rhs.ravel()


def solve_darcy_2d(
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
    permeability,
    p_left: float,
    p_right: float,
    viscosity: float = 1.0,
) -> DarcyResult:
    """Solve steady Darcy flow with fixed pressures left/right and no flow top/bottom.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y (each >= 1).
    length_x, length_y : float
        Domain size (> 0).
    permeability : float or (ny, nx) array
        Strictly positive, finite cell permeabilities.
    p_left, p_right : float
        Dirichlet pressures on ``x = 0`` and ``x = length_x``.
    viscosity : float
        Fluid viscosity (> 0). It does not affect the pressure, only the fluxes.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    nx = _positive_int("nx", nx)
    ny = _positive_int("ny", ny)
    length_x = _positive_real("length_x", length_x)
    length_y = _positive_real("length_y", length_y)
    p_left = _finite_real("p_left", p_left)
    p_right = _finite_real("p_right", p_right)
    viscosity = _positive_real("viscosity", viscosity)
    k = _permeability_field(permeability, nx, ny)

    dx, dy = length_x / nx, length_y / ny
    tx, ty = transmissibilities(k, dx, dy)
    matrix, rhs = _assemble(tx, ty, p_left, p_right)
    pressure = np.asarray(spsolve(matrix.tocsc(), rhs)).reshape(ny, nx)
    if not np.all(np.isfinite(pressure)):
        raise RuntimeError("linear solve produced non-finite pressures")

    # Fluxes use exactly the transmissibilities of the linear system, so the
    # discrete cell balances equal the solver residual (A p - b) / mu.
    mobility = 1.0 / viscosity
    flux_x = np.empty((ny, nx + 1))
    flux_x[:, 1:-1] = mobility * tx[:, 1:-1] * (pressure[:, :-1] - pressure[:, 1:])
    flux_x[:, 0] = mobility * tx[:, 0] * (p_left - pressure[:, 0])
    flux_x[:, -1] = mobility * tx[:, -1] * (pressure[:, -1] - p_right)

    flux_y = np.zeros((ny + 1, nx))
    flux_y[1:-1, :] = mobility * ty[1:-1, :] * (pressure[:-1, :] - pressure[1:, :])

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    return DarcyResult(
        pressure=pressure,
        flux_x=flux_x,
        flux_y=flux_y,
        x=x,
        y=y,
        permeability=k,
        length_x=length_x,
        length_y=length_y,
        p_left=p_left,
        p_right=p_right,
        viscosity=viscosity,
    )
