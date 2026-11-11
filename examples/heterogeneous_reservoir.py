"""Heterogeneous reservoir: correlated log-normal permeability with a high-permeability channel.

Run:  python examples/heterogeneous_reservoir.py [--seed N] [--show]
Writes outputs/heterogeneous_reservoir.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

from reservoir_agent import solve_darcy_2d, validate_solution

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

LENGTH_X, LENGTH_Y = 1000.0, 500.0
NX, NY = 120, 60
VISCOSITY = 1.0e-3
P_LEFT, P_RIGHT = 2.0e7, 1.0e7
MEAN_PERMEABILITY = 1.0e-13  # geometric mean, ~100 mD


def make_permeability(seed: int) -> np.ndarray:
    """Correlated log-normal field plus a sinuous channel 20x more permeable."""
    rng = np.random.default_rng(seed)
    log_k = gaussian_filter(rng.standard_normal((NY, NX)), sigma=4.0, mode="reflect")
    log_k = 1.5 * log_k / log_k.std()

    x = (np.arange(NX) + 0.5) / NX
    y = (np.arange(NY) + 0.5) / NY
    centre = 0.5 + 0.2 * np.sin(2.0 * np.pi * x)
    channel = np.abs(y[:, None] - centre[None, :]) < 0.06
    log_k[channel] += np.log(20.0)
    return MEAN_PERMEABILITY * np.exp(log_k)


def main(seed: int = 42, show: bool = False) -> int:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    k = make_permeability(seed)
    result = solve_darcy_2d(NX, NY, LENGTH_X, LENGTH_Y, k, P_LEFT, P_RIGHT, VISCOSITY)
    report = validate_solution(result)

    print(f"Grid {NX} x {NY}, permeability range {k.min():.2e} .. {k.max():.2e} m^2 "
          f"(contrast {k.max() / k.min():.0f}x)")
    print(f"Flow rate = {result.inflow:.6e} m^2/s per unit depth")
    k_eff = result.inflow * VISCOSITY * LENGTH_X / ((P_LEFT - P_RIGHT) * LENGTH_Y)
    print(f"Effective permeability = {k_eff:.3e} m^2 "
          f"(harmonic {1 / np.mean(1 / k):.3e} <= k_eff <= arithmetic {k.mean():.3e})")
    print(report.summary())

    u, v = result.cell_velocity()
    speed = np.hypot(u, v)
    extent = (0, LENGTH_X, 0, LENGTH_Y)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), constrained_layout=True)

    im = axes[0].imshow(k / 9.869e-16, origin="lower", extent=extent, cmap="cividis", norm=LogNorm())
    fig.colorbar(im, ax=axes[0], label="permeability [mD]")
    axes[0].set_title("Permeability")

    im = axes[1].imshow(result.pressure / 1e5, origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(im, ax=axes[1], label="pressure [bar]")
    axes[1].set_title("Pressure")

    im = axes[2].imshow(speed, origin="lower", extent=extent, cmap="magma", norm=LogNorm())
    fig.colorbar(im, ax=axes[2], label="|Darcy velocity| [m/s]")
    axes[2].streamplot(result.x, result.y, u, v, color="white", density=1.2, linewidth=0.6)
    axes[2].set_title("Darcy velocity and streamlines")
    for ax in axes:
        ax.set(xlabel="x [m]", ylabel="y [m]")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "heterogeneous_reservoir.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    return 0 if report.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args()
    raise SystemExit(main(args.seed, args.show))
