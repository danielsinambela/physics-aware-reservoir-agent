"""Train a Deep Ritz PINN for a synthetic heterogeneous reservoir.

Install the optional dependency with ``python -m pip install -e '.[pinn]'``.
Run: python examples/run_heterogeneous_pinn.py [--steps N] [--seed N] [--show]
Writes outputs/heterogeneous_pinn.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

from reservoir_agent.pinn import train_heterogeneous_pinn

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
NX, NY = 24, 12
LENGTH_X, LENGTH_Y = 1000.0, 500.0
P_LEFT, P_RIGHT = 2.0e7, 1.0e7
VISCOSITY = 1.0e-3


def make_permeability(seed: int) -> np.ndarray:
    """Create a correlated log-normal field with a permeable sinuous channel."""
    rng = np.random.default_rng(seed)
    log_k = gaussian_filter(rng.standard_normal((NY, NX)), sigma=1.5, mode="reflect")
    log_k = 1.0 * log_k / log_k.std()
    x = (np.arange(NX) + 0.5) / NX
    y = (np.arange(NY) + 0.5) / NY
    channel_center = 0.5 + 0.2 * np.sin(2.0 * np.pi * x)
    channel = np.abs(y[:, None] - channel_center[None, :]) < 0.09
    log_k[channel] += np.log(10.0)
    return 1.0e-13 * np.exp(log_k)


def main(
    steps: int = 500,
    seed: int = 42,
    quadrature_order: int = 8,
    show: bool = False,
) -> int:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    permeability = make_permeability(seed)
    result = train_heterogeneous_pinn(
        permeability,
        length_x=LENGTH_X,
        length_y=LENGTH_Y,
        p_left=P_LEFT,
        p_right=P_RIGHT,
        viscosity=VISCOSITY,
        steps=steps,
        quadrature_order=quadrature_order,
        seed=seed,
    )
    contrast = float(permeability.max() / permeability.min())
    print("Heterogeneous Darcy-flow Deep Ritz PINN")
    print(f"Grid = {NX} x {NY}, permeability contrast = {contrast:.3e}")
    print(f"Energy loss = {result.initial_loss:.6e} -> {result.final_loss:.6e}")
    print(f"Relative L2 pressure error vs finite volume = {result.relative_l2_error:.6e}")
    print(f"Relative inflow error vs finite volume = {result.relative_flow_error:.6e}")
    print(f"PINN pressure-derived inflow = {result.inflow:.6e} m^2/s per unit depth")
    print(f"PINN pressure-derived outflow = {result.outflow:.6e} m^2/s per unit depth")
    print(f"Relative boundary flow imbalance = {result.boundary_balance_error:.6e}")
    if result.relative_l2_error > 1.0e-2 or result.relative_flow_error > 5.0e-2:
        print(
            "WARNING: the PINN has not matched the finite-volume reference closely; "
            "increase --steps or --quadrature-order and compare again."
        )

    extent = (0, LENGTH_X, 0, LENGTH_Y)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    image = axes[0].imshow(
        permeability / 9.869e-16,
        origin="lower",
        extent=extent,
        cmap="cividis",
        norm=LogNorm(),
    )
    fig.colorbar(image, ax=axes[0], label="permeability [mD]")
    axes[0].set(title="Synthetic permeability", xlabel="x [m]", ylabel="y [m]")

    image = axes[1].imshow(
        result.pressure / 1e5,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    fig.colorbar(image, ax=axes[1], label="pressure [bar]")
    axes[1].set(title="PINN pressure", xlabel="x [m]", ylabel="y [m]")

    axes[2].bar(
        ["inflow", "outflow", "finite-volume inflow"],
        [result.inflow, result.outflow, result.finite_volume_inflow],
        color=["#2878a8", "#63a6c5", "#17324d"],
    )
    axes[2].tick_params(axis="x", labelrotation=20)
    axes[2].set(title="Recovered flow rates", ylabel="m²/s per unit depth")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "heterogeneous_pinn.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500, help="maximum L-BFGS iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quadrature-order",
        type=int,
        default=8,
        help="Gauss points per dimension in each permeability cell",
    )
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args()
    raise SystemExit(main(args.steps, args.seed, args.quadrature_order, args.show))
