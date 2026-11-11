"""Train a PINN for the homogeneous Darcy-flow baseline.

Install the optional dependency with ``python -m pip install -e '.[pinn]'``.
Run: python examples/run_pinn.py [--steps N] [--show]
Writes outputs/pinn_baseline.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from reservoir_agent.pinn import train_homogeneous_pinn

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main(steps: int = 3000, show: bool = False) -> int:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = train_homogeneous_pinn(steps=steps)
    exact_flow_error = abs(result.flow_rate - result.exact_flow_rate) / abs(
        result.exact_flow_rate
    )

    print("Homogeneous Darcy-flow PINN")
    print(f"Optimization steps = {steps}")
    print(f"PDE residual loss  = {result.initial_loss:.6e} -> {result.final_loss:.6e}")
    print(f"Max pressure error = {result.max_pressure_error:.6e} Pa")
    print(f"Relative L2 error vs finite volume = {result.relative_l2_error:.6e}")
    print(f"PINN flow rate     = {result.flow_rate:.6e} m^2/s per unit depth")
    print(f"Exact flow rate    = {result.exact_flow_rate:.6e} m^2/s per unit depth")
    print(f"Relative flow error = {exact_flow_error:.6e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    image = axes[0].imshow(
        result.pressure / 1e5,
        origin="lower",
        aspect="auto",
        extent=(0, 1000.0, 0, 200.0),
        cmap="viridis",
    )
    fig.colorbar(image, ax=axes[0], label="pressure [bar]")
    axes[0].set(title="PINN pressure", xlabel="x [m]", ylabel="y [m]")

    exact = 2.0e7 + (1.0e7 - 2.0e7) * result.x / 1000.0
    axes[1].plot(result.x, exact / 1e5, "k-", label="exact")
    axes[1].plot(result.x, result.pressure[result.pressure.shape[0] // 2] / 1e5,
                 "o", markersize=3, label="PINN")
    axes[1].set(title="Centreline pressure", xlabel="x [m]", ylabel="pressure [bar]")
    axes[1].legend()

    axes[2].semilogy(np.maximum(result.loss_history, np.finfo(float).tiny))
    axes[2].set(title="PDE residual during training", xlabel="optimization step",
                ylabel="mean squared residual")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "pinn_baseline.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000, help="number of Adam steps")
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args()
    raise SystemExit(main(args.steps, args.show))
