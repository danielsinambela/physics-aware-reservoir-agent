"""Baseline case: homogeneous reservoir, compared with the exact linear solution.

Run:  python examples/run_baseline.py [--show]
Writes outputs/baseline.png.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from reservoir_agent import solve_darcy_2d, validate_solution
from reservoir_agent.physics_validation import linear_pressure

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# SI units: m, m^2, Pa, Pa*s
LENGTH_X, LENGTH_Y = 1000.0, 200.0
NX, NY = 100, 20
PERMEABILITY = 1.0e-13          # ~100 mD
VISCOSITY = 1.0e-3              # water
P_LEFT, P_RIGHT = 2.0e7, 1.0e7  # 200 bar -> 100 bar


def main(show: bool = False) -> int:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = solve_darcy_2d(NX, NY, LENGTH_X, LENGTH_Y, PERMEABILITY, P_LEFT, P_RIGHT, VISCOSITY)
    exact = linear_pressure(result.x, LENGTH_X, P_LEFT, P_RIGHT)
    max_error = np.abs(result.pressure - exact).max()
    analytic_rate = PERMEABILITY / VISCOSITY * (P_LEFT - P_RIGHT) / LENGTH_X * LENGTH_Y

    print(f"Grid {NX} x {NY}, k = {PERMEABILITY:.2e} m^2, mu = {VISCOSITY:.1e} Pa*s")
    print(f"Max |p - p_exact|   = {max_error:.3e} Pa")
    print(f"Flow rate (numeric) = {result.inflow:.6e} m^2/s per unit depth")
    print(f"Flow rate (exact)   = {analytic_rate:.6e} m^2/s per unit depth")
    report = validate_solution(result)
    print(report.summary())

    fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    image = ax_map.imshow(
        result.pressure / 1e5, origin="lower", aspect="auto",
        extent=(0, LENGTH_X, 0, LENGTH_Y), cmap="viridis",
    )
    fig.colorbar(image, ax=ax_map, label="pressure [bar]")
    ax_map.set(title="Pressure, homogeneous reservoir", xlabel="x [m]", ylabel="y [m]")

    ax_line.plot(result.x, exact / 1e5, "k-", label="exact (linear)")
    ax_line.plot(result.x[::5], result.pressure[NY // 2, ::5] / 1e5, "o", label="finite volume")
    ax_line.set(title="Centreline pressure", xlabel="x [m]", ylabel="pressure [bar]")
    ax_line.legend()

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "baseline.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if show:
        plt.show()
    return 0 if report.passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    raise SystemExit(main(parser.parse_args().show))
