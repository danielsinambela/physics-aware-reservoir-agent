"""Run the physics validation suite on a set of reference cases.

Run:  python examples/validate_physics.py
Exit code is 0 only if every check passes.
"""

import numpy as np

from reservoir_agent import solve_darcy_2d, validate_solution


def cases():
    nx, ny, lx, ly = 40, 20, 400.0, 200.0
    rng = np.random.default_rng(0)
    yield "homogeneous", (nx, ny, lx, ly, 1e-13)
    yield "layers in series (x)", (nx, ny, lx, ly, np.tile(np.repeat([1e-13, 1e-15, 5e-13, 2e-14], nx // 4), (ny, 1)))
    yield "layers in parallel (y)", (nx, ny, lx, ly, np.tile(np.repeat([1e-13, 1e-15, 5e-13, 2e-14], ny // 4)[:, None], (1, nx)))
    yield "random log-normal", (nx, ny, lx, ly, 1e-13 * np.exp(2.0 * rng.standard_normal((ny, nx))))
    yield "single cell", (1, 1, 1.0, 1.0, 1.0)


def main() -> int:
    all_passed = True
    for name, (nx, ny, lx, ly, k) in cases():
        result = solve_darcy_2d(nx, ny, lx, ly, k, p_left=2.0e7, p_right=1.0e7, viscosity=1e-3)
        report = validate_solution(result)
        all_passed &= report.passed
        print(f"== {name} ==")
        print(report.summary(), end="\n\n")
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
