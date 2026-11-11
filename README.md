# physics-aware-reservoir-agent

An open-source physics-aware AI agent for PDE-based reservoir simulation, scientific computing, and engineering analysis.

The project combines a verified numerical PDE solver for Darcy flow with a
**local** large language model (`gpt-oss:20b` via [Ollama](https://ollama.com)).
No paid API or external model provider is required.

## The model problem

Steady, single-phase, incompressible Darcy flow on the rectangle $[0, L_x] \times [0, L_y]$:

$$
-\nabla\cdot\big(k(x,y)\,\nabla p\big) = 0, \qquad
\mathbf{u} = -\frac{k}{\mu}\nabla p
$$

* $p = p_L$ on $x = 0$, $p = p_R$ on $x = L_x$ (prescribed pressure)
* $\mathbf{u}\cdot\mathbf{n} = 0$ on $y = 0$ and $y = L_y$ (no flow)

### Discretization

Cell-centred finite volumes with a two-point flux approximation (TPFA) on a
uniform grid, with piecewise-constant permeability per cell:

* **Interior faces** use the harmonic mean $k_f = 2k_Lk_R/(k_L+k_R)$. This is
  the face permeability that makes the flux continuous across a material
  interface, because the two half-cell resistances act in series.
* **Dirichlet faces** use the half-cell distance $\Delta x/2$ from the cell centre to the boundary.
* **No-flow faces** have zero transmissibility.
* The face fluxes reported by the solver use exactly the same
  transmissibilities as the linear system, so the discrete mass balance holds to round-off.

### Physics validation

`reservoir_agent.physics_validation` checks every solution against
properties that the discrete scheme must satisfy exactly:

| Check | What it verifies |
|---|---|
| Local mass conservation | Net outflow of every cell is zero |
| Global balance | Inflow = outflow, and no flux through the no-flow boundaries |
| Maximum principle | $\min(p_L,p_R) \le p \le \max(p_L,p_R)$ |
| Exact linear solution | Constant $k$ reproduces $p = p_L + (p_R-p_L)x/L_x$ |
| Layered media | Series layers give $Q = L_y\Delta p / (\mu\sum \Delta x/k_i)$, parallel layers give $Q = \sum k_j\Delta y\,\Delta p/(\mu L_x)$ |

## Project layout

```text
physics-aware-reservoir-agent/
├── pyproject.toml
├── src/reservoir_agent/
│   ├── darcy_solver.py          # TPFA finite-volume Darcy solver
│   └── physics_validation.py    # discrete physics checks
├── examples/
│   ├── run_baseline.py          # homogeneous case vs exact solution
│   ├── run_pinn.py              # physics-informed neural network baseline
│   ├── run_heterogeneous_pinn.py # Deep Ritz PINN with heterogeneous rock
│   ├── heterogeneous_reservoir.py
│   ├── validate_physics.py      # validation suite over reference cases
│   └── test_local_llm.py        # query gpt-oss:20b through Ollama
└── tests/                       # pytest suite
```

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/danielsinambela/physics-aware-reservoir-agent.git
cd physics-aware-reservoir-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Local LLM (optional)

1. Install [Ollama](https://ollama.com/download) and start it (the desktop app, or `ollama serve`).
2. Download the model (about 13 GB, needs roughly 16 GB of RAM):

   ```bash
   ollama pull gpt-oss:20b
   ```

### Physics-informed neural network (optional)

The PyTorch PINN examples are separate numerical approaches from both the
finite-volume solver and the optional local language model. The homogeneous
example minimizes the strong-form Darcy PDE residual. The heterogeneous
example minimizes the weak-form Darcy energy (Deep Ritz method), using
cell-wise Gauss quadrature so quadrature does not cross permeability
discontinuities. Both enforce prescribed-pressure and no-flow boundaries in
the network output.

```bash
python -m pip install -e ".[pinn]"
python examples/run_pinn.py                    # trains and writes outputs/pinn_baseline.png
python examples/run_pinn.py --steps 5000       # use more Adam optimization steps
python examples/run_heterogeneous_pinn.py      # synthetic heterogeneous field
python examples/run_heterogeneous_pinn.py --steps 1000 --quadrature-order 12
```

The homogeneous example compares its prediction against the exact linear
solution and the finite-volume solution. The heterogeneous example generates a
reproducible correlated permeability field with a high-permeability channel and
compares its predicted pressure and recovered flow with the finite-volume
solution. The weak formulation represents continuity of pressure and normal
flux across cell interfaces. Heterogeneous PINN accuracy depends on network
capacity and quadrature resolution; the finite-volume solution remains the
reference for checking each run. The example prints pressure, flow, and
boundary-balance errors and warns when the PINN has not closely matched that
reference.

## Usage

```bash
python -m pytest tests/ -v                   # run the test suite
python examples/run_baseline.py              # writes outputs/baseline.png
python examples/run_pinn.py                  # homogeneous PINN (requires .[pinn])
python examples/run_heterogeneous_pinn.py    # heterogeneous PINN (requires .[pinn])
python examples/heterogeneous_reservoir.py   # writes outputs/heterogeneous_reservoir.png
python examples/validate_physics.py          # prints the validation report
python examples/test_local_llm.py            # asks gpt-oss:20b a Darcy-flow question
```

Add `--show` to the plotting examples to open the figure window.

Using the solver from Python:

```python
import numpy as np
from reservoir_agent import solve_darcy_2d, validate_solution

k = 1e-13 * np.exp(np.random.default_rng(0).standard_normal((50, 100)))  # m^2, shape (ny, nx)
result = solve_darcy_2d(nx=100, ny=50, length_x=1000.0, length_y=500.0,
                        permeability=k, p_left=2e7, p_right=1e7, viscosity=1e-3)
print(result.inflow)                      # total flow rate per unit depth
print(validate_solution(result).summary())
```

Any consistent unit system works. The examples use SI units (m, m², Pa, Pa·s).

## License

MIT. See [LICENSE](LICENSE).
