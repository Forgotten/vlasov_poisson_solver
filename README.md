# Vlasov-Poisson Solver

A semi-Lagrangian solver for the Vlasov-Poisson equation in 2D position space and 2D velocity space (4D phase space) using JAX.

## Features
- **Semi-Lagrangian Advection**: Efficient interpolation-based advection.
- **FFT-based Poisson Solver**: Fast solution for periodic domains.
- **Strang Splitting**: Second-order time accuracy.
- **Sharding**: Parallel computation across multiple devices (GPUs/TPUs).

## Installation

You can install the package using pip:

```bash
pip install .
```

Or for development (editable install):

```bash
pip install -e .
```

## Usage

```python
from vlasov_poisson_solver import VlasovPoissonSolver, DomainConfig, PhysicsConfig, SolverConfig

# Configure
domain = DomainConfig(Nx=64, Ny=64, Nvx=64, Nvy=64)
physics = PhysicsConfig(dt=0.1)
solver_config = SolverConfig(interpolation_order=1)

# Initialize Solver
solver = VlasovPoissonSolver(domain, physics, solver_config)

# Run Step
f_next = solver.step(f_current, t=0.0)
```

## Demo

Check out the [demo notebook](demo/demo.ipynb) for a complete example.
