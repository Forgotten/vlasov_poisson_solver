import pytest
import jax
import jax.numpy as jnp
from vlasov_poisson_solver import (
    DomainConfig, PhysicsConfig, SolverConfig, VlasovPoissonSolver, create_mesh
)

def test_vlasov_poisson_integration():
  """
  Integration test for the Vlasov-Poisson solver.
  Simulates Landau damping and checks for conservation of mass.
  """
  # Config
  domain_config = DomainConfig(
      x_min=0.0, x_max=4*jnp.pi,
      y_min=0.0, y_max=4*jnp.pi,
      vx_min=-6.0, vx_max=6.0,
      vy_min=-6.0, vy_max=6.0,
      nx=32, ny=32, nvx=32, nvy=32
  )
  physics_config = PhysicsConfig(dt=0.1, final_time=0.5)
  solver_config = SolverConfig(interpolation_order=1)
  
  # Create solver
  # Try to create a mesh (will fallback to 1 device if needed)
  mesh = create_mesh((1, 1), ('x', 'y'))
  solver = VlasovPoissonSolver(
      domain_config, physics_config, solver_config, mesh=mesh
  )
  
  # Initial condition: Landau damping
  # f(x, v) = (1 + alpha * cos(k*x)) * exp(-v^2/2) / sqrt(2pi)
  # In 2D+2D:
  # f(x, y, vx, vy) = (1 + alpha*(cos(kx*x) + cos(ky*y))) * Maxwellian(vx, vy)
  
  alpha = 0.01
  kx = 0.5
  ky = 0.5
  
  x_grid, y_grid = jnp.meshgrid(solver.x, solver.y, indexing='ij')
  vx_grid, vy_grid = jnp.meshgrid(solver.vx, solver.vy, indexing='ij')
  
  # Expand dims for broadcasting
  # X, Y: (nx, ny) -> (nx, ny, 1, 1)
  x_grid = x_grid[:, :, None, None]
  y_grid = y_grid[:, :, None, None]
  # VX, VY: (nvx, nvy) -> (1, 1, nvx, nvy)
  vx_grid = vx_grid[None, None, :, :]
  vy_grid = vy_grid[None, None, :, :]
  
  spatial_perturbation = 1.0 + alpha * (jnp.cos(kx * x_grid) + jnp.cos(ky * y_grid))
  velocity_dist = (1.0 / (2 * jnp.pi)) * jnp.exp(-(vx_grid**2 + vy_grid**2) / 2)
  
  f0 = spatial_perturbation * velocity_dist
  
  # Run one step
  f_next = solver.step(f0, 0.0)
  
  # Check mass conservation
  # Mass = integral f dx dy dvx dvy
  # Since grid is uniform, sum is proportional to integral
  mass_0 = jnp.sum(f0)
  mass_next = jnp.sum(f_next)
  
  # Allow small error due to semi-Lagrangian non-conservation
  assert jnp.isclose(mass_0, mass_next, rtol=1e-2)
  
  # Check positivity (semi-Lagrangian with linear interp should preserve
  # positivity? Actually cubic spline doesn't, linear does if coefficients
  # are positive. map_coordinates with order=1 is bilinear, so it should
  # preserve positivity if values are positive.)
  assert jnp.all(f_next >= -1e-10)

if __name__ == "__main__":
  test_vlasov_poisson_integration()
