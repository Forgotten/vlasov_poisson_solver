import pytest
import jax.numpy as jnp
from vlasov_poisson_solver import advect_2d, compute_advection_indices

def test_advection_constant_velocity():
  """Test advection of a Gaussian pulse with constant velocity."""
  nx, ny = 64, 64
  x = jnp.linspace(0, 1, nx, endpoint=False)
  y = jnp.linspace(0, 1, ny, endpoint=False)
  x_grid, y_grid = jnp.meshgrid(x, y, indexing='ij')
  dx = 1.0 / nx
  dy = 1.0 / ny
  
  # Initial Gaussian pulse centered at (0.5, 0.5).
  sigma = 0.1
  f0 = jnp.exp(-((x_grid - 0.5)**2 + (y_grid - 0.5)**2) / (2 * sigma**2))
  
  # Constant velocity.
  vx = 0.1
  vy = 0.1
  dt = 1.0
  
  # Expected center after time dt: (0.5 + vx*dt, 0.5 + vy*dt) = (0.6, 0.6).
  expected_center_x = 0.6
  expected_center_y = 0.6
  
  # Compute indices.
  idx_x, idx_y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
  
  depart_x, depart_y = compute_advection_indices(
      idx_x, idx_y, vx, vy, dt, dx, dy, nx, ny
  )
  
  # Advect.
  f_advected = advect_2d(f0, depart_x, depart_y, order=1, mode='wrap')
  
  # Find peak of advected field.
  max_idx = jnp.unravel_index(jnp.argmax(f_advected), f_advected.shape)
  peak_x = x[max_idx[0]]
  peak_y = y[max_idx[1]]
  
  assert jnp.isclose(peak_x, expected_center_x, atol=4*dx)
  assert jnp.isclose(peak_y, expected_center_y, atol=4*dy)
  
  # Check conservation (should be roughly conserved for periodic).
  assert jnp.isclose(jnp.sum(f0), jnp.sum(f_advected), rtol=5e-2)

if __name__ == "__main__":
  test_advection_constant_velocity()
