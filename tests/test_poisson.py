import pytest
import jax.numpy as jnp
from vlasov_poisson_solver import DomainConfig, solve_poisson_fft

def test_poisson_solver_sine_wave():
  """Test Poisson solver with a simple sine wave source."""
  # Setup.
  config = DomainConfig(
      x_min=0.0, x_max=2*jnp.pi,
      y_min=0.0, y_max=2*jnp.pi,
      nx=32, ny=32
  )
  
  # Analytical solution: phi = sin(x) + cos(y).
  # -Delta phi = -(-sin(x) - cos(y)) = sin(x) + cos(y) = rho.
  # So if rho = sin(x) + cos(y), we should recover phi (up to a constant).
  
  x = jnp.linspace(config.x_min, config.x_max, config.nx, endpoint=False)
  y = jnp.linspace(config.y_min, config.y_max, config.ny, endpoint=False)
  x_grid, y_grid = jnp.meshgrid(x, y, indexing='ij')
  
  rho = jnp.sin(x_grid) + jnp.cos(y_grid)
  
  # Solve.
  phi, (ex, ey) = solve_poisson_fft(rho, config)
  
  # Check potential (up to constant shift).
  expected_phi = jnp.sin(x_grid) + jnp.cos(y_grid)
  # Normalize by subtracting mean.
  phi = phi - jnp.mean(phi)
  expected_phi = expected_phi - jnp.mean(expected_phi)
  
  assert jnp.allclose(phi, expected_phi, atol=1e-3)
  
  # Check Electric Field E = -grad phi.
  # Ex = -cos(x), Ey = sin(y).
  expected_ex = -jnp.cos(x_grid)
  expected_ey = jnp.sin(y_grid)
  
  assert jnp.allclose(ex, expected_ex, atol=1e-3)
  assert jnp.allclose(ey, expected_ey, atol=1e-3)

if __name__ == "__main__":
  test_poisson_solver_sine_wave()
