import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from vlasov_poisson_solver import (
    DomainConfig, PhysicsConfig, SolverConfig, VlasovPoissonSolver, create_mesh,
    advect_2d, compute_advection_indices
)

class FreeStreamingSolver(VlasovPoissonSolver):
  """Solver for free streaming (E=0)."""
  
  def step(self, f, t):
    """Performs one time step of free streaming: f(x, v, t+dt) = f(x-v*dt, v, t)."""
    dt = self.physics_config.dt
    # Just advect in X for full dt.
    return self._advect_x(f, dt)

def main():
  print("Initializing Free Streaming Simulation...")
  
  # 1. Configuration.
  domain_config = DomainConfig(
      x_min=0.0, x_max=1.0,
      y_min=0.0, y_max=1.0,
      vx_min=-1.0, vx_max=1.0,
      vy_min=-1.0, vy_max=1.0,
      nx=64, ny=64, nvx=64, nvy=64
  )
  physics_config = PhysicsConfig(dt=0.01, final_time=1.0)
  solver_config = SolverConfig(interpolation_order=1)
  
  # 2. Create Solver.
  try:
    mesh = create_mesh((1, 1), ('x', 'y'))
  except:
    mesh = None
    
  solver = FreeStreamingSolver(
      domain_config, physics_config, solver_config, mesh=mesh
  )
  
  # 3. Initial Condition: Gaussian Pulse.
  print("Setting up initial conditions...")
  sigma = 0.1
  x_grid, y_grid = jnp.meshgrid(solver.x, solver.y, indexing='ij')
  vx_grid, vy_grid = jnp.meshgrid(solver.vx, solver.vy, indexing='ij')
  
  # Expand dims.
  x_grid = x_grid[:, :, None, None]
  y_grid = y_grid[:, :, None, None]
  vx_grid = vx_grid[None, None, :, :]
  vy_grid = vy_grid[None, None, :, :]
  
  # Gaussian in x, y, vx, vy centered at 0.5, 0.5, 0, 0.
  f0 = jnp.exp(
      -((x_grid - 0.5)**2 + (y_grid - 0.5)**2) / (2 * sigma**2)
      - (vx_grid**2 + vy_grid**2) / (2 * sigma**2)
  )
  
  f = f0
  
  # 4. Time Loop.
  num_steps = int(physics_config.final_time / physics_config.dt)
  print(f"Starting simulation for {num_steps} steps...")
  
  def step_fn(carry, _):
    f_curr, t_curr = carry
    f_next = solver.step(f_curr, t_curr)
    t_next = t_curr + physics_config.dt
    return (f_next, t_next), None

  start_time = time.time()
  (f_final, t_final), _ = jax.lax.scan(
      step_fn, (f, 0.0), None, length=num_steps
  )
  f_final.block_until_ready()
  end_time = time.time()
  print(f"Simulation complete in {end_time - start_time:.2f} seconds.")
  
  # 5. Analytical Solution Check.
  # f_analytical(x, v, t) = f0(x - v*t, v).
  # We need to compute f0 at shifted coordinates.
  # Since f0 is Gaussian, we can evaluate it directly.
  
  # Shifted coordinates (periodic wrapping).
  # x' = (x - vx*t) % L.
  L = domain_config.x_max - domain_config.x_min
  x_shifted = (x_grid - vx_grid * t_final) % L
  y_shifted = (y_grid - vy_grid * t_final) % L
  
  # Evaluate f0 at shifted coords.
  f_analytical = jnp.exp(
      -((x_shifted - 0.5)**2 + (y_shifted - 0.5)**2) / (2 * sigma**2)
      - (vx_grid**2 + vy_grid**2) / (2 * sigma**2)
  )
  
  # Error.
  error = jnp.abs(f_final - f_analytical)
  max_error = jnp.max(error)
  l2_error = jnp.sqrt(jnp.mean(error**2))
  
  print(f"Max Error: {max_error:.2e}")
  print(f"L2 Error:  {l2_error:.2e}")
  
  # 6. Plotting.
  print("Plotting results...")
  
  # Plot spatial density slice.
  dv = domain_config.dvx * domain_config.dvy
  rho_final = jnp.sum(f_final, axis=(2, 3)) * dv
  rho_analytical = jnp.sum(f_analytical, axis=(2, 3)) * dv
  
  mid_y = domain_config.ny // 2
  
  plt.figure(figsize=(8, 6))
  plt.plot(solver.x, rho_final[:, mid_y], label='Numerical', marker='o', markersize=4, linestyle='None')
  plt.plot(solver.x, rho_analytical[:, mid_y], label='Analytical', linestyle='--')
  plt.xlabel('x')
  plt.ylabel('Density')
  plt.title(f'Free Streaming: Density Slice at y={solver.y[mid_y]:.2f}, t={t_final:.1f}')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig('free_streaming_results.png', dpi=150)
  print("Plot saved to free_streaming_results.png")

if __name__ == "__main__":
  main()
