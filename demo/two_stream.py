import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from vlasov_poisson_solver import (
    DomainConfig, PhysicsConfig, SolverConfig, VlasovPoissonSolver, create_mesh,
    solve_poisson_fft
)

def main():
  print("Initializing two-stream instability simulation...")
  
  # Setting up configuration.
  # We need a longer domain or specific k to trigger instability.
  # Instability typically occurs for k < 1 (in normalized units).
  domain_config = DomainConfig(
      x_min=0.0, x_max=4*jnp.pi,
      y_min=0.0, y_max=4*jnp.pi,
      vx_min=-6.0, vx_max=6.0,
      vy_min=-6.0, vy_max=6.0,
      nx=64, ny=64, nvx=64, nvy=64
  )
  # Instability takes time to grow.
  physics_config = PhysicsConfig(dt=0.1, final_time=20.0)
  solver_config = SolverConfig(interpolation_order=1)
  
  print(f"Domain: {domain_config.nx}x{domain_config.ny} spatial, "
        f"{domain_config.nvx}x{domain_config.nvy} velocity")
  print(f"Physics: dt={physics_config.dt}, T_final={physics_config.final_time}")

  # Setting up solver.
  try:
    mesh = create_mesh((1, 1), ('x', 'y'))
    print("Solver initialized with mesh.")
  except Exception as e:
    print(f"Could not create mesh: {e}. Running without explicit sharding.")
    mesh = None
    
  solver = VlasovPoissonSolver(
      domain_config, physics_config, solver_config, mesh=mesh
  )
  
  # Setting up initial conditions.
  print("Setting up initial conditions...")
  alpha = 0.01
  kx = 0.5
  ky = 0.5
  v0 = 2.4  # Beam velocity.
  
  x_grid, y_grid = jnp.meshgrid(solver.x, solver.y, indexing='ij')
  vx_grid, vy_grid = jnp.meshgrid(solver.vx, solver.vy, indexing='ij')
  
  # Expand dims for broadcasting.
  x_grid = x_grid[:, :, None, None]
  y_grid = y_grid[:, :, None, None]
  vx_grid = vx_grid[None, None, :, :]
  vy_grid = vy_grid[None, None, :, :]
  
  # Counter-streaming beams, formula is given by:
  # f_v = 1/(2*sqrt(2pi)) * (exp(-(v-v0)^2/2) + exp(-(v+v0)^2/2)).
  # Standard 1D two-stream is in vx so we keep vy Maxwellian.
  
  f_vx = (1.0 / (2 * jnp.sqrt(2 * jnp.pi))) * (
      jnp.exp(-(vx_grid - v0)**2 / 2) + jnp.exp(-(vx_grid + v0)**2 / 2)
  )
  # Maxwellian distribution in vy.
  f_vy = (1.0 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-vy_grid**2 / 2)
  
  velocity_dist = f_vx * f_vy
  
  spatial_perturbation = 1.0 + alpha * (jnp.cos(kx * x_grid) + jnp.cos(ky * y_grid))
  
  f = spatial_perturbation * velocity_dist
  f_initial = f
  
  initial_mass = jnp.sum(f)
  print(f"Initial Mass: {initial_mass:.6f}")
  
  # Setting up time loop.
  num_steps = int(physics_config.final_time / physics_config.dt)
  
  print(f"Starting simulation for {num_steps} steps...")
  
  def step_fn(carry, _):
    f_curr, t_curr = carry
    f_next = solver.step(f_curr, t_curr)
    t_next = t_curr + physics_config.dt
    
    # Calculate electric field energy.
    rho = solver._compute_rho(f_curr)
    _, (ex, ey) = jax.lax.stop_gradient(
        solve_poisson_fft(rho, solver.domain_config)
    )
    dv = domain_config.dx * domain_config.dy
    electric_energy = 0.5 * jnp.sum(ex**2 + ey**2) * dv
    
    return (f_next, t_next), electric_energy

  # Running simulation and timing it.
  start_time = time.time()
  
  (f_final, t_final), electric_energies = jax.lax.scan(
      step_fn, (f, 0.0), None, length=num_steps
  )
  
  f_final.block_until_ready()
  end_time = time.time()
  print(f"Simulation complete in {end_time - start_time:.2f} seconds.")
  
  # Plotting results.
  print("Plotting results...")
  
  # Energy growth.
  times = jnp.arange(num_steps) * physics_config.dt
  plt.figure(figsize=(6, 4))
  plt.semilogy(times, electric_energies)
  plt.xlabel('Time')
  plt.ylabel('Electric field energy')
  plt.title('Two-stream instability: energy growth')
  plt.grid(True, alpha=0.3)
  plt.savefig('two_stream_energy.png', dpi=150)
  print("Energy plot saved to two_stream_energy.png")
  
  # Density plots.
  dv = domain_config.dvx * domain_config.dvy
  rho_initial = jnp.sum(f_initial, axis=(2, 3)) * dv
  rho_final = jnp.sum(f_final, axis=(2, 3)) * dv
  
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  
  im0 = axes[0].imshow(
      rho_initial.T, 
      extent=[solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]],
      origin='lower', cmap='viridis'
  )
  axes[0].set_title('Initial density')
  fig.colorbar(im0, ax=axes[0])
  
  im1 = axes[1].imshow(
      rho_final.T, 
      extent=[solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]],
      origin='lower', cmap='viridis'
  )
  axes[1].set_title(f'Final density (t={t_final:.1f})')
  fig.colorbar(im1, ax=axes[1])
  
  plt.tight_layout()
  plt.savefig('two_stream_density.png', dpi=150)
  print("Density plot saved to two_stream_density.png")

if __name__ == "__main__":
  main()
