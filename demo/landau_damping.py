import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from vlasov_poisson_solver import (
    DomainConfig, PhysicsConfig, SolverConfig, VlasovPoissonSolver, create_mesh,
    solve_poisson_fft
)

def main():
  print("Initializing Landau Damping Simulation...")
  
  # 1. Configuration.
  domain_config = DomainConfig(
      x_min=0.0, x_max=4*jnp.pi,
      y_min=0.0, y_max=4*jnp.pi,
      vx_min=-6.0, vx_max=6.0,
      vy_min=-6.0, vy_max=6.0,
      nx=64, ny=64, nvx=64, nvy=64
  )
  physics_config = PhysicsConfig(dt=0.1, final_time=5.0)
  solver_config = SolverConfig(interpolation_order=1)
  
  print(f"Domain: {domain_config.nx}x{domain_config.ny} spatial, "
        f"{domain_config.nvx}x{domain_config.nvy} velocity")
  print(f"Physics: dt={physics_config.dt}, T_final={physics_config.final_time}")

  # 2. Create Solver.
  try:
    mesh = create_mesh((1, 1), ('x', 'y'))
    print("Solver initialized with mesh.")
  except Exception as e:
    print(f"Could not create mesh: {e}. Running without explicit sharding.")
    mesh = None
    
  solver = VlasovPoissonSolver(
      domain_config, physics_config, solver_config, mesh=mesh
  )
  
  # 3. Initial Condition: Landau Damping.
  print("Setting up initial conditions...")
  alpha = 0.01
  kx = 0.5
  ky = 0.5
  
  x_grid, y_grid = jnp.meshgrid(solver.x, solver.y, indexing='ij')
  vx_grid, vy_grid = jnp.meshgrid(solver.vx, solver.vy, indexing='ij')
  
  # Expand dims for broadcasting.
  x_grid = x_grid[:, :, None, None]
  y_grid = y_grid[:, :, None, None]
  vx_grid = vx_grid[None, None, :, :]
  vy_grid = vy_grid[None, None, :, :]
  
  spatial_perturbation = 1.0 + alpha * (jnp.cos(kx * x_grid) + jnp.cos(ky * y_grid))
  velocity_dist = (1.0 / (2 * jnp.pi)) * jnp.exp(-(vx_grid**2 + vy_grid**2) / 2)
  
  f = spatial_perturbation * velocity_dist
  f_initial = f  # Keep reference to initial state.
  
  initial_mass = jnp.sum(f)
  print(f"Initial Mass: {initial_mass:.6f}")
  
  # 4. Time Loop using lax.scan.
  num_steps = int(physics_config.final_time / physics_config.dt)
  
  print(f"Starting simulation for {num_steps} steps using lax.scan...")
  
  def step_fn(carry, _):
    f_curr, t_curr = carry
    f_next = solver.step(f_curr, t_curr)
    t_next = t_curr + physics_config.dt
    
    # Calculate electric field energy for benchmarking.
    # We need to re-compute E from f_curr to get E(t).
    # Note: solver.step does this internally, but we need it here for diagnostics.
    # This adds some overhead but is fine for a demo.
    rho = solver._compute_rho(f_curr)
    _, (ex, ey) = jax.lax.stop_gradient(
        solve_poisson_fft(rho, solver.domain_config)
    )
    # Energy = 0.5 * integral (Ex^2 + Ey^2) dx dy.
    # Sum * dx * dy.
    dv = domain_config.dx * domain_config.dy
    electric_energy = 0.5 * jnp.sum(ex**2 + ey**2) * dv
    
    mass_err = jnp.abs(jnp.sum(f_next) - initial_mass) / initial_mass
    return (f_next, t_next), (mass_err, electric_energy)

  start_time = time.time()
  
  # Run scan.
  (f_final, t_final), (mass_errors, electric_energies) = jax.lax.scan(
      step_fn, (f, 0.0), None, length=num_steps
  )
  
  # Block until ready.
  f_final.block_until_ready()
  
  end_time = time.time()
  print("-" * 35)
  print(f"Simulation complete in {end_time - start_time:.2f} seconds.")
  
  final_mass = jnp.sum(f_final)
  final_error = jnp.abs(final_mass - initial_mass) / initial_mass
  print(f"Final Mass Error: {final_error:.2e}")
  
  # --- Benchmarking ---
  print("\n--- Landau Damping Benchmark ---")
  # Theoretical decay rate for k=0.5 is gamma approx -0.1533.
  # E(t) ~ exp(gamma * t) -> Energy ~ E^2 ~ exp(2 * gamma * t).
  # So log(Energy) ~ 2 * gamma * t + C.
  # Slope of log(Energy) vs t should be 2 * gamma.
  
  # Time array.
  times = jnp.arange(num_steps) * physics_config.dt
  
  # Fit line to log(energy).
  # Ignore first few steps (transient) and very late steps (noise floor).
  # Let's fit from t=0.5 to t=3.0.
  start_idx = int(0.5 / physics_config.dt)
  end_idx = int(3.0 / physics_config.dt)
  
  fit_times = times[start_idx:end_idx]
  fit_log_energy = jnp.log(electric_energies[start_idx:end_idx])
  
  # Linear regression: y = mx + c.
  # m = (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - sum(x)^2).
  N = len(fit_times)
  sum_x = jnp.sum(fit_times)
  sum_y = jnp.sum(fit_log_energy)
  sum_xy = jnp.sum(fit_times * fit_log_energy)
  sum_xx = jnp.sum(fit_times**2)
  
  slope = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x**2)
  
  fitted_gamma = slope / 2.0
  theoretical_gamma = -0.1533
  
  print(f"Fitted Decay Rate (gamma): {fitted_gamma:.4f}")
  print(f"Theoretical Decay Rate:    {theoretical_gamma:.4f}")
  print(f"Relative Error:            {abs((fitted_gamma - theoretical_gamma)/theoretical_gamma)*100:.2f}%")
  
  # Plot Energy.
  plt.figure(figsize=(6, 4))
  plt.semilogy(times, electric_energies, label='Simulation')
  # Plot fitted line.
  plt.semilogy(fit_times, jnp.exp(slope * fit_times + (sum_y - slope * sum_x)/N), 
               'r--', label=f'Fit (gamma={fitted_gamma:.3f})')
  plt.xlabel('Time')
  plt.ylabel('Electric Field Energy')
  plt.title('Landau Damping: Energy Decay')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig('landau_energy.png', dpi=150)
  print("Energy plot saved to landau_energy.png")

  # 5. Plotting.
  print("\nPlotting densities...")
  
  # Compute spatial density: rho(x, y) = integral f dvx dvy.
  dv = domain_config.dvx * domain_config.dvy
  rho_initial = jnp.sum(f_initial, axis=(2, 3)) * dv
  rho_final = jnp.sum(f_final, axis=(2, 3)) * dv
  
  # Create a figure with two subplots.
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  
  # Initial Density.
  im0 = axes[0].imshow(
      rho_initial.T, 
      extent=[solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]],
      origin='lower',
      cmap='viridis'
  )
  axes[0].set_title('Initial Density (t=0)')
  axes[0].set_xlabel('x')
  axes[0].set_ylabel('y')
  fig.colorbar(im0, ax=axes[0])
  
  # Final Density.
  im1 = axes[1].imshow(
      rho_final.T, 
      extent=[solver.x[0], solver.x[-1], solver.y[0], solver.y[-1]],
      origin='lower',
      cmap='viridis'
  )
  axes[1].set_title(f'Final Density (t={t_final:.1f})')
  axes[1].set_xlabel('x')
  axes[1].set_ylabel('y')
  fig.colorbar(im1, ax=axes[1])
  
  plt.tight_layout()
  
  output_filename = 'landau_damping_2d_results.png'
  plt.savefig(output_filename, dpi=150)
  print(f"Density plot saved to {output_filename}")

if __name__ == "__main__":
  main()
