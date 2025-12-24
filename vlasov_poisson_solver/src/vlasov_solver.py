import jax
import jax.numpy as jnp
from functools import partial
from .solver_types import PhaseSpaceField, SpatialField, VelocityField, Array
from .config import DomainConfig, PhysicsConfig, SolverConfig
from .poisson import solve_poisson_fft
from .advection import advect_2d, compute_advection_indices
from .sharding import get_phase_space_sharding
from jax.sharding import Mesh
import jax.lax

class VlasovPoissonSolver:
  def __init__(
      self,
      domain_config: DomainConfig,
      physics_config: PhysicsConfig,
      solver_config: SolverConfig,
      mesh: Mesh | None = None
  ):
    self.domain_config = domain_config
    self.physics_config = physics_config
    self.solver_config = solver_config
    self.mesh = mesh
    
    # Grid setup.
    self.x = jnp.linspace(
        domain_config.x_min, domain_config.x_max, domain_config.nx,
        endpoint=False
    )
    self.y = jnp.linspace(
        domain_config.y_min, domain_config.y_max, domain_config.ny,
        endpoint=False
    )
    self.vx = jnp.linspace(
        domain_config.vx_min, domain_config.vx_max, domain_config.nvx,
        endpoint=False
    )
    self.vy = jnp.linspace(
        domain_config.vy_min, domain_config.vy_max, domain_config.nvy,
        endpoint=False
    )
    
    # Meshgrids for indices (used in advection).
    self.idx_x, self.idx_y = jnp.meshgrid(
        jnp.arange(domain_config.nx), jnp.arange(domain_config.ny),
        indexing='ij'
    )
    self.idx_vx, self.idx_vy = jnp.meshgrid(
        jnp.arange(domain_config.nvx), jnp.arange(domain_config.nvy),
        indexing='ij'
    )

  @partial(jax.jit, static_argnums=(0,))
  def step(self, f: PhaseSpaceField, t: float) -> PhaseSpaceField:
    """
    Performs one time step using Strang splitting.
    Splitting: Advect X (dt/2) -> Advect V (dt) -> Advect X (dt/2)
    
    Args:
      f: Distribution function f(x, y, vx, vy) at time t.
      t: Current time.
      
    Returns:
      f at time t + dt.
    """
    dt = self.physics_config.dt
    
    # Apply sharding constraint to input if mesh is present.
    if self.mesh is not None:
      sharding = get_phase_space_sharding(self.mesh)
      f = jax.lax.with_sharding_constraint(f, sharding)
    
    # 1. Advect in X for dt/2.
    f = self._advect_x(f, dt / 2)
    
    # 2. Solve Poisson to get E.
    rho = self._compute_rho(f)
    _, (ex, ey) = solve_poisson_fft(rho, self.domain_config)
    
    # 3. Advect in V for dt.
    f = self._advect_v(f, ex, ey, dt)
    
    # 4. Advect in X for dt/2.
    f = self._advect_x(f, dt / 2)
    
    return f

  def _compute_rho(self, f: PhaseSpaceField) -> SpatialField:
    """Computes charge density rho(x, y) = integral f dvx dvy - background."""
    # Integrate over velocity space.
    # f has shape (nx, ny, nvx, nvy).
    # Sum over last two axes and multiply by dvx * dvy.
    rho = jnp.sum(f, axis=(2, 3)) * self.domain_config.dvx * self.domain_config.dvy
    
    # Assume neutralizing background (mean 0).
    rho = rho - jnp.mean(rho)
    return rho

  def _advect_x(self, f: PhaseSpaceField, dt: float) -> PhaseSpaceField:
    """Advects f in spatial coordinates: x' = x - v * dt."""
    # We need to advect each velocity slice (vx, vy) with velocity (vx, vy).
    # This can be vectorized.
    
    # V has shape (nvx, nvy).
    vx_grid, vy_grid = jnp.meshgrid(self.vx, self.vy, indexing='ij')
    
    # We want to map this over the velocity axes (2, 3) of f.
    # But advect_2d expects 2D fields.
    # We can use jax.vmap to batch over nvx, nvy.
    
    # Define a function that takes a single 2D spatial slice and its velocity.
    def advect_slice(f_slice, vx, vy):
      depart_x, depart_y = compute_advection_indices(
          self.idx_x, self.idx_y, vx, vy, dt,
          self.domain_config.dx, self.domain_config.dy,
          self.domain_config.nx, self.domain_config.ny
      )
      return advect_2d(
          f_slice, depart_x, depart_y,
          order=self.solver_config.interpolation_order
      )
      
    # vmap over vx, vy (axes 0, 1 of VX, VY) and f (axes 2, 3).
    # f: (nx, ny, nvx, nvy) -> move axes to (nvx, nvy, nx, ny) for easier vmap?
    # Or just specify in_axes.
    
    # Let's transpose f to (nvx, nvy, nx, ny).
    f_transposed = jnp.transpose(f, (2, 3, 0, 1))
    
    # vmap over the first two axes (nvx, nvy).
    # We can flatten nvx, nvy to a single batch dimension for a single vmap.
    
    f_flat = f_transposed.reshape(
        -1, self.domain_config.nx, self.domain_config.ny
    )
    vx_flat = vx_grid.flatten()
    vy_flat = vy_grid.flatten()
    
    advect_batch = jax.vmap(advect_slice, in_axes=(0, 0, 0))
    f_advected_flat = advect_batch(f_flat, vx_flat, vy_flat)
    
    # Reshape back and transpose.
    f_advected = f_advected_flat.reshape(
        self.domain_config.nvx, self.domain_config.nvy,
        self.domain_config.nx, self.domain_config.ny
    )
    return jnp.transpose(f_advected, (2, 3, 0, 1))

  def _advect_v(
      self,
      f: PhaseSpaceField,
      ex: SpatialField,
      ey: SpatialField,
      dt: float
  ) -> PhaseSpaceField:
    """Advects f in velocity coordinates: v' = v - a * dt = v - (q/m)E * dt."""
    # Assume q/m = -1 (electrons) -> a = -E -> v' = v + E * dt.
    # Or q/m = 1? Let's assume standard Vlasov-Poisson for electrons.
    # Equation: df/dt + v.grad_x f - E.grad_v f = 0.
    # Characteristics: dx/dt = v, dv/dt = -E.
    # So departure point: v_depart = v - (-E) * dt = v + E * dt.
    
    # E depends on (x, y). So for each spatial point (x, y), we have a different shift in v.
    
    # Define a function that takes a single 2D velocity slice and the local E.
    def advect_slice(f_slice, local_ex, local_ey):
      # f_slice: (nvx, nvy).
      # local_ex, local_ey: scalars.
      
      # Velocity shift is E * dt.
      # But wait, compute_advection_indices computes x - v*dt.
      # Here we want v - a*dt = v - (-E)*dt = v + E*dt.
      # So "velocity" in velocity space is -E.
      
      # Let's use compute_advection_indices with "velocity" = -E.
      # depart_v = v - (-E) * dt = v + E * dt.
      
      depart_vx, depart_vy = compute_advection_indices(
          self.idx_vx, self.idx_vy, -local_ex, -local_ey, dt,
          self.domain_config.dvx, self.domain_config.dvy,
          self.domain_config.nvx, self.domain_config.nvy
      )
      return advect_2d(
          f_slice, depart_vx, depart_vy,
          order=self.solver_config.interpolation_order
      )
      
    # vmap over x, y (axes 0, 1 of f and Ex, Ey).
    
    # f: (nx, ny, nvx, nvy) -> flatten first two dims.
    f_flat = f.reshape(-1, self.domain_config.nvx, self.domain_config.nvy)
    ex_flat = ex.flatten()
    ey_flat = ey.flatten()
    
    advect_batch = jax.vmap(advect_slice, in_axes=(0, 0, 0))
    f_advected_flat = advect_batch(f_flat, ex_flat, ey_flat)
    
    # Reshape back.
    return f_advected_flat.reshape(
        self.domain_config.nx, self.domain_config.ny,
        self.domain_config.nvx, self.domain_config.nvy
    )
