import jax
import jax.numpy as jnp
from functools import partial
from .solver_types import Array

@partial(jax.jit, static_argnames=['order', 'mode'])
def advect_2d(
    f: Array,
    coords_x: Array,
    coords_y: Array,
    order: int = 1,
    mode: str = 'wrap'
) -> Array:
  """
  Advects a 2D field f using interpolation at coordinates (coords_x, coords_y).
  
  Args:
    f: 2D field to advect, shape (nx, ny).
    coords_x: Coordinates to interpolate at along x-axis (index space).
    coords_y: Coordinates to interpolate at along y-axis (index space).
    order: Interpolation order (0-5). 1 is linear, 3 is cubic.
    mode: Boundary condition mode ('wrap', 'constant', 'nearest', 'mirror').
          'wrap' corresponds to periodic boundary conditions.
          
  Returns:
    Advected field evaluated at (coords_x, coords_y).
  """
  # jax.scipy.ndimage.map_coordinates expects coordinates as (ndim, points)
  # We flatten the coordinates to (2, nx*ny) and then reshape back
  
  coords = jnp.stack([coords_x, coords_y], axis=0)
  
  interpolated = jax.scipy.ndimage.map_coordinates(
      f, coords, order=order, mode=mode
  )
  
  return interpolated

def compute_advection_indices(
    x_indices: Array,
    y_indices: Array,
    velocity_x: Array,
    velocity_y: Array,
    dt: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int
) -> tuple[Array, Array]:
  """
  Computes the departure points (indices) for semi-Lagrangian advection.
  x_depart = x - v * dt
  
  Args:
    x_indices: Grid indices for x, shape (nx, ny).
    y_indices: Grid indices for y, shape (nx, ny).
    velocity_x: Velocity field in x direction, shape (nx, ny) or scalar.
    velocity_y: Velocity field in y direction, shape (nx, ny) or scalar.
    dt: Time step.
    dx: Grid spacing in x.
    dy: Grid spacing in y.
    nx: Number of points in x.
    ny: Number of points in y.
    
  Returns:
    (depart_x, depart_y): Departure indices.
  """
  # Displacement in physical units
  disp_x = velocity_x * dt
  disp_y = velocity_y * dt
  
  # Displacement in index units
  disp_idx_x = disp_x / dx
  disp_idx_y = disp_y / dy
  
  # Departure indices
  depart_x = x_indices - disp_idx_x
  depart_y = y_indices - disp_idx_y
  
  # We don't wrap here because map_coordinates handles 'mode'
  # But for 'wrap' mode in map_coordinates, it expects indices, and it wraps them
  # So we can just pass the raw indices.
  
  return depart_x, depart_y
