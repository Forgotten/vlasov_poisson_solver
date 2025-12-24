import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from .solver_types import Array

def create_mesh(mesh_shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh:
  """Creates a JAX device mesh.

  Args:
    mesh_shape: Shape of the mesh (e.g., (4, 2)).
    axis_names: Names of the axes (e.g., ('x', 'y')).

  Returns:
    JAX Mesh object.
  """
  devices = jax.devices()
  # Ensure we have enough devices.
  num_devices = len(devices)
  required_devices = np.prod(mesh_shape)
  if num_devices < required_devices:
    # Fallback to a smaller mesh or error?
    # For now, let's just use the first required_devices and reshape.
    # But if we don't have enough, we can't create the requested mesh.
    # For testing on CPU (usually 1 device), we might request (1, 1).
    pass
  
  # Reshape devices to match mesh_shape.
  # This assumes len(devices) >= prod(mesh_shape).
  mesh_devices = np.array(devices[:required_devices]).reshape(mesh_shape)
  mesh = Mesh(mesh_devices, axis_names)
  return mesh

def get_phase_space_sharding(mesh: Mesh) -> NamedSharding:
  """Returns sharding spec for phase space field (nx, ny, nvx, nvy).

  Strategy: Shard spatial dims (x, y), replicate velocity dims.
  """
  # x mapped to mesh axis 'x', y mapped to mesh axis 'y'.
  # vx, vy are replicated (None).
  return NamedSharding(mesh, PartitionSpec('x', 'y', None, None))

def get_spatial_field_sharding(mesh: Mesh) -> NamedSharding:
  """Returns sharding spec for spatial field (nx, ny)."""
  return NamedSharding(mesh, PartitionSpec('x', 'y'))
