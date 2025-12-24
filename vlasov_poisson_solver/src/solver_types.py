import jax
import jax.numpy as jnp
from typing import Any, TypeAlias

# Use chex if available, otherwise fallback to Any or jax.Array.
try:
  import chex
  Array = chex.Array
  Scalar = chex.Scalar
except ImportError:
  Array = jax.Array
  Scalar = Any

# Type aliases for clarity.
# 4D array: (nx, ny, nvx, nvy).
PhaseSpaceField: TypeAlias = Array

# 2D array: (nx, ny).
SpatialField: TypeAlias = Array

# 2D array: (nvx, nvy).
VelocityField: TypeAlias = Array

# 1D array.
AxisArray: TypeAlias = Array
