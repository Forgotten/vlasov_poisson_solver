from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp
from .solver_types import Scalar

@dataclass(frozen=True)
class DomainConfig:
  """Configuration for the physical domain and grid resolution."""
  x_min: float = 0.0
  x_max: float = 1.0
  y_min: float = 0.0
  y_max: float = 1.0
  vx_min: float = -5.0
  vx_max: float = 5.0
  vy_min: float = -5.0
  vy_max: float = 5.0
  
  nx: int = 64
  ny: int = 64
  nvx: int = 64
  nvy: int = 64

  @property
  def dx(self) -> float:
    return (self.x_max - self.x_min) / self.nx

  @property
  def dy(self) -> float:
    return (self.y_max - self.y_min) / self.ny

  @property
  def dvx(self) -> float:
    return (self.vx_max - self.vx_min) / self.nvx

  @property
  def dvy(self) -> float:
    return (self.vy_max - self.vy_min) / self.nvy

@dataclass(frozen=True)
class PhysicsConfig:
  """Configuration for physical parameters."""
  dt: float = 0.1
  final_time: float = 1.0
  # Add other physical constants if needed (e.g., charge, mass).

@dataclass(frozen=True)
class SolverConfig:
  """Configuration for solver parameters."""
  interpolation_order: int = 1  # 1: linear (JAX limitation).
  cfl_safety: float = 0.5
