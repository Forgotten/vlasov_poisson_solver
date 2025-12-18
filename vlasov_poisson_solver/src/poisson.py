import jax
import jax.numpy as jnp
from .solver_types import SpatialField, Array
from .config import DomainConfig

def solve_poisson_fft(
    rho: SpatialField,
    config: DomainConfig
) -> tuple[SpatialField, tuple[SpatialField, SpatialField]]:
  """
  Solves the Poisson equation -Delta phi = rho using FFT with periodic BCs.
  
  Args:
    rho: Charge density rho(x, y).
    config: Domain configuration.
      
  Returns:
    phi: Electrostatic potential phi(x, y).
    E: Electric field E(x, y) = -grad phi, as a tuple (Ex, Ey).
  """
  nx, ny = config.nx, config.ny
  lx = config.x_max - config.x_min
  ly = config.y_max - config.y_min
  
  # Wavenumbers
  kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=lx/nx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=ly/ny)
  
  kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
  
  # FFT of density
  rho_hat = jnp.fft.fft2(rho)
  
  # Solve in Fourier space: -k^2 phi_hat = rho_hat  => phi_hat = rho_hat / k^2
  k2 = kx_grid**2 + ky_grid**2
  
  # Avoid division by zero at k=0 (mean potential is arbitrary, set to 0)
  k2 = jnp.where(k2 == 0, 1.0, k2)
  phi_hat = rho_hat / k2
  phi_hat = jnp.where((kx_grid == 0) & (ky_grid == 0), 0.0, phi_hat)
  
  # Inverse FFT to get potential
  phi = jnp.real(jnp.fft.ifft2(phi_hat))
  
  # Electric field: E = -grad phi
  # In Fourier space: E_hat = -i k phi_hat
  ex_hat = -1j * kx_grid * phi_hat
  ey_hat = -1j * ky_grid * phi_hat
  
  ex = jnp.real(jnp.fft.ifft2(ex_hat))
  ey = jnp.real(jnp.fft.ifft2(ey_hat))
  
  return phi, (ex, ey)
