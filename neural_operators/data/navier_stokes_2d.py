"""Pseudo-spectral solver for the 2D incompressible Navier-Stokes equations in
vorticity-streamfunction form on the periodic torus [0,1)^2, matching the data
generation procedure described in the paper (Section 7.3), following the method of
Chandler & Kerswell (2013).

Governing equations (vorticity form):
    dw/dt + u . grad(w) = visc * Delta(w) + f
    u = (dpsi/dy, -dpsi/dx),   -Delta(psi) = w   (Biot-Savart law)

On a periodic domain, -Delta is diagonal in the Fourier basis, so psi_hat = w_hat / |k|^2
and all spatial derivatives become multiplications by i*k in Fourier space. We advance
the nonlinear advection term explicitly and the linear viscous term implicitly
(Crank-Nicolson), which avoids the severe time-step restriction that explicit treatment
of viscosity would otherwise require.
"""
import math

import torch


def _wavenumbers(n, device):
    k_max = n // 2
    k = torch.cat(
        (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0
    ).repeat(n, 1)
    k_y = k[..., : k_max + 1]
    k_x = k.transpose(0, 1)[..., : k_max + 1]
    return k_x, k_y


def solve_navier_stokes_2d(w0, forcing, visc, T, delta_t=1e-3, record_steps=1):
    """Solve the 2D vorticity Navier-Stokes equation forward in time.

    Args:
        w0: (batch, n, n) initial vorticity field.
        forcing: (n, n) or (batch, n, n) forcing function f(x), constant in time.
        visc: viscosity (1/Reynolds number for unit-scale velocity/length).
        T: total time to integrate.
        delta_t: internal integration time step.
        record_steps: number of (equally spaced in time) snapshots to save, not
            counting the initial condition.

    Returns:
        sol: (batch, n, n, record_steps) vorticity snapshots.
        sol_t: (record_steps,) the times at which snapshots were taken.
    """
    device = w0.device
    n = w0.shape[-1]
    k_x, k_y = _wavenumbers(n, device)

    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(forcing)
    if f_h.dim() < w_h.dim():
        f_h = f_h.unsqueeze(0)

    laplacian = (k_x ** 2 + k_y ** 2) * (4 * math.pi ** 2)
    laplacian[0, 0] = 1.0  # avoid division by zero for the mean mode (psi mean is arbitrary)

    k_max = n // 2
    dealias = (
        (k_x.abs() <= (2.0 / 3.0) * k_max) & (k_y.abs() <= (2.0 / 3.0) * k_max)
    ).float().unsqueeze(0)

    n_steps = math.ceil(T / delta_t)
    steps_per_record = max(1, n_steps // record_steps)

    sol = torch.zeros(*w0.shape, record_steps, device=device)
    sol_t = torch.zeros(record_steps, device=device)

    t = 0.0
    record_idx = 0
    for step in range(n_steps):
        psi_h = w_h / laplacian  # Biot-Savart: stream function from vorticity

        u_h = 2j * math.pi * k_y * psi_h    # u  =  d(psi)/dy
        v_h = -2j * math.pi * k_x * psi_h   # v  = -d(psi)/dx
        wx_h = 2j * math.pi * k_x * w_h
        wy_h = 2j * math.pi * k_y * w_h

        u = torch.fft.irfft2(u_h, s=(n, n))
        v = torch.fft.irfft2(v_h, s=(n, n))
        wx = torch.fft.irfft2(wx_h, s=(n, n))
        wy = torch.fft.irfft2(wy_h, s=(n, n))

        advection = u * wx + v * wy
        advection_h = dealias * torch.fft.rfft2(advection)

        # Crank-Nicolson on the viscous (linear) term, explicit Euler on advection + forcing.
        w_h = (
            (1.0 - 0.5 * delta_t * visc * laplacian) * w_h
            - delta_t * advection_h
            + delta_t * f_h
        ) / (1.0 + 0.5 * delta_t * visc * laplacian)

        t += delta_t

        if (step + 1) % steps_per_record == 0 and record_idx < record_steps:
            sol[..., record_idx] = torch.fft.irfft2(w_h, s=(n, n))
            sol_t[record_idx] = t
            record_idx += 1

    return sol, sol_t


def default_forcing(n, device=None):
    """Standard forcing used in the paper's Navier-Stokes benchmark: a fixed sinusoidal
    forcing that continuously injects energy, producing statistically steady turbulence.
    f(x, y) = 0.1 * (sin(2*pi*(x+y)) + cos(2*pi*(x+y)))
    """
    grid = torch.linspace(0, 1, n + 1, device=device)[:-1]
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    return 0.1 * (torch.sin(2 * math.pi * (x + y)) + torch.cos(2 * math.pi * (x + y)))
