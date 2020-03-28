import sys
import os
import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
from scipy.fftpack import fft
from scipy.interpolate import interp1d

# Plot functions
def plot_pulse(pulse, tlist, xlimit=None):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, None) for t in tlist])
    ax.plot(tlist, pulse)
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('pulse amplitude')
    if xlimit is not None:
        ax.set_xlim(xlimit)
    plt.show(fig)

def plot_population(result):
    '''Representation of the expected values for the initial states'''
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0], label='0')
    ax.plot(result.times, result.expect[1], label='1')
    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    plt.show(fig)

def plot_spectrum(pulse, tlist, xlim=None):

    if callable(pulse):
        pulse = np.array([pulse(t, None) for t in tlist])

    dt = tlist[1] - tlist[0]
    n = len(tlist)

    w = np.fft.fftfreq(n, d=dt/(2.0*np.pi))
    # the factor 2π in the normalization means that
    # the spectrum is in units of angular frequency,
    # which is normally what we want

    spectrum = np.fft.fft(pulse) / n
    # normalizing the spectrum with n means that
    # the y-axis is independent of dt

    # we assume a real-valued pulse, so we throw away
    # the half of the spectrum with negative frequencies
    w = w[range(int(n / 2))]
    spectrum = np.abs(spectrum[range(int(n / 2))])

    fig, ax = plt.subplots()
    ax.plot(w, spectrum, '-o')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('amplitude (arb. units)')
    if xlim is not None:
        ax.set_xlim(*xlim)
    plt.show(fig)


# Optimization
tlist = np.linspace(0, 10, 1000)

def eps0(t, args):
    T = tlist[-1]
    return 4 * np.exp(-40.0 * (t / T - 0.5) ** 2)

def transmon_hamiltonian(Ec=0.386, EjEc=45, nstates=8, ng=0.0, T=10.0):
    """Transmon Hamiltonian

    Args:
        Ec: capacitive energy
        EjEc: ratio `Ej` / `Ec`
        nstates: defines the maximum and minimum states for the basis. The
            truncated basis will have a total of ``2*nstates + 1`` states

        ng: offset charge
        T: gate duration
    """

    Ej = EjEc * Ec
    n = np.arange(-nstates, nstates + 1)
    up = np.diag(np.ones(2 * nstates), k=-1)
    do = up.T
    H0 = qutip.Qobj(np.diag(4 * Ec * (n - ng) ** 2) - Ej * (up + do) / 2.0)
    H1 = qutip.Qobj(-2 * np.diag(n))
    return [H0, [H1, eps0]]

def logical_basis(H):
    H0 = H[0]
    eigenvals, eigenvecs = scipy.linalg.eig(H0.full())
    ndx = np.argsort(eigenvals.real)
    E = eigenvals[ndx].real
    V = eigenvecs[:, ndx]
    psi0 = qutip.Qobj(V[:, 0])
    psi1 = qutip.Qobj(V[:, 1])
    w01 = E[1] - E[0]  # Transition energy between states
    print("Energy of qubit transition is %.3f" % w01)
    return psi0, psi1

def S(t):
    """Scales the Krotov methods update of the pulse value at the time t"""
    return krotov.shapes.flattop(
        t, t_start=0.0, t_stop=10.0, t_rise=0.5, func='sinsq'
    )

H = transmon_hamiltonian()
psi0, psi1 = logical_basis(H)
proj0 = qutip.ket2dm(psi0)
proj1 = qutip.ket2dm(psi1)

objectives = krotov.gate_objectives(
    basis_states=[psi0, psi1], gate=qutip.operators.sigmax(), H=H
)

guess_dynamics = [
    objectives[x].mesolve(tlist, e_ops=[proj0, proj1]) for x in [0, 1]
]


plot_population(guess_dynamics[0])
plot_population(guess_dynamics[1])


pulse_options = {H[1][1]: dict(lambda_a=1, update_shape=S)}

opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options,
    tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_re,
    info_hook=krotov.info_hooks.print_table(
        J_T=krotov.functionals.J_T_re,
        show_g_a_int_per_pulse=True,
        unicode=False,
    ),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below(1e-3, name='J_T'),
        krotov.convergence.delta_below(1e-5),
        krotov.convergence.check_monotonic_error,
    ),
    iter_stop=5,
    parallel_map=(
        krotov.parallelization.parallel_map,
        krotov.parallelization.parallel_map,
        krotov.parallelization.parallel_map_fw_prop_step,
    ),
)

plot_pulse(opt_result.optimized_controls[0], tlist)
plot_spectrum(opt_result.optimized_controls[0], tlist, xlim=(0, 40))

opt_dynamics = [
    opt_result.optimized_objectives[x].mesolve(tlist, e_ops=[proj0, proj1])
    for x in [0, 1]
]

plot_population(opt_dynamics[0])
plot_population(opt_dynamics[1])