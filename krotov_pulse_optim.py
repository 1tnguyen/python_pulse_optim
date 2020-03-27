# XACC Pulse Optimizer plugin implementation that can be used by IR transformation service.
# IMPORTANT: this file must be copied/installed to XACC_INSTALL_DIR/py-plugins
# e.g. $HOME/.xacc/py-plugins

import xacc
from pelix.ipopo.decorators import ComponentFactory, Property, Requires, Provides, \
    Validate, Invalidate, Instantiate

# Import packages required for krotov pulse optimization
import qutip
import krotov
import numpy as np
import scipy


tlist = np.linspace(0, 10, 1000)

def eps0(t, args):
    T = tlist[-1]
    return 4 * np.exp(-40.0 * (t / T - 0.5) ** 2)

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
# Pulse optimization plugin
@ComponentFactory("py_pulse_optimizer_factory")
@Provides("optimizer")
@Property("_optimizer", "optimizer", "krotov_optimizer")
@Property("_name", "name", "krotov_optimizer")
@Instantiate("krotovopt_instance")
class MlPulseOptimizer(xacc.Optimizer):
    def __init__(self):
        xacc.Optimizer.__init__(self)
        self.options = {}
        self.pulseOpts = None

    def name(self):
        return 'krotov_optimizer'
    
    def construct_hamiltonian(self):
        """Construct QuTip Hamiltonian object [H0, [H1, f1] etc.]
        from H0 and H_ops
        """
        
        Ec = 0.386
        EjEc = 45
        nstates = 8 
        ng = 0.0        

        Ej = EjEc * Ec
        n = np.arange(-nstates, nstates + 1)
        up = np.diag(np.ones(2 * nstates), k=-1)
        do = up.T
        H0 = qutip.Qobj(np.diag(4 * Ec * (n - ng) ** 2) - Ej * (up + do) / 2.0)
        H1 = qutip.Qobj(-2 * np.diag(n))

        return [H0, [H1, eps0]]
    
    def setOptions(self, opts):
        self.pulseOpts = opts
        # These params are always present in the option map/dict
        # when using the IR transformation service.
        if 'dimension' in opts:
            self.dimension = opts['dimension']
        # Target unitary matrix
        if 'target-U' in opts:
            self.targetU = opts['target-U']
        # Static Hamiltonian
        if 'static-H' in opts:
            self.H0 = opts['static-H']
        # Control Hamiltonian (list)
        if 'control-H' in opts:
            self.Hops = opts['control-H']
        # Max time horizon
        if 'max-time' in opts:
            self.tMax = opts['max-time']
        # Pulse sample dt (number of samples over the time horizon)
        if 'dt' in opts:
            self.dt = opts['dt']
        if 'hamiltonian-json' in opts:
            self.hamJson = opts['hamiltonian-json']
        # Note: if the method requires specific parameters,
        # we can require those params being specified in the IR transformation options
        # then propagate to here. 

    # This is main entry point that the high-level
    # IR transformation service will call.
    # This needs to return the pair (opt-val, pulses)
    # where opt-val is a floating point number (final value of the cost function);
    # pulses is a single array of all control pulses (one for each control-H term)
    # (appending one pulse array after another).
    def optimize(self):
        # TODO: we can now call any Python lib to
        # perform pulse optimization (marshalling the options/parameters if required)
        # For example, one can use Qutip pulse optimization:
        # Notes about data types: 
        # - targerU: flatten (row-by-row) U matrix into a 1-D array of complex numbers
        # - H0: string-type representation of the static Hamiltonian:
        # e.g.: 0.123 Z0Z1
        # - Hops: array of strings represent terms on the Hamiltonian which can be controlled.
        # Depending on the specific library we use for pulse optimization,
        # we may need to marshal these data types accordingly.
        # Run the optimization
        
        H = self.construct_hamiltonian()
        psi0, psi1 = logical_basis(H)
        
        
        pulse_options = {H[1][1]: dict(lambda_a=1, update_shape=S)}

        objectives = krotov.gate_objectives(
            # TODO:
            # gate = target-U
            basis_states=[psi0, psi1], gate=qutip.operators.sigmax(), H=H
        )
        
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
        




        # Total time, T, of control pulse
        #nbSamples = (int)(self.tMax/self.dt)
        # Optimized pulse ??
        pulse = opt_result.optimized_controls[0]
        return (0.0, pulse)