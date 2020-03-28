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
import matplotlib
import matplotlib.pylab as plt
from datetime import *

# Pulse optimization plugin
@ComponentFactory("py_pulse_optimizer_factory")
@Provides("optimizer")
@Property("_optimizer", "optimizer", "krotov")
@Property("_name", "name", "krotov")
@Instantiate("krotovopt_instance")
class MlPulseOptimizer(xacc.Optimizer):
    def __init__(self):
        xacc.Optimizer.__init__(self)
        self.options = {}
        self.pulseOpts = None
        # Krotov parameter that determines overall magnitude of 
        # the respective field in each iteration 
        # (the smaller, the larger the update)
        self.lambda_a = 5
        # Scale factor of rise/fall edge
        self.rise_fall_coeff = 20.0
        # TODO: This should become an option for Krotov
        self.func_name = 'blackman'
    
    def name(self):
        return 'krotov'
    
    # This is *ONLY* for demo purposes: it can only handle *single* qubit Pauli 
    def parse_pauli_str(self, pauli_string):
        coeff_and_op = pauli_string.split()
        coeff = (float(coeff_and_op[-2]))
        pauliOp = coeff_and_op[-1]
        if pauliOp[0] == 'X':
            return coeff * qutip.operators.sigmax()
        if pauliOp[0] == 'Y':
            return coeff * qutip.operators.sigmay()
        if pauliOp[0] == 'Z':
            return coeff * qutip.operators.sigmaz()
    
    def construct_hamiltonian(self):
        """Construct QuTip Hamiltonian object [H0, [H1, f1] etc.]
        from H0 and H_ops
        Note: Currently, only support 1 channel, i.e. len(Hops) == 1
        """
        # TODO: This is a *HACK* only handle one qubit case
        H0 = self.parse_pauli_str(self.H0)
        H1 = self.parse_pauli_str(self.Hops[0])
        
        # Initial guess control
        def guess_control(t, args):
            return (1.0/self.lambda_a) * krotov.shapes.flattop(
                t, t_start=0, t_stop=self.tMax, t_rise=self.tMax/self.rise_fall_coeff, func=self.func_name
            )
        return [H0, [H1, guess_control]]
    
    def logical_basis(self):
        N = self.dimension
        result = np.zeros(2**(N), dtype = qutip.Qobj)
        for i in range(2**(N)):
            result[i] = qutip.Qobj(qutip.ket(format(i, 'b').zfill(N)))
        return result

    def targetObj(self):
        dim = self.dimension
        return qutip.Qobj(np.array(self.targetU).reshape((2**dim, 2**dim)), dims=[[2] * dim, [2] * dim])

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
    
    def plot_control_field_iterations(self, opt_result):
        """Plot the control fields over all iterations.
        This depends on ``store_all_pulses=True`` in the call to
        `optimize_pulses`.
        """
        fig, ax_ctr = plt.subplots(figsize=(8, 5))
        n_iters = len(opt_result.iters)
        for (iteration, pulses) in zip(opt_result.iters, opt_result.all_pulses):
            controls = [
                krotov.conversions.pulse_onto_tlist(pulse)
                for pulse in pulses
            ]
            
            if iteration == 0:
                ls = '--'  # dashed
                alpha = 1  # full opacity
                ctr_label = 'guess'
            elif iteration == opt_result.iters[-1]:
                ls = '-'  # solid
                alpha = 1  # full opacity
                ctr_label = 'optimized'
            else:
                ls = '-'  # solid
                alpha = 0.5 * float(iteration) / float(n_iters)  # max 50%
                ctr_label = None
            ax_ctr.plot(
                opt_result.tlist,
                controls[0],
                label=ctr_label,
                color='black',
                ls=ls,
                alpha=alpha,
            )
        ax_ctr.legend()
        ax_ctr.set_xlabel('time')
        ax_ctr.set_ylabel('control amplitude')
        plot_file_name = 'control_field_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.pdf'
        plt.savefig(plot_file_name)
        return plot_file_name

    # This is main entry point that the high-level
    # IR transformation service will call.
    # This needs to return the pair (opt-val, pulses)
    # where opt-val is a floating point number (final value of the cost function);
    # pulses is a single array of all control pulses (one for each control-H term)
    # (appending one pulse array after another).
    def optimize(self):
        # We can now call any Python lib to
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
        tlist = np.arange(0.0, self.tMax, self.dt)
        # TODO: this should be an IR transformation option
        def S(t):
            """Shape function for the field update"""
            return krotov.shapes.flattop(
                t, t_start=0, t_stop=self.tMax, t_rise=self.tMax/self.rise_fall_coeff, t_fall=self.tMax/self.rise_fall_coeff, func=self.func_name
            )

        pulse_options = {
            H[1][1]: dict(lambda_a=self.lambda_a, update_shape=S)
        }
        print('Krotov: Target Unitary Object:')
        print(self.targetObj())
        
        objectives = krotov.gate_objectives(
            # Target is the unitary: i.e. transform all states -> expected result.
            basis_states=self.logical_basis(), gate=self.targetObj(), H=H
        )
        
        opt_result = krotov.optimize_pulses(
            objectives,
            pulse_options,
            tlist,
            propagator=krotov.propagators.expm,
            chi_constructor=krotov.functionals.chis_ss,
            info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
            check_convergence=krotov.convergence.Or(
                krotov.convergence.value_below('1e-3', name='J_T'),
                krotov.convergence.check_monotonic_error,
            ),
            store_all_pulses=True,
        )

        plotFile = self.plot_control_field_iterations(opt_result)
        print(opt_result)
        print('*************************************************')
        print('**Plot of control field iterations is saved to:**')
        print(plotFile)
        print('*************************************************')

        pulse = opt_result.optimized_controls[0]
        return (0.0, pulse)