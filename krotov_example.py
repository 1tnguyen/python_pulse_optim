# Example of pulse-level IR transformation using ML-based method
# IMPORTANT: krotov_pulse_optim.py *MUST* be installed to the XACC python 
# plugin folder in order to run this.

# The following lines are for that installation
# This assumes the regular $HOME/.xacc location of XACC installation
import sys, os, shutil
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.xacc')
destFile = str(Path.home()) + '/.xacc/py-plugins/krotov_pulse_optim.py'
shutil.copyfile('krotov_pulse_optim.py', destFile) 
#########################################################################
import xacc, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# XACC IR transformation
hamiltonianJson = {
        "description": "Hamiltonian of a one-qubit system.\n",
        "h_str": ["-0.5*omega0*Z0", "omegaa*X0||D0"],
        "osc": {},
        "qub": {
            "0": 2
        },
        "vars": {
            "omega0": 1.0,
            "omegaa": 1.0
        } 
}

# Create a pulse system model object 
model = xacc.createPulseModel()

# Load the Hamiltonian JSON (string) to the system model
loadResult = model.loadHamiltonianJson(json.dumps(hamiltonianJson))

if loadResult is True :
    qpu = xacc.getAccelerator('QuaC', {'system-model': model.name(), 'logging-period': 0.1})    
    channelConfigs = xacc.BackendChannelConfigs()
    T = 5.0
    channelConfigs.dt = 0.01
    # Krotov will be able to compute things even without resonance
    # i.e. it will figure out how to modulate things into resonance (if needed)
    channelConfigs.loFregs_dChannels = [0.0]
    model.setChannelConfigs(channelConfigs)
   
    # Get the XASM compiler
    xasmCompiler = xacc.getCompiler('xasm');
    # Composite to be transform to pulse: X gate = H-Z-H
    ir = xasmCompiler.compile('''__qpu__ void f(qbit q) {
        Ry(q[0], pi/2);
        X(q[0]); 
    }''', qpu);
    program = ir.getComposites()[0]

    # Run the pulse IRTransformation 
    optimizer = xacc.getIRTransformation('quantum-control')
    optimizer.apply(program, qpu, {
        # Using the Python-contributed pulse optimizer
        # This will propagate to setOptions() then optimize()
        # calls on the optimizer implementation. 
        # Note: this is currently doing nothing
        'method': 'krotov',
        'max-time': T
    })
   
    # Verify the result
    # Run the simulation of the optimized pulse program
    qubitReg = xacc.qalloc(1)
    qpu.execute(qubitReg, program)
    print(qubitReg)
    # Retrieve time-stepping raw data
    csvFile = qubitReg['csvFile']
    data = np.genfromtxt(csvFile, delimiter = ',', dtype=float, names=True)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (8, 5))
    plt.tight_layout()
    ax[0].plot(data['Time'], data['Channel0'], 'b', label = '$D_0(t)$')
    ax[1].plot(data['Time'], data['X0'], 'b', label = '$\\langle X \\rangle$')
    ax[1].plot(data['Time'], data['Y0'], 'g', label = '$\\langle Y \\rangle$')
    ax[1].plot(data['Time'], data['Z0'], 'r', label = '$\\langle Z \\rangle$')
    ax[1].plot(data['Time'], data['Population0'], 'k', label = '$Prob(1)$')

    # ax[0].set_xlim([0, 5])
    # ax[0].set_ylim([-0.1, 2.0])
    # ax[1].set_xlim([0, 5])
    # ax[1].set_ylim([-1.1, 1.1])
    ax[1].legend()
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=5)
    ax[0].legend()
    ax[1].set_xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.1)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig('Krotov_Pulse_Response.pdf')

