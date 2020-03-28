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
    qpu = xacc.getAccelerator('QuaC', {'system-model': model.name()})    
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
        H(q[0]);
        Z(q[0]);
        H(q[0]);
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

