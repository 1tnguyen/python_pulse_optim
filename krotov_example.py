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
            "omega0": 6.2831853,
            "omegaa": 0.0314159
        } 
}

# Create a pulse system model object 
model = xacc.createPulseModel()

# Load the Hamiltonian JSON (string) to the system model
loadResult = model.loadHamiltonianJson(json.dumps(hamiltonianJson))

if loadResult is True :
    qpu = xacc.getAccelerator('QuaC', {'system-model': model.name()})    
    channelConfigs = xacc.BackendChannelConfigs()
    T = 100.0 
    channelConfigs.dt = 1.0

    # Resonance: f = 1
    channelConfigs.loFregs_dChannels = [1.0]
    model.setChannelConfigs(channelConfigs)
   
    # Get the XASM compiler
    xasmCompiler = xacc.getCompiler('xasm');
    # Composite to be transform to pulse: X gate
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
        'method': 'krotov_optimizer',
        'max-time': T
    })
    # This composite should be a pulse composite now
    print(program)
    
    # Run the simulation of the optimized pulse program
    qubitReg = xacc.qalloc(1)
    qpu.execute(qubitReg, program)
    print(qubitReg)

