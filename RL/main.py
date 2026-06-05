from synthesize import synthesize
from qiskit.quantum_info import random_clifford
from qiskit.transpiler import CouplingMap


if __name__ == "__main__":
    #you can provide as input the elements of the available test datasets present in the respective directory

    # you can use 3 or 5 qubits
    num_qubits = 5
    clifford = random_clifford(num_qubits)

    # you can use one of the possible coupling maps for the number of qubits you decided    
    coupling_map = [[0, 2], [2, 1], [2, 3], [3, 4]]

    # you can provide as input either a list of egdes representing the coupling map or an object of the CouplingMap class
    synthesized_qc = synthesize(clifford, coupling_map)
    
    print(synthesized_qc)
