import gymnasium as gym
import numpy as np
import random
from statistics import mean
import numpy as np
from gymnasium.spaces import Discrete, Box
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.transpiler import CouplingMap, PassManager, StagedPassManager
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate
from sb3_contrib import MaskablePPO
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import Target
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary




class SynthesisEnv(gym.Env):

    def __init__(self, num_qubits):
        super(SynthesisEnv, self).__init__()
        self.difficulty = 1
        self.n_qubits = num_qubits  
        if self.n_qubits == 3:
            self.maps = [[[0, 1], [1, 2]], [[0, 1], [1, 2], [0, 2]]]
        else:
            self.maps = [[[0, 1], [1, 2], [2, 3], [3, 4]], [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], [[0, 2], [2, 1], [2, 3], [3, 4]], [[0, 2], [2, 1], [2, 3], [2, 4]]]
        
        self.tot_dict = {}
        i = 0
        for map in self.maps:
            dict = self._mapping__coupling_to_dict(
            CouplingMap(map), [HGate(), SGate(), CXGate()])
            for action in dict.values():
                if action not in self.tot_dict.values():
                    self.tot_dict[str(i)] = action
                    i += 1
        self.dict = {}

        self.action_space = Discrete(len(self.tot_dict.keys()))
              
        self.observation_space = Box(low=0, high=1, shape=(
            2*self.n_qubits, 2*self.n_qubits+1), dtype=int)

        self.i = random.randint(0, len(self.maps) - 1)
        self.dict = self._mapping__coupling_to_dict(
        CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        self.target_qc = self.generate_circuit(self.difficulty, self.n_qubits, CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        self.target_qc_size = self.target_qc.size()

        self.cliff_state = Clifford(self.target_qc)

        self.target_tableau_state = self.cliff_state.tableau       

        self.state = self.target_tableau_state
        
        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}        
        self.seed()
        self.success_count = 0
        self.total_episodes_rollout = 0
        self.qc_cx_size = 0
        self.qc_cx_depth = 0
        self.qc_size = 0
        self.qc_depth = 0      


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _mapping__coupling_to_dict(self, coupling_map, gates_set):
        i = 0
        dict_map = {}

        edges = coupling_map.graph.edge_list()
        nodes_indexes = coupling_map.graph.node_indexes()
        for gate in gates_set:
            num_qubits = gate.num_qubits

            if num_qubits == 1:
                for node_index in nodes_indexes:
                    dict_map[str(i)] = (gate, [node_index])
                    i += 1
            if num_qubits == 2:
                for edge in edges:
                    dict_map[str(i)] = (gate, list(edge))
                    i += 1
                    e = list(edge)
                    e.reverse()
                    dict_map[str(i)] = (gate, e)
                    i += 1
        return dict_map

    def count_q_qubit_gates(self, circuit, q):
        count = 0
        for gate in circuit:
            if gate[0].num_qubits == q:
                count += 1
        return count

    def generate_circuit(self, d, num_qubits, coupling_map, gates_set):
        edges = coupling_map.graph.edge_list()
        qc = QuantumCircuit(num_qubits)
        for i in range(d):
            gate = random.choice(gates_set)
            #print("gate: ", gate)
            nq_gate = gate.num_qubits
            if nq_gate == 1:
                q = random.randint(0, num_qubits-1)
                qc.append(gate, [q])
            else:
                q1, q2 = random.choice(edges)
                qc.append(gate, [q1, q2])
        return qc

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        self.reset()
    

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.i = random.randint(0, len(self.maps) - 1)
        self.dict = self._mapping__coupling_to_dict(
        CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        pm_staged = StagedPassManager()
        pm_staged.routing = PassManager(SabreSwap(CouplingMap(self.maps[self.i])))

        equiv_lib = SessionEquivalenceLibrary
        target = Target()
        target.add_instruction(SGate(), name='s')
        target.add_instruction(HGate(), name='h')
        target.add_instruction(CXGate(), name='cx')

        pass_manager = PassManager(BasisTranslator(equiv_lib, target))

        if self.difficulty > 1024:

            cliff = random_clifford(self.n_qubits)            

            cliff_qc = cliff.to_circuit()

            cliff_qc_routed = pm_staged.run(cliff_qc)

            self.target_qc = pass_manager.run(cliff_qc_routed)

            self.target_qc_size = self.target_qc.size()  

            self.cliff_state = Clifford(self.target_qc)      
                                
            self.target_tableau_state = self.cliff_state.tableau            
            
            self.state = self.target_tableau_state
            
        else:
            self.target_qc = self.generate_circuit(self.difficulty, self.n_qubits, CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

            self.target_qc_size = self.target_qc.size()

            self.cliff_state = Clifford(self.target_qc)

            self.target_tableau_state = self.cliff_state.tableau       

            self.state = self.target_tableau_state

        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}

        return (self.state, self.info)
    


    def step(self, action):

        gate, q = self.tot_dict[str(action)]      
        qc_tmp = QuantumCircuit(self.n_qubits)
        qc_tmp.append(gate, q)

        c = Clifford(qc_tmp)
        self.cliff_state = self.cliff_state.compose(c)

        self.target_tableau_state = self.cliff_state.tableau        

        depth_before = self.qc.depth()
        oneq_gates_before = self.count_q_qubit_gates(self.qc, 1)
        twoq_gates_before = self.count_q_qubit_gates(self.qc, 2)

        self.qc.append(gate, q)        

        depth_after = self.qc.depth()

        if gate.num_qubits == 1:
            oneq_gates_after = oneq_gates_before + 1
            twoq_gates_after = twoq_gates_before
        else:
            oneq_gates_after = oneq_gates_before
            twoq_gates_after = twoq_gates_before + 1

        if oneq_gates_after + twoq_gates_after >= self.target_qc_size:
            truncated = True
        else: truncated = False
    
        identity = np.identity(2 * self.n_qubits)

        identity_distance = 1 - \
            abs(identity - self.target_tableau_state.astype(int)[:, 0:-1]).mean()

        depth_coeff = 0.5
        twoq_gate_coeff = 0.15
        oneq_gate_coeff = 0.1

        reward = identity_distance - depth_coeff * \
            (depth_after - depth_before) - twoq_gate_coeff * (twoq_gates_after - twoq_gates_before) - oneq_gate_coeff * (oneq_gates_after - oneq_gates_before)

        if identity_distance == 1:
            reward = 1000 - depth_after * (depth_after - depth_before) - 10 * twoq_gates_after
            terminated = True
        else:
            terminated = False

        if truncated or terminated:
            self.total_episodes_rollout += 1
            self.qc_cx_size += self.count_q_qubit_gates(self.qc, 2)
            self.qc_cx_depth += self.qc.depth(lambda gate: gate[0].name in ['cx'])
            self.qc_size += self.qc.size()
            self.qc_depth += self.qc.depth()

        if terminated: self.success_count += 1

        self.info["synthetized_qc"]= self.qc
        self.info["tableau"] = self.target_tableau_state

        self.state = self.target_tableau_state

        return self.state, reward, terminated, truncated, self.info

    def reset_success_data(self):
        print('reset total episodes rollout ', self.total_episodes_rollout)
        self.success_count = 0
        self.total_episodes_rollout = 0
        self.qc_cx_size = 0
        self.qc_cx_depth = 0
        self.qc_size = 0
        self.qc_depth = 0

    def render(self, mode='human'):
        pass
    def close(self):
        pass

    def action_masks(self):
        l = []
        for action in self.tot_dict.values():
            if action in self.dict.values():
                l.append(True)
            else:
                l.append(False)
        return l
    
    def set_state(self, obs, map, target_qc_size, cliff_state):
        self.state = obs
        self.dict = self._mapping__coupling_to_dict(CouplingMap(map), [HGate(), SGate(), CXGate()])
        self.target_qc_size = target_qc_size
        self.cliff_state = cliff_state

        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}