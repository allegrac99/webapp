import numpy as np
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap



def apply_cnot(tableau, control, target):
    """
    Applica CNOT(control, target) a un tableau stile Aaronson–Gottesman.

    tableau shape: (2n, 2n+1)
        righe: destabilizer (0..n-1) + stabilizer (n..2n-1)
        colonne: [X | Z | r]

    Regole tratte da Aaronson–Gottesman.
    """

    t = tableau.copy()
    rows, cols = t.shape
    n = rows // 2

    # Per comodità
    a = control
    b = target

    for i in range(2*n):

        # Update phase
        t[i, -1] ^= (t[i, a] & t[i, n+b] & (t[i, b] ^ t[i, n+a] ^ 1))

        # Update X part
        t[i, b] ^= t[i, a]

        # Update Z part
        t[i, n+a] ^= t[i, n+b]

    return t


def apply_swap(tableau, i, j):
    """
    Applica SWAP(i, j) al tableau stile Aaronson–Gottesman,
    usando la decomposizione standard:

        SWAP(i, j) = CNOT(i, j) • CNOT(j, i) • CNOT(i, j)

    dove la prima che applichi è l'ultima a destra.
    """

    t = tableau
    rows, cols = t.shape
    n = rows // 2

    # apply CNOT(i, j)
    t = apply_cnot(t, i, j)

    # apply CNOT(j, i)
    t = apply_cnot(t, j, i)

    # apply CNOT(i, j)
    t = apply_cnot(t, i, j)


    a = copy.deepcopy(t[i,:])
    b = copy.deepcopy(t[j,:])
    c = copy.deepcopy(t[n+i,:])
    d = copy.deepcopy(t[n+j,:])

    t[i,:] = b
    t[j,:] = a
    t[n+i,:] = d
    t[n+j,:] = c

    return t


def apply_layout(tableau, layout):
    n = len(layout)
    # Copia per non modificare l'originale, se vuoi
    layout = list(layout)

    for i in range(n):
        # Finché il qubit logico sulla riga i non è quello giusto
        while layout[i] != i:
            # il qubit logico che dovrebbe essere qui è i
            # cerco dove si trova attualmente
            target_logical = i
            j = layout.index(target_logical)

            # swap tra i qubit fisici i e j nel tableau
            res = apply_swap(tableau, i, j)

            # Se swap_func restituisce un tableau nuovo, aggiornalo
            if res is not None:
                tableau = res

            # Aggiorno anche il layout, perché ora i contenuti sono swappati
            layout[i], layout[j] = layout[j], layout[i]

    return tableau


    import numpy as np

def permute_clifford_tableau(tab, layout_log2phys, has_phase=False):
    """
    Permuta un tableau Clifford in stile Aaronson–Gottesman come RELABELING (layout),
    senza applicare SWAP/CNOT (cioè senza costo).

    tab: shape (2n, 2n) oppure (2n, 2n+1) se has_phase=True
    layout_log2phys: lista lunghezza n, layout_log2phys[i] = qubit fisico per il logical i
    has_phase: True se tab include colonna finale r.
    """
    tab = np.array(tab, copy=True)
    n = len(layout_log2phys)

    # Indici delle righe/colonne dopo la permutazione:
    # X rows: 0..n-1, Z rows: n..2n-1
    row_perm = list(layout_log2phys) + [n + p for p in layout_log2phys]

    # X cols: 0..n-1, Z cols: n..2n-1
    col_perm = list(layout_log2phys) + [n + p for p in layout_log2phys]

    if has_phase:
        core = tab[:, :2*n]
        r = tab[:, 2*n:]          # ultima colonna (o colonne) di fase
        core2 = core[row_perm, :][:, col_perm]
        r2 = r[row_perm, :]       # la fase si permuta con le righe
        return np.concatenate([core2, r2], axis=1)

    return tab[row_perm, :][:, col_perm]


def optimize_qc_and_correct_phase(qc):
    pm = PassManager([Optimize1qGates()])
    optimized_qc = pm.run(qc)



    


def synthesize_circuit_for_layout(layout,
                                  env,
                                  model,
                                  base_tab,
                                  base_map,
                                  cliff,
                                  max_steps):
    """
    Data una permutazione 'layout', esegue la sintesi RL e restituisce:
        - rl_synth_qc  : il circuito sintetizzato (inverse() già applicata)
        - cx_depth     : la cx depth (o 'penalty' se non termina)
        - terminated   : True se l'episodio è terminato correttamente
    """

    layout = list(layout)

    #new_tab = apply_layout(base_tab, layout)
    new_tab = permute_clifford_tableau(base_tab, layout, has_phase=True)
    map_ = base_map

    env.set_state(new_tab, map_, max_steps, cliff)

    obs = new_tab
    terminated = False
    truncated = False

    while not terminated and not truncated:
        env.render()

        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

    if not terminated:
        print("genetic false")
        return None, None, False

    rl_synth_qc = env.info["synthetized_qc"].inverse()
    cx_depth = rl_synth_qc.depth(lambda gate: gate[0].name in ['cx'])

    #optimize and correct phase
    


    return rl_synth_qc, cx_depth, True


def synthesize_circuit(env, model, base_tab, base_map, cliff, max_steps):

    map_ = base_map

    env.set_state(base_tab, map_, max_steps, cliff)

    obs = base_tab
    terminated = False
    truncated = False

    while not terminated and not truncated:
        env.render()

        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

    if not terminated:
        print("original false")
        return None, None, False

    rl_synth_qc = env.info["synthetized_qc"].inverse()
    cx_depth = rl_synth_qc.depth(lambda gate: gate[0].name in ['cx'])

    #optimize and correct phase

    return rl_synth_qc, cx_depth, True


def mapping_physical_to_logical(qc, layout):
    qc_perm = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        new_qargs = [qc_perm.qubits[layout[q._index]] for q in qargs]
        qc_perm.append(instr, new_qargs, cargs)
    return qc_perm

def invert_perm_log2phys(layout_log2phys):
    n = len(layout_log2phys)
    inv = [0]*n
    for logical, phys in enumerate(layout_log2phys):
        inv[phys] = logical
    return inv

def inverse_layout_log2phys(layout_log2phys):
    """
    Dato layout_log2phys (logical -> physical), restituisce inv_layout (physical -> logical).
    Se inv_layout viene passato a permute_clifford_tableau, ottieni l'inverso della permutazione.
    """
    layout = list(layout_log2phys)
    n = len(layout)

    # controlli: deve essere una permutazione di 0..n-1
    if sorted(layout) != list(range(n)):
        raise ValueError(
            f"layout_log2phys deve essere una permutazione di 0..{n-1}. "
            f"Ricevuto: {layout}"
        )

    inv = [0] * n
    for logical, phys in enumerate(layout):
        inv[phys] = logical
    return inv