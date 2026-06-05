from flask import Flask, render_template, request
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager, Target, StagedPassManager
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import Clifford, random_clifford, Operator
import ast
import hashlib
import sys
import os
from itertools import permutations
import json
from flask import session
import dotenv
from flask import jsonify, request  # gia' importati in app.py, ma per chiarezza


from qiskit.circuit.library.standard_gates import (
    HGate, SGate, CXGate,
    U1Gate, U2Gate, U3Gate,
    RZGate, SXGate, XGate
)

from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.transpiler.passes.routing import BasicSwap, LookaheadSwap
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

from qiskit.synthesis import synth_clifford_greedy, synth_clifford_ag

from qiskit.transpiler.passes import TrivialLayout, DenseLayout, SabreLayout
from qiskit.transpiler.passes import SetLayout, ApplyLayout

from utils_webapp import *
from flask import Flask, render_template, request, make_response
import json
import numpy as np
from qiskit.quantum_info import Clifford
import io

dotenv.load_dotenv()

def clifford_from_tableau_numpy_bytes(file_bytes: bytes) -> Clifford:
    """
    Load a Clifford tableau from a NumPy file (.npy or .npz).
    Expected array shape: (2n, 2n+1), dtype bool or 0/1 integers (also accepts 0.0/1.0).
    Validates shape + entries + Clifford symplectic constraints (via Qiskit's Clifford()).
    Returns a qiskit.quantum_info.Clifford if valid, otherwise raises ValueError.
    """
    bio = io.BytesIO(file_bytes)

    try:
        obj = np.load(bio, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Invalid NumPy file: {type(e).__name__}: {e}")

    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.files)
        if not keys:
            raise ValueError("Empty .npz file (no arrays found).")
        if "tableau" in keys:
            arr = obj["tableau"]
        elif len(keys) == 1:
            arr = obj[keys[0]]
        else:
            raise ValueError(
                f".npz contains multiple arrays {keys}. Please store the tableau as key 'tableau' (e.g., np.savez(..., tableau=T))."
            )
    else:
        arr = obj

    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Tableau must be a 2D matrix, got ndim={arr.ndim}.")
    r, c = arr.shape
    if r % 2 != 0:
        raise ValueError(f"Invalid tableau shape: rows must be even (2n). Got {r}.")
    if c != r + 1:
        raise ValueError(f"Invalid tableau shape: expected (2n, 2n+1). Got ({r}, {c}).")

    if arr.dtype == bool:
        tab_bool = arr
    else:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Tableau dtype must be bool or numeric 0/1. Got dtype={arr.dtype}.")
        unique_vals = np.unique(arr)
        if unique_vals.size == 0:
            raise ValueError("Empty tableau array.")
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(f"Tableau entries must be 0/1 (or True/False). Found values: {unique_vals.tolist()}")
        tab_bool = arr.astype(bool)

    try:
        return Clifford(tab_bool)
    except Exception as e:
        raise ValueError(f"Invalid Clifford tableau (symplectic/phase constraints not satisfied): {type(e).__name__}: {e}")


def clifford_from_tableau_json_bytes(file_bytes: bytes) -> Clifford:
    obj = json.loads(file_bytes.decode("utf-8"))
    if "tableau" not in obj:
        raise ValueError("Missing 'tableau' field in JSON.")
    tab = np.array(obj["tableau"], dtype=bool)
    if tab.ndim != 2:
        raise ValueError("'tableau' must be a 2D matrix.")
    r, c = tab.shape
    if c != r + 1 or r % 2 != 0:
        raise ValueError("Expected shape (2n, 2n+1).")
    return Clifford(tab)

def builder_json_from_clifford(c: Clifford, depth_hint: int = 10) -> str:
    qc = c.to_circuit()
    return builder_json_from_qc(qc, depth_hint=depth_hint)


def add_project_root_to_sys_path(current_file, target_subdir):
    current_dir = os.path.dirname(os.path.abspath(current_file))
    while current_dir and not os.path.exists(os.path.join(current_dir, target_subdir)):
        current_dir = os.path.dirname(current_dir)
    if current_dir and current_dir not in sys.path:
        sys.path.append(current_dir)


add_project_root_to_sys_path(__file__, 'RL')
'''from RL.synthesis_env import SynthesisEnv
from RL.synthesize import *'''


app = Flask(__name__)
app.secret_key = "change-this-secret-key"

try:
    from google import genai
    from google.genai import types as genai_types

    _gemini_api_key = os.getenv("GEMINI_API_KEY")
    _gemini_client = genai.Client(api_key=_gemini_api_key) if _gemini_api_key else None
except Exception:
    _gemini_client = None
    genai_types = None


def _gemini_is_configured() -> bool:
    return _gemini_client is not None and genai_types is not None


def _gemini_generate_text(
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
) -> str:
    """
    Calls Gemini and returns plain text.
    Uses GEMINI_API_KEY and optional GEMINI_MODEL from environment variables.
    """
    if not _gemini_is_configured():
        raise RuntimeError("Gemini not configured. Missing GEMINI_API_KEY or google-genai.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    response = _gemini_client.models.generate_content(
        model=model,
        contents=[
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )

    reply = (response.text or "").strip() if hasattr(response, "text") else ""
    if reply:
        return reply

    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        finish_reason = "unknown"

    raise RuntimeError(f"Empty Gemini response. Finish reason: {finish_reason}")


def _gemini_error_message(e: Exception) -> str:
    msg = str(e)
    lower = msg.lower()

    if "api_key" in lower or "api key" in lower or "unauthenticated" in lower or "401" in msg:
        return "Gemini not configured (invalid or missing GEMINI_API_KEY)."
    if "resource_exhausted" in lower or "429" in msg or "rate" in lower or "quota" in lower:
        return "Gemini analysis unavailable: rate limit or quota exceeded."
    if "not found" in lower or ("model" in lower and "404" in msg):
        return "Gemini analysis unavailable: model not found or not accessible."
    if "timeout" in lower or "network" in lower:
        return "Gemini analysis unavailable: network error or timeout."
    return f"Gemini analysis unavailable: {type(e).__name__}: {e}"

def format_clifford_tableau_bool(cliff: Clifford) -> str:
    import numpy as np
    arr = np.asarray(cliff.tableau, dtype=int)
    rows = [" ".join(str(int(x)) for x in row) for row in arr]
    return f"Shape: {arr.shape[0]} x {arr.shape[1]}\n\n" + "\n".join(rows)


def generate_routing_analysis(original_text: str, routed_text: str, original_qasm: str | None = None, routed_qasm: str | None = None) -> str:
    if not _gemini_is_configured():
        return "Gemini not configured (missing library or GEMINI_API_KEY)."

    system_instruction = "You are an expert in Qiskit transpilation and routing on coupling maps."
    user_prompt = (
        "You are a quantum circuit routing assistant. "
        "Analyze differences between the original and the routed circuits."
        "In particular, at the beginning indicate how many SWAP gates have been added."
        "Then, for each 2-qubit operation that can not be executed with the"
        "chosen coupling map and layout, explain which swap gates have been inserted"
        "and why now it is ok."
        "Then, say the differences in terms of gate counts, depth, CX depth, inserted SWAPs, and the impact of layout/coupling."
        "Remember: if SWAP/CX gates have a qubit in common, they can not be executed in parallel."
        "Provide a concise summary in English (10–12 lines), bullet-style, no code. Do NOT use Markdown (no **, no backticks, no headings).\n\n"
        "[Original circuit - ASCII]\n" + (original_text or "(n/a)") + "\n\n"
        + ("[Original circuit - QASM]\n" + original_qasm + "\n\n" if original_qasm else "")
        + "[Routed circuit - ASCII]\n" + (routed_text or "(n/a)") + "\n\n"
        + ("[Routed circuit - QASM]\n" + routed_qasm + "\n\n" if routed_qasm else "")
    )

    try:
        return _gemini_generate_text(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=0.2,
            max_output_tokens=700,
        )
    except Exception as e:
        return _gemini_error_message(e)

def generate_synthesis_comparison_analysis(
    operator_kind: str,
    operator_payload: str,
    compare_results: list[dict],
) -> str:
    """
    compare_results items expected keys:
      - method (label)
      - swap_count
      - stats (dict or None)
      - circuit_text (str)
      - circuit_qasm (str or None)
      - docs (str or "")
      - error (str or None)
    """
    if not _gemini_is_configured():
        return "Gemini not configured (missing library or GEMINI_API_KEY)."

    blocks = []
    blocks.append(f"Operator ({operator_kind}):\n{operator_payload}\n")

    for item in compare_results:
        method = item.get("method", "Unknown")
        err = item.get("error")
        docs = item.get("docs") or ""
        swap_count = item.get("swap_count", None)
        stats = item.get("stats", None)
        txt = item.get("circuit_text", "") or ""
        qasm = item.get("circuit_qasm", None)

        blocks.append(f"=== METHOD: {method} ===")
        if docs:
            blocks.append(f"Docs: {docs}")
        if err:
            blocks.append(f"Status: FAILED\nError: {err}\n")
            continue

        blocks.append("Status: OK")
        blocks.append(f"SWAP count: {swap_count if swap_count is not None else '—'}")
        if stats:
            blocks.append(
                f"Stats: size={stats.get('size')} depth={stats.get('depth')} cx_depth={stats.get('cx_depth')} qubits={stats.get('num_qubits')}"
            )
        blocks.append("[Circuit - ASCII]")
        blocks.append(txt.strip() if txt else "(empty)")
        if qasm:
            blocks.append("[Circuit - QASM]")
            blocks.append(qasm.strip())
        blocks.append("")

    system_instruction = "You are an expert in Qiskit Clifford synthesis and circuit optimization."
    user_prompt = (
        "You are a quantum circuit synthesis comparison assistant.\n"
        "Given the same Clifford operator and the synthesized circuits produced by different methods,\n"
        "compare the methods and explain trade-offs.\n\n"
        "Requirements:\n"
        "- Start with a 1-line verdict: which method is best under depth, which under gate count, which under CX depth.\n"
        "- Then give 8–12 bullet points (plain text, no Markdown) comparing:\n"
        "  * size, depth, CX depth, SWAPs\n"
        "  * routing/translation side-effects (if present)\n"
        "  * notable structural differences you can infer from the circuits\n"
        "- If a method failed, mention it briefly and ignore it for 'best method' selection.\n"
        "- Be concise and concrete; cite exact numbers when available.\n\n"
        + "\n".join(blocks)
    )

    try:
        return _gemini_generate_text(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=0.2,
            max_output_tokens=800,
        )
    except Exception as e:
        return _gemini_error_message(e)

def generate_routing_comparison_analysis(
    original_text: str,
    compare_results: list[dict],
    original_qasm: str | None = None,
) -> str:
    """
    compare_results items expected keys:
      - method
      - docs
      - swap_count
      - stats (dict or None)
      - circuit_text (str)
      - circuit_text_cx (str or None)
    """
    if not _gemini_is_configured():
        return "Gemini not configured (missing library or GEMINI_API_KEY)."

    blocks = []
    blocks.append("[Original circuit - ASCII]\n" + (original_text or "(n/a)") + "\n")
    if original_qasm:
        blocks.append("[Original circuit - QASM]\n" + original_qasm + "\n")

    for item in compare_results:
        m = item.get("method", "unknown")
        docs = item.get("docs") or ""
        sw = item.get("swap_count", None)
        st = item.get("stats", None)
        txt_obj = item.get("circuit_text") or ""
        txt = str(txt_obj).strip()

        blocks.append(f"=== ROUTING METHOD: {m} ===")
        if docs:
            blocks.append(f"Docs: {docs}")
        blocks.append(f"SWAP count: {sw if sw is not None else '—'}")
        if st:
            blocks.append(
                f"Stats: size={st.get('size')} depth={st.get('depth')} cx_depth={st.get('cx_depth')} qubits={st.get('num_qubits')}"
            )
        blocks.append("[Routed circuit - ASCII]")
        blocks.append(txt if txt else "(empty)")
        blocks.append("")

    system_instruction = "You are an expert in Qiskit transpilation and routing on coupling maps."
    user_prompt = (
        "You are a quantum circuit routing comparison assistant.\n"
        "Given one original circuit and multiple routed versions produced by different routing methods,\n"
        "compare the methods and explain trade-offs.\n\n"
        "Requirements:\n"
        "- Start with a 1-line verdict: best under SWAP count, best under depth, best under CX depth.\n"
        "- Then give 8–12 bullet points (plain text, no Markdown) comparing:\n"
        "  * SWAPs, depth, CX depth, size\n"
        "  * structural differences you can infer from the circuits\n"
        "  * any notable behavior of methods (e.g., why one inserts more SWAPs)\n"
        "- Be concise and cite exact numbers when available.\n\n"
        + "\n".join(blocks)
    )

    try:
        return _gemini_generate_text(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            temperature=0.2,
            max_output_tokens=800,
        )
    except Exception as e:
        return _gemini_error_message(e)

DOCS = {
    "BasisTranslator": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.BasisTranslator",
}

LAYOUT_DOCS = {
    "none": "",
    "trivial": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.TrivialLayout",
    "dense": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.DenseLayout",
    "sabre": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SabreLayout",
    "custom": "",
}

ROUTING_DOCS = {
    "none": "",
    "basic": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.BasicSwap",
    "lookahead": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.LookaheadSwap",
    "sabre": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.SabreSwap",
}

SYNTHESIS_DOCS = {
    "clifford_greedy": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/synthesis#synth_clifford_greedy",
    "clifford_ag": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/synthesis#synth_clifford_ag",
    "RL": "https://github.com/Quasar-UniNA/Topology_Aware_Quantum_Circuit_Synthesis_with_RL",
}


BASIS_GATES_OPTIONS = {
    "u3_cx": ["u3", "cx"],
    "rz_sx_x_cx": ["rz", "sx", "x", "cx"],
    "u1_u2_u3_cx": ["u1", "u2", "u3", "cx"],
    "none": None,
}

LAYOUT_OPTIONS = {
    "none": None,
    "trivial": "trivial",
    "dense": "dense",
    "sabre": "sabre",
    "custom": "custom",
}

ROUTING_OPTIONS = {
    "none": None,
    "basic": "basic",
    "lookahead": "lookahead",
    "sabre": "sabre",
}

COUPLING_MAP_OPTIONS = {
    "none": "none",
    "line": "line",
    "ring": "ring",
}

CLIFFORD_ALGO_OPTIONS = {
    "clifford_greedy": "clifford_greedy",
    "clifford_ag": "clifford_ag",
    "RL": "RL",
}

SYNTH_COMPARE_ORDER = ["clifford_greedy", "clifford_ag", "RL"]

SYNTH_METHOD_LABELS = {
    "clifford_greedy": "Greedy",
    "clifford_ag": "Aaronson-Gottesman",
    "RL": "Topology-Aware Reinforcement Learning-based Clifford synthesizer",
}


def _json_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def build_coupling_map(map_type: str, num_qubits: int):
    if map_type == "none":
        return None
    if num_qubits <= 1:
        return None

    if map_type == "line":
        edges = [[i, i + 1] for i in range(num_qubits - 1)]
        edges += [[i + 1, i] for i in range(num_qubits - 1)]
    elif map_type == "ring":
        edges = [[i, (i + 1) % num_qubits] for i in range(num_qubits)]
        edges += [[(i + 1) % num_qubits, i] for i in range(num_qubits)]
    elif map_type == "star":
        if num_qubits != 5:
            return None
        edges = []
        for i in range(1, 5):
            edges.append([0, i])
            edges.append([i, 0])
    elif map_type == "further_map":
        if num_qubits != 5:
            return None
        undirected = [(0, 2), (2, 1), (2, 3), (3, 4)]
        edges = []
        for a, b in undirected:
            edges.append([a, b])
            edges.append([b, a])
    else:
        return None

    return CouplingMap(edges)


def compute_stats(circ: QuantumCircuit):
    depth = circ.depth()
    size = circ.size()
    cx_depth = circ.depth(filter_function=lambda inst: inst.operation.name == "cx")
    return {"num_qubits": circ.num_qubits, "size": size, "depth": depth, "cx_depth": cx_depth}


def count_ops(qc: QuantumCircuit, name: str) -> int:
    return sum(1 for inst, _, _ in qc.data if inst.name == name)


def routing_comments(original: QuantumCircuit, routed: QuantumCircuit):
    orig_swap = count_ops(original, "swap")
    routed_swap = count_ops(routed, "swap")
    added_swap = max(0, routed_swap - orig_swap)
    if added_swap > 0:
        return [f"Routing inserted {added_swap} SWAP gate(s) to satisfy the coupling map."]
    return ["Routing did not insert any SWAP gates."]


def _layout_to_list_for_qc(qc: QuantumCircuit, layout_obj):
    v2p = layout_obj.get_virtual_bits()
    out = []
    for vq in qc.qubits:
        if vq not in v2p:
            return None
        out.append(int(v2p[vq]))
    return out


def compute_layout_only(qc: QuantumCircuit, coupling_map: CouplingMap, layout_method: str, custom_layout_list=None):
    if coupling_map is None:
        layout_list = list(custom_layout_list) if custom_layout_list is not None else list(range(qc.num_qubits))
    else:
        if layout_method == "custom":
            layout_list = list(custom_layout_list)
        elif layout_method == "trivial":
            pm = PassManager([TrivialLayout(coupling_map)])
            _ = pm.run(qc)
            layout_obj = pm.property_set.get("layout", None)
            layout_list = _layout_to_list_for_qc(qc, layout_obj) if layout_obj is not None else list(range(qc.num_qubits))
            if layout_list is None:
                layout_list = list(range(qc.num_qubits))
        elif layout_method == "dense":
            pm = PassManager([DenseLayout(coupling_map)])
            _ = pm.run(qc)
            layout_obj = pm.property_set.get("layout", None)
            layout_list = _layout_to_list_for_qc(qc, layout_obj) if layout_obj is not None else list(range(qc.num_qubits))
            if layout_list is None:
                layout_list = list(range(qc.num_qubits))
        elif layout_method == "sabre":
            pm = PassManager([SabreLayout(coupling_map)])
            _ = pm.run(qc)
            layout_obj = pm.property_set.get("layout", None)
            layout_list = _layout_to_list_for_qc(qc, layout_obj) if layout_obj is not None else list(range(qc.num_qubits))
            if layout_list is None:
                layout_list = list(range(qc.num_qubits))
        else:
            layout_list = list(range(qc.num_qubits))

    layout_pairs = [(i, layout_list[i]) for i in range(qc.num_qubits)]
    return layout_list, layout_pairs


def circuit_from_builder_json(circuit_json: str) -> QuantumCircuit:
    import json
    if not circuit_json or not circuit_json.strip():
        raise ValueError("Empty circuit JSON.")
    data = json.loads(circuit_json)

    n = int(data["num_qubits"])
    ops = data.get("ops", [])
    qc = QuantumCircuit(n)

    def _key(op):
        return (int(op.get("col", 0)), op.get("gate", ""), op.get("qubits", [0])[0])

    for op in sorted(ops, key=_key):
        gate = str(op["gate"]).lower().strip()
        qs = op["qubits"]

        if gate == "id":
            qc.id(int(qs[0]))
        elif gate == "h":
            qc.h(int(qs[0]))
        elif gate == "x":
            qc.x(int(qs[0]))
        elif gate == "y":
            qc.y(int(qs[0]))
        elif gate == "z":
            qc.z(int(qs[0]))
        elif gate == "s":
            qc.s(int(qs[0]))
        elif gate == "sdg":
            qc.sdg(int(qs[0]))
        elif gate == "t":
            qc.t(int(qs[0]))
        elif gate == "tdg":
            qc.tdg(int(qs[0]))
        elif gate == "sx":
            qc.sx(int(qs[0]))
        elif gate == "sxdg":
            qc.sxdg(int(qs[0]))

        elif gate == "cx":
            qc.cx(int(qs[0]), int(qs[1]))
        elif gate == "cy":
            qc.cy(int(qs[0]), int(qs[1]))
        elif gate == "cz":
            qc.cz(int(qs[0]), int(qs[1]))
        elif gate == "swap":
            qc.swap(int(qs[0]), int(qs[1]))

        else:
            raise ValueError(f"Unsupported gate in builder: {gate}")

    return qc

def builder_json_from_qc(qc: QuantumCircuit, depth_hint: int = 10) -> str:
    n = qc.num_qubits
    cols = max(10, int(depth_hint or 10))

    ops_out = []
    occ_1q = {q: set() for q in range(n)}
    occ_range = []

    def col_blocked_for_range(col: int, lo: int, hi: int) -> bool:
        for (c, a, b) in occ_range:
            if c != col:
                continue
            if not (b < lo or a > hi):
                return True
        for q in range(lo, hi + 1):
            if col in occ_1q[q]:
                return True
        return False

    nonlocal_cols = [cols]

    allowed_1q = {"id","h","x","y","z","s","sdg","t","tdg","sx","sxdg"}
    allowed_2q = {"cx","cy","cz","swap"}

    for inst, qargs, _ in qc.data:
        name = inst.name.lower().strip()
        qubits = [qc.find_bit(q).index for q in qargs]

        if name in allowed_1q and len(qubits) == 1:
            q = qubits[0]
            c = 0
            while True:
                if c >= nonlocal_cols[0]:
                    nonlocal_cols[0] += 5
                if not col_blocked_for_range(c, q, q):
                    break
                c += 1
            occ_1q[q].add(c)
            ops_out.append({"gate": name, "qubits": [q], "col": c})

        elif name in allowed_2q and len(qubits) == 2:
            a, b = qubits
            lo, hi = min(a, b), max(a, b)
            c = 0
            while True:
                if c >= nonlocal_cols[0]:
                    nonlocal_cols[0] += 5
                if not col_blocked_for_range(c, lo, hi):
                    break
                c += 1
            occ_range.append((c, lo, hi))
            ops_out.append({"gate": name, "qubits": [a, b], "col": c})

        else:
            raise ValueError(f"Unsupported gate in loaded QASM for editor: {name}")

    payload = {"num_qubits": n, "depth": int(nonlocal_cols[0]), "ops": ops_out}
    return json.dumps(payload)


def swap_to_3cx_text(qc: QuantumCircuit):
    try:
        decomp = qc.decompose(gates_to_decompose=["swap"])
        if count_ops(qc, "swap") == count_ops(decomp, "swap"):
            return None, None
        return decomp.draw(output="text"), compute_stats(decomp)
    except Exception:
        return None, None


def run_routing_only(qc: QuantumCircuit, coupling_map: CouplingMap, layout_list: list[int], routing_method: str):
    passes = [SetLayout(layout_list), ApplyLayout()]

    if routing_method == "none":
        pass
    elif routing_method == "basic":
        passes.append(BasicSwap(coupling_map))
    elif routing_method == "lookahead":
        passes.append(LookaheadSwap(coupling_map))
    else:
        passes.append(SabreSwap(coupling_map))

    pm = PassManager(passes)
    return pm.run(qc)


def _target_from_basis_gates(basis_gates: list[str]) -> Target:
    t = Target()

    gate_map = {
        "h": HGate(),
        "s": SGate(),
        "x": XGate(),
        "sx": SXGate(),
        "rz": RZGate(0.0),
        "cx": CXGate(),
        "u1": U1Gate(0.0),
        "u2": U2Gate(0.0, 0.0),
        "u3": U3Gate(0.0, 0.0, 0.0),
    }

    for g in basis_gates:
        g = g.lower().strip()
        if g not in gate_map:
            raise ValueError(f"Unsupported basis gate for translator: {g}")
        t.add_instruction(gate_map[g], name=g)

    return t


def translate_circuit_basis_translator(qc: QuantumCircuit, basis_gates: list[str] | None) -> QuantumCircuit:
    if basis_gates is None:
        return qc

    equiv_lib = SessionEquivalenceLibrary
    target = _target_from_basis_gates(basis_gates)
    pm = PassManager(BasisTranslator(equiv_lib, target))
    return pm.run(qc)


def _format_complex(z: complex, prec: int = 3) -> str:
    a = float(z.real)
    b = float(z.imag)
    if abs(a) < 10 ** (-prec):
        a = 0.0
    if abs(b) < 10 ** (-prec):
        b = 0.0
    sign = "+" if b >= 0 else "-"
    return f"{a:.{prec}f}{sign}{abs(b):.{prec}f}i"


def format_operator_matrix(op_data, prec: int = 3, max_dim: int = 16) -> str:
    import numpy as np

    mat = np.asarray(op_data)
    d = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        return f"(Not a square matrix: shape={mat.shape})"

    clipped = False
    if d > max_dim:
        mat = mat[:max_dim, :max_dim]
        clipped = True

    rows = []
    for i in range(mat.shape[0]):
        row = "  ".join(_format_complex(mat[i, j], prec=prec) for j in range(mat.shape[1]))
        rows.append(row)

    header = f"Dimension: {d} x {d}"
    if clipped:
        header += f"\n(Displayed top-left {max_dim} x {max_dim} block for readability.)"
    return header + "\n\n" + "\n".join(rows)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/transpile", methods=["GET", "POST"])
def transpile_view():
    # UNCHANGED (come nel tuo file)
    show_qiskit = session.get("show_qiskit", False)
    show_qiskit = session.get("show_qiskit", False)
    original_circuit_text = session.get("qiskit_text", None) if show_qiskit else None
    original_stats = session.get("qiskit_stats", None) if show_qiskit else None


    num_qubits = 3
    builder_depth = 10

    selected_map = "line"
    selected_layout = "trivial"
    custom_layout_str = ""
    selected_routing = "sabre"

    selected_translation_basis = "u3_cx"

    circuit_json = ""

    original_circuit_text = None
    original_stats = None

    layout_list_str = ""
    layout_for_hash = ""

    layout_pairs = None
    layout_compare_results = None

    routed_circuit_text = None
    routed_stats = None
    routed_swap_count = None
    routed_circuit_text_cx = None
    routed_stats_cx = None
    routing_notes = None
    routing_compare_results = None

    routed_qasm_str = ""
    routed_for_hash = ""

    translated_circuit_text = None
    translated_stats = None
    translation_error = None

    editor_error = None
    layout_error = None
    routing_error = None

    focus_section = None
    qc_for_view = None

    routing_analysis_text = session.get("routing_analysis_text", None)
    routing_analysis_hash = session.get("routing_analysis_hash", "")

    routing_compare_analysis_text = session.get("routing_compare_analysis_text", None)
    routing_compare_analysis_hash = session.get("routing_compare_analysis_hash", "")


    def _routing_hash(circuit_json_: str, layout_list_str_: str, selected_routing_: str, selected_map_: str) -> str:
        payload = f"{circuit_json_}|{layout_list_str_}|{selected_routing_}|{selected_map_}"
        return _json_hash(payload)

    if request.method == "POST":
        actions = request.form.getlist("action")
        action = actions[-1] if actions else None

        focus_section = request.form.get("focus_section") or None

        num_qubits = int(request.form.get("num_qubits", 3))
        builder_depth = int(request.form.get("builder_depth", 10))

        selected_map = request.form.get("map_type", "line")
        selected_layout = request.form.get("layout_method", "trivial")
        custom_layout_str = request.form.get("custom_layout", "").strip()
        selected_routing = request.form.get("routing_method", "sabre")

        selected_translation_basis = request.form.get("translation_basis", "u3_cx")

        circuit_json = request.form.get("circuit_json", "")
        layout_list_str = request.form.get("layout_list_str", "")
        layout_for_hash = request.form.get("layout_for_hash", "")

        routed_qasm_str = request.form.get("routed_qasm_str", "")
        routed_for_hash = request.form.get("routed_for_hash", "")

        def section_of_action(act: str):
            if act in ("render_circuit", "hide_qiskit", "clear_all"):
                return "editor"
            if act in ("apply_layout", "compare_layout", "clear_layout"):
                return "layout"
            if act in ("apply_routing", "compare_routing", "clear_routing"):
                return "routing"
            if act in ("apply_translation", "clear_translation"):
                return "translation"
            return "editor"

        def set_error_for_action(msg: str):
            nonlocal editor_error, layout_error, routing_error, translation_error, focus_section
            sec = section_of_action(action)
            focus_section = sec
            if sec == "layout":
                layout_error = msg
            elif sec == "routing":
                routing_error = msg
            elif sec == "translation":
                translation_error = msg
            else:
                editor_error = msg

        def _load_qc_or_error():
            if not circuit_json or not circuit_json.strip():
                set_error_for_action(
                    "You must build a circuit first. Use the editor and click 'Render circuit' before applying layout."
                )
                return None
            try:
                return circuit_from_builder_json(circuit_json)
            except Exception as e:
                set_error_for_action(f"Invalid circuit: {type(e).__name__}: {e}")
                return None

        def _parse_custom_layout_or_error(qc: QuantumCircuit):
            if selected_layout != "custom":
                return None
            if not custom_layout_str:
                set_error_for_action("You selected 'Custom' layout, but no permutation has been chosen.")
                return None
            try:
                parsed = ast.literal_eval(custom_layout_str)
                if isinstance(parsed, (list, tuple)) and len(parsed) == qc.num_qubits:
                    return list(parsed)
                raise ValueError("Custom layout must be a list of length num_qubits.")
            except Exception as e:
                set_error_for_action(f"Invalid custom layout: {type(e).__name__}: {e}")
                return None

        def _clear_layout_outputs_only():
            nonlocal layout_pairs, layout_compare_results
            layout_pairs = None
            layout_compare_results = None

        def _clear_routing_outputs_only():
            nonlocal routed_circuit_text, routed_stats, routed_swap_count
            nonlocal routed_circuit_text_cx, routed_stats_cx
            nonlocal routing_notes, routing_compare_results
            routed_circuit_text = None
            routed_stats = None
            routed_swap_count = None
            routed_circuit_text_cx = None
            routed_stats_cx = None
            routing_notes = None
            routing_compare_results = None

        def _clear_layout_memory():
            nonlocal layout_list_str, layout_for_hash
            layout_list_str = ""
            layout_for_hash = ""

        def _clear_routing_memory_for_translation():
            nonlocal routed_qasm_str, routed_for_hash
            routed_qasm_str = ""
            routed_for_hash = ""

        def _clear_translation_outputs_only():
            nonlocal translated_circuit_text, translated_stats, translation_error
            translated_circuit_text = None
            translated_stats = None
            translation_error = None

        if action != "clear_all":
            qc_for_view = _load_qc_or_error()

            if qc_for_view is not None and action == "render_circuit":
                original_circuit_text = str(qc_for_view.draw(output="text"))
                original_stats = compute_stats(qc_for_view)

        if action == "clear_all":
            focus_section = "editor"
            circuit_json = ""
            original_circuit_text = None
            original_stats = None
            session.pop("routing_analysis_text", None)
            session.pop("routing_analysis_hash", None)
            session.pop("show_qiskit", None)
            session.pop("qiskit_text", None)
            session.pop("qiskit_stats", None)
            session.pop("routing_compare_analysis_text", None)
            session.pop("routing_compare_analysis_hash", None)



            _clear_layout_outputs_only()
            _clear_routing_outputs_only()
            _clear_layout_memory()
            _clear_routing_memory_for_translation()
            _clear_translation_outputs_only()

        elif action == "render_circuit":
            focus_section = "editor"
            qc = qc_for_view
            if qc is not None:
                show_qiskit = True
                session["show_qiskit"] = True

                original_circuit_text = str(qc.draw(output="text"))
                original_stats = compute_stats(qc)

                session["qiskit_text"] = original_circuit_text
                session["qiskit_stats"] = original_stats


        elif action == "hide_qiskit":
            focus_section = "editor"
            session["show_qiskit"] = False
            session.pop("qiskit_text", None)
            session.pop("qiskit_stats", None)
            show_qiskit = False
            original_circuit_text = None
            original_stats = None


        elif action == "load_qasm":
            focus_section = "editor"

            file = request.files.get("qasm_file", None)
            if file is None or file.filename == "":
                set_error_for_action("Please select a QASM file to load.")
            else:
                try:
                    qasm_text = file.read().decode("utf-8", errors="replace")
                    qc_loaded = QuantumCircuit.from_qasm_str(qasm_text)

                    circuit_json = builder_json_from_qc(qc_loaded, depth_hint=builder_depth)

                    num_qubits = qc_loaded.num_qubits
                    builder_depth = max(builder_depth, 10)

                    _clear_layout_outputs_only()
                    _clear_routing_outputs_only()
                    _clear_layout_memory()
                    _clear_routing_memory_for_translation()
                    _clear_translation_outputs_only()

                    qc_for_view = qc_loaded
                    original_circuit_text = None
                    original_stats = None

                except Exception as e:
                    set_error_for_action(f"Failed to load QASM: {type(e).__name__}: {e}")

        elif action == "save_circuit":
            focus_section = "editor"

            qc = qc_for_view
            if qc is not None:
                try:
                    qasm_str = qasm2_dumps(qc)
                    filename = f"circuit_{qc.num_qubits}q.qasm"

                    resp = make_response(qasm_str)
                    resp.headers["Content-Type"] = "application/qasm"
                    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
                    return resp

                except Exception as e:
                    set_error_for_action(f"Save failed: {type(e).__name__}: {e}")

        elif action == "clear_layout":
            focus_section = "layout"
            _clear_layout_outputs_only()
            _clear_layout_memory()

            _clear_routing_outputs_only()
            _clear_routing_memory_for_translation()
            _clear_translation_outputs_only()
            session.pop("routing_analysis_text", None)
            session.pop("routing_analysis_hash", None)
            session.pop("routing_compare_analysis_text", None)
            session.pop("routing_compare_analysis_hash", None)


        elif action == "clear_routing":
            focus_section = "routing"
            _clear_routing_outputs_only()
            _clear_routing_memory_for_translation()
            _clear_translation_outputs_only()
            session.pop("routing_analysis_text", None)
            session.pop("routing_analysis_hash", None)
            session.pop("routing_compare_analysis_text", None)
            session.pop("routing_compare_analysis_hash", None)


        elif action == "clear_translation":
            focus_section = "translation"
            _clear_translation_outputs_only()

        elif action == "apply_layout":
            focus_section = "layout"
            qc = qc_for_view
            if qc is not None:
                coupling_map = build_coupling_map(selected_map, qc.num_qubits)
                custom_layout_list = _parse_custom_layout_or_error(qc)
                if (selected_layout != "custom") or (custom_layout_list is not None):
                    layout_list, layout_pairs = compute_layout_only(
                        qc=qc,
                        coupling_map=coupling_map,
                        layout_method=selected_layout,
                        custom_layout_list=custom_layout_list,
                    )
                    layout_list_str = str(layout_list)
                    layout_for_hash = _json_hash(circuit_json)

                    _clear_routing_outputs_only()
                    _clear_routing_memory_for_translation()
                    _clear_translation_outputs_only()
                    layout_compare_results = None

        elif action == "compare_layout":
            focus_section = "layout"
            qc = qc_for_view
            if qc is not None:
                coupling_map = build_coupling_map(selected_map, qc.num_qubits)
                _clear_layout_memory()

                custom_layout_list = _parse_custom_layout_or_error(qc)

                compare = []
                for m in ["trivial", "dense", "sabre"]:
                    ll, lp = compute_layout_only(qc, coupling_map, m, custom_layout_list=None)
                    compare.append({"method": m, "layout_pairs": lp})

                if selected_layout == "custom" and custom_layout_list is not None:
                    ll, lp = compute_layout_only(qc, coupling_map, "custom", custom_layout_list=custom_layout_list)
                    compare.append({"method": "custom", "layout_pairs": lp})

                layout_compare_results = compare
                layout_pairs = None

                _clear_routing_outputs_only()
                _clear_routing_memory_for_translation()
                _clear_translation_outputs_only()

        elif action == "apply_routing":
            focus_section = "routing"
            qc = qc_for_view
            if qc is not None:
                if (not layout_list_str.strip()) or (layout_for_hash != _json_hash(circuit_json)):
                    set_error_for_action("Please apply layout before routing")
                else:
                    coupling_map = build_coupling_map(selected_map, qc.num_qubits)

                    if coupling_map is None and selected_routing != "none":
                        set_error_for_action("Routing requires a coupling map. Please select a coupling map first.")
                    else:
                        try:
                            layout_list = ast.literal_eval(layout_list_str)
                            if not isinstance(layout_list, (list, tuple)) or len(layout_list) != qc.num_qubits:
                                raise ValueError("Stored layout is invalid.")
                            layout_list = list(layout_list)
                        except Exception as e:
                            set_error_for_action(f"Invalid stored layout: {type(e).__name__}: {e}")
                            layout_list = None

                        if routing_error is None:
                            try:
                                tqc_routed = run_routing_only(qc, coupling_map, layout_list, selected_routing)
                                routed_circuit_text = str(tqc_routed.draw(output="text"))
                                routed_stats = compute_stats(tqc_routed)
                                routed_swap_count = count_ops(tqc_routed, "swap")
                                routing_notes = routing_comments(qc, tqc_routed)
                                routed_circuit_text_cx, routed_stats_cx = swap_to_3cx_text(tqc_routed)
                                routing_compare_results = None

                                routed_qasm_str = qasm2_dumps(tqc_routed)
                                routed_for_hash = _routing_hash(circuit_json, layout_list_str, selected_routing, selected_map)

                                try:
                                    original_qasm_str = qasm2_dumps(qc)
                                except Exception:
                                    original_qasm_str = None
                                routing_analysis_text = generate_routing_analysis(
                                    original_text=original_circuit_text or "",
                                    routed_text=routed_circuit_text or "",
                                    original_qasm=original_qasm_str,
                                    routed_qasm=routed_qasm_str,
                                )
                                session["routing_analysis_text"] = routing_analysis_text
                                session["routing_analysis_hash"] = routed_for_hash

                                _clear_translation_outputs_only()

                            except Exception as e:
                                set_error_for_action(f"Routing stage failed: {type(e).__name__}: {e}")
                                _clear_routing_memory_for_translation()
                                _clear_translation_outputs_only()

        elif action == "compare_routing":
            focus_section = "routing"
            qc = qc_for_view
            if qc is not None:
                if (not layout_list_str.strip()) or (layout_for_hash != _json_hash(circuit_json)):
                    set_error_for_action("Please apply layout before routing")
                else:
                    coupling_map = build_coupling_map(selected_map, qc.num_qubits)
                    if coupling_map is None:
                        set_error_for_action("Routing requires a coupling map. Please select a coupling map first.")
                    else:
                        try:
                            layout_list = ast.literal_eval(layout_list_str)
                            if not isinstance(layout_list, (list, tuple)) or len(layout_list) != qc.num_qubits:
                                raise ValueError("Stored layout is invalid.")
                            layout_list = list(layout_list)
                        except Exception as e:
                            set_error_for_action(f"Invalid stored layout: {type(e).__name__}: {e}")
                            layout_list = None

                        if routing_error is None:
                            methods = ["none", "basic", "lookahead", "sabre"]
                            compare = []
                            for m in methods:
                                doc = ROUTING_DOCS.get(m, "")
                                try:
                                    tq = run_routing_only(qc, coupling_map, layout_list, m)
                                    txt = str(tq.draw(output="text"))
                                    sc = count_ops(tq, "swap")
                                    st = compute_stats(tq)
                                    txt_cx, _ = swap_to_3cx_text(tq)
                                    compare.append({
                                        "method": m,
                                        "docs": doc,
                                        "swap_count": sc,
                                        "stats": st,
                                        "circuit_text": txt,
                                        "circuit_text_cx": txt_cx,
                                    })
                                except Exception as e:
                                    compare.append({
                                        "method": m,
                                        "docs": doc,
                                        "swap_count": None,
                                        "stats": None,
                                        "circuit_text": f"Failed: {type(e).__name__}: {e}",
                                        "circuit_text_cx": None,
                                    })


                            routing_compare_results = compare

                            # --- ChatGPT routing compare analysis (cached) ---
                            try:
                                # testo originale (prendilo dal qc vero, non dipendere dal "Display Qiskit version")
                                original_txt = str(qc.draw(output="text"))
                                try:
                                    original_qasm_str = qasm2_dumps(qc)
                                except Exception:
                                    original_qasm_str = None

                                # hash robusto: circuito + layout + coupling map + contenuto sintetico dei risultati
                                payload_join = "\n".join(
                                    f"{it.get('method','')}|{it.get('swap_count')}|{(it.get('stats') or {}).get('depth','')}|{(it.get('stats') or {}).get('size','')}|{(it.get('stats') or {}).get('cx_depth','')}"
                                    for it in compare
                                )
                                ch = _json_hash(f"{circuit_json}|{layout_list_str}|{selected_map}|{payload_join}")

                                if session.get("routing_compare_analysis_hash", "") != ch:
                                    analysis = generate_routing_comparison_analysis(
                                        original_text=original_txt,
                                        compare_results=compare,
                                        original_qasm=original_qasm_str,
                                    )
                                    session["routing_compare_analysis_text"] = analysis
                                    session["routing_compare_analysis_hash"] = ch

                                routing_compare_analysis_text = session.get("routing_compare_analysis_text", None)
                                routing_compare_analysis_hash = session.get("routing_compare_analysis_hash", "")

                            except Exception as e:
                                routing_compare_analysis_text = f"ChatGPT analysis unavailable: {type(e).__name__}: {e}"
                                routing_compare_analysis_hash = ""
                                session["routing_compare_analysis_text"] = routing_compare_analysis_text
                                session["routing_compare_analysis_hash"] = ""


                            _clear_routing_memory_for_translation()
                            _clear_translation_outputs_only()

                            routed_circuit_text = None
                            routed_stats = None
                            routed_swap_count = None
                            routed_circuit_text_cx = None
                            routed_stats_cx = None
                            routing_notes = None

        elif action == "apply_translation":
            focus_section = "translation"

            if not routed_qasm_str.strip():
                set_error_for_action("Please apply routing first (Translation needs a routed circuit).")
            else:
                expected = _routing_hash(circuit_json, layout_list_str, selected_routing, selected_map)
                if routed_for_hash != expected:
                    set_error_for_action("Routing result is outdated. Please apply routing again before translation.")
                else:
                    try:
                        qc_routed = QuantumCircuit.from_qasm_str(routed_qasm_str)
                        basis_gates = BASIS_GATES_OPTIONS.get(selected_translation_basis)
                        qc_translated = translate_circuit_basis_translator(qc_routed, basis_gates)

                        translated_circuit_text = str(qc_translated.draw(output="text"))
                        translated_stats = compute_stats(qc_translated)

                    except Exception as e:
                        set_error_for_action(f"Translation failed: {type(e).__name__}: {e}")

        if focus_section is None and action:
            focus_section = section_of_action(action)

    if layout_pairs is None and layout_compare_results is None:
        if layout_list_str.strip() and layout_for_hash == _json_hash(circuit_json):
            try:
                ll = ast.literal_eval(layout_list_str)
                if isinstance(ll, (list, tuple)):
                    ll = list(ll)
                    nq = qc_for_view.num_qubits if qc_for_view is not None else num_qubits
                    if len(ll) == nq:
                        layout_pairs = [(i, int(ll[i])) for i in range(nq)]
            except Exception:
                pass

    if routed_circuit_text is None and routing_compare_results is None:
        if routed_qasm_str.strip():
            expected = _routing_hash(circuit_json, layout_list_str, selected_routing, selected_map)
            if routed_for_hash == expected:
                try:
                    qc_routed = QuantumCircuit.from_qasm_str(routed_qasm_str)
                    routed_circuit_text = str(qc_routed.draw(output="text"))
                    routed_stats = compute_stats(qc_routed)
                    routed_swap_count = count_ops(qc_routed, "swap")
                    routed_circuit_text_cx, routed_stats_cx = swap_to_3cx_text(qc_routed)
                    if qc_for_view is not None:
                        routing_notes = routing_comments(qc_for_view, qc_routed)
                except Exception:
                    pass

    layout_permutations = [
        "[" + ", ".join(str(i) for i in perm) + "]"
        for perm in permutations(range(num_qubits))
    ]

    if routing_analysis_text is None:
        routing_analysis_text = session.get("routing_analysis_text", None)

    return render_template(
        "transpile.html",
        num_qubits=num_qubits,
        builder_depth=builder_depth,

        map_options=COUPLING_MAP_OPTIONS,
        layout_options=LAYOUT_OPTIONS,
        routing_options=ROUTING_OPTIONS,

        translation_basis_options=BASIS_GATES_OPTIONS,

        selected_map=selected_map,
        selected_layout=selected_layout,
        selected_routing=selected_routing,

        selected_translation_basis=selected_translation_basis,

        custom_layout_str=custom_layout_str,
        layout_permutations=layout_permutations,

        original_circuit_text=original_circuit_text,
        original_stats=original_stats,

        layout_list_str=layout_list_str,
        layout_for_hash=layout_for_hash,

        layout_pairs=layout_pairs,
        layout_compare_results=layout_compare_results,

        routed_circuit_text=routed_circuit_text,
        routed_stats=routed_stats,
        routed_swap_count=routed_swap_count,
        routed_circuit_text_cx=routed_circuit_text_cx,
        routed_stats_cx=routed_stats_cx,
        routing_notes=routing_notes,
        routing_compare_results=routing_compare_results,

        routing_analysis_text=routing_analysis_text,

        routed_qasm_str=routed_qasm_str,
        routed_for_hash=routed_for_hash,

        translated_circuit_text=translated_circuit_text,
        translated_stats=translated_stats,
        translation_error=translation_error,

        circuit_json=circuit_json,

        editor_error=editor_error,
        layout_error=layout_error,
        routing_error=routing_error,
        focus_section=focus_section,

        docs=DOCS,
        layout_docs=LAYOUT_DOCS,
        routing_docs=ROUTING_DOCS,
        show_qiskit=show_qiskit,
        routing_compare_analysis_text=routing_compare_analysis_text,

    )


@app.route("/synthesis", methods=["GET", "POST"])
def synthesis_view():
    # UNCHANGED (come nel tuo file)
    op_num_qubits = 3
    op_builder_depth = 10
    op_circuit_json = ""

    op_selected_map = "line"
    op_selected_routing = "sabre"
    op_selected_translation_basis = "u3_cx"

    op_error = None
    op_circuit_text = None
    op_stats = None
    op_operator_text = None

    op_synth_circuit_text = None
    op_synth_stats = None
    op_synth_swap_count = None
    op_synth_circuit_text_cx = None
    op_synth_stats_cx = None

    focus_section = "operator"

    if request.method == "POST":
        actions = request.form.getlist("action")
        action = actions[-1] if actions else ""

        focus_section = (request.form.get("focus_section") or "").strip() or "operator"

        op_circuit_json = request.form.get("op_circuit_json", "") or ""
        op_builder_depth = int(request.form.get("op_builder_depth", 10))
        op_num_qubits = int(request.form.get("op_num_qubits", 3))

        op_selected_map = request.form.get("op_map_type", "line")
        op_selected_routing = request.form.get("op_routing_method", "sabre")
        op_selected_translation_basis = request.form.get("op_translation_basis", "u3_cx")

        def _load_op_qc_or_error():
            nonlocal op_error
            if not op_circuit_json.strip():
                op_error = "You must build a circuit first."
                return None
            try:
                return circuit_from_builder_json(op_circuit_json)
            except Exception as e:
                op_error = f"Invalid circuit: {type(e).__name__}: {e}"
                return None

        if action == "op_render_circuit":
            qc = _load_op_qc_or_error()
            if qc is not None:
                op_circuit_text = str(qc.draw(output="text"))
                op_stats = compute_stats(qc)

        elif action == "op_load_qasm":
            file = request.files.get("op_qasm_file", None)
            if file is None or file.filename == "":
                op_error = "Please select a QASM file to load."
            else:
                try:
                    qasm_text = file.read().decode("utf-8", errors="replace")
                    qc_loaded = QuantumCircuit.from_qasm_str(qasm_text)

                    op_circuit_json = builder_json_from_qc(qc_loaded, depth_hint=op_builder_depth)
                    op_num_qubits = qc_loaded.num_qubits
                    op_builder_depth = max(op_builder_depth, 10)

                    op_circuit_text = None
                    op_stats = None
                    op_operator_text = None
                    op_synth_circuit_text = None
                    op_synth_stats = None
                    op_synth_swap_count = None
                    op_synth_circuit_text_cx = None
                    op_synth_stats_cx = None

                except Exception as e:
                    op_error = f"Failed to load QASM: {type(e).__name__}: {e}"

        elif action == "op_generate_operator":
            qc = _load_op_qc_or_error()
            if qc is not None:
                op_circuit_text = str(qc.draw(output="text"))
                op_stats = compute_stats(qc)
                try:
                    op = Operator(qc)
                    op_operator_text = format_operator_matrix(op.data, prec=3, max_dim=16)
                except Exception as e:
                    op_error = f"Operator generation failed: {type(e).__name__}: {e}"

        elif action == "op_apply_synthesis":
            qc = _load_op_qc_or_error()
            if qc is not None:
                op_circuit_text = str(qc.draw(output="text"))
                op_stats = compute_stats(qc)

                try:
                    coupling_map = build_coupling_map(op_selected_map, qc.num_qubits)
                    if coupling_map is None and op_selected_routing != "none":
                        op_error = "Routing requires a coupling map. Please select a coupling map first."
                    else:
                        layout_list = list(range(qc.num_qubits))

                        routed = run_routing_only(
                            qc=qc,
                            coupling_map=coupling_map,
                            layout_list=layout_list,
                            routing_method=op_selected_routing
                        )

                        basis_gates = BASIS_GATES_OPTIONS.get(op_selected_translation_basis)
                        routed = translate_circuit_basis_translator(routed, basis_gates)

                        op_synth_circuit_text = str(routed.draw(output="text"))
                        op_synth_stats = compute_stats(routed)
                        op_synth_swap_count = count_ops(routed, "swap")

                        op_synth_circuit_text_cx, op_synth_stats_cx = swap_to_3cx_text(routed)

                except Exception as e:
                    op_error = f"Synthesis failed: {type(e).__name__}: {e}"

    return render_template(
        "synthesis.html",
        focus_section=focus_section,

        op_num_qubits=op_num_qubits,
        op_builder_depth=op_builder_depth,
        op_circuit_json=op_circuit_json,

        op_selected_map=op_selected_map,
        op_selected_routing=op_selected_routing,
        op_selected_translation_basis=op_selected_translation_basis,

        op_error=op_error,
        op_circuit_text=op_circuit_text,
        op_stats=op_stats,
        op_operator_text=op_operator_text,

        op_synth_circuit_text=op_synth_circuit_text,
        op_synth_stats=op_synth_stats,
        op_synth_swap_count=op_synth_swap_count,
        op_synth_circuit_text_cx=op_synth_circuit_text_cx,
        op_synth_stats_cx=op_synth_stats_cx,

        map_options=COUPLING_MAP_OPTIONS,
        routing_options=ROUTING_OPTIONS,
        translation_basis_options=BASIS_GATES_OPTIONS,
    )


@app.route("/clifford_synthesis", methods=["GET", "POST"])
def clifford_synthesis_view():

    def _cliff_ui_options(n: int):
        if n == 3:
            map_opts = {"line": "Line", "ring": "Ring"}
            rl_enabled = True
        elif n == 5:
            map_opts = {"line": "Line", "ring": "Ring", "star": "Star", "further_map": "Further map"}
            rl_enabled = True
        else:
            map_opts = {"line": "Line", "ring": "Ring"}
            rl_enabled = False

        if rl_enabled:
            algo_opts = {"clifford_greedy": "Greedy", "clifford_ag": "Aaranson Gottesman", "RL": "Topology-Aware Reinforcement Learning-based Cifford synthesizer"}
        else:
            algo_opts = {
                "clifford_greedy": "Greedy",
                "clifford_ag": "AG",
                "RL": "RL (only 3/5 qubits)"
            }

        return map_opts, algo_opts, rl_enabled

    # UI mode: "start" | "build" | "load" | "loaded"
    mode = "start"

    # Synthesis compare (not persisted)
    cliff_synth_compare_results = None

    # Defaults (Clifford-only)
    cliff_num_qubits = 3
    cliff_builder_depth = 10
    cliff_circuit_json = ""

    cliff_selected_map = "line"
    cliff_selected_algo = "clifford_greedy"
    cliff_layout_str = ""

    # Single error + where to show it
    cliff_error = None
    cliff_error_section = ""  # "editor" | "qiskit" | "tableau" | "synthesis" | "operator"

    # Qiskit version toggle (persisted)
    cliff_show_qiskit = session.get("cliff_show_qiskit", False)
    cliff_circuit_text = session.get("cliff_qiskit_text", None) if cliff_show_qiskit else None
    cliff_stats = session.get("cliff_qiskit_stats", None) if cliff_show_qiskit else None

    # Tableau (persisted only if hash matches)
    cliff_tableau_bool_text = None

    # Synthesis output (persisted only if hash matches)
    cliff_synth_circuit_text = None
    cliff_synth_stats = None
    cliff_synth_swap_count = None
    cliff_synth_circuit_text_cx = None
    cliff_synth_stats_cx = None

    focus_section = "clifford"
    last_action = ""

    def _circuit_hash() -> str:
        return _json_hash(cliff_circuit_json or "")

    def _loaded_tableau_hash() -> str:
        tab = session.get("loaded_cliff_tableau")
        if tab is None:
            return ""
        return _json_hash(json.dumps(tab))

    def _has_loaded_operator() -> bool:
        return session.get("loaded_cliff_tableau") is not None

    def _tableau_is_ready() -> bool:
        if _has_loaded_operator() and mode == "loaded":
            return True
        return session.get("cliff_tableau_hash") == _circuit_hash()

    def _clear_synth_session():
        session.pop("cliff_synth_src", None)   # "build" or "loaded"
        session.pop("cliff_synth_hash", None)  # hash circuito o operatore
        session.pop("cliff_synth_text", None)
        session.pop("cliff_synth_stats", None)
        session.pop("cliff_synth_swap_count", None)
        session.pop("cliff_synth_text_cx", None)
        session.pop("cliff_synth_stats_cx", None)

    def _restore_synth_from_session():
        nonlocal cliff_synth_circuit_text, cliff_synth_stats, cliff_synth_swap_count
        nonlocal cliff_synth_circuit_text_cx, cliff_synth_stats_cx

        src = session.get("cliff_synth_src", None)
        h = session.get("cliff_synth_hash", "")

        if src == "build" and h and h == _circuit_hash():
            cliff_synth_circuit_text = session.get("cliff_synth_text", None)
            cliff_synth_stats = session.get("cliff_synth_stats", None)
            cliff_synth_swap_count = session.get("cliff_synth_swap_count", None)
            cliff_synth_circuit_text_cx = session.get("cliff_synth_text_cx", None)
            cliff_synth_stats_cx = session.get("cliff_synth_stats_cx", None)
            return

        if src == "loaded" and h and h == _loaded_tableau_hash():
            cliff_synth_circuit_text = session.get("cliff_synth_text", None)
            cliff_synth_stats = session.get("cliff_synth_stats", None)
            cliff_synth_swap_count = session.get("cliff_synth_swap_count", None)
            cliff_synth_circuit_text_cx = session.get("cliff_synth_text_cx", None)
            cliff_synth_stats_cx = session.get("cliff_synth_stats_cx", None)
            return

        cliff_synth_circuit_text = None
        cliff_synth_stats = None
        cliff_synth_swap_count = None
        cliff_synth_circuit_text_cx = None
        cliff_synth_stats_cx = None

    def _restore_tableau_if_valid():
        nonlocal cliff_tableau_bool_text
        if session.get("cliff_tableau_hash") == _circuit_hash():
            cliff_tableau_bool_text = session.get("cliff_tableau_text", None)
        else:
            cliff_tableau_bool_text = None

    def _set_error(msg: str, section: str):
        nonlocal cliff_error, cliff_error_section
        cliff_error = msg
        cliff_error_section = section or ""

    # ✅ ChatGPT compare analysis (persisted, SEPARATE caches)
    cliff_compare_analysis_text_build = session.get("cliff_compare_analysis_text_build", None)
    cliff_compare_analysis_hash_build = session.get("cliff_compare_analysis_hash_build", "")

    cliff_compare_analysis_text_loaded = session.get("cliff_compare_analysis_text_loaded", None)
    cliff_compare_analysis_hash_loaded = session.get("cliff_compare_analysis_hash_loaded", "")

    if request.method == "POST":
        actions = request.form.getlist("action")
        action = actions[-1] if actions else ""
        last_action = (request.form.get("last_action") or action or "").strip()

        # 🔥 CLEAR_ALL must wipe EVERYTHING
        if action in ("cliff_back_to_start", "clear_all"):
            # tableau (build)
            session.pop("cliff_tableau_hash", None)
            session.pop("cliff_tableau_text", None)

            # loaded operator
            session.pop("loaded_cliff_tableau", None)
            session.pop("loaded_cliff_num_qubits", None)

            # qiskit version
            session.pop("cliff_show_qiskit", None)
            session.pop("cliff_qiskit_text", None)
            session.pop("cliff_qiskit_stats", None)

            # synthesis
            _clear_synth_session()

            # ✅ wipe BOTH compare caches
            session.pop("cliff_compare_analysis_text_build", None)
            session.pop("cliff_compare_analysis_hash_build", None)
            session.pop("cliff_compare_analysis_text_loaded", None)
            session.pop("cliff_compare_analysis_hash_loaded", None)

            mode = "start"
            focus_section = "clifford"

            cliff_circuit_json = ""
            cliff_error = None
            cliff_error_section = ""

            cliff_show_qiskit = False
            cliff_circuit_text = None
            cliff_stats = None

            cliff_tableau_bool_text = None

            cliff_synth_circuit_text = None
            cliff_synth_stats = None
            cliff_synth_swap_count = None
            cliff_synth_circuit_text_cx = None
            cliff_synth_stats_cx = None

            cliff_compare_analysis_text_build = None
            cliff_compare_analysis_hash_build = ""
            cliff_compare_analysis_text_loaded = None
            cliff_compare_analysis_hash_loaded = ""

        else:
            focus_section = (request.form.get("focus_section") or "").strip() or "clifford"
            mode = (request.form.get("mode") or "").strip() or "start"

            cliff_circuit_json = request.form.get("cliff_circuit_json", "") or ""
            cliff_builder_depth = int(request.form.get("cliff_builder_depth", 10))
            cliff_num_qubits = int(request.form.get("cliff_num_qubits", 3))

            cliff_selected_map = request.form.get("cliff_map_type", "line")
            cliff_selected_algo = request.form.get("clifford_algo", "clifford_greedy")
            cliff_layout_str = ""

            # restore persisted outputs (so Display/Hide do not kill tableau/synthesis)
            cliff_show_qiskit = session.get("cliff_show_qiskit", False)
            cliff_circuit_text = session.get("cliff_qiskit_text", None) if cliff_show_qiskit else None
            cliff_stats = session.get("cliff_qiskit_stats", None) if cliff_show_qiskit else None

            _restore_tableau_if_valid()
            _restore_synth_from_session()

            def _load_cliff_qc_or_error(section_for_error: str):
                if not cliff_circuit_json.strip():
                    _set_error("You must build a circuit first.", section_for_error)
                    return None
                try:
                    return circuit_from_builder_json(cliff_circuit_json)
                except Exception as e:
                    _set_error(f"Invalid circuit: {type(e).__name__}: {e}", section_for_error)
                    return None

            # START screen actions
            if action == "cliff_choose_build":
                mode = "build"
                session.pop("cliff_tableau_hash", None)
                session.pop("cliff_tableau_text", None)
                session.pop("loaded_cliff_tableau", None)
                session.pop("loaded_cliff_num_qubits", None)

                # ✅ clear BOTH compare caches
                session.pop("cliff_compare_analysis_text_build", None)
                session.pop("cliff_compare_analysis_hash_build", None)
                session.pop("cliff_compare_analysis_text_loaded", None)
                session.pop("cliff_compare_analysis_hash_loaded", None)
                cliff_compare_analysis_text_build = None
                cliff_compare_analysis_hash_build = ""
                cliff_compare_analysis_text_loaded = None
                cliff_compare_analysis_hash_loaded = ""

                _clear_synth_session()

            elif action == "cliff_choose_load":
                mode = "load"
                _clear_synth_session()

            # Load Clifford operator (npy tableau)
            elif action == "cliff_load_operator":
                mode = "loaded"
                _clear_synth_session()

                # ✅ clear ONLY loaded compare cache (this mode)
                session.pop("cliff_compare_analysis_text_loaded", None)
                session.pop("cliff_compare_analysis_hash_loaded", None)
                cliff_compare_analysis_text_loaded = None
                cliff_compare_analysis_hash_loaded = ""

                # ✅ IMPORTANT: also clear locals in this same response,
                # so previous synthesis can't be rendered until Apply synthesis is clicked.
                cliff_synth_circuit_text = None
                cliff_synth_stats = None
                cliff_synth_swap_count = None
                cliff_synth_circuit_text_cx = None
                cliff_synth_stats_cx = None

                file = request.files.get("cliff_operator_file", None)
                if file is None or file.filename == "":
                    _set_error("Please select a NumPy file (.npy or .npz) containing a Clifford tableau.", "operator")
                else:
                    try:
                        file_bytes = file.read()
                        cliff = clifford_from_tableau_numpy_bytes(file_bytes)

                        cliff_tableau_bool_text = format_clifford_tableau_bool(cliff)

                        tab = np.asarray(cliff.tableau, dtype=bool)
                        session["loaded_cliff_tableau"] = tab.astype(int).tolist()
                        session["loaded_cliff_num_qubits"] = cliff.num_qubits

                        cliff_num_qubits = cliff.num_qubits
                        cliff_circuit_json = ""
                        session["cliff_tableau_hash"] = _loaded_tableau_hash()

                    except ValueError as e:
                        _set_error(str(e), "operator")
                    except Exception as e:
                        _set_error(f"Failed to load Clifford operator: {type(e).__name__}: {e}", "operator")

            # Clear editor (build mode): wipe circuit + tableau + synthesis (but stay in build)
            elif action == "cliff_clear_editor":
                mode = "build"
                focus_section = "build"
                cliff_circuit_json = ""
                session.pop("cliff_tableau_hash", None)
                session.pop("cliff_tableau_text", None)
                _clear_synth_session()

                # do not touch loaded operator (still in build mode anyway)
                # also reset qiskit version display
                session["cliff_show_qiskit"] = False
                session.pop("cliff_qiskit_text", None)
                session.pop("cliff_qiskit_stats", None)
                cliff_show_qiskit = False
                cliff_circuit_text = None
                cliff_stats = None
                cliff_tableau_bool_text = None
                cliff_synth_circuit_text = None
                cliff_synth_stats = None
                cliff_synth_swap_count = None
                cliff_synth_circuit_text_cx = None
                cliff_synth_stats_cx = None

            # Hide Qiskit version (must NOT clear tableau/synthesis)
            elif action == "cliff_hide_qiskit":
                mode = "build"
                session["cliff_show_qiskit"] = False
                session.pop("cliff_qiskit_text", None)
                session.pop("cliff_qiskit_stats", None)
                cliff_show_qiskit = False
                cliff_circuit_text = None
                cliff_stats = None

            # Display Qiskit version (must NOT clear tableau/synthesis)
            elif action == "cliff_render_circuit":
                mode = "build"
                qc = _load_cliff_qc_or_error("qiskit")
                if qc is not None:
                    cliff_show_qiskit = True
                    session["cliff_show_qiskit"] = True

                    cliff_circuit_text = str(qc.draw(output="text"))
                    cliff_stats = compute_stats(qc)

                    session["cliff_qiskit_text"] = cliff_circuit_text
                    session["cliff_qiskit_stats"] = cliff_stats

            # Compute tableau (must NOT clear synthesis; must NOT rely on qiskit render)
            elif action == "cliff_show_tableau":
                mode = "build"
                qc = _load_cliff_qc_or_error("tableau")
                if qc is not None:
                    try:
                        cliff = Clifford(qc)
                        cliff_tableau_bool_text = format_clifford_tableau_bool(cliff)
                        session["cliff_tableau_hash"] = _circuit_hash()
                        session["cliff_tableau_text"] = cliff_tableau_bool_text
                    except Exception as e:
                        _set_error(f"Failed to compute Clifford tableau: {type(e).__name__}: {e}", "tableau")

            # Load QASM (changes circuit => clear tableau+synthesis; keep mode build)
            elif action == "cliff_load_qasm":
                mode = "build"
                file = request.files.get("cliff_qasm_file", None)
                if file is None or file.filename == "":
                    _set_error("Please select a QASM file to load.", "editor")
                else:
                    try:
                        qasm_text = file.read().decode("utf-8", errors="replace")
                        qc_loaded = QuantumCircuit.from_qasm_str(qasm_text)
                        _ = Clifford(qc_loaded)  # raise if not Clifford

                        cliff_circuit_json = builder_json_from_qc(qc_loaded, depth_hint=cliff_builder_depth)
                        cliff_num_qubits = qc_loaded.num_qubits
                        cliff_builder_depth = max(cliff_builder_depth, 10)

                        session.pop("cliff_tableau_hash", None)
                        session.pop("cliff_tableau_text", None)
                        _clear_synth_session()

                        # qiskit version stays as user choice, but content might be outdated -> hide it
                        session["cliff_show_qiskit"] = False
                        session.pop("cliff_qiskit_text", None)
                        session.pop("cliff_qiskit_stats", None)
                        cliff_show_qiskit = False
                        cliff_circuit_text = None
                        cliff_stats = None
                        cliff_tableau_bool_text = None

                    except Exception as e:
                        _set_error(f"Failed to load QASM: {type(e).__name__}: {e}", "editor")

            # Apply synthesis
            elif action == "cliff_apply_synthesis":
                # CASE 1: loaded operator
                if mode == "loaded" and session.get("loaded_cliff_tableau") is not None:
                    try:
                        tab01 = np.array(session["loaded_cliff_tableau"], dtype=int).astype(bool)
                        cliff = Clifford(tab01)
                        cliff_num_qubits = cliff.num_qubits
                        cliff_tableau_bool_text = format_clifford_tableau_bool(cliff)

                        n_eff = cliff.num_qubits
                        if cliff_selected_algo == "RL" and n_eff not in (3, 5):
                            _set_error("RL synthesis is available only for 3- or 5-qubit Clifford operators.", "synthesis")
                        else:
                            coupling_map = build_coupling_map(cliff_selected_map, n_eff)

                            tqc = None
                            if cliff_selected_algo == "clifford_greedy":
                                tqc = synth_clifford_greedy(cliff)
                            elif cliff_selected_algo == "clifford_ag":
                                tqc = synth_clifford_ag(cliff)
                            elif cliff_selected_algo == "RL":
                                rl_synth_qc, terminated = synthesize(cliff, coupling_map)
                                if not terminated:
                                    _set_error("Synthesis failed.", "synthesis")
                                    tqc = None
                                else:
                                    tqc = rl_synth_qc
                            else:
                                raise ValueError("Unknown synthesis algorithm selected.")

                            if tqc is not None and cliff_selected_algo != "RL":
                                if coupling_map is not None:
                                    pm_staged = StagedPassManager()
                                    pm_staged.routing = PassManager(SabreSwap(coupling_map))
                                    tqc = pm_staged.run(tqc)

                                equiv_lib = SessionEquivalenceLibrary
                                target = Target()
                                target.add_instruction(SGate(), name="s")
                                target.add_instruction(HGate(), name="h")
                                target.add_instruction(CXGate(), name="cx")
                                tqc = PassManager(BasisTranslator(equiv_lib, target)).run(tqc)

                            if tqc is not None:
                                cliff_synth_circuit_text = str(tqc.draw(output="text"))
                                cliff_synth_stats = compute_stats(tqc)
                                cliff_synth_swap_count = count_ops(tqc, "swap")
                                cliff_synth_circuit_text_cx, cliff_synth_stats_cx = swap_to_3cx_text(tqc)

                                session["cliff_synth_src"] = "loaded"
                                session["cliff_synth_hash"] = _loaded_tableau_hash()
                                session["cliff_synth_text"] = cliff_synth_circuit_text
                                session["cliff_synth_stats"] = cliff_synth_stats
                                session["cliff_synth_swap_count"] = cliff_synth_swap_count
                                session["cliff_synth_text_cx"] = cliff_synth_circuit_text_cx
                                session["cliff_synth_stats_cx"] = cliff_synth_stats_cx

                    except Exception as e:
                        _set_error(f"Synthesis failed: {type(e).__name__}: {e}", "synthesis")

                # CASE 2: build-mode workflow (requires tableau)
                else:
                    qc = _load_cliff_qc_or_error("synthesis")
                    if qc is not None and not _tableau_is_ready():
                        _set_error("You must click 'Compute tableau' before applying synthesis.", "synthesis")
                    else:
                        n_eff = qc.num_qubits if qc is not None else int(cliff_num_qubits)
                        if cliff_selected_algo == "RL" and n_eff not in (3, 5):
                            _set_error("RL synthesis is available only for 3- or 5-qubit Clifford operators.", "synthesis")
                        else:
                            try:
                                cliff = Clifford(qc)
                            except Exception:
                                _set_error("This circuit is not Clifford. Use only Clifford gates.", "synthesis")
                                cliff = None

                            if qc is not None and cliff is not None:
                                try:
                                    coupling_map = build_coupling_map(cliff_selected_map, qc.num_qubits)

                                    tqc = None
                                    if cliff_selected_algo == "clifford_greedy":
                                        tqc = synth_clifford_greedy(cliff)
                                    elif cliff_selected_algo == "clifford_ag":
                                        tqc = synth_clifford_ag(cliff)
                                    elif cliff_selected_algo == "RL":
                                        rl_synth_qc, terminated = synthesize(cliff, coupling_map)
                                        if not terminated:
                                            _set_error("Synthesis failed.", "synthesis")
                                            tqc = None
                                        else:
                                            tqc = rl_synth_qc
                                    else:
                                        raise ValueError("Unknown synthesis algorithm selected.")

                                    if tqc is not None and cliff_selected_algo != "RL":
                                        if coupling_map is not None:
                                            pm_staged = StagedPassManager()
                                            pm_staged.routing = PassManager(SabreSwap(coupling_map))
                                            tqc = pm_staged.run(tqc)

                                        equiv_lib = SessionEquivalenceLibrary
                                        target = Target()
                                        target.add_instruction(SGate(), name="s")
                                        target.add_instruction(HGate(), name="h")
                                        target.add_instruction(CXGate(), name="cx")
                                        tqc = PassManager(BasisTranslator(equiv_lib, target)).run(tqc)

                                    if tqc is not None:
                                        cliff_synth_circuit_text = str(tqc.draw(output="text"))
                                        cliff_synth_stats = compute_stats(tqc)
                                        cliff_synth_swap_count = count_ops(tqc, "swap")
                                        cliff_synth_circuit_text_cx, cliff_synth_stats_cx = swap_to_3cx_text(tqc)

                                        session["cliff_synth_src"] = "build"
                                        session["cliff_synth_hash"] = _circuit_hash()
                                        session["cliff_synth_text"] = cliff_synth_circuit_text
                                        session["cliff_synth_stats"] = cliff_synth_stats
                                        session["cliff_synth_swap_count"] = cliff_synth_swap_count
                                        session["cliff_synth_text_cx"] = cliff_synth_circuit_text_cx
                                        session["cliff_synth_stats_cx"] = cliff_synth_stats_cx

                                except Exception as e:
                                    _set_error(f"Synthesis failed: {type(e).__name__}: {e}", "synthesis")

            elif action == "compare_synthesis":
                def _compare_synth_hash(mode_: str, algo_map: str, operator_hash: str, compare_payload: str) -> str:
                    # compare_payload può essere anche solo un join dei circuit_text (è ok)
                    return _json_hash(f"{mode_}|{algo_map}|{operator_hash}|{compare_payload}")

                focus_section = "synthesis"

                # reset (not persisted)
                cliff_synth_compare_results = []

                # --- Build the Clifford operator (loaded vs build) ---
                cliff = None

                # CASE 1: loaded operator
                if mode == "loaded" and session.get("loaded_cliff_tableau") is not None:
                    try:
                        tab01 = np.array(session["loaded_cliff_tableau"], dtype=int).astype(bool)
                        cliff = Clifford(tab01)
                        cliff_num_qubits = cliff.num_qubits
                        cliff_tableau_bool_text = format_clifford_tableau_bool(cliff)
                    except Exception as e:
                        _set_error(f"Failed to read loaded operator: {type(e).__name__}: {e}", "synthesis")
                        cliff = None

                # CASE 2: build-mode workflow (requires tableau)
                else:
                    qc = _load_cliff_qc_or_error("synthesis")
                    if qc is not None and not _tableau_is_ready():
                        _set_error("You must click 'Compute tableau' before comparing synthesis.", "synthesis")
                    else:
                        if qc is not None:
                            try:
                                cliff = Clifford(qc)
                            except Exception:
                                _set_error("This circuit is not Clifford. Use only Clifford gates.", "synthesis")
                                cliff = None

                # --- Run compare if operator is valid ---
                if cliff is not None and cliff_error is None:
                    n_eff = cliff.num_qubits
                    coupling_map = build_coupling_map(cliff_selected_map, n_eff)

                    for algo in SYNTH_COMPARE_ORDER:
                        label = SYNTH_METHOD_LABELS.get(algo, algo)
                        doc = SYNTHESIS_DOCS.get(algo, "")

                        # RL availability check
                        if algo == "RL" and n_eff not in (3, 5):
                            cliff_synth_compare_results.append({
                                "method": label,
                                "docs": doc,
                                "error": "RL synthesis is available only for 3- or 5-qubit Clifford operators.",
                                "circuit_text": "",
                                "stats": None,
                                "swap_count": None,
                                "circuit_text_cx": None,
                            })
                            continue

                        try:
                            tqc = None

                            if algo == "clifford_greedy":
                                tqc = synth_clifford_greedy(cliff)
                            elif algo == "clifford_ag":
                                tqc = synth_clifford_ag(cliff)
                            elif algo == "RL":
                                rl_synth_qc, terminated = synthesize(cliff, coupling_map)
                                if not terminated:
                                    raise ValueError("RL synthesis failed.")
                                tqc = rl_synth_qc
                            else:
                                raise ValueError("Unknown synthesis algorithm selected.")

                            # Post-processing for non-RL (same as apply_synthesis)
                            if tqc is not None and algo != "RL":
                                if coupling_map is not None:
                                    pm_staged = StagedPassManager()
                                    pm_staged.routing = PassManager(SabreSwap(coupling_map))
                                    tqc = pm_staged.run(tqc)

                                equiv_lib = SessionEquivalenceLibrary
                                target = Target()
                                target.add_instruction(SGate(), name="s")
                                target.add_instruction(HGate(), name="h")
                                target.add_instruction(CXGate(), name="cx")
                                tqc = PassManager(BasisTranslator(equiv_lib, target)).run(tqc)

                            txt = str(tqc.draw(output="text"))
                            stats = compute_stats(tqc)
                            sw = count_ops(tqc, "swap")
                            txt_cx, _stats_cx = swap_to_3cx_text(tqc)

                            try:
                                qasm_out = qasm2_dumps(tqc)
                            except Exception:
                                qasm_out = None

                            cliff_synth_compare_results.append({
                                "method": label,
                                "docs": doc,
                                "error": None,
                                "circuit_text": txt,
                                "circuit_qasm": qasm_out,
                                "stats": stats,
                                "swap_count": sw,
                                "circuit_text_cx": txt_cx,
                            })

                        except Exception as e:
                            cliff_synth_compare_results.append({
                                "method": label,
                                "docs": doc,
                                "error": f"{type(e).__name__}: {e}",
                                "circuit_text": "",
                                "circuit_qasm": None,
                                "stats": None,
                                "swap_count": None,
                                "circuit_text_cx": None,
                            })

                    # --- ChatGPT compare analysis (cached, SEPARATE per mode) ---
                    try:
                        # operator payload: tableau se disponibile, altrimenti un fallback
                        if mode == "loaded" and session.get("loaded_cliff_tableau") is not None:
                            tab01 = np.array(session["loaded_cliff_tableau"], dtype=int).astype(bool)
                            cliff_tmp = Clifford(tab01)
                            operator_kind = "tableau"
                            operator_payload = format_clifford_tableau_bool(cliff_tmp)
                            op_hash = _loaded_tableau_hash()
                        else:
                            qc_tmp = _load_cliff_qc_or_error("synthesis")
                            cliff_tmp = Clifford(qc_tmp) if qc_tmp is not None else None
                            operator_kind = "tableau"
                            operator_payload = format_clifford_tableau_bool(cliff_tmp) if cliff_tmp is not None else "(n/a)"
                            op_hash = _circuit_hash()

                        payload_join = "\n".join(
                            f"{it.get('method','')}|{it.get('error') or ''}|{(it.get('stats') or {}).get('depth','')}|{(it.get('stats') or {}).get('size','')}|{it.get('swap_count')}"
                            for it in cliff_synth_compare_results
                        )
                        ch = _json_hash(f"{mode}|{cliff_selected_map}|{op_hash}|{payload_join}")

                        if mode == "loaded":
                            text_key = "cliff_compare_analysis_text_loaded"
                            hash_key = "cliff_compare_analysis_hash_loaded"
                        else:
                            text_key = "cliff_compare_analysis_text_build"
                            hash_key = "cliff_compare_analysis_hash_build"

                        if session.get(hash_key, "") != ch:
                            analysis = generate_synthesis_comparison_analysis(
                                operator_kind=operator_kind,
                                operator_payload=operator_payload,
                                compare_results=cliff_synth_compare_results
                            )
                            session[text_key] = analysis
                            session[hash_key] = ch

                        if mode == "loaded":
                            cliff_compare_analysis_text_loaded = session.get(text_key, None)
                            cliff_compare_analysis_hash_loaded = session.get(hash_key, "")
                        else:
                            cliff_compare_analysis_text_build = session.get(text_key, None)
                            cliff_compare_analysis_hash_build = session.get(hash_key, "")

                    except Exception as e:
                        analysis_fallback = f"ChatGPT analysis unavailable: {type(e).__name__}: {e}"
                        if mode == "loaded":
                            cliff_compare_analysis_text_loaded = analysis_fallback
                            cliff_compare_analysis_hash_loaded = ""
                            session["cliff_compare_analysis_text_loaded"] = cliff_compare_analysis_text_loaded
                            session["cliff_compare_analysis_hash_loaded"] = ""
                        else:
                            cliff_compare_analysis_text_build = analysis_fallback
                            cliff_compare_analysis_hash_build = ""
                            session["cliff_compare_analysis_text_build"] = cliff_compare_analysis_text_build
                            session["cliff_compare_analysis_hash_build"] = ""

                    # When comparing, don't show the single "apply" output
                    cliff_synth_circuit_text = None
                    cliff_synth_stats = None
                    cliff_synth_swap_count = None
                    cliff_synth_circuit_text_cx = None
                    cliff_synth_stats_cx = None

    ui_map_options, ui_algo_options, rl_enabled = _cliff_ui_options(int(cliff_num_qubits))

    if cliff_selected_map not in ui_map_options:
        cliff_selected_map = "line"

    rl_note = None
    if cliff_selected_algo == "RL" and not rl_enabled:
        cliff_selected_algo = "clifford_greedy"

    # ✅ IMPORTANT: show analysis only in the current UI mode (no bleed)
    if mode == "build":
        cliff_compare_analysis_text_loaded = None
    elif mode == "loaded":
        cliff_compare_analysis_text_build = None
    else:
        cliff_compare_analysis_text_build = None
        cliff_compare_analysis_text_loaded = None

    return render_template(
        "clifford_synthesis.html",
        focus_section=focus_section,
        last_action=last_action,
        mode=mode,

        cliff_num_qubits=cliff_num_qubits,
        cliff_builder_depth=cliff_builder_depth,
        cliff_circuit_json=cliff_circuit_json,

        cliff_selected_map=cliff_selected_map,
        cliff_selected_algo=cliff_selected_algo,
        cliff_layout_str=cliff_layout_str,

        cliff_error=cliff_error,
        cliff_error_section=cliff_error_section,

        cliff_show_qiskit=cliff_show_qiskit,
        cliff_circuit_text=cliff_circuit_text,
        cliff_stats=cliff_stats,

        cliff_tableau_bool_text=cliff_tableau_bool_text,

        cliff_synth_circuit_text=cliff_synth_circuit_text,
        cliff_synth_stats=cliff_synth_stats,
        cliff_synth_swap_count=cliff_synth_swap_count,
        cliff_synth_circuit_text_cx=cliff_synth_circuit_text_cx,
        cliff_synth_stats_cx=cliff_synth_stats_cx,

        map_options=ui_map_options,
        clifford_algo_options=ui_algo_options,
        rl_enabled=rl_enabled,
        rl_note=rl_note,

        synthesis_docs=SYNTHESIS_DOCS,
        cliff_synth_compare_results=cliff_synth_compare_results,

        # ✅ separate vars (template must use the right one per mode)
        cliff_compare_analysis_text_build=cliff_compare_analysis_text_build,
        cliff_compare_analysis_text_loaded=cliff_compare_analysis_text_loaded,
    )







# ---- System prompt: definisce il tutor ----
TUTOR_SYSTEM_PROMPT = """Sei un tutor esperto di traspilazione di circuiti quantistici Qiskit.
Stai aiutando uno studente universitario a CAPIRE come funziona la traspilazione, non solo a ricevere risposte.

Linee guida pedagogiche (importanti):
1. Approccio socratico: se lo studente fa una domanda concettuale aperta ("perche X?", "come funziona Y?"),
PRIMA fai una domanda di ritorno breve per stimolare il suo ragionamento, POI spiega.
Se la domanda e' molto specifica e tecnica, rispondi direttamente.
2. Usa SEMPRE il circuito e i dati dello studente come esempio concreto.
Non parlare di casi generici se hai il suo contesto.
3. Sii conciso: 3-5 frasi per risposta. Solo se chiede dettagli, espandi.
4. Rispondi in italiano.
5. Niente Markdown elaborato. No headers (#), no tabelle. Liste brevi con trattini ok.
Codice inline tra backtick: `cx q[0],q[1]`.
6. Se la domanda e' ambigua, chiedi chiarimenti invece di indovinare.
7. Se non hai abbastanza contesto per rispondere, dillo esplicitamente.
8. Non inventare fatti su Qiskit. Se non sei sicuro, dillo.

Cosa NON fare:
- Non essere condiscendente ("ottima domanda!", "esattamente!")
- Non ripetere quello che lo studente ha appena detto
- Non dare lunghe lezioni teoriche se non richieste
- Non fornire codice Python a meno che lo studente lo chieda esplicitamente
"""


def _build_context_block(ctx: dict) -> str:
    """Compatta il context inviato dal frontend in un blocco testuale."""
    if not ctx:
        return "(Nessun contesto disponibile.)"

    lines = []

    section_labels = {
        "editor": "sta lavorando sull'editor del circuito",
        "target": "sta scegliendo il coupling map",
        "layout": "sta lavorando sul layout stage",
        "routing": "sta lavorando sul routing stage",
        "translation": "sta lavorando sulla traduzione",
    }
    sec = ctx.get("section")
    if sec:
        lines.append(f"Sezione corrente: lo studente {section_labels.get(sec, sec)}.")

    if ctx.get("num_qubits"):
        lines.append(f"Numero di qubit: {ctx['num_qubits']}.")
    if ctx.get("coupling_map"):
        lines.append(f"Coupling map scelto: {ctx['coupling_map']}.")
    if ctx.get("layout_method"):
        lines.append(f"Metodo di layout scelto: {ctx['layout_method']}.")
    if ctx.get("routing_method"):
        lines.append(f"Metodo di routing scelto: {ctx['routing_method']}.")
    if ctx.get("translation_basis"):
        lines.append(f"Basis di traduzione scelto: {ctx['translation_basis']}.")

    if ctx.get("circuit_text"):
        lines.append(f"\nCircuito originale dello studente:\n{ctx['circuit_text']}")

    if ctx.get("layout_pairs"):
        lines.append(f"\nLayout applicato (logico -> fisico): {ctx['layout_pairs']}")

    if "routed_swap_count" in ctx:
        lines.append(f"\nDopo il routing: {ctx['routed_swap_count']} gate SWAP inseriti.")
    if "routed_depth" in ctx:
        lines.append(f"Depth dopo il routing: {ctx['routed_depth']}.")
    if ctx.get("routed_text"):
        lines.append(f"\nCircuito dopo il routing:\n{ctx['routed_text']}")

    return "\n".join(lines)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Endpoint per il tutor contestuale (Gemini backend).

    Riceve JSON:
    - messages: lista di {role, content} (storia conversazione)
    - context:  dict con stato corrente della pagina

    Risponde JSON: {reply: str, error: bool}
    """
    if not _gemini_is_configured():
        return jsonify({
            "reply": "Il tutor non e' configurato. Verifica GEMINI_API_KEY nelle Environment Variables di Vercel "
                    "e che google-genai sia installato.",
            "error": True,
        })

    try:
        data = request.get_json(silent=True) or {}
        client_messages = data.get("messages", []) or []
        context = data.get("context", {}) or {}

        if not isinstance(client_messages, list):
            return jsonify({"reply": "Formato messaggi non valido.", "error": True})

        ctx_block = _build_context_block(context)
        system_instruction = TUTOR_SYSTEM_PROMPT + "\n\nCONTESTO CORRENTE:\n" + ctx_block

        gemini_contents = []
        recent = client_messages[-20:]
        for m in recent:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                continue

            if len(content) > 4000:
                content = content[:4000] + "... (truncated)"

            gemini_contents.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        if not gemini_contents:
            return jsonify({"reply": "Scrivi una domanda per iniziare.", "error": True})

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = _gemini_client.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=500,
            ),
        )

        reply = (response.text or "").strip() if hasattr(response, "text") else ""

        if not reply:
            try:
                finish_reason = str(response.candidates[0].finish_reason)
            except Exception:
                finish_reason = ""

            if "SAFETY" in finish_reason.upper():
                return jsonify({
                    "reply": "La risposta e' stata bloccata dai filtri di sicurezza di Gemini. Prova a riformulare la domanda.",
                    "error": True,
                })
            if "MAX_TOKENS" in finish_reason.upper():
                return jsonify({
                    "reply": "Risposta troncata per limite di lunghezza. Riprova chiedendo qualcosa di piu' specifico.",
                    "error": True,
                })
            return jsonify({
                "reply": f"(Risposta vuota. Motivo: {finish_reason or 'sconosciuto'}.)",
                "error": True,
            })

        return jsonify({"reply": reply, "error": False})

    except Exception as e:
        msg = str(e)
        lower = msg.lower()

        if "api_key" in lower or "api key" in lower or "unauthenticated" in lower or "401" in msg:
            return jsonify({
                "reply": "API key Gemini invalida o mancante. Controlla GEMINI_API_KEY su Vercel.",
                "error": True,
            })

        if "resource_exhausted" in lower or "429" in msg or "rate" in lower or "quota" in lower:
            return jsonify({
                "reply": "Limite di richieste Gemini raggiunto. Aspetta un minuto e riprova.",
                "error": True,
            })

        if "not found" in lower or ("model" in lower and "404" in msg):
            return jsonify({
                "reply": f"Modello Gemini non disponibile. Verifica GEMINI_MODEL. Errore: {e}",
                "error": True,
            })

        return jsonify({
            "reply": f"Errore: {type(e).__name__}: {e}",
            "error": True,
        })


if __name__ == "__main__":
    # ============================================================
    # Gemini chat API
    # Deve stare PRIMA di: if __name__ == "__main__":
    # ============================================================

    try:
        from google import genai
        from google.genai import types as genai_types

        _gemini_api_key = os.getenv("GEMINI_API_KEY")
        _gemini_client = genai.Client(api_key=_gemini_api_key) if _gemini_api_key else None
    except Exception:
        _gemini_client = None


    TUTOR_SYSTEM_PROMPT = """Sei un tutor esperto di traspilazione di circuiti quantistici Qiskit.
    Stai aiutando uno studente universitario a capire come funziona la traspilazione, non solo a ricevere risposte.

    Linee guida:
    - Rispondi in modo chiaro e breve.
    - Se la domanda è concettuale, puoi spiegare passo passo.
    - Se la domanda riguarda il circuito o il routing, usa il contesto fornito.
    - Non inventare dati non presenti nel contesto.
    - Non usare Markdown pesante.
    """


    def _build_context_block(ctx: dict) -> str:
        """Compatta il context inviato dal frontend in un blocco testuale."""
        if not ctx:
            return "(Nessun contesto disponibile.)"

        lines = []

        section_labels = {
            "editor": "sta lavorando sull'editor del circuito",
            "target": "sta scegliendo il coupling map",
            "layout": "sta lavorando sul layout stage",
            "routing": "sta lavorando sul routing stage",
            "translation": "sta lavorando sulla traduzione",
        }

        sec = ctx.get("section")
        if sec:
            lines.append(f"Sezione corrente: lo studente {section_labels.get(sec, sec)}.")

        if ctx.get("num_qubits"):
            lines.append(f"Numero di qubit: {ctx['num_qubits']}.")

        if ctx.get("coupling_map"):
            lines.append(f"Coupling map scelto: {ctx['coupling_map']}.")

        if ctx.get("layout_method"):
            lines.append(f"Metodo di layout scelto: {ctx['layout_method']}.")

        if ctx.get("routing_method"):
            lines.append(f"Metodo di routing scelto: {ctx['routing_method']}.")

        if ctx.get("translation_basis"):
            lines.append(f"Basis di traduzione scelto: {ctx['translation_basis']}.")

        if ctx.get("circuit_text"):
            lines.append(f"\nCircuito originale dello studente:\n{ctx['circuit_text']}")

        if ctx.get("layout_pairs"):
            lines.append(f"\nLayout applicato, logico -> fisico: {ctx['layout_pairs']}")

        if "routed_swap_count" in ctx:
            lines.append(f"\nDopo il routing: {ctx['routed_swap_count']} gate SWAP inseriti.")

        if "routed_depth" in ctx:
            lines.append(f"Depth dopo il routing: {ctx['routed_depth']}.")

        if ctx.get("routed_text"):
            lines.append(f"\nCircuito dopo il routing:\n{ctx['routed_text']}")

        return "\n".join(lines)


    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """
        Endpoint per il tutor contestuale con Gemini.

        Riceve JSON:
        {
            "messages": [{"role": "user", "content": "..."}],
            "context": {...}
        }

        Risponde JSON:
        {
            "reply": "...",
            "error": false
        }
        """

        if _gemini_client is None:
            return jsonify({
                "reply": "Il tutor non è configurato. Verifica che GEMINI_API_KEY sia impostata su Vercel e che google-genai sia in requirements.txt.",
                "error": True,
            })

        try:
            data = request.get_json(silent=True) or {}
            client_messages = data.get("messages", []) or []
            context = data.get("context", {}) or {}

            if not isinstance(client_messages, list):
                return jsonify({
                    "reply": "Formato messaggi non valido.",
                    "error": True,
                })

            ctx_block = _build_context_block(context)

            prompt_parts = []
            prompt_parts.append(TUTOR_SYSTEM_PROMPT)
            prompt_parts.append("\nCONTESTO CORRENTE:\n" + ctx_block)
            prompt_parts.append("\nCONVERSAZIONE:")

            recent = client_messages[-20:]

            for m in recent:
                if not isinstance(m, dict):
                    continue

                role = m.get("role", "")
                content = (m.get("content", "") or "").strip()

                if not content:
                    continue

                if len(content) > 4000:
                    content = content[:4000] + "..."

                if role == "user":
                    prompt_parts.append(f"\nStudente: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"\nTutor: {content}")

            prompt_parts.append("\nTutor:")

            final_prompt = "\n".join(prompt_parts)

            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

            response = _gemini_client.models.generate_content(
                model=model,
                contents=final_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=600,
                ),
            )

            reply = (getattr(response, "text", "") or "").strip()

            if not reply:
                reply = "(Risposta vuota dal modello. Riprova.)"

            return jsonify({
                "reply": reply,
                "error": False,
            })

        except Exception as e:
            return jsonify({
                "reply": f"Errore Gemini: {type(e).__name__}: {e}",
                "error": True,
            })
    app.run(debug=True)
