"""Quantum Playground · VQE  -  Qollab entry point.

Runs the same variational loop as the interactive playground (VQA_playground.html) headlessly, on whatever
backend you pick in Qollab's "Select QPU" dialog (`backend` is pre-created by the runtime):

    prepare |psi(theta)>  ->  measure <H> on the backend  ->  classical optimizer updates theta  ->  repeat

Problems: H2 (0.735 A), LiH (1.595 A) and stretched LiH (3.0 A), each a 2-qubit STO-3G model (parity mapping).
    H = g0*II + z0*Z0 + z1*Z1 + zz*Z0Z1 + xx*X0X1

One job = one energy evaluation. Both measurement bases (Z for the Z-terms, H-rotated for X0X1) go into ONE
4-qubit circuit (two copies of the state), so a single `backend.run` call measures the whole Hamiltonian.
IonQ backends accept one job per run, so there the optimizer runs on an exact NumPy simulator and the optimum
is then measured with a single job on the chosen backend.
"""
import math
import numpy as np
from qiskit import QuantumCircuit

# ------------------------------------------------------------------ problems
MOLS = {
    "h2":   dict(label="H2, 0.735 A",          g0=-1.0523732, z0=0.39793742, z1=-0.39793742, zz=-0.0112801, xx=0.18093119, nuc=0.7199689944),
    "lih":  dict(label="LiH, 1.595 A",         g0=-8.6096017, z0=0.1711307,  z1=-0.1711307,  zz=-0.0945217, xx=0.013014,   nuc=0.9953176),
    "lih3": dict(label="LiH stretched, 3.0 A", g0=-8.113068,  z0=0.0937509,  z1=-0.0937509,  zz=-0.0605627, xx=0.0610303,  nuc=0.5291772),
}
ALIASES = {"h2": "h2", "lih": "lih", "lih3": "lih3", "lihstretched": "lih3", "stretchedlih": "lih3", "lih3.0": "lih3"}


def hamiltonian_matrix(c):
    """4x4 matrix of H in the basis |q1 q0> (index = 2*q1 + q0)."""
    m = np.zeros((4, 4))
    for i in range(4):
        a, b = 1 - 2 * (i & 1), 1 - 2 * (i >> 1)
        m[i, i] = c["g0"] + c["z0"] * a + c["z1"] * b + c["zz"] * a * b
        m[i, 3 - i] += c["xx"]
    return m


# ------------------------------------------------------------------ ansatz
def build_ops(ansatz, layers):
    """Gate list of (name, qubit, parameter index or None, scale). Hartree-Fock |01> is prepared with an X on qubit 0."""
    ops, n = [("X", 0, None, 0.0)], 0
    if ansatz == "ucc":  # Givens excitation: rotates |01> <-> |10>, one parameter per layer
        for _ in range(layers):
            ops += [("CX", 1, None, 0), ("RY", 1, n, 0.5), ("CX", 0, None, 0), ("RY", 1, n, -0.5), ("CX", 0, None, 0), ("CX", 1, None, 0)]
            n += 1
    else:  # hardware-efficient: Ry on both qubits + CNOT per layer, then a final Ry layer
        for _ in range(layers):
            ops += [("RY", 0, n, 1.0), ("RY", 1, n + 1, 1.0), ("CX", 0, None, 0)]
            n += 2
        ops += [("RY", 0, n, 1.0), ("RY", 1, n + 1, 1.0)]
        n += 2
    return ops, n


def build_circuit(ops, theta, shift=None):
    """4-qubit measurement circuit: qubits 0-1 are measured in Z, qubits 2-3 (a second copy) after H in the X basis."""
    qc = QuantumCircuit(4, 4)
    for off in (0, 2):
        for k, (g, q, p, s) in enumerate(ops):
            a = 0.0 if p is None else s * theta[p]
            if shift is not None and shift[0] == k:
                a += shift[1]
            if g == "X":
                qc.x(q + off)
            elif g == "RY":
                qc.ry(a, q + off)
            else:
                qc.cx(q + off, (1 - q) + off)
    qc.h(2)
    qc.h(3)
    qc.measure(range(4), range(4))
    return qc


def energy_from_counts(counts, c):
    n = sum(counts.values())
    z0 = z1 = zz = xx = 0.0
    for key, k in counts.items():
        s = [1 - 2 * int(ch) for ch in key.replace(" ", "")[::-1]]  # s[i] = +/-1 for classical bit i
        z0 += k * s[0]
        z1 += k * s[1]
        zz += k * s[0] * s[1]
        xx += k * s[2] * s[3]
    z0, z1, zz, xx = z0 / n, z1 / n, zz / n, xx / n
    return c["g0"] + c["z0"] * z0 + c["z1"] * z1 + c["zz"] * zz + c["xx"] * xx, dict(Z0=z0, Z1=z1, Z0Z1=zz, X0X1=xx)


# ------------------------------------------------------------------ exact reference (NumPy statevector)
def _apply(psi, g, q, a):
    if g == "CX":
        out = psi.copy()
        for i in range(4):
            if (i >> q) & 1:
                out[i] = psi[i ^ (1 << (1 - q))]
        return out
    c, s = math.cos(a / 2), math.sin(a / 2)
    u = np.array({"RY": [[c, -s], [s, c]], "X": [[0, 1], [1, 0]]}[g], dtype=complex)
    ax = 1 - q
    return np.moveaxis(np.tensordot(u, psi.reshape(2, 2), axes=([1], [ax])), 0, ax).reshape(4)


def exact_energy(ops, theta, hm, shift=None):
    psi = np.zeros(4, complex)
    psi[0] = 1
    for k, (g, q, p, s) in enumerate(ops):
        a = 0.0 if p is None else s * theta[p]
        if shift is not None and shift[0] == k:
            a += shift[1]
        psi = _apply(psi, g, q, a)
    return float(np.real(psi.conj() @ hm @ psi))


# ------------------------------------------------------------------ optimizer: Adam on parameter-shift gradients
def gradient(ops, theta, energy_fn):
    """dE/dtheta_p = sum over gates using p of scale * (E(angle + pi/2) - E(angle - pi/2)) / 2."""
    g = np.zeros(len(theta))
    for k, (_, _, p, s) in enumerate(ops):
        if p is not None:
            g[p] += s * (energy_fn(theta, (k, math.pi / 2)) - energy_fn(theta, (k, -math.pi / 2))) / 2
    return g


class Adam:
    def __init__(self, n, lr):
        self.lr, self.m, self.v, self.t = lr, np.zeros(n), np.zeros(n), 0

    def step(self, theta, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        return theta - self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


# ------------------------------------------------------------------ talking to the backend
def run_counts(qc, shots):
    try:
        job = backend.run(qc, shots=shots)  # `backend` is pre-created by Qollab: do not create your own
    except Exception as e:
        if "job" in str(e).lower() and "one" in str(e).lower():
            raise  # IonQ's one-job-per-run rule: do not retry
        from qiskit import transpile
        job = backend.run(transpile(qc, backend), shots=shots)  # some backends need the circuit in their native gates
    return job.result().get_counts()


def is_ionq():
    return "ionq" in (type(backend).__module__ + " " + str(getattr(backend, "name", ""))).lower()


def plot(history, ideal, exact, title):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.axhspan(exact - 0.0016, exact + 0.0016, color="#b7e4cf", label="chemical accuracy (1.6 mHa)")
        ax.axhline(exact, color="#0d7a4b", ls="--", lw=1, label="exact ground state")
        ax.plot(history, color="#9aa3b8", marker="o", ms=3, lw=1, label="measured on backend (shot noise)")
        if len(ideal) > 1:
            ax.plot(ideal, color="#1d2b53", lw=2, label="exact energy at the same angles")
        ax.set_xlabel("iteration")
        ax.set_ylabel("energy (Hartree)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plt.show()
    except Exception as e:  # plotting is a nicety, never a failure
        print("(plot skipped: %s)" % e)


def main(shots: int = 2000, iterations: int = 30, molecule: str = "h2", ansatz: str = "ucc", layers: int = 1, learning_rate: float = 0.1, single_job: bool = False):
    key = ALIASES.get(molecule.lower().replace(" ", "").replace("_", "").replace("-", ""), "h2")
    c = MOLS[key]
    hm = hamiltonian_matrix(c)
    exact = float(np.linalg.eigvalsh(hm)[0])
    hf = float(hm[1, 1])
    ops, n = build_ops("ry" if ansatz.lower().startswith("ry") or ansatz.lower().startswith("hard") else "ucc", max(1, layers))
    ansatz_name = "hardware-efficient Ry+CNOT" if n > layers else "UCC-inspired Givens excitation"
    theta = np.random.default_rng(7).uniform(-0.3, 0.3, n)

    print("Quantum Playground · VQE on Qollab")
    print("problem : %s  (2-qubit STO-3G model)" % c["label"])
    print("ansatz  : %s, %d layer(s), %d parameter(s)" % (ansatz_name, layers, n))
    print("backend : %s" % getattr(backend, "name", type(backend).__name__))
    print("exact ground state %.6f Ha | Hartree-Fock %.6f Ha | (add %.4f Ha nuclear repulsion for the total)\n" % (exact, hf, c["nuc"]))

    exact_fn = lambda th, sh=None: exact_energy(ops, th, hm, sh)
    history, ideal = [], []
    if single_job or is_ionq():
        print("One-job mode (IonQ allows one job per run): optimizing on the exact NumPy simulator,\nthen measuring the optimum with a single job on the backend.\n")
        opt = Adam(n, 0.2)
        for _ in range(200):
            theta = opt.step(theta, gradient(ops, theta, exact_fn))
        counts = run_counts(build_circuit(ops, theta), shots)
        e, terms = energy_from_counts(counts, c)
        ideal = exact_fn(theta)
        print("optimal angles          :", np.round(theta, 4))
        print("ideal energy at optimum : %.6f Ha" % ideal)
        print("measured on backend     : %.6f Ha  (%d shots)  error vs exact %.2f mHa" % (e, shots, (e - exact) * 1000))
        print("term expectations       :", {k: round(v, 3) for k, v in terms.items()})
        history, ideal = [e], [ideal]
        final = e
    else:
        energy_fn = lambda th, sh=None: energy_from_counts(run_counts(build_circuit(ops, th, sh), shots), c)[0]
        opt = Adam(n, learning_rate)
        print("iter   measured (Ha)   exact-at-angles (Ha)   measured error (mHa)   backend jobs")
        jobs = 0
        for it in range(iterations):
            e = energy_fn(theta)
            jobs += 1
            history.append(e)
            ideal.append(exact_fn(theta))
            print("%4d   %.6f      %.6f            %8.2f               %d" % (it, e, ideal[-1], (e - exact) * 1000, jobs))
            g = gradient(ops, theta, energy_fn)
            jobs += 2 * sum(1 for o in ops if o[2] is not None)
            theta = opt.step(theta, g)
        big = shots * 10  # final read-out with 10x the shots
        final, terms = energy_from_counts(run_counts(build_circuit(ops, theta), big), c)
        history.append(final)
        ideal.append(exact_fn(theta))
        noise = math.sqrt(sum(c[k] ** 2 * (1 - terms[t] ** 2) for k, t in (("z0", "Z0"), ("z1", "Z1"), ("zz", "Z0Z1"), ("xx", "X0X1"))) / big)
        print("\nfinal read-out (%d shots): %.6f Ha | error vs exact %.2f mHa (shot noise ~ +/-%.1f mHa)" % (big, final, (final - exact) * 1000, noise * 1000))
        print("exact energy at the final angles: %.6f Ha | error %.2f mHa" % (ideal[-1], (ideal[-1] - exact) * 1000))
        print("chemical accuracy (1.6 mHa) %s" % ("reached" if abs(ideal[-1] - exact) < 0.0016 else "not reached yet: raise iterations or lower the learning rate"))
    plot(history, ideal, exact, "%s, %s" % (c["label"], ansatz_name))
    try:
        import json
        from js import sendFile
        sendFile("vqe_result.json", "application/json", json.dumps(dict(problem=c["label"], exact=exact, hartree_fock=hf, final=float(final), theta=[float(x) for x in theta], history=[float(x) for x in history])))
    except Exception:
        pass
