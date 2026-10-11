# Quantum Playground: Variational Quantum Algorithms

## What it is
This is a 2026 Qollab Hackathon submission. **An interactive, TensorFlow-Playground-style sandbox that shows how a variational quantum algorithm (VQA) works:
a parameterized circuit, its measurements, and a classical optimizer that keeps readjusting the circuit's angles.**

Hackathon theme: **Quantum Unlocked** 

Variational algorithms are perhaps the leading idea for useful quantum computing on today's noisy devices, but their backbone processes
(prepare, measure, optimize, repeat) are hard to picture. In this project you can watch every stage of it, hover over any
step to see what it actually holds, and change everything: the problem, the ansatz, the optimizer and the number of shots.

## What it does

| Problem | What is solved |
|---|---|
| **VQE: H₂** | Ground-state energy of hydrogen at 0.735 Å (2 qubits, STO-3G) |
| **VQE: LiH** | Lithium hydride at 1.595 Å and stretched to 3.0 Å, where Hartree–Fock starts to fail |
| **QAOA: MaxCut** | Max-cut on small graphs (triangle, 5-cycle, K4, Petersen, random) with three mixers |

**Everything is adjustable:** ansatz (hardware-efficient Ry+CNOT, Ry+Rz+CNOT, UCC-inspired), circuit depth, reference state,
optimizer (gradient descent, Adam, SPSA for VQE), learning rate, shots (exact to 10,000), QAOA layers, initial state and mixer (X, XY ring, Grover).

**It explains itself:**
- The five stages (Prepare, Ansatz, Measure, Cost, Optimize) expand on hover. They show the full state of the qubits,
  one square per measurement shot (so shot noise becomes visible), how each Pauli term adds to the energy, and the
  parameter-shift gradients and the step the optimizer takes.
- Annotated circuits (gates are colour-coded by type) explain what the rotation gates, CNOT, cost layer and each mixer do.
- An interactive molecule explains every Pauli term of the Hamiltonian, what each qubit means and what a 0 or 1 reading tells you.
- For MaxCut you can cut example graphs yourself, see the single curve that slices exactly the cut edges, and learn how QAOA finds good cuts.
- A live energy landscape shows the optimizer's path over the parameters.

## Run it on Qollab

`main.py` is the Qollab entry point. It runs the same VQE loop headlessly on **whatever backend you choose in Select QPU**.
It works on the built-in simulator, the IBM noise-model simulators and the IonQ remote simulators.

1. Open the project on Qollab and press **Run**.
2. Pick a backend (the built-in simulator is the quickest) and, in **Run settings**, a molecule (`h2`, `lih`, `lih stretched`),
   ansatz (`ucc` or `ry`), shots, iterations and learning rate.
3. The console prints every iteration (measured energy, exact energy at the same angles, number of backend jobs) and plots the convergence.

How it uses the backend: one job is one energy evaluation. Both measurement bases (Z for Z₀, Z₁, Z₀Z₁ and H-rotated for X₀X₁) are
packed into a single 4-qubit circuit, and gradients come from the parameter-shift rule. IonQ backends accept one job per run,
so there (or with *Single job* ticked) the optimizer runs on an exact NumPy simulator and the optimum is measured with one job on the chosen backend.

## Run the full interactive playground locally

The playground (`VQA_playground.html`) talks to a Qiskit backend over WebSockets (`server.py`), which cannot run inside Qollab.

```bash
pip install -r requirements-server.txt
uvicorn server:app --reload
```

Then open `VQA_playground.html` in your browser; the status in the top right turns green when the backend is connected.

## How it works

```
 VQA_playground.html  ──WebSocket──▶  server.py (FastAPI + Qiskit)
  circuit, charts, explainers          builds circuits, measures, runs Adam / GD / SPSA,
  (draws state locally)                parameter-shift gradients, sends results back each iteration

 main.py  ──▶  Qollab `backend`       the same loop, headless, on any Qollab simulator
```

**The Hamiltonians.** Each molecule is a 2-qubit model after parity mapping and symmetry tapering:
`H = g₀·II + z₀·Z₀ + z₁·Z₁ + zz·Z₀Z₁ + xx·X₀X₁`.
Find the data used for each on the dashboard.

Energies are electronic, in Hartree (z₁ = −z₀; add the nuclear repulsion, 0.7200 / 0.9953 / 0.5292 Ha, for the total).
The H₂ values match the standard Qiskit coefficients. The LiH values were computed with PySCF (STO-3G, Li 1s² core frozen,
2 electrons in the 2σ/3σ orbitals).

## Roadmap

Future developments could add a bond-length slider, adjustable noise models, and more VQA examples (like the full 4-qubit LiH) or SQD.

## Files

| File | Purpose |
|---|---|
| `main.py` | Headless VQE on the selected backend (just for Qollab)|
| `VQA_playground.html` | The interactive playground (single file, no build step) |
| `server.py` | Qiskit backend for the playground (`/ws/vqe`, `/ws/qaoa`) |
| `requirements.txt` | Read by Qollab (nothing extra needed) |
| `requirements-server.txt` | Dependencies of the local playground server |
| `tests/` | Tests |

## Disclosure and credits

Built during the hackathon window (Oct 9–11, 2026)
The code is fully AI (generated with Claude, then with modifications made on top).
Built on [Qiskit](https://qiskit.org), NumPy, [FastAPI](https://fastapi.tiangolo.com) and the Roboto font; [PySCF](https://pyscf.org) was used once offline, to generate the LiH Hamiltonian coefficients. The layout and look are heavily inspired by the [TensorFlow Playground](https://playground.tensorflow.org).

## License

MIT. See [LICENSE](LICENSE).
