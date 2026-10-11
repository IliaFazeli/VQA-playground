# VQA Playground

An interactive, TensorFlow-Playground-style sandbox for variational quantum algorithms.
Watch a parameterized circuit, its measurements and a classical optimizer work together. This app was built using FastAPI.

- **VQE:** ground-state energy of H₂ (STO-3G, 0.735 Å, 2 qubits), with an interactive explainer of the Hamiltonian
- **QAOA:** MaxCut on small graphs
- **Adjustable:** ansatz, depth, reference state, mixer, optimizer (GD / Adam / SPSA), learning rate, shots
- **Hover over each step** (prepare, ansatz, measure, cost, optimize) to see what happens inside it

## Run

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

Then open `vqa_playground.html` in your browser.

No backend? Set **Engine** to *Browser (JS)* for H₂ VQE. QAOA needs the Qiskit backend.

## Files

| File | Purpose |
|---|---|
| `vqa_playground.html` | Front end (single file, no build step) |
| `server.py` | Qiskit backend over WebSockets (`/ws/vqe`, `/ws/qaoa`) |
| `requirements.txt` | Python dependencies |

## Roadmap

Bond-length sweep, LiH, noise models.
