"""Qiskit backend for the Quantum Playground front end.

    pip install fastapi "uvicorn[standard]" qiskit numpy
    uvicorn server:app --reload        ->  ws://localhost:8000/ws/vqe  and  /ws/qaoa
"""
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

app = FastAPI()

# H2 / STO-3G / 0.735 A / parity mapping, 2-qubit reduced (same numbers as the front end)
C = dict(g0=-1.0523732, z0=0.39793742, z1=-0.39793742, zz=-0.0112801, xx=0.18093119)
H = SparsePauliOp.from_list([("II", C["g0"]), ("IZ", C["z0"]), ("ZI", C["z1"]),
                             ("ZZ", C["zz"]), ("XX", C["xx"])])  # rightmost char = qubit 0
HM = H.to_matrix()
EXACT, HF = float(np.linalg.eigvalsh(HM)[0]), float(HM[1, 1].real)
B0 = np.array([1 - 2 * (i & 1) for i in range(4)])
B1 = np.array([1 - 2 * (i >> 1) for i in range(4)])


def build_ops(ans, ref, L):
    """Same circuit as the front end's build()."""
    ops, n = [], [0]

    def add(t, q, p=None, s=1.0):
        ops.append(dict(t=t, q=q, p=p, s=s))

    def par():
        n[0] += 1
        return n[0] - 1

    if ref == "hf":
        add("X", 0)
    if ref == "plus":
        add("H", 0); add("H", 1)
    if ans == "ucc":
        for _ in range(L):
            p = par()
            add("CX", 1); add("RY", 1, p, .5); add("CX", 0); add("RY", 1, p, -.5); add("CX", 0); add("CX", 1)
    else:
        def rot():
            for q in (0, 1):
                add("RY", q, par())
                if ans == "ryrz":
                    add("RZ", q, par())
        for _ in range(L):
            rot(); add("CX", 0)
        rot()
    return ops, n[0]


class Session:
    def __init__(self, m):
        self.ops, n = build_ops(m["ans"], m["ref"], int(m["L"]))
        self.theta = np.array(m["theta"], float)
        if len(self.theta) != n:
            raise ValueError(f"expected {n} parameters, got {len(self.theta)}")
        self.rng = np.random.default_rng()
        self.opt, self.lr, self.shots = None, 0.15, 0
        self.it = self.evals = self.t = self.k = 0
        self.m1, self.m2 = np.zeros(n), np.zeros(n)
        self.configure(m)

    def configure(self, m):
        if m.get("opt") and m["opt"] != self.opt:
            self.opt = m["opt"]
            self.m1, self.m2 = np.zeros_like(self.theta), np.zeros_like(self.theta)
            self.t = self.k = 0
        self.lr = float(m.get("lr", self.lr))
        self.shots = int(m.get("shots", self.shots))

    # ---- the one place that talks to a quantum backend: swap for Aer / Runtime later ----
    def _probs(self, qc, shots):
        p = np.clip(Statevector.from_instruction(qc).probabilities(), 0, None)
        p /= p.sum()
        return p if not shots else self.rng.multinomial(shots, p) / shots

    def evaluate(self, theta, shots, shift=None):
        qc = QuantumCircuit(2)
        for k, o in enumerate(self.ops):
            a = 0.0 if o["p"] is None else o["s"] * theta[o["p"]]
            if shift and shift[0] == k:
                a += shift[1]
            t, q = o["t"], o["q"]
            if t == "RY": qc.ry(a, q)
            elif t == "RZ": qc.rz(a, q)
            elif t == "H": qc.h(q)
            elif t == "X": qc.x(q)
            else: qc.cx(q, 1 - q)
        pz = self._probs(qc, shots)
        qx = qc.copy(); qx.h([0, 1])          # rotate to the X basis
        px = self._probs(qx, shots)
        z0, z1, zz, xx = pz @ B0, pz @ B1, pz @ (B0 * B1), px @ (B0 * B1)
        E = C["g0"] + C["z0"] * z0 + C["z1"] * z1 + C["zz"] * zz + C["xx"] * xx
        return dict(E=float(E), pz=pz.tolist(), px=px.tolist(),
                    ev=dict(g0=1.0, z0=float(z0), z1=float(z1), zz=float(zz), xx=float(xx)))

    def grad(self):
        g = np.zeros_like(self.theta)
        for k, o in enumerate(self.ops):
            if o["p"] is not None:
                ep = self.evaluate(self.theta, self.shots, (k, np.pi / 2))["E"]
                em = self.evaluate(self.theta, self.shots, (k, -np.pi / 2))["E"]
                g[o["p"]] += o["s"] * (ep - em) / 2
                self.evals += 2
        return g

    def snapshot(self):
        r = self.evaluate(self.theta, self.shots)
        ex = r["E"] if not self.shots else self.evaluate(self.theta, 0)["E"]
        return dict(type="ready", exact=EXACT, hf=HF, E=r["E"], E_exact=ex,
                    ev=r["ev"], pz=r["pz"], px=r["px"])

    def step(self):
        th = self.theta
        r = self.evaluate(th, self.shots); self.evals += 1
        ex = r["E"] if not self.shots else self.evaluate(th, 0)["E"]
        if self.opt == "spsa":
            self.k += 1
            c = .25 / self.k ** .1
            d = self.rng.choice([-1.0, 1.0], len(th))
            kk = (self.evaluate(th + c * d, self.shots)["E"] - self.evaluate(th - c * d, self.shots)["E"]) / (2 * c)
            self.evals += 2
            g = d * kk
            new = th - self.lr * g
        else:
            g = self.grad()
            if self.opt == "gd":
                new = th - self.lr * g
            else:  # adam
                self.t += 1
                self.m1 = .9 * self.m1 + .1 * g
                self.m2 = .999 * self.m2 + .001 * g ** 2
                new = th - self.lr * (self.m1 / (1 - .9 ** self.t)) / (np.sqrt(self.m2 / (1 - .999 ** self.t)) + 1e-8)
        self.it += 1
        self.theta = new
        return dict(type="iter", iteration=self.it, evals=self.evals, theta=th.tolist(),
                    grad=g.tolist(), E_exact=ex, **r)


async def serve(ws: WebSocket, make):
    await ws.accept()
    sess, task, state = None, None, {"run": False}

    async def loop(delay, target):
        try:
            while state["run"] and sess.it < target:
                await ws.send_json(await asyncio.to_thread(sess.step))
                await asyncio.sleep(delay)
            if state["run"]:
                await ws.send_json({"type": "done"})
        except Exception:
            pass
        state["run"] = False

    async def halt():
        nonlocal task
        state["run"] = False
        if task:
            await task
            task = None

    try:
        while True:
            m = await ws.receive_json()
            cmd = m.get("cmd")
            if cmd == "init":
                await halt()
                try:
                    sess = make(m)
                    await ws.send_json(await asyncio.to_thread(sess.snapshot))
                except Exception as e:
                    sess = None
                    await ws.send_json({"type": "error", "msg": str(e)})
            elif sess is None:
                continue
            elif cmd == "config":
                sess.configure(m)
            elif cmd == "stop":
                await halt()
            elif cmd == "step" and not state["run"]:
                await ws.send_json(await asyncio.to_thread(sess.step))
            elif cmd == "run" and not state["run"]:
                state["run"] = True
                task = asyncio.create_task(loop(float(m.get("delay", .12)), sess.it + int(m.get("maxiter", 300))))
    except WebSocketDisconnect:
        state["run"] = False


# ===================== MaxCut QAOA =====================
DEFAULT_GRAPH = dict(n=6, edges=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 2], [3, 5], [1, 4]])


class QAOASession(Session):
    """MaxCut QAOA. Minimises E = -<cut>.  theta = [gamma_1, beta_1, ..., gamma_p, beta_p].
    Cost layer  exp(-i*gamma*H_C), H_C = sum_e w*(1 - Z_i Z_j)/2  ->  RZZ(-gamma*w) per edge
    Mixer layer exp(-i*beta*sum_q X_q)                           ->  RX(2*beta) per qubit
    mixer "xy": exp(-i*beta*sum_ring (XX+YY)/2) -> RXX(beta)RYY(beta) per ring bond (conserves Hamming weight)
    mixer "grover": exp(-i*beta*|s><s|), s = initial state (needs a |+> or Dicke start)
    init: "plus" | "zero" | "bits" (init_bits) | "dicke" (n//2 ones)
    """

    def __init__(self, m):
        g = m.get("graph") or DEFAULT_GRAPH
        self.n = int(g["n"])
        if not 2 <= self.n <= 12:
            raise ValueError("n must be between 2 and 12")
        self.edges = []
        for e in g["edges"]:
            i, j, w = int(e[0]), int(e[1]), (float(e[2]) if len(e) > 2 else 1.0)
            if i == j or not (0 <= i < self.n and 0 <= j < self.n):
                raise ValueError(f"bad edge {e}")
            self.edges.append((i, j, w))
        if not self.edges:
            raise ValueError("graph needs at least one edge")
        self.rng = np.random.default_rng(m.get("seed"))
        bits = (np.arange(2 ** self.n)[:, None] >> np.arange(self.n)) & 1   # bits[x, q] = qubit q of state x
        pop = bits.sum(axis=1)
        self.cutvals = sum(w * (bits[:, i] != bits[:, j]) for i, j, w in self.edges).astype(float)
        self.maxcut = float(self.cutvals.max())

        # initial state (prepared exactly, not decomposed into gates)
        self.init = m.get("init", "plus")
        v = np.zeros(2 ** self.n, complex)
        if self.init == "plus":
            v[:] = 1 / np.sqrt(2 ** self.n)
        elif self.init == "zero":
            v[0] = 1
        elif self.init == "bits":
            s = m.get("init_bits") or "".join(str(int(c)) for c in self.rng.integers(0, 2, self.n))
            if len(s) != self.n or set(s) - {"0", "1"}:
                raise ValueError("init_bits must be a 0/1 string with one character per node")
            v[sum(int(c) << q for q, c in enumerate(s))] = 1
        elif self.init == "dicke":
            sel = pop == self.n // 2
            v[sel] = 1 / np.sqrt(sel.sum())
        else:
            raise ValueError(f"unknown initial state {self.init!r}")
        self.init_vec = v

        # cost + mixer layers
        self.mixer = m.get("mixer", "x")
        if self.mixer == "grover" and self.init in ("zero", "bits"):
            raise ValueError("Grover mixer needs a spread-out initial state (|+> or Dicke)")
        p = int(m.get("p", 1))
        self.ops = []
        for l in range(p):
            self.ops += [dict(t="RZZ", q=i, r=j, p=2 * l, s=-w) for i, j, w in self.edges]
            if self.mixer == "x":
                self.ops += [dict(t="RX", q=q, p=2 * l + 1, s=2.0) for q in range(self.n)]
            elif self.mixer == "xy":
                if self.n < 3:
                    raise ValueError("XY ring mixer needs at least 3 nodes")
                for i in list(range(0, self.n, 2)) + list(range(1, self.n, 2)):   # even bonds, then odd bonds
                    a_, b_ = i, (i + 1) % self.n
                    self.ops += [dict(t="RXX", q=a_, r=b_, p=2 * l + 1, s=1.0),
                                 dict(t="RYY", q=a_, r=b_, p=2 * l + 1, s=1.0)]
            elif self.mixer == "grover":
                self.ops.append(dict(t="GROVER", q=None, p=2 * l + 1, s=1.0))
            else:
                raise ValueError(f"unknown mixer {self.mixer!r}")

        # best cut reachable from the initial state (the XY mixer conserves Hamming weight)
        if self.mixer == "xy":
            reach = np.isin(pop, np.unique(pop[np.abs(v) > 0]))
        elif self.mixer == "grover":   # amplitude never leaves the support of the initial state
            reach = np.abs(v) > 0
        else:
            reach = np.ones(2 ** self.n, bool)
        self.maxreach = float(self.cutvals[reach].max())
        self.theta = np.array(m["theta"], float) if m.get("theta") else self.rng.uniform(-.5, .5, 2 * p)
        if len(self.theta) != 2 * p:
            raise ValueError(f"expected {2 * p} parameters, got {len(self.theta)}")
        self.opt, self.lr, self.shots = None, 0.1, 0
        self.it = self.evals = self.t = self.k = 0
        self.m1, self.m2 = np.zeros(2 * p), np.zeros(2 * p)
        self.configure({"opt": "adam", **m})

    def _state(self, theta, shift):
        sv, qc = Statevector(self.init_vec), QuantumCircuit(self.n)
        for k, o in enumerate(self.ops):
            a = 0.0 if o["p"] is None else o["s"] * theta[o["p"]]
            if shift and shift[0] == k:
                a += shift[1]
            t = o["t"]
            if t == "GROVER":   # exp(-i a |s><s|), applied exactly on the statevector
                if qc.size():
                    sv, qc = sv.evolve(qc), QuantumCircuit(self.n)
                amp = np.vdot(self.init_vec, sv.data)
                sv = Statevector(sv.data + (np.exp(-1j * a) - 1) * amp * self.init_vec)
            elif t == "RX": qc.rx(a, o["q"])
            elif t == "RXX": qc.rxx(a, o["q"], o["r"])
            elif t == "RYY": qc.ryy(a, o["q"], o["r"])
            else: qc.rzz(a, o["q"], o["r"])
        return sv.evolve(qc) if qc.size() else sv

    def evaluate(self, theta, shots, shift=None):
        probs = np.clip(self._state(theta, shift).probabilities(), 0, None)
        probs /= probs.sum()
        if shots:
            probs = self.rng.multinomial(shots, probs) / shots
        cut = float(probs @ self.cutvals)
        top = np.argsort(probs)[::-1][:8]
        return dict(E=-cut, cut=cut, ratio=cut / self.maxcut,
                    p_opt=float(probs[self.cutvals >= self.maxcut - 1e-9].sum()),
                    top=[dict(bits=self.bits(x), prob=float(probs[x]), cut=float(self.cutvals[x])) for x in top])

    def bits(self, x):  # node 0 first: character i is the side (0/1) of node i
        return "".join(str(int(x) >> q & 1) for q in range(self.n))

    def snapshot(self):
        r = self.evaluate(self.theta, self.shots)
        ex = r["E"] if not self.shots else self.evaluate(self.theta, 0)["E"]
        best = [self.bits(x) for x in np.flatnonzero(self.cutvals >= self.maxcut - 1e-9)[:8]]
        return dict(type="ready", n=self.n, edges=[list(e) for e in self.edges], maxcut=self.maxcut,
                    optimal=best, reach=self.maxreach, theta=self.theta.tolist(), E_exact=ex, **r)


@app.websocket("/ws/vqe")
async def ws_vqe(ws: WebSocket):
    await serve(ws, Session)


@app.websocket("/ws/qaoa")
async def ws_qaoa(ws: WebSocket):
    await serve(ws, QAOASession)
