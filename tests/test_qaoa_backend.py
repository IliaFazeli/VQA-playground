"""Unit tests for the MaxCut QAOA half of server.py.

    pip install pytest
    pytest -q
"""
import itertools

import numpy as np
import pytest

from server import DEFAULT_GRAPH, QAOASession


def ring(n):
    return [[i, (i + 1) % n] for i in range(n)]


TRIANGLE = dict(n=3, edges=ring(3))
STAR = dict(n=4, edges=[[0, 1], [0, 2], [0, 3]])
K4_MINUS = dict(n=4, edges=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
PETERSEN = dict(n=10, edges=ring(5) + [[i, i + 5] for i in range(5)] + [[5, 7], [7, 9], [9, 6], [6, 8], [8, 5]])


def cut_of(bits, edges):
    """Independent of the server: bits[i] is the side (0/1) of node i."""
    return sum(1.0 for e in edges if bits[e[0]] != bits[e[1]])


def brute(graph):
    return {"".join(map(str, t)): cut_of("".join(map(str, t)), graph["edges"])
            for t in itertools.product((0, 1), repeat=graph["n"])}


def make(graph=None, p=1, theta=None, **kw):
    m = {"p": p, "seed": 0, **kw}
    if graph:
        m["graph"] = graph
    if theta is not None:
        m["theta"] = list(theta)
    return QAOASession(m)


# ---------------------------------------------------------------- problem encoding

@pytest.mark.parametrize("graph", [TRIANGLE, STAR, DEFAULT_GRAPH, PETERSEN])
def test_cut_table_matches_brute_force(graph):
    s = make(graph)
    table = brute(graph)
    for x in range(2 ** graph["n"]):
        assert s.cutvals[x] == cut_of(s.bits(x), graph["edges"])
    assert s.maxcut == max(table.values())


@pytest.mark.parametrize("graph,expected", [(TRIANGLE, 2), (STAR, 3), (PETERSEN, 12)])
def test_known_maximum_cuts(graph, expected):
    assert make(graph).maxcut == expected


def test_bitstring_convention_character_i_is_node_i():
    s = make(TRIANGLE, theta=[0, 0], init="bits", init_bits="100")
    r = s.evaluate(s.theta, 0)
    assert r["top"][0]["bits"] == "100" and r["top"][0]["prob"] == pytest.approx(1)
    assert r["cut"] == cut_of("100", TRIANGLE["edges"]) == 2


# ---------------------------------------------------------------- circuit conventions

@pytest.mark.parametrize("theta", [[0, 0], [0.9, 0], [0, 0.7]])
def test_trivial_angles_leave_the_uniform_superposition(theta):
    s = make(DEFAULT_GRAPH, theta=theta)
    assert s.evaluate(s.theta, 0)["cut"] == pytest.approx(len(DEFAULT_GRAPH["edges"]) / 2)


@pytest.mark.parametrize("g,b", [(0.7, 0.3), (-1.1, 0.2), (2.0, -0.45), (np.pi / 2, np.pi / 8)])
def test_single_edge_closed_form(g, b):
    s = make(dict(n=2, edges=[[0, 1]]), theta=[g, b])
    assert s.evaluate(s.theta, 0)["cut"] == pytest.approx(0.5 + 0.5 * np.sin(4 * b) * np.sin(g), abs=1e-9)


def test_single_edge_reaches_the_optimum():
    s = make(dict(n=2, edges=[[0, 1]]), theta=[np.pi / 2, np.pi / 8])
    r = s.evaluate(s.theta, 0)
    assert r["ratio"] == pytest.approx(1) and r["p_opt"] == pytest.approx(1)


def apply_1q(psi, u, q, n):
    t = np.tensordot(u, psi.reshape([2] * n), axes=([1], [n - 1 - q]))   # qubit q is axis n-1-q
    return np.moveaxis(t, 0, n - 1 - q).reshape(-1)


def xy_bond(n, a, b, beta):
    """exp(-i*beta*(XX+YY)/2) on qubits a, b: rotates |01> <-> |10>, identity on |00>, |11>."""
    u = np.eye(2 ** n, dtype=complex)
    for x in range(2 ** n):
        if ((x >> a) & 1) != ((x >> b) & 1):
            y = x ^ (1 << a) ^ (1 << b)
            u[x, x], u[y, x] = np.cos(beta), -1j * np.sin(beta)
    return u


def dense_cut(s, theta, p):
    """Reference QAOA with plain matrices: diagonal cost phase, then the chosen mixer."""
    psi = s.init_vec.copy()
    for l in range(p):
        g, b = theta[2 * l], theta[2 * l + 1]
        psi = np.exp(-1j * g * s.cutvals) * psi
        if s.mixer == "x":
            u = np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])
            for q in range(s.n):
                psi = apply_1q(psi, u, q, s.n)
        elif s.mixer == "xy":
            for i in list(range(0, s.n, 2)) + list(range(1, s.n, 2)):
                psi = xy_bond(s.n, i, (i + 1) % s.n, b) @ psi
        else:  # grover: exp(-i b |s><s|)
            psi = psi + (np.exp(-1j * b) - 1) * np.vdot(s.init_vec, psi) * s.init_vec
    return float(np.abs(psi) ** 2 @ s.cutvals)


CASES = [("x", "plus"), ("x", "zero"), ("x", "dicke"), ("xy", "plus"), ("xy", "dicke"),
         ("grover", "plus"), ("grover", "dicke")]


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("mixer,init", CASES)
def test_circuit_matches_dense_matrix_reference(mixer, init, p):
    theta = np.random.default_rng(3).uniform(-1.5, 1.5, 2 * p)
    s = make(K4_MINUS, p=p, theta=theta, mixer=mixer, init=init)
    assert s.evaluate(s.theta, 0)["cut"] == pytest.approx(dense_cut(s, theta, p), abs=1e-9)


# ---------------------------------------------------------------- mixers and initial states

def popcount_table(n):
    return np.array([bin(x).count("1") for x in range(2 ** n)])


@pytest.mark.parametrize("mixer", ["xy", "grover"])
def test_constrained_mixers_stay_in_the_initial_weight_sector(mixer):
    s = make(K4_MINUS, p=2, theta=[0.8, 1.1, -0.4, 0.6], mixer=mixer, init="dicke")
    sv = s._state(s.theta, None)
    assert np.linalg.norm(sv.data) == pytest.approx(1)                  # unitary
    assert sv.probabilities()[popcount_table(4) != 2].sum() < 1e-12


def test_x_mixer_does_leave_the_weight_sector():
    s = make(K4_MINUS, p=2, theta=[0.8, 1.1, -0.4, 0.6], mixer="x", init="dicke")
    assert s._state(s.theta, None).probabilities()[popcount_table(4) != 2].sum() > 1e-3


def test_initial_states():
    n = DEFAULT_GRAPH["n"]
    plus, zero, dicke = (make(init=i) for i in ("plus", "zero", "dicke"))
    assert np.allclose(np.abs(plus.init_vec) ** 2, 1 / 2 ** n)
    assert zero.init_vec[0] == 1 and np.count_nonzero(zero.init_vec) == 1
    support = np.flatnonzero(dicke.init_vec)
    assert all(bin(x).count("1") == n // 2 for x in support) and len(support) == 20     # C(6, 3)
    assert np.linalg.norm(dicke.init_vec) == pytest.approx(1)
    assert make(init="zero").evaluate(np.zeros(2), 0)["cut"] == 0


def test_reachable_cut_with_constrained_mixers():
    assert (make(STAR, mixer="xy", init="dicke").snapshot()["maxcut"],
            make(STAR, mixer="xy", init="dicke").snapshot()["reach"]) == (3.0, 2.0)
    assert make(STAR, mixer="grover", init="dicke").snapshot()["reach"] == 2.0
    assert make(STAR, mixer="x", init="dicke").snapshot()["reach"] == 3.0
    assert make(TRIANGLE, mixer="xy", init="bits", init_bits="000").snapshot()["reach"] == 0.0


# ---------------------------------------------------------------- gradients and training

@pytest.mark.parametrize("mixer,init", [("x", "plus"), ("xy", "dicke"), ("grover", "plus")])
def test_parameter_shift_matches_finite_difference(mixer, init):
    s = make(K4_MINUS, p=2, theta=[0.4, -0.7, 1.2, 0.3], mixer=mixer, init=init)
    g, eps = s.grad(), 1e-5
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        fd = (s.evaluate(s.theta + d, 0)["E"] - s.evaluate(s.theta - d, 0)["E"]) / (2 * eps)
        assert g[i] == pytest.approx(fd, abs=1e-6)


def test_adam_improves_the_expected_cut():
    s = make(DEFAULT_GRAPH, p=2, seed=3)
    start = s.evaluate(s.theta, 0)["cut"]
    for _ in range(40):
        out = s.step()
    assert s.evaluate(s.theta, 0)["cut"] > start + 0.05
    assert out["ratio"] <= 1 + 1e-9 and out["iteration"] == 40


def test_step_returns_the_theta_it_measured_at():
    s = make(DEFAULT_GRAPH, p=2, theta=[0.1, 0.2, 0.3, 0.4])
    out = s.step()
    assert out["theta"] == [0.1, 0.2, 0.3, 0.4] and not np.allclose(s.theta, out["theta"])
    assert out["E"] == pytest.approx(-out["cut"])


# ---------------------------------------------------------------- reporting

def test_snapshot_and_top_list():
    s = make(PETERSEN, theta=[0, 0])
    snap = s.snapshot()
    table = brute(PETERSEN)
    assert snap["type"] == "ready" and snap["n"] == 10 and snap["maxcut"] == 12 and snap["reach"] == 12
    assert 1 <= len(snap["optimal"]) <= 8 and all(table[b] == 12 for b in snap["optimal"])
    probs = [t["prob"] for t in snap["top"]]
    assert len(probs) == 8 and probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-9
    assert all(t["cut"] == table[t["bits"]] for t in snap["top"])
    assert snap["p_opt"] == pytest.approx(sum(v == 12 for v in table.values()) / 1024)


def test_shot_sampling_quantises_probabilities():
    s = make(DEFAULT_GRAPH, theta=[0.4, 0.3], shots=200)
    snap = s.snapshot()
    assert all(abs(t["prob"] * 200 - round(t["prob"] * 200)) < 1e-9 for t in snap["top"])
    assert snap["E_exact"] == pytest.approx(s.evaluate(s.theta, 0)["E"])


def test_seed_controls_the_initial_angles():
    a, b, c = (QAOASession(dict(p=2, seed=sd)) for sd in (5, 5, 6))
    assert np.allclose(a.theta, b.theta) and not np.allclose(a.theta, c.theta)


@pytest.mark.parametrize("kw,msg", [
    (dict(graph=dict(n=1, edges=[[0, 0]])), "n must be"),
    (dict(graph=dict(n=13, edges=[[0, 1]])), "n must be"),
    (dict(graph=dict(n=3, edges=[])), "at least one edge"),
    (dict(graph=dict(n=3, edges=[[1, 1]])), "bad edge"),
    (dict(graph=dict(n=3, edges=[[0, 5]])), "bad edge"),
    (dict(mixer="warp"), "unknown mixer"),
    (dict(init="warp"), "unknown initial state"),
    (dict(mixer="grover", init="zero"), "Grover"),
    (dict(mixer="grover", init="bits"), "Grover"),
    (dict(init="bits", init_bits="01"), "init_bits"),
    (dict(graph=dict(n=2, edges=[[0, 1]]), mixer="xy"), "at least 3"),
    (dict(theta=[0.1, 0.2, 0.3]), "expected 2"),
])
def test_invalid_requests_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        QAOASession(kw)
