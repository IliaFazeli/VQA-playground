"""WebSocket protocol tests (init / step / run / config) against the real app, no server needed.

    pip install pytest httpx
    pytest -q
"""
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from server import app, mol_data  # noqa: E402

client = TestClient(app)

VQE = dict(cmd="init", mol="h2", ans="ucc", ref="hf", L=1, theta=[0.0], opt="gd", lr=0.3, shots=0)
QAOA = dict(cmd="init", p=2, theta=[0.1, 0.2, 0.3, 0.4], opt="adam", lr=0.1, shots=0, seed=1)


def drain_until(ws, kind, limit=200):
    """Skip stray iterations (a halted run may still deliver its in-flight step) until `kind` arrives."""
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind!r} message received")


# ---------------------------------------------------------------- VQE endpoint

def test_vqe_handshake_and_single_step():
    with client.websocket_connect("/ws/vqe") as ws:
        ws.send_json(VQE)
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["exact"] == mol_data("h2")[1]
        ws.send_json({"cmd": "step"})
        it = ws.receive_json()
        assert it["type"] == "iter" and it["iteration"] == 1 and it["theta"] == [0.0]


def test_vqe_run_streams_then_reports_done():
    with client.websocket_connect("/ws/vqe") as ws:
        ws.send_json(VQE)
        ws.receive_json()
        ws.send_json({"cmd": "run", "delay": 0, "maxiter": 3})
        msgs = [ws.receive_json() for _ in range(4)]
        assert [m["type"] for m in msgs] == ["iter", "iter", "iter", "done"]
        assert [m["iteration"] for m in msgs[:3]] == [1, 2, 3]
        assert msgs[2]["E"] < msgs[0]["E"]


def test_init_while_running_halts_the_old_loop():
    with client.websocket_connect("/ws/vqe") as ws:
        ws.send_json(VQE)
        ws.receive_json()
        ws.send_json({"cmd": "run", "delay": 0.01, "maxiter": 1000})
        assert ws.receive_json()["type"] == "iter"
        ws.send_json(VQE)                                   # re-init while the loop is running
        drain_until(ws, "ready")
        ws.send_json({"cmd": "step"})
        assert ws.receive_json()["iteration"] == 1          # fresh session; nothing else is still running


def test_config_applies_to_the_next_step():
    with client.websocket_connect("/ws/vqe") as ws:
        ws.send_json(VQE)
        ws.receive_json()
        ws.send_json({"cmd": "config", "shots": 50})
        ws.send_json({"cmd": "step"})
        it = ws.receive_json()
        assert all(abs(p * 50 - round(p * 50)) < 1e-9 for p in it["pz"] + it["px"])


def test_bad_init_reports_an_error_and_the_connection_survives():
    with client.websocket_connect("/ws/vqe") as ws:
        ws.send_json({**VQE, "theta": [0.0, 1.0, 2.0]})
        err = ws.receive_json()
        assert err["type"] == "error" and "expected 1" in err["msg"]
        ws.send_json({"cmd": "step"})                       # no session yet: silently ignored
        ws.send_json(VQE)
        assert ws.receive_json()["type"] == "ready"


# ---------------------------------------------------------------- QAOA endpoint

def test_qaoa_stream():
    with client.websocket_connect("/ws/qaoa") as ws:
        ws.send_json(QAOA)
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["n"] == 6 and 0 < ready["ratio"] <= 1
        ws.send_json({"cmd": "run", "delay": 0, "maxiter": 5})
        msgs = [ws.receive_json() for _ in range(6)]
        assert [m["type"] for m in msgs] == ["iter"] * 5 + ["done"]
        assert all({"cut", "ratio", "p_opt", "top", "grad", "theta"} <= m.keys() for m in msgs[:5])
        assert msgs[4]["cut"] > msgs[0]["cut"]


def test_qaoa_custom_graph_and_initial_bitstring():
    with client.websocket_connect("/ws/qaoa") as ws:
        graph = dict(n=3, edges=[[0, 1], [1, 2], [2, 0]])
        ws.send_json({**QAOA, "graph": graph, "p": 1, "theta": [0.0, 0.0], "init": "bits", "init_bits": "100"})
        ready = ws.receive_json()
        assert ready["maxcut"] == 2 and ready["top"][0]["bits"] == "100" and ready["cut"] == 2


def test_qaoa_rejects_grover_from_a_basis_state():
    with client.websocket_connect("/ws/qaoa") as ws:
        ws.send_json({**QAOA, "mixer": "grover", "init": "zero"})
        err = ws.receive_json()
        assert err["type"] == "error" and "Grover" in err["msg"]
        ws.send_json({**QAOA, "mixer": "grover", "init": "dicke"})
        assert ws.receive_json()["type"] == "ready"
