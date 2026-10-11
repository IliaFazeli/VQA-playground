"""Unit tests for the VQE half of server.py.

    pip install pytest
    pytest -q
"""
import numpy as np
import pytest

from server import MOLS, Session, build_ops, mol_data

MOL_NAMES = list(MOLS)
I2, X, Z = np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])


def hamiltonian(c):
    """Independent numpy build; kron(q1, q0) so qubit 0 is the least significant bit."""
    return (c["g0"] * np.eye(4) + c["z0"] * np.kron(I2, Z) + c["z1"] * np.kron(Z, I2)
            + c["zz"] * np.kron(Z, Z) + c["xx"] * np.kron(X, X))


def make(mol="h2", ans="ucc", ref="hf", L=1, opt="gd", lr=0.3, shots=0, theta=None):
    n = build_ops(ans, ref, L)[1]
    theta = np.zeros(n) if theta is None else theta
    return Session(dict(mol=mol, ans=ans, ref=ref, L=L, opt=opt, lr=lr, shots=shots, theta=list(theta)))


# ---------------------------------------------------------------- Hamiltonians

@pytest.mark.parametrize("mol", MOL_NAMES)
def test_reference_energies_match_independent_matrix(mol):
    c, exact, hf = mol_data(mol)
    h = hamiltonian(c)
    assert exact == pytest.approx(np.linalg.eigvalsh(h)[0], abs=1e-9)
    assert hf == pytest.approx(h[1, 1], abs=1e-9)                       # |01>: qubit 0 occupied
    assert hf == pytest.approx(c["g0"] - c["z0"] + c["z1"] - c["zz"], abs=1e-9)
    assert exact < hf                                                   # correlation energy is negative


def test_correlation_grows_with_bond_length():
    def corr(m):
        _, exact, hf = mol_data(m)
        return hf - exact
    assert corr("lih3") > corr("lih") > 0                               # lih3 = stretched LiH


def test_unknown_molecule_is_rejected():
    with pytest.raises(KeyError):
        make(mol="unobtainium")


# ---------------------------------------------------------------- circuits

@pytest.mark.parametrize("mol", MOL_NAMES)
def test_ucc_at_zero_is_hartree_fock(mol):
    s = make(mol)
    assert s.evaluate(s.theta, 0)["E"] == pytest.approx(s.hf, abs=1e-9)


@pytest.mark.parametrize("mol", MOL_NAMES)
def test_ucc_scan_reaches_exact_ground_state(mol):
    s = make(mol)
    best = min(s.evaluate(np.array([t]), 0)["E"] for t in np.linspace(-np.pi, np.pi, 401))
    assert best == pytest.approx(s.exact, abs=1e-4)


@pytest.mark.parametrize("ans,ref,L", [("ry", "hf", 2), ("ryrz", "plus", 1), ("ucc", "zero", 2), ("ry", "zero", 1)])
def test_variational_principle(ans, ref, L):
    n = build_ops(ans, ref, L)[1]
    rng = np.random.default_rng(11)
    s = make(ans=ans, ref=ref, L=L, theta=rng.uniform(-np.pi, np.pi, n))
    for _ in range(25):
        assert s.evaluate(rng.uniform(-np.pi, np.pi, n), 0)["E"] >= s.exact - 1e-9


def test_probabilities_are_normalised_and_shots_are_quantised():
    n = build_ops("ry", "plus", 2)[1]
    s = make(ans="ry", ref="plus", L=2, theta=np.random.default_rng(1).uniform(-1, 1, n))
    s.rng = np.random.default_rng(0)
    r = s.evaluate(s.theta, 0)
    assert sum(r["pz"]) == pytest.approx(1) and sum(r["px"]) == pytest.approx(1)
    r = s.evaluate(s.theta, 1000)
    assert all(abs(p * 1000 - round(p * 1000)) < 1e-9 for p in r["pz"] + r["px"])
    assert r["E"] == pytest.approx(s.evaluate(s.theta, 0)["E"], abs=0.15)   # ~5 sigma


# ---------------------------------------------------------------- gradients and optimisers

@pytest.mark.parametrize("ans,L", [("ry", 2), ("ryrz", 1), ("ucc", 2)])
def test_parameter_shift_matches_finite_difference(ans, L):
    n = build_ops(ans, "hf", L)[1]
    s = make(ans=ans, L=L, theta=np.random.default_rng(7).uniform(-np.pi, np.pi, n))
    g, eps = s.grad(), 1e-5
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        fd = (s.evaluate(s.theta + d, 0)["E"] - s.evaluate(s.theta - d, 0)["E"]) / (2 * eps)
        assert g[i] == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("opt,lr,steps,tol", [("gd", 0.3, 150, 1e-6), ("spsa", 0.3, 250, 1e-6), ("adam", 0.1, 400, 1e-3)])
@pytest.mark.parametrize("mol", MOL_NAMES)
def test_optimisers_converge_on_ucc(mol, opt, lr, steps, tol):
    s = make(mol, opt=opt, lr=lr)
    s.rng = np.random.default_rng(0)
    for _ in range(steps):
        s.step()
    gap = s.evaluate(s.theta, 0)["E"] - s.exact
    assert -1e-9 <= gap < tol


def test_step_bookkeeping():
    s = make(opt="gd", theta=np.array([0.2]))
    out = s.step()
    assert out["type"] == "iter" and out["iteration"] == 1
    assert out["theta"] == [0.2]                         # theta the energy was measured at
    assert s.theta[0] != 0.2                             # the session has moved on
    assert out["evals"] == 5                             # 1 energy + 2 shifts x 2 UCC rotation gates
    assert out["E"] == pytest.approx(out["E_exact"])     # exact mode
    assert {"grad", "pz", "px", "ev"} <= out.keys()
    assert s.step()["iteration"] == 2


def test_configure_resets_adam_state_only_when_optimiser_changes():
    s = make(opt="adam", lr=0.1)
    s.step()
    assert s.t == 1 and np.any(s.m1 != 0)
    s.configure({"lr": 0.2, "shots": 50})
    assert (s.t, s.lr, s.shots) == (1, 0.2, 50)
    s.configure({"opt": "gd"})
    assert s.opt == "gd" and s.t == 0 and not np.any(s.m1)


def test_wrong_number_of_parameters_is_rejected():
    with pytest.raises(ValueError, match="expected 1"):
        make(theta=np.zeros(3))


def test_snapshot_reports_reference_energies():
    s = make("lih")
    snap = s.snapshot()
    assert snap["type"] == "ready" and snap["exact"] == s.exact and snap["hf"] == s.hf
    assert snap["E"] == pytest.approx(s.hf)
