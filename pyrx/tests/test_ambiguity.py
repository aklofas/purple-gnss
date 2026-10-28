import numpy as np
import pytest

from pyrx.ambiguity import decorrelate, lambda_fix


def _problem(rng, n=6, rho=0.95):
    # strongly correlated float ambiguities, like a few epochs of L1 double differences
    base = np.ones((n, n)) * rho + np.eye(n) * (1 - rho)
    Q = 0.0025 * base  # sigma 0.05 cycles per ambiguity, 0.12 along the common mode
    truth = rng.integers(-50, 50, n)
    a = truth + rng.multivariate_normal(np.zeros(n), Q)
    return a, Q, truth


def test_decorrelate_is_unimodular_and_reconstructs_q():
    rng = np.random.default_rng(1)
    _, Q, _ = _problem(rng)
    Z, L, D = decorrelate(Q)
    assert abs(round(np.linalg.det(Z))) == 1 and Z.dtype.kind == "i"
    Qz = Z @ Q @ Z.T
    np.testing.assert_allclose(L.T @ np.diag(D) @ L, Qz, atol=1e-10, rtol=0)
    c = Qz / np.sqrt(np.outer(np.diag(Qz), np.diag(Qz)))
    # decorrelated: correlations at or below one half, the bound of the integer reduction
    assert np.abs(c - np.eye(len(c))).max() <= 0.5 + 1e-9
    assert np.abs(Q / 0.0025 - np.eye(len(c))).max() > 0.9


def test_lambda_fix_recovers_integers():
    rng = np.random.default_rng(2)
    for _ in range(20):
        a, Q, truth = _problem(rng)
        cands, norms = lambda_fix(a, Q, 2)
        assert cands.shape == (2, 6) and norms[0] <= norms[1]
        np.testing.assert_array_equal(cands[0], truth)


def test_lambda_ratio_orders_candidates():
    rng = np.random.default_rng(3)
    a, Q, _ = _problem(rng)
    cands, norms = lambda_fix(a, Q, 2)
    assert norms[1] / norms[0] > 3.0  # a clear case passes the ratio test
    # an ambiguous float vector (half-integer, uncorrelated) has two equally good candidates
    cands, norms = lambda_fix(np.array([0.5, 1.5]), np.eye(2) * 0.01, 2)
    assert norms[0] == pytest.approx(norms[1], rel=1e-9)  # ratio 1: must not pass a 3.0 test
    assert np.abs(cands - np.array([0.5, 1.5])).max() == 0.5
    assert not np.array_equal(cands[0], cands[1])
