"""LAMBDA integer least squares: decorrelation and the modified (MLAMBDA) search.

Follows RTKLIB's `lambda.c` step for step (LDLᵀ factorization from the last row,
integer Gauss transformations with permutations, depth-first search keeping the
`m` best candidates), so the ratio test sees the same candidates as RTKLIB.
"""

from math import floor, sqrt

import numpy as np

LOOPMAX = 10000  # search iterations before giving up


def _round(x: float) -> float:
    return floor(x + 0.5)


def _sgn(x: float) -> float:
    return -1.0 if x <= 0.0 else 1.0


def _ld(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`Q = Lᵀ diag(D) L` with `L` unit lower triangular."""
    n = len(Q)
    A = Q.astype(float).copy()
    L = np.zeros((n, n))
    D = np.zeros(n)
    for i in range(n - 1, -1, -1):
        D[i] = A[i, i]
        if D[i] <= 0.0:
            raise ValueError("covariance is not positive definite")
        a = sqrt(D[i])
        L[i, : i + 1] = A[i, : i + 1] / a
        for j in range(i):
            A[j, : j + 1] -= L[i, : j + 1] * L[i, j]
        L[i, : i + 1] /= L[i, i]
    return L, D


def _gauss(L, Z, i, j):
    mu = _round(L[i, j])
    if mu != 0.0:
        L[i:, j] -= mu * L[i:, i]
        Z[:, j] -= mu * Z[:, i]


def _perm(L, D, j, delta, Z):
    n = len(D)
    eta = D[j] / delta
    lam = D[j + 1] * L[j + 1, j] / delta
    D[j] = eta * D[j + 1]
    D[j + 1] = delta
    for k in range(j):
        a0, a1 = L[j, k], L[j + 1, k]
        L[j, k] = -L[j + 1, j] * a0 + a1
        L[j + 1, k] = eta * a0 + lam * a1
    L[j + 1, j] = lam
    for k in range(j + 2, n):
        L[k, j], L[k, j + 1] = L[k, j + 1], L[k, j]
    Z[:, [j, j + 1]] = Z[:, [j + 1, j]]


def decorrelate(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer transformation `Z` with `Z Q Zᵀ = Lᵀ diag(D) L` nearly diagonal."""
    n = len(Q)
    L, D = _ld(np.asarray(Q, dtype=float))
    Zr = np.eye(n)  # RTKLIB's Z: z = Zrᵀ a
    j = k = n - 2
    while j >= 0:
        if j <= k:
            for i in range(j + 1, n):
                _gauss(L, Zr, i, j)
        delta = D[j] + L[j + 1, j] ** 2 * D[j + 1]
        if delta + 1e-6 < D[j + 1]:
            _perm(L, D, j, delta, Zr)
            k = j
            j = n - 2
        else:
            j -= 1
    return np.rint(Zr.T).astype(int), L, D


def search(L: np.ndarray, D: np.ndarray, z: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """The `m` integer vectors nearest to `z` in the metric `Lᵀ diag(D) L`, best first."""
    n = len(z)
    S = np.zeros((n, n))
    dist, zb, zi, step = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    zn = np.zeros((m, n))
    s = np.zeros(m)
    nn = imax = 0
    maxdist = 1e99
    k = n - 1
    zb[k] = z[k]
    zi[k] = _round(zb[k])
    y = zb[k] - zi[k]
    step[k] = _sgn(y)
    for _ in range(LOOPMAX):
        newdist = dist[k] + y * y / D[k]
        if newdist < maxdist:
            if k != 0:
                k -= 1
                dist[k] = newdist
                S[k, : k + 1] = S[k + 1, : k + 1] + (zi[k + 1] - zb[k + 1]) * L[k + 1, : k + 1]
                zb[k] = z[k] + S[k, k]
                zi[k] = _round(zb[k])
                y = zb[k] - zi[k]
                step[k] = _sgn(y)
            else:
                if nn < m:
                    if nn == 0 or newdist > s[imax]:
                        imax = nn
                    zn[nn] = zi
                    s[nn] = newdist
                    nn += 1
                else:
                    if newdist < s[imax]:
                        zn[imax] = zi
                        s[imax] = newdist
                        imax = int(np.argmax(s))
                    maxdist = s[imax]
                zi[0] += step[0]
                y = zb[0] - zi[0]
                step[0] = -step[0] - _sgn(step[0])
        else:
            if k == n - 1:
                break
            k += 1
            zi[k] += step[k]
            y = zb[k] - zi[k]
            step[k] = -step[k] - _sgn(step[k])
    else:
        raise ValueError("ambiguity search did not finish")
    order = np.argsort(s[:nn], kind="stable")
    return zn[order].astype(int), s[order]


def lambda_fix(a: np.ndarray, Q: np.ndarray, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """The `m` best integer candidates for float ambiguities `a` with covariance `Q`.

    Returns the candidates `(m, n)` and their squared norms `(a - b)ᵀ Q⁻¹ (a - b)`,
    best first.
    """
    a = np.asarray(a, dtype=float)
    Z, L, D = decorrelate(Q)
    zn, s = search(L, D, Z @ a, m)
    cands = np.linalg.solve(Z.astype(float), zn.T).T
    return np.rint(cands).astype(int), s
