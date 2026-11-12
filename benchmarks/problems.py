"""Canonical 9-instance blind benchmark problem suite for fortchuck.

Implements exact closed-form smooth objectives and Euclidean partial gradients
(grad_z f, grad_{X_1} f, ..., grad_{X_R} f) following:
  - docs/ARCHITECTURE_SPEC.md (§6.2)
  - docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md (§6)

Maintaining strict separation from Fortran solver internals (`src/*.f90`), all
problem matrices are deterministically generated from fixed canonical problem
seeds via PCG64, independent of initial-point seeds (101..110 or 201..210).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import numpy as np

BENCHMARK_IDS: tuple[str, ...] = (
    "B1",
    "B2A",
    "B2B",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7A",
    "B7B",
)

EvalFGCallable = Callable[
    [np.ndarray, Sequence[np.ndarray]],
    tuple[float, np.ndarray, list[np.ndarray]],
]


def _random_orthonormal(
    rng: np.random.Generator,
    n: int,
    p: int,
    special_orthogonal: bool = False,
) -> np.ndarray:
    """Draw a deterministic Haar-distributed Stiefel / SO(n) frame via positive-R QR."""
    raw = rng.standard_normal((n, p))
    q_mat, r_mat = np.linalg.qr(raw, mode="reduced")
    diag_r = np.diag(r_mat)
    signs = np.where(diag_r >= 0.0, 1.0, -1.0)
    q_mat = q_mat * signs[None, :]
    if special_orthogonal and n == p and np.linalg.det(q_mat) < 0.0:
        q_mat[:, -1] *= -1.0
    return np.asfortranarray(q_mat, dtype=np.float64)


@dataclass(frozen=True)
class BenchmarkProblem:
    """Immutable specification of a Coupled Box x Multi-Stiefel benchmark instance."""

    problem_id: str
    name: str
    regime_summary: str
    d: int
    lower: np.ndarray
    upper: np.ndarray
    n_list: tuple[int, ...]
    p_list: tuple[int, ...]
    eval_fg: EvalFGCallable
    f_star: float | None = None
    z_star: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def R(self) -> int:
        return len(self.n_list)

    @property
    def n_stiefel_tot(self) -> int:
        return int(sum(n * p for n, p in zip(self.n_list, self.p_list)))

    @property
    def n_tot(self) -> int:
        return int(self.d + self.n_stiefel_tot)

    @property
    def intrinsic_dim(self) -> int:
        st_dim = sum(n * p - (p * (p + 1)) // 2 for n, p in zip(self.n_list, self.p_list))
        return int(self.d + st_dim)


def _build_b1_ensemble_dft() -> BenchmarkProblem:
    """B1: Ensemble Kohn-Sham DFT with Fractional Occupations.

    Domain: z in [0, 1]^10 x St(120, 10) (Branch A: 2p = 20 < n = 120).
    Tests small-gap metallic orbital-occupation cross-Hessian sloshing and
    clamping at both core (z_i* = 1) and virtual (z_i* = 0) orbitals.
    """
    rng = np.random.Generator(np.random.PCG64(202601))
    n, p = 120, 10
    dom_len = 24.0
    dx = dom_len / n
    grid = np.linspace(0.0, dom_len, n, endpoint=False)

    # 1D periodic kinetic operator -0.5 * d^2/dx^2 (second-order finite difference)
    lap = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    lap[idx, idx] = -2.0 / (dx * dx)
    lap[idx, (idx + 1) % n] = 1.0 / (dx * dx)
    lap[idx, (idx - 1) % n] = 1.0 / (dx * dx)

    # Asymmetric double-well dimer external potential inducing small frontier gap
    v_ext = (
        -3.6 * np.exp(-0.5 * ((grid - 0.36 * dom_len) / 1.35) ** 2)
        - 3.45 * np.exp(-0.5 * ((grid - 0.64 * dom_len) / 1.35) ** 2)
        + 0.12 * np.cos(2.0 * np.pi * grid / dom_len)
        + 0.04 * rng.standard_normal(n)
    )
    h0 = -0.5 * lap + np.diag(v_ext)
    h0 = 0.5 * (h0 + h0.T)

    # Positive-definite screened Yukawa Hartree interaction kernel V in S_{++}^n
    diff = np.abs(grid[:, None] - grid[None, :])
    per_dist = np.minimum(diff, dom_len - diff)
    v_hartree = 0.45 * np.exp(-0.28 * per_dist) / np.sqrt(per_dist * per_dist + 0.36)
    v_hartree = 0.5 * (v_hartree + v_hartree.T) + 1.0e-3 * np.eye(n)

    n_elec = 5.5
    temp_e = 0.08
    delta_fd = 1.0e-4
    kappa_n = 10.0

    lower = np.zeros(p, dtype=np.float64)
    upper = np.ones(p, dtype=np.float64)

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = x_list[0]
        one_minus_d = 1.0 - delta_fd
        a = delta_fd + one_minus_d * z
        b = 1.0 - one_minus_d * z
        if np.any(a <= 0.0) or np.any(b <= 0.0) or not (np.all(np.isfinite(z)) and np.all(np.isfinite(X))):
            return (
                float("nan"),
                np.full(p, np.nan, dtype=np.float64),
                [np.full((n, p), np.nan, dtype=np.float64, order="F")],
            )

        # Electron density rho = (X odot X) z
        x_sq = X * X
        rho = x_sq @ z
        v_rho = v_hartree @ rho

        # Fock-like action F_X = (H_0 + diag(V rho)) X
        hx = h0 @ X
        fx = hx + v_rho[:, None] * X

        e_kin_ext = 0.5 * float(np.sum((X * hx) @ z))
        e_hartree = 0.25 * float(rho @ v_rho)

        # Regularized Fermi-Dirac entropy S_delta(z)
        log_a = np.log(a)
        log_b = np.log(b)
        s_delta = -float(np.sum(z * log_a + (1.0 - z) * log_b))
        ds_dz = -(log_a - log_b + (one_minus_d * z) / a - (one_minus_d * (1.0 - z)) / b)

        occ_err = float(np.sum(z) - n_elec)
        e_occ = 0.5 * kappa_n * occ_err * occ_err

        f_val = e_kin_ext + e_hartree - temp_e * s_delta + e_occ

        rayleigh_diag = 0.5 * np.sum(X * fx, axis=0)
        gz = rayleigh_diag - temp_e * ds_dz + kappa_n * occ_err
        gx = np.asfortranarray(fx * z[None, :], dtype=np.float64)
        return float(f_val), gz.astype(np.float64, copy=False), [gx]

    return BenchmarkProblem(
        problem_id="B1",
        name="Ensemble Kohn-Sham DFT Fractional Occupations",
        regime_summary="z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)",
        d=p,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        metadata={"n_elec": n_elec, "temp_e": temp_e, "delta_fd": delta_fd, "kappa_n": kappa_n},
    )


def _build_b2a_amplitude_limited_ica() -> BenchmarkProblem:
    """B2A: Amplitude-Limited Blind Source Separation (ICA).

    Domain: Z in [-2, 2]^{5 x 150} (d = 750) x W in St(20, 5) (Branch A).
    Tests nonconvex log-cosh source recovery with d >> n*p and ~30% active bounds.
    """
    rng = np.random.Generator(np.random.PCG64(202602))
    n, p, m_samples = 20, 5, 150
    d = p * m_samples
    amp_limit = 2.0
    lam_ica = 0.65

    w_true = _random_orthonormal(rng, n, p)
    # Impulsive / Laplace-like latent sources so ~30% exceed amplitude limit +-2
    s_true = 1.65 * rng.laplace(loc=0.0, scale=1.0, size=(p, m_samples))
    noise = 0.12 * rng.standard_normal((n, m_samples))
    y_obs = w_true @ s_true + noise

    lower = np.full(d, -amp_limit, dtype=np.float64)
    upper = np.full(d, amp_limit, dtype=np.float64)
    log2 = float(np.log(2.0))

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        W = x_list[0]
        Z = z.reshape((p, m_samples), order="F")
        wty = W.T @ y_obs
        diff = Z - wty

        # Overflow-safe log(cosh(Z)) = |Z| + log1p(exp(-2|Z|)) - log(2)
        abs_z = np.abs(Z)
        log_cosh_z = abs_z + np.log1p(np.exp(-2.0 * abs_z)) - log2

        f_val = 0.5 * float(np.sum(diff * diff)) - lam_ica * float(np.sum(log_cosh_z))
        gz_mat = diff - lam_ica * np.tanh(Z)
        gw = np.asfortranarray(y_obs @ (-diff.T), dtype=np.float64)
        return float(f_val), gz_mat.reshape(d, order="F"), [gw]

    return BenchmarkProblem(
        problem_id="B2A",
        name="Amplitude-Limited Blind Source Separation (ICA)",
        regime_summary="Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        metadata={"m_samples": m_samples, "lambda_ica": lam_ica, "amp_limit": amp_limit},
    )


def _build_b2b_nonnegative_sparse_pca() -> BenchmarkProblem:
    """B2B: Smooth Penalty-Split Non-Negative Sparse PCA.

    Domain: Z in [0, 5]^{120 x 6} (d = 720) x X in St(120, 6) (Branch A).
    Tests FM-2 Lagrange-multiplier masking when ~80% of Z_ij clamp at 0 with
    strictly positive multipliers mu - rho * X_ij > 0.
    """
    rng = np.random.Generator(np.random.PCG64(202603))
    n, p = 120, 6
    d = n * p
    rho_pen = 25.0
    mu_l1 = 2.5

    # Construct disjoint sparse non-negative principal loadings (18/120 support per factor)
    v_sp = np.zeros((n, p), dtype=np.float64)
    block_sz = 18
    for col in range(p):
        idx_start = col * block_sz
        weights = rng.uniform(0.5, 1.5, size=block_sz)
        v_sp[idx_start : idx_start + block_sz, col] = weights / np.linalg.norm(weights)

    spikes = np.array([14.0, 11.5, 9.2, 7.3, 5.6, 4.1], dtype=np.float64)
    u_bg = _random_orthonormal(rng, n, n)
    bg_eigs = np.linspace(0.15, 0.85, n, dtype=np.float64)
    sigma = (v_sp * spikes[None, :]) @ v_sp.T + (u_bg * bg_eigs[None, :]) @ u_bg.T
    sigma = 0.5 * (sigma + sigma.T)

    lower = np.zeros(d, dtype=np.float64)
    upper = np.full(d, 5.0, dtype=np.float64)

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = x_list[0]
        Z = z.reshape((n, p), order="F")
        sig_x = sigma @ X
        diff = X - Z

        f_val = (
            -0.5 * float(np.sum(X * sig_x))
            + 0.5 * rho_pen * float(np.sum(diff * diff))
            + mu_l1 * float(np.sum(Z))
        )
        gz_mat = rho_pen * (Z - X) + mu_l1
        gx = np.asfortranarray(-sig_x + rho_pen * diff, dtype=np.float64)
        return float(f_val), gz_mat.reshape(d, order="F"), [gx]

    return BenchmarkProblem(
        problem_id="B2B",
        name="Smooth Penalty-Split Non-Negative Sparse PCA",
        regime_summary="Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        metadata={"rho": rho_pen, "mu": mu_l1},
    )


def _build_b3_multicamera_wahba() -> BenchmarkProblem:
    """B3: Multi-Camera Anisotropic Bounded-Scale Rig Self-Calibration (Wahba).

    Domain: z in B^15 x St(3, 3)^4 (R = 4, Branch B Direct LU on SO(3)^4, N_tot = 51).
    Tests multi-factor product loops, 2p >= n direct LU Cayley dispatch, and
    kappa = 10^3 anisotropic sensor precision with active scale/translation bounds.
    """
    rng = np.random.Generator(np.random.PCG64(202604))
    r_cams = 4
    m_pts = 25
    d = 3 + 3 * r_cams

    landmarks = rng.standard_normal((3, m_pts))
    s_raw = np.array([0.35, 1.25, 2.35], dtype=np.float64)  # Clamps at 0.5 and 2.0
    t_raw = np.array(
        [
            [-1.85, 0.40, 1.10],
            [0.75, -1.75, 0.25],
            [-0.30, 1.80, -0.90],
            [1.65, -0.55, 0.80],
        ],
        dtype=np.float64,
    )
    rot_true = [_random_orthonormal(rng, 3, 3, special_orthogonal=True) for _ in range(r_cams)]

    # Anisotropic precision matrices W_{r,m} in S_{++}^3 with exact condition number 10^3
    eig_profile = np.array([0.02, 0.02 * np.sqrt(1000.0), 20.0], dtype=np.float64) / m_pts
    w_prec = np.zeros((r_cams, m_pts, 3, 3), dtype=np.float64)
    q_obs = np.zeros((r_cams, 3, m_pts), dtype=np.float64)
    for r in range(r_cams):
        scaled_pts = s_raw[:, None] * landmarks
        q_clean = rot_true[r] @ scaled_pts + t_raw[r][:, None]
        q_obs[r] = q_clean + 0.015 * rng.standard_normal((3, m_pts))
        for m in range(m_pts):
            u_loc = _random_orthonormal(rng, 3, 3, special_orthogonal=True)
            w_mat = (u_loc * eig_profile[None, :]) @ u_loc.T
            w_prec[r, m] = 0.5 * (w_mat + w_mat.T)

    lower = np.concatenate(
        [np.full(3, 0.5, dtype=np.float64), np.full(3 * r_cams, -1.5, dtype=np.float64)]
    )
    upper = np.concatenate(
        [np.full(3, 2.0, dtype=np.float64), np.full(3 * r_cams, 1.5, dtype=np.float64)]
    )

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        s_vec = z[:3]
        sp = s_vec[:, None] * landmarks  # (3, M)
        f_val = 0.0
        gs = np.zeros(3, dtype=np.float64)
        gz = np.zeros(d, dtype=np.float64)
        gx_list: list[np.ndarray] = []

        for r in range(r_cams):
            X_r = x_list[r]
            t_r = z[3 + 3 * r : 3 + 3 * (r + 1)]
            res_r = X_r @ sp + t_r[:, None] - q_obs[r]  # (3, M)
            # Weighted residual e_{r,m} = W_{r,m} res_{r,m}
            e_r = np.einsum("mab,bm->am", w_prec[r], res_r)
            f_val += 0.5 * float(np.sum(res_r * e_r))

            gx_r = np.asfortranarray(e_r @ sp.T, dtype=np.float64)
            gx_list.append(gx_r)
            gz[3 + 3 * r : 3 + 3 * (r + 1)] = np.sum(e_r, axis=1)
            gs += np.sum((X_r.T @ e_r) * landmarks, axis=1)

        gz[:3] = gs
        return float(f_val), gz, gx_list

    return BenchmarkProblem(
        problem_id="B3",
        name="Multi-Camera Anisotropic Bounded-Scale Rig Wahba",
        regime_summary="z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(3, 3, 3, 3),
        p_list=(3, 3, 3, 3),
        eval_fg=eval_fg,
        metadata={"r_cams": r_cams, "m_pts": m_pts, "kappa_w": 1000.0},
    )


def _build_b4_bounded_ml_cpc() -> BenchmarkProblem:
    """B4: Bounded-Variance Maximum Likelihood Common Principal Components (ML-CPC).

    Domain: z in [0.05, 20.0]^150 (K=5, p=30) x X in St(30, 30) (Branch B O(30)).
    Tests log-variance domain barrier and 1.6e5 cross-block Hessian scale ratio
    between lower-clamped (0.05) and upper-clamped (20.0) variances.
    """
    rng = np.random.Generator(np.random.PCG64(202605))
    k_classes = 5
    n = 30
    p = 30
    d = k_classes * p

    u_true = _random_orthonormal(rng, n, n, special_orthogonal=True)
    weights = np.array([0.24, 0.21, 0.20, 0.19, 0.16], dtype=np.float64)

    cov_list = np.zeros((k_classes, n, n), dtype=np.float64)
    for k in range(k_classes):
        # 5 near-null modes (< 0.05 -> clamp at 0.05), 4 spiked modes (> 20 -> clamp at 20),
        # and 21 well-separated intermediate variances
        low_modes = rng.uniform(0.004, 0.022, size=5)
        mid_modes = np.geomspace(0.18, 11.5, num=21) * rng.uniform(0.92, 1.08, size=21)
        high_modes = rng.uniform(27.0, 42.0, size=4)
        lam_k = np.concatenate([high_modes, mid_modes, low_modes])
        # Slight class-specific cyclic shift of extreme modes to couple basis & variances
        lam_k = np.roll(lam_k, shift=k)
        noise_g = rng.standard_normal((n, n))
        s_k = (u_true * lam_k[None, :]) @ u_true.T + 0.01 * (noise_g @ noise_g.T) / n
        cov_list[k] = 0.5 * (s_k + s_k.T)

    lower = np.full(d, 0.05, dtype=np.float64)
    upper = np.full(d, 20.0, dtype=np.float64)

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = x_list[0]
        if np.any(z <= 0.0) or not (np.all(np.isfinite(z)) and np.all(np.isfinite(X))):
            return (
                float("nan"),
                np.full(d, np.nan, dtype=np.float64),
                [np.full((n, p), np.nan, dtype=np.float64, order="F")],
            )

        d_mat = z.reshape((k_classes, p), order="C")
        inv_d = 1.0 / d_mat
        log_d = np.log(d_mat)

        # sx[k] = S_k @ X for k = 0..K-1 -> shape (K, n, p)
        sx = np.einsum("kab,bp->kap", cov_list, X)
        # q[k, j] = x_j^T S_k x_j -> shape (K, p)
        q_mat = np.sum(X[None, :, :] * sx, axis=1)

        f_val = 0.5 * float(np.sum(weights[:, None] * (log_d + q_mat * inv_d)))
        gz_mat = 0.5 * weights[:, None] * (inv_d - q_mat * (inv_d * inv_d))
        gx = np.einsum("k,kap,kp->ap", weights, sx, inv_d)
        return float(f_val), gz_mat.reshape(d, order="C"), [np.asfortranarray(gx, dtype=np.float64)]

    return BenchmarkProblem(
        problem_id="B4",
        name="Bounded-Variance ML Common Principal Components",
        regime_summary="z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        metadata={"k_classes": k_classes, "weights": weights.tolist()},
    )


def _build_b5_factor_correlation_calibration() -> BenchmarkProblem:
    """B5: Bounded Factor Correlation Matrix Calibration.

    Domain: z = (lambda, psi) in [0.2, 25]^10 x [0.02, 1.0]^100 (d=110) x St(100, 10).
    Tests quartic curved banana-like valleys X diag(lambda) X^T and clamped
    idiosyncratic variances psi_i* = 0.02 on high-communality assets.
    """
    rng = np.random.Generator(np.random.PCG64(202606))
    n, p = 100, 10
    d = p + n
    beta_diag = 5.0

    x_true = _random_orthonormal(rng, n, p)
    # Boost loadings on first 22 rows so communalities exceed 0.98, clamping psi_i* = 0.02
    x_boost = x_true.copy()
    x_boost[:22, :4] *= 2.4
    q_true, r_true = np.linalg.qr(x_boost, mode="reduced")
    q_true *= np.where(np.diag(r_true) >= 0.0, 1.0, -1.0)[None, :]

    lam_true = np.array([16.5, 11.8, 9.4, 7.1, 5.3, 4.0, 2.9, 1.8, 0.95, 0.11], dtype=np.float64)
    comm = (q_true * q_true) @ lam_true
    psi_true = np.maximum(-0.08, 1.0 - comm)  # Negative entries force active lower bound 0.02

    noise_sym = rng.standard_normal((n, n))
    noise_sym = 0.05 * (noise_sym + noise_sym.T)
    c_emp = (q_true * lam_true[None, :]) @ q_true.T + np.diag(psi_true) + noise_sym
    c_emp = 0.5 * (c_emp + c_emp.T)
    np.fill_diagonal(c_emp, 1.0)

    lower = np.concatenate(
        [np.full(p, 0.2, dtype=np.float64), np.full(n, 0.02, dtype=np.float64)]
    )
    upper = np.concatenate(
        [np.full(p, 25.0, dtype=np.float64), np.full(n, 1.0, dtype=np.float64)]
    )

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = x_list[0]
        lam = z[:p]
        psi = z[p:]

        x_lam = X * lam[None, :]
        c_hat = x_lam @ X.T
        c_hat[np.diag_indices(n)] += psi
        err = c_hat - c_emp

        x_sq = X * X
        v_diag = (x_sq @ lam) + psi - 1.0

        f_val = 0.5 * float(np.sum(err * err)) + 0.5 * beta_diag * float(v_diag @ v_diag)

        ex = err @ X
        g_lam = np.sum(X * ex, axis=0) + beta_diag * (x_sq.T @ v_diag)
        g_psi = np.diag(err) + beta_diag * v_diag
        gz = np.concatenate([g_lam, g_psi])
        gx = 2.0 * (ex + beta_diag * (v_diag[:, None] * X)) * lam[None, :]
        return float(f_val), gz, [np.asfortranarray(gx, dtype=np.float64)]

    return BenchmarkProblem(
        problem_id="B5",
        name="Bounded Factor Correlation Matrix Calibration",
        regime_summary="z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        metadata={"beta_diag": beta_diag},
    )


def _build_b6_ky_fan_brockett() -> BenchmarkProblem:
    """B6: Box-Coupled Generalized Brockett with Certified Ky Fan QP Global Truth.

    Domain: z in [0.1, 5.0]^10 x St(150, 10) (Branch A, kappa(M) = 10^4).
    By Ky Fan's extremal trace inequality and rearrangement dominance
    (delta_alpha = 0.25 < mu_min * delta_z = 1.0 * (4.9 / 9) = 0.5444...),
    (z_star, U_{A, :, :10}) is the unique global minimizer over the entire
    coupled product space [0.1, 5.0]^10 x St(150, 10), yielding an exact
    analytical machine-precision certificate f_star!
    """
    rng = np.random.Generator(np.random.PCG64(202607))
    n, p = 150, 10
    d = p

    lower = np.full(p, 0.1, dtype=np.float64)
    upper = np.full(p, 5.0, dtype=np.float64)

    # Strictly increasing spectrum alpha_1 < ... < alpha_n:
    # First p eigenvalues spaced by delta_alpha = 0.004 < mu_min * delta_z = 0.005444...,
    # guaranteeing unique global optimality across all p! column permutations while
    # keeping mu_max = 100.0 (kappa(M) = 10^4) well above the IEEE-754 float64 ULP floor.
    alphas_low = -0.80 + 0.004 * np.arange(p, dtype=np.float64)
    alphas_high = alphas_low[-1] + 0.12 * np.arange(1, n - p + 1, dtype=np.float64)
    alphas = np.concatenate([alphas_low, alphas_high])
    u_a = _random_orthonormal(rng, n, n, special_orthogonal=True)
    a_mat = (u_a * alphas[None, :]) @ u_a.T
    a_mat = 0.5 * (a_mat + a_mat.T)

    # Positive-definite coupling matrix M in S_{++}^p with exact kappa(M) = 10^4
    u_m = _random_orthonormal(rng, p, p, special_orthogonal=True)
    m_eigs = np.geomspace(0.01, 100.0, num=p).astype(np.float64)
    m_mat = (u_m * m_eigs[None, :]) @ u_m.T
    m_mat = 0.5 * (m_mat + m_mat.T)
    m_inv = (u_m * (1.0 / m_eigs)[None, :]) @ u_m.T

    # Certified strictly decreasing optimal z_star with active upper bound z_1*=5.0
    # and active lower bound z_10*=0.1
    z_star = np.linspace(5.0, 0.1, num=p, dtype=np.float64)
    lam_lower = np.zeros(p, dtype=np.float64)
    lam_upper = np.zeros(p, dtype=np.float64)
    lam_upper[0] = 0.25
    lam_lower[-1] = 0.25

    # Exact KKT stationarity alpha_{1:p} + M (z_star - z0_ref) = lam_lower - lam_upper
    z0_ref = z_star + m_inv @ (alphas[:p] - lam_lower + lam_upper)
    dz_star = z_star - z0_ref
    f_star = float(np.dot(alphas[:p], z_star) + 0.5 * float(dz_star @ (m_mat @ dz_star)))

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = x_list[0]
        ax = a_mat @ X
        rayleigh = np.sum(X * ax, axis=0)
        dz = z - z0_ref
        m_dz = m_mat @ dz

        f_val = float(np.dot(rayleigh, z)) + 0.5 * float(np.dot(dz, m_dz))
        gz = rayleigh + m_dz
        gx = np.asfortranarray(2.0 * ax * z[None, :], dtype=np.float64)
        return float(f_val), gz, [gx]

    return BenchmarkProblem(
        problem_id="B6",
        name="Box-Coupled Brockett with Exact Ky Fan QP Truth",
        regime_summary="z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(n,),
        p_list=(p,),
        eval_fg=eval_fg,
        f_star=f_star,
        z_star=z_star,
        metadata={"kappa_m": 1.0e4, "alpha_min": float(alphas[0]), "alpha_max": float(alphas[-1])},
    )


def _build_b7a_pure_box_rosenbrock_quad() -> BenchmarkProblem:
    """B7A: Pure-Box Limit: Active-Bound Chained Rosenbrock + Ill-Conditioned Quadratic.

    Domain: z in [l, u]^500, R = 0 (including 5 fixed coordinates l_i == u_i).
    Tests pure-box parity vs SciPy Fortran L-BFGS-B under ~60% active bounds,
    curved Rosenbrock valleys, and fixed variables.
    """
    rng = np.random.Generator(np.random.PCG64(202608))
    d = 500
    d_ros = 200
    d_quad = 300

    lower = np.empty(d, dtype=np.float64)
    upper = np.empty(d, dtype=np.float64)

    # First 200 coords: Chained Rosenbrock; odd indices capped at 0.65 < 1.0 so ~50% bind
    lower[:d_ros] = -1.5
    upper[:d_ros] = 1.5
    upper[1:d_ros:2] = 0.65

    # Next 300 coords: [0.0, 1.0]^300 with 5 fixed coordinates l_i == u_i == 0.25
    lower[d_ros:] = 0.0
    upper[d_ros:] = 1.0
    fixed_indices = (250, 300, 350, 400, 450)
    for idx in fixed_indices:
        lower[idx] = 0.25
        upper[idx] = 0.25

    # Ill-conditioned SPD matrix Q_quad in S_{++}^300 with exact kappa = 10^3
    u_q = _random_orthonormal(rng, d_quad, d_quad, special_orthogonal=True)
    q_eigs = np.geomspace(0.05, 50.0, num=d_quad).astype(np.float64)
    q_quad = (u_q * q_eigs[None, :]) @ u_q.T
    q_quad = 0.5 * (q_quad + q_quad.T)

    # Unconstrained target placed outside [0, 1] on ~70% of coordinates so bounds actively bind
    z_quad_target = rng.uniform(-0.8, 1.8, size=d_quad)
    ros_weight = 2.0
    link_weight = 0.10

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        del x_list
        zr = z[:d_ros]
        zq = z[d_ros:]

        # Chained Rosenbrock: sum_{i=0}^{198} [ros_weight*(z_{i+1} - z_i^2)^2 + (1 - z_i)^2]
        r1 = zr[1:] - zr[:-1] * zr[:-1]
        r2 = 1.0 - zr[:-1]
        f_ros = ros_weight * float(np.dot(r1, r1)) + float(np.dot(r2, r2))

        gz_ros = np.zeros(d_ros, dtype=np.float64)
        gz_ros[:-1] += -4.0 * ros_weight * zr[:-1] * r1 - 2.0 * r2
        gz_ros[1:] += 2.0 * ros_weight * r1

        # Coupled active-boundary quadratic
        dq = zq - z_quad_target
        q_dq = q_quad @ dq
        f_quad = 0.5 * float(np.dot(dq, q_dq))
        gz_quad = q_dq

        # Cross-block spring coupling between Rosenbrock even indices and quadratic block
        diff_link = zr[0:200:2] - zq[0:200:2]
        f_link = 0.5 * link_weight * float(np.dot(diff_link, diff_link))
        gz_ros[0:200:2] += link_weight * diff_link
        gz_quad[0:200:2] -= link_weight * diff_link

        gz = np.concatenate([gz_ros, gz_quad])
        return float(f_ros + f_quad + f_link), gz, []

    return BenchmarkProblem(
        problem_id="B7A",
        name="Pure-Box Active-Bound Chained Rosenbrock + Quadratic",
        regime_summary="z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)",
        d=d,
        lower=lower,
        upper=upper,
        n_list=(),
        p_list=(),
        eval_fg=eval_fg,
        metadata={"fixed_indices": list(fixed_indices), "kappa_quad": 1.0e3},
    )


def _build_b7b_pure_stiefel_brockett_procrustes() -> BenchmarkProblem:
    """B7B: Pure-Stiefel Limit: Tall Brockett & Square Orthogonal Procrustes.

    Domain: d = 0, R = 2: X_1 in St(200, 8) (Branch A, 2p < n) x X_2 in St(24, 24)
    (Branch B, 2p >= n on SO(24)).
    Tests pure-Stiefel dual-regime Cayley retraction & isometric transport
    head-to-head against native pymanopt Stiefel optimizers, with exact analytical
    global optimum f_star (Ky Fan + SO(24) Kabsch SVD).
    """
    rng = np.random.Generator(np.random.PCG64(202609))
    n1, p1 = 200, 8
    n2, p2 = 24, 24

    # Factor 1: Tall Brockett 0.5 * tr(X_1^T A_1 X_1 diag(mu))
    eigs_a1 = np.linspace(-4.0, 18.0, num=n1, dtype=np.float64)
    u_a1 = _random_orthonormal(rng, n1, n1, special_orthogonal=True)
    a1 = (u_a1 * eigs_a1[None, :]) @ u_a1.T
    a1 = 0.5 * (a1 + a1.T)
    mu1 = np.arange(p1, 0, -1, dtype=np.float64)  # [8, 7, ..., 1]
    f1_star = 0.5 * float(np.dot(eigs_a1[:p1], mu1))

    # Factor 2: Square Orthogonal Procrustes 0.5 * ||X_2 B_2 - C_2||_F^2 on SO(24)
    u_b2 = _random_orthonormal(rng, n2, n2, special_orthogonal=True)
    v_b2 = _random_orthonormal(rng, n2, n2, special_orthogonal=True)
    sv_b2 = np.geomspace(0.25, 15.0, num=n2).astype(np.float64)
    b2 = (u_b2 * sv_b2[None, :]) @ v_b2.T
    rot2_true = _random_orthonormal(rng, n2, n2, special_orthogonal=True)
    c2 = rot2_true @ b2 + 0.05 * rng.standard_normal((n2, n2))

    # Exact SO(24) Kabsch global minimum for f_2(X_2)
    u_svd, s_svd, vt_svd = np.linalg.svd(c2 @ b2.T, full_matrices=True)
    det_sign = 1.0 if np.linalg.det(u_svd @ vt_svd) >= 0.0 else -1.0
    max_trace = float(np.sum(s_svd[:-1]) + det_sign * s_svd[-1])
    f2_star = 0.5 * float(np.sum(b2 * b2) + np.sum(c2 * c2)) - max_trace
    f_star = f1_star + f2_star

    empty = np.zeros(0, dtype=np.float64)

    def eval_fg(
        z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        del z
        X1, X2 = x_list[0], x_list[1]

        a1_x1 = a1 @ X1
        f1 = 0.5 * float(np.sum((X1 * a1_x1) * mu1[None, :]))
        gx1 = np.asfortranarray(a1_x1 * mu1[None, :], dtype=np.float64)

        res2 = X2 @ b2 - c2
        f2 = 0.5 * float(np.sum(res2 * res2))
        gx2 = np.asfortranarray(res2 @ b2.T, dtype=np.float64)

        return float(f1 + f2), empty, [gx1, gx2]

    return BenchmarkProblem(
        problem_id="B7B",
        name="Pure-Stiefel Tall Brockett & Orthogonal Procrustes",
        regime_summary="d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)",
        d=0,
        lower=empty,
        upper=empty,
        n_list=(n1, n2),
        p_list=(p1, p2),
        eval_fg=eval_fg,
        f_star=f_star,
        metadata={"f1_star": f1_star, "f2_star": f2_star},
    )


_PROBLEM_BUILDERS: dict[str, Callable[[], BenchmarkProblem]] = {
    "B1": _build_b1_ensemble_dft,
    "B2A": _build_b2a_amplitude_limited_ica,
    "B2B": _build_b2b_nonnegative_sparse_pca,
    "B3": _build_b3_multicamera_wahba,
    "B4": _build_b4_bounded_ml_cpc,
    "B5": _build_b5_factor_correlation_calibration,
    "B6": _build_b6_ky_fan_brockett,
    "B7A": _build_b7a_pure_box_rosenbrock_quad,
    "B7B": _build_b7b_pure_stiefel_brockett_procrustes,
}

_PROBLEM_CACHE: dict[str, BenchmarkProblem] = {}


def get_benchmark_problem(problem_id: str) -> BenchmarkProblem:
    """Return a cached deterministic canonical BenchmarkProblem by ID."""
    key = problem_id.strip().upper()
    if key not in _PROBLEM_BUILDERS:
        raise KeyError(
            f"Unknown benchmark problem_id={problem_id!r}. Expected one of {BENCHMARK_IDS}."
        )
    if key not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[key] = _PROBLEM_BUILDERS[key]()
    return _PROBLEM_CACHE[key]


def get_all_benchmark_problems() -> list[BenchmarkProblem]:
    """Return all 9 canonical benchmark instances in standard suite order."""
    return [get_benchmark_problem(pid) for pid in BENCHMARK_IDS]
