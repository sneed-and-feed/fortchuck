# Stiefel and Riemannian Quasi-Newton Optimization (2012--2026): ArXiv Literature Synthesis and Architectural Design for `fortchuck`

> [!NOTE]
> **Literature Search Provenance and Licensing Notice.** This technical survey was compiled using the `literature-search-arxiv` workflow in full compliance with the [arXiv API Terms of Use](https://info.arxiv.org/help/api/index.html) (rate-limited below one query per three seconds). Metadata and primary LaTeX source manuscripts retrieved from arXiv (`https://arxiv.org`) carry individual author-selected licenses (`arXiv.org perpetual, non-exclusive license` or Creative Commons `CC BY` / `CC BY-NC-SA`); license records were verified per retrieved manuscript prior to mathematical synthesis, no third-party copyrighted prose or code was copied, and explicit URLs and DOIs for all referenced works are cataloged in [Section 8](#8-complete-bibliography-and-arxiv-url-registry).

## Table of Contents

- [1. Executive Summary and Concrete Design Matrix for fortchuck](#1-executive-summary-and-concrete-design-matrix-for-fortchuck)
- [2. Problem Formulation and Stiefel Manifold Geometry](#2-problem-formulation-and-stiefel-manifold-geometry)
- [3. Feasible Cayley Curvilinear Search and Multi-Regime Linear Algebra](#3-feasible-cayley-curvilinear-search-and-multi-regime-linear-algebra)
- [4. Riemannian L-BFGS and Vector Transport Comparison on St(n, p)](#4-riemannian-l-bfgs-and-vector-transport-comparison-on-stn-p)
- [5. Spectral Initialization, Curvature Safeguards, and Line Search](#5-spectral-initialization-curvature-safeguards-and-line-search)
- [6. Numerical Stability: Antipodal Conditioning and Drift Reorthogonalization](#6-numerical-stability-antipodal-conditioning-and-drift-reorthogonalization)
- [7. Assessment of 2020--2026 Inversion-Free and Landing Alternatives](#7-assessment-of-2020--2026-inversion-free-and-landing-alternatives)
- [8. Complete Bibliography and ArXiv URL Registry](#8-complete-bibliography-and-arxiv-url-registry)

---

## 1. Executive Summary and Concrete Design Matrix for `fortchuck`

`fortchuck` targets standalone, zero-dependency, thread-safe Modern Fortran (`F2018` + `fpm` + `iso_c_binding`) quasi-Newton optimization on coupled Euclidean box and multi-factor Stiefel product manifolds:

```math
\min_{z,\, X_1, \dots, X_R} \; f(z, X_1, \dots, X_R) \quad \text{subject to} \quad l \le z \le u, \quad X_r^\top X_r = I_{p_r} \;\; (r = 1, \dots, R),
```

where $z \in \mathbb R^d$ with componentwise bounds $l, u \in (\mathbb R \cup \lbrace -\infty, +\infty \rbrace)^d$, and each matrix factor $X_r \in \mathrm{St}(n_r, p_r) \subset \mathbb R^{n_r \times p_r}$ satisfies
$1 \le p_r \le n_r$.

A synthesis of foundational Riemannian optimization theory (2010--2018) and recent arXiv advances (2019--2026) yields eight concrete architectural decisions for `fortchuck`:

| Subsystem | Recommended `fortchuck` Choice | Mathematical and Computational Rationale | Primary References |
| :--- | :--- | :--- | :--- |
| **Stiefel Riemannian Metric** | Canonical metric $(\beta = 1/2)$ as default; optional Euclidean $(\beta = 1)$ switch | Weights each of the $p(p-1)/2$ independent vertical $\mathfrak{so}(p)$ coordinates equally with horizontal coordinates and box variables $z_i$; uniquely makes Cayley retraction second-order. | Edelman et al. (1998); Stoye et al. (2025, `2511.01563`); Jensen & Zimmermann (2026, `2608.28073`) |
| **Feasible Retraction** | Generalized Wen--Yin Cayley curve $R_X^{\mathrm{Cay}}(\tau \xi), \; \xi \in T_X \mathrm{St}(n, p)$ | Strictly feasible to machine precision, smooth $C^\infty$, inversion-only (no SVD, QR, or matrix exponential inside backtracking), second-order under $\beta = 1/2$. | Wen & Yin (2013); Jiang & Dai (2015, `1301.0172`); Zhu (2017) |
| **Linear Algebra Dispatch** | Two-regime threshold at $2 p \le n$ vs $2 p > n$, using $p \times p$ Schur-Cayley for $p \le n/2$ and direct $n \times n$ LU for $p > n/2$ | Both low-rank LU dimension $(2p \text{ vs } n)$ and batched $2m$-vector L-BFGS transport FLOPs $(8 m n p^2 \text{ vs } 4 m n^2 p)$ cross over at exact $p = n / 2$; Schur reduction cuts backtracking LU by $8\times$ and removes SMW block imbalance. | Wen & Yin (2013); Jiang & Dai (2015, `1301.0172`); Jensen & Zimmermann (2026, `2608.28073`) |
| **L-BFGS Vector Transport** | Lock-free Isometric Cayley Transport $\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\zeta)$ with eager batched ring-buffer updates | Exact linear isometry under all $\beta$-metrics, self-inverse-adjoint, reuses accepted step LU factors with zero extra factorizations, avoids rank loss and canonical norm inflation of projection transport. | Zhu (2017); Huang, Absil & Gallivan (2018); Wang et al. (2024, `2311.00907`) |
| **Curvature Positivity Guard** | Cautious update check (default) with optional Powell damping toward $H_k^{(0)-1} s_k = \gamma_k^{-1} s_k$ | Guarantees $\langle s, y \rangle > 0$ and global + local superlinear convergence on curved manifolds without the Huang--Gallivan--Absil locking condition or differentiated retraction. | Huang, Absil & Gallivan (2018); Nocedal & Wright (2006) |
| **Initial Scaling** | Scalar BB2 scaling $\gamma_k = \langle s_k, y_k \rangle / \langle y_k, y_k \rangle$ for L-BFGS (`tau_0 = 1`); alternating ABB1/ABB2 on steepest-descent steps | Matches local Rayleigh quotient along the most recent secant pair; unit step `tau_0 = 1` preserves asymptotic superlinear rate while ABB accelerates $k = 0$ and restarts. | Wen & Yin (2013); Jiang & Dai (2015, `1301.0172`); Hu et al. (2020, `1906.05450`) |
| **Line Search** | Generalized Zhang--Hager / Armijo backtracking ($\eta \in [0, 0.85]$, default $\eta = 0$ in L-BFGS mode) | Zero extra gradient evaluations on rejected trials; avoids nonsmooth directional-derivative kinks on box bounds $[l, u]$ and avoids nonconvex Wolfe pathologies. | Zhang & Hager (2004); Ding, Li & Zhang (2024, `2404.19392`); Diao (2026, `2609.09686`) |
| **Singularity & Drift Guards** | Step-velocity cap + LU reciprocal-pivot guard; drift threshold $\lVert X_r^\top X_r - I_{p_r} \rVert_{\mathrm F} > 10^{-13}$ | Prevents floating-point blow-up near antipodal locus $\det(I + X^\top Y) = 0$; 1-step Newton--Schulz normal polish (when drift $\le 10^{-4}$) or positive-diagonal Householder QR restores `eps_mach` orthogonality without sign flips. | Jiang & Dai (2015, `1301.0172`); Kume & Yamada (2023, `2305.17901`); Xiong, Gao & Absil (2026, `2605.02838`) |

---

## 2. Problem Formulation and Stiefel Manifold Geometry

### 2.1 Tangent Space and Horizontal-Vertical Decomposition

For integers $1 \le p \le n$, the compact Stiefel manifold of orthonormal $p$-frames in $\mathbb R^n$ is an embedded real-analytic submanifold of $\mathbb R^{n \times p}$:

```math
\mathrm{St}(n, p) = \left\lbrace X \in \mathbb R^{n \times p} : X^\top X = I_p \right\rbrace, \qquad \dim \mathrm{St}(n, p) = n p - \frac{p(p + 1)}{2}.
```

Boundary cases recover the unit sphere $S^{n-1}$ when $p = 1$ and the orthogonal group $\mathrm{O}(n)$ (whose connected components include $\mathrm{SO}(n)$) when $p = n$. Differentiating $X(t)^\top X(t) = I_p$ at $X(0) = X$ gives the tangent space:

```math
T_X \mathrm{St}(n, p) = \left\lbrace \xi \in \mathbb R^{n \times p} : X^\top \xi + \xi^\top X = 0 \right\rbrace = \left\lbrace X A + K : A \in \mathfrak{so}(p), \; K \in \mathbb R^{n \times p}, \; X^\top K = 0 \right\rbrace,
```

where $\mathfrak{so}(p) = \lbrace A \in \mathbb R^{p \times p} : A^\top = -A \rbrace$, and the symmetric/skew-symmetric matrix parts are $\mathrm{sym}(M) = \tfrac{1}{2}(M + M^\top)$ and $\mathrm{skew}(M) = \tfrac{1}{2}(M - M^\top)$. Crucially for `fortchuck`, the horizontal component $K = (I_n - X X^\top)\xi = \xi - X(X^\top \xi)$ never materializes an explicit $n \times (n - p)$ orthogonal completion
$X_\perp$, avoiding $O(n^2)$ memory when $p \ll n$.

### 2.2 The $\beta$-Metric Family: Canonical $(\beta = 1/2)$ vs Euclidean $(\beta = 1)$

Hüper, Markina, and Silva Leite (2021), Zimmermann and Hüper (2022, `2203.00389`), and Stoye, Mataigne, Absil, and Zimmermann (2025, `2511.01563`) parameterize the homogeneous Riemannian metrics on $\mathrm{St}(n, p)$ by $\beta > 0$. Writing $\xi = X A_1 + K_1$ and
$\eta = X A_2 + K_2$ with
$A_i = X^\top \xi_i \in \mathfrak{so}(p)$ and
$X^\top K_i = 0$:

```math
g_X^{(\beta)}(\xi, \eta) = \beta\,\mathrm{tr}(A_1^\top A_2) + \mathrm{tr}(K_1^\top K_2) = \mathrm{tr}\!\left(\xi^\top \left(I_n - (1 - \beta) X X^\top\right) \eta\right).
```

Let $G = \nabla_X f(z, X_1, \dots, X_R) \in \mathbb R^{n \times p}$ denote the ambient Euclidean partial gradient with respect to $X$. The Riesz representation
$g_X^{(\beta)}(\mathrm{grad}^{(\beta)} f(X), \eta) = \mathrm{tr}(G^\top \eta)$ for all
$\eta \in T_X \mathrm{St}(n, p)$ yields the Riemannian gradient under arbitrary $\beta > 0$:

```math
\mathrm{grad}^{(\beta)} f(X) = G - X\,\mathrm{sym}(X^\top G) - \frac{1 - \beta}{\beta} X\,\mathrm{skew}(X^\top G) = G - \frac{1}{2\beta} X \left(G^\top X + (2\beta - 1) X^\top G\right).
```

Note that orthogonal tangent-space projection under the ambient Frobenius metric is $P_{T_X}(Z) = Z - X\,\mathrm{sym}(X^\top Z)$, whereas the two flagship metrics specialize as follows:

1. **Euclidean Submanifold Metric** $(\beta = 1)$:

```math
g_X^{\mathrm{Euc}}(\xi, \eta) = \mathrm{tr}(\xi^\top \eta), \qquad \mathrm{grad}^{\mathrm{Euc}} f(X) = G - X\,\mathrm{sym}(X^\top G) = G - \frac{1}{2} X (X^\top G + G^\top X).
```

2. **Canonical Quotient-Induced Metric** $(\beta = 1/2)$ (Edelman--Arias--Smith 1998, `physics/9806030`):

```math
g_X^{\mathrm{can}}(\xi, \eta) = \mathrm{tr}\!\left(\xi^\top \left(I_n - \frac{1}{2} X X^\top\right) \eta\right), \qquad \mathrm{grad}^{\mathrm{can}} f(X) = G - X G^\top X.
```

### 2.3 Curvature, Injectivity Radius, and Product-Manifold Coupling Implications

Recent geometric analyses by Zimmermann and Stoye (2024, `2403.01879`; 2025, `2405.02268`), Absil and Mataigne (2025, `2403.03782`), and Stoye et al. (2025, `2511.01563`) establish exact curvature and injectivity radius properties across $\mathrm{St}^{(\beta)}(n, p)$ for $2 \le p \le n - 1$:

| Property | Canonical Metric $(\beta = 1/2)$ | Euclidean Metric $(\beta = 1)$ | Impact on `fortchuck` |
| :--- | :--- | :--- | :--- |
| **Tangent Norm Squared** | $\tfrac{1}{2}\lVert A \rVert_{\mathrm F}^2 + \lVert K \rVert_{\mathrm F}^2 = \sum_{i < j} A_{ij}^2 + \sum_{a,j} K_{aj}^2$ | $\lVert A \rVert^2 + \lVert K \rVert^2$ (Frobenius, weight $2$ on strict upper triangle of `A`) | $\beta = 1/2$ assigns unit weight $1$ (not $2$) to each independent $\mathfrak{so}(p)$ coordinate, matching Euclidean box `z_i` scaling. |
| **Riemannian Gradient FLOPs** | $4 n p^2$ via $X^\top G$ then $G - X(X^\top G)^\top$ | $4 n p^2 + p^2$ via $X^\top G$, symmetrize, $G - X S$ | Identical leading $4 n p^2$ cost; canonical gradient avoids forming $\mathrm{sym}(X^\top G)$. |
| **Max Sectional Curvature** $K_{\max}$ | $5/4$ (sharp for $p \ge 2, n \ge p + 2$) | $1$ (sharp for $n \ge p + 2$; $2$ when $n = p + 1$) | Comparable positive curvature bounds; governs trust-region / step-length ceiling. |
| **Shortest Geodesic Loop** | $2\pi$ (`2511.01563`, Thm. 3.3) | $2\pi$ (`2405.02268`) | Single trial step lengths $\tau \lVert \xi \rVert_\beta \ge \pi$ wrap halfway around closed loops. |
| **Global Injectivity Radius** $\mathrm{inj}$ | $\mathrm{inj} \in [\sqrt{4/5}\,\pi, \sqrt{2}\,t^{\mathrm r}] \approx [2.8099, 2.8691]$ | $\mathrm{inj} = \pi \approx 3.1416$ (exact) | Motivates per-factor trial step velocity safeguard $\tau \lVert \xi_r \rVert_{\mathrm{can}} \le \Delta_{\max} \approx 2\pi$. |
| **Cayley Retraction Order** | **Second-order** $(D_{\dot{\gamma}} \dot{\gamma}(0) = 0)$ | **First-order only** when $1 < p < n$ | Second-order retraction eliminates extrinsic acceleration bias in quasi-Newton secant equations (`2608.28073`, `2602.19923`). |

> [!IMPORTANT]
> *Design Recommendation 1 (Default Metric on Product Manifold).* Use the **canonical metric** $(\beta = 1/2)$ on every Stiefel factor $\mathrm{St}(n_r, p_r)$,
> paired with the standard Euclidean inner product on the box block $z \in \mathbb R^d$:

```math
\left\langle (\delta z, \xi_1, \dots, \xi_R), (\tilde{\delta z}, \eta_1, \dots, \eta_R) \right\rangle_{\mathcal M} = \delta z^\top \tilde{\delta z} + \sum_{r=1}^R \mathrm{tr}\!\left(\xi_r^\top \left(I_{n_r} - \frac{1}{2} X_r X_r^\top\right) \eta_r\right).
```

When $p_r = 1$ (unit sphere),
$X_r^\top \xi_r = 0$ identically, so the canonical and Euclidean metrics coincide automatically; when
$p_r = n_r$
($\mathrm{O}(n_r)$ /
$\mathrm{SO}(n_r)$),
$I_n - \tfrac{1}{2} X X^\top = \tfrac{1}{2} I_n$, giving each of the $n(n-1)/2$ Lie-algebra coordinates unit variance
$\sum_{i < j} A_{ij}^2$ consistent with $z \in \mathbb R^d$.

---

## 3. Feasible Cayley Curvilinear Search and Multi-Regime Linear Algebra

### 3.1 Generalized Wen--Yin Skew-Symmetric Generator for Arbitrary Quasi-Newton Directions

In Wen and Yin (2013, *Math. Program.* 142, 397--428), the curvilinear steepest-descent arc uses $W_{\mathrm{WY}}(X, G) = G X^\top - X G^\top \in \mathfrak{so}(n)$, for which
$-W_{\mathrm{WY}}(X, G) X = -(G - X G^\top X) = -\mathrm{grad}^{\mathrm{can}} f(X)$. In a quasi-Newton solver (`LRBFGS`), however, the search direction
$\xi \in T_X \mathrm{St}(n, p)$ is produced by the two-loop recursion and is **not** a scalar multiple of $-\mathrm{grad}^{\mathrm{can}} f(X)$.

Following Jiang and Dai (2015, `1301.0172`), Zhu (2017), and Wang et al. (2024, `2311.00907`), define the canonical projector $P_X = I_n - \tfrac{1}{2} X X^\top$. For **any** tangent vector
$\xi \in T_X \mathrm{St}(n, p)$ (satisfying $X^\top \xi + \xi^\top X = 0$), set:

```math
\tilde{\xi} = P_X \xi = \xi - \frac{1}{2} X (X^\top \xi), \qquad W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top \in \mathfrak{so}(n).
```

Direct multiplication verifies exact initial-velocity consistency:

```math
W_\xi X = \tilde{\xi}(X^\top X) - X(\tilde{\xi}^\top X) = \left(\xi - \frac{1}{2} X X^\top \xi\right) - X\left(\xi^\top X - \frac{1}{2} \xi^\top X\right) = \xi - \frac{1}{2} X(X^\top \xi + \xi^\top X) = \xi.
```

The **Cayley retraction** along $\xi \in T_X \mathrm{St}(n, p)$ with step size $\tau \ge 0$ is then defined by the Cayley transform
$Q_\xi(\tau) \in \mathrm{SO}(n)$:

```math
Q_\xi(\tau) = \left(I_n - \frac{\tau}{2} W_\xi\right)^{-1} \left(I_n + \frac{\tau}{2} W_\xi\right), \qquad R_X^{\mathrm{Cay}}(\tau \xi) = Q_\xi(\tau) X.
```

Because $W_\xi^\top = -W_\xi$, all eigenvalues of
$W_\xi$ are purely imaginary
$\lbrace \pm i \theta_j, 0 \rbrace$, so
$\det(I_n - \tfrac{\tau}{2} W_\xi) = \prod_j (1 + \tfrac{\tau^2}{4}\theta_j^2) \ge 1$ for all $\tau \in \mathbb R$ and
$Q_\xi(\tau)^\top Q_\xi(\tau) = I_n$. Moreover, the curve
$Y(\tau) = R_X^{\mathrm{Cay}}(\tau \xi)$ satisfies:

```math
Y(0) = X, \qquad \dot{Y}(0) = W_\xi X = \xi, \qquad \ddot{Y}(0) = \frac{1}{2} W_\xi^2 X,
```

whose canonical Riemannian acceleration vanishes identically (`2608.28073`, Prop. 2.1).

### 3.2 Three Algebraically Identical Linear-Algebra Realizations

Evaluating $R_X^{\mathrm{Cay}}(\tau \xi) = Q_\xi(\tau) X$ (and, upon step acceptance, transporting $2m$ L-BFGS tangent vectors via
$Q_\xi(\tau_k) \zeta$) admits three exact algebraic realizations whose computational complexities depend sharply on the aspect ratio $p / n$.

#### Formulation 1A: Classical Wen--Yin Low-Rank ($2p$) Sherman--Morrison--Woodbury ($2p \times 2p$ LU)

Writing $W_\xi = U_\xi V_\xi^\top$ with
$U_\xi = [\tilde{\xi},\, X] \in \mathbb R^{n \times 2p}$ and
$V_\xi = [X,\, -\tilde{\xi}] \in \mathbb R^{n \times 2p}$, the Sherman--Morrison--Woodbury (SMW) identity (`Wen & Yin 2013`, Lemma 4) gives, for any $\zeta \in \mathbb R^{n \times p}$:

```math
Q_\xi(\tau) \zeta = \zeta + \tau U_\xi M_{2p}(\tau)^{-1} \left(V_\xi^\top \zeta\right), \qquad M_{2p}(\tau) = I_{2p} - \frac{\tau}{2} V_\xi^\top U_\xi = \begin{bmatrix} I_p - \frac{\tau}{4} A & -\frac{\tau}{2} I_p \\ \frac{\tau}{2}\left(K^\top K - \frac{1}{4} A^2\right) & I_p - \frac{\tau}{4} A \end{bmatrix},
```

where $A = \mathrm{skew}(X^\top \xi) \in \mathfrak{so}(p)$ and $K = \xi - X A$. Setting $\zeta = X$ yields $V_\xi^\top X = [I_p,\, \tfrac{1}{2} A^\top]^\top = [I_p,\, -\tfrac{1}{2} A]^\top$ and
$R_X^{\mathrm{Cay}}(\tau \xi) = X + \tau U_\xi M_{2p}(\tau)^{-1} V_\xi^\top X$.

- **Critical Conditioning Caveat (`2608.28073`, Jensen & Zimmermann 2026)**: Notice that the $(1, 2)$ block of $M_{2p}(\tau)$ has norm $\tau / 2$, whereas the $(2, 1)$ block has norm
  $O(\tau \lVert \xi \rVert_{\mathrm F}^2)$. As
  $\lVert \xi \rVert_{\mathrm F} \to 0$ near convergence (or when
  $\lVert \xi \rVert_{\mathrm F} \gg 1$),
  $M_{2p}(\tau)$ becomes artificially unbalanced for unscaled $2p \times 2p$ Gaussian elimination, even though the underlying Cayley operator has unit spectral norm! Moreover, factoring a $2p \times 2p$ matrix costs $\tfrac{16}{3} p^3$ FLOPs ($8\times$ more than a $p \times p$ LU).

#### Formulation 1B: Exact $p \times p$ Schur-Complement Reduction for Both Retraction and Isometric Transport

Jiang and Dai (2015, `1301.0172`, Prop. 3.2 & Appendix D) and Jensen and Zimmermann (2026, `2608.28073`, Sec. 3.2) prove that block elimination of $M_{2p}(\tau)$ reduces the **exact same Cayley curve** to a single $p \times p$ strictly dissipative system. Specifically, let:

```math
A = \mathrm{skew}(X^\top \xi) \in \mathfrak{so}(p), \qquad K = \xi - X A \in \mathbb R^{n \times p}, \qquad S = K^\top K \in \mathbb R^{p \times p},
```

and for any trial step $\tau \ge 0$ define the $p \times p$ matrix:

```math
J(\tau) = I_p - \frac{\tau}{2} A + \frac{\tau^2}{4} S \in \mathbb R^{p \times p}.
```

Because $\mathrm{sym}(J(\tau)) = I_p + \tfrac{\tau^2}{4} K^\top K \succeq I_p$, $J(\tau)$ is **unconditionally nonsingular** for all $\tau \in \mathbb R$, satisfies
$\lVert J(\tau)^{-1} \rVert_2 \le 1$, and enjoys the sharp condition-number bound (`1301.0172`, Lemma 3.1(iv)):

```math
\kappa_2(J(\tau)) = \lVert J(\tau) \rVert_2 \lVert J(\tau)^{-1} \rVert_2 \le \frac{5 + \tau^2 \lVert \xi \rVert_{\mathrm F}^2}{4},
```

completely eliminating the small-gradient block imbalance of $M_{2p}(\tau)$, since
$\kappa_2(J(\tau)) \to 1$ smoothly as
$\tau \lVert \xi \rVert_{\mathrm F} \to 0$.

1. **Retraction Evaluation** ($2 n p^2 + \tfrac{2}{3} p^3$ FLOPs per backtracking trial):

```math
R_X^{\mathrm{Cay}}(\tau \xi) = (2 X + \tau K) J(\tau)^{-1} - X.
```

In Fortran column-major arithmetic, rather than inverting $J(\tau)$, factor $J(\tau)^\top = L U$ via $p \times p$ kernel `GETRF` ($\tfrac{2}{3} p^3$ FLOPs) and solve $J(\tau)^\top Z^\top = (2 X + \tau K)^\top$ (or apply right-triangular solves `TRSM` directly on the $n \times p$ panel $2 X + \tau K$ in $2 n p^2$ FLOPs without transposing $n \times p$ arrays!), then set $Y(\tau) = Z - X$.

2. **Division-Free Isometric Cayley Transport Reusing the Factored** $J(\tau_k)$:
Eliminating $Z_2$ from
$M_{2p}(\tau_k) [Z_1^\top, Z_2^\top]^\top = V_\xi^\top \zeta = [R_1^\top, R_2^\top]^\top$ using
$\tfrac{\tau_k}{2} T_{21} = J(\tau_k) - C^2$ where
$C = I_p - \tfrac{\tau_k}{4} A$ cancels
$C^2 Z_1$ identically and yields an **exact, division-free formula** for
$Q_\xi(\tau_k) \zeta$ requiring *only* the already-computed $p \times p$ LU factors of
$J(\tau_k)$:

```math
R_1 = X^\top \zeta, \qquad R_2 = -\tilde{\xi}^\top \zeta, \qquad C = I_p - \frac{\tau_k}{4} A,
```

```math
J(\tau_k) Z_1 = C R_1 + \frac{\tau_k}{2} R_2, \qquad \mathcal T_{\tau_k \xi}^{\mathrm{Cay}}(\zeta) = Q_\xi(\tau_k) \zeta = \zeta + \tau_k \tilde{\xi} Z_1 + 2 X (C Z_1 - R_1).
```

Note that at $\tau_k = 0$,
$C = I_p$,
$J(0) = I_p$,
$Z_1 = R_1$, and
$\mathcal T_0^{\mathrm{Cay}}(\zeta) = \zeta$ with zero division by
$\tau_k$; at $\zeta = X$,
$R_1 = I_p$ and
$R_2 = \tfrac{1}{2} A$, recovering
$(2 X + \tau_k K) J(\tau_k)^{-1} - X$ identically. Stacking all $2m$ L-BFGS tangent matrices
$\lbrace S_i, Y_i \rbrace_{i=1}^m$ into a contiguous $n \times (2mp)$ panel executes the entire history transport in two batched `GEMM` calls and one $p \times p$ solve `GETRS` at $8 m n p^2 + O(m p^3)$ FLOPs!

#### Formulation 2: Direct $n \times n$ Skew-Cayley LU with Resolvent Doubling ($p > n/2$, including $\mathrm{SO}(n)$ and $\mathrm{O}(n)$)

When $p > n/2$ (so $2p > n$, peaking at $p = n$ for $\mathrm{SO}(n)$ and $\mathrm{O}(n)$), low-rank representations introduce redundant $2p$-column auxiliary bases in an $n$-dimensional ambient space. Instead, form $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top \in \mathfrak{so}(n)$ explicitly once per outer iteration ($2 n^2 p$ FLOPs to compute $P = \tilde{\xi} X^\top$, followed by
$W_\xi = P - P^\top$). Next, exploit the **resolvent doubling identity**:

```math
Q_\xi(\tau) = \left(I_n - \frac{\tau}{2} W_\xi\right)^{-1} \left(2 I_n - \left(I_n - \frac{\tau}{2} W_\xi\right)\right) = 2 M_n(\tau)^{-1} - I_n, \qquad M_n(\tau) = I_n - \frac{\tau}{2} W_\xi \in \mathbb R^{n \times n}.
```

This identity eliminates the $2 n^2 p$ matrix product $(I_n + \tfrac{\tau}{2} W_\xi) \zeta$ on every single backtracking trial and every single L-BFGS vector transport:

1. **Retraction Evaluation** ($\tfrac{2}{3} n^3 + 2 n^2 p$ FLOPs per trial $\tau$): Factor $M_n(\tau) = L U$ via $n \times n$ kernel `GETRF` ($\tfrac{2}{3} n^3$ FLOPs), solve
  $M_n(\tau) Z = 2 X$ via `GETRS` ($2 n^2 p$ FLOPs), and set
  $R_X^{\mathrm{Cay}}(\tau \xi) = Z - X$.
2. **Batched Isometric Transport** ($4 m n^2 p$ FLOPs for $2m$ vectors): Reusing the factored $M_n(\tau_k)$ from the accepted step, solve
  $M_n(\tau_k) Z_\zeta = 2 \zeta$ and set
  $\mathcal T_{\tau_k \xi}^{\mathrm{Cay}}(\zeta) = Z_\zeta - \zeta$ (`GETRS` + `AXPY` only, zero `GEMM`!).

### 3.3 Exact FLOP Counts, Workspace Footprint, and the $p = n/2$ Dispatch Theorem

Comparing the leading-order floating-point operations (`1 FLOP = 1 add or 1 multiply`) across all three realizations establishes the exact dispatch boundary:

| Operation Phase | Formulation 1A: Low-Rank ($2p$) SMW | Formulation 1B: Schur-Cayley ($p \times p$ LU) | Formulation 2: Direct $n \times n$ LU |
| :--- | :--- | :--- | :--- |
| **Outer Setup** (once per $k$) | $5 n p^2$ for $\tilde{\xi}, \; V_\xi^\top U_\xi$ | $5 n p^2$ for $A, K, \tilde{\xi}, S = K^\top K$ | $2 n p^2 + 2 n^2 p$ for $\tilde{\xi}$ and `W_xi` |
| **First Trial Step** $\tau_0$ | $4 n p^2 + \tfrac{40}{3} p^3$ | $2 n p^2 + \tfrac{2}{3} p^3$ | $\tfrac{2}{3} n^3 + 2 n^2 p$ |
| **Each Extra Backtrack** $\tau_j$ | $4 n p^2 + \tfrac{40}{3} p^3$ | $2 n p^2 + \tfrac{2}{3} p^3$ | $\tfrac{2}{3} n^3 + 2 n^2 p$ |
| **Transport of** $2m$ **Pairs** | $8 m n p^2 + 16 m p^3$ | $8 m n p^2 + 10 m p^3$ | $4 m n^2 p$ |
| **Cost at** $p = n, m = 5$ | $\tfrac{280}{3} n^3 \approx 93.3 n^3$ | $57.7 n^3$ ($\tfrac{8}{3} n^3$ step + $80 n^3$ transport) | $\tfrac{140}{3} n^3 \approx 46.7 n^3$ ($\tfrac{14}{3} n^3$ step + $40 n^3$ transport) |
| **Scratch `real64` Words** | $4 n p + 8 p^2$ | $2 n p + 4 p^2$ | $2 n^2 + n p$ |

Look at the exact crossover between the low-rank family (Formulation 1B) and Direct $n \times n$ LU (Formulation 2):
- **Transport FLOP equality**: $8 m n p^2 = 4 m n^2 p \iff 2 p = n \iff p = n / 2$. For a typical L-BFGS memory $m \in [5, 10]$, history transport dominates per-iteration linear algebra ($80 n p^2$ vs $40 n^2 p$ at $m = 5$), making Formulation 2 strictly faster whenever $p > n / 2$!
- **Dense LU dimension**: Rank $2p$ SMW (1A) factors a $2p \times 2p$ matrix while Direct LU (2) factors an $n \times n$ matrix, crossing at exact $2p = n$, i.e. $p = n / 2$.
- **Within** $p \le n / 2$: Formulation 1B (`p x p` Schur-Cayley) strictly dominates Formulation 1A (`2p x 2p` SMW) on every metric: $2\times$ fewer $O(n p^2)$ backtracking FLOPs ($2 n p^2$ vs $4 n p^2$), $20\times$ fewer $O(p^3)$ backtracking FLOPs ($\tfrac{2}{3} p^3$ vs $\tfrac{40}{3} p^3$), half the scratch memory, and guaranteed $\kappa_2(J(\tau)) \le (5 + \tau^2 \lVert \xi \rVert_{\mathrm F}^2)/4$.
- **Boundary Specializations**: When $p = 1$ (unit sphere $S^{n-1}$), $A = 0$ and $J(\tau) = 1 + \tfrac{\tau^2}{4} \lVert \xi \rVert_2^2$ is a scalar ($O(n)$ FLOPs, zero matrix factorizations). When $p = n$ (orthogonal groups $\mathrm{O}(n)$ and $\mathrm{SO}(n)$),
  $K = (I_n - X X^\top)\xi = 0$ identically, and Formulation 2 reduces to $\tfrac{14}{3} n^3$ setup+trial FLOPs ($\tfrac{8}{3} n^3$ per extra backtrack) plus $4 m n^3$ transport FLOPs.

> [!TIP]
> **Design Recommendation 2 (Dual-Regime Cayley Engine for `fortchuck`).** Dispatch per Stiefel factor $r \in \lbrace 1, \dots, R \rbrace$ using the static dimension check `2 * p_r <= n_r`:
> - **Branch A** (`2 * p_r <= n_r`): Execute *Formulation 1B* ($p_r \times p_r$ Schur-Cayley) for both curvilinear backtracking and post-step batched isometric L-BFGS transport (with scalar fast-path when
> $p_r = 1$).
> - **Branch B** (`2 * p_r > n_r`): Execute *Formulation 2* (Direct $n_r \times n_r$ LU with resolvent doubling
> $2 M_n(\tau)^{-1} - I_n$) for both curvilinear backtracking and post-step batched isometric L-BFGS transport.
> Both branches require only pure `F2018` blocked `GEMM`, `TRSM`, and panel-pivoted `GETRF` — zero `SVD`, `QR`, or `EXPM` calls inside the optimization loop.

---

## 4. Riemannian L-BFGS and Vector Transport Comparison on $\mathrm{St}(n, p)$

### 4.1 The Three Candidate Vector Transports Associated with $R_X^{\mathrm{Cay}}$

In Riemannian L-BFGS (`LRBFGS`; Ring & Wirth 2012; Huang, Gallivan & Absil 2015; Huang, Absil & Gallivan 2018; Zhu 2017; Sato 2022, `2112.02572`; Wang et al. 2024, `2311.00907`), moving from $X_k$ to
$X_{k+1} = R_{X_k}^{\mathrm{Cay}}(\tau_k \xi_k)$ requires a linear vector transport
$\mathcal T_{\tau_k \xi_k} : T_{X_k} \mathrm{St}(n, p) \to T_{X_{k+1}} \mathrm{St}(n, p)$ to form the secant pair
$(s_k, y_k) \in (T_{X_{k+1}} \mathrm{St}(n, p))^2$ and to carry stored curvature pairs forward. Three candidate transports exist on $\mathrm{St}(n, p)$:

#### Candidate 1: Isometric Cayley Vector Transport ($\mathcal T^{\mathrm{Cay}}$; Zhu 2017; Wen & Yin 2013)

```math
\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\zeta) = Q_\xi(\tau) \zeta = \left(I_n - \frac{\tau}{2} W_\xi\right)^{-1} \left(I_n + \frac{\tau}{2} W_\xi\right) \zeta, \qquad \zeta \in T_X \mathrm{St}(n, p).
```

- **Exact Tangent Membership**: Since $Q_\xi(\tau) \in \mathrm{SO}(n)$ and
  $Y = Q_\xi(\tau) X$,
  $Y^\top (Q_\xi(\tau)\zeta) = X^\top Q_\xi(\tau)^\top Q_\xi(\tau) \zeta = X^\top \zeta \in \mathfrak{so}(p)$, so
  $\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\zeta) \in T_Y \mathrm{St}(n, p)$.
- **Simultaneous Isometry Across Every Metric Parameter**: For any $\zeta_1, \zeta_2 \in T_X \mathrm{St}(n, p)$ and *every* $\beta > 0$:

```math
g_Y^{(\beta)}\!\left(Q_\xi(\tau)\zeta_1, Q_\xi(\tau)\zeta_2\right) = \mathrm{tr}\!\left(\zeta_1^\top Q_\xi(\tau)^\top \left(I_n - (1 - \beta) Q_\xi(\tau) X X^\top Q_\xi(\tau)^\top\right) Q_\xi(\tau) \zeta_2\right) = g_X^{(\beta)}(\zeta_1, \zeta_2).
```

- **Self-Inverse-Adjointness**: Because $\mathcal T_{\tau \xi}^{\mathrm{Cay}}$ is an orthogonal linear operator between
  $(T_X, g_X^{(\beta)})$ and
  $(T_Y, g_Y^{(\beta)})$, its inverse adjoint satisfies
  $(\mathcal T_{\tau \xi}^{\mathrm{Cay}})^{-\sharp} = \mathcal T_{\tau \xi}^{\mathrm{Cay}}$. Consequently, inverse-Hessian transport reduces directly to vector transport while *strictly preserving symmetry and positive definiteness*:

```math
H_{k+1} = \mathcal T_k \widehat H_k \mathcal T_k^\sharp \quad \Longleftrightarrow \quad (S_i, Y_i) \longmapsto (\mathcal T_k S_i, \mathcal T_k Y_i).
```

#### Candidate 2: Differentiated Cayley Retraction ($\mathcal T^R$; Zhu 2017; Wang et al. 2024, `2311.00907`)

Differentiating $R_X^{\mathrm{Cay}}(\eta) = (I_n - \tfrac{1}{2} W_\eta)^{-1}(I_n + \tfrac{1}{2} W_\eta) X$ at $\eta = \tau \xi$ in direction
$\zeta \in T_X \mathrm{St}(n, p)$ yields (`Zhu 2017`, Eq. 21; `2311.00907`, Prop. 2):

```math
\mathcal T_{\tau \xi}^R(\zeta) = \mathrm D R_X^{\mathrm{Cay}}(\tau \xi)[\zeta] = \left(I_n - \frac{\tau}{2} W_\xi\right)^{-1} W_\zeta \left(I_n - \frac{\tau}{2} W_\xi\right)^{-1} X.
```

Along the step direction itself ($\zeta = \xi$), since $W_\xi$ commutes with
$(I_n - \tfrac{\tau}{2} W_\xi)^{-1}$ and
$W_\xi X = \xi$, this simplifies to the curve velocity:

```math
\mathcal T_{\tau \xi}^R(\xi) = \frac{\mathrm d}{\mathrm d\tau} R_X^{\mathrm{Cay}}(\tau \xi) = \left(I_n - \frac{\tau}{2} W_\xi\right)^{-2} \xi.
```

- **Severe Drawbacks for `LRBFGS`**:
  1. **Non-isometric and strictly contractive**: $\lVert \mathcal T_{\tau \xi}^R(\xi) \rVert_\beta \le \lVert \xi \rVert_\beta$ with strict inequality whenever
   $\tau W_\xi \ne 0$. Because
   $(\mathcal T_{\tau \xi}^R)^{-\sharp} \ne \mathcal T_{\tau \xi}^R$, standard two-loop L-BFGS requires inverting the adjoint
   $(\mathcal T_{\tau \xi}^R)^{-\sharp}$ at every iteration (or forming a dense isometric Gage--Hamilton--Householder locking map).
  2. **Loss of full rank at large steps**: As noted by Zhu (2017) and Wang et al. (2024, `2311.00907`, Sec. 3.1), while $\mathcal T_{\tau \xi}^R(\xi) \ne 0$ for $\xi \ne 0$, the linear map
   $\zeta \mapsto \mathcal T_{\tau \xi}^R(\zeta)$ on
   $T_X \mathrm{St}(n, p)$ is *not guaranteed to remain nonsingular* for arbitrary $\tau > 0$ when $1 < p < n$, risking singular inverse-adjoint solves during aggressive global steps.
  3. **Double solve cost**: Evaluating $\mathcal T_{\tau \xi}^R(\zeta)$ for general $\zeta \ne \xi$ requires constructing
   $W_\zeta$ for each of the $2m$ history vectors and performing two sequential resolvent applications.

#### Candidate 3: Orthogonal Projection-Based Transport ($\mathcal T^{\mathrm{proj}}$; Absil, Mahony & Sepulchre 2008)

```math
\mathcal T_{\tau \xi}^{\mathrm{proj}}(\zeta) = P_{T_Y}(\zeta) = \zeta - Y\,\mathrm{sym}(Y^\top \zeta), \qquad Y = R_X^{\mathrm{Cay}}(\tau \xi).
```

- **Severe Drawbacks Under Canonical Metric and Large Steps**:
  1. **Violates Ring--Wirth non-expansiveness under the canonical metric!** Under the Euclidean metric $(\beta = 1)$, $P_{T_Y}$ is an ambient orthogonal projector so
   $\lVert P_{T_Y}(\zeta) \rVert_{\mathrm{Euc}} \le \lVert \zeta \rVert_{\mathrm{Euc}}$. Under the *canonical metric* $(\beta = 1/2)$, however, let $n \ge 2p$, take a purely vertical tangent vector $\zeta = X A$ ($A \in \mathfrak{so}(p) \setminus \lbrace 0 \rbrace$, so
   $\lVert \zeta \rVert_{\mathrm{can}}^2 = \tfrac{1}{2} \lVert A \rVert_{\mathrm F}^2$), and step to a frame $Y$ with $Y^\top X = 0$. Then $Y^\top \zeta = 0$, so
   $P_{T_Y}(\zeta) = \zeta$ becomes purely horizontal at $Y$, giving:

```math
\left\lVert \mathcal T_{\tau \xi}^{\mathrm{proj}}(\zeta) \right\rVert_{\mathrm{can}}^2 = \lVert A \rVert_{\mathrm F}^2 = 2 \lVert \zeta \rVert_{\mathrm{can}}^2 > \lVert \zeta \rVert_{\mathrm{can}}^2.
```

  Thus $\mathcal T^{\mathrm{proj}}$ can **inflate canonical tangent norms** by up to $\sqrt{2}$, violating Ring and Wirth's (2012) bounded-transport hypothesis $\lVert \mathcal T(\eta_k) \rVert_{x_{k+1}} \le \lVert \eta_k \rVert_{x_k}$ (cf. Sato 2022, `2112.02572`, Sec. 2.2)!
  2. **Non-isometric and rank-deficient at orthogonal rotations**: Because $(\mathcal T^{\mathrm{proj}})^{-\sharp} \ne \mathcal T^{\mathrm{proj}}$, two-loop L-BFGS loses exact secant interpolation $H_{k+1} y_k = s_k$ unless an inverse-projection system is solved, which becomes singular at $\pi/2$ subspace rotations where
   $\sigma_{\min}(X^\top Y) = 0$.

### 4.2 Head-to-Head Transport Comparison Table

| Criterion | Isometric Cayley $\mathcal T^{\mathrm{Cay}}$ | Differentiated Cayley $\mathcal T^R$ | Projection Transport $\mathcal T^{\mathrm{proj}}$ |
| :--- | :--- | :--- | :--- |
| **Exact Isometry** ($\beta = 1/2, 1$) | **Yes** — $\lVert \mathcal T^{\mathrm{Cay}}(\zeta) \rVert_\beta = \lVert \zeta \rVert_\beta$ for all $\tau$ | No (strictly contractive for $\tau > 0$) | No (contracts under $\beta = 1$; can inflate by $\sqrt{2}$ under $\beta = 1/2$) |
| **Self-Inverse-Adjoint** $((\mathcal T)^{-\sharp} = \mathcal T)$ | **Yes** (preserves SPD two-loop `H_k` directly) | No (requires adjoint inverse or locking) | No (distorts secant equation $H_{k+1} y_k = s_k$) |
| **Global Nonsingularity** ($\tau \in \mathbb R$) | **Yes** — $\det Q(\tau) = +1$ everywhere | No (can lose rank on `T_X` at large $\tau$) | No — singular when $\sigma_{\min}(X^\top Y) = 0$ |
| **Deviation from** $\mathcal T^R(\xi)$ | $O(\tau^2 \lVert \xi \rVert^3)$ (`Zhu 2017`, Lemma 4) | $0$ — exact curve velocity $\dot{Y}(\tau)$ | $O(\tau \lVert \xi \rVert^2)$ (first-order discrepancy) |
| **Transport Cost** ($2m$ pairs, $p \le n/2$) | $8 m n p^2 + 10 m p^3$ — reuses $J(\tau_k)$ LU | $> 16 m n p^2 + O(m p^3)$ + inverse solves | $8 m n p^2 + 2 m p^2$ ($2m \times 4 n p^2$) |
| **Transport Cost** ($2m$ pairs, $p > n/2$) | $4 m n^2 p$ — reuses $M_n(\tau_k)$ LU | $> 8 m n^2 p$ + inverse solves | $8 m n p^2$ ($> 4 m n^2 p$ whenever $p > n/2$!) |

Notice a remarkable bonus in the bottom two rows: for $p \le n/2$, $\mathcal T^{\mathrm{Cay}}$ via Formulation 1B has the *exact same* $8 m n p^2$ leading FLOP count as naive projection transport $\mathcal T^{\mathrm{proj}}$, and for $p > n/2$, $\mathcal T^{\mathrm{Cay}}$ via Formulation 2 ($4 m n^2 p$ FLOPs) is actually **up to twice as fast** as $\mathcal T^{\mathrm{proj}}$ ($8 m n p^2$ FLOPs)!

### 4.3 Locking Condition vs Lock-Free Curvature-Guarded `LRBFGS`

In Huang, Gallivan, and Absil (2015, *SIAM J. Optim.* 25(3), 1660--1685), convergence under Wolfe line search used an isometric transport $\mathcal T^{\mathrm{lock}}$ modified by two Householder reflections so as to enforce the **locking condition**:

```math
\mathcal T_{\eta_k}^{\mathrm{lock}}(\eta_k) = \beta_k \mathcal T_{\eta_k}^R(\eta_k), \qquad \beta_k = \frac{\lVert \eta_k \rVert_{X_k}}{\lVert \mathcal T_{\eta_k}^R(\eta_k) \rVert_{X_{k+1}}}.
```

Why does `fortchuck` **not** need Householder locking?
1. **Second-Order Agreement & Acute-Angle Bound**: Comparing $\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\xi) = (I_n - \tfrac{\tau}{2} W_\xi)^{-1}(I_n + \tfrac{\tau}{2} W_\xi)\xi$ with
  $\mathcal T_{\tau \xi}^R(\xi) = (I_n - \tfrac{\tau}{2} W_\xi)^{-2} \xi$ yields:

```math
\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\xi) = \left(I_n - \frac{\tau^2}{4} W_\xi^2\right) \mathcal T_{\tau \xi}^R(\xi) \implies \left\lVert \mathcal T_{\tau \xi}^{\mathrm{Cay}}(\xi) - \mathcal T_{\tau \xi}^R(\xi) \right\rVert_\beta = O(\tau^3 \lVert \xi \rVert_\beta^3) \text{ (absolute)}, \; O(\tau^2 \lVert \xi \rVert_\beta^2) \text{ (relative)}.
```

Moreover, Zhu (2017, Lemma 4) and Wang et al. (2024, `2311.00907`, Eq. 26) prove that the angle $\theta_k$ between
$\mathcal T_{\tau \xi}^{\mathrm{Cay}}(\xi)$ and
$\mathcal T_{\tau \xi}^R(\xi)$ is strictly acute for all $\tau > 0$, bounded below by
$\cos \theta_k \ge \sqrt{(4 + \tau^2 \sigma_{\min}^2)/(4 + \tau^2 \sigma_{\max}^2)} = 1 - O(\tau^2 \lVert \xi \rVert^2)$.
2. **Huang--Absil--Gallivan (2018) Lock-Free Convergence Theorem**: Huang, Absil, and Gallivan (2018, *SIAM J. Optim.* 28(1), 470--495) proved that replacing the locking condition and Wolfe line search with *any isometric smooth vector transport* (specifically $\mathcal T^{\mathrm{Cay}}$), *Armijo backtracking*, and a *cautious curvature guard* yields global convergence on general nonconvex Riemannian manifolds and local superlinear convergence near nondegenerate local minimizers, without ever evaluating $\mathcal T^R$!

### 4.4 Eager Ring-Buffer Two-Loop Recursion for `fortchuck`

Huang, Gallivan, and Absil (2015) originally stored $(S_i, Y_i)$ in historical tangent spaces
$T_{X_i}$ and cascaded transports during each two-loop call ($O(m^2)$ transports). Because $m$ is small ($m \in [5, 12]$, default $m = 7$) and $\mathcal T^{\mathrm{Cay}}$ reuses the single step factorization at
$X_k \to X_{k+1}$, **eager ring-buffer transport** is strictly superior:
- Maintain a circular buffer of at most $m$ pairs $(S_i, Y_i)$ residing in the **current** tangent space
  $T_{X_k} \mathcal M$, together with cached scalars
  $\rho_i = 1 / \langle S_i, Y_i \rangle_{\mathcal M} > 0$.
- Because $\mathcal T^{\mathrm{Cay}}$ is an exact linear isometry on every Stiefel block (and the identity map on free Euclidean box coordinates), inner products $\langle S_i, Y_i \rangle_{\mathcal M}$ are **invariant under transport**
  ($\rho_i$ never needs recomputation after creation!).
- At iteration $k$:
  1. Compute $q \leftarrow g_k = \mathrm{grad}\,f(x_k)$ (masked on active box bounds per `arxiv-coupled`'s specification).
  2. **Backward loop** ($i = m_{\mathrm{cur}}, \dots, 1$):
   $\alpha_i = \rho_i \langle S_i, q \rangle_{\mathcal M}$,
   $q \leftarrow q - \alpha_i Y_i$.
  3. **Initial scaling**: $r \leftarrow \gamma_k q$.
  4. **Forward loop** ($i = 1, \dots, m_{\mathrm{cur}}$):
   $\beta_{\mathrm{loop}} = \rho_i \langle Y_i, r \rangle_{\mathcal M}$,
   $r \leftarrow r + (\alpha_i - \beta_{\mathrm{loop}}) S_i$.
  5. Set search direction $d_k = -r$ (total two-loop cost:
   $4 m_{\mathrm{cur}} \dim_{\mathrm{amb}}$ fused-multiply-add `FMA` FLOPs, pure Level-1 vector sweeps).
  6. Once $\tau_k$ is accepted and
   $x_{k+1}$ is formed, transport all stored Stiefel blocks
   $\lbrace S_{i,r}, Y_{i,r} \rbrace_{i=1}^{m_{\mathrm{cur}}}$ plus the current step
   $\tau_k \xi_{k,r}$ and old gradient
   $g_{k,r}$ from
   $T_{X_{k,r}}$ to
   $T_{X_{k+1,r}}$ in **one single batched call** reusing the already-factored
   $J_r(\tau_k)$ or
   $M_{n_r}(\tau_k)$!

---

## 5. Spectral Initialization, Curvature Safeguards, and Line Search

### 5.1 Barzilai--Borwein Spectral Scaling ($\gamma_k$ and Fallback
$\tau_k^{\mathrm{BB}}$)

Let $s_k = (s_{k,z}, s_{k,1}, \dots, s_{k,R})$ and
$y_k = (y_{k,z}, y_{k,1}, \dots, y_{k,R})$ denote the joint secant displacement and gradient difference in
$T_{x_{k+1}} \mathcal M$, where on each Stiefel factor $r$:

```math
s_{k,r} = \mathcal T_{\tau_k \xi_{k,r}}^{\mathrm{Cay}}(\tau_k \xi_{k,r}) \in T_{X_{k+1,r}} \mathrm{St}(n_r, p_r), \qquad y_{k,r} = \mathrm{grad}^{\mathrm{can}} f_r(x_{k+1}) - \mathcal T_{\tau_k \xi_{k,r}}^{\mathrm{Cay}}\!\left(\mathrm{grad}^{\mathrm{can}} f_r(x_k)\right),
```

and on the Euclidean box block $s_{k,z} = z_{k+1} - z_k$ (projected to free coordinates at $k+1$; see `ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md`). Under the product metric
$\langle \cdot, \cdot \rangle_{\mathcal M}$:

```math
\sigma_{ss} = \langle s_k, s_k \rangle_{\mathcal M}, \qquad \sigma_{sy} = \langle s_k, y_k \rangle_{\mathcal M}, \qquad \sigma_{yy} = \langle y_k, y_k \rangle_{\mathcal M}.
```

Following Wen and Yin (2013), Jiang and Dai (2015, `1301.0172`), Iannazzo and Porcelli (2018, *IMA J. Numer. Anal.* 38, 495--517), and Hu et al. (2020, `1906.05450`):
1. **L-BFGS Initial Inverse-Hessian Scaling** ($m_{\mathrm{cur}} \ge 1$): Use the short Barzilai--Borwein (BB2 / Oren--Luenberger) Rayleigh quotient safeguarded to
  $[\gamma_{\min}, \gamma_{\max}]$ (with
  $\gamma_{\min} = 10^{-12}, \gamma_{\max} = 10^{12}$):

```math
\gamma_{k+1} = \mathrm{clip}\!\left(\frac{\sigma_{sy}}{\sigma_{yy}}, \, \gamma_{\min}, \, \gamma_{\max}\right), \qquad H_{k+1}^{(0)} = \gamma_{k+1} I, \qquad \tau_{\mathrm{init}} = 1.0.
```

Trying $\tau_{\mathrm{init}} = 1.0$ first whenever
$m_{\mathrm{cur}} \ge 1$ is mandatory to achieve superlinear convergence near local minimizers without step-size distortion.
2. **Initial Iteration and Steepest-Descent Fallback** ($k = 0$ or $m_{\mathrm{cur}} = 0$): At $k = 0$ (no history), use normalized gradient step
  $\tau_0 = \min(1.0, \, 1.0 / \max(1.0, \lVert g_0 \rVert_{\mathcal M}))$. If `LRBFGS` runs with $m = 0$ (pure Riemannian BB mode) or restarts after an active-set / descent reset, use *Alternating Barzilai--Borwein (ABB)* (`Wen & Yin 2013`; `1301.0172`, Eq. 4.3):

```math
\tau_k^{\mathrm{BB1}} = \frac{\sigma_{ss}}{\lvert \sigma_{sy} \rvert}, \qquad \tau_k^{\mathrm{BB2}} = \frac{\lvert \sigma_{sy} \rvert}{\sigma_{yy}}, \qquad \tau_k^{\mathrm{ABB}} = \mathrm{clip}\!\left(\begin{cases} \tau_k^{\mathrm{BB1}} & \text{if } k \bmod 2 = 1 \\ \tau_k^{\mathrm{BB2}} & \text{if } k \bmod 2 = 0 \end{cases}, \, \tau_{\min}, \, \tau_{\max}\right),
```

with $\tau_{\min} = 10^{-16}$ and
$\tau_{\max} = 10^{6}$.

### 5.2 Cautious Update vs Riemannian Powell Damping for $\langle s_k, y_k \rangle_{\mathcal M} > 0$

On nonconvex manifolds with positive sectional curvature ($K_{\max} = 5/4$) and Armijo line search (which imposes no lower slope condition),
$\sigma_{sy} = \langle s_k, y_k \rangle_{\mathcal M}$ can be zero or negative. Two literature-proven remedies maintain strict positive definiteness of
$H_{k+1}$:

1. **Huang--Absil--Gallivan (2018) Cautious Update (Recommended Default)**:
Accept the pair $(s_k, y_k)$ into the L-BFGS ring buffer if and only if:

```math
\langle s_k, y_k \rangle_{\mathcal M} \ge \max\!\left(\epsilon_{\mathrm{caut}} \lVert s_k \rVert_{\mathcal M}^2 \min\!\left(1, \lVert g_k \rVert_{\mathcal M}^\nu\right), \; \epsilon_{\mathrm{mach}} \lVert s_k \rVert_{\mathcal M} \lVert y_k \rVert_{\mathcal M}\right),
```

with $\epsilon_{\mathrm{caut}} = 10^{-8}$ and $\nu = 1$ (`Huang, Absil & Gallivan 2018`, Sec. 2.2). Notice the
$\min(1, \lVert g_k \rVert_{\mathcal M}^\nu)$ cap: far from stationarity it prevents
$\lVert g_k \rVert^\nu \gg 1$ from rejecting informative secant pairs, while as
$\lVert g_k \rVert_{\mathcal M} \to 0$ near a strong local minimizer (where positive Riemannian Hessian eigenvalues exceed
$\epsilon_{\mathrm{caut}} \lVert g_k \rVert$), **every update is accepted asymptotically**, guaranteeing undisrupted local superlinear convergence!

2. **Limited-Memory Riemannian Powell Damping** (Optional `curvature_mode = POWELL`):
In dense BFGS, Powell (1978) damps $y_k$ toward
$B_k s_k$. In limited-memory L-BFGS, forming
$B_k s_k$ without compact matrix inversions is expensive; instead, standard L-BFGS Powell damping (`Nocedal & Wright 2006`; `Al-Baali 2014`) damps
$y_k$ toward the spectral prior
$\gamma_k^{-1} s_k$ (using the previous accepted BB2 inverse-Hessian scale
$\gamma_k$) whenever
$\sigma_{sy} < \omega \gamma_k^{-1} \sigma_{ss}$ with damping threshold $\omega \in (0, 0.2]$ (default $\omega = 0.1$, provided
$\sigma_{yy} > 0$):

```math
\theta_k = \frac{(1 - \omega) \gamma_k^{-1} \sigma_{ss}}{\gamma_k^{-1} \sigma_{ss} - \sigma_{sy}} \in (0, 1), \qquad \tilde y_k = \theta_k y_k + (1 - \theta_k) \gamma_k^{-1} s_k,
```

which enforces $\langle s_k, \tilde y_k \rangle_{\mathcal M} = \omega \gamma_k^{-1} \lVert s_k \rVert_{\mathcal M}^2 > 0$ identically. Because cautious skipping avoids injecting synthetic isotropic curvature along negative-curvature saddle directions (where negative curvature helps escape saddles!), **cautious skipping should be default**, with Powell damping selectable via solver options.

3. **Descent Safeguard**: Before entering line search, verify $\langle g_k, d_k \rangle_{\mathcal M} \le -\epsilon_{\mathrm{desc}} \lVert g_k \rVert_{\mathcal M} \lVert d_k \rVert_{\mathcal M}$
  ($\epsilon_{\mathrm{desc}} = 10^{-12}$); if violated due to extreme roundoff or active-set shift, flush the ring buffer
  ($m_{\mathrm{cur}} \leftarrow 0$) and reset
  $d_k = -g_k$.

### 5.3 Line Search on Coupled Arcs: Why Generalized Armijo / Zhang--Hager Beats Strong Wolfe

Three recent theoretical and structural facts rule out Strong Wolfe as the primary line search for `fortchuck`:
1. **Nonconvex Wolfe Counterexample** (`2609.09686`, Diao, Sep 2026): Diao proved that even in smooth Euclidean unconstrained optimization on bounded-below nonconvex functions, unmodified BFGS and L-BFGS can *fail to converge under exact Strong Wolfe line search* for every parameter pair $0 < c_1 < c_2 < 1$! Thus Strong Wolfe alone does *not* eliminate the need for cautious/damped curvature safeguards on nonconvex manifolds, while doubling or tripling per-iteration gradient calls during bisection/zoom phases.
2. **Nonsmooth Kinks Along Coupled Box Projection**: Along the trial curve $z(\tau) = P_{[l, u]}(z_k + \tau d_{k,z})$,
  $\tau \mapsto f(z(\tau), R_{X_k}^{\mathrm{Cay}}(\tau \xi_k))$ is only piecewise smooth, with downward or upward slope jumps at every breakpoint
  $\tau_i = (l_i - z_{k,i})/d_{k,z,i}$ or
  $(u_i - z_{k,i})/d_{k,z,i}$. Strong Wolfe's curvature condition
  $\lvert \phi'(\tau) \rvert \le c_2 \lvert \phi'(0) \rvert$ can cease to admit any point in an interval straddling a bound kink!
3. **Extra Differentiated-Retraction Cost**: Evaluating $\phi'(\tau)$ on $\mathrm{St}(n, p)$ requires both a full user gradient evaluation $\nabla f$ at every rejected trial $\tau$ and evaluating $\mathcal T_{\tau \xi}^R(\xi) = (I_n - \tfrac{\tau}{2} W_\xi)^{-2} \xi$.

By contrast, the **Zhang--Hager Nonmonotone / Monotone Armijo Backtracking Scheme** (Zhang & Hager 2004, *SIAM J. Optim.* 14(4), 1043--1056; Wen & Yin 2013; Oviedo 2022; Ding, Li & Zhang 2024, `2404.19392`) requires *function evaluations only* on rejected steps and unifies monotone and nonmonotone search via a single parameter $\eta_{\mathrm{ZH}} \in [0, 1)$:

```math
f\!\left(z(\tau_k), R_{X_k}^{\mathrm{Cay}}(\tau_k \xi_k)\right) \le C_k + c_1 \tau_k \Delta_k(\tau_k),
```

where $\Delta_k(\tau_k) = \tau_k^{-1} g_k^\top_{\text{box}}(z(\tau_k) - z_k) + \sum_{r=1}^R g_{X_{k,r}}^{\mathrm{can}}(\mathrm{grad}^{\mathrm{can}} f_r(x_k), \xi_{k,r}) < 0$ is the coupled Bertsekas/Armijo reference slope, and the reference level
$(Q_k, C_k)$ updates in $O(1)$ time via:

```math
Q_0 = 1, \quad C_0 = f(x_0), \qquad Q_{k+1} = \eta_{\mathrm{ZH}} Q_k + 1, \qquad C_{k+1} = \frac{\eta_{\mathrm{ZH}} Q_k C_k + f(x_{k+1})}{Q_{k+1}}.
```

- **Default Nonmonotonicity Policy** for $\eta_{\mathrm{ZH}}$:
  - In `LRBFGS` **mode** ($m \ge 1$): Default to $\eta_{\mathrm{ZH}} = 0$ (strictly monotone Armijo,
   $C_k = f(x_k)$) or very mild nonmonotonicity
   ($\eta_{\mathrm{ZH}} \in [0, 0.2]$), with
   $c_1 = 10^{-4}$ and safeguarded quadratic/backtracking shrinkage
   $\tau \leftarrow \beta_{\mathrm{ls}} \tau$
   ($\beta_{\mathrm{ls}} = 0.5$). Strict monotonicity prevents unit quasi-Newton steps
   ($\tau_0 = 1$) from overshooting narrow basins while guaranteeing
   $\sum_k \tau_k \lvert \Delta_k \rvert < \infty$.
  - In **BB / Gradient mode** ($m = 0$): Default to $\eta_{\mathrm{ZH}} = 0.85$ (`Wen & Yin 2013` `OptStiefelGBB` standard), since BB spectral steps intentionally oscillate nonmonotonically across the Hessian spectrum to achieve $R$-linear acceleration, with Ding, Li, and Zhang (2024, `2404.19392`, Thm. 4.3) establishing global $O(\epsilon^{-2})$ gradient complexity
   $\min_{0 \le k \le K} \lVert \mathrm{grad}\,f(X_k) \rVert \le O(1 / \sqrt{(1 - \eta_{\mathrm{ZH}})(K + 1)})$.

---

## 6. Numerical Stability: Antipodal Conditioning and Drift Reorthogonalization

### 6.1 Antipodal Cayley Singularity Analysis and Step-Conditioning Guards

Kume and Yamada (2023, `2305.17901`) and Jensen and Zimmermann (2026, `2608.28073`) give the complete geometric picture of Cayley singularities on $\mathrm{St}(n, p)$:
1. **Why Forward Cayley Steps Are Mathematically Nonsingular Everywhere**: For any skew-symmetric $W_\xi \in \mathfrak{so}(n)$ and any finite $\tau \in \mathbb R$, $-1$ is *never* an eigenvalue of
  $Q_\xi(\tau) = (I_n - \tfrac{\tau}{2} W_\xi)^{-1}(I_n + \tfrac{\tau}{2} W_\xi)$ because $(1 + i\lambda)/(1 - i\lambda) \ne -1$ for all $\lambda \in \mathbb R$. The exceptional antipodal set relative to $X$:

```math
E_{n,p}(X) = \left\lbrace Y \in \mathrm{St}(n, p) : \det(I_p + X^\top Y) = 0 \right\rbrace = \left\lbrace Y \in \mathrm{St}(n, p) : -1 \in \sigma(X^\top Y) \right\rbrace
```

has measure zero (it is reached only in the projective limit $\tau \to \pm \infty$). Moreover, whereas global single-center Cayley parameterizations stagnate near $E_{n,p}(X_0)$, `fortchuck` **re-centers the Cayley chart** at
$X_k$ on every iteration, so `Fact 2.3` of Kume and Yamada (`2305.17901`) guarantees full local mobility around any target $X^\star \in \mathrm{St}(n, p)$ in at most two steps.
2. **Where Floating-Point Singularity Hazards Actually Arise**:
   - **Asymptotic Antipodal Winding** ($\tau \lVert \xi_r \rVert_{\mathrm{can}} \gg \pi$): If an initial step
   $\tau_0$ produces
   $\tau_0 \lVert \xi_r \rVert_{\mathrm{can}} \gg 1$, then
   $2 J_r(\tau_0)^{-1} = I_{p_r} + X_r^\top Y_r(\tau_0)$ approaches
   $E_{n,p}(X_r)$, driving
   $\sigma_{\min}(J_r(\tau_0)) \to 0$ and growing
   $\kappa_2(J_r(\tau_0)) \le (5 + \tau_0^2 \lVert \xi_r \rVert_{\mathrm F}^2)/4$ quadratically (`1301.0172`, Eq. 1378; `2608.28073`).
   - **Pivot / Conditioning Safeguard for `fortchuck`**:
     1. **Geometric velocity cap before line search**: Ensure the initial trial step satisfies $\tau_0 \max_{1 \le r \le R} \lVert \xi_r \rVert_{\mathrm{can}} \le \Delta_{\max}$ with default
   $\Delta_{\max} = 2\pi$ (matching the shortest Stiefel geodesic loop
   $\ell_{1/2} = 2\pi$ from `2511.01563` and bounding
   $\kappa_2(J_r) \le (5 + 8\pi^2)/4 \approx 21$!).
     2. **LU pivot health check inside backtracking**: During `GETRF` of $J_r(\tau)$ (Branch A) or
   $M_{n_r}(\tau)$ (Branch B), if `info /= 0`, any `IEEE_IS_NAN` appears, or the diagonal U-factor ratio violates
   $\min_i \lvert U_{ii} \rvert < \epsilon_{\mathrm{piv}} \max_i \lvert U_{ii} \rvert$
   ($\epsilon_{\mathrm{piv}} = 10^{-12}$), immediately reject $\tau$ and halve $\tau \leftarrow 0.5 \tau$ without calling the user objective $f$! Since
   $J_r(\tau) \to I_{p_r}$ and
   $M_{n_r}(\tau) \to I_{n_r}$ as $\tau \to 0^+$, halving $\tau$ is guaranteed to restore
   $\kappa_2 \to 1$ in finitely many halvings.

### 6.2 Feasibility Drift Amplification and Sign-Preserving Reorthogonalization

In exact arithmetic, $Y_r = Q_{\xi_r}(\tau) X_r$ preserves
$Y_r^\top Y_r = X_r^\top X_r = I_{p_r}$. In IEEE-754 `real64`
($\epsilon_{\mathrm{mach}} \approx 2.22 \times 10^{-16}$), let
$\Xi_X = X^\top X - I_p$ denote accumulated orthogonality drift. Jiang and Dai (2015, `1301.0172`, Sec. 4.3, Eq. 4.9) prove that because
$X^\top K = -\Xi_X (X^\top \xi) \ne 0$ when
$\Xi_X \ne 0$, each Cayley step can amplify existing drift multiplicatively:

```math
\lVert \Xi_Y \rVert_{\mathrm F} \le \left(1 + 2 \tau \lVert X^\top \xi \rVert_{\mathrm F}\right) \lVert \Xi_X \rVert_{\mathrm F} + O(\epsilon_{\mathrm{mach}}).
```

Over hundreds of iterations, unchecked roundoff can grow from $10^{-16}$ past $10^{-12}$, corrupting $X^\top \xi + \xi^\top X = 0$ and degrading quasi-Newton secants.

- **Drift Check & Trigger Threshold (`OptStiefelGBB` Standard)**:
  At the end of each accepted iteration (for each factor $r = 1, \dots, R$), compute the residual matrix $E_r = X_r^\top X_r - I_{p_r}$ (`SYRK`,
  $n_r p_r^2$ FLOPs) and check:

```math
\delta_r = \lVert X_r^\top X_r - I_{p_r} \rVert_{\mathrm F} > \epsilon_{\mathrm{orth}}, \qquad \epsilon_{\mathrm{orth}} = 10^{-13}.
```

- **How to Reorthogonalize Without Corrupting Quasi-Newton State (Critical Sign Trap!)**:
  Standard textbook Householder QR (`LAPACK` `DGEQRF`) chooses reflection signs to avoid cancellation, yielding $X = Q R$ where diagonal entries $R_{ii}$ have *arbitrary signs* $\pm 1$! If called mid-optimization when $X \approx \tilde{X} \in \mathrm{St}(n, p)$, naive QR flips columns of $X$ by $-1$, jumping to a distant component/point of $\mathrm{St}(n, p)$ and invalidating $f(X)$ and the entire L-BFGS history! `fortchuck` must use a **two-tier distance-minimizing reorthogonalizer**:

#### Tier 1: Fast Normal-Space Newton--Schulz Polish when $10^{-13} < \delta_r \le 10^{-4}$ (Typical Mid-Run Drift)

Reusing the **already-computed** residual $E_r = X_r^\top X_r - I_{p_r}$, apply one order-1 Newton--Schulz step (`2007.03638` Hauru et al. 2021; `2605.02838` Xiong, Gao & Absil 2026, Eq. 1.7; `2608.22774` Su & Huang 2026):

```math
X_r^{\mathrm{new}} = X_r \left(I_{p_r} - \frac{1}{2} E_r\right) = \frac{1}{2} X_r \left(3 I_{p_r} - X_r^\top X_r\right).
```

Because $(X_r^{\mathrm{new}})^\top X_r^{\mathrm{new}} - I_{p_r} = -\tfrac{3}{4} E_r^2 + \tfrac{1}{4} E_r^3$, a drift of
$\delta_r = 10^{-13}$ contracts quadratically in **one single `GEMM` call**
($2 n_r p_r^2$ FLOPs) to
$\tfrac{3}{4} \delta_r^2 \approx 7.5 \times 10^{-27} \to O(\epsilon_{\mathrm{mach}})$! Moreover,
$- \tfrac{1}{2} X_r E_r$ lies purely in the normal space
$N_{X_r} \mathrm{St}(n_r, p_r)$, approximates the exact Riemannian polar projection
$X_r (X_r^\top X_r)^{-1/2}$ to
$O(\delta_r^2) \approx 10^{-26}$ (zero column coupling or sign flips!), and shifts
$X_r$ by only
$\tfrac{1}{2} \delta_r \approx 5 \times 10^{-14}$ — small enough that stored L-BFGS tangent pairs only need a cheap tangent re-projection
$S_i \leftarrow P_{T_{X_r^{\mathrm{new}}}}(S_i)$ and
$Y_i \leftarrow P_{T_{X_r^{\mathrm{new}}}}(Y_i)$ without flushing history!

#### Tier 2: Unique Positive-Diagonal Thin Householder QR ($\mathrm{qf}$) when $\delta_r > 10^{-4}$ (User Input Projection or Extreme Recovery)

Compute thin Householder QR $X_r = Q R$
($2 n_r p_r^2 - \tfrac{2}{3} p_r^3$ FLOPs) and **explicitly enforce**
$R_{ii} > 0$ (`Sato 2022`, `2112.02572`, Example 2.1) by multiplying column $i$ of $Q$ by
$\mathrm{sign}(R_{ii}) \in \lbrace -1, +1 \rbrace$, taking $\mathrm{sign}(0) = +1$. When
$\delta_r > 10^{-4}$ occurs mid-solve, also re-evaluate $f$ and flush the L-BFGS history
($m_{\mathrm{cur}} \leftarrow 0$). Note that on $\mathrm{SO}(n)$
($p_r = n_r$ and
$\det X_0 = +1$), both Cayley steps
($\det Q_\xi(\tau) = +1$ for all $\tau$) and continuous Tier 1 / Tier 2 polishing preserve
$\det X_r = +1$ automatically.

---

## 7. Assessment of 2020--2026 Inversion-Free and Landing Alternatives

A major 2020--2026 research wave explored avoiding matrix inversions on $\mathrm{St}(n, p)$. We evaluate each family for `fortchuck` and explain why feasible dual-regime Cayley remains the optimal core engine:

| Method Family | Representative Papers | Core Update Mechanism | Why Not Primary Engine for `fortchuck` |
| :--- | :--- | :--- | :--- |
| **Iterative Cayley Fixed-Point** | Li, Fuxin & Todorovic (2020, `2002.01113`) | $s$-step Picard iteration $Y^{(j+1)} = X + \tfrac{\tau}{2} W(X + Y^{(j)})$ ($s = 2$) | Contraction requires $\tau \lVert W \rVert_2 < 2$ (fails on aggressive L-BFGS/BB steps!), leaves $O(\tau^{2+s})$ orthogonality error per step, and targets GPU deep-learning SGD rather than high-accuracy CPU `real64` QN. |
| **First-Order Landing Flows** | Gao, Vary, Ablin & Absil (2022, `2202.09058`); Ablin et al. (2024, `2303.16510`); Goyens, Absil & Feppon (2025, `2507.15638`) | $X_{k+1} = X_k - \eta_k \left(2\,\mathrm{skew}(G_k X_k^\top) X_k + \lambda X_k (X_k^\top X_k - I_p)\right)$ | Iterates leave $\mathrm{St}(n, p)$: *breaks user callbacks* $f(z, X)$ requiring strictly orthonormal inputs (e.g., $\log\det(X^\top A X)$, Cholesky, Kohn--Sham DFT orbitals), requires tuning hyperparameter $\lambda > 0$, and converges only linearly. |
| **1-Step Newton--Schulz Inexact RGD** | Su & Huang (2026, `2608.22774`) | $Y_k = X_k - \eta_k \mathrm{grad}\,f(X_k), \; X_{k+1} = \tfrac{1}{2} Y_k (3 I_p - Y_k^\top Y_k)$ | Avoids penalty $\lambda$, but intermediate iterates remain infeasible (`O(eta_k^4)` drift), step size is restricted to a safe basin around $\mathrm{St}(n, p)$, and no isometric L-BFGS transport exists across infeasible layers. (Adopted instead as `fortchuck`'s Tier 1 drift polish!) |
| **Second-Order Newton--Schulz Landing (`SOL`)** | Xiong, Gao & Absil (2026, `2605.02838`) | Coupled modified Riemannian Newton tangent equation + NS normal step $N_1(X) = -\tfrac{1}{2} X(X^\top X - I_p)$ | Requires second-order user Hessian-vector products $\nabla^2 f(X)[T]$ (violating `fortchuck`'s gradient-only quasi-Newton interface) and still evaluates $f$ at infeasible iterates before landing. |
| **Constraint-Dissolving (`CDF` / `ExPen`)** | Xiao, Liu & Toh (2022/2024, `2203.10319`) | Unconstrained L-BFGS on smooth exact penalty $h(X) = f(\tfrac{1}{2} X(3 I_p - X^\top X)) + \tfrac{\beta}{4}\lVert X^\top X - I_p \rVert_{\mathrm F}^2$ | Evaluates $f$ at $\tilde{X} = \tfrac{1}{2} X(3 I - X^\top X)$ (still infeasible!), alters the user gradient formula via chain-rule Jacobian of $\tilde{X}(X)$, and introduces problem-dependent penalty scaling $\beta$ that worsens Hessian conditioning. |

---

## 8. Complete Bibliography and ArXiv URL Registry

All papers synthesized in this report are listed below with explicit arXiv URLs and publication DOIs:

1. **Absil, P.-A., Mahony, R., & Sepulchre, R. (2008).** *Optimization Algorithms on Matrix Manifolds*. Princeton University Press. URL: https://press.princeton.edu/absil
2. **Absil, P.-A., & Mataigne, S. (2025).** The ultimate upper bound on the injectivity radius of the Stiefel manifold. *SIAM Journal on Matrix Analysis and Applications*, 46(2), 1145--1167. arXiv: https://arxiv.org/abs/2403.03782 | DOI: https://doi.org/10.1137/24M1644572
3. **Ablin, P., Vary, S., Gao, B., & Absil, P.-A. (2024).** Infeasible deterministic, stochastic, and variance-reduction algorithms for optimization under orthogonality constraints. *Journal of Machine Learning Research*, 25(304), 1--45. arXiv: https://arxiv.org/abs/2303.16510
4. **Boumal, N. (2023).** *An Introduction to Optimization on Smooth Manifolds*. Cambridge University Press. URL: https://www.nicolasboumal.net/book/ | DOI: https://doi.org/10.1017/9781009166164
5. **Diao, Z. (2026).** Non-convergence of the BFGS method under strong Wolfe line search. arXiv preprint. arXiv: https://arxiv.org/abs/2609.09686
6. **Ding, W., Li, J., & Zhang, S. (2024).** Transformed gradient projection algorithms for optimization problems over compact submanifolds. arXiv preprint. arXiv: https://arxiv.org/abs/2404.19392
7. **Edelman, A., Arias, T. A., & Smith, S. T. (1998).** The geometry of algorithms with orthogonality constraints. *SIAM Journal on Matrix Analysis and Applications*, 20(2), 303--353. arXiv: https://arxiv.org/abs/physics/9806030 | DOI: https://doi.org/10.1137/S0895479895290954
8. **Gao, B., Liu, X., Chen, X., & Yuan, Y.-X. (2018).** A new first-order algorithmic framework for optimization problems with orthogonality constraints. *SIAM Journal on Optimization*, 28(1), 302--332. arXiv: https://arxiv.org/abs/1609.06049 | DOI: https://doi.org/10.1137/16M1098759
9. **Gao, B., Vary, S., Ablin, P., & Absil, P.-A. (2022).** Optimization flows landing on the Stiefel manifold. *IFAC-PapersOnLine*, 58(17), 24--29. arXiv: https://arxiv.org/abs/2202.09058 | DOI: https://doi.org/10.1016/j.ifacol.2024.10.141
10. **Goyens, F., Absil, P.-A., & Feppon, F. (2025).** Geometric design of the tangent step in the landing algorithm for orthogonality constraints. arXiv preprint. arXiv: https://arxiv.org/abs/2507.15638
11. **Hauru, M., Van Damme, M., & Haegeman, J. (2021).** Riemannian optimization of isometric tensor networks. *SciPost Physics*, 10(2), 040. arXiv: https://arxiv.org/abs/2007.03638 | DOI: https://doi.org/10.21468/SciPostPhys.10.2.040
12. **Hu, J., Liu, X., Wen, Z., & Yuan, Y.-X. (2020).** A brief introduction to manifold optimization. *Journal of the Operations Research Society of China*, 8(2), 199--248. arXiv: https://arxiv.org/abs/1906.05450 | DOI: https://doi.org/10.1007/s40305-020-00295-9
13. **Huang, W., Gallivan, K. A., & Absil, P.-A. (2015).** A Broyden class of quasi-Newton methods for Riemannian optimization. *SIAM Journal on Optimization*, 25(3), 1660--1685. DOI: https://doi.org/10.1137/140955483
14. **Huang, W., Absil, P.-A., & Gallivan, K. A. (2018).** A Riemannian BFGS method without differentiated retraction for nonconvex optimization problems. *SIAM Journal on Optimization*, 28(1), 470--495. DOI: https://doi.org/10.1137/17M1127582
15. **Hüper, K., Markina, I., & Silva Leite, F. (2021).** A Lagrangian approach to extremal curves on Stiefel manifolds. *Journal of Geometric Mechanics*, 13(1), 55--72. DOI: https://doi.org/10.3934/jgm.2020031
16. **Iannazzo, B., & Porcelli, M. (2018).** The Riemannian Barzilai--Borwein method with nonmonotone line search and the matrix geometric mean computation. *IMA Journal of Numerical Analysis*, 38(1), 495--517. DOI: https://doi.org/10.1093/imanum/drx015
17. **Jensen, M. L., & Zimmermann, R. (2026).** Properties of the Cayley retraction on the Stiefel manifold with focus on data interpolation. arXiv preprint. arXiv: https://arxiv.org/abs/2608.28073
18. **Jiang, B., & Dai, Y.-H. (2015).** A framework of constraint preserving update schemes for optimization on Stiefel manifold. *Mathematical Programming*, 153(2), 535--575. arXiv: https://arxiv.org/abs/1301.0172 | DOI: https://doi.org/10.1007/s10107-014-0816-7
19. **Kume, K., & Yamada, I. (2024).** Global-support singular-value-free parameterization of the Stiefel manifold via generalized Cayley transforms. arXiv preprint. arXiv: https://arxiv.org/abs/2305.17901
20. **Li, J., Fuxin, L., & Todorovic, S. (2020).** Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. *International Conference on Learning Representations (ICLR 2020)*. arXiv: https://arxiv.org/abs/2002.01113
21. **Mataigne, S., Gallivan, K. A., & Absil, P.-A. (2026).** Acceleration-free retractions on the Stiefel manifold. arXiv preprint. arXiv: https://arxiv.org/abs/2602.19923
22. **Oviedo, H. (2022).** Global convergence of Riemannian line search methods with a Zhang--Hager-type condition. *Numerical Algorithms*, 91, 1183--1203. DOI: https://doi.org/10.1007/s11075-022-01298-6
23. **Qi, C., Gallivan, K. A., & Absil, P.-A. (2010).** Riemannian BFGS algorithm with applications. In *Recent Advances in Optimization and its Applications in Engineering* (pp. 183--192). Springer. DOI: https://doi.org/10.1007/978-3-642-12598-0_16
24. **Ring, W., & Wirth, B. (2012).** Optimization methods on Riemannian manifolds and their application to shape space. *SIAM Journal on Optimization*, 22(2), 596--627. DOI: https://doi.org/10.1137/11082885X
25. **Sato, H. (2022).** Riemannian conjugate gradient methods: General framework and specific algorithms with convergence analyses. *SIAM Journal on Optimization*, 32(4), 2690--2717. arXiv: https://arxiv.org/abs/2112.02572 | DOI: https://doi.org/10.1137/21M1464178
26. **Stoye, J., Mataigne, S., Absil, P.-A., & Zimmermann, R. (2025).** Shortest geodesic loops, sectional curvature, and injectivity radius of the Stiefel manifold. *BIT Numerical Mathematics* (accepted / preprint). arXiv: https://arxiv.org/abs/2511.01563
27. **Su, K., & Huang, W. (2026).** An inexact Riemannian gradient descent based on Newton--Schulz iteration on the Stiefel manifold. arXiv preprint. arXiv: https://arxiv.org/abs/2608.22774
28. **Wang, X., Deng, K., Peng, Z., & Yan, C. (2024).** New vector transport operators extending a Riemannian CG algorithm to generalized Stiefel manifold with low-rank applications. *Journal of Computational and Applied Mathematics*, 451, 116024. arXiv: https://arxiv.org/abs/2311.00907 | DOI: https://doi.org/10.1016/j.cam.2024.116024
29. **Wen, Z., & Yin, W. (2013).** A feasible method for optimization with orthogonality constraints. *Mathematical Programming*, 142(1--2), 397--428. DOI: https://doi.org/10.1007/s10107-012-0584-1
30. **Xiao, N., Liu, X., & Toh, K.-C. (2024).** Dissolving constraints for Riemannian optimization. *Mathematics of Operations Research*, 49(1), 366--397. arXiv: https://arxiv.org/abs/2203.10319 | DOI: https://doi.org/10.1287/moor.2023.1363
31. **Xiong, X., Gao, B., & Absil, P.-A. (2026).** A second-order method landing on the Stiefel manifold via Newton--Schulz iteration. arXiv preprint. arXiv: https://arxiv.org/abs/2605.02838
32. **Zhang, H., & Hager, W. W. (2004).** A nonmonotone line search technique and its application to unconstrained optimization. *SIAM Journal on Optimization*, 14(4), 1043--1056. DOI: https://doi.org/10.1137/S1052623403428208
33. **Zhu, X. (2017).** A Riemannian conjugate gradient method for optimization on the Stiefel manifold. *Computational Optimization and Applications*, 67(1), 73--110. DOI: https://doi.org/10.1007/s10589-016-9883-4
34. **Zhu, X., & Sato, H. (2020).** Riemannian conjugate gradient methods based on inverse retractions. *Computational Optimization and Applications*, 77(3), 779--810. DOI: https://doi.org/10.1007/s10589-020-00217-8
35. **Zimmermann, R., & Hüper, K. (2022).** Computing the Riemannian logarithm on the Stiefel manifold: Metrics, methods, and performance. *SIAM Journal on Matrix Analysis and Applications*, 43(2), 953--980. arXiv: https://arxiv.org/abs/2203.00389 | DOI: https://doi.org/10.1137/21M1425426
36. **Zimmermann, R., & Stoye, J. (2024).** High-precision bounds on the sectional curvature of the Stiefel manifold under the $\beta$-metrics. arXiv preprint. arXiv: https://arxiv.org/abs/2403.01879
37. **Zimmermann, R., & Stoye, J. (2025).** The injectivity radius of the compact Stiefel manifold under the Euclidean metric. *SIAM Journal on Matrix Analysis and Applications*, 46(1), 298--309. arXiv: https://arxiv.org/abs/2405.02268 | DOI: https://doi.org/10.1137/24M1663818
