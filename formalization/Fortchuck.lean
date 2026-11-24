import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Topology.Algebra.Ring.Real
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Module

/-!
# `Fortchuck`: Machine-Checked Algebraic & Variational Invariants (`v0.2.1`)

This module provides complete, self-contained, fully closed, zero-warning Lean 4
proofs of the foundational mathematical theorems underlying **`fortchuck`**
(`docs/ARCHITECTURE_SPEC.md` §2.2–§2.9 and §5.3):

1. **Theorem 1 (`skew_generator_properties`)**:
   For $X \in \mathrm{St}(n, p)$ ($X^\top X = I_p$) and Stiefel tangent $\xi \in T_X \mathrm{St}(n, p)$
   ($X^\top \xi + \xi^\top X = 0$), the modified tangent $\tilde{\xi} = \xi - \tfrac{1}{2} X (X^\top \xi)$
   and rank-$2p$ generator $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top$ satisfy
   $W_\xi^\top = -W_\xi$ and $W_\xi X = \xi$.
2. **Theorem 2 (`cayley_resolvent_isUnit`, `cayley_rotor_orthogonality_and_isometry`)**:
   For every skew-symmetric $W \in \mathfrak{so}(n)$ ($W^\top = -W$) and every $\tau \in \mathbb{R}$,
   $I_n - \tfrac{\tau}{2} W$ is unconditionally invertible (`IsUnit`), and the Cayley rotor
   $Q = (I_n - \tfrac{\tau}{2} W)^{-1}(I_n + \tfrac{\tau}{2} W)$ satisfies resolvent doubling
   $Q = 2M - I_n$, two-sided orthogonality $Q^\top Q = Q Q^\top = I_n$, Stiefel retraction
   preservation $(Q X)^\top (Q X) = X^\top X$, and exact isometric vector transport invariance of
   tangent skew-symmetry, skew projections, Frobenius cross-Gramians, and canonical Stiefel inner
   products.
3. **Theorem 3 (`schur_woodbury_isUnit`, `schur_woodbury_cayley_orthogonality`)**:
   For $X^\top X = I_p$, $A^\top = -A$, $X^\top K = 0$, and every $\tau \in \mathbb{R}$, the
   dissipative Schur complement $J(\tau) = I_p - \tfrac{\tau}{2} A + \tfrac{\tau^2}{4} K^\top K$
   is unconditionally invertible (`IsUnit`), and the Branch A Schur-Woodbury retraction
   $Y = (2X + \tau K) J(\tau)^{-1} - X$ satisfies $Y^\top Y = I_p$ identically.
4. **Theorem 4 (`cayley_branch_a_eq_branch_b`, `cayley_branch_a_transport_eq_branch_b`)**:
   For $X \in \mathrm{St}(n, p)$ and $\xi \in T_X \mathrm{St}(n, p)$, Branch A Schur-Woodbury
   retraction and rank-$2p$ single/batched-panel vector transport agree algebraically
   ($Q X = Y$ and $Q \zeta = \mathcal{T}_A(\zeta)$ for any $\zeta \in \mathbb{R}^{n \times q}$)
   with the Branch B direct $n \times n$ Cayley rotor, certifying exact canonical isometry of
   `fc_cayley_transport_branch_a` and `fc_cayley_transport_batch_branch_a`.
5. **Theorem 5 (`lbfgs_two_loop_symm_posDef_bound`, `coupled_two_metric_strict_descent`,
   `masked_lbfgs_coupled_strict_descent`)**:
   First proves inductively that the filtered two-loop L-BFGS recursion (`fc_lbfgs_apply_H`)
   preserves symmetry and strict positive-definiteness with an explicit closed-form eigenvalue
   lower bound whenever every retained secant pair satisfies $\langle s_j, y_j \rangle > 0$
   (`filter_free_subspace_pairs`). Then formalizes Theorem 2.1 of `ARCHITECTURE_SPEC.md`
   (unconditional strict descent $\phi'(0^+) < 0$ and exact first-order KKT equivalence
   $\phi'(0^+) = 0 \iff \mathrm{KKT}(z, X)$) and instantiates it directly with the masked product
   L-BFGS operator on $\mathcal{F}_\epsilon \times \prod_r T_{X_r}\mathrm{St}(n_r, p_r)$.
6. **Theorem 6 (`simplex_box_dual_root_exists`, `simplex_box_projection_variational`,
   `coupled_spg_stiefel_strict_descent`)**:
   Establishes the exact feasibility certificate $\mathcal{C} \neq \emptyset \iff \underline{b} \le b \le \overline{b}$
   (`fc_box_lin_validate`), unconditional dual root existence via continuity and IVT
   (`fc_box_lin_project`), primal feasibility, variational optimality, strong-convexity unique
   Euclidean projection, dual shift invariance $P_{\mathcal{C}}(y + \mu a) = P_{\mathcal{C}}(y)$,
   and unconditional coupled isotropic SPG $\times$ Stiefel ABB strict descent on
   $\mathcal{C} \times \prod_r \mathrm{St}(n_r, p_r)$.
7. **Theorem 7 (`rank1_kkt_nullspace_quasinewton_descent`)**:
   Establishes well-posedness ($a^\top H a > 0$), homogeneous and inhomogeneous hyperplane
   tangency ($a^\top d = 0$ and $a^\top d_{\mathrm{inhom}} = \delta_b$), multiplier shift
   invariance ($g \mapsto g + \mu a$), symmetric rank-1 operator form, unconditional homogeneous
   descent ($g^\top d = -v^\top H v \le 0$, strictly $< 0$ whenever $g \notin \mathrm{span}(a)$),
   and the exact inhomogeneous directional-derivative identity
   $g^\top d_{\mathrm{inhom}} = -v^\top H v + \tfrac{\delta_b}{a^\top H a}(g^\top H a)$ showing
   precisely why `fc_solve_core` pairs inhomogeneous boundary snapping with a runtime descent
   guard and fallback to Theorem 6 coupled SPG.
-/

namespace Fortchuck

open scoped Matrix BigOperators Classical
open Finset

noncomputable section

-- ============================================================================
-- Section 1: Rank-2p Stiefel Skew-Symmetric Lie Generator (Theorem 1)
-- ============================================================================

/-- Modified Stiefel tangent matrix $\tilde{\xi} = \xi - \frac{1}{2} X (X^\top \xi)$. -/
def stiefelModifiedTangent {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ) : Matrix (Fin n) (Fin p) ℝ :=
  ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ))

/-- Canonical rank-$2p$ skew-symmetric generator $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top$. -/
def skewGenerator {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  let ξ_tilde := stiefelModifiedTangent X ξ
  ξ_tilde * Xᵀ - X * ξ_tildeᵀ

/-- **Theorem 1 (`skew_generator_properties`)**:
For orthonormal $X \in \mathbb{R}^{n \times p}$ ($X^\top X = I_p$) and Stiefel tangent
$\xi \in T_X \mathrm{St}(n, p)$ ($X^\top \xi + \xi^\top X = 0$), the canonical rank-$2p$
generator $W_\xi = \mathrm{skewGenerator}(X, \xi)$ guarantees both skew-symmetry
$W_\xi^\top = -W_\xi$ and exact tangent reproduction $W_\xi X = \xi$. -/
theorem skew_generator_properties {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0) :
    (skewGenerator X ξ)ᵀ = -(skewGenerator X ξ) ∧ (skewGenerator X ξ) * X = ξ := by
  unfold skewGenerator stiefelModifiedTangent
  dsimp only
  have hξ_swap : ξᵀ * X = -(Xᵀ * ξ) := eq_neg_of_add_eq_zero_right hξ
  constructor
  · simp only [Matrix.transpose_sub, Matrix.transpose_mul, Matrix.transpose_transpose]
    abel
  · simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.transpose_sub, Matrix.transpose_smul, Matrix.transpose_mul,
      Matrix.transpose_transpose, Matrix.mul_assoc, hX, hξ_swap,
      Matrix.mul_one, Matrix.neg_mul, Matrix.mul_neg]
    module

-- ============================================================================
-- Section 2: Unconditional Resolvent Invertibility & Cayley Rotor (Theorem 2)
-- ============================================================================

/-- Squared Euclidean norm `dotProduct v v` is nonnegative on `Fin n → ℝ`. -/
lemma dotProduct_self_nonneg {n : ℕ} (v : Fin n → ℝ) : 0 ≤ dotProduct v v :=
  Finset.sum_nonneg fun i _ => mul_self_nonneg (v i)

/-- Squared Euclidean norm `dotProduct v v` vanishes if and only if `v = 0`. -/
lemma dotProduct_self_eq_zero_iff {n : ℕ} (v : Fin n → ℝ) : dotProduct v v = 0 ↔ v = 0 := by
  constructor
  · intro h
    have hi := (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => mul_self_nonneg (v i))).mp h
    ext i
    exact mul_self_eq_zero.mp (hi i (Finset.mem_univ i))
  · rintro rfl
    simp

/-- Strictly positive Euclidean norm `0 < dotProduct v v` for every nonzero `v : Fin n → ℝ`. -/
lemma dotProduct_self_pos {n : ℕ} {v : Fin n → ℝ} (hv : v ≠ 0) : 0 < dotProduct v v :=
  lt_of_le_of_ne (dotProduct_self_nonneg v) (Ne.symm (mt (dotProduct_self_eq_zero_iff v).mp hv))

/-- Any skew-symmetric matrix $W^\top = -W$ has vanishing quadratic form $v^\top W v = 0$. -/
lemma dotProduct_skew_mulVec {n : ℕ}
    (W : Matrix (Fin n) (Fin n) ℝ) (hW : Wᵀ = -W) (v : Fin n → ℝ) :
    dotProduct v (W.mulVec v) = 0 := by
  have h : dotProduct v (W.mulVec v) = -dotProduct v (W.mulVec v) := by
    calc
      dotProduct v (W.mulVec v) = dotProduct (W.vecMul v) v := Matrix.dotProduct_mulVec v W v
      _ = dotProduct (Wᵀ.mulVec v) v := by rw [Matrix.mulVec_transpose]
      _ = dotProduct ((-W).mulVec v) v := by rw [hW]
      _ = -dotProduct (W.mulVec v) v := by rw [Matrix.neg_mulVec, neg_dotProduct]
      _ = -dotProduct v (W.mulVec v) := by rw [dotProduct_comm]
  linarith

/-- Any real square matrix whose quadratic form is strictly positive on nonzero vectors is a unit. -/
lemma isUnit_of_posDef_mulVec {n : ℕ}
    (A : Matrix (Fin n) (Fin n) ℝ)
    (hpos : ∀ v : Fin n → ℝ, v ≠ 0 → 0 < dotProduct v (A.mulVec v)) :
    IsUnit A := by
  rw [← Matrix.mulVec_injective_iff_isUnit]
  intro x y hxy
  by_contra hne
  have hv_ne : x - y ≠ 0 := sub_ne_zero.mpr hne
  have hAv : A.mulVec (x - y) = 0 := by
    rw [Matrix.mulVec_sub, hxy, sub_self]
  have h0 : dotProduct (x - y) (A.mulVec (x - y)) = 0 := by
    rw [hAv, dotProduct_zero]
  exact absurd h0 (ne_of_gt (hpos (x - y) hv_ne))

/-- Unconditional Cayley resolvent inverse $M(\tau) = (I_n - \tfrac{\tau}{2} W)^{-1}$. -/
def cayleyResolvent {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  ((1 : Matrix (Fin n) (Fin n) ℝ) - (τ / 2 : ℝ) • W)⁻¹

/-- Direct $n \times n$ Cayley rotor $Q(\tau) = (I_n - \tfrac{\tau}{2} W)^{-1} (I_n + \tfrac{\tau}{2} W)$. -/
def cayleyRotor {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  cayleyResolvent W τ * (1 + (τ / 2 : ℝ) • W)

/-- Parametric Stiefel Riemannian metric ($\beta = 1/2$ canonical, $\beta = 1$ Euclidean):
$\langle \xi, \eta \rangle_{X, \beta} = \mathrm{tr}\!\left(\xi^\top (I_n - (1 - \beta) X X^\top) \eta\right)$. -/
def stiefelInner {n p : ℕ} (β : ℝ)
    (X ξ η : Matrix (Fin n) (Fin p) ℝ) : ℝ :=
  Matrix.trace (ξᵀ * ((1 : Matrix (Fin n) (Fin n) ℝ) - (1 - β) • (X * Xᵀ)) * η)

/-- **Unconditional Cayley Resolvent Invertibility (`cayley_resolvent_isUnit`)**:
For every skew-symmetric $W \in \mathfrak{so}(n)$ ($W^\top = -W$) and every $\tau \in \mathbb{R}$,
the quadratic form satisfies $v^\top (I_n - \tfrac{\tau}{2} W) v = \|v\|_2^2 > 0$ for all $v \neq 0$,
so $I_n - \tfrac{\tau}{2} W$ is unconditionally invertible. -/
theorem cayley_resolvent_isUnit {n : ℕ}
    (W : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ) (hW : Wᵀ = -W) :
    IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) - (τ / 2 : ℝ) • W) := by
  refine isUnit_of_posDef_mulVec _ fun v hv => ?_
  have hquad :
      dotProduct v (((1 : Matrix (Fin n) (Fin n) ℝ) - (τ / 2 : ℝ) • W).mulVec v) =
        dotProduct v v := by
    rw [Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec, dotProduct_sub,
      dotProduct_smul, smul_eq_mul, dotProduct_skew_mulVec W hW v, mul_zero, sub_zero]
  rw [hquad]
  exact dotProduct_self_pos hv

/-- Left multiplication by any column-orthonormal matrix $Q$ ($Q^\top Q = I_n$) preserves all
cross-Gramian products $(Q A)^\top (Q B) = A^\top B$. -/
lemma orthogonal_gram_invariance {n p q : ℕ}
    (Q : Matrix (Fin n) (Fin n) ℝ) (hQtQ : Qᵀ * Q = 1)
    (A : Matrix (Fin n) (Fin p) ℝ) (B : Matrix (Fin n) (Fin q) ℝ) :
    (Q * A)ᵀ * (Q * B) = Aᵀ * B := by
  calc
    (Q * A)ᵀ * (Q * B) = (Aᵀ * Qᵀ) * (Q * B) := by rw [Matrix.transpose_mul]
    _ = Aᵀ * ((Qᵀ * Q) * B) := by simp only [Matrix.mul_assoc]
    _ = Aᵀ * B := by rw [hQtQ, Matrix.one_mul]

/-- **Theorem 2 (`cayley_rotor_orthogonality_and_isometry`)**:
For any skew-symmetric $W \in \mathfrak{so}(n)$ ($W^\top = -W$) and any $\tau \in \mathbb{R}$,
setting $M = \mathrm{cayleyResolvent}(W, \tau) = (I_n - \tfrac{\tau}{2} W)^{-1}$ and
$Q = \mathrm{cayleyRotor}(W, \tau) = M(I_n + \tfrac{\tau}{2} W)$ unconditionally guarantees:
- **(a)** Two-sided invertibility of $I_n - \tfrac{\tau}{2} W$ and resolvent doubling $Q = 2M - I_n$,
- **(b)** Two-sided orthogonality $Q^\top Q = I_n$ and $Q Q^\top = I_n$,
- **(c)** Stiefel retraction preservation $(Q X)^\top (Q X) = X^\top X$,
- **(d)** Isometric transport invariance of tangent skew-symmetry, skew projections,
  Frobenius cross-Gramians, and canonical Stiefel metric inner products `stiefelInner (1 / 2)`. -/
theorem cayley_rotor_orthogonality_and_isometry {n p : ℕ}
    (W : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ)
    (X ζ ζ₁ ζ₂ : Matrix (Fin n) (Fin p) ℝ)
    (hW : Wᵀ = -W) :
    let M := cayleyResolvent W τ
    let Q := cayleyRotor W τ
    (IsUnit ((1 : Matrix (Fin n) (Fin n) ℝ) - (τ / 2 : ℝ) • W) ∧
      M * (1 - (τ / 2 : ℝ) • W) = 1 ∧ (1 - (τ / 2 : ℝ) • W) * M = 1) ∧
    Q = (2 : ℝ) • M - 1 ∧
    (Qᵀ * Q = 1 ∧ Q * Qᵀ = 1) ∧
    (Q * X)ᵀ * (Q * X) = Xᵀ * X ∧
    (((Q * X)ᵀ * (Q * ζ) + (Q * ζ)ᵀ * (Q * X) = Xᵀ * ζ + ζᵀ * X) ∧
     ((Q * X)ᵀ * (Q * ζ) = Xᵀ * ζ ∧ (Q * ζ₁)ᵀ * (Q * ζ₂) = ζ₁ᵀ * ζ₂) ∧
     ((Q * ζ₁)ᵀ * (1 - (1 / 2 : ℝ) • ((Q * X) * (Q * X)ᵀ)) * (Q * ζ₂) =
        ζ₁ᵀ * (1 - (1 / 2 : ℝ) • (X * Xᵀ)) * ζ₂) ∧
     (stiefelInner (1 / 2) (Q * X) (Q * ζ₁) (Q * ζ₂) = stiefelInner (1 / 2) X ζ₁ ζ₂)) := by
  dsimp only
  set c : ℝ := τ / 2
  set N : Matrix (Fin n) (Fin n) ℝ := 1 - c • W with hN_def
  set M := cayleyResolvent W τ with hM_def
  set Q := cayleyRotor W τ with hQ_def
  have hN_unit : IsUnit N := cayley_resolvent_isUnit W τ hW
  have hN_det : IsUnit N.det := (Matrix.isUnit_iff_isUnit_det N).mp hN_unit
  have hM_right : M * N = 1 := Matrix.nonsing_inv_mul N hN_det
  have hM_left : N * M = 1 := Matrix.mul_nonsing_inv N hN_det
  have h_split : (1 : Matrix (Fin n) (Fin n) ℝ) + c • W = (2 : ℝ) • 1 - N := by
    rw [hN_def]; module
  have hQ_res : Q = (2 : ℝ) • M - 1 := by
    calc
      Q = M * ((2 : ℝ) • 1 - N) := by rw [hQ_def, cayleyRotor, ← hM_def, h_split]
      _ = (2 : ℝ) • M - 1 := by
        rw [Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_one, hM_right]
  have hQ_comm : Q = (1 + c • W) * M := by
    calc
      Q = (2 : ℝ) • M - 1 := hQ_res
      _ = ((2 : ℝ) • 1 - N) * M := by
        rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul, hM_left]
      _ = (1 + c • W) * M := by rw [← h_split]
  have hN_sub_tr : Nᵀ = 1 + c • W := by
    rw [hN_def, Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_smul, hW,
      smul_neg, sub_neg_eq_add]
  have hN_add_tr : ((1 : Matrix (Fin n) (Fin n) ℝ) + c • W)ᵀ = N := by
    rw [hN_def, Matrix.transpose_add, Matrix.transpose_one, Matrix.transpose_smul, hW,
      smul_neg, ← sub_eq_add_neg]
  have hMt_right : Mᵀ * (1 + c • W) = 1 := by
    have h := congrArg Matrix.transpose hM_left
    rwa [Matrix.transpose_mul, hN_sub_tr, Matrix.transpose_one] at h
  have hMt_left : (1 + c • W) * Mᵀ = 1 := by
    have h := congrArg Matrix.transpose hM_right
    rwa [Matrix.transpose_mul, hN_sub_tr, Matrix.transpose_one] at h
  have hQtQ : Qᵀ * Q = 1 := by
    calc
      Qᵀ * Q = ((1 + c • W) * M)ᵀ * (M * (1 + c • W)) := by
        rw [← hQ_comm, hQ_def, cayleyRotor, ← hM_def]
      _ = (Mᵀ * N) * (M * (1 + c • W)) := by rw [Matrix.transpose_mul, hN_add_tr]
      _ = Mᵀ * ((N * M) * (1 + c • W)) := by simp only [Matrix.mul_assoc]
      _ = Mᵀ * (1 + c • W) := by rw [hM_left, Matrix.one_mul]
      _ = 1 := hMt_right
  have hQQt : Q * Qᵀ = 1 := by
    calc
      Q * Qᵀ = ((1 + c • W) * M) * (M * (1 + c • W))ᵀ := by
        rw [← hQ_comm, hQ_def, cayleyRotor, ← hM_def]
      _ = ((1 + c • W) * M) * (N * Mᵀ) := by rw [Matrix.transpose_mul, hN_add_tr]
      _ = (1 + c • W) * ((M * N) * Mᵀ) := by simp only [Matrix.mul_assoc]
      _ = (1 + c • W) * Mᵀ := by rw [hM_right, Matrix.one_mul]
      _ = 1 := hMt_left
  have hXX := orthogonal_gram_invariance Q hQtQ X X
  have hXζ := orthogonal_gram_invariance Q hQtQ X ζ
  have hζX := orthogonal_gram_invariance Q hQtQ ζ X
  have hζ₁ζ₂ := orthogonal_gram_invariance Q hQtQ ζ₁ ζ₂
  have hζ₁X := orthogonal_gram_invariance Q hQtQ ζ₁ X
  have hXζ₂ := orthogonal_gram_invariance Q hQtQ X ζ₂
  have h_can_mat :
      (Q * ζ₁)ᵀ * (1 - (1 / 2 : ℝ) • ((Q * X) * (Q * X)ᵀ)) * (Q * ζ₂) =
        ζ₁ᵀ * (1 - (1 / 2 : ℝ) • (X * Xᵀ)) * ζ₂ := by
    calc
      (Q * ζ₁)ᵀ * (1 - (1 / 2 : ℝ) • ((Q * X) * (Q * X)ᵀ)) * (Q * ζ₂)
          = (Q * ζ₁)ᵀ * (Q * ζ₂) - (1 / 2 : ℝ) • (((Q * ζ₁)ᵀ * (Q * X)) * ((Q * X)ᵀ * (Q * ζ₂))) := by
            simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one,
              Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]
      _ = ζ₁ᵀ * ζ₂ - (1 / 2 : ℝ) • ((ζ₁ᵀ * X) * (Xᵀ * ζ₂)) := by
            rw [hζ₁ζ₂, hζ₁X, hXζ₂]
      _ = ζ₁ᵀ * (1 - (1 / 2 : ℝ) • (X * Xᵀ)) * ζ₂ := by
            simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one,
              Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]
  have h_inner : stiefelInner (1 / 2) (Q * X) (Q * ζ₁) (Q * ζ₂) = stiefelInner (1 / 2) X ζ₁ ζ₂ := by
    unfold stiefelInner
    have h_half : (1 : ℝ) - (1 / 2 : ℝ) = (1 / 2 : ℝ) := by norm_num
    rw [h_half]
    exact congrArg Matrix.trace h_can_mat
  exact ⟨⟨hN_unit, hM_right, hM_left⟩, hQ_res, ⟨hQtQ, hQQt⟩, hXX,
    ⟨by rw [hXζ, hζX], ⟨hXζ, hζ₁ζ₂⟩, h_can_mat, h_inner⟩⟩

-- ============================================================================
-- Section 3: Branch A Schur-Woodbury Invertibility, Retraction & Transport (Theorems 3 & 4)
-- ============================================================================

/-- $p \times p$ strictly dissipative Schur-Woodbury matrix
$J(\tau) = I_p - \frac{\tau}{2} A + \frac{\tau^2}{4} K^\top K$. -/
def schurWoodburyMatrix {n p : ℕ}
    (A : Matrix (Fin p) (Fin p) ℝ) (K : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ) :
    Matrix (Fin p) (Fin p) ℝ :=
  1 - (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K)

/-- Branch A rank-$2p$ Schur-Woodbury Stiefel retraction
$Y(\tau) = (2X + \tau K) J(\tau)^{-1} - X$. -/
def schurWoodburyRetract {n p : ℕ}
    (X K : Matrix (Fin n) (Fin p) ℝ) (A : Matrix (Fin p) (Fin p) ℝ) (τ : ℝ) :
    Matrix (Fin n) (Fin p) ℝ :=
  ((2 : ℝ) • X + τ • K) * (schurWoodburyMatrix A K τ)⁻¹ - X

/-- Branch A rank-$2p$ Schur-Woodbury single-vector / batched-panel vector transport
matching `transport_branch_a_arrays` (`q = p`) and `transport_batch_branch_a_arrays` (`q = p * K`)
in `src/fortchuck_stiefel.f90`. -/
def schurWoodburyTransport {n p q : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ) (ζ : Matrix (Fin n) (Fin q) ℝ) :
    Matrix (Fin n) (Fin q) ℝ :=
  let A := Xᵀ * ξ
  let K := ξ - X * A
  let ξ_tilde := stiefelModifiedTangent X ξ
  let C := (1 : Matrix (Fin p) (Fin p) ℝ) - (τ / 4 : ℝ) • A
  let R1 := Xᵀ * ζ
  let R2 := -(ξ_tildeᵀ * ζ)
  let Z1 := (schurWoodburyMatrix A K τ)⁻¹ * (C * R1 + (τ / 2 : ℝ) • R2)
  ζ + τ • (ξ_tilde * Z1) + (2 : ℝ) • (X * (C * Z1 - R1))

/-- **Unconditional Schur-Woodbury Invertibility (`schur_woodbury_isUnit`)**:
For every skew-symmetric $A \in \mathfrak{so}(p)$ ($A^\top = -A$), any $K \in \mathbb{R}^{n \times p}$,
and every $\tau \in \mathbb{R}$, $v^\top J(\tau) v = \|v\|_2^2 + \tfrac{\tau^2}{4}\|K v\|_2^2 > 0$
for all $v \neq 0$, so $J(\tau) = \mathrm{schurWoodburyMatrix}(A, K, \tau)$ is unconditionally invertible. -/
theorem schur_woodbury_isUnit {n p : ℕ}
    (A : Matrix (Fin p) (Fin p) ℝ) (K : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ)
    (hA : Aᵀ = -A) :
    IsUnit (schurWoodburyMatrix A K τ) := by
  refine isUnit_of_posDef_mulVec _ fun v hv => ?_
  have hKtK : dotProduct v ((Kᵀ * K).mulVec v) = dotProduct (K.mulVec v) (K.mulVec v) := by
    rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose]
  have hquad :
      dotProduct v ((schurWoodburyMatrix A K τ).mulVec v) =
        dotProduct v v + (τ ^ 2 / 4 : ℝ) * dotProduct (K.mulVec v) (K.mulVec v) := by
    unfold schurWoodburyMatrix
    rw [Matrix.add_mulVec, Matrix.sub_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec,
      Matrix.smul_mulVec, dotProduct_add, dotProduct_sub, dotProduct_smul, dotProduct_smul,
      smul_eq_mul, smul_eq_mul, dotProduct_skew_mulVec A hA v, hKtK, mul_zero, sub_zero]
  rw [hquad]
  have hv_pos : 0 < dotProduct v v := dotProduct_self_pos hv
  have hKv_nonneg : 0 ≤ (τ ^ 2 / 4 : ℝ) * dotProduct (K.mulVec v) (K.mulVec v) :=
    mul_nonneg (by positivity) (dotProduct_self_nonneg (K.mulVec v))
  linarith

/-- **Theorem 3 (`schur_woodbury_cayley_orthogonality`)**:
Given $X^\top X = I_p$, skew-symmetric $A^\top = -A$, and normal block $X^\top K = 0$, the
Schur complement $J(\tau) = \mathrm{schurWoodburyMatrix}(A, K, \tau)$ is unconditionally invertible
for every $\tau \in \mathbb{R}$, and $Y = \mathrm{schurWoodburyRetract}(X, K, A, \tau)$ satisfies
$Y^\top Y = I_p$ identically. -/
theorem schur_woodbury_cayley_orthogonality {n p : ℕ}
    (X K : Matrix (Fin n) (Fin p) ℝ)
    (A : Matrix (Fin p) (Fin p) ℝ) (τ : ℝ)
    (hX : Xᵀ * X = 1)
    (hA : Aᵀ = -A)
    (hXK : Xᵀ * K = 0) :
    IsUnit (schurWoodburyMatrix A K τ) ∧
    (schurWoodburyRetract X K A τ)ᵀ * (schurWoodburyRetract X K A τ) = 1 := by
  set J := schurWoodburyMatrix A K τ with hJ_def
  set J_inv := J⁻¹ with hJinv_def
  have hJ_unit : IsUnit J := schur_woodbury_isUnit A K τ hA
  have hJ_det : IsUnit J.det := (Matrix.isUnit_iff_isUnit_det J).mp hJ_unit
  have hJ_right : J * J_inv = 1 := Matrix.mul_nonsing_inv J hJ_det
  refine ⟨hJ_unit, ?_⟩
  unfold schurWoodburyRetract
  rw [← hJ_def, ← hJinv_def]
  set Z := (2 : ℝ) • X + τ • K
  set Y := Z * J_inv - X
  have hKX : Kᵀ * X = 0 := by
    have h := congrArg Matrix.transpose hXK
    rwa [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_zero] at h
  have hJt : Jᵀ = 1 + (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K) := by
    rw [hJ_def, schurWoodburyMatrix, Matrix.transpose_add, Matrix.transpose_sub,
      Matrix.transpose_one, Matrix.transpose_smul, Matrix.transpose_smul, hA,
      Matrix.transpose_mul, Matrix.transpose_transpose, smul_neg, sub_neg_eq_add]
  have h_sum : (2 : ℝ) • (Jᵀ + J) = (4 : ℝ) • 1 + (τ ^ 2 : ℝ) • (Kᵀ * K) := by
    rw [hJt, hJ_def, schurWoodburyMatrix]
    module
  have hZtZ : Zᵀ * Z = (2 : ℝ) • (Jᵀ + J) := by
    have h_exp : Zᵀ * Z = (4 : ℝ) • (1 : Matrix (Fin p) (Fin p) ℝ) + (τ ^ 2 : ℝ) • (Kᵀ * K) := by
      simp only [Z, Matrix.transpose_add, Matrix.transpose_smul, Matrix.add_mul, Matrix.mul_add,
        Matrix.smul_mul, Matrix.mul_smul, hX, hXK, hKX, smul_zero, add_zero, zero_add]
      ext i j
      simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      ring
    rw [h_exp, ← h_sum]
  have hXtZ : Xᵀ * Z = (2 : ℝ) • 1 := by
    simp only [Z, Matrix.mul_add, Matrix.mul_smul, hX, hXK, smul_zero, add_zero]
  have hZtX : Zᵀ * X = (2 : ℝ) • 1 := by
    simp only [Z, Matrix.transpose_add, Matrix.transpose_smul, Matrix.add_mul,
      Matrix.smul_mul, hX, hKX, smul_zero, add_zero]
  have hJinv_tr_J : J_invᵀ * Jᵀ = 1 := by
    have h := congrArg Matrix.transpose hJ_right
    rwa [Matrix.transpose_mul, Matrix.transpose_one] at h
  have h_mid : J_invᵀ * (Zᵀ * Z) * J_inv = (2 : ℝ) • J_inv + (2 : ℝ) • J_invᵀ := by
    rw [hZtZ, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_add, Matrix.add_mul,
      hJinv_tr_J, Matrix.one_mul,
      Matrix.mul_assoc J_invᵀ J J_inv, hJ_right, Matrix.mul_one, smul_add]
  calc
    Yᵀ * Y = (J_invᵀ * Zᵀ - Xᵀ) * (Z * J_inv - X) := by
      simp only [Y, Matrix.transpose_sub, Matrix.transpose_mul]
    _ = J_invᵀ * (Zᵀ * Z) * J_inv - J_invᵀ * (Zᵀ * X) - (Xᵀ * Z) * J_inv + Xᵀ * X := by
      simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_assoc]
      abel
    _ = ((2 : ℝ) • J_inv + (2 : ℝ) • J_invᵀ) - (2 : ℝ) • J_invᵀ - (2 : ℝ) • J_inv + 1 := by
      rw [h_mid, hZtX, hXtZ, hX, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, Matrix.one_mul]
    _ = 1 := by module

/-- **Theorem 4a (`cayley_branch_a_eq_branch_b`)**:
For $X \in \mathrm{St}(n, p)$ ($X^\top X = I_p$), Stiefel tangent $\xi \in T_X \mathrm{St}(n, p)$
($X^\top \xi + \xi^\top X = 0$), and any $\tau \in \mathbb{R}$, the Branch B direct Cayley
retraction $\mathrm{cayleyRotor}(W_\xi, \tau) X$ and Branch A Schur-Woodbury retraction
$\mathrm{schurWoodburyRetract}(X, \xi - X(X^\top \xi), X^\top \xi, \tau)$ are algebraically
identical. -/
theorem cayley_branch_a_eq_branch_b {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0) :
    cayleyRotor (skewGenerator X ξ) τ * X =
      schurWoodburyRetract X (ξ - X * (Xᵀ * ξ)) (Xᵀ * ξ) τ := by
  set A := Xᵀ * ξ with hA_def
  set K := ξ - X * A with hK_def
  set ξ_tilde := stiefelModifiedTangent X ξ with hξt_def
  set W_ξ := skewGenerator X ξ with hW_def
  set M := cayleyResolvent W_ξ τ
  set Q := cayleyRotor W_ξ τ
  set J := schurWoodburyMatrix A K τ with hJ_def
  have hA_skew : Aᵀ = -A := by
    have hξ_swap : ξᵀ * X = -A := eq_neg_of_add_eq_zero_right hξ
    rw [hA_def, Matrix.transpose_mul, Matrix.transpose_transpose, hξ_swap]
  have hW_props := skew_generator_properties X ξ hX hξ
  have hWt : W_ξᵀ = -W_ξ := hW_props.1
  have hWX : W_ξ * X = ξ := hW_props.2
  have h2 := cayley_rotor_orthogonality_and_isometry W_ξ τ X ξ ξ ξ hWt
  have hM_right : M * (1 - (τ / 2 : ℝ) • W_ξ) = 1 := h2.1.2.1
  have hQ_res : Q = (2 : ℝ) • M - 1 := h2.2.1
  have hJ_det : IsUnit J.det :=
    (Matrix.isUnit_iff_isUnit_det J).mp (schur_woodbury_isUnit A K τ hA_skew)
  have hJ_right : J * J⁻¹ = 1 := Matrix.mul_nonsing_inv J hJ_det
  have hXK : Xᵀ * K = 0 := by
    simp only [K, A, Matrix.mul_sub, ← Matrix.mul_assoc Xᵀ X (Xᵀ * ξ), hX, Matrix.one_mul, sub_self]
  have hξ_tilde_K : ξ_tilde = K + (1 / 2 : ℝ) • (X * A) := by
    rw [hξt_def, stiefelModifiedTangent, hK_def, hA_def]
    module
  have hWK : W_ξ * K = -(X * (Kᵀ * K)) := by
    calc
      W_ξ * K = ξ_tilde * (Xᵀ * K) - X * (ξ_tildeᵀ * K) := by
        rw [hW_def, skewGenerator, ← hξt_def, Matrix.sub_mul, Matrix.mul_assoc, Matrix.mul_assoc]
      _ = -(X * (ξ_tildeᵀ * K)) := by rw [hXK, Matrix.mul_zero, zero_sub]
      _ = -(X * (Kᵀ * K)) := by
        congr 2
        rw [hξ_tilde_K, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul,
          Matrix.add_mul, Matrix.smul_mul, Matrix.mul_assoc, hXK, Matrix.mul_zero,
          smul_zero, add_zero]
  have h_intertwine : (1 - (τ / 2 : ℝ) • W_ξ) * (X + (τ / 2 : ℝ) • K) = X * J := by
    have hξ_split : ξ = X * A + K := by simp only [K]; abel
    rw [hJ_def, schurWoodburyMatrix]
    simp only [Matrix.sub_mul, Matrix.mul_add, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one,
      Matrix.smul_mul, Matrix.mul_smul, hWX, hWK, hξ_split, smul_neg, smul_add]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have hMX : M * X = (X + (τ / 2 : ℝ) • K) * J⁻¹ := by
    calc
      M * X = M * (X * (J * J⁻¹)) := by rw [hJ_right, Matrix.mul_one]
      _ = M * ((X * J) * J⁻¹) := by rw [Matrix.mul_assoc X J J⁻¹]
      _ = M * (((1 - (τ / 2 : ℝ) • W_ξ) * (X + (τ / 2 : ℝ) • K)) * J⁻¹) := by
        rw [← h_intertwine]
      _ = ((M * (1 - (τ / 2 : ℝ) • W_ξ)) * (X + (τ / 2 : ℝ) • K)) * J⁻¹ := by
        simp only [Matrix.mul_assoc]
      _ = (X + (τ / 2 : ℝ) • K) * J⁻¹ := by rw [hM_right, Matrix.one_mul]
  have h2XK : (2 : ℝ) • (X + (τ / 2 : ℝ) • K) = (2 : ℝ) • X + τ • K := by
    ext i j
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  unfold schurWoodburyRetract
  rw [← hJ_def]
  calc
    Q * X = ((2 : ℝ) • M - 1) * X := by rw [hQ_res]
    _ = (2 : ℝ) • (M * X) - X := by rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul]
    _ = (2 : ℝ) • ((X + (τ / 2 : ℝ) • K) * J⁻¹) - X := by rw [hMX]
    _ = ((2 : ℝ) • (X + (τ / 2 : ℝ) • K)) * J⁻¹ - X := by rw [Matrix.smul_mul]
    _ = ((2 : ℝ) • X + τ • K) * J⁻¹ - X := by rw [h2XK]

/-- **Theorem 4b (`cayley_branch_a_transport_eq_branch_b`)**:
For $X \in \mathrm{St}(n, p)$, Stiefel tangent $\xi \in T_X \mathrm{St}(n, p)$, any $\tau \in \mathbb{R}$,
and any matrix/panel $\zeta \in \mathbb{R}^{n \times q}$ (covering both single-vector transport
$q = p$ and batched Level-3 panel transport $q = p K_{\mathrm{batch}}$ in `fortchuck_stiefel.f90`):
1. Branch A rank-$2p$ Schur-Woodbury transport equals Branch B direct Cayley transport identically:
   `cayleyRotor (skewGenerator X ξ) τ * ζ = schurWoodburyTransport X ξ τ ζ`; and
2. Consequently, Branch A transport preserves Stiefel tangent skew-symmetry, cached skew
   projections $Y^\top \mathcal{T}_A(\zeta) = X^\top \zeta$, and canonical Stiefel inner products
   `stiefelInner (1 / 2)` identically. -/
theorem cayley_branch_a_transport_eq_branch_b {n p q : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ)
    (ζ : Matrix (Fin n) (Fin q) ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0) :
    cayleyRotor (skewGenerator X ξ) τ * ζ = schurWoodburyTransport X ξ τ ζ := by
  set A := Xᵀ * ξ with hA_def
  set K := ξ - X * A with hK_def
  set ξ_tilde := stiefelModifiedTangent X ξ with hξt_def
  set W_ξ := skewGenerator X ξ with hW_def
  set M := cayleyResolvent W_ξ τ with hM_def
  set J := schurWoodburyMatrix A K τ with hJ_def
  set C := (1 : Matrix (Fin p) (Fin p) ℝ) - (τ / 4 : ℝ) • A with hC_def
  set R1 := Xᵀ * ζ with hR1_def
  set R2 := -(ξ_tildeᵀ * ζ) with hR2_def
  set Z1 := J⁻¹ * (C * R1 + (τ / 2 : ℝ) • R2) with hZ1_def
  set V := ζ + τ • (ξ_tilde * Z1) + (2 : ℝ) • (X * (C * Z1 - R1)) with hV_def
  have hA_skew : Aᵀ = -A := by
    have hξ_swap : ξᵀ * X = -A := eq_neg_of_add_eq_zero_right hξ
    rw [hA_def, Matrix.transpose_mul, Matrix.transpose_transpose, hξ_swap]
  have hWt : W_ξᵀ = -W_ξ := (skew_generator_properties X ξ hX hξ).1
  have hM_right : M * (1 - (τ / 2 : ℝ) • W_ξ) = 1 :=
    (cayley_rotor_orthogonality_and_isometry W_ξ τ X ξ ξ ξ hWt).1.2.1
  have hJ_det : IsUnit J.det :=
    (Matrix.isUnit_iff_isUnit_det J).mp (schur_woodbury_isUnit A K τ hA_skew)
  have hJ_right : J * J⁻¹ = 1 := Matrix.mul_nonsing_inv J hJ_det
  have hJZ1 : J * Z1 = C * R1 + (τ / 2 : ℝ) • R2 := by
    rw [hZ1_def, ← Matrix.mul_assoc, hJ_right, Matrix.one_mul]
  have hXK : Xᵀ * K = 0 := by
    simp only [K, A, Matrix.mul_sub, ← Matrix.mul_assoc Xᵀ X (Xᵀ * ξ), hX, Matrix.one_mul, sub_self]
  have hKX : Kᵀ * X = 0 := by
    have h := congrArg Matrix.transpose hXK
    rwa [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_zero] at h
  have hξ_tilde_K : ξ_tilde = K + (1 / 2 : ℝ) • (X * A) := by
    rw [hξt_def, stiefelModifiedTangent, hK_def, hA_def]
    module
  have hXt_ξt : Xᵀ * ξ_tilde = (1 / 2 : ℝ) • A := by
    rw [hξ_tilde_K, Matrix.mul_add, Matrix.mul_smul, hXK, zero_add,
      ← Matrix.mul_assoc, hX, Matrix.one_mul]
  have hξt_X : ξ_tildeᵀ * X = -(1 / 2 : ℝ) • A := by
    have h := congrArg Matrix.transpose hXt_ξt
    rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_smul, hA_skew] at h
    simpa [smul_neg, neg_smul] using h
  have hξt_ξt : ξ_tildeᵀ * ξ_tilde = Kᵀ * K - (1 / 4 : ℝ) • (A * A) := by
    rw [hξ_tilde_K, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul, hA_skew]
    simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul, Matrix.mul_assoc,
      hXK, Matrix.mul_zero, smul_zero, add_zero]
    have hKX_term : Kᵀ * (X * A) = 0 := by rw [← Matrix.mul_assoc, hKX, Matrix.zero_mul]
    have hXtX_term : Xᵀ * (X * A) = A := by rw [← Matrix.mul_assoc, hX, Matrix.one_mul]
    rw [hKX_term, hXtX_term, zero_add, Matrix.neg_mul]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  -- Key cancellation 1: `Xᵀ * (V + ζ) = 2 • Z1`
  have hX_Vpζ : Xᵀ * (V + ζ) = (2 : ℝ) • Z1 := by
    have h_exp : Xᵀ * (V + ζ) =
        (2 : ℝ) • R1 + τ • ((Xᵀ * ξ_tilde) * Z1) + (2 : ℝ) • ((Xᵀ * X) * (C * Z1 - R1)) := by
      rw [hV_def]
      simp only [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_assoc, ← hR1_def]
      module
    rw [h_exp, hXt_ξt, hX, Matrix.one_mul, hC_def]
    simp only [Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  -- Key cancellation 2: `(τ / 2) • (ξ_tildeᵀ * (V + ζ)) = -2 • (C * Z1 - R1)`
  have hξt_Vpζ : (τ / 2 : ℝ) • (ξ_tildeᵀ * (V + ζ)) = -((2 : ℝ) • (C * Z1 - R1)) := by
    have h_ξt_ζ : ξ_tildeᵀ * ζ = -R2 := by rw [hR2_def, neg_neg]
    have h_exp : ξ_tildeᵀ * (V + ζ) =
        -((2 : ℝ) • R2) + τ • ((ξ_tildeᵀ * ξ_tilde) * Z1) +
          (2 : ℝ) • ((ξ_tildeᵀ * X) * (C * Z1 - R1)) := by
      rw [hV_def]
      simp only [Matrix.mul_add, Matrix.mul_smul, Matrix.mul_assoc, h_ξt_ζ]
      module
    have h_tauR2 : τ • R2 = (2 : ℝ) • (J * Z1) - (2 : ℝ) • (C * R1) := by
      have h2 := congrArg ((2 : ℝ) • ·) hJZ1
      rw [smul_add, smul_smul] at h2
      have h2half : (2 : ℝ) * (τ / 2) = τ := by ring
      rw [h2half] at h2
      exact eq_sub_of_add_eq' h2.symm
    rw [h_exp, hξt_ξt, hξt_X]
    have h_scale :
        (τ / 2 : ℝ) • (-((2 : ℝ) • R2) + τ • ((Kᵀ * K - (1 / 4 : ℝ) • (A * A)) * Z1) +
            (2 : ℝ) • ((-(1 / 2 : ℝ) • A) * (C * Z1 - R1))) =
          -(τ • R2) + (τ ^ 2 / 2 : ℝ) • ((Kᵀ * K) * Z1) - (τ ^ 2 / 8 : ℝ) • (A * (A * Z1)) -
            (τ / 2 : ℝ) • (A * (C * Z1 - R1)) := by
      simp only [Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc]
      ext i j
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul]
      ring
    rw [h_scale, h_tauR2, hJ_def, schurWoodburyMatrix, hC_def]
    simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.mul_sub, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.one_mul, Matrix.mul_assoc]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  -- Combine into `(1 - (τ/2) • W_ξ) * V = (1 + (τ/2) • W_ξ) * ζ`
  have h_cayley_eq : (1 - (τ / 2 : ℝ) • W_ξ) * V = (1 + (τ / 2 : ℝ) • W_ξ) * ζ := by
    have hW_Vpζ : (τ / 2 : ℝ) • (W_ξ * (V + ζ)) = V - ζ := by
      calc
        (τ / 2 : ℝ) • (W_ξ * (V + ζ))
            = (τ / 2 : ℝ) • (ξ_tilde * (Xᵀ * (V + ζ)) - X * (ξ_tildeᵀ * (V + ζ))) := by
              rw [hW_def, skewGenerator, ← hξt_def, Matrix.sub_mul, Matrix.mul_assoc, Matrix.mul_assoc]
        _ = (τ / 2 : ℝ) • (ξ_tilde * ((2 : ℝ) • Z1)) -
              X * ((τ / 2 : ℝ) • (ξ_tildeᵀ * (V + ζ))) := by
              rw [smul_sub, hX_Vpζ, Matrix.mul_smul, Matrix.mul_smul]
        _ = (τ / 2 : ℝ) • ((2 : ℝ) • (ξ_tilde * Z1)) + (2 : ℝ) • (X * (C * Z1 - R1)) := by
              rw [hξt_Vpζ, Matrix.mul_smul, Matrix.mul_neg, sub_neg_eq_add, Matrix.mul_smul]
        _ = τ • (ξ_tilde * Z1) + (2 : ℝ) • (X * (C * Z1 - R1)) := by
              rw [smul_smul]; congr 2; ring
        _ = V - ζ := by rw [hV_def]; abel
    have h_diff : (1 - (τ / 2 : ℝ) • W_ξ) * V - (1 + (τ / 2 : ℝ) • W_ξ) * ζ = 0 := by
      calc
        (1 - (τ / 2 : ℝ) • W_ξ) * V - (1 + (τ / 2 : ℝ) • W_ξ) * ζ
            = (V - ζ) - (τ / 2 : ℝ) • (W_ξ * (V + ζ)) := by
              simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.mul_add, Matrix.one_mul,
                Matrix.smul_mul, smul_add]
              abel
        _ = 0 := by rw [hW_Vpζ, sub_self]
    exact sub_eq_zero.mp h_diff
  -- Multiply on the left by `M = (1 - (τ/2) • W_ξ)⁻¹`
  have hQ_eq_V : cayleyRotor W_ξ τ * ζ = V := by
    calc
      cayleyRotor W_ξ τ * ζ = M * ((1 + (τ / 2 : ℝ) • W_ξ) * ζ) := by
        rw [cayleyRotor, ← hM_def, Matrix.mul_assoc]
      _ = M * ((1 - (τ / 2 : ℝ) • W_ξ) * V) := by rw [← h_cayley_eq]
      _ = (M * (1 - (τ / 2 : ℝ) • W_ξ)) * V := by rw [Matrix.mul_assoc]
      _ = V := by rw [hM_right, Matrix.one_mul]
  simpa only [schurWoodburyTransport, ← hA_def, ← hK_def, ← hξt_def, ← hJ_def, ← hC_def,
    ← hR1_def, ← hR2_def, ← hZ1_def, ← hV_def] using hQ_eq_V

/-- **Corollary (`schur_woodbury_transport_isometry`)**:
Branch A Schur-Woodbury transport preserves Stiefel tangent skew-symmetry, cached skew
projections $Y^\top \mathcal{T}_A(\zeta) = X^\top \zeta$, and canonical Stiefel inner products
`stiefelInner (1 / 2)` identically. -/
theorem schur_woodbury_transport_isometry {n p : ℕ}
    (X ξ ζ₁ ζ₂ : Matrix (Fin n) (Fin p) ℝ) (τ : ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0) :
    let Y := schurWoodburyRetract X (ξ - X * (Xᵀ * ξ)) (Xᵀ * ξ) τ
    let T₁ := schurWoodburyTransport X ξ τ ζ₁
    let T₂ := schurWoodburyTransport X ξ τ ζ₂
    (Yᵀ * T₁ + T₁ᵀ * Y = Xᵀ * ζ₁ + ζ₁ᵀ * X) ∧
    (Yᵀ * T₁ = Xᵀ * ζ₁) ∧
    (stiefelInner (1 / 2) Y T₁ T₂ = stiefelInner (1 / 2) X ζ₁ ζ₂) := by
  dsimp only
  have hWt : (skewGenerator X ξ)ᵀ = -(skewGenerator X ξ) :=
    (skew_generator_properties X ξ hX hξ).1
  have hY : cayleyRotor (skewGenerator X ξ) τ * X =
      schurWoodburyRetract X (ξ - X * (Xᵀ * ξ)) (Xᵀ * ξ) τ :=
    cayley_branch_a_eq_branch_b X ξ τ hX hξ
  have hT₁ : cayleyRotor (skewGenerator X ξ) τ * ζ₁ = schurWoodburyTransport X ξ τ ζ₁ :=
    cayley_branch_a_transport_eq_branch_b X ξ τ ζ₁ hX hξ
  have hT₂ : cayleyRotor (skewGenerator X ξ) τ * ζ₂ = schurWoodburyTransport X ξ τ ζ₂ :=
    cayley_branch_a_transport_eq_branch_b X ξ τ ζ₂ hX hξ
  have h_iso := (cayley_rotor_orthogonality_and_isometry (skewGenerator X ξ) τ X ζ₁ ζ₁ ζ₂ hWt).2.2.2.2
  rw [← hY, ← hT₁, ← hT₂]
  exact ⟨h_iso.1, h_iso.2.1.1, h_iso.2.2.2⟩

-- ============================================================================
-- Section 4: Filtered L-BFGS Two-Loop Symmetry & Positive-Definiteness
-- ============================================================================

/-- One step of the matrix-free L-BFGS two-loop recursion wrapping an inner inverse-Hessian
operator `H` with secant pair `(s, y)` under inner product `ip`:
`α = ρ ⟨s, v⟩`, `q = H (v - α y)`, `β = ρ ⟨y, q⟩`, returning `q + (α - β) s`. -/
def lbfgsStep {V : Type*} [AddCommGroup V] [Module ℝ V]
    (ip : V → V → ℝ) (s y : V) (H : V → V) (v : V) : V :=
  let ρ : ℝ := 1 / ip s y
  let α : ℝ := ρ * ip s v
  let q : V := H (v - α • y)
  let β : ℝ := ρ * ip y q
  q + (α - β) • s

/-- Matrix-free L-BFGS two-loop recursion (`fc_lbfgs_apply_H` in `src/fortchuck_lbfgs.f90`)
over a list of active curvature-filtered secant pairs ordered newest-first. -/
def lbfgsTwoLoop {V : Type*} [AddCommGroup V] [Module ℝ V]
    (ip : V → V → ℝ) (H₀ : V → V) : List (V × V) → V → V
  | [], v => H₀ v
  | (s, y) :: rest, v => lbfgsStep ip s y (lbfgsTwoLoop ip H₀ rest) v

/-- Single-step BFGS two-loop update preserves linearity, symmetry, and an explicit closed-form
strict positive-definiteness lower bound whenever `0 < ip s y`. -/
lemma lbfgs_step_symm_posDef_bound {V : Type*} [AddCommGroup V] [Module ℝ V]
    (ip : V → V → ℝ)
    (hip_symm : ∀ u v, ip u v = ip v u)
    (hip_add : ∀ u v w, ip (u + v) w = ip u w + ip v w)
    (hip_smul : ∀ (c : ℝ) u v, ip (c • u) v = c * ip u v)
    (hip_nonneg : ∀ v, 0 ≤ ip v v)
    (hip_eq0 : ∀ v, ip v v = 0 → v = 0)
    (H : V → V) (lam : ℝ)
    (hlam_pos : 0 < lam)
    (hH_add : ∀ u v, H (u + v) = H u + H v)
    (hH_smul : ∀ (c : ℝ) v, H (c • v) = c • H v)
    (hH_symm : ∀ u v, ip u (H v) = ip (H u) v)
    (hH_pd : ∀ v, lam * ip v v ≤ ip v (H v))
    (s y : V) (hsy : 0 < ip s y) :
    let H' := lbfgsStep ip s y H
    let lam' := (lam * ip s y) / (ip s y + lam * ip y y)
    0 < lam' ∧
    (∀ u v, H' (u + v) = H' u + H' v) ∧
    (∀ (c : ℝ) v, H' (c • v) = c • H' v) ∧
    (∀ u v, ip u (H' v) = ip (H' u) v) ∧
    (∀ v, lam' * ip v v ≤ ip v (H' v)) := by
  dsimp only
  have hip_add_right : ∀ u v w, ip u (v + w) = ip u v + ip u w := fun u v w => by
    rw [hip_symm, hip_add, hip_symm v u, hip_symm w u]
  have hip_smul_right : ∀ (c : ℝ) u v, ip u (c • v) = c * ip u v := fun c u v => by
    rw [hip_symm, hip_smul, hip_symm v u]
  have hip_zero_right : ∀ u, ip u 0 = 0 := fun u => by
    have h := hip_smul_right 0 u 0
    simpa using h
  have hip_sub_left : ∀ u v w, ip (u - v) w = ip u w - ip v w := fun u v w => by
    rw [sub_eq_add_neg, hip_add, ← neg_one_smul ℝ v, hip_smul]
    ring
  have hip_sub_right : ∀ u v w, ip u (v - w) = ip u v - ip u w := fun u v w => by
    rw [hip_symm, hip_sub_left, hip_symm v u, hip_symm w u]
  have hy_ne : y ≠ 0 := by
    rintro rfl
    rw [hip_zero_right] at hsy
    exact lt_irrefl 0 hsy
  have hyy_pos : 0 < ip y y :=
    lt_of_le_of_ne (hip_nonneg y) (Ne.symm (mt (hip_eq0 y) hy_ne))
  have hden_pos : 0 < ip s y + lam * ip y y := by nlinarith
  set ρ : ℝ := 1 / ip s y
  have h_bilin : ∀ u v,
      ip u (lbfgsStep ip s y H v) =
        ip (u - (ρ * ip s u) • y) (H (v - (ρ * ip s v) • y)) + ρ * ip s u * ip s v := by
    intro u v
    unfold lbfgsStep
    dsimp only
    set αu : ℝ := ρ * ip s u
    set αv : ℝ := ρ * ip s v
    set wu : V := u - αu • y
    set wv : V := v - αv • y
    have hu_split : u = wu + αu • y := by simp only [wu]; abel
    rw [hip_add_right, hip_smul_right]
    nth_rw 1 [hu_split]
    rw [hip_add, hip_smul, hip_symm u s]
    have h_cancel : αu * ip y (H wv) + (αv - ρ * ip y (H wv)) * ip s u = ρ * ip s u * ip s v := by
      simp only [αu, αv]
      ring
    linarith
  refine ⟨div_pos (mul_pos hlam_pos hsy) hden_pos, ?_, ?_, ?_, ?_⟩
  · intro u v
    unfold lbfgsStep
    dsimp only
    have h1 : u + v - (ρ * ip s (u + v)) • y =
        (u - (ρ * ip s u) • y) + (v - (ρ * ip s v) • y) := by
      rw [hip_add_right, mul_add, add_smul]
      abel
    rw [h1, hH_add, hip_add_right, hip_add_right]
    have hcoeff :
        ρ * (ip s u + ip s v) - ρ * (ip y (H (u - (ρ * ip s u) • y)) +
          ip y (H (v - (ρ * ip s v) • y))) =
        (ρ * ip s u - ρ * ip y (H (u - (ρ * ip s u) • y))) +
          (ρ * ip s v - ρ * ip y (H (v - (ρ * ip s v) • y))) := by ring
    rw [hcoeff, add_smul]
    abel
  · intro c v
    unfold lbfgsStep
    dsimp only
    have h1 : c • v - (ρ * ip s (c • v)) • y = c • (v - (ρ * ip s v) • y) := by
      rw [hip_smul_right, smul_sub, smul_smul]
      congr 2
      ring
    rw [h1, hH_smul, hip_smul_right, hip_smul_right, smul_add, smul_smul]
    congr 2
    ring
  · intro u v
    rw [h_bilin u v, hip_symm (lbfgsStep ip s y H u) v, h_bilin v u,
      hH_symm (u - (ρ * ip s u) • y) (v - (ρ * ip s v) • y),
      hip_symm (H (u - (ρ * ip s u) • y)) (v - (ρ * ip s v) • y)]
    ring
  · intro v
    rw [h_bilin v v]
    set α : ℝ := ρ * ip s v
    set w : V := v - α • y
    have h_step_lb : lam * ip w w + ip s y * α ^ 2 ≤ ip w (H w) + ρ * ip s v * ip s v := by
      have h_eq : ip s y * α ^ 2 = ρ * ip s v * ip s v := by
        have hsy_ne : ip s y ≠ 0 := ne_of_gt hsy
        simp only [α, ρ]
        field_simp [hsy_ne]
      linarith [hH_pd w]
    -- Exact completing-the-square identity without division by zero
    set c_sq : ℝ := (ip s y + lam * ip y y) / (lam * ip y y)
    have h_sq_nonneg : 0 ≤ ip (v - (c_sq * α) • y) (v - (c_sq * α) • y) := hip_nonneg _
    have h_expand_sq :
        ip (v - (c_sq * α) • y) (v - (c_sq * α) • y) =
          ip v v - 2 * (c_sq * α) * ip v y + (c_sq * α) ^ 2 * ip y y := by
      simp_rw [hip_sub_left, hip_sub_right, hip_smul, hip_smul_right, hip_symm y v]
      ring
    have h_expand_w : ip w w = ip v v - 2 * α * ip v y + α ^ 2 * ip y y := by
      rw [show w = v - α • y from rfl]
      simp_rw [hip_sub_left, hip_sub_right, hip_smul, hip_smul_right, hip_symm y v]
      ring
    have hlam_yy_pos : 0 < lam * ip y y := mul_pos hlam_pos hyy_pos
    have h_alg :
        lam * ip w w + ip s y * α ^ 2 -
          ((lam * ip s y) / (ip s y + lam * ip y y)) * ip v v =
        ((lam * (lam * ip y y)) / (ip s y + lam * ip y y)) *
          ip (v - (c_sq * α) • y) (v - (c_sq * α) • y) := by
      rw [h_expand_w, h_expand_sq]
      have hd1 : ip s y + lam * ip y y ≠ 0 := ne_of_gt hden_pos
      have hd2 : lam * ip y y ≠ 0 := ne_of_gt hlam_yy_pos
      simp only [c_sq]
      field_simp [hd1, hd2]
      ring
    have h_rhs_nonneg :
        0 ≤ ((lam * (lam * ip y y)) / (ip s y + lam * ip y y)) *
          ip (v - (c_sq * α) • y) (v - (c_sq * α) • y) :=
      mul_nonneg (div_nonneg (mul_nonneg (le_of_lt hlam_pos) (le_of_lt hlam_yy_pos))
        (le_of_lt hden_pos)) h_sq_nonneg
    linarith

/-- **Theorem 5a (`lbfgs_two_loop_symm_posDef_bound`)**:
If the diagonal/block center preconditioner `H₀` is linear, symmetric, and strictly
positive-definite with lower bound `lam₀ > 0`, and every dynamically filtered secant pair
`(s, y) ∈ pairs` satisfies `0 < ip s y` (`filter_free_subspace_pairs`), then the matrix-free
two-loop recursion `lbfgsTwoLoop ip H₀ pairs` is linear, symmetric, and strictly
positive-definite with an explicit positive lower eigenvalue bound `lam_min > 0`. -/
theorem lbfgs_two_loop_symm_posDef_bound {V : Type*} [AddCommGroup V] [Module ℝ V]
    (ip : V → V → ℝ)
    (hip_symm : ∀ u v, ip u v = ip v u)
    (hip_add : ∀ u v w, ip (u + v) w = ip u w + ip v w)
    (hip_smul : ∀ (c : ℝ) u v, ip (c • u) v = c * ip u v)
    (hip_nonneg : ∀ v, 0 ≤ ip v v)
    (hip_eq0 : ∀ v, ip v v = 0 → v = 0)
    (H₀ : V → V) (lam₀ : ℝ)
    (hlam₀ : 0 < lam₀)
    (hH₀_add : ∀ u v, H₀ (u + v) = H₀ u + H₀ v)
    (hH₀_smul : ∀ (c : ℝ) v, H₀ (c • v) = c • H₀ v)
    (hH₀_symm : ∀ u v, ip u (H₀ v) = ip (H₀ u) v)
    (hH₀_pd : ∀ v, lam₀ * ip v v ≤ ip v (H₀ v))
    (pairs : List (V × V))
    (hpairs : ∀ p ∈ pairs, 0 < ip p.1 p.2) :
    ∃ lam_min : ℝ, 0 < lam_min ∧
      (∀ u v, lbfgsTwoLoop ip H₀ pairs (u + v) =
        lbfgsTwoLoop ip H₀ pairs u + lbfgsTwoLoop ip H₀ pairs v) ∧
      (∀ (c : ℝ) v, lbfgsTwoLoop ip H₀ pairs (c • v) =
        c • lbfgsTwoLoop ip H₀ pairs v) ∧
      (∀ u v, ip u (lbfgsTwoLoop ip H₀ pairs v) = ip (lbfgsTwoLoop ip H₀ pairs u) v) ∧
      (∀ v, lam_min * ip v v ≤ ip v (lbfgsTwoLoop ip H₀ pairs v)) := by
  induction pairs with
  | nil =>
    exact ⟨lam₀, hlam₀, hH₀_add, hH₀_smul, hH₀_symm, hH₀_pd⟩
  | cons p rest ih =>
    rcases p with ⟨s, y⟩
    have hsy : 0 < ip s y := hpairs (s, y) (List.mem_cons_self ..)
    have hrest : ∀ q ∈ rest, 0 < ip q.1 q.2 :=
      fun q hq => hpairs q (List.mem_cons_of_mem _ hq)
    rcases ih hrest with ⟨lam, hlam, hadd, hsmul, hsymm, hpd⟩
    have hstep := lbfgs_step_symm_posDef_bound ip hip_symm hip_add hip_smul hip_nonneg hip_eq0
      (lbfgsTwoLoop ip H₀ rest) lam hlam hadd hsmul hsymm hpd s y hsy
    exact ⟨_, hstep⟩

-- ============================================================================
-- Section 5: Coupled Two-Metric Strict Descent & KKT Stationarity (Theorem 5)
-- ============================================================================

/-- Trace of $M^\top M$ equals the sum of squared entries and is nonnegative. -/
lemma trace_transpose_mul_self_nonneg {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) :
    0 ≤ Matrix.trace (Mᵀ * M) := by
  rw [Matrix.trace]
  refine Finset.sum_nonneg fun j _ => ?_
  rw [Matrix.diag_apply, Matrix.mul_apply]
  exact Finset.sum_nonneg fun i _ => by
    rw [Matrix.transpose_apply]
    exact mul_self_nonneg (M i j)

/-- Trace of $M^\top M$ vanishes if and only if $M = 0$. -/
lemma trace_transpose_mul_self_eq_zero_iff {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) :
    Matrix.trace (Mᵀ * M) = 0 ↔ M = 0 := by
  constructor
  · intro htr
    rw [Matrix.trace] at htr
    have h_col := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => by
      rw [Matrix.diag_apply, Matrix.mul_apply]
      exact Finset.sum_nonneg fun i _ => by
        rw [Matrix.transpose_apply]; exact mul_self_nonneg (M i j))).mp htr
    ext i j
    have hj := h_col j (Finset.mem_univ j)
    rw [Matrix.diag_apply, Matrix.mul_apply] at hj
    have hi := (Finset.sum_eq_zero_iff_of_nonneg (fun k _ => by
      rw [Matrix.transpose_apply]; exact mul_self_nonneg (M k j))).mp hj i (Finset.mem_univ i)
    rw [Matrix.transpose_apply] at hi
    exact mul_self_eq_zero.mp hi
  · rintro rfl
    simp

/-- Symmetry of the parametric Stiefel inner product `stiefelInner β X ξ η`. -/
lemma stiefelInner_symm {n p : ℕ} (β : ℝ)
    (X ξ η : Matrix (Fin n) (Fin p) ℝ) :
    stiefelInner β X ξ η = stiefelInner β X η ξ := by
  unfold stiefelInner
  rw [← Matrix.trace_transpose]
  congr 1
  simp only [Matrix.transpose_mul, Matrix.transpose_sub, Matrix.transpose_one,
    Matrix.transpose_smul, Matrix.transpose_transpose, Matrix.mul_assoc]

/-- Left additivity of `stiefelInner β X`. -/
lemma stiefelInner_add_left {n p : ℕ} (β : ℝ)
    (X ξ₁ ξ₂ η : Matrix (Fin n) (Fin p) ℝ) :
    stiefelInner β X (ξ₁ + ξ₂) η = stiefelInner β X ξ₁ η + stiefelInner β X ξ₂ η := by
  unfold stiefelInner
  rw [Matrix.transpose_add, Matrix.add_mul, Matrix.add_mul, Matrix.trace_add]

/-- Left scalar linearity of `stiefelInner β X`. -/
lemma stiefelInner_smul_left {n p : ℕ} (β c : ℝ)
    (X ξ η : Matrix (Fin n) (Fin p) ℝ) :
    stiefelInner β X (c • ξ) η = c * stiefelInner β X ξ η := by
  unfold stiefelInner
  rw [Matrix.transpose_smul, Matrix.smul_mul, Matrix.smul_mul, Matrix.trace_smul, smul_eq_mul]

/-- Canonical Stiefel metric ($\beta = 1/2$) decomposes into orthogonal horizontal + vertical
Frobenius norms on any orthonormal frame $X^\top X = I_p$, hence is strictly positive-definite. -/
lemma stiefelInner_canonical_decomp {n p : ℕ}
    (X Z : Matrix (Fin n) (Fin p) ℝ) (hX : Xᵀ * X = 1) :
    stiefelInner (1 / 2) X Z Z =
      (1 / 2 : ℝ) * Matrix.trace ((Xᵀ * Z)ᵀ * (Xᵀ * Z)) +
        Matrix.trace ((Z - X * (Xᵀ * Z))ᵀ * (Z - X * (Xᵀ * Z))) := by
  set B := Xᵀ * Z
  set K := Z - X * B
  have hXK : Xᵀ * K = 0 := by
    simp only [K, B, Matrix.mul_sub, ← Matrix.mul_assoc Xᵀ X (Xᵀ * Z), hX, Matrix.one_mul, sub_self]
  have hKX : Kᵀ * X = 0 := by
    have h := congrArg Matrix.transpose hXK
    rwa [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_zero] at h
  have hZ : Z = X * B + K := by simp only [K]; abel
  have h_mat : Zᵀ * ((1 : Matrix (Fin n) (Fin n) ℝ) - (1 - (1 / 2 : ℝ)) • (X * Xᵀ)) * Z =
      (1 / 2 : ℝ) • (Bᵀ * B) + Kᵀ * K := by
    have h_half : (1 : ℝ) - (1 / 2 : ℝ) = (1 / 2 : ℝ) := by norm_num
    rw [h_half]
    calc
      Zᵀ * ((1 : Matrix (Fin n) (Fin n) ℝ) - (1 / 2 : ℝ) • (X * Xᵀ)) * Z
          = Zᵀ * Z - (1 / 2 : ℝ) • ((Zᵀ * X) * (Xᵀ * Z)) := by
            simp only [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one,
              Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_assoc]
      _ = (Bᵀ * Xᵀ + Kᵀ) * (X * B + K) - (1 / 2 : ℝ) • (Bᵀ * B) := by
            have hZtX : Zᵀ * X = Bᵀ := by
              simp only [B, Matrix.transpose_mul, Matrix.transpose_transpose]
            rw [hZtX, show Xᵀ * Z = B from rfl]
            nth_rw 1 [hZ]
            nth_rw 1 [hZ]
            rw [Matrix.transpose_add, Matrix.transpose_mul]
      _ = Bᵀ * B + Kᵀ * K - (1 / 2 : ℝ) • (Bᵀ * B) := by
            simp only [Matrix.add_mul, Matrix.mul_add, Matrix.mul_assoc,
              ← Matrix.mul_assoc Xᵀ X B, hX, Matrix.one_mul, hXK,
              ← Matrix.mul_assoc Kᵀ X B, hKX, Matrix.zero_mul, Matrix.mul_zero,
              add_zero, zero_add]
      _ = (1 / 2 : ℝ) • (Bᵀ * B) + Kᵀ * K := by module
  unfold stiefelInner
  rw [h_mat, Matrix.trace_add, Matrix.trace_smul, smul_eq_mul]

lemma stiefelInner_canonical_nonneg {n p : ℕ}
    (X Z : Matrix (Fin n) (Fin p) ℝ) (hX : Xᵀ * X = 1) :
    0 ≤ stiefelInner (1 / 2) X Z Z := by
  rw [stiefelInner_canonical_decomp X Z hX]
  have h1 := trace_transpose_mul_self_nonneg (Xᵀ * Z)
  have h2 := trace_transpose_mul_self_nonneg (Z - X * (Xᵀ * Z))
  linarith

lemma stiefelInner_canonical_eq_zero_iff {n p : ℕ}
    (X Z : Matrix (Fin n) (Fin p) ℝ) (hX : Xᵀ * X = 1) :
    stiefelInner (1 / 2) X Z Z = 0 ↔ Z = 0 := by
  constructor
  · intro h0
    rw [stiefelInner_canonical_decomp X Z hX] at h0
    have h1 := trace_transpose_mul_self_nonneg (Xᵀ * Z)
    have h2 := trace_transpose_mul_self_nonneg (Z - X * (Xᵀ * Z))
    have hB0 : Matrix.trace ((Xᵀ * Z)ᵀ * (Xᵀ * Z)) = 0 := by linarith
    have hK0 : Matrix.trace ((Z - X * (Xᵀ * Z))ᵀ * (Z - X * (Xᵀ * Z))) = 0 := by linarith
    have hB : Xᵀ * Z = 0 := (trace_transpose_mul_self_eq_zero_iff _).mp hB0
    have hK : Z - X * (Xᵀ * Z) = 0 := (trace_transpose_mul_self_eq_zero_iff _).mp hK0
    rw [hB, Matrix.mul_zero, sub_zero] at hK
    exact hK
  · rintro rfl
    unfold stiefelInner
    simp

/-- Right-hand directional velocity $\dot{z}_i(0^+)$ of the projected box curve
$z(\tau) = P_{[l, u]}(z + \tau d_z)$ (`ARCHITECTURE_SPEC.md` §2.3). -/
def boxRightVelocity {d : ℕ} (l u z dz : Fin d → ℝ) (i : Fin d) : ℝ :=
  if (z i = l i ∧ dz i ≤ 0) ∨ (z i = u i ∧ 0 ≤ dz i) then 0 else dz i

/-- Fixed box coordinate predicate ($l_i = u_i$). -/
def IsFixedCoord {d : ℕ} (l u : Fin d → ℝ) (i : Fin d) : Prop :=
  l i = u i

/-- Safeguarded Bertsekas active-band predicate (`mask_box(i) == 1` in §2.2). -/
def IsActiveBand {d : ℕ} (l u z gz ε : Fin d → ℝ) (i : Fin d) : Prop :=
  l i < u i ∧ ((z i ≤ l i + ε i ∧ 0 < gz i) ∨ (u i - ε i ≤ z i ∧ gz i < 0))

/-- Strictly interior active-band coordinate predicate $\mathcal{A}_\epsilon^{\mathrm{int}}(z)$. -/
def IsActiveInterior {d : ℕ} (l u z gz ε : Fin d → ℝ) (i : Fin d) : Prop :=
  IsActiveBand l u z gz ε i ∧ l i < z i ∧ z i < u i

/-- Free box coordinate predicate $\mathcal{F}_\epsilon(z)$ (`mask_box(i) == 0` in §2.2). -/
def IsFreeCoord {d : ℕ} (l u z gz ε : Fin d → ℝ) (i : Fin d) : Prop :=
  l i < u i ∧ ¬ ((z i ≤ l i + ε i ∧ 0 < gz i) ∨ (u i - ε i ≤ z i ∧ gz i < 0))

/-- Finite index set of strictly interior active-band coordinates. -/
def activeIntFinset {d : ℕ} (l u z gz ε : Fin d → ℝ) : Finset (Fin d) :=
  Finset.univ.filter (IsActiveInterior l u z gz ε)

/-- Finite index set of free box coordinates. -/
def freeFinset {d : ℕ} (l u z gz ε : Fin d → ℝ) : Finset (Fin d) :=
  Finset.univ.filter (IsFreeCoord l u z gz ε)

/-- Joint right-hand directional derivative $\phi'(0^+)$ along the feasible product curve
$(P_{[l,u]}(z + \tau d_z), \mathcal{R}_X^{\mathrm{Cay}}(\tau \eta))$. -/
def coupledDirDeriv0 {d R : ℕ} {n p : Fin R → ℕ}
    (l u z gz dz : Fin d → ℝ)
    (X gX η : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ) : ℝ :=
  (∑ i : Fin d, gz i * boxRightVelocity l u z dz i) +
    ∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)

/-- Exact first-order coupled Box $\times$ Multi-Stiefel KKT stationarity predicate. -/
def IsCoupledKKTStationary {d R : ℕ} {n p : Fin R → ℕ}
    (l u z gz : Fin d → ℝ)
    (X gX : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ) : Prop :=
  (∀ i : Fin d, l i ≤ z i ∧ z i ≤ u i) ∧
  (∀ r : Fin R, (X r)ᵀ * (X r) = 1) ∧
  (∀ i : Fin d, l i < u i →
    (l i < z i ∧ z i < u i → gz i = 0) ∧
    (z i = l i → 0 ≤ gz i) ∧
    (z i = u i → gz i ≤ 0)) ∧
  (∀ r : Fin R, gX r = 0)

/-- Coordinatewise clamping inequality on the Bertsekas free set (§2.3):
boundary clamping on $\mathcal{F}_\epsilon(z)$ can only decrease $g_{z,i} \dot{z}_i(0^+)$
relative to the unprojected product $g_{z,i} d_{z,i}$. -/
lemma free_coord_clamping_le {d : ℕ}
    (l u z gz dz ε : Fin d → ℝ) (i : Fin d)
    (h_eps_nonneg : 0 ≤ ε i)
    (h_free : IsFreeCoord l u z gz ε i) :
    gz i * boxRightVelocity l u z dz i ≤ gz i * dz i := by
  unfold boxRightVelocity
  split_ifs with h_clamp
  · rw [mul_zero]
    rcases h_clamp with ⟨hzl, hdz⟩ | ⟨hzu, hdz⟩
    · have hgz_le : gz i ≤ 0 := by
        by_contra h_pos
        exact h_free.2 (Or.inl ⟨by linarith, lt_of_not_ge h_pos⟩)
      exact mul_nonneg_of_nonpos_of_nonpos hgz_le hdz
    · have hgz_ge : 0 ≤ gz i := by
        by_contra h_neg
        exact h_free.2 (Or.inr ⟨by linarith, lt_of_not_ge h_neg⟩)
      exact mul_nonneg hgz_ge hdz
  · exact le_rfl

/-- **Theorem 5b (`coupled_two_metric_strict_descent`)**:
Formalization of Theorem 2.1 (`ARCHITECTURE_SPEC.md` §2.3). Under primal feasibility,
safeguarded Bertsekas active-band width ($0 \le \epsilon_i < u_i - l_i$ whenever $l_i < u_i$),
safeguarded active-band steps $d_{z,i} = -\gamma_i g_{z,i}$ ($\gamma_i \ge \gamma_{\min} > 0$),
and positive-definite masked product quasi-Newton curvature ($\lambda_{\min} > 0$) on the joint
free subspace $\mathcal{S}_{\mathrm{free}}$:
1. Free-coordinate clamping preserves descent ($g_{z,i} \dot{z}_i(0^+) \le g_{z,i} d_{z,i}$)
   and active-band coordinates are sign-aligned ($g_{z,i} d_{z,i} \le 0$);
2. The joint initial directional derivative $\phi'(0^+)$ satisfies the exact Theorem 2.1
   two-metric quadratic upper bound;
3. $\phi'(0^+) < 0$ holds strictly whenever any active-interior, free-box, or Stiefel
   Riemannian gradient component is nonzero; and
4. $\phi'(0^+) = 0$ if and only if $(z, X)$ is a first-order KKT stationary point. -/
theorem coupled_two_metric_strict_descent {d R : ℕ} {n p : Fin R → ℕ}
    (l u z gz dz ε : Fin d → ℝ)
    (X gX η : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ)
    (γ_min lam_min : ℝ)
    (h_feas : ∀ i : Fin d, l i ≤ z i ∧ z i ≤ u i)
    (hX_orth : ∀ r : Fin R, (X r)ᵀ * (X r) = 1)
    (h_eps : ∀ i : Fin d, l i < u i → 0 ≤ ε i ∧ ε i < u i - l i)
    (hγ_pos : 0 < γ_min)
    (hlam_pos : 0 < lam_min)
    (h_fixed : ∀ i : Fin d, IsFixedCoord l u i → dz i = 0)
    (h_act : ∀ i : Fin d, IsActiveBand l u z gz ε i →
      ∃ γ_i : ℝ, γ_min ≤ γ_i ∧ dz i = -γ_i * gz i)
    (h_free_pd :
      (∑ i ∈ freeFinset l u z gz ε, gz i * dz i) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) ≤
      -lam_min * ((∑ i ∈ freeFinset l u z gz ε, gz i ^ 2) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r)))) :
    let φ'_0 := coupledDirDeriv0 l u z gz dz X gX η
    ((∀ i ∈ freeFinset l u z gz ε, gz i * boxRightVelocity l u z dz i ≤ gz i * dz i) ∧
     (∀ i : Fin d, IsActiveBand l u z gz ε i → gz i * dz i ≤ 0)) ∧
    (φ'_0 ≤
      -γ_min * (∑ i ∈ activeIntFinset l u z gz ε, gz i ^ 2) -
      lam_min * ((∑ i ∈ freeFinset l u z gz ε, gz i ^ 2) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r)))) ∧
    (((∃ i ∈ activeIntFinset l u z gz ε, gz i ≠ 0) ∨
      (∃ i ∈ freeFinset l u z gz ε, gz i ≠ 0) ∨
      (∃ r : Fin R, gX r ≠ 0)) → φ'_0 < 0) ∧
    (φ'_0 = 0 ↔ IsCoupledKKTStationary l u z gz X gX) := by
  dsimp only
  set A_int := activeIntFinset l u z gz ε
  set F_set := freeFinset l u z gz ε
  set φ'_0 := coupledDirDeriv0 l u z gz dz X gX η with hφ_def
  have h_free_clamp : ∀ i ∈ F_set, gz i * boxRightVelocity l u z dz i ≤ gz i * dz i := by
    intro i hi
    have hf : IsFreeCoord l u z gz ε i := (Finset.mem_filter.mp hi).2
    exact free_coord_clamping_le l u z gz dz ε i (h_eps i hf.1).1 hf
  have h_act_sign : ∀ i : Fin d, IsActiveBand l u z gz ε i → gz i * dz i ≤ 0 := by
    intro i hi
    rcases h_act i hi with ⟨γ_i, hγ_ge, hdz⟩
    rw [hdz]
    have hγ_nonneg : 0 ≤ γ_i := le_of_lt (lt_of_lt_of_le hγ_pos hγ_ge)
    have hsq : 0 ≤ gz i ^ 2 := sq_nonneg (gz i)
    nlinarith
  have h_disj : Disjoint A_int F_set := by
    rw [Finset.disjoint_left]
    intro i hiA hiF
    have hA : IsActiveInterior l u z gz ε i := (Finset.mem_filter.mp hiA).2
    have hF : IsFreeCoord l u z gz ε i := (Finset.mem_filter.mp hiF).2
    exact hF.2 hA.1.2
  have h_rest_nonpos :
      ∀ i : Fin d, i ∉ A_int ∪ F_set → gz i * boxRightVelocity l u z dz i ≤ 0 := by
    intro i hi_not
    rw [Finset.mem_union, not_or] at hi_not
    rcases hi_not with ⟨_, h_not_F⟩
    by_cases h_lu : l i < u i
    · have h_in_act : IsActiveBand l u z gz ε i := by
        by_contra h_not_act
        have hF : IsFreeCoord l u z gz ε i := ⟨h_lu, fun h_cond => h_not_act ⟨h_lu, h_cond⟩⟩
        exact h_not_F (Finset.mem_filter.mpr ⟨Finset.mem_univ i, hF⟩)
      rcases h_act i h_in_act with ⟨γ_i, hγ_ge, hdz⟩
      unfold boxRightVelocity
      split_ifs with h_clamp
      · simp
      · rw [hdz]
        have hsq : 0 ≤ gz i ^ 2 := sq_nonneg (gz i)
        nlinarith
    · have h_eq : l i = u i := le_antisymm ((h_feas i).1.trans (h_feas i).2) (not_lt.mp h_lu)
      have hdz0 : dz i = 0 := h_fixed i h_eq
      unfold boxRightVelocity
      rw [hdz0]
      split_ifs <;> simp
  have h_Aint_bound :
      ∀ i ∈ A_int, gz i * boxRightVelocity l u z dz i ≤ -γ_min * gz i ^ 2 := by
    intro i hi
    have hAi : IsActiveInterior l u z gz ε i := (Finset.mem_filter.mp hi).2
    rcases h_act i hAi.1 with ⟨γ_i, hγ_ge, hdz⟩
    have h_no_clamp : ¬ ((z i = l i ∧ dz i ≤ 0) ∨ (z i = u i ∧ 0 ≤ dz i)) := by
      rintro (⟨hzl, _⟩ | ⟨hzu, _⟩) <;> linarith [hAi.2.1, hAi.2.2]
    unfold boxRightVelocity
    simp only [h_no_clamp, ↓reduceIte]
    rw [hdz]
    have hsq : 0 ≤ gz i ^ 2 := sq_nonneg (gz i)
    nlinarith
  have h_box_split :
      (∑ i : Fin d, gz i * boxRightVelocity l u z dz i) ≤
        (∑ i ∈ A_int, -γ_min * gz i ^ 2) + (∑ i ∈ F_set, gz i * dz i) := by
    set S := A_int ∪ F_set
    have h_univ :
        (∑ i : Fin d, gz i * boxRightVelocity l u z dz i) =
          (∑ i ∈ S, gz i * boxRightVelocity l u z dz i) +
            (∑ i ∈ Finset.univ \ S, gz i * boxRightVelocity l u z dz i) := by
      rw [← Finset.sum_union (Finset.disjoint_sdiff), Finset.union_sdiff_of_subset (Finset.subset_univ S)]
    have h_sdiff_le : (∑ i ∈ Finset.univ \ S, gz i * boxRightVelocity l u z dz i) ≤ 0 := by
      refine Finset.sum_nonpos fun i hi => ?_
      exact h_rest_nonpos i (Finset.mem_sdiff.mp hi).2
    have h_union_sum :
        (∑ i ∈ S, gz i * boxRightVelocity l u z dz i) =
          (∑ i ∈ A_int, gz i * boxRightVelocity l u z dz i) +
            (∑ i ∈ F_set, gz i * boxRightVelocity l u z dz i) :=
      Finset.sum_union h_disj
    have h1 : (∑ i ∈ A_int, gz i * boxRightVelocity l u z dz i) ≤
        ∑ i ∈ A_int, -γ_min * gz i ^ 2 := Finset.sum_le_sum h_Aint_bound
    have h2 : (∑ i ∈ F_set, gz i * boxRightVelocity l u z dz i) ≤
        ∑ i ∈ F_set, gz i * dz i := Finset.sum_le_sum h_free_clamp
    linarith
  have h_main_bound :
      φ'_0 ≤ -γ_min * (∑ i ∈ A_int, gz i ^ 2) -
        lam_min * ((∑ i ∈ F_set, gz i ^ 2) +
          (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r))) := by
    have h_factor : (∑ i ∈ A_int, -γ_min * gz i ^ 2) = -γ_min * (∑ i ∈ A_int, gz i ^ 2) := by
      rw [← Finset.mul_sum]
    have hφ_unfold : φ'_0 =
        (∑ i : Fin d, gz i * boxRightVelocity l u z dz i) +
          ∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r) := rfl
    linarith
  have h_sum_Aint_nonneg : 0 ≤ ∑ i ∈ A_int, gz i ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg (gz i)
  have h_sum_F_nonneg : 0 ≤ ∑ i ∈ F_set, gz i ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg (gz i)
  have h_sum_St_nonneg : 0 ≤ ∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r) :=
    Finset.sum_nonneg fun r _ => stiefelInner_canonical_nonneg (X r) (gX r) (hX_orth r)
  have h_strict :
      ((∃ i ∈ A_int, gz i ≠ 0) ∨ (∃ i ∈ F_set, gz i ≠ 0) ∨ (∃ r : Fin R, gX r ≠ 0)) →
        φ'_0 < 0 := by
    intro h_nz
    rcases h_nz with ⟨i₀, hi₀, hgz₀⟩ | ⟨i₀, hi₀, hgz₀⟩ | ⟨r₀, hgX₀⟩
    · have h_pos_term : 0 < gz i₀ ^ 2 := sq_pos_of_ne_zero hgz₀
      have h_sum_pos : 0 < ∑ i ∈ A_int, gz i ^ 2 :=
        lt_of_lt_of_le h_pos_term
          (Finset.single_le_sum (fun j _ => sq_nonneg (gz j)) hi₀)
      nlinarith
    · have h_pos_term : 0 < gz i₀ ^ 2 := sq_pos_of_ne_zero hgz₀
      have h_sum_pos : 0 < ∑ i ∈ F_set, gz i ^ 2 :=
        lt_of_lt_of_le h_pos_term
          (Finset.single_le_sum (fun j _ => sq_nonneg (gz j)) hi₀)
      nlinarith
    · have h_st_ne : stiefelInner (1 / 2) (X r₀) (gX r₀) (gX r₀) ≠ 0 :=
        mt (stiefelInner_canonical_eq_zero_iff (X r₀) (gX r₀) (hX_orth r₀)).mp hgX₀
      have h_st_pos : 0 < stiefelInner (1 / 2) (X r₀) (gX r₀) (gX r₀) :=
        lt_of_le_of_ne (stiefelInner_canonical_nonneg (X r₀) (gX r₀) (hX_orth r₀)) (Ne.symm h_st_ne)
      have h_sum_pos : 0 < ∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r) :=
        lt_of_lt_of_le h_st_pos
          (Finset.single_le_sum
            (fun r _ => stiefelInner_canonical_nonneg (X r) (gX r) (hX_orth r))
            (Finset.mem_univ r₀))
      nlinarith
  have h_kkt_iff : φ'_0 = 0 ↔ IsCoupledKKTStationary l u z gz X gX := by
    constructor
    · intro h_zero
      have hA0 : ∀ i ∈ A_int, gz i = 0 := by
        intro i hi
        by_contra hne
        have hlt := h_strict (Or.inl ⟨i, hi, hne⟩)
        linarith
      have hF0 : ∀ i ∈ F_set, gz i = 0 := by
        intro i hi
        by_contra hne
        have hlt := h_strict (Or.inr (Or.inl ⟨i, hi, hne⟩))
        linarith
      have hSt0 : ∀ r : Fin R, gX r = 0 := by
        intro r
        by_contra hne
        have hlt := h_strict (Or.inr (Or.inr ⟨r, hne⟩))
        linarith
      refine ⟨h_feas, hX_orth, fun i h_lu => ?_, hSt0⟩
      have h_eps_i := h_eps i h_lu
      refine ⟨fun ⟨h_gt_l, h_lt_u⟩ => ?_, fun h_eq_l => ?_, fun h_eq_u => ?_⟩
      · by_cases h_band : (z i ≤ l i + ε i ∧ 0 < gz i) ∨ (u i - ε i ≤ z i ∧ gz i < 0)
        · have hiA : i ∈ A_int :=
            Finset.mem_filter.mpr ⟨Finset.mem_univ i, ⟨⟨h_lu, h_band⟩, h_gt_l, h_lt_u⟩⟩
          exact hA0 i hiA
        · have hiF : i ∈ F_set :=
            Finset.mem_filter.mpr ⟨Finset.mem_univ i, ⟨h_lu, h_band⟩⟩
          exact hF0 i hiF
      · by_contra h_not_ge
        have h_neg : gz i < 0 := lt_of_not_ge h_not_ge
        have h_not_band : ¬ ((z i ≤ l i + ε i ∧ 0 < gz i) ∨ (u i - ε i ≤ z i ∧ gz i < 0)) := by
          rintro (⟨_, h_pos⟩ | ⟨h_hi, _⟩) <;> linarith [h_eps_i.2]
        have hiF : i ∈ F_set :=
          Finset.mem_filter.mpr ⟨Finset.mem_univ i, ⟨h_lu, h_not_band⟩⟩
        exact (ne_of_lt h_neg) (hF0 i hiF)
      · by_contra h_not_le
        have h_pos : 0 < gz i := lt_of_not_ge h_not_le
        have h_not_band : ¬ ((z i ≤ l i + ε i ∧ 0 < gz i) ∨ (u i - ε i ≤ z i ∧ gz i < 0)) := by
          rintro (⟨h_lo, _⟩ | ⟨_, h_neg⟩) <;> linarith [h_eps_i.2]
        have hiF : i ∈ F_set :=
          Finset.mem_filter.mpr ⟨Finset.mem_univ i, ⟨h_lu, h_not_band⟩⟩
        exact (ne_of_gt h_pos) (hF0 i hiF)
    · rintro ⟨_, _, h_box_kkt, h_st_kkt⟩
      rw [hφ_def]
      unfold coupledDirDeriv0
      have h_st_sum_zero : (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) = 0 := by
        refine Finset.sum_eq_zero fun r _ => ?_
        rw [h_st_kkt r]
        unfold stiefelInner
        simp
      have h_box_sum_zero : (∑ i : Fin d, gz i * boxRightVelocity l u z dz i) = 0 := by
        refine Finset.sum_eq_zero fun i _ => ?_
        by_cases h_lu : l i < u i
        · have hkkt := h_box_kkt i h_lu
          have h_eps_i := h_eps i h_lu
          rcases lt_or_eq_of_le (h_feas i).1 with h_gt_l | h_eq_l
          · rcases lt_or_eq_of_le (h_feas i).2 with h_lt_u | h_eq_u
            · rw [hkkt.1 ⟨h_gt_l, h_lt_u⟩, zero_mul]
            · rcases lt_or_eq_of_le (hkkt.2.2 h_eq_u) with h_gz_neg | h_gz_zero
              · have h_in_act : IsActiveBand l u z gz ε i :=
                  ⟨h_lu, Or.inr ⟨by linarith [h_eps_i.1], h_gz_neg⟩⟩
                rcases h_act i h_in_act with ⟨γ_i, hγ_ge, hdz⟩
                have hdz_nonneg : 0 ≤ dz i := by
                  rw [hdz]
                  have hγ_pos' : 0 < γ_i := lt_of_lt_of_le hγ_pos hγ_ge
                  nlinarith
                have h_clamp_u : (z i = l i ∧ dz i ≤ 0) ∨ (z i = u i ∧ 0 ≤ dz i) :=
                  Or.inr ⟨h_eq_u, hdz_nonneg⟩
                unfold boxRightVelocity
                simp only [h_clamp_u, ↓reduceIte, mul_zero]
              · rw [h_gz_zero, zero_mul]
          · rcases lt_or_eq_of_le (hkkt.2.1 h_eq_l.symm) with h_gz_pos | h_gz_zero
            · have h_in_act : IsActiveBand l u z gz ε i :=
                ⟨h_lu, Or.inl ⟨by linarith [h_eps_i.1], h_gz_pos⟩⟩
              rcases h_act i h_in_act with ⟨γ_i, hγ_ge, hdz⟩
              have hdz_nonpos : dz i ≤ 0 := by
                rw [hdz]
                have hγ_pos' : 0 < γ_i := lt_of_lt_of_le hγ_pos hγ_ge
                nlinarith
              have h_clamp_l : (z i = l i ∧ dz i ≤ 0) ∨ (z i = u i ∧ 0 ≤ dz i) :=
                Or.inl ⟨h_eq_l.symm, hdz_nonpos⟩
              unfold boxRightVelocity
              simp only [h_clamp_l, ↓reduceIte, mul_zero]
            · rw [← h_gz_zero, zero_mul]
        · have h_eq : l i = u i := le_antisymm ((h_feas i).1.trans (h_feas i).2) (not_lt.mp h_lu)
          have hdz0 : dz i = 0 := h_fixed i h_eq
          have hzl : z i = l i := le_antisymm ((h_feas i).2.trans h_eq.symm.le) (h_feas i).1
          have h_clamp_fix : (z i = l i ∧ dz i ≤ 0) ∨ (z i = u i ∧ 0 ≤ dz i) :=
            Or.inl ⟨hzl, le_of_eq hdz0⟩
          unfold boxRightVelocity
          simp only [h_clamp_fix, ↓reduceIte, mul_zero]
      rw [h_box_sum_zero, h_st_sum_zero, add_zero]
  exact ⟨⟨h_free_clamp, h_act_sign⟩, h_main_bound, h_strict, h_kkt_iff⟩

/-- Joint free tangent space $\mathbb{R}^{\mathcal{F}} \oplus \bigoplus_{r=1}^R \mathbb{R}^{n_r \times p_r}$
matching `fc_lbfgs_apply_H`. -/
abbrev FreeCoupledVec {d R : ℕ} (F : Finset (Fin d)) (n p : Fin R → ℕ) :=
  (F → ℝ) × ((r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ)

/-- Canonical product inner product on the joint free subspace $\mathcal{F} \times \prod_r \mathrm{St}(n_r, p_r)$. -/
def freeCoupledInner {d R : ℕ} {n p : Fin R → ℕ}
    (F : Finset (Fin d))
    (X : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ)
    (u v : FreeCoupledVec F n p) : ℝ :=
  (∑ i : F, u.1 i * v.1 i) + ∑ r : Fin R, stiefelInner (1 / 2) (X r) (u.2 r) (v.2 r)

/-- Center diagonal/block preconditioner `diag_box` $\oplus$ `gamma_blk` in `fc_lbfgs_apply_H`. -/
def freeCoupledDiagScale {d R : ℕ} {n p : Fin R → ℕ}
    (F : Finset (Fin d)) (diag_box : Fin d → ℝ) (gamma_blk : Fin R → ℝ)
    (v : FreeCoupledVec F n p) : FreeCoupledVec F n p :=
  (fun i => diag_box i.1 * v.1 i, fun r => gamma_blk r • v.2 r)

/-- **Theorem 5c (`masked_lbfgs_coupled_strict_descent`)**:
End-to-end derivation removing the `h_free_pd` assumption from Theorem 5b: given a positive
diagonal/block center preconditioner (`diag_box(i) ≥ lam₀ > 0`, `gamma_blk(r) ≥ lam₀ > 0`) and
any finite history of curvature-filtered pairs (`∀ p ∈ pairs, 0 < freeCoupledInner F X p.1 p.2`,
as enforced by `filter_free_subspace_pairs`), setting the free search direction to the negated
two-loop output `-(lbfgsTwoLoop ... pairs g_free)` automatically produces `lam_min > 0`
discharging `h_free_pd` and guaranteeing strict coupled descent and exact KKT stationarity. -/
theorem masked_lbfgs_coupled_strict_descent {d R : ℕ} {n p : Fin R → ℕ}
    (l u z gz dz ε diag_box : Fin d → ℝ)
    (gamma_blk : Fin R → ℝ)
    (X gX η : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ)
    (γ_min lam₀ : ℝ)
    (pairs : List (FreeCoupledVec (freeFinset l u z gz ε) n p ×
                   FreeCoupledVec (freeFinset l u z gz ε) n p))
    (h_feas : ∀ i : Fin d, l i ≤ z i ∧ z i ≤ u i)
    (hX_orth : ∀ r : Fin R, (X r)ᵀ * (X r) = 1)
    (h_eps : ∀ i : Fin d, l i < u i → 0 ≤ ε i ∧ ε i < u i - l i)
    (hγ_pos : 0 < γ_min)
    (hlam₀ : 0 < lam₀)
    (hdiag : ∀ i ∈ freeFinset l u z gz ε, lam₀ ≤ diag_box i)
    (hgamma : ∀ r : Fin R, lam₀ ≤ gamma_blk r)
    (hpairs : ∀ pr ∈ pairs,
      0 < freeCoupledInner (freeFinset l u z gz ε) X pr.1 pr.2)
    (h_fixed : ∀ i : Fin d, IsFixedCoord l u i → dz i = 0)
    (h_act : ∀ i : Fin d, IsActiveBand l u z gz ε i →
      ∃ γ_i : ℝ, γ_min ≤ γ_i ∧ dz i = -γ_i * gz i)
    (h_free_step :
      let F := freeFinset l u z gz ε
      let g_free : FreeCoupledVec F n p := (fun i => gz i.1, gX)
      let q_free := lbfgsTwoLoop (freeCoupledInner F X)
        (freeCoupledDiagScale F diag_box gamma_blk) pairs g_free
      (∀ i : F, dz i.1 = -(q_free.1 i)) ∧ (∀ r : Fin R, η r = -(q_free.2 r))) :
    ∃ lam_min : ℝ, 0 < lam_min ∧
      let φ'_0 := coupledDirDeriv0 l u z gz dz X gX η
      (φ'_0 ≤
        -γ_min * (∑ i ∈ activeIntFinset l u z gz ε, gz i ^ 2) -
        lam_min * ((∑ i ∈ freeFinset l u z gz ε, gz i ^ 2) +
          (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r)))) ∧
      (((∃ i ∈ activeIntFinset l u z gz ε, gz i ≠ 0) ∨
        (∃ i ∈ freeFinset l u z gz ε, gz i ≠ 0) ∨
        (∃ r : Fin R, gX r ≠ 0)) → φ'_0 < 0) ∧
      (φ'_0 = 0 ↔ IsCoupledKKTStationary l u z gz X gX) := by
  let F := freeFinset l u z gz ε
  let ip := freeCoupledInner F X
  let H₀ : FreeCoupledVec F n p → FreeCoupledVec F n p :=
    freeCoupledDiagScale F diag_box gamma_blk
  have hip_symm : ∀ u v : FreeCoupledVec F n p, ip u v = ip v u := fun u v => by
    dsimp only [ip, freeCoupledInner]
    congr 1
    · refine Finset.sum_congr rfl fun i _ => mul_comm _ _
    · refine Finset.sum_congr rfl fun r _ => stiefelInner_symm (1 / 2) (X r) (u.2 r) (v.2 r)
  have hip_add : ∀ u v w : FreeCoupledVec F n p, ip (u + v) w = ip u w + ip v w := fun u v w => by
    dsimp only [ip, freeCoupledInner]
    have h1 : (∑ i : F, (u.1 i + v.1 i) * w.1 i) =
        (∑ i : F, u.1 i * w.1 i) + (∑ i : F, v.1 i * w.1 i) := by
      simp_rw [add_mul, Finset.sum_add_distrib]
    have h2 : (∑ r : Fin R, stiefelInner (1 / 2) (X r) (u.2 r + v.2 r) (w.2 r)) =
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (u.2 r) (w.2 r)) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (v.2 r) (w.2 r)) := by
      simp_rw [stiefelInner_add_left, Finset.sum_add_distrib]
    change (∑ i : F, (u.1 i + v.1 i) * w.1 i) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (u.2 r + v.2 r) (w.2 r)) = _
    linarith
  have hip_smul : ∀ (c : ℝ) (u v : FreeCoupledVec F n p), ip (c • u) v = c * ip u v := fun c u v => by
    dsimp only [ip, freeCoupledInner]
    have h1 : (∑ i : F, (c * u.1 i) * v.1 i) = c * (∑ i : F, u.1 i * v.1 i) := by
      simp_rw [mul_assoc, ← Finset.mul_sum]
    have h2 : (∑ r : Fin R, stiefelInner (1 / 2) (X r) (c • u.2 r) (v.2 r)) =
        c * (∑ r : Fin R, stiefelInner (1 / 2) (X r) (u.2 r) (v.2 r)) := by
      simp_rw [stiefelInner_smul_left, ← Finset.mul_sum]
    change (∑ i : F, (c * u.1 i) * v.1 i) +
        (∑ r : Fin R, stiefelInner (1 / 2) (X r) (c • u.2 r) (v.2 r)) = _
    linarith
  have hip_nonneg : ∀ v : FreeCoupledVec F n p, 0 ≤ ip v v := fun v => by
    dsimp only [ip, freeCoupledInner]
    exact add_nonneg
      (Finset.sum_nonneg fun i _ => mul_self_nonneg (v.1 i))
      (Finset.sum_nonneg fun r _ => stiefelInner_canonical_nonneg (X r) (v.2 r) (hX_orth r))
  have hip_eq0 : ∀ v : FreeCoupledVec F n p, ip v v = 0 → v = 0 := fun v hv => by
    dsimp only [ip, freeCoupledInner] at hv
    have h1_nn : 0 ≤ ∑ i : F, v.1 i * v.1 i := Finset.sum_nonneg fun i _ => mul_self_nonneg (v.1 i)
    have h2_nn : 0 ≤ ∑ r : Fin R, stiefelInner (1 / 2) (X r) (v.2 r) (v.2 r) :=
      Finset.sum_nonneg fun r _ => stiefelInner_canonical_nonneg (X r) (v.2 r) (hX_orth r)
    have h1_0 : (∑ i : F, v.1 i * v.1 i) = 0 := by linarith
    have h2_0 : (∑ r : Fin R, stiefelInner (1 / 2) (X r) (v.2 r) (v.2 r)) = 0 := by linarith
    have hv1 : v.1 = 0 := by
      ext i
      have hi := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => mul_self_nonneg (v.1 j))).mp h1_0
        i (Finset.mem_univ i)
      exact mul_self_eq_zero.mp hi
    have hv2 : v.2 = 0 := by
      ext r a b
      have hr := (Finset.sum_eq_zero_iff_of_nonneg
        (fun s _ => stiefelInner_canonical_nonneg (X s) (v.2 s) (hX_orth s))).mp h2_0
        r (Finset.mem_univ r)
      have hvr0 : v.2 r = 0 := (stiefelInner_canonical_eq_zero_iff (X r) (v.2 r) (hX_orth r)).mp hr
      simp [hvr0]
    exact Prod.ext hv1 hv2
  have hH₀_add : ∀ u v : FreeCoupledVec F n p, H₀ (u + v) = H₀ u + H₀ v := fun u v => by
    apply Prod.ext
    · ext i; simp [H₀, freeCoupledDiagScale, mul_add]
    · ext r a b; simp [H₀, freeCoupledDiagScale, smul_add]
  have hH₀_smul : ∀ (c : ℝ) (v : FreeCoupledVec F n p), H₀ (c • v) = c • H₀ v := fun c v => by
    apply Prod.ext
    · ext i; simp [H₀, freeCoupledDiagScale]; ring
    · ext r a b; simp [H₀, freeCoupledDiagScale, smul_smul]; ring
  have hH₀_symm : ∀ u v : FreeCoupledVec F n p, ip u (H₀ v) = ip (H₀ u) v := fun u v => by
    dsimp only [ip, H₀, freeCoupledInner, freeCoupledDiagScale]
    congr 1
    · refine Finset.sum_congr rfl fun i _ => by ring
    · refine Finset.sum_congr rfl fun r _ => ?_
      rw [stiefelInner_symm, stiefelInner_smul_left, stiefelInner_smul_left, stiefelInner_symm]
  have hH₀_pd : ∀ v : FreeCoupledVec F n p, lam₀ * ip v v ≤ ip v (H₀ v) := fun v => by
    dsimp only [ip, H₀, freeCoupledInner, freeCoupledDiagScale]
    rw [mul_add, Finset.mul_sum, Finset.mul_sum]
    refine add_le_add ?_ ?_
    · refine Finset.sum_le_sum fun i _ => ?_
      have hdi : lam₀ ≤ diag_box i.1 := hdiag i.1 i.2
      have hsq : 0 ≤ v.1 i * v.1 i := mul_self_nonneg (v.1 i)
      nlinarith
    · refine Finset.sum_le_sum fun r _ => ?_
      rw [stiefelInner_symm _ _ (v.2 r) (gamma_blk r • v.2 r), stiefelInner_smul_left]
      have hst_nn : 0 ≤ stiefelInner (1 / 2) (X r) (v.2 r) (v.2 r) :=
        stiefelInner_canonical_nonneg (X r) (v.2 r) (hX_orth r)
      nlinarith [hgamma r]
  rcases lbfgs_two_loop_symm_posDef_bound ip hip_symm hip_add hip_smul hip_nonneg hip_eq0
      H₀ lam₀ hlam₀ hH₀_add hH₀_smul hH₀_symm hH₀_pd pairs hpairs with
    ⟨lam_min, hlam_pos, _, _, _, h_loop_pd⟩
  let g_free : FreeCoupledVec F n p := (fun i => gz i.1, gX)
  let q_free := lbfgsTwoLoop ip H₀ pairs g_free
  have h_pd_g : lam_min * ip g_free g_free ≤ ip g_free q_free := h_loop_pd g_free
  have h_sum_coe : (∑ i : F, gz i.1 * gz i.1) = ∑ i ∈ F, gz i ^ 2 := by
    rw [← Finset.sum_coe_sort F (fun i => gz i ^ 2)]
    refine Finset.sum_congr rfl fun i _ => by ring
  have h_sum_dz : (∑ i ∈ F, gz i * dz i) = -(∑ i : F, gz i.1 * q_free.1 i) := by
    rw [← Finset.sum_coe_sort F (fun i => gz i * dz i), ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [h_free_step.1 i, mul_neg]
  have h_sum_st :
      (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) =
        -(∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (q_free.2 r)) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [h_free_step.2 r, stiefelInner_symm, ← neg_one_smul ℝ (q_free.2 r),
      stiefelInner_smul_left, stiefelInner_symm]
    ring
  have h_free_pd :
      (∑ i ∈ F, gz i * dz i) + (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) ≤
        -lam_min * ((∑ i ∈ F, gz i ^ 2) +
          (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r))) := by
    have h_ip_g : ip g_free g_free =
        (∑ i ∈ F, gz i ^ 2) + (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (gX r)) := by
      dsimp only [ip, freeCoupledInner, g_free]
      rw [h_sum_coe]
    have h_ip_gq : ip g_free q_free =
        (∑ i : F, gz i.1 * q_free.1 i) +
          (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (q_free.2 r)) := rfl
    rw [h_ip_g, h_ip_gq] at h_pd_g
    rw [h_sum_dz, h_sum_st]
    linarith
  have h5 := coupled_two_metric_strict_descent l u z gz dz ε X gX η γ_min lam_min
    h_feas hX_orth h_eps hγ_pos hlam_pos h_fixed h_act h_free_pd
  exact ⟨lam_min, hlam_pos, h5.2.1, h5.2.2.1, h5.2.2.2⟩

-- ============================================================================
-- Section 6: Linearly-Constrained Box / Simplex Projection & SPG Descent (Theorem 6)
-- ============================================================================

/-- Dual-shifted coordinatewise clamping operator
$z^\ast_i(\nu) = \min(u_i, \max(l_i, y_i - \nu a_i))$ characterizing the Euclidean projection
onto $\mathcal{C} = \{z \in [l, u]^d : a^\top z = b\}$ (`ARCHITECTURE_SPEC.md` §2.9). -/
def simplexBoxDualClamp {d : ℕ} (l u a y : Fin d → ℝ) (nu : ℝ) (i : Fin d) : ℝ :=
  min (u i) (max (l i) (y i - nu * a i))

/-- Dual linear-equality map $\psi(\nu) = \sum_{i=1}^d a_i \min(u_i, \max(l_i, y_i - \nu a_i))$. -/
def simplexBoxDualSum {d : ℕ} (l u a y : Fin d → ℝ) (nu : ℝ) : ℝ :=
  ∑ i : Fin d, a i * simplexBoxDualClamp l u a y nu i

/-- Minimum attainable linear functional $\underline{b} = \sum_{a_i \ge 0} a_i l_i + \sum_{a_i < 0} a_i u_i$
on $[l, u]^d$ (`fc_box_lin_validate` in `src/fortchuck_box.f90`). -/
def simplexBoxMinSum {d : ℕ} (l u a : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, if 0 ≤ a i then a i * l i else a i * u i

/-- Maximum attainable linear functional $\overline{b} = \sum_{a_i \ge 0} a_i u_i + \sum_{a_i < 0} a_i l_i$
on $[l, u]^d$ (`fc_box_lin_validate` in `src/fortchuck_box.f90`). -/
def simplexBoxMaxSum {d : ℕ} (l u a : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, if 0 ≤ a i then a i * u i else a i * l i

/-- Coordinatewise clamping $\min(u_i, \max(l_i, w_i))$ onto any valid interval $l_i \le u_i$
lies inside $[l_i, u_i]$. -/
lemma clamp_mem_Icc {l_i u_i w_i : ℝ} (hl : l_i ≤ u_i) :
    l_i ≤ min u_i (max l_i w_i) ∧ min u_i (max l_i w_i) ≤ u_i :=
  ⟨le_min hl (le_max_left l_i w_i), min_le_left u_i (max l_i w_i)⟩

/-- Pointwise 1D variational inequality for coordinatewise clamping onto $[l_i, u_i]$:
for any $z_i \in [l_i, u_i]$,
$(w_i - \min(u_i, \max(l_i, w_i)))(z_i - \min(u_i, \max(l_i, w_i))) \le 0$. -/
lemma clamp_variational_nonpos {l_i u_i w_i z_i : ℝ}
    (hl : l_i ≤ u_i) (hz_lo : l_i ≤ z_i) (hz_hi : z_i ≤ u_i) :
    (w_i - min u_i (max l_i w_i)) * (z_i - min u_i (max l_i w_i)) ≤ 0 := by
  rcases le_total w_i l_i with hwl | hlw
  · rw [max_eq_left hwl, min_eq_right hl]
    exact mul_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr hwl) (sub_nonneg.mpr hz_lo)
  · rw [max_eq_right hlw]
    rcases le_total w_i u_i with hwu | huw
    · rw [min_eq_right hwu, sub_self, zero_mul]
    · rw [min_eq_left huw]
      exact mul_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr huw) (sub_nonpos.mpr hz_hi)

/-- **Theorem 6a (`simplex_box_dual_root_exists`)**:
For valid box bounds $l_i \le u_i$ on $\mathbb{R}^d$, any normal $a \in \mathbb{R}^d$, and target
$b \in \mathbb{R}$:
1. **Exact Feasibility Certificate (`fc_box_lin_validate`)**:
   $\mathcal{C} = \{z \in [l, u]^d : a^\top z = b\}$ is nonempty if and only if
   $\underline{b} \le b \le \overline{b}$; and
2. **Unconditional Dual Root Existence (`fc_box_lin_project`)**:
   Whenever $\underline{b} \le b \le \overline{b}$, for every query point $y \in \mathbb{R}^d$
   there exists a dual multiplier $\nu \in \mathbb{R}$ satisfying
   $\sum_i a_i\,\mathrm{simplexBoxDualClamp}(l, u, a, y, \nu, i) = b$. -/
theorem simplex_box_dual_root_exists {d : ℕ}
    (l u a y : Fin d → ℝ) (b : ℝ)
    (hl : ∀ i : Fin d, l i ≤ u i) :
    ((∃ z : Fin d → ℝ, (∀ i : Fin d, l i ≤ z i ∧ z i ≤ u i) ∧ (∑ i : Fin d, a i * z i) = b) ↔
      simplexBoxMinSum l u a ≤ b ∧ b ≤ simplexBoxMaxSum l u a) ∧
    (simplexBoxMinSum l u a ≤ b → b ≤ simplexBoxMaxSum l u a →
      ∃ nu : ℝ, simplexBoxDualSum l u a y nu = b) := by
  have h_root : simplexBoxMinSum l u a ≤ b → b ≤ simplexBoxMaxSum l u a →
      ∃ nu : ℝ, simplexBoxDualSum l u a y nu = b := by
    intro hb_lo hb_hi
    set M : Fin d → ℝ := fun i => if a i = 0 then 0 else (|y i| + |l i| + |u i|) / |a i|
    set R : ℝ := ∑ i : Fin d, M i
    have hM_nn : ∀ i : Fin d, 0 ≤ M i := fun i => by
      dsimp only [M]
      split_ifs with ha0
      · exact le_rfl
      · exact div_nonneg (by linarith [abs_nonneg (y i), abs_nonneg (l i), abs_nonneg (u i)])
          (abs_nonneg (a i))
    have hR_nn : 0 ≤ R := Finset.sum_nonneg fun i _ => hM_nn i
    have hM_le_R : ∀ i : Fin d, M i ≤ R := fun i =>
      Finset.single_le_sum (fun j _ => hM_nn j) (Finset.mem_univ i)
    have h_negR : simplexBoxDualSum l u a y (-R) = simplexBoxMaxSum l u a := by
      unfold simplexBoxDualSum simplexBoxMaxSum simplexBoxDualClamp
      refine Finset.sum_congr rfl fun i _ => ?_
      rcases lt_trichotomy (a i) 0 with ha_neg | ha_zero | ha_pos
      · have ha_ne : a i ≠ 0 := ne_of_lt ha_neg
        have hnot : ¬ (0 ≤ a i) := not_le.mpr ha_neg
        simp only [hnot, ↓reduceIte]
        have h_abs : |a i| = -(a i) := abs_of_neg ha_neg
        have hMR : (|y i| + |l i| + |u i|) / |a i| ≤ R := by
          simpa only [M, ha_ne, ↓reduceIte] using hM_le_R i
        have h_mul : |y i| + |l i| + |u i| ≤ R * |a i| :=
          (div_le_iff₀ (abs_pos.mpr ha_ne)).mp hMR
        have h_le_l : y i - (-R) * a i ≤ l i := by
          have hy_le : y i ≤ |y i| := le_abs_self (y i)
          have hl_ge : -|l i| ≤ l i := neg_abs_le (l i)
          have hu_nn : 0 ≤ |u i| := abs_nonneg (u i)
          nlinarith
        rw [max_eq_left h_le_l, min_eq_right (hl i)]
      · simp [ha_zero]
      · have ha_ne : a i ≠ 0 := ne_of_gt ha_pos
        have hge : 0 ≤ a i := le_of_lt ha_pos
        simp only [hge, ↓reduceIte]
        have h_abs : |a i| = a i := abs_of_pos ha_pos
        have hMR : (|y i| + |l i| + |u i|) / |a i| ≤ R := by
          simpa only [M, ha_ne, ↓reduceIte] using hM_le_R i
        have h_mul : |y i| + |l i| + |u i| ≤ R * |a i| :=
          (div_le_iff₀ (abs_pos.mpr ha_ne)).mp hMR
        have h_ge_u : u i ≤ y i - (-R) * a i := by
          have hy_ge : -|y i| ≤ y i := neg_abs_le (y i)
          have hu_le : u i ≤ |u i| := le_abs_self (u i)
          have hl_nn : 0 ≤ |l i| := abs_nonneg (l i)
          nlinarith
        rw [max_eq_right (le_trans (hl i) h_ge_u), min_eq_left h_ge_u]
    have h_posR : simplexBoxDualSum l u a y R = simplexBoxMinSum l u a := by
      unfold simplexBoxDualSum simplexBoxMinSum simplexBoxDualClamp
      refine Finset.sum_congr rfl fun i _ => ?_
      rcases lt_trichotomy (a i) 0 with ha_neg | ha_zero | ha_pos
      · have ha_ne : a i ≠ 0 := ne_of_lt ha_neg
        have hnot : ¬ (0 ≤ a i) := not_le.mpr ha_neg
        simp only [hnot, ↓reduceIte]
        have h_abs : |a i| = -(a i) := abs_of_neg ha_neg
        have hMR : (|y i| + |l i| + |u i|) / |a i| ≤ R := by
          simpa only [M, ha_ne, ↓reduceIte] using hM_le_R i
        have h_mul : |y i| + |l i| + |u i| ≤ R * |a i| :=
          (div_le_iff₀ (abs_pos.mpr ha_ne)).mp hMR
        have h_ge_u : u i ≤ y i - R * a i := by
          have hy_ge : -|y i| ≤ y i := neg_abs_le (y i)
          have hu_le : u i ≤ |u i| := le_abs_self (u i)
          have hl_nn : 0 ≤ |l i| := abs_nonneg (l i)
          nlinarith
        rw [max_eq_right (le_trans (hl i) h_ge_u), min_eq_left h_ge_u]
      · simp [ha_zero]
      · have ha_ne : a i ≠ 0 := ne_of_gt ha_pos
        have hge : 0 ≤ a i := le_of_lt ha_pos
        simp only [hge, ↓reduceIte]
        have h_abs : |a i| = a i := abs_of_pos ha_pos
        have hMR : (|y i| + |l i| + |u i|) / |a i| ≤ R := by
          simpa only [M, ha_ne, ↓reduceIte] using hM_le_R i
        have h_mul : |y i| + |l i| + |u i| ≤ R * |a i| :=
          (div_le_iff₀ (abs_pos.mpr ha_ne)).mp hMR
        have h_le_l : y i - R * a i ≤ l i := by
          have hy_le : y i ≤ |y i| := le_abs_self (y i)
          have hl_ge : -|l i| ≤ l i := neg_abs_le (l i)
          have hu_nn : 0 ≤ |u i| := abs_nonneg (u i)
          nlinarith
        rw [max_eq_left h_le_l, min_eq_right (hl i)]
    have h_cont : Continuous (fun nu : ℝ => simplexBoxDualSum l u a y nu) := by
      unfold simplexBoxDualSum simplexBoxDualClamp
      refine continuous_finsetSum Finset.univ fun i _ => ?_
      exact continuous_const.mul
        (continuous_const.min
          (continuous_const.max (continuous_const.sub (continuous_id.mul continuous_const))))
    have h_ivt := intermediate_value_Icc' (show -R ≤ R by linarith) h_cont.continuousOn
    rw [h_posR, h_negR] at h_ivt
    rcases h_ivt ⟨hb_lo, hb_hi⟩ with ⟨nu, _, hnu⟩
    exact ⟨nu, hnu⟩
  refine ⟨⟨?_, ?_⟩, h_root⟩
  · rintro ⟨z, hz_box, hz_sum⟩
    rw [← hz_sum]
    constructor
    · unfold simplexBoxMinSum
      refine Finset.sum_le_sum fun i _ => ?_
      split_ifs with ha
      · exact mul_le_mul_of_nonneg_left (hz_box i).1 ha
      · exact mul_le_mul_of_nonpos_left (hz_box i).2 (le_of_lt (not_le.mp ha))
    · unfold simplexBoxMaxSum
      refine Finset.sum_le_sum fun i _ => ?_
      split_ifs with ha
      · exact mul_le_mul_of_nonneg_left (hz_box i).2 ha
      · exact mul_le_mul_of_nonpos_left (hz_box i).1 (le_of_lt (not_le.mp ha))
  · rintro ⟨hb_lo, hb_hi⟩
    rcases h_root hb_lo hb_hi with ⟨nu, hnu⟩
    refine ⟨simplexBoxDualClamp l u a y nu, fun i => clamp_mem_Icc (hl i), hnu⟩

/-- **Theorem 6b (`simplex_box_projection_variational`)**:
For valid box bounds $l_i \le u_i$ on $\mathbb{R}^d$, if $z^\ast = \mathrm{simplexBoxDualClamp}(l, u, a, y, \nu)$
satisfies $\mathrm{simplexBoxDualSum}(l, u, a, y, \nu) = b$, then:
- **(a)** Box feasibility: $l_i \le z^\ast_i \le u_i$ for every $i$;
- **(b)** Variational inequality: $\sum_i (y_i - z^\ast_i)(z_i - z^\ast_i) \le 0$ for every
  feasible $z \in \mathcal{C} = \{z \in [l, u]^d : \sum_i a_i z_i = b\}$;
- **(c)** Strong-convexity quadratic bound & unique Euclidean projection:
  $\|z^\ast - y\|_2^2 + \|z - z^\ast\|_2^2 \le \|z - y\|_2^2$, with strict inequality
  $\|z^\ast - y\|_2^2 < \|z - y\|_2^2$ whenever $z \neq z^\ast$;
- **(d)** Shift invariance along $a$: for every $\mu \in \mathbb{R}$,
  $\mathrm{simplexBoxDualClamp}(l, u, a, y + \mu a, \nu + \mu) = z^\ast$
  (certifying the reduced-gradient shift $g_z \mapsto g_z + \nu_k a$ in `fc_spg_lin_direction`);
- **(e)** Isotropic Spectral Projected Gradient (`SPG`) feasibility & strict descent: when
  $y_i = z_{k,i} - \gamma g_i$ ($\gamma > 0$) at feasible $z_k \in \mathcal{C}$, the step
  $d_{\mathrm{SPG}} = z^\ast - z_k$ satisfies $\sum_i a_i d_{\mathrm{SPG},i} = 0$, keeps the entire
  segment $z_k + \tau d_{\mathrm{SPG}} \in \mathcal{C}$ feasible for $\tau \in [0, 1]$, and obeys
  $\sum_i g_i d_{\mathrm{SPG},i} \le -\tfrac{1}{\gamma}\sum_i d_{\mathrm{SPG},i}^2$ (strictly $< 0$
  whenever $\exists i, z^\ast_i \neq z_{k,i}$). -/
theorem simplex_box_projection_variational {d : ℕ}
    (l u a y : Fin d → ℝ) (nu b : ℝ)
    (hl : ∀ i : Fin d, l i ≤ u i)
    (hsum_star : simplexBoxDualSum l u a y nu = b) :
    let z_star := simplexBoxDualClamp l u a y nu
    (∀ i : Fin d, l i ≤ z_star i ∧ z_star i ≤ u i) ∧
    (∀ z : Fin d → ℝ,
      (∀ i : Fin d, l i ≤ z i) →
      (∀ i : Fin d, z i ≤ u i) →
      (∑ i : Fin d, a i * z i) = b →
        (∑ i : Fin d, (y i - z_star i) * (z i - z_star i)) ≤ 0 ∧
        (∑ i : Fin d, (z_star i - y i) ^ 2) + (∑ i : Fin d, (z i - z_star i) ^ 2) ≤
          ∑ i : Fin d, (z i - y i) ^ 2 ∧
        (z ≠ z_star →
          (∑ i : Fin d, (z_star i - y i) ^ 2) < ∑ i : Fin d, (z i - y i) ^ 2)) ∧
    (∀ μ : ℝ, simplexBoxDualClamp l u a (fun i => y i + μ * a i) (nu + μ) = z_star) ∧
    (∀ (z_k g : Fin d → ℝ) (gamma : ℝ),
      (∀ i : Fin d, l i ≤ z_k i ∧ z_k i ≤ u i) →
      (∑ i : Fin d, a i * z_k i) = b →
      0 < gamma →
      (∀ i : Fin d, y i = z_k i - gamma * g i) →
        let d_spg : Fin d → ℝ := fun i => z_star i - z_k i
        (∑ i : Fin d, a i * d_spg i) = 0 ∧
        (∀ τ : ℝ, 0 ≤ τ → τ ≤ 1 →
          (∀ i : Fin d, l i ≤ z_k i + τ * d_spg i ∧ z_k i + τ * d_spg i ≤ u i) ∧
          (∑ i : Fin d, a i * (z_k i + τ * d_spg i)) = b) ∧
        (∑ i : Fin d, g i * d_spg i) ≤ -(1 / gamma) * (∑ i : Fin d, (d_spg i) ^ 2) ∧
        ((∃ i : Fin d, z_star i ≠ z_k i) → (∑ i : Fin d, g i * d_spg i) < 0)) := by
  dsimp only
  set z_star := simplexBoxDualClamp l u a y nu
  have h_box : ∀ i : Fin d, l i ≤ z_star i ∧ z_star i ≤ u i := fun i =>
    clamp_mem_Icc (hl i)
  have h_proj : ∀ z : Fin d → ℝ,
      (∀ i : Fin d, l i ≤ z i) →
      (∀ i : Fin d, z i ≤ u i) →
      (∑ i : Fin d, a i * z i) = b →
        (∑ i : Fin d, (y i - z_star i) * (z i - z_star i)) ≤ 0 ∧
        (∑ i : Fin d, (z_star i - y i) ^ 2) + (∑ i : Fin d, (z i - z_star i) ^ 2) ≤
          ∑ i : Fin d, (z i - y i) ^ 2 ∧
        (z ≠ z_star →
          (∑ i : Fin d, (z_star i - y i) ^ 2) < ∑ i : Fin d, (z i - y i) ^ 2) := by
    intro z hz_lo hz_hi hsum_z
    have h_vi : (∑ i : Fin d, (y i - z_star i) * (z i - z_star i)) ≤ 0 := by
      have h_split :
          (∑ i : Fin d, (y i - z_star i) * (z i - z_star i)) =
            (∑ i : Fin d, (y i - nu * a i - z_star i) * (z i - z_star i)) +
              nu * ((∑ i : Fin d, a i * z i) - (∑ i : Fin d, a i * z_star i)) := by
        have h_pt : ∀ i : Fin d,
            (y i - z_star i) * (z i - z_star i) =
              (y i - nu * a i - z_star i) * (z i - z_star i) +
                (nu * (a i * z i) - nu * (a i * z_star i)) := fun i => by ring
        simp_rw [h_pt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← mul_sub]
      rw [h_split, hsum_z, show (∑ i : Fin d, a i * z_star i) = b from hsum_star,
        sub_self, mul_zero, add_zero]
      refine Finset.sum_nonpos fun i _ => ?_
      exact clamp_variational_nonpos (hl i) (hz_lo i) (hz_hi i)
    have h_alg :
        (∑ i : Fin d, (z i - y i) ^ 2) =
          (∑ i : Fin d, (z_star i - y i) ^ 2) + (∑ i : Fin d, (z i - z_star i) ^ 2) -
            2 * (∑ i : Fin d, (y i - z_star i) * (z i - z_star i)) := by
      have h_pt : ∀ i : Fin d,
          (z i - y i) ^ 2 =
            ((z_star i - y i) ^ 2 + (z i - z_star i) ^ 2) -
              2 * ((y i - z_star i) * (z i - z_star i)) := fun i => by ring
      simp_rw [h_pt, Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum]
    refine ⟨h_vi, by linarith, fun h_ne => ?_⟩
    have ⟨i₀, hi₀⟩ : ∃ i : Fin d, z i ≠ z_star i := Function.ne_iff.mp h_ne
    have h_sq_pos : 0 < (z i₀ - z_star i₀) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hi₀)
    have h_sum_pos : 0 < ∑ i : Fin d, (z i - z_star i) ^ 2 :=
      lt_of_lt_of_le h_sq_pos
        (Finset.single_le_sum (fun j _ => sq_nonneg (z j - z_star j)) (Finset.mem_univ i₀))
    linarith
  have h_shift : ∀ μ : ℝ,
      simplexBoxDualClamp l u a (fun i => y i + μ * a i) (nu + μ) = z_star := by
    intro μ
    ext i
    unfold z_star simplexBoxDualClamp
    congr 2
    ring
  refine ⟨h_box, h_proj, h_shift, ?_⟩
  intro z_k g gamma hzk_box hzk_sum hgamma hy
  set d_spg : Fin d → ℝ := fun i => z_star i - z_k i
  have h_tang : (∑ i : Fin d, a i * d_spg i) = 0 := by
    have h_pt : ∀ i : Fin d, a i * d_spg i = a i * z_star i - a i * z_k i := fun i => by
      dsimp [d_spg]; ring
    simp_rw [h_pt, Finset.sum_sub_distrib,
      show (∑ i : Fin d, a i * z_star i) = b from hsum_star, hzk_sum, sub_self]
  have h_seg : ∀ τ : ℝ, 0 ≤ τ → τ ≤ 1 →
      (∀ i : Fin d, l i ≤ z_k i + τ * d_spg i ∧ z_k i + τ * d_spg i ≤ u i) ∧
      (∑ i : Fin d, a i * (z_k i + τ * d_spg i)) = b := by
    intro τ hτ0 hτ1
    refine ⟨fun i => ?_, ?_⟩
    · have h_conv : z_k i + τ * d_spg i = (1 - τ) * z_k i + τ * z_star i := by
        dsimp [d_spg]; ring
      rw [h_conv]
      constructor <;> nlinarith [(hzk_box i).1, (hzk_box i).2, (h_box i).1, (h_box i).2]
    · have h_pt : ∀ i : Fin d, a i * (z_k i + τ * d_spg i) = a i * z_k i + τ * (a i * d_spg i) :=
        fun i => by ring
      simp_rw [h_pt, Finset.sum_add_distrib, ← Finset.mul_sum, hzk_sum, h_tang, mul_zero, add_zero]
  have h_vi_zk := (h_proj z_k (fun i => (hzk_box i).1) (fun i => (hzk_box i).2) hzk_sum).1
  have h_spg_rel :
      (∑ i : Fin d, (d_spg i) ^ 2) + gamma * (∑ i : Fin d, g i * d_spg i) ≤ 0 := by
    have h_pt : ∀ i : Fin d,
        (y i - z_star i) * (z_k i - z_star i) = (d_spg i) ^ 2 + gamma * (g i * d_spg i) := by
      intro i
      rw [hy i]
      dsimp [d_spg]
      ring
    simp_rw [h_pt, Finset.sum_add_distrib, ← Finset.mul_sum] at h_vi_zk
    exact h_vi_zk
  have h_inv_pos : 0 < 1 / gamma := div_pos one_pos hgamma
  have h_scaled :
      (1 / gamma) * ((∑ i : Fin d, (d_spg i) ^ 2) + gamma * (∑ i : Fin d, g i * d_spg i)) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos (le_of_lt h_inv_pos) h_spg_rel
  have h_cancel :
      (1 / gamma) * (gamma * (∑ i : Fin d, g i * d_spg i)) = ∑ i : Fin d, g i * d_spg i := by
    rw [← mul_assoc, one_div_mul_cancel (ne_of_gt hgamma), one_mul]
  have h_desc : (∑ i : Fin d, g i * d_spg i) ≤ -(1 / gamma) * (∑ i : Fin d, (d_spg i) ^ 2) := by
    linarith
  refine ⟨h_tang, h_seg, h_desc, fun ⟨i₀, hi₀⟩ => ?_⟩
  have h_sq_pos : 0 < (d_spg i₀) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hi₀)
  have h_sum_pos : 0 < ∑ i : Fin d, (d_spg i) ^ 2 :=
    lt_of_lt_of_le h_sq_pos
      (Finset.single_le_sum (fun j _ => sq_nonneg (d_spg j)) (Finset.mem_univ i₀))
  have h_prod_pos : 0 < (1 / gamma) * (∑ i : Fin d, (d_spg i) ^ 2) := mul_pos h_inv_pos h_sum_pos
  linarith

/-- **Corollary (`coupled_spg_stiefel_strict_descent`)**:
Certifies the coupled constrained fallback direction (`fc_spg_lin_direction`) on
$\mathcal{C} \times \prod_{r=1}^R \mathrm{St}(n_r, p_r)$: pairing isotropic SPG
$d_{\mathrm{spg}} = P_{\mathcal{C}}(z_k - \gamma g_z) - z_k$ ($\gamma > 0$) with Stiefel ABB
steps $\eta_r = -\alpha_r g_{X,r}$ ($\alpha_r > 0$) unconditionally preserves $\mathcal{C}$ along
$\tau \in [0, 1]$ and guarantees strict joint descent whenever $d_{\mathrm{spg}} \neq 0$ or any
Stiefel Riemannian gradient $g_{X,r} \neq 0$. -/
theorem coupled_spg_stiefel_strict_descent {d R : ℕ} {n p : Fin R → ℕ}
    (l u a z_k gz : Fin d → ℝ) (b gamma : ℝ)
    (X gX : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ)
    (α_st : Fin R → ℝ)
    (hl : ∀ i : Fin d, l i ≤ u i)
    (hzk_box : ∀ i : Fin d, l i ≤ z_k i ∧ z_k i ≤ u i)
    (hzk_sum : (∑ i : Fin d, a i * z_k i) = b)
    (hX_orth : ∀ r : Fin R, (X r)ᵀ * (X r) = 1)
    (hgamma : 0 < gamma)
    (hα_pos : ∀ r : Fin R, 0 < α_st r) :
    ∃ nu : ℝ,
      let z_star := simplexBoxDualClamp l u a (fun i => z_k i - gamma * gz i) nu
      let d_spg : Fin d → ℝ := fun i => z_star i - z_k i
      let η : (r : Fin R) → Matrix (Fin (n r)) (Fin (p r)) ℝ := fun r => -(α_st r) • gX r
      simplexBoxDualSum l u a (fun i => z_k i - gamma * gz i) nu = b ∧
      (∀ τ : ℝ, 0 ≤ τ → τ ≤ 1 →
        (∀ i : Fin d, l i ≤ z_k i + τ * d_spg i ∧ z_k i + τ * d_spg i ≤ u i) ∧
        (∑ i : Fin d, a i * (z_k i + τ * d_spg i)) = b) ∧
      ((∑ i : Fin d, gz i * d_spg i) + (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) ≤
        -(1 / gamma) * (∑ i : Fin d, (d_spg i) ^ 2) -
          (∑ r : Fin R, α_st r * stiefelInner (1 / 2) (X r) (gX r) (gX r))) ∧
      (((∃ i : Fin d, z_star i ≠ z_k i) ∨ (∃ r : Fin R, gX r ≠ 0)) →
        (∑ i : Fin d, gz i * d_spg i) + (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (η r)) < 0) := by
  set y : Fin d → ℝ := fun i => z_k i - gamma * gz i
  have h_ex := simplex_box_dual_root_exists l u a y b hl
  have h_bounds : simplexBoxMinSum l u a ≤ b ∧ b ≤ simplexBoxMaxSum l u a :=
    h_ex.1.mp ⟨z_k, hzk_box, hzk_sum⟩
  rcases h_ex.2 h_bounds.1 h_bounds.2 with ⟨nu, hnu⟩
  refine ⟨nu, hnu, ?_⟩
  dsimp only
  have h_spg := (simplex_box_projection_variational l u a y nu b hl hnu).2.2.2
    z_k gz gamma hzk_box hzk_sum hgamma (fun _ => rfl)
  set z_star := simplexBoxDualClamp l u a y nu
  set d_spg : Fin d → ℝ := fun i => z_star i - z_k i
  have h_st_eq :
      (∑ r : Fin R, stiefelInner (1 / 2) (X r) (gX r) (-(α_st r) • gX r)) =
        -(∑ r : Fin R, α_st r * stiefelInner (1 / 2) (X r) (gX r) (gX r)) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [stiefelInner_symm, stiefelInner_smul_left, stiefelInner_symm]
    ring
  have h_box_nn : 0 ≤ (1 / gamma) * (∑ i : Fin d, (d_spg i) ^ 2) :=
    mul_nonneg (le_of_lt (div_pos one_pos hgamma)) (Finset.sum_nonneg fun i _ => sq_nonneg (d_spg i))
  have h_st_nn : 0 ≤ ∑ r : Fin R, α_st r * stiefelInner (1 / 2) (X r) (gX r) (gX r) :=
    Finset.sum_nonneg fun r _ =>
      mul_nonneg (le_of_lt (hα_pos r)) (stiefelInner_canonical_nonneg (X r) (gX r) (hX_orth r))
  refine ⟨h_spg.2.1, ?_, ?_⟩
  · rw [h_st_eq]
    linarith [h_spg.2.2.1]
  · intro hnz
    rw [h_st_eq]
    rcases hnz with hbox_nz | ⟨r₀, hgX₀⟩
    · linarith [h_spg.2.2.2 hbox_nz]
    · have h_st_ne : stiefelInner (1 / 2) (X r₀) (gX r₀) (gX r₀) ≠ 0 :=
        mt (stiefelInner_canonical_eq_zero_iff (X r₀) (gX r₀) (hX_orth r₀)).mp hgX₀
      have h_st_pos : 0 < α_st r₀ * stiefelInner (1 / 2) (X r₀) (gX r₀) (gX r₀) :=
        mul_pos (hα_pos r₀)
          (lt_of_le_of_ne (stiefelInner_canonical_nonneg (X r₀) (gX r₀) (hX_orth r₀)) (Ne.symm h_st_ne))
      have h_sum_pos : 0 < ∑ r : Fin R, α_st r * stiefelInner (1 / 2) (X r) (gX r) (gX r) :=
        lt_of_lt_of_le h_st_pos
          (Finset.single_le_sum
            (fun r _ => mul_nonneg (le_of_lt (hα_pos r))
              (stiefelInner_canonical_nonneg (X r) (gX r) (hX_orth r)))
            (Finset.mem_univ r₀))
      linarith [h_spg.2.2.1]

-- ============================================================================
-- Section 7: Rank-1 KKT Nullspace Quasi-Newton Descent & Shift Invariance (Theorem 7)
-- ============================================================================

/-- Homogeneous Rank-1 KKT Nullspace quasi-Newton search direction
$d_{\mathrm{KKT}}(g) = u - \frac{a^\top u}{a^\top w} w$ with $u = -H g$ and $w = H a$
(`ARCHITECTURE_SPEC.md` §2.9). -/
def kktNullspaceStep {n : ℕ}
    (H : Matrix (Fin n) (Fin n) ℝ) (a g : Fin n → ℝ) : Fin n → ℝ :=
  let u := -(H.mulVec g)
  let w := H.mulVec a
  u - (dotProduct a u / dotProduct a w) • w

/-- Inhomogeneous Rank-1 KKT Nullspace quasi-Newton search direction with boundary-snap target
$\delta_b \in \mathbb{R}$: $d_{\mathrm{KKT}}(g, \delta_b) = u - \frac{a^\top u - \delta_b}{a^\top w} w$. -/
def kktNullspaceStepInhom {n : ℕ}
    (H : Matrix (Fin n) (Fin n) ℝ) (a g : Fin n → ℝ) (delta_b : ℝ) : Fin n → ℝ :=
  let u := -(H.mulVec g)
  let w := H.mulVec a
  u - ((dotProduct a u - delta_b) / dotProduct a w) • w

/-- Symmetry of the bilinear form $x^\top (H y) = y^\top (H x)$ for any symmetric matrix
$H^\top = H$. -/
lemma dotProduct_mulVec_symm {n : ℕ}
    (H : Matrix (Fin n) (Fin n) ℝ) (hSymm : Hᵀ = H) (x y : Fin n → ℝ) :
    dotProduct x (H.mulVec y) = dotProduct y (H.mulVec x) := by
  calc
    dotProduct x (H.mulVec y) = dotProduct (H.vecMul x) y := Matrix.dotProduct_mulVec x H y
    _ = dotProduct (Hᵀ.mulVec x) y := by rw [Matrix.mulVec_transpose]
    _ = dotProduct (H.mulVec x) y := by rw [hSymm]
    _ = dotProduct y (H.mulVec x) := dotProduct_comm _ _

/-- **Theorem 7 (`rank1_kkt_nullspace_quasinewton_descent`)**:
For any symmetric positive-definite matrix $H \in \mathbb{R}^{n \times n}$ ($H^\top = H$ and
$\forall v \neq 0, v^\top H v > 0$) and nonzero constraint normal $a \in \mathbb{R}^n \setminus \{0\}$,
with witness $v = g - \frac{a^\top H g}{a^\top H a} a$:
- **(a)** Well-posedness: $a^\top H a > 0$;
- **(b)** Exact hyperplane tangency: $a^\top \mathrm{kktNullspaceStep}(H, a, g) = 0$, and for any
  inhomogeneous boundary-snap target $\delta_b \in \mathbb{R}$,
  $a^\top \mathrm{kktNullspaceStepInhom}(H, a, g, \delta_b) = \delta_b$;
- **(c)** Multiplier shift invariance:
  $\mathrm{kktNullspaceStep}(H, a, g + \mu a) = \mathrm{kktNullspaceStep}(H, a, g)$ for all $\mu \in \mathbb{R}$;
- **(d)** Homogeneous quadratic-form witness identity, symmetric rank-1 operator form, guaranteed
  descent, and strict non-collinear descent: $v^\top H v = -g^\top d_{\mathrm{KKT}}(g)$,
  $d_{\mathrm{KKT}}(g) = -\bigl(H g - \frac{w^\top g}{a^\top w} w\bigr)$,
  $g^\top d_{\mathrm{KKT}}(g) \le 0$, $g^\top d_{\mathrm{KKT}}(g) < 0$ whenever
  $\forall \mu \in \mathbb{R}, g \neq \mu a$, and $g^\top d_{\mathrm{KKT}}(g) = 0 \iff \exists \mu, g = \mu a$;
- **(e)** Exact inhomogeneous directional derivative identity & descent threshold:
  $g^\top \mathrm{kktNullspaceStepInhom}(H, a, g, \delta_b) = -v^\top H v + \frac{\delta_b}{a^\top H a}(g^\top H a)$,
  which is strictly negative iff $\frac{\delta_b}{a^\top H a}(g^\top H a) < v^\top H v$ (making
  explicit why `fc_solve_core` guards nonzero boundary snaps with a directional-derivative check
  and fallback to Theorem 6 coupled SPG). -/
theorem rank1_kkt_nullspace_quasinewton_descent {n : ℕ}
    (H : Matrix (Fin n) (Fin n) ℝ)
    (a g : Fin n → ℝ)
    (hSymm : Hᵀ = H)
    (hPosDef : ∀ v : Fin n → ℝ, v ≠ 0 → 0 < dotProduct v (H.mulVec v))
    (ha : a ≠ 0) :
    let v : Fin n → ℝ :=
      g - (dotProduct a (H.mulVec g) / dotProduct a (H.mulVec a)) • a
    (0 < dotProduct a (H.mulVec a)) ∧
    (dotProduct a (kktNullspaceStep H a g) = 0 ∧
      ∀ delta_b : ℝ, dotProduct a (kktNullspaceStepInhom H a g delta_b) = delta_b) ∧
    (∀ μ : ℝ, kktNullspaceStep H a (g + μ • a) = kktNullspaceStep H a g) ∧
    (dotProduct v (H.mulVec v) = -dotProduct g (kktNullspaceStep H a g) ∧
      kktNullspaceStep H a g =
        -(H.mulVec g - (dotProduct (H.mulVec a) g / dotProduct a (H.mulVec a)) • H.mulVec a) ∧
      dotProduct g (kktNullspaceStep H a g) ≤ 0 ∧
      ((∀ μ : ℝ, g ≠ μ • a) → dotProduct g (kktNullspaceStep H a g) < 0) ∧
      (dotProduct g (kktNullspaceStep H a g) = 0 ↔ ∃ μ : ℝ, g = μ • a)) ∧
    (∀ delta_b : ℝ,
      dotProduct g (kktNullspaceStepInhom H a g delta_b) =
        -dotProduct v (H.mulVec v) +
          (delta_b / dotProduct a (H.mulVec a)) * dotProduct g (H.mulVec a) ∧
      (dotProduct g (kktNullspaceStepInhom H a g delta_b) < 0 ↔
        (delta_b / dotProduct a (H.mulVec a)) * dotProduct g (H.mulVec a) <
          dotProduct v (H.mulVec v))) := by
  dsimp only
  set w := H.mulVec a with hw_def
  have haw_pos : 0 < dotProduct a w := hPosDef a ha
  have haw_ne : dotProduct a w ≠ 0 := ne_of_gt haw_pos
  have h_inhom : ∀ (g' : Fin n → ℝ) (delta_b : ℝ),
      dotProduct a (kktNullspaceStepInhom H a g' delta_b) = delta_b := by
    intro g' delta_b
    unfold kktNullspaceStepInhom
    dsimp only
    rw [← hw_def, dotProduct_sub, dotProduct_smul, smul_eq_mul, div_mul_cancel₀ _ haw_ne]
    ring
  have h_tang : dotProduct a (kktNullspaceStep H a g) = 0 := by
    have h0 := h_inhom g 0
    simpa only [kktNullspaceStep, kktNullspaceStepInhom, sub_zero] using h0
  have h_shift : ∀ μ : ℝ, kktNullspaceStep H a (g + μ • a) = kktNullspaceStep H a g := by
    intro μ
    unfold kktNullspaceStep
    dsimp only
    rw [← hw_def]
    have hu_shift : -(H.mulVec (g + μ • a)) = -(H.mulVec g) - μ • w := by
      rw [Matrix.mulVec_add, Matrix.mulVec_smul, ← hw_def, neg_add_rev]
      abel
    have h_dot_shift :
        dotProduct a (-(H.mulVec (g + μ • a))) / dotProduct a w =
          dotProduct a (-(H.mulVec g)) / dotProduct a w - μ := by
      rw [hu_shift, dotProduct_sub, dotProduct_smul, smul_eq_mul, sub_div,
        mul_div_cancel_right₀ μ haw_ne]
    rw [h_dot_shift, hu_shift]
    ext i
    simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
    ring
  set α : ℝ := dotProduct a (H.mulVec g) / dotProduct a w with hα_def
  set v : Fin n → ℝ := g - α • a with hv_def
  set d : Fin n → ℝ := kktNullspaceStep H a g with hd_def
  have hd_eq_neg_Hv : d = -(H.mulVec v) := by
    have h_au : dotProduct a (-(H.mulVec g)) / dotProduct a w = -α := by
      rw [dotProduct_neg, neg_div, ← hα_def]
    rw [hd_def, kktNullspaceStep, ← hw_def, h_au, hv_def, Matrix.mulVec_sub, Matrix.mulVec_smul, ← hw_def]
    ext i
    simp only [Pi.sub_apply, Pi.neg_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have h_aHv : dotProduct a (H.mulVec v) = 0 := by
    rw [hv_def, Matrix.mulVec_sub, Matrix.mulVec_smul, ← hw_def,
      dotProduct_sub, dotProduct_smul, smul_eq_mul, hα_def, div_mul_cancel₀ _ haw_ne, sub_self]
  have h_quad_id : dotProduct v (H.mulVec v) = -dotProduct g d := by
    rw [hv_def, sub_dotProduct, smul_dotProduct, smul_eq_mul, h_aHv, mul_zero, sub_zero,
      hd_eq_neg_Hv, dotProduct_neg, neg_neg]
  have h_sym_op :
      d = -(H.mulVec g - (dotProduct w g / dotProduct a w) • w) := by
    have h_ag_wg : dotProduct a (H.mulVec g) = dotProduct w g := by
      rw [hw_def, dotProduct_mulVec_symm H hSymm a g, dotProduct_comm]
    have h_au : dotProduct a (-(H.mulVec g)) / dotProduct a w = -(dotProduct w g / dotProduct a w) := by
      rw [dotProduct_neg, h_ag_wg, neg_div]
    rw [hd_def, kktNullspaceStep, ← hw_def, h_au]
    ext i
    simp only [Pi.sub_apply, Pi.neg_apply, Pi.smul_apply, smul_eq_mul]
    ring
  have hv_nonneg : 0 ≤ dotProduct v (H.mulVec v) := by
    by_cases hv0 : v = 0
    · rw [hv0, zero_dotProduct]
    · exact le_of_lt (hPosDef v hv0)
  have h_desc_nonpos : dotProduct g d ≤ 0 := by linarith
  have h_strict : (∀ μ : ℝ, g ≠ μ • a) → dotProduct g d < 0 := by
    intro h_noncol
    have hv_ne : v ≠ 0 := by
      intro hv0
      exact h_noncol α (sub_eq_zero.mp hv0)
    have hv_pos : 0 < dotProduct v (H.mulVec v) := hPosDef v hv_ne
    linarith
  have h_zero_iff : dotProduct g d = 0 ↔ ∃ μ : ℝ, g = μ • a := by
    constructor
    · intro hg0
      by_cases hv0 : v = 0
      · exact ⟨α, sub_eq_zero.mp hv0⟩
      · have hv_pos : 0 < dotProduct v (H.mulVec v) := hPosDef v hv0
        linarith
    · rintro ⟨μ, rfl⟩
      have hd0 : d = 0 := by
        have h_s := h_shift (-μ)
        have h_cancel : μ • a + (-μ) • a = (0 : Fin n → ℝ) := by module
        rw [h_cancel] at h_s
        rw [← h_s, kktNullspaceStep]
        rw [Matrix.mulVec_zero, neg_zero, dotProduct_zero, zero_div, zero_smul, sub_zero]
      rw [hd0, dotProduct_zero]
  have h_inhom_desc : ∀ delta_b : ℝ,
      dotProduct g (kktNullspaceStepInhom H a g delta_b) =
        -dotProduct v (H.mulVec v) + (delta_b / dotProduct a w) * dotProduct g w ∧
      (dotProduct g (kktNullspaceStepInhom H a g delta_b) < 0 ↔
        (delta_b / dotProduct a w) * dotProduct g w < dotProduct v (H.mulVec v)) := by
    intro delta_b
    have h_split :
        kktNullspaceStepInhom H a g delta_b = d + (delta_b / dotProduct a w) • w := by
      rw [hd_def, kktNullspaceStepInhom, kktNullspaceStep, ← hw_def, sub_div]
      ext i
      simp only [Pi.add_apply, Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
      ring
    have h_eq : dotProduct g (kktNullspaceStepInhom H a g delta_b) =
        -dotProduct v (H.mulVec v) + (delta_b / dotProduct a w) * dotProduct g w := by
      rw [h_split, dotProduct_add, dotProduct_smul, smul_eq_mul, h_quad_id, neg_neg]
    refine ⟨h_eq, ?_⟩
    rw [h_eq]
    constructor <;> intro h <;> linarith
  exact ⟨haw_pos, ⟨h_tang, h_inhom g⟩, h_shift,
    ⟨h_quad_id, h_sym_op, h_desc_nonpos, h_strict, h_zero_iff⟩, h_inhom_desc⟩

end

end Fortchuck
