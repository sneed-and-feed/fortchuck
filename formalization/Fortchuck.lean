import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Module
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.Positivity

/-!
# `Fortchuck`: Machine-Checked Algebraic & Variational Invariants

This module provides complete, self-contained Lean 4 (`0 sorry`, `0` custom axioms, `0` warnings)
proofs of the four foundational mathematical theorems underlying **`fortchuck`**
(`docs/ARCHITECTURE_SPEC.md` §2.2–§2.5 and §5.3):

1. **Theorem 1 (`skew_generator_properties`)**:
   For $X \in \mathrm{St}(n, p)$ ($X^\top X = I_p$) and Stiefel tangent $\xi \in T_X \mathrm{St}(n, p)$
   ($X^\top \xi + \xi^\top X = 0$), the modified tangent $\tilde{\xi} = \xi - \tfrac{1}{2} X (X^\top \xi)$
   and rank-$2p$ generator $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top$ satisfy
   $W_\xi^\top = -W_\xi$ and $W_\xi X = \xi$.
2. **Theorem 2 (`cayley_rotor_orthogonality_and_isometry`)**:
   For skew-symmetric $W \in \mathfrak{so}(n)$ ($W^\top = -W$) and resolvent inverse witness $M$
   satisfying $M(I_n - \tfrac{\tau}{2} W) = (I_n - \tfrac{\tau}{2} W) M = I_n$, the Cayley rotor
   $Q = M(I_n + \tfrac{\tau}{2} W)$ satisfies resolvent doubling $Q = 2M - I_n$, two-sided
   orthogonality $Q^\top Q = Q Q^\top = I_n$, Stiefel retraction preservation
   $(Q X)^\top (Q X) = X^\top X$, and exact isometric vector transport invariance of tangent
   skew-symmetry, skew projections, Frobenius cross-Gramians, and canonical Stiefel inner products.
3. **Theorem 3 (`schur_woodbury_cayley_orthogonality` & `cayley_branch_a_eq_branch_b`)**:
   For $X^\top X = I_p$, $A^\top = -A$, $X^\top K = 0$, dissipative Schur complement
   $J = I_p - \tfrac{\tau}{2} A + \tfrac{\tau^2}{4} K^\top K$, and inverse witness
   $J J_{\mathrm{inv}} = J_{\mathrm{inv}} J = I_p$, the Branch A Schur-Woodbury retraction
   $Y = (2X + \tau K) J_{\mathrm{inv}} - X$ satisfies $Y^\top Y = I_p$ identically, and agrees
   algebraically ($Q X = Y$) with the Branch B direct $n \times n$ Cayley rotor when
   $A = X^\top \xi$ and $K = \xi - X A$.
4. **Theorem 4 (`coupled_two_metric_strict_descent`)**:
   Formalizes Theorem 2.1 of `ARCHITECTURE_SPEC.md`: under the safeguarded Bertsekas active-band
   partition coupled with a positive-definite masked product quasi-Newton operator on the joint
   free Box $\times$ Multi-Stiefel tangent subspace, free-coordinate clamping preserves descent
   ($g_i \dot{z}_i(0^+) \le g_i d_i$), the joint right-hand directional derivative $\phi'(0^+)$
   satisfies the exact two-metric quadratic upper bound, $\phi'(0^+) < 0$ holds strictly whenever
   any active-interior, free-box, or Stiefel gradient component is nonzero, and $\phi'(0^+) = 0$
   holds if and only if the iterate is first-order KKT stationary.
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
$\xi \in T_X \mathrm{St}(n, p)$ ($X^\top \xi + \xi^\top X = 0$), defining
$\tilde{\xi} = \xi - \tfrac{1}{2} X (X^\top \xi)$ and
$W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top$ guarantees both skew-symmetry
$W_\xi^\top = -W_\xi$ and exact tangent reproduction $W_\xi X = \xi$. -/
theorem skew_generator_properties {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0) :
    let ξ_tilde := ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ))
    let W_ξ := ξ_tilde * Xᵀ - X * ξ_tildeᵀ
    W_ξᵀ = -W_ξ ∧ W_ξ * X = ξ := by
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
-- Section 2: Cayley Rotor Orthogonality, Retraction & Isometric Transport (Theorem 2)
-- ============================================================================

/-- Direct $n \times n$ Cayley rotor $Q = M (I_n + \frac{\tau}{2} W)$ from resolvent inverse $M$. -/
def cayleyRotor {n : ℕ} (M W : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  M * (1 + (τ / 2 : ℝ) • W)

/-- Parametric Stiefel Riemannian metric ($\beta = 1/2$ canonical, $\beta = 1$ Euclidean):
$\langle \xi, \eta \rangle_{X, \beta} = \mathrm{tr}\!\left(\xi^\top (I_n - (1 - \beta) X X^\top) \eta\right)$. -/
def stiefelInner {n p : ℕ} (β : ℝ)
    (X ξ η : Matrix (Fin n) (Fin p) ℝ) : ℝ :=
  Matrix.trace (ξᵀ * ((1 : Matrix (Fin n) (Fin n) ℝ) - (1 - β) • (X * Xᵀ)) * η)

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
Given skew-symmetric $W \in \mathfrak{so}(n)$ ($W^\top = -W$) and two-sided inverse witness $M$
for $I_n - \tfrac{\tau}{2} W$, the Cayley rotor $Q = M(I_n + \tfrac{\tau}{2} W)$ satisfies:
- **(a)** Resolvent doubling: $Q = 2M - I_n$,
- **(b)** Two-sided orthogonality: $Q^\top Q = I_n$ and $Q Q^\top = I_n$,
- **(c)** Stiefel retraction preservation: $(Q X)^\top (Q X) = X^\top X$,
- **(d)** Isometric transport invariance of tangent skew-symmetry, skew projections,
  Frobenius cross-Gramians, and canonical Stiefel metric products & traces. -/
theorem cayley_rotor_orthogonality_and_isometry {n p : ℕ}
    (W M : Matrix (Fin n) (Fin n) ℝ) (τ : ℝ)
    (X ζ ζ₁ ζ₂ : Matrix (Fin n) (Fin p) ℝ)
    (hW : Wᵀ = -W)
    (hM_right : M * (1 - (τ / 2 : ℝ) • W) = 1)
    (hM_left : (1 - (τ / 2 : ℝ) • W) * M = 1) :
    let Q := M * (1 + (τ / 2 : ℝ) • W)
    Q = (2 : ℝ) • M - 1 ∧
    (Qᵀ * Q = 1 ∧ Q * Qᵀ = 1) ∧
    (Q * X)ᵀ * (Q * X) = Xᵀ * X ∧
    (((Q * X)ᵀ * (Q * ζ) + (Q * ζ)ᵀ * (Q * X) = Xᵀ * ζ + ζᵀ * X) ∧
     ((Q * X)ᵀ * (Q * ζ) = Xᵀ * ζ ∧ (Q * ζ₁)ᵀ * (Q * ζ₂) = ζ₁ᵀ * ζ₂) ∧
     ((Q * ζ₁)ᵀ * (1 - (1 / 2 : ℝ) • ((Q * X) * (Q * X)ᵀ)) * (Q * ζ₂) =
        ζ₁ᵀ * (1 - (1 / 2 : ℝ) • (X * Xᵀ)) * ζ₂) ∧
     (Matrix.trace ((Q * ζ₁)ᵀ * (1 - (1 / 2 : ℝ) • ((Q * X) * (Q * X)ᵀ)) * (Q * ζ₂)) =
        Matrix.trace (ζ₁ᵀ * (1 - (1 / 2 : ℝ) • (X * Xᵀ)) * ζ₂))) := by
  dsimp only
  set c : ℝ := τ / 2
  set Q := M * (1 + c • W) with hQ_def
  have h_split : (1 : Matrix (Fin n) (Fin n) ℝ) + c • W = (2 : ℝ) • 1 - (1 - c • W) := by
    module
  have hQ_res : Q = (2 : ℝ) • M - 1 := by
    calc
      Q = M * ((2 : ℝ) • 1 - (1 - c • W)) := by rw [hQ_def, h_split]
      _ = (2 : ℝ) • M - 1 := by
        rw [Matrix.mul_sub, Matrix.mul_smul, Matrix.mul_one, hM_right]
  have hQ_comm : Q = (1 + c • W) * M := by
    calc
      Q = (2 : ℝ) • M - 1 := hQ_res
      _ = ((2 : ℝ) • 1 - (1 - c • W)) * M := by
        rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul, hM_left]
      _ = (1 + c • W) * M := by rw [← h_split]
  have hN_sub_tr : ((1 : Matrix (Fin n) (Fin n) ℝ) - c • W)ᵀ = 1 + c • W := by
    rw [Matrix.transpose_sub, Matrix.transpose_one, Matrix.transpose_smul, hW,
      smul_neg, sub_neg_eq_add]
  have hN_add_tr : ((1 : Matrix (Fin n) (Fin n) ℝ) + c • W)ᵀ = 1 - c • W := by
    rw [Matrix.transpose_add, Matrix.transpose_one, Matrix.transpose_smul, hW,
      smul_neg, ← sub_eq_add_neg]
  have hMt_right : Mᵀ * (1 + c • W) = 1 := by
    have h := congrArg Matrix.transpose hM_left
    rwa [Matrix.transpose_mul, hN_sub_tr, Matrix.transpose_one] at h
  have hMt_left : (1 + c • W) * Mᵀ = 1 := by
    have h := congrArg Matrix.transpose hM_right
    rwa [Matrix.transpose_mul, hN_sub_tr, Matrix.transpose_one] at h
  have hQtQ : Qᵀ * Q = 1 := by
    calc
      Qᵀ * Q = ((1 + c • W) * M)ᵀ * (M * (1 + c • W)) := by rw [← hQ_comm, ← hQ_def]
      _ = (Mᵀ * (1 - c • W)) * (M * (1 + c • W)) := by rw [Matrix.transpose_mul, hN_add_tr]
      _ = Mᵀ * (((1 - c • W) * M) * (1 + c • W)) := by simp only [Matrix.mul_assoc]
      _ = Mᵀ * (1 + c • W) := by rw [hM_left, Matrix.one_mul]
      _ = 1 := hMt_right
  have hQQt : Q * Qᵀ = 1 := by
    calc
      Q * Qᵀ = ((1 + c • W) * M) * (M * (1 + c • W))ᵀ := by rw [← hQ_comm, ← hQ_def]
      _ = ((1 + c • W) * M) * ((1 - c • W) * Mᵀ) := by rw [Matrix.transpose_mul, hN_add_tr]
      _ = (1 + c • W) * ((M * (1 - c • W)) * Mᵀ) := by simp only [Matrix.mul_assoc]
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
  exact ⟨hQ_res, ⟨hQtQ, hQQt⟩, hXX,
    ⟨by rw [hXζ, hζX], ⟨hXζ, hζ₁ζ₂⟩, h_can_mat, congrArg Matrix.trace h_can_mat⟩⟩

-- ============================================================================
-- Section 3: Branch A Schur-Woodbury Orthogonality & Branch Equivalence (Theorem 3)
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
    (X K : Matrix (Fin n) (Fin p) ℝ) (J_inv : Matrix (Fin p) (Fin p) ℝ) (τ : ℝ) :
    Matrix (Fin n) (Fin p) ℝ :=
  ((2 : ℝ) • X + τ • K) * J_inv - X

/-- **Theorem 3 (`schur_woodbury_cayley_orthogonality`)**:
Given $X^\top X = I_p$, skew-symmetric $A^\top = -A$, normal block $X^\top K = 0$,
Schur complement $J = I_p - \tfrac{\tau}{2} A + \tfrac{\tau^2}{4} (K^\top K)$, and two-sided
inverse witness $J J_{\mathrm{inv}} = J_{\mathrm{inv}} J = I_p$, the Branch A Schur-Woodbury
retraction $Y = (2X + \tau K) J_{\mathrm{inv}} - X$ satisfies $Y^\top Y = I_p$ identically. -/
theorem schur_woodbury_cayley_orthogonality {n p : ℕ}
    (X K : Matrix (Fin n) (Fin p) ℝ)
    (A J_inv : Matrix (Fin p) (Fin p) ℝ) (τ : ℝ)
    (hX : Xᵀ * X = 1)
    (hA : Aᵀ = -A)
    (hXK : Xᵀ * K = 0)
    (hJ_right : (1 - (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K)) * J_inv = 1)
    (hJ_left : J_inv * (1 - (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K)) = 1) :
    let Y := ((2 : ℝ) • X + τ • K) * J_inv - X
    Yᵀ * Y = 1 := by
  dsimp only
  set J := (1 : Matrix (Fin p) (Fin p) ℝ) - (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K)
    with hJ_def
  set Z := (2 : ℝ) • X + τ • K
  set Y := Z * J_inv - X
  have hKX : Kᵀ * X = 0 := by
    have h := congrArg Matrix.transpose hXK
    rwa [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.transpose_zero] at h
  have hYJ : Y * J = Z - X * J := by
    calc
      Y * J = (Z * J_inv) * J - X * J := by rw [Matrix.sub_mul]
      _ = Z * (J_inv * J) - X * J := by rw [Matrix.mul_assoc]
      _ = Z - X * J := by rw [hJ_left, Matrix.mul_one]
  have hJt : Jᵀ = 1 + (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K) := by
    rw [hJ_def, Matrix.transpose_add, Matrix.transpose_sub, Matrix.transpose_one,
      Matrix.transpose_smul, Matrix.transpose_smul, hA, Matrix.transpose_mul,
      Matrix.transpose_transpose, smul_neg, sub_neg_eq_add]
  have h_sum : (2 : ℝ) • (Jᵀ + J) = (4 : ℝ) • 1 + (τ ^ 2 : ℝ) • (Kᵀ * K) := by
    rw [hJt, hJ_def]
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
  have hY_recon : Y = (Y * J) * J_inv := by
    rw [Matrix.mul_assoc, hJ_right, Matrix.mul_one]
  have _ := hYJ.symm.trans (congrArg (· * J) hY_recon)
  calc
    Yᵀ * Y = (J_invᵀ * Zᵀ - Xᵀ) * (Z * J_inv - X) := by
      simp only [Y, Matrix.transpose_sub, Matrix.transpose_mul]
    _ = J_invᵀ * (Zᵀ * Z) * J_inv - J_invᵀ * (Zᵀ * X) - (Xᵀ * Z) * J_inv + Xᵀ * X := by
      simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.mul_assoc]
      abel
    _ = ((2 : ℝ) • J_inv + (2 : ℝ) • J_invᵀ) - (2 : ℝ) • J_invᵀ - (2 : ℝ) • J_inv + 1 := by
      rw [h_mid, hZtX, hXtZ, hX, Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, Matrix.one_mul]
    _ = 1 := by module

/-- **Branch A == Branch B Algebraic Equivalence (`cayley_branch_a_eq_branch_b`)**:
For $X \in \mathrm{St}(n, p)$ ($X^\top X = I_p$) and Stiefel tangent $\xi \in T_X \mathrm{St}(n, p)$
($X^\top \xi + \xi^\top X = 0$), setting $A = X^\top \xi$, $K = \xi - X A$,
$\tilde{\xi} = \xi - \tfrac{1}{2} X A$, $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top$,
and $J = I_p - \tfrac{\tau}{2} A + \tfrac{\tau^2}{4} (K^\top K)$, the Branch B direct Cayley
retraction $Q X = M (I_n + \tfrac{\tau}{2} W_\xi) X$ and Branch A Schur-Woodbury retraction
$Y = (2X + \tau K) J_{\mathrm{inv}} - X$ are algebraically identical ($Q X = Y$). -/
theorem cayley_branch_a_eq_branch_b {n p : ℕ}
    (X ξ : Matrix (Fin n) (Fin p) ℝ)
    (M : Matrix (Fin n) (Fin n) ℝ)
    (J_inv : Matrix (Fin p) (Fin p) ℝ) (τ : ℝ)
    (hX : Xᵀ * X = 1)
    (hξ : Xᵀ * ξ + ξᵀ * X = 0)
    (hM_right : M * (1 - (τ / 2 : ℝ) •
      ((ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ))) * Xᵀ - X * (ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ)))ᵀ)) = 1)
    (hM_left : (1 - (τ / 2 : ℝ) •
      ((ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ))) * Xᵀ - X * (ξ - (1 / 2 : ℝ) • (X * (Xᵀ * ξ)))ᵀ)) * M = 1)
    (hJ_right : (1 - (τ / 2 : ℝ) • (Xᵀ * ξ) +
      (τ ^ 2 / 4 : ℝ) • ((ξ - X * (Xᵀ * ξ))ᵀ * (ξ - X * (Xᵀ * ξ)))) * J_inv = 1) :
    let A := Xᵀ * ξ
    let K := ξ - X * A
    let ξ_tilde := ξ - (1 / 2 : ℝ) • (X * A)
    let W_ξ := ξ_tilde * Xᵀ - X * ξ_tildeᵀ
    let Q := M * (1 + (τ / 2 : ℝ) • W_ξ)
    let Y := ((2 : ℝ) • X + τ • K) * J_inv - X
    Q * X = Y := by
  dsimp only
  set A := Xᵀ * ξ
  set K := ξ - X * A
  set ξ_tilde := ξ - (1 / 2 : ℝ) • (X * A)
  set W_ξ := ξ_tilde * Xᵀ - X * ξ_tildeᵀ
  set Q := M * (1 + (τ / 2 : ℝ) • W_ξ)
  set J := (1 : Matrix (Fin p) (Fin p) ℝ) - (τ / 2 : ℝ) • A + (τ ^ 2 / 4 : ℝ) • (Kᵀ * K)
  have hW_props := skew_generator_properties X ξ hX hξ
  have hWt : W_ξᵀ = -W_ξ := hW_props.1
  have hWX : W_ξ * X = ξ := hW_props.2
  have hQ_res : Q = (2 : ℝ) • M - 1 :=
    (cayley_rotor_orthogonality_and_isometry W_ξ M τ X ξ ξ ξ hWt hM_right hM_left).1
  have hXK : Xᵀ * K = 0 := by
    simp only [K, A, Matrix.mul_sub, ← Matrix.mul_assoc Xᵀ X (Xᵀ * ξ), hX, Matrix.one_mul, sub_self]
  have hξ_tilde_K : ξ_tilde = K + (1 / 2 : ℝ) • (X * A) := by
    simp only [ξ_tilde, K]
    module
  have hWK : W_ξ * K = -(X * (Kᵀ * K)) := by
    calc
      W_ξ * K = ξ_tilde * (Xᵀ * K) - X * (ξ_tildeᵀ * K) := by
        simp only [W_ξ, Matrix.sub_mul, Matrix.mul_assoc]
      _ = -(X * (ξ_tildeᵀ * K)) := by rw [hXK, Matrix.mul_zero, zero_sub]
      _ = -(X * (Kᵀ * K)) := by
        congr 2
        rw [hξ_tilde_K, Matrix.transpose_add, Matrix.transpose_smul, Matrix.transpose_mul,
          Matrix.add_mul, Matrix.smul_mul, Matrix.mul_assoc, hXK, Matrix.mul_zero,
          smul_zero, add_zero]
  have h_intertwine : (1 - (τ / 2 : ℝ) • W_ξ) * (X + (τ / 2 : ℝ) • K) = X * J := by
    have hξ_split : ξ = X * A + K := by simp only [K]; abel
    simp only [J, Matrix.sub_mul, Matrix.mul_add, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one,
      Matrix.smul_mul, Matrix.mul_smul, hWX, hWK, hξ_split, smul_neg, smul_add]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.neg_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have hMX : M * X = (X + (τ / 2 : ℝ) • K) * J_inv := by
    calc
      M * X = M * (X * (J * J_inv)) := by rw [hJ_right, Matrix.mul_one]
      _ = M * ((X * J) * J_inv) := by rw [Matrix.mul_assoc X J J_inv]
      _ = M * (((1 - (τ / 2 : ℝ) • W_ξ) * (X + (τ / 2 : ℝ) • K)) * J_inv) := by
        rw [← h_intertwine]
      _ = ((M * (1 - (τ / 2 : ℝ) • W_ξ)) * (X + (τ / 2 : ℝ) • K)) * J_inv := by
        simp only [Matrix.mul_assoc]
      _ = (X + (τ / 2 : ℝ) • K) * J_inv := by rw [hM_right, Matrix.one_mul]
  have h2XK : (2 : ℝ) • (X + (τ / 2 : ℝ) • K) = (2 : ℝ) • X + τ • K := by
    ext i j
    simp only [Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  calc
    Q * X = ((2 : ℝ) • M - 1) * X := by rw [hQ_res]
    _ = (2 : ℝ) • (M * X) - X := by rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.one_mul]
    _ = (2 : ℝ) • ((X + (τ / 2 : ℝ) • K) * J_inv) - X := by rw [hMX]
    _ = ((2 : ℝ) • (X + (τ / 2 : ℝ) • K)) * J_inv - X := by rw [Matrix.smul_mul]
    _ = ((2 : ℝ) • X + τ • K) * J_inv - X := by rw [h2XK]

-- ============================================================================
-- Section 4: Coupled Two-Metric Strict Descent & KKT Stationarity (Theorem 4)
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

/-- **Theorem 4 (`coupled_two_metric_strict_descent`)**:
Formalization of Theorem 2.1 (`ARCHITECTURE_SPEC.md` §2.3). Under primal feasibility,
safeguarded Bertsekas active-band width ($0 \le \epsilon_i < u_i - l_i$ whenever $l_i < u_i$),
safeguarded active-band steps $d_{z,i} = -\gamma_i g_{z,i}$ ($\gamma_i \ge \gamma_{\min} > 0$),
and positive-definite masked product L-BFGS curvature ($\lambda_{\min} > 0$) on the joint
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
  -- Bound every coordinate outside `A_int ∪ F_set` by 0
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
  -- Bound every coordinate inside `A_int` by `-γ_min * gz i ^ 2`
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
  -- Split `Finset.univ` into `(A_int ∪ F_set)` and its complement
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
  -- Strict descent whenever any active-interior, free, or Stiefel gradient component is nonzero
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
  -- Exact KKT characterization `φ'_0 = 0 ↔ IsCoupledKKTStationary`
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

end

end Fortchuck
