module fortchuck_stiefel
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: &
    dp, i32, i64, FC_HUGE, FC_ERR_INVALID_DIM, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN
  use fortchuck_linalg, only: &
    fc_gemm_nn, fc_gemm_tn, fc_gemm_nt, fc_syrk_tn, &
    fc_frob_inner, fc_lu_factor, fc_lu_solve, fc_lu_solve_trans, fc_lu_solve_right
  implicit none (type, external)
  private

  real(dp), parameter :: CAYLEY_PIV_TOL = 1.0e-12_dp

  public :: fc_stiefel_offsets
  public :: fc_stiefel_rgrad
  public :: fc_stiefel_proj_tangent
  public :: fc_stiefel_inner
  public :: fc_stiefel_orth_err
  public :: fc_stiefel_orth_mat
  public :: fc_cayley_setup
  public :: fc_cayley_retract
  public :: fc_cayley_transport_vec
  public :: fc_cayley_setup_branch_a
  public :: fc_cayley_retract_branch_a
  public :: fc_cayley_transport_branch_a
  public :: fc_cayley_setup_branch_b
  public :: fc_cayley_retract_branch_b
  public :: fc_cayley_transport_branch_b

contains

  !---------------------------------------------------------------------------
  ! Compute 1-based packed offsets and total buffer lengths across R Stiefel
  ! factors St(n_r, p_r), matching ARCHITECTURE_SPEC.md section 3.4:
  !   c_r = 2*n_r*p_r + 3*p_r^2 (if 2*p_r < n_r) else 2*n_r^2
  !   q_r = p_r                 (if 2*p_r < n_r) else n_r
  !---------------------------------------------------------------------------
  pure subroutine fc_stiefel_offsets(R, n_arr, p_arr, &
      x_off, skew_off, cayley_off, ipiv_off, &
      n_st_tot, n_skew_tot, n_cayley_tot, n_ipiv_tot, status)
    integer(i32), intent(in)  :: R
    integer(i32), intent(in)  :: n_arr(*), p_arr(*)
    integer(i32), intent(out) :: x_off(*), skew_off(*), cayley_off(*), ipiv_off(*)
    integer(i32), intent(out) :: n_st_tot, n_skew_tot, n_cayley_tot, n_ipiv_tot
    integer(i32), intent(out) :: status

    integer(i32) :: r_idx, nr, pr
    integer(i64) :: st_acc, skew_acc, cay_acc, piv_acc, nr64, pr64, cr64, qr64, lim64

    n_st_tot     = 0_i32
    n_skew_tot   = 0_i32
    n_cayley_tot = 0_i32
    n_ipiv_tot   = 0_i32

    if (R < 0_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    x_off(1)      = 1_i32
    skew_off(1)   = 1_i32
    cayley_off(1) = 1_i32
    ipiv_off(1)   = 1_i32
    status        = 0_i32
    if (R == 0_i32) return

    st_acc   = 0_i64
    skew_acc = 0_i64
    cay_acc  = 0_i64
    piv_acc  = 0_i64
    lim64    = int(huge(0_i32) - 1_i32, i64)

    do r_idx = 1_i32, R
      nr = n_arr(r_idx)
      pr = p_arr(r_idx)
      if (nr < 1_i32 .or. pr < 1_i32 .or. pr > nr) then
        status = FC_ERR_INVALID_DIM
        return
      end if

      nr64 = int(nr, i64)
      pr64 = int(pr, i64)
      if (2_i64 * pr64 < nr64) then
        cr64 = 2_i64 * nr64 * pr64 + 3_i64 * pr64 * pr64
        qr64 = pr64
      else
        cr64 = 2_i64 * nr64 * nr64
        qr64 = nr64
      end if

      st_acc   = st_acc   + nr64 * pr64
      skew_acc = skew_acc + pr64 * pr64
      cay_acc  = cay_acc  + cr64
      piv_acc  = piv_acc  + qr64

      if (st_acc > lim64 .or. skew_acc > lim64 .or. &
          cay_acc > lim64 .or. piv_acc > lim64) then
        status = FC_ERR_INVALID_DIM
        return
      end if

      x_off(r_idx + 1_i32)      = int(st_acc + 1_i64, i32)
      skew_off(r_idx + 1_i32)   = int(skew_acc + 1_i64, i32)
      cayley_off(r_idx + 1_i32) = int(cay_acc + 1_i64, i32)
      ipiv_off(r_idx + 1_i32)   = int(piv_acc + 1_i64, i32)
    end do

    n_st_tot     = int(st_acc, i32)
    n_skew_tot   = int(skew_acc, i32)
    n_cayley_tot = int(cay_acc, i32)
    n_ipiv_tot   = int(piv_acc, i32)
  end subroutine fc_stiefel_offsets

  !---------------------------------------------------------------------------
  ! Stiefel Riemannian gradient and cached skew projection A_skew = X^T * rgrad:
  !   Canonical (beta = 1/2): rgrad = GX - X * GX^T * X, A_skew = 2 * skew(X^T * GX)
  !   Euclidean (beta = 1):   rgrad = GX - X * sym(X^T * GX), A_skew = skew(X^T * GX)
  !---------------------------------------------------------------------------
  pure subroutine fc_stiefel_rgrad(n, p, X, GX, metric, rgrad, A_skew, scratch_p2)
    integer(i32), intent(in) :: n, p, metric
    real(dp), intent(in)     :: X(n, p), GX(n, p)
    real(dp), intent(out)    :: rgrad(n, p), A_skew(p, p)
    real(dp), intent(inout)  :: scratch_p2(p, p)

    integer(i32) :: i, j
    real(dp)     :: sk, sy

    if (n <= 0_i32 .or. p <= 0_i32) return

    ! scratch_p2 := M = X^T * GX
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, GX, n, 0.0_dp, scratch_p2, p)

    do j = 1_i32, p
      do i = 1_i32, n
        rgrad(i, j) = GX(i, j)
      end do
    end do

    if (metric == FC_METRIC_EUCLIDEAN) then
      do j = 1_i32, p
        A_skew(j, j) = 0.0_dp
        do i = 1_i32, j - 1_i32
          sk               = 0.5_dp * (scratch_p2(i, j) - scratch_p2(j, i))
          sy               = 0.5_dp * (scratch_p2(i, j) + scratch_p2(j, i))
          A_skew(i, j)     = sk
          A_skew(j, i)     = -sk
          scratch_p2(i, j) = sy
          scratch_p2(j, i) = sy
        end do
      end do
      call fc_gemm_nn(n, p, p, -1.0_dp, X, n, scratch_p2, p, 1.0_dp, rgrad, n)
    else
      ! Default canonical metric (FC_METRIC_CANONICAL)
      do j = 1_i32, p
        A_skew(j, j) = 0.0_dp
        do i = 1_i32, j - 1_i32
          sk           = scratch_p2(i, j) - scratch_p2(j, i)
          A_skew(i, j) = sk
          A_skew(j, i) = -sk
        end do
      end do
      ! rgrad := GX - X * M^T = GX - X * (GX^T * X)
      call fc_gemm_nt(n, p, p, -1.0_dp, X, n, scratch_p2, p, 1.0_dp, rgrad, n)
    end if
  end subroutine fc_stiefel_rgrad

  !---------------------------------------------------------------------------
  ! Ambient orthogonal projection Z := Z - X * sym(X^T * Z) onto T_X St(n, p),
  ! also outputting A_skew_out = X^T * Z_proj = skew(X^T * Z_in) in so(p).
  !---------------------------------------------------------------------------
  pure subroutine fc_stiefel_proj_tangent(n, p, X, Z, A_skew_out, scratch_p2)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p)
    real(dp), intent(inout)  :: Z(n, p)
    real(dp), intent(out)    :: A_skew_out(p, p)
    real(dp), intent(inout)  :: scratch_p2(p, p)

    integer(i32) :: i, j
    real(dp)     :: sk, sy

    if (n <= 0_i32 .or. p <= 0_i32) return

    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, Z, n, 0.0_dp, scratch_p2, p)

    do j = 1_i32, p
      A_skew_out(j, j) = 0.0_dp
      do i = 1_i32, j - 1_i32
        sk                = 0.5_dp * (scratch_p2(i, j) - scratch_p2(j, i))
        sy                = 0.5_dp * (scratch_p2(i, j) + scratch_p2(j, i))
        A_skew_out(i, j)  = sk
        A_skew_out(j, i)  = -sk
        scratch_p2(i, j)  = sy
        scratch_p2(j, i)  = sy
      end do
    end do

    call fc_gemm_nn(n, p, p, -1.0_dp, X, n, scratch_p2, p, 1.0_dp, Z, n)
  end subroutine fc_stiefel_proj_tangent

  !---------------------------------------------------------------------------
  ! Stiefel beta-metric inner product:
  !   <U, V>_X = <U, V>_F - (1 - beta) * <X^T U, X^T V>_F
  ! Computed in scalar registers (zero scratch writes -> pure function C1583).
  !---------------------------------------------------------------------------
  pure function fc_stiefel_inner(n, p, X, U, V, metric, scratch_2p2) result(val)
    integer(i32), intent(in)           :: n, p, metric
    real(dp), intent(in)               :: X(n, p), U(n, p), V(n, p)
    real(dp), intent(in), optional     :: scratch_2p2(*)
    real(dp)                           :: val

    integer(i32) :: i, j, l
    real(dp)     :: frob_uv, xt_dot, xu_ij, xv_ij

    val = 0.0_dp
    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. present(scratch_2p2)) then
      if (scratch_2p2(1) > 0.0_dp) continue
    end if

    frob_uv = fc_frob_inner(n, p, U, V)
    if (metric == FC_METRIC_EUCLIDEAN) then
      val = frob_uv
      return
    end if

    xt_dot = 0.0_dp
    do j = 1_i32, p
      do i = 1_i32, p
        xu_ij = 0.0_dp
        xv_ij = 0.0_dp
        do l = 1_i32, n
          xu_ij = xu_ij + X(l, i) * U(l, j)
          xv_ij = xv_ij + X(l, i) * V(l, j)
        end do
        xt_dot = xt_dot + xu_ij * xv_ij
      end do
    end do

    val = frob_uv - 0.5_dp * xt_dot
  end function fc_stiefel_inner

  !---------------------------------------------------------------------------
  ! Overflow-safe Frobenius orthogonality residual ||X^T * X - I_p||_F.
  !---------------------------------------------------------------------------
  pure function fc_stiefel_orth_err(n, p, X, E_out) result(err_frob)
    integer(i32), intent(in)       :: n, p
    real(dp), intent(in)           :: X(n, p)
    real(dp), intent(in), optional :: E_out(*)
    real(dp)                       :: err_frob

    integer(i32) :: i, j, l
    real(dp)     :: acc, e_ij, abse, scale, ssq, ratio, mult

    err_frob = 0.0_dp
    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. present(E_out)) then
      if (E_out(1) > 0.0_dp) continue
    end if

    scale = 0.0_dp
    ssq   = 1.0_dp
    do j = 1_i32, p
      do i = 1_i32, j
        acc = 0.0_dp
        do l = 1_i32, n
          acc = acc + X(l, i) * X(l, j)
        end do
        if (i == j) then
          e_ij = acc - 1.0_dp
          mult = 1.0_dp
        else
          e_ij = acc
          mult = 2.0_dp
        end if

        if (ieee_is_nan(e_ij)) then
          err_frob = e_ij
          return
        end if

        abse = abs(e_ij)
        if (abse > 0.0_dp) then
          if (.not. ieee_is_finite(abse)) then
            err_frob = abse
            return
          end if
          if (scale < abse) then
            ratio = scale / abse
            ssq   = mult + ssq * (ratio * ratio)
            scale = abse
          else
            ratio = abse / scale
            ssq   = ssq + mult * (ratio * ratio)
          end if
        end if
      end do
    end do

    err_frob = scale * sqrt(ssq)
  end function fc_stiefel_orth_err

  !---------------------------------------------------------------------------
  ! Materialize E_out = X^T * X - I_p together with ||E_out||_F.
  !---------------------------------------------------------------------------
  pure subroutine fc_stiefel_orth_mat(n, p, X, E_out, err_frob)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p)
    real(dp), intent(out)    :: E_out(p, p)
    real(dp), intent(out)    :: err_frob

    integer(i32) :: i

    err_frob = 0.0_dp
    if (n <= 0_i32 .or. p <= 0_i32) return

    call fc_syrk_tn(n, p, X, E_out)
    do i = 1_i32, p
      E_out(i, i) = E_out(i, i) - 1.0_dp
    end do
    err_frob = fc_stiefel_orth_err(n, p, X)
  end subroutine fc_stiefel_orth_mat

  !===========================================================================
  ! Branch A: Rank-2p Schur-Woodbury Cayley Engine (p x p LU + p=1 fast-path)
  ! Persistent cay_buf layout (3*p^2 + 2*n*p doubles):
  !   [1             .. p^2          ] : A_r         (p x p)
  !   [p^2 + 1       .. 2*p^2        ] : S_r         (p x p)
  !   [2*p^2 + 1     .. 3*p^2        ] : J_lu_r      (p x p, LU of J(tau)^T)
  !   [3*p^2 + 1     .. 3*p^2 + n*p  ] : K_r         (n x p)
  !   [3*p^2+n*p + 1 .. 3*p^2 + 2*n*p] : eta_tilde_r (n x p)
  !===========================================================================

  pure subroutine setup_branch_a_arrays(n, p, X, eta, A_r, S_r, J_lu_r, K_r, eta_tilde_r, M_scr)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p), eta(n, p)
    real(dp), intent(out)    :: A_r(p, p), S_r(p, p), J_lu_r(p, p)
    real(dp), intent(out)    :: K_r(n, p), eta_tilde_r(n, p)
    real(dp), intent(inout)  :: M_scr(p, p)

    integer(i32) :: i, j
    real(dp)     :: sk, acc

    do j = 1_i32, p
      do i = 1_i32, p
        J_lu_r(i, j) = 0.0_dp
      end do
      J_lu_r(j, j) = 1.0_dp
    end do

    if (p == 1_i32) then
      A_r(1, 1) = 0.0_dp
      acc = 0.0_dp
      do i = 1_i32, n
        acc = acc + X(i, 1) * eta(i, 1)
      end do
      do i = 1_i32, n
        K_r(i, 1)         = eta(i, 1) - acc * X(i, 1)
        eta_tilde_r(i, 1) = K_r(i, 1)
      end do
      acc = 0.0_dp
      do i = 1_i32, n
        acc = acc + K_r(i, 1) * K_r(i, 1)
      end do
      S_r(1, 1) = acc
      return
    end if

    ! M_scr := X^T * eta
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, eta, n, 0.0_dp, M_scr, p)

    ! A_r := skew(M_scr)
    do j = 1_i32, p
      A_r(j, j) = 0.0_dp
      do i = 1_i32, j - 1_i32
        sk        = 0.5_dp * (M_scr(i, j) - M_scr(j, i))
        A_r(i, j) = sk
        A_r(j, i) = -sk
      end do
    end do

    ! K_r := eta_tan - X * A_r = eta - X * (X^T * eta) = eta - X * M_scr
    do j = 1_i32, p
      do i = 1_i32, n
        K_r(i, j) = eta(i, j)
      end do
    end do
    call fc_gemm_nn(n, p, p, -1.0_dp, X, n, M_scr, p, 1.0_dp, K_r, n)

    ! eta_tilde_r := K_r + 0.5 * X * A_r = eta_tan - 0.5 * X * A_r
    do j = 1_i32, p
      do i = 1_i32, n
        eta_tilde_r(i, j) = K_r(i, j)
      end do
    end do
    call fc_gemm_nn(n, p, p, 0.5_dp, X, n, A_r, p, 1.0_dp, eta_tilde_r, n)

    ! S_r := K_r^T * K_r
    call fc_syrk_tn(n, p, K_r, S_r)
  end subroutine setup_branch_a_arrays

  pure subroutine retract_branch_a_arrays(n, p, X, tau, A_r, S_r, J_lu_r, K_r, ipiv, Y_out, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(in)      :: X(n, p), tau, A_r(p, p), S_r(p, p), K_r(n, p)
    real(dp), intent(out)     :: J_lu_r(p, p), Y_out(n, p)
    integer(i32), intent(out) :: ipiv(p), info

    integer(i32) :: i, j
    real(dp)     :: half_tau, q_tau, s11, j_scalar, inv_j, c_x, c_k, sij

    info = 0_i32
    do j = 1_i32, p
      ipiv(j) = j
      do i = 1_i32, p
        J_lu_r(i, j) = 0.0_dp
      end do
      J_lu_r(j, j) = 1.0_dp
    end do

    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = X(i, j)
      end do
    end do

    if (ieee_is_nan(tau) .or. .not. ieee_is_finite(tau) .or. abs(tau) > 1.0e150_dp) then
      info = 1_i32
      return
    end if

    if (.not. (abs(tau) > 0.0_dp)) return

    half_tau = 0.5_dp * tau
    q_tau    = half_tau * half_tau

    if (p == 1_i32) then
      s11 = S_r(1, 1)
      if (ieee_is_nan(s11) .or. .not. ieee_is_finite(s11) .or. s11 < 0.0_dp) then
        info = 1_i32
        return
      end if
      if (s11 > 1.0_dp) then
        if (q_tau > 0.45_dp * FC_HUGE / s11) then
          info = 1_i32
          return
        end if
      end if
      j_scalar     = 1.0_dp + q_tau * s11
      J_lu_r(1, 1) = j_scalar
      inv_j        = 1.0_dp / j_scalar
      c_x          = (1.0_dp - q_tau * s11) * inv_j
      c_k          = tau * inv_j
      do i = 1_i32, n
        Y_out(i, 1) = c_x * X(i, 1) + c_k * K_r(i, 1)
      end do
      return
    end if

    ! J(tau)^T = I_p + (tau/2) * A_r + (tau^2/4) * S_r
    do j = 1_i32, p
      do i = 1_i32, p
        sij = S_r(i, j)
        if (ieee_is_nan(sij) .or. .not. ieee_is_finite(sij) .or. &
            ieee_is_nan(A_r(i, j)) .or. .not. ieee_is_finite(A_r(i, j))) then
          info = 1_i32
          return
        end if
        if (abs(sij) > 1.0_dp) then
          if (q_tau > 0.2_dp * FC_HUGE / abs(sij)) then
            info = 1_i32
            return
          end if
        end if
        if (abs(A_r(i, j)) > 1.0_dp) then
          if (abs(half_tau) > 0.2_dp * FC_HUGE / abs(A_r(i, j))) then
            info = 1_i32
            return
          end if
        end if
        J_lu_r(i, j) = half_tau * A_r(i, j) + q_tau * sij
      end do
      J_lu_r(j, j) = J_lu_r(j, j) + 1.0_dp
    end do

    call fc_lu_factor(p, J_lu_r, ipiv, CAYLEY_PIV_TOL, info)
    if (info /= 0_i32) return

    ! Y_out := 2 * X + tau * K_r, then solve Z * J(tau) = Y_out in-place
    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = 2.0_dp * X(i, j) + tau * K_r(i, j)
      end do
    end do
    call fc_lu_solve_right(n, p, J_lu_r, ipiv, Y_out)

    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = Y_out(i, j) - X(i, j)
      end do
    end do
  end subroutine retract_branch_a_arrays

  pure subroutine transport_branch_a_arrays(n, p, X_old, tau, &
      A_r, J_lu_r, eta_tilde_r, ipiv, zeta_inout, R1, R2, C_mat, Z1, Tmp)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X_old(n, p), tau
    real(dp), intent(in)     :: A_r(p, p), J_lu_r(p, p), eta_tilde_r(n, p)
    integer(i32), intent(in) :: ipiv(p)
    real(dp), intent(inout)  :: zeta_inout(n, p)
    real(dp), intent(out)    :: R1(p, p), R2(p, p), C_mat(p, p), Z1(p, p), Tmp(p, p)

    integer(i32) :: i, j
    real(dp)     :: q_tau, r1_s, r2_s, z1_s, coeff_et, coeff_x

    R1    = 0.0_dp
    R2    = 0.0_dp
    C_mat = 0.0_dp
    Z1    = 0.0_dp
    Tmp   = 0.0_dp

    if (.not. (abs(tau) > 0.0_dp)) return

    if (p == 1_i32) then
      r1_s = 0.0_dp
      r2_s = 0.0_dp
      do i = 1_i32, n
        r1_s = r1_s + X_old(i, 1) * zeta_inout(i, 1)
        r2_s = r2_s - eta_tilde_r(i, 1) * zeta_inout(i, 1)
      end do
      z1_s     = (r1_s + 0.5_dp * tau * r2_s) / J_lu_r(1, 1)
      coeff_et = tau * z1_s
      coeff_x  = 2.0_dp * (z1_s - r1_s)
      do i = 1_i32, n
        zeta_inout(i, 1) = zeta_inout(i, 1) + coeff_et * eta_tilde_r(i, 1) + coeff_x * X_old(i, 1)
      end do
      return
    end if

    ! R1 = X_old^T * zeta, R2 = -eta_tilde_r^T * zeta
    call fc_gemm_tn(p, p, n,  1.0_dp, X_old,       n, zeta_inout, n, 0.0_dp, R1, p)
    call fc_gemm_tn(p, p, n, -1.0_dp, eta_tilde_r, n, zeta_inout, n, 0.0_dp, R2, p)

    ! C_mat = I_p - (tau/4) * A_r
    q_tau = 0.25_dp * tau
    do j = 1_i32, p
      do i = 1_i32, p
        C_mat(i, j) = -q_tau * A_r(i, j)
      end do
      C_mat(j, j) = C_mat(j, j) + 1.0_dp
    end do

    ! Z1 := C_mat * R1 + (tau/2) * R2, then solve J(tau) * Z1 = RHS
    ! Note J_lu_r factored J(tau)^T, so (J(tau)^T)^T * Z1 = J(tau) * Z1 uses fc_lu_solve_trans!
    do j = 1_i32, p
      do i = 1_i32, p
        Z1(i, j) = R2(i, j)
      end do
    end do
    call fc_gemm_nn(p, p, p, 1.0_dp, C_mat, p, R1, p, 0.5_dp * tau, Z1, p)
    call fc_lu_solve_trans(p, p, J_lu_r, ipiv, Z1)

    ! Tmp := C_mat * Z1 - R1
    do j = 1_i32, p
      do i = 1_i32, p
        Tmp(i, j) = R1(i, j)
      end do
    end do
    call fc_gemm_nn(p, p, p, 1.0_dp, C_mat, p, Z1, p, -1.0_dp, Tmp, p)

    ! zeta := zeta + tau * eta_tilde_r * Z1 + 2 * X_old * Tmp
    call fc_gemm_nn(n, p, p, tau,    eta_tilde_r, n, Z1,  p, 1.0_dp, zeta_inout, n)
    call fc_gemm_nn(n, p, p, 2.0_dp, X_old,       n, Tmp, p, 1.0_dp, zeta_inout, n)
  end subroutine transport_branch_a_arrays

  pure subroutine fc_cayley_setup_branch_a(n, p, X, eta, cay_buf, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p), eta(n, p)
    real(dp), intent(inout)  :: cay_buf(*), scratch(*)

    integer(i32) :: p2, np, i1, i2, i3, i4, i5

    if (n <= 0_i32 .or. p <= 0_i32) return
    p2 = p * p
    np = n * p
    i1 = 1_i32
    i2 = i1 + p2
    i3 = i2 + p2
    i4 = i3 + p2
    i5 = i4 + np
    call setup_branch_a_arrays(n, p, X, eta, &
      cay_buf(i1:i2 - 1_i32), cay_buf(i2:i3 - 1_i32), cay_buf(i3:i4 - 1_i32), &
      cay_buf(i4:i5 - 1_i32), cay_buf(i5:i5 + np - 1_i32), scratch(1_i32:p2))
  end subroutine fc_cayley_setup_branch_a

  pure subroutine fc_cayley_retract_branch_a(n, p, X, eta, tau, cay_buf, ipiv, Y_out, scratch, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(in)      :: X(n, p), eta(n, p), tau
    real(dp), intent(inout)   :: cay_buf(*), scratch(*)
    integer(i32), intent(out) :: ipiv(p), info
    real(dp), intent(out)     :: Y_out(n, p)

    integer(i32) :: p2, np, i1, i2, i3, i4

    info = 0_i32
    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. eta(1, 1) > 0.0_dp) continue
    scratch(1) = 0.0_dp

    p2 = p * p
    np = n * p
    i1 = 1_i32
    i2 = i1 + p2
    i3 = i2 + p2
    i4 = i3 + p2
    call retract_branch_a_arrays(n, p, X, tau, &
      cay_buf(i1:i2 - 1_i32), cay_buf(i2:i3 - 1_i32), cay_buf(i3:i4 - 1_i32), &
      cay_buf(i4:i4 + np - 1_i32), ipiv, Y_out, info)
  end subroutine fc_cayley_retract_branch_a

  pure subroutine fc_cayley_transport_branch_a(n, p, X_old, eta, tau, &
      cay_buf, ipiv, zeta_inout, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X_old(n, p), eta(n, p), tau
    real(dp), intent(in)     :: cay_buf(*)
    integer(i32), intent(in) :: ipiv(p)
    real(dp), intent(inout)  :: zeta_inout(n, p), scratch(*)

    integer(i32) :: p2, np, i1, i3, i5

    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. eta(1, 1) > 0.0_dp) continue

    p2 = p * p
    np = n * p
    i1 = 1_i32
    i3 = 1_i32 + 2_i32 * p2
    i5 = 1_i32 + 3_i32 * p2 + np
    call transport_branch_a_arrays(n, p, X_old, tau, &
      cay_buf(i1:i1 + p2 - 1_i32), cay_buf(i3:i3 + p2 - 1_i32), cay_buf(i5:i5 + np - 1_i32), &
      ipiv, zeta_inout, &
      scratch(1_i32:p2), scratch(p2 + 1_i32:2_i32 * p2), &
      scratch(2_i32 * p2 + 1_i32:3_i32 * p2), scratch(3_i32 * p2 + 1_i32:4_i32 * p2), &
      scratch(4_i32 * p2 + 1_i32:5_i32 * p2))
  end subroutine fc_cayley_transport_branch_a

  !===========================================================================
  ! Branch B: Direct n x n Skew-Cayley LU with Resolvent Doubling 2*M^{-1} - I
  ! Persistent cay_buf layout (2*n^2 doubles):
  !   [1       .. n^2  ] : W_eta (n x n)
  !   [n^2 + 1 .. 2*n^2] : M_lu  (n x n, LU of I_n - (tau/2) * W_eta)
  !===========================================================================

  pure subroutine setup_branch_b_arrays(n, p, X, eta, W_eta, M_lu, M_scr, H_scr, eta_tilde)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p), eta(n, p)
    real(dp), intent(out)    :: W_eta(n, n), M_lu(n, n)
    real(dp), intent(inout)  :: M_scr(p, p), H_scr(p, p), eta_tilde(n, p)

    integer(i32) :: i, j

    ! M_scr := X^T * eta
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, eta, n, 0.0_dp, M_scr, p)

    ! H_scr := sym(M_scr) + 0.5 * skew(M_scr) = 0.75 * M_scr + 0.25 * M_scr^T
    do j = 1_i32, p
      do i = 1_i32, p
        H_scr(i, j) = 0.75_dp * M_scr(i, j) + 0.25_dp * M_scr(j, i)
      end do
    end do

    ! eta_tilde := eta - X * H_scr = eta_tan - 0.5 * X * (X^T * eta_tan)
    do j = 1_i32, p
      do i = 1_i32, n
        eta_tilde(i, j) = eta(i, j)
      end do
    end do
    call fc_gemm_nn(n, p, p, -1.0_dp, X, n, H_scr, p, 1.0_dp, eta_tilde, n)

    ! M_lu temporarily holds P_r = eta_tilde * X^T; W_eta = P_r - P_r^T
    call fc_gemm_nt(n, n, p, 1.0_dp, eta_tilde, n, X, n, 0.0_dp, M_lu, n)
    do j = 1_i32, n
      W_eta(j, j) = 0.0_dp
      do i = 1_i32, j - 1_i32
        W_eta(i, j) = M_lu(i, j) - M_lu(j, i)
        W_eta(j, i) = -W_eta(i, j)
      end do
    end do
  end subroutine setup_branch_b_arrays

  pure subroutine retract_branch_b_arrays(n, p, X, tau, W_eta, M_lu, ipiv, Y_out, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(in)      :: X(n, p), tau, W_eta(n, n)
    real(dp), intent(out)     :: M_lu(n, n), Y_out(n, p)
    integer(i32), intent(out) :: ipiv(n), info

    integer(i32) :: i, j
    real(dp)     :: half_tau, wij

    info = 0_i32
    do j = 1_i32, n
      ipiv(j) = j
      do i = 1_i32, n
        M_lu(i, j) = 0.0_dp
      end do
      M_lu(j, j) = 1.0_dp
    end do

    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = X(i, j)
      end do
    end do

    if (ieee_is_nan(tau) .or. .not. ieee_is_finite(tau) .or. abs(tau) > 1.0e150_dp) then
      info = 1_i32
      return
    end if

    if (.not. (abs(tau) > 0.0_dp)) return

    half_tau = 0.5_dp * tau
    do j = 1_i32, n
      do i = 1_i32, n
        wij = W_eta(i, j)
        if (ieee_is_nan(wij) .or. .not. ieee_is_finite(wij)) then
          info = 1_i32
          return
        end if
        if (abs(wij) > 1.0_dp) then
          if (abs(half_tau) > 0.45_dp * FC_HUGE / abs(wij)) then
            info = 1_i32
            return
          end if
        end if
        M_lu(i, j) = -half_tau * wij
      end do
      M_lu(j, j) = M_lu(j, j) + 1.0_dp
    end do

    call fc_lu_factor(n, M_lu, ipiv, CAYLEY_PIV_TOL, info)
    if (info /= 0_i32) return

    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = 2.0_dp * X(i, j)
      end do
    end do
    call fc_lu_solve(n, p, M_lu, ipiv, Y_out)

    do j = 1_i32, p
      do i = 1_i32, n
        Y_out(i, j) = Y_out(i, j) - X(i, j)
      end do
    end do
  end subroutine retract_branch_b_arrays

  pure subroutine transport_branch_b_arrays(n, p, tau, M_lu, ipiv, zeta_inout, Z_scr)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: tau, M_lu(n, n)
    integer(i32), intent(in) :: ipiv(n)
    real(dp), intent(inout)  :: zeta_inout(n, p)
    real(dp), intent(out)    :: Z_scr(n, p)

    integer(i32) :: i, j

    Z_scr = 0.0_dp
    if (.not. (abs(tau) > 0.0_dp)) return

    do j = 1_i32, p
      do i = 1_i32, n
        Z_scr(i, j) = 2.0_dp * zeta_inout(i, j)
      end do
    end do
    call fc_lu_solve(n, p, M_lu, ipiv, Z_scr)

    do j = 1_i32, p
      do i = 1_i32, n
        zeta_inout(i, j) = Z_scr(i, j) - zeta_inout(i, j)
      end do
    end do
  end subroutine transport_branch_b_arrays

  pure subroutine fc_cayley_setup_branch_b(n, p, X, eta, cay_buf, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p), eta(n, p)
    real(dp), intent(inout)  :: cay_buf(*), scratch(*)

    integer(i32) :: n2, p2, np

    if (n <= 0_i32 .or. p <= 0_i32) return
    n2 = n * n
    p2 = p * p
    np = n * p
    call setup_branch_b_arrays(n, p, X, eta, &
      cay_buf(1_i32:n2), cay_buf(n2 + 1_i32:2_i32 * n2), &
      scratch(1_i32:p2), scratch(p2 + 1_i32:2_i32 * p2), &
      scratch(2_i32 * p2 + 1_i32:2_i32 * p2 + np))
  end subroutine fc_cayley_setup_branch_b

  pure subroutine fc_cayley_retract_branch_b(n, p, X, eta, tau, cay_buf, ipiv, Y_out, scratch, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(in)      :: X(n, p), eta(n, p), tau
    real(dp), intent(inout)   :: cay_buf(*), scratch(*)
    integer(i32), intent(out) :: ipiv(n), info
    real(dp), intent(out)     :: Y_out(n, p)

    integer(i32) :: n2

    info = 0_i32
    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. eta(1, 1) > 0.0_dp) continue
    scratch(1) = 0.0_dp

    n2 = n * n
    call retract_branch_b_arrays(n, p, X, tau, &
      cay_buf(1_i32:n2), cay_buf(n2 + 1_i32:2_i32 * n2), ipiv, Y_out, info)
  end subroutine fc_cayley_retract_branch_b

  pure subroutine fc_cayley_transport_branch_b(n, p, X_old, eta, tau, &
      cay_buf, ipiv, zeta_inout, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X_old(n, p), eta(n, p), tau
    real(dp), intent(in)     :: cay_buf(*)
    integer(i32), intent(in) :: ipiv(n)
    real(dp), intent(inout)  :: zeta_inout(n, p), scratch(*)

    integer(i32) :: n2, np

    if (n <= 0_i32 .or. p <= 0_i32) return
    if (.false. .and. X_old(1, 1) > 0.0_dp .and. eta(1, 1) > 0.0_dp) continue

    n2 = n * n
    np = n * p
    call transport_branch_b_arrays(n, p, tau, &
      cay_buf(n2 + 1_i32:2_i32 * n2), ipiv, zeta_inout, scratch(1_i32:np))
  end subroutine fc_cayley_transport_branch_b

  !===========================================================================
  ! Auto-Dispatching Dual-Regime Public Entry Points
  !   2 * p <  n -> Branch A (Schur-Woodbury + p == 1 fast-path)
  !   2 * p >= n -> Branch B (Direct n x n LU + Resolvent Doubling)
  !===========================================================================

  pure subroutine fc_cayley_setup(n, p, X, eta, cay_buf, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X(n, p), eta(n, p)
    real(dp), intent(inout)  :: cay_buf(*), scratch(*)

    if (2_i32 * p < n) then
      call fc_cayley_setup_branch_a(n, p, X, eta, cay_buf, scratch)
    else
      call fc_cayley_setup_branch_b(n, p, X, eta, cay_buf, scratch)
    end if
  end subroutine fc_cayley_setup

  pure subroutine fc_cayley_retract(n, p, X, eta, tau, cay_buf, ipiv, Y_out, scratch, info)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(in)      :: X(n, p), eta(n, p), tau
    real(dp), intent(inout)   :: cay_buf(*), scratch(*)
    integer(i32), intent(out) :: ipiv(*)
    real(dp), intent(out)     :: Y_out(n, p)
    integer(i32), intent(out) :: info

    if (2_i32 * p < n) then
      call fc_cayley_retract_branch_a(n, p, X, eta, tau, cay_buf, ipiv(1_i32:p), Y_out, scratch, info)
    else
      call fc_cayley_retract_branch_b(n, p, X, eta, tau, cay_buf, ipiv(1_i32:n), Y_out, scratch, info)
    end if
  end subroutine fc_cayley_retract

  pure subroutine fc_cayley_transport_vec(n, p, X_old, eta, tau, &
      cay_buf, ipiv, zeta_inout, scratch)
    integer(i32), intent(in) :: n, p
    real(dp), intent(in)     :: X_old(n, p), eta(n, p), tau
    real(dp), intent(in)     :: cay_buf(*)
    integer(i32), intent(in) :: ipiv(*)
    real(dp), intent(inout)  :: zeta_inout(n, p), scratch(*)

    if (2_i32 * p < n) then
      call fc_cayley_transport_branch_a(n, p, X_old, eta, tau, &
        cay_buf, ipiv(1_i32:p), zeta_inout, scratch)
    else
      call fc_cayley_transport_branch_b(n, p, X_old, eta, tau, &
        cay_buf, ipiv(1_i32:n), zeta_inout, scratch)
    end if
  end subroutine fc_cayley_transport_vec

end module fortchuck_stiefel
