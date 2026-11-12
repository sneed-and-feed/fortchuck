module coupled_test_callbacks
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_double, c_ptr, c_associated, c_f_pointer
  use fortchuck, only: dp, i32
  use fortchuck_linalg, only: fc_gemm_nn, fc_gemm_tn
  implicit none (type, external)
  public

  type :: ky_fan_ctx_t
    real(dp) :: A(14, 14)
    real(dp) :: M(3, 3)
    real(dp) :: z_ref(3)
  end type ky_fan_ctx_t

  type :: so5_ctx_t
    real(dp) :: B(5, 5)
    real(dp) :: C(5, 5)
  end type so5_ctx_t

  type :: brockett12_ctx_t
    real(dp) :: A(12, 12)
    real(dp) :: w(3)
  end type brockett12_ctx_t

  type :: barrier_ctx_t
    integer(i32) :: mode = 0_i32  ! 1 = domain barrier rc=+1 when z(1) < 0.25; -1 = abort on eval 3
    integer(i32) :: calls = 0_i32
  end type barrier_ctx_t

contains

  !---------------------------------------------------------------------------
  ! 1. Pure-Box callback (d = 12, R = 0):
  !    Coupled Rosenbrock + active-bound quadratic with +-FC_HUGE & fixed coords
  !---------------------------------------------------------------------------
  function cb_pure_box12(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    integer(i32) :: i
    real(dp)     :: t1, t2, targets(12), weights(12), diff_i, cpl

    if (.false. .and. R == 0_i32 .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. X_packed(1) > 0.0_dp .and. GX_packed_out(1) > 0.0_dp) continue
    if (.false. .and. c_associated(udata)) continue

    targets(1)  =  1.0_dp
    targets(2)  =  1.0_dp
    targets(3)  =  1.2_dp   ! Clamps at upper bound u(3) = 0.5
    targets(4)  = -1.4_dp   ! Clamps at lower bound l(4) = -0.5
    targets(5)  =  0.0_dp   ! Fixed l(5) = u(5) = 1.25
    targets(6)  =  0.0_dp   ! Fixed l(6) = u(6) = -0.75
    targets(7)  = -0.8_dp   ! Clamps at l(7) = 0.0
    targets(8)  =  1.5_dp   ! Clamps at u(8) = 0.8
    targets(9)  =  0.35_dp  ! Strictly interior in [0, 0.8]
    targets(10) =  0.55_dp  ! Strictly interior in [0, 0.8]
    targets(11) = -2.0_dp   ! Clamps at l(11) = 0.0 with large multiplier
    targets(12) =  2.5_dp   ! Clamps at u(12) = 0.8 with large multiplier

    do i = 1_i32, d
      weights(i) = 2.0_dp * real(i, dp)
      gz_out(i)  = 0.0_dp
    end do

    ! Rosenbrock on unconstrained [-FC_HUGE, +FC_HUGE] pair (z_1, z_2)
    t1    = 1.0_dp - z(1)
    t2    = z(2) - z(1) * z(1)
    f_out = 0.5_dp * (t1 * t1 + 50.0_dp * t2 * t2)
    gz_out(1) = -t1 - 200.0_dp * z(1) * t2
    gz_out(2) = 100.0_dp * t2

    ! Coupled quadratic across remaining coordinates i = 3..12
    do i = 3_i32, d
      diff_i    = z(i) - targets(i)
      f_out     = f_out + 0.5_dp * weights(i) * diff_i * diff_i
      gz_out(i) = gz_out(i) + weights(i) * diff_i
    end do

    do i = 3_i32, d - 1_i32
      cpl           = z(i) - 0.5_dp * z(i + 1_i32)
      f_out         = f_out + 0.5_dp * cpl * cpl
      gz_out(i)     = gz_out(i) + cpl
      gz_out(i + 1) = gz_out(i + 1) - 0.5_dp * cpl
    end do

    rc = 0_i32
  end function cb_pure_box12

  !---------------------------------------------------------------------------
  ! 2a. Sphere Rayleigh quotient on St(10, 1) (p = 1 scalar Cayley fast-path)
  !---------------------------------------------------------------------------
  function cb_sphere10(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    integer(i32) :: i, n
    real(dp)     :: lam_i

    if (.false. .and. d == 0_i32 .and. R == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. z(1) > 0.0_dp .and. gz_out(1) > 0.0_dp) continue
    if (.false. .and. c_associated(udata)) continue

    n     = n_arr(1)
    f_out = 0.0_dp
    do i = 1_i32, n
      lam_i            = 0.5_dp + real(i, dp)   ! Minimum eigenvalue = 1.5 at i = 1 -> f* = 0.75
      f_out            = f_out + 0.5_dp * lam_i * X_packed(i) * X_packed(i)
      GX_packed_out(i) = lam_i * X_packed(i)
    end do
    rc = 0_i32
  end function cb_sphere10

  !---------------------------------------------------------------------------
  ! 2b. Pure-Stiefel Branch A Brockett on St(12, 3): f(X) = 0.5 * tr(X^T A X diag(w))
  !---------------------------------------------------------------------------
  function cb_brockett12(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(brockett12_ctx_t), pointer :: ctx
    real(dp)                        :: AX(12, 3)
    integer(i32)                    :: i, j, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. z(1) > 0.0_dp .and. gz_out(1) > 0.0_dp) continue

    call c_f_pointer(udata, ctx)
    call fc_gemm_nn(12_i32, 3_i32, 12_i32, 1.0_dp, ctx%A, 12_i32, X_packed, 12_i32, &
      0.0_dp, AX, 12_i32)

    f_out = 0.0_dp
    idx   = 0_i32
    do j = 1_i32, 3_i32
      do i = 1_i32, 12_i32
        idx                = idx + 1_i32
        f_out              = f_out + 0.5_dp * ctx%w(j) * X_packed(idx) * AX(i, j)
        GX_packed_out(idx) = ctx%w(j) * AX(i, j)
      end do
    end do
    rc = 0_i32
  end function cb_brockett12

  !---------------------------------------------------------------------------
  ! 2c. Pure-Stiefel Branch B Special Orthogonal Wahba on SO(5) subset St(5, 5):
  !     f(X) = 0.5 * ||X * B - C||_F^2
  !---------------------------------------------------------------------------
  function cb_so5_wahba(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(so5_ctx_t), pointer :: ctx
    real(dp)                 :: Res_mat(5, 5), Grad_mat(5, 5)
    integer(i32)             :: i, j, l, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. z(1) > 0.0_dp .and. gz_out(1) > 0.0_dp) continue

    call c_f_pointer(udata, ctx)
    call fc_gemm_nn(5_i32, 5_i32, 5_i32, 1.0_dp, X_packed, 5_i32, ctx%B, 5_i32, &
      0.0_dp, Res_mat, 5_i32)

    f_out = 0.0_dp
    do j = 1_i32, 5_i32
      do i = 1_i32, 5_i32
        Res_mat(i, j) = Res_mat(i, j) - ctx%C(i, j)
        f_out         = f_out + 0.5_dp * Res_mat(i, j) * Res_mat(i, j)
      end do
    end do

    ! Grad = Res_mat * B^T
    idx = 0_i32
    do j = 1_i32, 5_i32
      do i = 1_i32, 5_i32
        Grad_mat(i, j) = 0.0_dp
        do l = 1_i32, 5_i32
          Grad_mat(i, j) = Grad_mat(i, j) + Res_mat(i, l) * ctx%B(j, l)
        end do
        idx                = idx + 1_i32
        GX_packed_out(idx) = Grad_mat(i, j)
      end do
    end do
    rc = 0_i32
  end function cb_so5_wahba

  !---------------------------------------------------------------------------
  ! 3. Multi-factor Coupled [l, u]^6 x St(10, 2) x St(4, 4) (Branch A + Branch B)
  !---------------------------------------------------------------------------
  function cb_coupled_multi(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    integer(i32) :: i, j, idx1, idx2
    real(dp)     :: X1(10, 2), X2(4, 4), GX1(10, 2), GX2(4, 4)
    real(dp)     :: z_targ(6), d_i, lam1, lam2, c12, s1, s2

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. c_associated(udata)) continue

    z_targ(1) = -0.5_dp   ! Active at l(1) = 0.1
    z_targ(2) =  1.8_dp   ! Active at u(2) = 1.2
    z_targ(3) =  0.5_dp   ! Fixed l(3) = u(3) = 0.5
    z_targ(4) =  0.65_dp  ! Interior in [0.1, 1.2]
    z_targ(5) =  0.9_dp   ! Interior in [0.1, 1.2]
    z_targ(6) = -1.2_dp   ! Active at l(6) = 0.1

    idx1 = 0_i32
    do j = 1_i32, 2_i32
      do i = 1_i32, 10_i32
        idx1      = idx1 + 1_i32
        X1(i, j)  = X_packed(idx1)
        GX1(i, j) = 0.0_dp
      end do
    end do

    idx2 = 20_i32
    do j = 1_i32, 4_i32
      do i = 1_i32, 4_i32
        idx2      = idx2 + 1_i32
        X2(i, j)  = X_packed(idx2)
        GX2(i, j) = 0.0_dp
      end do
    end do

    f_out = 0.0_dp
    do i = 1_i32, 6_i32
      d_i       = z(i) - z_targ(i)
      f_out     = f_out + 0.5_dp * (3.0_dp + real(i, dp)) * d_i * d_i
      gz_out(i) = (3.0_dp + real(i, dp)) * d_i
    end do

    ! Coupled z * St(10, 2) Brockett + linear alignment
    do j = 1_i32, 2_i32
      do i = 1_i32, 10_i32
        lam1      = 0.8_dp * real(i, dp) + 0.15_dp * real(j, dp)
        f_out     = f_out + 0.5_dp * z(j) * lam1 * X1(i, j) * X1(i, j)
        gz_out(j) = gz_out(j) + 0.5_dp * lam1 * X1(i, j) * X1(i, j)
        GX1(i, j) = GX1(i, j) + z(j) * lam1 * X1(i, j)
      end do
      f_out            = f_out - z(3_i32 + j) * X1(j, j)
      gz_out(3_i32 + j) = gz_out(3_i32 + j) - X1(j, j)
      GX1(j, j)        = GX1(j, j) - z(3_i32 + j)
    end do

    ! Coupled z * St(4, 4) Procrustes + cross-factor St(10,2) <-> St(4,4) coupling
    do j = 1_i32, 4_i32
      do i = 1_i32, 4_i32
        lam2 = 0.2_dp * real(i + j, dp)
        if (i == j) then
          d_i = X2(i, j) - 1.0_dp
        else
          d_i = X2(i, j)
        end if
        f_out     = f_out + 0.5_dp * (1.0_dp + z(6)) * lam2 * d_i * d_i
        gz_out(6) = gz_out(6) + 0.5_dp * lam2 * d_i * d_i
        GX2(i, j) = GX2(i, j) + (1.0_dp + z(6)) * lam2 * d_i
      end do
    end do

    ! Trilinear cross-coupling z(5) * (X1(1,1) + X1(2,2)) * (X2(1,1) + X2(2,2))
    s1    = X1(1, 1) + X1(2, 2)
    s2    = X2(1, 1) + X2(2, 2)
    c12   = 0.35_dp
    f_out = f_out + c12 * z(5) * s1 * s2
    gz_out(5) = gz_out(5) + c12 * s1 * s2
    GX1(1, 1) = GX1(1, 1) + c12 * z(5) * s2
    GX1(2, 2) = GX1(2, 2) + c12 * z(5) * s2
    GX2(1, 1) = GX2(1, 1) + c12 * z(5) * s1
    GX2(2, 2) = GX2(2, 2) + c12 * z(5) * s1

    idx1 = 0_i32
    do j = 1_i32, 2_i32
      do i = 1_i32, 10_i32
        idx1                = idx1 + 1_i32
        GX_packed_out(idx1) = GX1(i, j)
      end do
    end do
    idx2 = 20_i32
    do j = 1_i32, 4_i32
      do i = 1_i32, 4_i32
        idx2                = idx2 + 1_i32
        GX_packed_out(idx2) = GX2(i, j)
      end do
    end do

    rc = 0_i32
  end function cb_coupled_multi

  !---------------------------------------------------------------------------
  ! 4. Box-Coupled Brockett with Certified Ky Fan QP Global Minimum:
  !    f(z, X) = tr(X^T A X diag(z)) + 0.5 * (z - z_ref)^T M (z - z_ref)
  !---------------------------------------------------------------------------
  function cb_ky_fan_brockett(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(ky_fan_ctx_t), pointer :: ctx
    real(dp)                    :: AX(14, 3), dz_vec(3), Mdz(3), ray_j
    integer(i32)                :: i, j, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue

    call c_f_pointer(udata, ctx)
    call fc_gemm_nn(14_i32, 3_i32, 14_i32, 1.0_dp, ctx%A, 14_i32, X_packed, 14_i32, &
      0.0_dp, AX, 14_i32)

    do j = 1_i32, 3_i32
      dz_vec(j) = z(j) - ctx%z_ref(j)
    end do

    f_out = 0.0_dp
    do i = 1_i32, 3_i32
      Mdz(i) = 0.0_dp
      do j = 1_i32, 3_i32
        Mdz(i) = Mdz(i) + ctx%M(i, j) * dz_vec(j)
      end do
      f_out = f_out + 0.5_dp * dz_vec(i) * Mdz(i)
    end do

    idx = 0_i32
    do j = 1_i32, 3_i32
      ray_j = 0.0_dp
      do i = 1_i32, 14_i32
        idx                = idx + 1_i32
        ray_j              = ray_j + X_packed(idx) * AX(i, j)
        GX_packed_out(idx) = 2.0_dp * z(j) * AX(i, j)
      end do
      f_out     = f_out + z(j) * ray_j
      gz_out(j) = ray_j + Mdz(j)
    end do

    rc = 0_i32
  end function cb_ky_fan_brockett

  !---------------------------------------------------------------------------
  ! 5a. FM-2 Clamped Multiplier Injection Test:
  !     f(z, X) = 10^4 * z(1) * (1 + X(1,1)) + 0.5 * tr(X^T * diag(d_i) * X)
  !---------------------------------------------------------------------------
  function cb_fm2_multiplier(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    integer(i32) :: i, j, idx
    real(dp)     :: di

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. c_associated(udata)) continue

    f_out     = 1.0e4_dp * z(1) * (1.0_dp + X_packed(1))
    gz_out(1) = 1.0e4_dp * (1.0_dp + X_packed(1))

    idx = 0_i32
    do j = 1_i32, 3_i32
      do i = 1_i32, 8_i32
        idx                = idx + 1_i32
        di                 = real(i, dp) * 1.5_dp
        f_out              = f_out + 0.5_dp * di * X_packed(idx) * X_packed(idx)
        GX_packed_out(idx) = di * X_packed(idx)
      end do
    end do
    GX_packed_out(1) = GX_packed_out(1) + 1.0e4_dp * z(1)
    rc = 0_i32
  end function cb_fm2_multiplier

  !---------------------------------------------------------------------------
  ! 5b. Domain Barrier (rc > 0 halving) and Callback Abort (rc < 0) Test
  !---------------------------------------------------------------------------
  function cb_barrier_or_abort(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(barrier_ctx_t), pointer :: ctx
    integer(i32)                 :: i

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue

    call c_f_pointer(udata, ctx)
    ctx%calls = ctx%calls + 1_i32

    if (ctx%mode < 0_i32 .and. ctx%calls >= 3_i32) then
      f_out = 0.0_dp
      rc    = -1_i32
      return
    end if

    ! Domain barrier: reject z(1) <= 0.25 with recoverable rc = +1
    if (ctx%mode > 0_i32 .and. z(1) <= 0.25_dp) then
      f_out = 0.0_dp
      rc    = 1_i32
      return
    end if

    ! Smooth log-barrier + St(4, 2) objective with interior minimizer z(1)* > 0.25
    f_out     = -log(z(1) - 0.25_dp) + 2.0_dp * (z(1) - 0.75_dp)**2 + (z(2) - 0.4_dp)**2
    gz_out(1) = -1.0_dp / (z(1) - 0.25_dp) + 4.0_dp * (z(1) - 0.75_dp)
    gz_out(2) = 2.0_dp * (z(2) - 0.4_dp)

    do i = 1_i32, 8_i32
      f_out            = f_out + 0.5_dp * real(i, dp) * X_packed(i) * X_packed(i)
      GX_packed_out(i) = real(i, dp) * X_packed(i)
    end do
    rc = 0_i32
  end function cb_barrier_or_abort

end module coupled_test_callbacks

program test_coupled_solver
  use, intrinsic :: iso_c_binding, only: c_loc, c_funloc, c_null_ptr, c_null_funptr
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_positive_inf
  use fortchuck, only: &
    dp, i32, i64, FC_HUGE, &
    FC_SUCCESS_QUERY, FC_CONVERGED_GRAD, FC_CONVERGED_FTOL, FC_CONVERGED_XTOL, &
    FC_MAX_ITER_REACHED, FC_MAX_EVAL_REACHED, &
    FC_ERR_INVALID_DIM, FC_ERR_NULL_PTR, FC_ERR_INVALID_BOUNDS, FC_ERR_INVALID_X0, &
    FC_ERR_WORK_TOO_SMALL, FC_ERR_CALLBACK_NAN, FC_ERR_CALLBACK_ABORT, &
    FC_ERR_INVALID_OPTS, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, &
    FC_DAMP_CAUTIOUS, FC_DAMP_POWELL, &
    fc_options_t, fc_result_t, fc_init_default_options, &
    fc_workspace_query, fc_solve, &
    fortchuck_default_options, fortchuck_workspace_query, fortchuck_solve, &
    fortchuck_project_stiefel, fortchuck_kkt_residual
  use fortchuck_linalg, only: fc_gemm_nn, fc_gemm_nt, fc_lu_factor, fc_lu_solve, fc_geqrf_pos
  use fortchuck_stiefel, only: fc_stiefel_orth_err
  use coupled_test_callbacks, only: &
    ky_fan_ctx_t, so5_ctx_t, brockett12_ctx_t, barrier_ctx_t, &
    cb_pure_box12, cb_sphere10, cb_brockett12, cb_so5_wahba, &
    cb_coupled_multi, cb_ky_fan_brockett, cb_fm2_multiplier, cb_barrier_or_abort
  implicit none (type, external)

  integer(i32) :: rng_state

  rng_state = 271828_i32

  call test_pure_box_regimes()
  call test_pure_stiefel_branch_a_and_so5()
  call test_multifactor_coupled_solver()
  call test_ky_fan_box_coupled_brockett_and_modes()
  call test_fm2_guards_and_c_abi_errors()

  write (*, '(A)') '[PASS] test_coupled_solver: all assertions passed.'

contains

  subroutine check_true(cond, msg)
    logical, intent(in)      :: cond
    character(*), intent(in) :: msg
    if (.not. cond) then
      write (*, '(A,A)') '[FAIL] ', msg
      error stop 1
    end if
  end subroutine check_true

  function next_rand(state) result(u)
    integer(i32), intent(inout) :: state
    real(dp)                    :: u
    integer(8)                  :: s64
    s64   = mod(int(state, 8) * 48271_8, 2147483647_8)
    state = int(s64, i32)
    u     = 2.0_dp * (real(s64, dp) / 2147483647.0_dp) - 1.0_dp
  end function next_rand

  subroutine fill_rand_vec(n, v, state)
    integer(i32), intent(in)    :: n
    real(dp), intent(out)       :: v(n)
    integer(i32), intent(inout) :: state
    integer(i32)                :: i
    do i = 1_i32, n
      v(i) = next_rand(state)
    end do
  end subroutine fill_rand_vec

  function det_5x5(A_in) result(det_val)
    real(dp), intent(in) :: A_in(5, 5)
    real(dp)             :: det_val
    real(dp)             :: LU(5, 5)
    integer(i32)         :: ipiv(5), info, i
    LU = A_in
    call fc_lu_factor(5_i32, LU, ipiv, 1.0e-12_dp, info)
    if (info /= 0_i32) then
      det_val = 0.0_dp
      return
    end if
    det_val = 1.0_dp
    do i = 1_i32, 5_i32
      if (ipiv(i) /= i) det_val = -det_val
      det_val = det_val * LU(i, i)
    end do
  end function det_5x5

  !---------------------------------------------------------------------------
  ! 1. Pure-Box regime (R = 0): +-FC_HUGE, +-Inf, fixed l_i == u_i, active bounds,
  !    and C-ABI null-pointer safety when R == 0 (n_arr/p_arr/X_packed == NULL)
  !---------------------------------------------------------------------------
  subroutine test_pure_box_regimes()
    integer(i32), parameter :: d = 12_i32
    real(dp), target        :: z1(d), z2(d), l(d), u(d), gz_chk(d)
    real(dp), target        :: work(4096)
    integer(i32), target    :: iwork(512), dummy_n(1)
    real(dp)                :: dummy_X(1), dummy_GX(1), f_chk, pinf
    real(dp), target        :: kkt_chk, orth_chk
    type(fc_options_t), target :: opts
    type(fc_result_t), target  :: res1, res2
    integer(i64), target       :: lwork_q, liwork_q
    integer(i32)               :: st, i, rc

    pinf = ieee_value(1.0_dp, ieee_positive_inf)

    l(1)  = -FC_HUGE; u(1)  =  FC_HUGE
    l(2)  = -FC_HUGE; u(2)  =  FC_HUGE
    l(3)  = -pinf;    u(3)  =  0.5_dp
    l(4)  = -0.5_dp;  u(4)  =  pinf
    l(5)  =  1.25_dp; u(5)  =  1.25_dp
    l(6)  = -0.75_dp; u(6)  = -0.75_dp
    do i = 7_i32, 12_i32
      l(i) = 0.0_dp
      u(i) = 0.8_dp
    end do

    call fill_rand_vec(d, z1, rng_state)
    z1(1) = -1.2_dp
    z1(2) =  1.0_dp
    z1(5) =  9.0_dp   ! Out-of-bounds initial value projected to fixed 1.25
    z1(6) = -9.0_dp   ! Out-of-bounds initial value projected to fixed -0.75
    z2    = z1

    dummy_n(1) = 1_i32
    dummy_X(1) = 0.0_dp
    call fc_init_default_options(opts)
    opts%gtol = 1.0e-10_dp

    ! Verify C-ABI workspace query with R = 0 and n_arr/p_arr = NULL
    st = fortchuck_workspace_query(d, 0_i32, c_null_ptr, c_null_ptr, opts%memory, &
      c_loc(lwork_q), c_loc(liwork_q))
    call check_true(st == FC_SUCCESS_QUERY, 'Pure-box fortchuck_workspace_query R=0 NULL')
    call check_true(lwork_q <= 4096_i64 .and. liwork_q <= 512_i64, 'Pure-box workspace fits')

    ! Native fc_solve
    call fc_solve(d, 0_i32, dummy_n, dummy_n, z1, l, u, dummy_X, &
      cb_pure_box12, c_null_ptr, opts, work, lwork_q, iwork, liwork_q, res1)
    call check_true(res1%status == FC_CONVERGED_GRAD, 'Pure-box native fc_solve CONVERGED_GRAD')
    call check_true(res1%kkt_residual <= 1.0e-10_dp, 'Pure-box KKT <= 1e-10')
    call check_true(abs(z1(1) - 1.0_dp) <= 1.0e-9_dp, 'Pure-box Rosenbrock z(1) == 1')
    call check_true(abs(z1(2) - 1.0_dp) <= 1.0e-9_dp, 'Pure-box Rosenbrock z(2) == 1')
    call check_true(abs(z1(5) - 1.25_dp) <= 1.0e-16_dp, 'Pure-box fixed z(5) == 1.25')
    call check_true(abs(z1(6) + 0.75_dp) <= 1.0e-16_dp, 'Pure-box fixed z(6) == -0.75')
    call check_true(abs(z1(3) - 0.5_dp) <= 1.0e-15_dp, 'Pure-box active upper z(3) == 0.5')
    call check_true(abs(z1(4) + 0.5_dp) <= 1.0e-15_dp, 'Pure-box active lower z(4) == -0.5')

    ! C-ABI fortchuck_solve with R = 0 and NULL Stiefel pointers
    st = fortchuck_solve(d, 0_i32, c_null_ptr, c_null_ptr, &
      c_loc(z2(1)), c_loc(l(1)), c_loc(u(1)), c_null_ptr, &
      c_funloc(cb_pure_box12), c_null_ptr, c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res2))
    call check_true(st == FC_CONVERGED_GRAD, 'Pure-box C-ABI fortchuck_solve CONVERGED_GRAD')
    call check_true(abs(res1%f_val - res2%f_val) <= 1.0e-15_dp, 'Pure-box native == C-ABI parity')

    ! C-ABI fortchuck_kkt_residual with R = 0 and NULL Stiefel pointers
    dummy_GX(1) = 0.0_dp
    rc = cb_pure_box12(d, 0_i32, dummy_n, dummy_n, z2, dummy_X, f_chk, gz_chk, dummy_GX, c_null_ptr)
    call check_true(rc == 0_i32, 'Pure-box callback check')
    st = fortchuck_kkt_residual(d, 0_i32, c_null_ptr, c_null_ptr, &
      c_loc(z2(1)), c_loc(l(1)), c_loc(u(1)), c_null_ptr, &
      c_loc(gz_chk(1)), c_null_ptr, FC_METRIC_CANONICAL, &
      c_loc(work(1)), lwork_q, c_loc(kkt_chk), c_loc(orth_chk))
    call check_true(st == FC_SUCCESS_QUERY, 'Pure-box fortchuck_kkt_residual status==0')
    call check_true(abs(kkt_chk - res2%kkt_residual) <= 1.0e-14_dp, 'Pure-box KKT residual match')
    call check_true(abs(orth_chk) <= 1.0e-16_dp, 'Pure-box orth_err == 0')
  end subroutine test_pure_box_regimes

  !---------------------------------------------------------------------------
  ! 2. Pure-Stiefel regimes (d = 0):
  !    - Sphere St(10, 1) scalar fast-path
  !    - Branch A (2p < n) Brockett on St(12, 3) with C-ABI z/l/u == NULL
  !    - Branch B (2p >= n) SO(5) Wahba preserving det(X*) == +1
  !---------------------------------------------------------------------------
  subroutine test_pure_stiefel_branch_a_and_so5()
    integer(i32), target       :: n_sph(1), p_sph(1), n_br(1), p_br(1), n_so(1), p_so(1)
    real(dp), target           :: X_sph(10), X_br(36), X_so(25)
    real(dp)                   :: Q12(12, 12), AQ(12, 12), R_true(5, 5), tau_qr(12), w_qr(12)
    real(dp), target           :: work(16384)
    integer(i32), target       :: iwork(1024)
    type(brockett12_ctx_t), target :: br_ctx
    type(so5_ctx_t), target        :: so_ctx
    type(fc_options_t), target     :: opts
    type(fc_result_t), target      :: res
    integer(i64), target           :: lwork_q, liwork_q
    integer(i32)                   :: info, st, i, j
    real(dp)                       :: det_init, det_final

    call fc_init_default_options(opts)
    opts%gtol = 1.0e-10_dp

    ! 2a. Sphere St(10, 1)
    n_sph(1) = 10_i32
    p_sph(1) = 1_i32
    call fill_rand_vec(10_i32, X_sph, rng_state)
    st = fortchuck_workspace_query(0_i32, 1_i32, c_loc(n_sph(1)), c_loc(p_sph(1)), &
      opts%memory, c_loc(lwork_q), c_loc(liwork_q))
    call check_true(st == FC_SUCCESS_QUERY, 'Sphere workspace query')
    st = fortchuck_solve(0_i32, 1_i32, c_loc(n_sph(1)), c_loc(p_sph(1)), &
      c_null_ptr, c_null_ptr, c_null_ptr, c_loc(X_sph(1)), &
      c_funloc(cb_sphere10), c_null_ptr, c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_CONVERGED_GRAD, 'Sphere St(10,1) CONVERGED_GRAD')
    call check_true(abs(res%f_val - 0.75_dp) <= 1.0e-12_dp, 'Sphere St(10,1) exact f* == 0.75')
    call check_true(res%orth_error <= 1.0e-14_dp, 'Sphere St(10,1) orth <= 1e-14')

    ! 2b. Branch A St(12, 3) Brockett with dense A = Q * diag(1..12) * Q^T, w = [3, 2, 1] -> f* = 5.0
    n_br(1) = 12_i32
    p_br(1) = 3_i32
    call fill_rand_vec(144_i32, Q12, rng_state)
    call fc_geqrf_pos(12_i32, 12_i32, Q12, tau_qr, w_qr, info)
    call check_true(info == 0_i32, 'Q12 orthogonal setup')
    do j = 1_i32, 12_i32
      do i = 1_i32, 12_i32
        AQ(i, j) = Q12(i, j) * real(j, dp)
      end do
    end do
    call fc_gemm_nt(12_i32, 12_i32, 12_i32, 1.0_dp, AQ, 12_i32, Q12, 12_i32, 0.0_dp, br_ctx%A, 12_i32)
    br_ctx%w = [3.0_dp, 2.0_dp, 1.0_dp]

    call fill_rand_vec(36_i32, X_br, rng_state)
    st = fortchuck_workspace_query(0_i32, 1_i32, c_loc(n_br(1)), c_loc(p_br(1)), &
      opts%memory, c_loc(lwork_q), c_loc(liwork_q))
    call check_true(st == FC_SUCCESS_QUERY, 'Branch A St(12,3) workspace query')
    st = fortchuck_solve(0_i32, 1_i32, c_loc(n_br(1)), c_loc(p_br(1)), &
      c_null_ptr, c_null_ptr, c_null_ptr, c_loc(X_br(1)), &
      c_funloc(cb_brockett12), c_loc(br_ctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_CONVERGED_GRAD, 'Branch A St(12,3) CONVERGED_GRAD')
    call check_true(abs(res%f_val - 5.0_dp) <= 1.0e-12_dp, 'Branch A St(12,3) exact f* == 5.0')
    call check_true(res%kkt_residual <= 1.0e-10_dp, 'Branch A St(12,3) KKT <= 1e-10')
    call check_true(res%orth_error <= 1.0e-13_dp, 'Branch A St(12,3) orth <= 1e-13')

    ! 2c. Branch B SO(5) Wahba: verify det(X*) == +1 preserved invariant!
    n_so(1) = 5_i32
    p_so(1) = 5_i32
    call fill_rand_vec(25_i32, R_true, rng_state)
    call fc_geqrf_pos(5_i32, 5_i32, R_true, tau_qr(1:5), w_qr(1:5), info)
    if (det_5x5(R_true) < 0.0_dp) then
      do i = 1_i32, 5_i32
        R_true(i, 1) = -R_true(i, 1)
      end do
    end if
    call check_true(abs(det_5x5(R_true) - 1.0_dp) <= 1.0e-14_dp, 'SO(5) R_true det == +1')

    call fill_rand_vec(25_i32, so_ctx%B, rng_state)
    do i = 1_i32, 5_i32
      so_ctx%B(i, i) = so_ctx%B(i, i) + 2.0_dp
    end do
    call fc_gemm_nn(5_i32, 5_i32, 5_i32, 1.0_dp, R_true, 5_i32, so_ctx%B, 5_i32, &
      0.0_dp, so_ctx%C, 5_i32)

    call fill_rand_vec(25_i32, X_so, rng_state)
    call fc_geqrf_pos(5_i32, 5_i32, X_so, tau_qr(1:5), w_qr(1:5), info)
    det_init = det_5x5(X_so)
    if (det_init < 0.0_dp) then
      do i = 1_i32, 5_i32
        X_so(i) = -X_so(i)
      end do
    end if
    det_init = det_5x5(X_so)
    call check_true(abs(det_init - 1.0_dp) <= 1.0e-14_dp, 'SO(5) X0 det == +1')

    st = fortchuck_workspace_query(0_i32, 1_i32, c_loc(n_so(1)), c_loc(p_so(1)), &
      opts%memory, c_loc(lwork_q), c_loc(liwork_q))
    st = fortchuck_solve(0_i32, 1_i32, c_loc(n_so(1)), c_loc(p_so(1)), &
      c_null_ptr, c_null_ptr, c_null_ptr, c_loc(X_so(1)), &
      c_funloc(cb_so5_wahba), c_loc(so_ctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_CONVERGED_GRAD, 'Branch B SO(5) CONVERGED_GRAD')
    call check_true(res%f_val <= 1.0e-22_dp, 'Branch B SO(5) recovers exact R_true (f <= 1e-22)')
    call check_true(res%orth_error <= 1.0e-13_dp, 'Branch B SO(5) orth <= 1e-13')
    det_final = det_5x5(X_so)
    call check_true(abs(det_final - 1.0_dp) <= 1.0e-13_dp, 'Branch B SO(5) preserves det(X*) == +1')
  end subroutine test_pure_stiefel_branch_a_and_so5

  !---------------------------------------------------------------------------
  ! 3. Multi-factor Coupled [l, u]^6 x St(10, 2) x St(4, 4)
  !---------------------------------------------------------------------------
  subroutine test_multifactor_coupled_solver()
    integer(i32), parameter    :: d = 6_i32, R = 2_i32
    integer(i32), target       :: n_arr(2), p_arr(2), iwork(1024)
    real(dp), target           :: z(d), l(d), u(d), X_packed(36), gz_chk(d), GX_chk(36)
    real(dp), target           :: work(16384), kkt_chk, orth_chk
    type(fc_options_t), target :: opts
    type(fc_result_t), target  :: res
    integer(i64), target       :: lwork_q, liwork_q
    integer(i32)               :: st, rc
    real(dp)                   :: f_chk

    n_arr = [10_i32, 4_i32]
    p_arr = [ 2_i32, 4_i32]

    l = [0.1_dp, 0.1_dp, 0.5_dp, 0.1_dp, 0.1_dp, 0.1_dp]
    u = [1.2_dp, 1.2_dp, 0.5_dp, 1.2_dp, 1.2_dp, 1.2_dp]
    call fill_rand_vec(d, z, rng_state)
    call fill_rand_vec(36_i32, X_packed, rng_state)

    call fc_init_default_options(opts)
    opts%gtol = 1.0e-9_dp

    call fc_workspace_query(d, R, n_arr, p_arr, opts%memory, lwork_q, liwork_q, st)
    call check_true(st == FC_SUCCESS_QUERY, 'Multi-factor workspace query')

    call fc_solve(d, R, n_arr, p_arr, z, l, u, X_packed, &
      cb_coupled_multi, c_null_ptr, opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_CONVERGED_GRAD, 'Multi-factor [l,u]^6 x St(10,2) x St(4,4) CONVERGED')
    call check_true(res%kkt_residual <= 1.0e-9_dp, 'Multi-factor KKT <= 1e-9')
    call check_true(res%orth_error <= 1.0e-13_dp, 'Multi-factor orth_error <= 1e-13')
    call check_true(abs(z(3) - 0.5_dp) <= 1.0e-16_dp, 'Multi-factor fixed z(3) == 0.5')

    gz_chk = 0.0_dp
    GX_chk = 0.0_dp
    rc = cb_coupled_multi(d, R, n_arr, p_arr, z, X_packed, f_chk, gz_chk, GX_chk, c_null_ptr)
    call check_true(rc == 0_i32, 'Multi-factor callback verify')
    st = fortchuck_kkt_residual(d, R, c_loc(n_arr(1)), c_loc(p_arr(1)), &
      c_loc(z(1)), c_loc(l(1)), c_loc(u(1)), c_loc(X_packed(1)), &
      c_loc(gz_chk(1)), c_loc(GX_chk(1)), opts%metric, &
      c_loc(work(1)), lwork_q, c_loc(kkt_chk), c_loc(orth_chk))
    call check_true(st == FC_SUCCESS_QUERY, 'Multi-factor fortchuck_kkt_residual status==0')
    call check_true(abs(kkt_chk - res%kkt_residual) <= 1.0e-14_dp, 'Multi-factor KKT parity')
    call check_true(abs(orth_chk - res%orth_error) <= 1.0e-15_dp, 'Multi-factor orth parity')
  end subroutine test_multifactor_coupled_solver

  !---------------------------------------------------------------------------
  ! 4. Certified Ky Fan Ground-Truth Box-Coupled Brockett on [0.2, 3.0]^3 x St(14, 3)
  !    Tested across BOTH metrics, BOTH damping modes, and memory = 0 vs memory = 7!
  !---------------------------------------------------------------------------
  subroutine test_ky_fan_box_coupled_brockett_and_modes()
    integer(i32), parameter     :: d = 3_i32, R = 1_i32
    integer(i32), target        :: n_arr(1), p_arr(1), iwork(1024)
    real(dp), target            :: z0(3), z(3), l(3), u(3), X0(42), X(42), work(16384)
    type(ky_fan_ctx_t), target  :: ctx
    type(fc_options_t)          :: opts
    type(fc_result_t)           :: res
    real(dp)                    :: Q14(14, 14), AQ(14, 14), alphas(14), tau_qr(14), w_qr(14)
    real(dp)                    :: M_lu(3, 3), rhs_w(3, 1), z_star(3), f_star
    integer(i32)                :: ipiv3(3), info, st, i, j, m_idx
    integer(i32)                :: metrics(5), damps(5), mems(5)
    integer(i64)                :: lwork_q, liwork_q

    n_arr(1) = 14_i32
    p_arr(1) = 3_i32
    l        = [0.2_dp, 0.2_dp, 0.2_dp]
    u        = [3.0_dp, 3.0_dp, 3.0_dp]

    alphas(1) = -8.0_dp
    alphas(2) = -5.0_dp
    alphas(3) = -2.5_dp
    do i = 4_i32, 14_i32
      alphas(i) = real(i - 3_i32, dp) * 1.25_dp
    end do

    call fill_rand_vec(196_i32, Q14, rng_state)
    call fc_geqrf_pos(14_i32, 14_i32, Q14, tau_qr, w_qr, info)
    call check_true(info == 0_i32, 'Q14 orthogonal setup')
    do j = 1_i32, 14_i32
      do i = 1_i32, 14_i32
        AQ(i, j) = Q14(i, j) * alphas(j)
      end do
    end do
    call fc_gemm_nt(14_i32, 14_i32, 14_i32, 1.0_dp, AQ, 14_i32, Q14, 14_i32, 0.0_dp, ctx%A, 14_i32)

    ! Supermodular SPD coupling matrix M (M_ij < 0 off-diagonal)
    ctx%M(1, 1) = 10.0_dp; ctx%M(1, 2) = -1.0_dp; ctx%M(1, 3) = -0.5_dp
    ctx%M(2, 1) = -1.0_dp; ctx%M(2, 2) = 10.0_dp; ctx%M(2, 3) = -1.0_dp
    ctx%M(3, 1) = -0.5_dp; ctx%M(3, 2) = -1.0_dp; ctx%M(3, 3) = 10.0_dp

    ! Target z* = [3.0 (upper active), 1.6 (interior), 0.2 (lower active)]
    ! Grad_z* = [-4.0, 0.0, +3.5] => M * (z* - z_ref) = [4.0, 5.0, 6.0]
    z_star     = [3.0_dp, 1.6_dp, 0.2_dp]
    rhs_w(:,1) = [4.0_dp, 5.0_dp, 6.0_dp]
    M_lu       = ctx%M
    call fc_lu_factor(3_i32, M_lu, ipiv3, 1.0e-12_dp, info)
    call fc_lu_solve(3_i32, 1_i32, M_lu, ipiv3, rhs_w)
    do i = 1_i32, 3_i32
      ctx%z_ref(i) = z_star(i) - rhs_w(i, 1)
    end do

    f_star = alphas(1) * z_star(1) + alphas(2) * z_star(2) + alphas(3) * z_star(3) + &
      0.5_dp * (rhs_w(1, 1) * 4.0_dp + rhs_w(2, 1) * 5.0_dp + rhs_w(3, 1) * 6.0_dp)

    z0 = [0.7_dp, 2.4_dp, 1.9_dp]
    call fill_rand_vec(42_i32, X0, rng_state)

    metrics = [FC_METRIC_CANONICAL, FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, &
               FC_METRIC_EUCLIDEAN, FC_METRIC_CANONICAL]
    damps   = [FC_DAMP_CAUTIOUS, FC_DAMP_POWELL, FC_DAMP_CAUTIOUS, &
               FC_DAMP_POWELL,   FC_DAMP_CAUTIOUS]
    mems    = [7_i32, 7_i32, 7_i32, 7_i32, 0_i32]

    do m_idx = 1_i32, 5_i32
      call fc_init_default_options(opts)
      opts%gtol         = 1.0e-9_dp
      opts%metric       = metrics(m_idx)
      opts%damping_mode = damps(m_idx)
      opts%memory       = mems(m_idx)
      opts%max_iter     = 2000_i32
      opts%max_eval     = 10000_i32

      z = z0
      X = X0
      call fc_workspace_query(d, R, n_arr, p_arr, opts%memory, lwork_q, liwork_q, st)
      call check_true(st == FC_SUCCESS_QUERY, 'Ky Fan workspace query')

      call fc_solve(d, R, n_arr, p_arr, z, l, u, X, &
        cb_ky_fan_brockett, c_loc(ctx), opts, work, lwork_q, iwork, liwork_q, res)

      call check_true(res%status == FC_CONVERGED_GRAD, 'Ky Fan mode CONVERGED_GRAD')
      call check_true(abs(res%f_val - f_star) <= 1.0e-11_dp, 'Ky Fan |f - f*| <= 1e-11')
      call check_true(res%kkt_residual <= 1.0e-8_dp, 'Ky Fan E_KKT <= 1e-8')
      call check_true(res%orth_error <= 1.0e-13_dp, 'Ky Fan orth_err <= 1e-13')
      do i = 1_i32, 3_i32
        call check_true(abs(z(i) - z_star(i)) <= 1.0e-8_dp, 'Ky Fan z == z_star')
      end do
    end do
  end subroutine test_ky_fan_box_coupled_brockett_and_modes

  !---------------------------------------------------------------------------
  ! 5. FM-2 clamped multiplier test, domain barrier recovery, polish re-proj,
  !    max_eval hard cap, fortchuck_project_stiefel, and C-ABI error codes
  !---------------------------------------------------------------------------
  subroutine test_fm2_guards_and_c_abi_errors()
    integer(i32), target        :: n_fm(1), p_fm(1), n_bar(1), p_bar(1), iwork(1024)
    real(dp), target            :: z_fm(1), l_fm(1), u_fm(1), X_fm(24)
    real(dp), target            :: z_bar(2), l_bar(2), u_bar(2), X_bar(8)
    real(dp), target            :: X_proj(18), work(8192), w_q(1)
    integer(i32), target        :: iw_q(1)
    type(barrier_ctx_t), target :: bctx
    type(fc_options_t), target  :: opts
    type(fc_result_t), target   :: res
    integer(i64)                :: lwork_q, liwork_q
    integer(i32)                :: st, i

    ! 5a. FM-2 clamped Lagrange-multiplier Stiefel injection guard (<= 35 iters to 1e-10!)
    n_fm(1) = 8_i32
    p_fm(1) = 3_i32
    z_fm(1) = 0.65_dp
    l_fm(1) = 0.0_dp
    u_fm(1) = 1.0_dp
    call fill_rand_vec(24_i32, X_fm, rng_state)

    call fc_init_default_options(opts)
    opts%gtol     = 1.0e-10_dp
    opts%orth_tol = 1.0e-15_dp  ! Also exercises pre-callback fc_geqrf_pos + history re-projection!
    call fc_workspace_query(1_i32, 1_i32, n_fm, p_fm, opts%memory, lwork_q, liwork_q, st)
    call fc_solve(1_i32, 1_i32, n_fm, p_fm, z_fm, l_fm, u_fm, X_fm, &
      cb_fm2_multiplier, c_null_ptr, opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_CONVERGED_GRAD, 'FM-2 multiplier test CONVERGED_GRAD')
    call check_true(res%kkt_residual <= 1.0e-10_dp, 'FM-2 multiplier test KKT <= 1e-10')
    call check_true(res%n_iter <= 35_i32, 'FM-2 multiplier superlinear <= 35 iters')
    call check_true(abs(z_fm(1)) <= 1.0e-16_dp, 'FM-2 z(1)* == 0')

    ! 5b. Domain barrier rc = +1 halving recovery & rc = -1 immediate abort
    n_bar(1) = 4_i32
    p_bar(1) = 2_i32
    l_bar    = [0.0_dp, -2.0_dp]
    u_bar    = [5.0_dp,  2.0_dp]
    z_bar    = [3.5_dp,  1.2_dp]
    call fill_rand_vec(8_i32, X_bar, rng_state)

    call fc_init_default_options(opts)
    opts%gtol  = 1.0e-9_dp
    bctx%mode  = 1_i32
    bctx%calls = 0_i32
    call fc_workspace_query(2_i32, 1_i32, n_bar, p_bar, opts%memory, lwork_q, liwork_q, st)
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_CONVERGED_GRAD, 'Domain barrier rc=+1 recovers & converges')
    call check_true(z_bar(1) > 0.25_dp, 'Domain barrier respected z(1) > 0.25')

    ! Callback abort rc = -1 on eval 3
    z_bar      = [3.5_dp, 1.2_dp]
    bctx%mode  = -1_i32
    bctx%calls = 0_i32
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_ERR_CALLBACK_ABORT, 'Callback rc < 0 returns FC_ERR_CALLBACK_ABORT')
    call check_true(res%n_eval == 3_i32, 'Callback abort exact n_eval == 3')

    ! Hard max_eval budget check (n_eval never exceeds max_eval!)
    z_bar      = [3.5_dp, 1.2_dp]
    bctx%mode  = 0_i32
    bctx%calls = 0_i32
    opts%max_eval = 4_i32
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_MAX_EVAL_REACHED, 'Hard max_eval returns FC_MAX_EVAL_REACHED')
    call check_true(res%n_eval == 4_i32 .and. bctx%calls == 4_i32, 'Strict n_eval == max_eval == 4')

    ! max_iter = 2 check
    z_bar         = [3.5_dp, 1.2_dp]
    opts%max_eval = 5000_i32
    opts%max_iter = 2_i32
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_MAX_ITER_REACHED, 'max_iter returns FC_MAX_ITER_REACHED')
    call check_true(res%n_iter == 2_i32, 'n_iter == 2')

    ! Explicit custom ftol (status = 2) and xtol (status = 3) stopping checks
    z_bar         = [3.5_dp, 1.2_dp]
    call fill_rand_vec(8_i32, X_bar, rng_state)
    opts%max_iter = 1000_i32
    opts%gtol     = 0.0_dp
    opts%ftol     = 1.0e-4_dp
    opts%xtol     = 0.0_dp
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_CONVERGED_FTOL, 'explicit ftol returns FC_CONVERGED_FTOL')

    z_bar         = [3.5_dp, 1.2_dp]
    call fill_rand_vec(8_i32, X_bar, rng_state)
    opts%gtol     = 0.0_dp
    opts%ftol     = 0.0_dp
    opts%xtol     = 1.0e-4_dp
    call fc_solve(2_i32, 1_i32, n_bar, p_bar, z_bar, l_bar, u_bar, X_bar, &
      cb_barrier_or_abort, c_loc(bctx), opts, work, lwork_q, iwork, liwork_q, res)
    call check_true(res%status == FC_CONVERGED_XTOL, 'explicit xtol returns FC_CONVERGED_XTOL')

    ! 5c. C-ABI fortchuck_default_options, query mode lwork=-1, project_stiefel, and error codes
    call fortchuck_default_options(c_null_ptr)
    call fortchuck_default_options(c_loc(opts))
    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
      c_null_funptr, c_null_ptr, c_loc(opts), &
      c_loc(w_q(1)), -1_i64, c_loc(iw_q(1)), -1_i64, c_loc(res))
    call check_true(st == FC_SUCCESS_QUERY, 'fortchuck_solve lwork=-1 query')
    call check_true(int(w_q(1), i64) == lwork_q .and. int(iw_q(1), i64) == liwork_q, &
      'fortchuck_solve lwork=-1 matches fortchuck_workspace_query')

    st = fortchuck_project_stiefel(6_i32, 3_i32, c_null_ptr, c_loc(w_q(1)), -1_i64)
    call check_true(st == FC_SUCCESS_QUERY .and. int(w_q(1), i32) == 6_i32, &
      'fortchuck_project_stiefel query lwork=-1')
    call fill_rand_vec(18_i32, X_proj, rng_state)
    st = fortchuck_project_stiefel(6_i32, 3_i32, c_loc(X_proj(1)), c_loc(work(1)), 6_i64)
    call check_true(st == FC_SUCCESS_QUERY, 'fortchuck_project_stiefel success')
    call check_true(fc_stiefel_orth_err(6_i32, 3_i32, X_proj) <= 1.0e-15_dp, &
      'fortchuck_project_stiefel orth <= 1e-15')

    ! Error codes: -1 (DIM), -2 (NULL), -3 (BOUNDS), -4 (X0), -5 (WORK), -6 (NAN), -9 (OPTS)
    st = fortchuck_solve(0_i32, 0_i32, c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr, &
      c_null_ptr, c_null_ptr, c_funloc(cb_pure_box12), c_null_ptr, c_loc(opts), &
      c_loc(work(1)), 100_i64, c_loc(iwork(1)), 100_i64, c_loc(res))
    call check_true(st == FC_ERR_INVALID_DIM, 'd=0, R=0 -> FC_ERR_INVALID_DIM')

    st = fortchuck_solve(2_i32, 0_i32, c_null_ptr, c_null_ptr, c_null_ptr, c_loc(l_bar(1)), &
      c_loc(u_bar(1)), c_null_ptr, c_funloc(cb_pure_box12), c_null_ptr, c_loc(opts), &
      c_loc(work(1)), 4096_i64, c_loc(iwork(1)), 256_i64, c_loc(res))
    call check_true(st == FC_ERR_NULL_PTR, 'd>0, z=NULL -> FC_ERR_NULL_PTR')

    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_loc(z_bar(1)), c_loc(l_bar(1)), c_loc(u_bar(1)), c_loc(X_bar(1)), &
      c_funloc(cb_barrier_or_abort), c_loc(bctx), c_loc(opts), &
      c_loc(work(1)), 8_i64, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_ERR_WORK_TOO_SMALL, 'undersized lwork -> FC_ERR_WORK_TOO_SMALL')

    l_bar(1) = 3.0_dp
    u_bar(1) = 1.0_dp
    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_loc(z_bar(1)), c_loc(l_bar(1)), c_loc(u_bar(1)), c_loc(X_bar(1)), &
      c_funloc(cb_barrier_or_abort), c_loc(bctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_ERR_INVALID_BOUNDS, 'l > u -> FC_ERR_INVALID_BOUNDS')

    l_bar(1) = 0.0_dp
    u_bar(1) = 5.0_dp
    do i = 1_i32, 8_i32
      X_bar(i) = 0.0_dp
    end do
    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_loc(z_bar(1)), c_loc(l_bar(1)), c_loc(u_bar(1)), c_loc(X_bar(1)), &
      c_funloc(cb_barrier_or_abort), c_loc(bctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_ERR_INVALID_X0, 'rank-deficient X0 -> FC_ERR_INVALID_X0')

    call fill_rand_vec(8_i32, X_bar, rng_state)
    z_bar(1)  = 0.1_dp   ! Inside forbidden barrier at p_0 -> rc = +1 -> FC_ERR_CALLBACK_NAN
    bctx%mode = 1_i32
    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_loc(z_bar(1)), c_loc(l_bar(1)), c_loc(u_bar(1)), c_loc(X_bar(1)), &
      c_funloc(cb_barrier_or_abort), c_loc(bctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_ERR_CALLBACK_NAN, 'p_0 rc > 0 -> FC_ERR_CALLBACK_NAN')

    opts%c1_armijo = 1.5_dp
    st = fortchuck_solve(2_i32, 1_i32, c_loc(n_bar(1)), c_loc(p_bar(1)), &
      c_loc(z_bar(1)), c_loc(l_bar(1)), c_loc(u_bar(1)), c_loc(X_bar(1)), &
      c_funloc(cb_barrier_or_abort), c_loc(bctx), c_loc(opts), &
      c_loc(work(1)), lwork_q, c_loc(iwork(1)), liwork_q, c_loc(res))
    call check_true(st == FC_ERR_INVALID_OPTS, 'c1_armijo >= 1 -> FC_ERR_INVALID_OPTS')
  end subroutine test_fm2_guards_and_c_abi_errors

end program test_coupled_solver
