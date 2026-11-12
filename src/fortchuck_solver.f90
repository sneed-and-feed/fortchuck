module fortchuck_solver
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: &
    dp, i32, i64, c_ptr, &
    FC_EPS, FC_HUGE, &
    FC_SUCCESS_QUERY, FC_CONVERGED_GRAD, FC_CONVERGED_FTOL, FC_CONVERGED_XTOL, &
    FC_MAX_ITER_REACHED, FC_MAX_EVAL_REACHED, &
    FC_ERR_INVALID_DIM, FC_ERR_INVALID_BOUNDS, FC_ERR_INVALID_X0, &
    FC_ERR_WORK_TOO_SMALL, FC_ERR_CALLBACK_NAN, FC_ERR_CALLBACK_ABORT, &
    FC_ERR_LS_FAILED, FC_ERR_INVALID_OPTS, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, &
    FC_DAMP_CAUTIOUS, FC_DAMP_POWELL, &
    fc_options_t, fc_result_t, fc_eval_fg_cb, fc_init_result
  use fortchuck_linalg, only: fc_frob_norm, fc_geqrf_pos
  use fortchuck_box, only: &
    fc_box_validate, fc_box_project, fc_box_kkt_residual, &
    fc_box_partition, fc_box_dir_deriv0, fc_box_trial_step
  use fortchuck_stiefel, only: &
    fc_stiefel_offsets, fc_stiefel_rgrad, fc_stiefel_proj_tangent, &
    fc_stiefel_inner, fc_stiefel_orth_err, &
    fc_cayley_setup, fc_cayley_retract, fc_cayley_transport_vec
  use fortchuck_lbfgs, only: &
    fc_lbfgs_direction, fc_abb_direction, fc_lbfgs_transport_and_push
  implicit none (type, external)
  private

  public :: fc_validate_options
  public :: fc_workspace_query
  public :: fc_solve
  public :: fc_project_stiefel
  public :: fc_kkt_residual

contains

  !---------------------------------------------------------------------------
  ! Validate fc_options_t hyperparameters (100% pure).
  !---------------------------------------------------------------------------
  pure subroutine fc_validate_options(opts, status)
    type(fc_options_t), intent(in) :: opts
    integer(i32), intent(out)      :: status

    if (opts%memory < 0_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    if (ieee_is_nan(opts%gtol) .or. .not. ieee_is_finite(opts%gtol) .or. &
        opts%gtol < 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%ftol) .or. .not. ieee_is_finite(opts%ftol) .or. &
        opts%ftol < 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%xtol) .or. .not. ieee_is_finite(opts%xtol) .or. &
        opts%xtol < 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%c1_armijo) .or. .not. ieee_is_finite(opts%c1_armijo) .or. &
        opts%c1_armijo <= 0.0_dp .or. opts%c1_armijo >= 1.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%zh_eta) .or. .not. ieee_is_finite(opts%zh_eta) .or. &
        opts%zh_eta >= 1.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%eps_band) .or. .not. ieee_is_finite(opts%eps_band) .or. &
        opts%eps_band < 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%orth_tol) .or. .not. ieee_is_finite(opts%orth_tol) .or. &
        opts%orth_tol <= 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (ieee_is_nan(opts%max_step_vel) .or. .not. ieee_is_finite(opts%max_step_vel) .or. &
        opts%max_step_vel <= 0.0_dp) then
      status = FC_ERR_INVALID_OPTS
      return
    end if

    if (opts%max_iter < 0_i32 .or. opts%max_eval < 1_i32 .or. opts%max_ls_steps < 1_i32) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (opts%metric /= FC_METRIC_CANONICAL .and. opts%metric /= FC_METRIC_EUCLIDEAN) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (opts%damping_mode /= FC_DAMP_CAUTIOUS .and. opts%damping_mode /= FC_DAMP_POWELL) then
      status = FC_ERR_INVALID_OPTS
      return
    end if
    if (opts%project_initial /= 0_i32 .and. opts%project_initial /= 1_i32) then
      status = FC_ERR_INVALID_OPTS
      return
    end if

    status = FC_SUCCESS_QUERY
  end subroutine fc_validate_options

  !---------------------------------------------------------------------------
  ! Internal helper computing exact padded workspace dimensions (ARCHITECTURE_SPEC 3.4).
  !---------------------------------------------------------------------------
  pure subroutine compute_workspace_dims(d, R, n_arr, p_arr, m, &
      d_pad, n_st_pad, n_tot_pad, n_skew_pad, m_pad, n_cay_pad, n_piv_pad, n_scr, &
      lwork_min, liwork_min, status)
    integer(i32), intent(in)  :: d, R, m
    integer(i32), intent(in)  :: n_arr(*), p_arr(*)
    integer(i64), intent(out) :: d_pad, n_st_pad, n_tot_pad, n_skew_pad
    integer(i64), intent(out) :: m_pad, n_cay_pad, n_piv_pad, n_scr
    integer(i64), intent(out) :: lwork_min, liwork_min
    integer(i32), intent(out) :: status

    integer(i32) :: r_idx, nr, pr
    integer(i64) :: d64, m64, rp1_64, nr64, pr64, cr64, qr64, scr_r
    integer(i64) :: n_st, n_skew, n_cay, n_piv, max_scr_r, lim64

    d_pad      = 1_i64
    n_st_pad   = 1_i64
    n_tot_pad  = 1_i64
    n_skew_pad = 1_i64
    m_pad      = 1_i64
    n_cay_pad  = 1_i64
    n_piv_pad  = 1_i64
    n_scr      = 64_i64
    lwork_min  = 0_i64
    liwork_min = 0_i64

    if (d < 0_i32 .or. R < 0_i32 .or. (d == 0_i32 .and. R == 0_i32) .or. m < 0_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    d64       = int(d, i64)
    m64       = int(m, i64)
    rp1_64    = int(R, i64) + 1_i64
    lim64     = int(huge(0_i32) - 1_i32, i64)
    n_st      = 0_i64
    n_skew    = 0_i64
    n_cay     = 0_i64
    n_piv     = 0_i64
    max_scr_r = 0_i64

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
      scr_r = 2_i64 * nr64 * pr64 + 6_i64 * pr64 * pr64 + 4_i64 * pr64

      n_st   = n_st   + nr64 * pr64
      n_skew = n_skew + pr64 * pr64
      n_cay  = n_cay  + cr64
      n_piv  = n_piv  + qr64
      if (scr_r > max_scr_r) max_scr_r = scr_r

      if (d64 + n_st > lim64 .or. n_skew > lim64 .or. &
          n_cay > lim64 .or. n_piv > lim64 .or. scr_r > lim64) then
        status = FC_ERR_INVALID_DIM
        return
      end if
    end do

    d_pad      = max(d64, 1_i64)
    n_st_pad   = max(n_st, 1_i64)
    n_tot_pad  = max(d64 + n_st, 1_i64)
    n_skew_pad = max(1_i64, n_skew)
    m_pad      = max(m64, 1_i64)
    n_cay_pad  = max(1_i64, n_cay)
    n_piv_pad  = max(1_i64, n_piv)
    n_scr      = max(64_i64, max_scr_r)

    lwork_min = 5_i64 * d_pad + 6_i64 * n_st_pad + 3_i64 * n_tot_pad + &
      n_skew_pad + 4_i64 * rp1_64 + 2_i64 * m64 * n_tot_pad + &
      2_i64 * m64 * n_skew_pad + 5_i64 * m_pad + n_cay_pad + n_scr

    liwork_min = d_pad + 2_i64 * m_pad + 4_i64 * rp1_64 + n_piv_pad
    status     = FC_SUCCESS_QUERY
  end subroutine compute_workspace_dims

  !---------------------------------------------------------------------------
  ! Exact caller-owned workspace query (100% pure).
  !---------------------------------------------------------------------------
  pure subroutine fc_workspace_query(d, R, n_arr, p_arr, m, lwork, liwork, status)
    integer(i32), intent(in)  :: d, R, m
    integer(i32), intent(in)  :: n_arr(*), p_arr(*)
    integer(i64), intent(out) :: lwork, liwork
    integer(i32), intent(out) :: status

    integer(i64) :: d_pad, n_st_pad, n_tot_pad, n_skew_pad
    integer(i64) :: m_pad, n_cay_pad, n_piv_pad, n_scr

    call compute_workspace_dims(d, R, n_arr, p_arr, m, &
      d_pad, n_st_pad, n_tot_pad, n_skew_pad, m_pad, n_cay_pad, n_piv_pad, n_scr, &
      lwork, liwork, status)
  end subroutine fc_workspace_query

  !---------------------------------------------------------------------------
  ! Overflow-safe hypot(a, b) = sqrt(a^2 + b^2).
  !---------------------------------------------------------------------------
  pure function safe_hypot(a, b) result(val)
    real(dp), intent(in) :: a, b
    real(dp)             :: val
    real(dp)             :: x, y, mx, mn, q

    x  = abs(a)
    y  = abs(b)
    mx = max(x, y)
    mn = min(x, y)
    if (.not. (mx > 0.0_dp)) then
      val = 0.0_dp
    else if (.not. ieee_is_finite(mx)) then
      val = mx
    else
      q   = mn / mx
      val = mx * sqrt(1.0_dp + q * q)
    end if
  end function safe_hypot

  !---------------------------------------------------------------------------
  ! Overflow-safe abs(a - b) avoiding FE_OVERFLOW near +-FC_HUGE.
  !---------------------------------------------------------------------------
  pure function safe_abs_diff(a, b) result(ad)
    real(dp), intent(in) :: a, b
    real(dp)             :: ad
    real(dp)             :: hd

    if (abs(a) <= 0.45_dp * FC_HUGE .and. abs(b) <= 0.45_dp * FC_HUGE) then
      ad = abs(a - b)
    else
      hd = abs(0.5_dp * a - 0.5_dp * b)
      if (hd >= 0.5_dp * FC_HUGE) then
        ad = FC_HUGE
      else
        ad = 2.0_dp * hd
      end if
    end if
  end function safe_abs_diff

  !---------------------------------------------------------------------------
  ! Overflow-safe Stiefel metric norm ||U||_{X, metric} from U and A_u = X^T*U.
  !---------------------------------------------------------------------------
  pure function stiefel_norm_from_skew(n, p, U, A_u, metric) result(nrm)
    integer(i32), intent(in) :: n, p, metric
    real(dp), intent(in)     :: U(n, p), A_u(p, p)
    real(dp)                 :: nrm
    real(dp)                 :: nrm_f, nrm_a, q_rat

    nrm_f = fc_frob_norm(n, p, U)
    if (.not. (nrm_f > 0.0_dp) .or. metric == FC_METRIC_EUCLIDEAN) then
      nrm = nrm_f
      return
    end if
    nrm_a = fc_frob_norm(p, p, A_u)
    if (.not. ieee_is_finite(nrm_f) .or. .not. ieee_is_finite(nrm_a)) then
      nrm = nrm_f
      return
    end if
    q_rat = nrm_a / nrm_f
    nrm   = nrm_f * sqrt(max(0.0_dp, 1.0_dp - 0.5_dp * (q_rat * q_rat)))
  end function stiefel_norm_from_skew

  !---------------------------------------------------------------------------
  ! Check finiteness of trial callback outputs (f, gz, GX).
  !---------------------------------------------------------------------------
  pure function is_trial_finite(d, n_st_tot, f_val, gz, GX) result(ok)
    integer(i32), intent(in) :: d, n_st_tot
    real(dp), intent(in)     :: f_val
    real(dp), intent(in)     :: gz(*), GX(*)
    logical                  :: ok

    integer(i32) :: i

    ok = .false.
    if (ieee_is_nan(f_val) .or. .not. ieee_is_finite(f_val)) return

    do i = 1_i32, d
      if (ieee_is_nan(gz(i)) .or. .not. ieee_is_finite(gz(i))) return
    end do

    do i = 1_i32, n_st_tot
      if (ieee_is_nan(GX(i)) .or. .not. ieee_is_finite(GX(i))) return
    end do

    ok = .true.
  end function is_trial_finite

  !---------------------------------------------------------------------------
  ! Standalone Stiefel projection utility (100% pure, supports lwork = -1 query).
  !---------------------------------------------------------------------------
  pure subroutine fc_project_stiefel(n, p, X, work, lwork, status)
    integer(i32), intent(in)  :: n, p
    real(dp), intent(inout)   :: X(*)
    real(dp), intent(inout)   :: work(*)
    integer(i64), intent(in)  :: lwork
    integer(i32), intent(out) :: status

    integer(i64) :: lwork_min
    integer(i32) :: info_qr

    if (n < 1_i32 .or. p < 1_i32 .or. p > n) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    lwork_min = 2_i64 * int(p, i64)
    if (lwork == -1_i64) then
      work(1) = real(lwork_min, dp)
      status  = FC_SUCCESS_QUERY
      return
    end if

    if (lwork < lwork_min) then
      status = FC_ERR_WORK_TOO_SMALL
      return
    end if

    call fc_geqrf_pos(n, p, X(1:n * p), work(1:p), work(p + 1_i32:2_i32 * p), info_qr)
    if (info_qr /= 0_i32) then
      status = FC_ERR_INVALID_X0
      return
    end if

    status = FC_SUCCESS_QUERY
  end subroutine fc_project_stiefel

  !---------------------------------------------------------------------------
  ! Standalone coupled KKT residual & manifold orthogonality evaluator (100% pure).
  !---------------------------------------------------------------------------
  pure subroutine fc_kkt_residual( &
      d, R, n_arr, p_arr, z, l, u, X_packed, gz, GX_packed, metric, &
      work, lwork, kkt_inf_out, orth_err_out, status)
    integer(i32), intent(in)  :: d, R, metric
    integer(i32), intent(in)  :: n_arr(*), p_arr(*)
    real(dp), intent(in)      :: z(*), l(*), u(*), X_packed(*)
    real(dp), intent(in)      :: gz(*), GX_packed(*)
    real(dp), intent(inout)   :: work(*)
    integer(i64), intent(in)  :: lwork
    real(dp), intent(out)     :: kkt_inf_out, orth_err_out
    integer(i32), intent(out) :: status

    integer(i32) :: r_idx, nr, pr, np, p2, i, x_pos, st_box
    integer(i64) :: lwork_min, req_r, tot_st, lim64
    real(dp)     :: box_inf, ri_inf, ri_l2, st_inf, rg_nrm, orth_r

    kkt_inf_out  = 0.0_dp
    orth_err_out = 0.0_dp

    if (d < 0_i32 .or. R < 0_i32 .or. (d == 0_i32 .and. R == 0_i32)) then
      status = FC_ERR_INVALID_DIM
      return
    end if
    if (metric /= FC_METRIC_CANONICAL .and. metric /= FC_METRIC_EUCLIDEAN) then
      status = FC_ERR_INVALID_OPTS
      return
    end if

    lwork_min = 1_i64
    tot_st    = 0_i64
    lim64     = int(huge(0_i32) - 1_i32, i64)
    do r_idx = 1_i32, R
      nr = n_arr(r_idx)
      pr = p_arr(r_idx)
      if (nr < 1_i32 .or. pr < 1_i32 .or. pr > nr) then
        status = FC_ERR_INVALID_DIM
        return
      end if
      tot_st = tot_st + int(nr, i64) * int(pr, i64)
      req_r  = int(nr, i64) * int(pr, i64) + 2_i64 * int(pr, i64) * int(pr, i64)
      if (tot_st > lim64 .or. req_r > lim64) then
        status = FC_ERR_INVALID_DIM
        return
      end if
      if (req_r > lwork_min) lwork_min = req_r
    end do

    if (lwork == -1_i64) then
      work(1) = real(lwork_min, dp)
      status  = FC_SUCCESS_QUERY
      return
    end if

    if (lwork < lwork_min) then
      status = FC_ERR_WORK_TOO_SMALL
      return
    end if

    box_inf = 0.0_dp
    if (d > 0_i32) then
      call fc_box_validate(d, l, u, st_box)
      if (st_box /= 0_i32) then
        status = st_box
        return
      end if
      do i = 1_i32, d
        if (ieee_is_nan(z(i)) .or. .not. ieee_is_finite(z(i))) then
          status = FC_ERR_INVALID_X0
          return
        end if
        if (ieee_is_nan(gz(i)) .or. .not. ieee_is_finite(gz(i))) then
          status = FC_ERR_CALLBACK_NAN
          return
        end if
        call fc_box_kkt_residual(1_i32, z(i), l(i), u(i), gz(i), work(1), ri_inf, ri_l2)
        if (ri_inf > box_inf) box_inf = ri_inf
      end do
    end if

    st_inf       = 0.0_dp
    orth_err_out = 0.0_dp
    x_pos        = 1_i32
    do r_idx = 1_i32, R
      nr = n_arr(r_idx)
      pr = p_arr(r_idx)
      np = nr * pr
      p2 = pr * pr

      do i = 0_i32, np - 1_i32
        if (ieee_is_nan(X_packed(x_pos + i)) .or. .not. ieee_is_finite(X_packed(x_pos + i))) then
          status = FC_ERR_INVALID_X0
          return
        end if
        if (ieee_is_nan(GX_packed(x_pos + i)) .or. .not. ieee_is_finite(GX_packed(x_pos + i))) then
          status = FC_ERR_CALLBACK_NAN
          return
        end if
      end do

      orth_r = fc_stiefel_orth_err(nr, pr, X_packed(x_pos:x_pos + np - 1_i32))
      if (orth_r > orth_err_out) orth_err_out = orth_r

      call fc_stiefel_rgrad(nr, pr, &
        X_packed(x_pos:x_pos + np - 1_i32), GX_packed(x_pos:x_pos + np - 1_i32), metric, &
        work(1:np), work(np + 1_i32:np + p2), work(np + p2 + 1_i32:np + 2_i32 * p2))
      rg_nrm = stiefel_norm_from_skew(nr, pr, &
        work(1:np), work(np + 1_i32:np + p2), metric)
      if (rg_nrm > st_inf) st_inf = rg_nrm

      x_pos = x_pos + np
    end do

    kkt_inf_out = max(box_inf, st_inf)
    status      = FC_SUCCESS_QUERY
  end subroutine fc_kkt_residual

  !---------------------------------------------------------------------------
  ! Evaluate search direction directional derivative phi'(0+) and norms.
  !---------------------------------------------------------------------------
  pure subroutine eval_dir_slope_and_norms( &
      d, R, n_arr, p_arr, metric, x_off, &
      z, l, u, gz, dz, X_packed, rgrad_X, eta_X, scratch, &
      slope_box, slope_st, phi_prime0, v_st_sq, v_0, g_0, max_eta_can)
    integer(i32), intent(in) :: d, R, metric
    integer(i32), intent(in) :: n_arr(*), p_arr(*), x_off(*)
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*), dz(*)
    real(dp), intent(in)     :: X_packed(*), rgrad_X(*)
    real(dp), intent(inout)  :: eta_X(*), scratch(*)
    real(dp), intent(out)    :: slope_box, slope_st, phi_prime0
    real(dp), intent(out)    :: v_st_sq, v_0, g_0, max_eta_can

    integer(i32) :: r_idx, nr, pr, p2, x_s, x_e
    real(dp)     :: v_box_sq, g_box_sq, g_st_sq
    real(dp)     :: slope_r, eta_nrm_m, rg_nrm_m, eta_nrm_can

    if (d > 0_i32) then
      call fc_box_dir_deriv0(d, z, l, u, gz, dz, slope_box, v_box_sq, g_box_sq)
    else
      slope_box = 0.0_dp
      v_box_sq  = 0.0_dp
      g_box_sq  = 0.0_dp
    end if

    slope_st    = 0.0_dp
    v_st_sq     = 0.0_dp
    g_st_sq     = 0.0_dp
    max_eta_can = 0.0_dp

    do r_idx = 1_i32, R
      nr  = n_arr(r_idx)
      pr  = p_arr(r_idx)
      p2  = pr * pr
      x_s = x_off(r_idx)
      x_e = x_off(r_idx + 1_i32) - 1_i32

      ! Project eta_r onto T_{X_r} St(n_r, p_r), placing A_eta = X_r^T * eta_r in scratch(1:p2)
      call fc_stiefel_proj_tangent(nr, pr, X_packed(x_s:x_e), eta_X(x_s:x_e), &
        scratch(1:p2), scratch(p2 + 1_i32:2_i32 * p2))

      slope_r   = fc_stiefel_inner(nr, pr, X_packed(x_s:x_e), &
        rgrad_X(x_s:x_e), eta_X(x_s:x_e), metric)
      eta_nrm_m = stiefel_norm_from_skew(nr, pr, eta_X(x_s:x_e), scratch(1:p2), metric)
      eta_nrm_can = stiefel_norm_from_skew(nr, pr, eta_X(x_s:x_e), scratch(1:p2), &
        FC_METRIC_CANONICAL)
      rg_nrm_m  = sqrt(max(0.0_dp, fc_stiefel_inner(nr, pr, X_packed(x_s:x_e), &
        rgrad_X(x_s:x_e), rgrad_X(x_s:x_e), metric)))

      slope_st = slope_st + slope_r
      if (eta_nrm_m <= 1.0e153_dp) then
        v_st_sq = v_st_sq + eta_nrm_m * eta_nrm_m
      else
        v_st_sq = FC_HUGE
      end if
      if (rg_nrm_m <= 1.0e153_dp) then
        g_st_sq = g_st_sq + rg_nrm_m * rg_nrm_m
      else
        g_st_sq = FC_HUGE
      end if
      if (eta_nrm_can > max_eta_can) max_eta_can = eta_nrm_can
    end do

    phi_prime0 = slope_box + slope_st
    v_0        = safe_hypot(sqrt(max(0.0_dp, v_box_sq)), sqrt(max(0.0_dp, v_st_sq)))
    g_0        = safe_hypot(sqrt(max(0.0_dp, g_box_sq)), sqrt(max(0.0_dp, g_st_sq)))
  end subroutine eval_dir_slope_and_norms

  !---------------------------------------------------------------------------
  ! Evaluate transported directional derivative <grad(p_trial), T(s)> at p_trial
  ! for exact 53-bit Hermite Armijo verification in the sub-ULP roundoff regime.
  !---------------------------------------------------------------------------
  pure subroutine eval_trial_slope_fast( &
      d, R, n_arr, p_arr, x_off, cay_off, ipiv_off, &
      z_old, z_t, gz_t, X_old, X_t, GX_t, eta_X, tau, &
      cayley_persist, ipiv_all, polished_step, scratch, slope_t)
    integer(i32), intent(in) :: d, R
    integer(i32), intent(in) :: n_arr(*), p_arr(*), x_off(*), cay_off(*), ipiv_off(*)
    real(dp), intent(in)     :: z_old(*), z_t(*), gz_t(*)
    real(dp), intent(in)     :: X_old(*), X_t(*), GX_t(*), eta_X(*), tau
    real(dp), intent(in)     :: cayley_persist(*)
    integer(i32), intent(in) :: ipiv_all(*)
    logical, intent(in)      :: polished_step
    real(dp), intent(inout)  :: scratch(*)
    real(dp), intent(out)    :: slope_t

    integer(i32) :: i, r_idx, nr, pr, np, p2, x_s, x_e, c_s, p_s
    real(dp)     :: dz_i

    slope_t = 0.0_dp
    do i = 1_i32, d
      dz_i    = z_t(i) - z_old(i)
      slope_t = slope_t + gz_t(i) * dz_i
    end do

    do r_idx = 1_i32, R
      nr  = n_arr(r_idx)
      pr  = p_arr(r_idx)
      np  = nr * pr
      p2  = pr * pr
      x_s = x_off(r_idx)
      x_e = x_off(r_idx + 1_i32) - 1_i32
      c_s = cay_off(r_idx)
      p_s = ipiv_off(r_idx)

      do i = 1_i32, np
        scratch(i) = tau * eta_X(x_s + i - 1_i32)
      end do
      call fc_cayley_transport_vec(nr, pr, X_old(x_s:x_e), eta_X(x_s:x_e), tau, &
        cayley_persist(c_s), ipiv_all(p_s), scratch(1:np), scratch(np + 1_i32))
      if (polished_step) then
        call fc_stiefel_proj_tangent(nr, pr, X_t(x_s:x_e), scratch(1:np), &
          scratch(np + 1_i32:np + p2), scratch(np + p2 + 1_i32:np + 2_i32 * p2))
      end if
      do i = 1_i32, np
        slope_t = slope_t + GX_t(x_s + i - 1_i32) * scratch(i)
      end do
    end do
  end subroutine eval_trial_slope_fast

  !---------------------------------------------------------------------------
  ! Master Solver Core operating on pre-sliced caller-owned workspace arrays.
  !---------------------------------------------------------------------------
  subroutine fc_solve_core( &
      d, R, n_arr, p_arr, z, l, u, X_packed, eval_fg, udata, opts, res, &
      d_pad, n_st_pad, n_tot_pad, n_skew_pad, m_pad, n_cay_pad, n_piv_pad, n_scr, &
      z_trial, gz, gz_trial, dz, rbox, &
      X_trial, GX, GX_trial, rgrad_X, rgrad_old, eta_X, &
      q_buf, s_cur, y_cur, Aq_buf, &
      gamma_blk, abb_ss, abb_sy, abb_yy, &
      s_hist, y_hist, As_hist, Ay_hist, &
      st_sy, st_ss, st_yy, rho_m, alpha_m, &
      cayley_persist, scratch, &
      mask_box, active_pair_flag, ring_order, &
      x_off, skew_off, cay_off, ipiv_off, ipiv_all)
    integer(i32), intent(in)         :: d, R
    integer(i32), intent(in)         :: n_arr(*), p_arr(*)
    real(dp), intent(inout)          :: z(*), X_packed(*)
    real(dp), intent(in)             :: l(*), u(*)
    procedure(fc_eval_fg_cb)         :: eval_fg
    type(c_ptr), value, intent(in)   :: udata
    type(fc_options_t), intent(in)   :: opts
    type(fc_result_t), intent(inout) :: res
    integer(i32), intent(in)         :: d_pad, n_st_pad, n_tot_pad, n_skew_pad
    integer(i32), intent(in)         :: m_pad, n_cay_pad, n_piv_pad, n_scr
    real(dp), intent(inout)          :: z_trial(d_pad), gz(d_pad), gz_trial(d_pad)
    real(dp), intent(inout)          :: dz(d_pad), rbox(d_pad)
    real(dp), intent(inout)          :: X_trial(n_st_pad), GX(n_st_pad), GX_trial(n_st_pad)
    real(dp), intent(inout)          :: rgrad_X(n_st_pad), rgrad_old(n_st_pad), eta_X(n_st_pad)
    real(dp), intent(inout)          :: q_buf(n_tot_pad), s_cur(n_tot_pad), y_cur(n_tot_pad)
    real(dp), intent(inout)          :: Aq_buf(n_skew_pad)
    real(dp), intent(inout)          :: gamma_blk(R + 1_i32), abb_ss(R + 1_i32)
    real(dp), intent(inout)          :: abb_sy(R + 1_i32), abb_yy(R + 1_i32)
    real(dp), intent(inout)          :: s_hist(n_tot_pad, opts%memory)
    real(dp), intent(inout)          :: y_hist(n_tot_pad, opts%memory)
    real(dp), intent(inout)          :: As_hist(n_skew_pad, opts%memory)
    real(dp), intent(inout)          :: Ay_hist(n_skew_pad, opts%memory)
    real(dp), intent(inout)          :: st_sy(m_pad), st_ss(m_pad), st_yy(m_pad)
    real(dp), intent(inout)          :: rho_m(m_pad), alpha_m(m_pad)
    real(dp), intent(inout)          :: cayley_persist(n_cay_pad), scratch(n_scr)
    integer(i32), intent(inout)      :: mask_box(d_pad)
    integer(i32), intent(inout)      :: active_pair_flag(m_pad), ring_order(m_pad)
    integer(i32), intent(inout)      :: x_off(R + 1_i32), skew_off(R + 1_i32)
    integer(i32), intent(inout)      :: cay_off(R + 1_i32), ipiv_off(R + 1_i32)
    integer(i32), intent(inout)      :: ipiv_all(n_piv_pad)

    integer(i32) :: n_st_tot, n_skew_tot, n_cayley_tot, n_ipiv_tot, st
    integer(i32) :: i, idx, r_idx, nr, pr, np, p2, x_s, x_e, sk_s, sk_e, c_s, p_s
    integer(i32) :: info_qr, info_lu, rc, n_eval, n_active_box, p_max
    integer(i32) :: k_iter, ls_step, attempt, m_cur, ring_head, tiny_df_streak, stag_streak
    real(dp)     :: orth_r, orth_max, f_k, f_trial, box_res, box_l2, st_res, rg_nrm
    real(dp)     :: kkt_res, kkt_old, best_kkt, zh_eta_eff, Q_zh, C_zh, Q_next
    real(dp)     :: slope_box, slope_st, phi_prime0, v_st_sq, v_0, g_0, max_eta_can
    real(dp)     :: tau_init, tau, chord_gz, box_step_sq, delta_tau, tot_step_sq
    real(dp)     :: drift_r, max_drift_step, f_scale, f_diff
    real(dp)     :: z_diff_inf, z_norm_inf, X_diff_max, x_diff_r, sqrt_p_max
    real(dp)     :: step_diff, x_scale, ftol_eff, xtol_eff, slope_trial, c_scale
    logical      :: has_prev, used_qn, descent_ok, ls_accepted, lu_failed, qr_failed
    logical      :: polished_step, is_default_ftol, is_default_xtol

    ! 1. Compute packed Stiefel offsets
    call fc_stiefel_offsets(R, n_arr, p_arr, &
      x_off, skew_off, cay_off, ipiv_off, &
      n_st_tot, n_skew_tot, n_cayley_tot, n_ipiv_tot, st)
    if (st /= 0_i32) then
      res%status = st
      return
    end if

    ! Zero workspace sentinel slots for clean determinism
    z_trial(1) = 0.0_dp
    gz(1)      = 0.0_dp
    X_trial(1) = 0.0_dp
    GX(1)      = 0.0_dp

    ! 2. Validate bounds & project/verify initial z0
    if (d > 0_i32) then
      call fc_box_validate(d, l, u, st)
      if (st /= 0_i32) then
        res%status = FC_ERR_INVALID_BOUNDS
        return
      end if

      do i = 1_i32, d
        if (ieee_is_nan(z(i)) .or. .not. ieee_is_finite(z(i))) then
          res%status = FC_ERR_INVALID_X0
          return
        end if
      end do

      if (opts%project_initial == 1_i32) then
        call fc_box_project(d, l, u, z)
      else
        do i = 1_i32, d
          if (z(i) < l(i) .or. z(i) > u(i)) then
            res%status = FC_ERR_INVALID_X0
            return
          end if
        end do
      end if
    end if

    ! 3. Validate & polish/verify initial Stiefel factors X0
    orth_max = 0.0_dp
    p_max    = 0_i32
    do r_idx = 1_i32, R
      nr  = n_arr(r_idx)
      pr  = p_arr(r_idx)
      if (pr > p_max) p_max = pr
      x_s = x_off(r_idx)
      x_e = x_off(r_idx + 1_i32) - 1_i32

      do idx = x_s, x_e
        if (ieee_is_nan(X_packed(idx)) .or. .not. ieee_is_finite(X_packed(idx))) then
          res%status = FC_ERR_INVALID_X0
          return
        end if
      end do

      orth_r = fc_stiefel_orth_err(nr, pr, X_packed(x_s:x_e))
      if (ieee_is_nan(orth_r) .or. .not. ieee_is_finite(orth_r)) then
        res%status = FC_ERR_INVALID_X0
        return
      end if

      if (opts%project_initial == 1_i32) then
        if (orth_r > opts%orth_tol) then
          call fc_geqrf_pos(nr, pr, X_packed(x_s:x_e), &
            scratch(1:pr), scratch(pr + 1_i32:2_i32 * pr), info_qr)
          if (info_qr /= 0_i32) then
            res%status = FC_ERR_INVALID_X0
            return
          end if
          res%n_reorth = res%n_reorth + 1_i32
          orth_r       = fc_stiefel_orth_err(nr, pr, X_packed(x_s:x_e))
        end if
      else
        if (orth_r > 1.0e-8_dp) then
          res%status = FC_ERR_INVALID_X0
          return
        end if
      end if
      if (orth_r > orth_max) orth_max = orth_r
    end do
    sqrt_p_max = sqrt(real(p_max, dp))

    ! 4. Evaluate initial objective and Euclidean gradients at p_0 (n_eval = 1)
    do i = 1_i32, d
      z_trial(i) = z(i)
      gz(i)      = 0.0_dp
    end do
    do idx = 1_i32, n_st_tot
      X_trial(idx) = X_packed(idx)
      GX(idx)      = 0.0_dp
    end do

    f_k    = 0.0_dp
    rc     = eval_fg(d, R, n_arr, p_arr, z_trial, X_trial, f_k, gz, GX, udata)
    n_eval = 1_i32
    res%n_eval     = n_eval
    res%orth_error = orth_max

    if (rc < 0_i32) then
      res%status = FC_ERR_CALLBACK_ABORT
      return
    end if
    if (rc > 0_i32 .or. .not. is_trial_finite(d, n_st_tot, f_k, gz, GX)) then
      res%status = FC_ERR_CALLBACK_NAN
      return
    end if

    ! Compute initial KKT residuals and active-band partition
    if (d > 0_i32) then
      call fc_box_kkt_residual(d, z, l, u, gz, rbox, box_res, box_l2)
      call fc_box_partition(d, z, l, u, gz, box_res, opts%eps_band, mask_box, n_active_box)
    else
      box_res      = 0.0_dp
      n_active_box = 0_i32
    end if

    st_res = 0.0_dp
    do r_idx = 1_i32, R
      nr   = n_arr(r_idx)
      pr   = p_arr(r_idx)
      p2   = pr * pr
      x_s  = x_off(r_idx)
      x_e  = x_off(r_idx + 1_i32) - 1_i32
      sk_s = skew_off(r_idx)
      sk_e = skew_off(r_idx + 1_i32) - 1_i32

      call fc_stiefel_rgrad(nr, pr, X_packed(x_s:x_e), GX(x_s:x_e), opts%metric, &
        rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), scratch(1:p2))
      rg_nrm = stiefel_norm_from_skew(nr, pr, rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), opts%metric)
      if (rg_nrm > st_res) st_res = rg_nrm
    end do

    kkt_res = max(box_res, st_res)

    res%f_val            = f_k
    res%kkt_residual     = kkt_res
    res%box_residual     = box_res
    res%stiefel_residual = st_res
    res%orth_error       = orth_max
    res%n_iter           = 0_i32
    res%n_eval           = n_eval
    res%n_active_box     = n_active_box

    if (kkt_res <= opts%gtol) then
      res%status = FC_CONVERGED_GRAD
      return
    end if
    if (opts%max_iter == 0_i32) then
      res%status = FC_MAX_ITER_REACHED
      return
    end if
    if (n_eval >= opts%max_eval) then
      res%status = FC_MAX_EVAL_REACHED
      return
    end if

    ! Initialize Zhang-Hager & L-BFGS ring state
    if (opts%zh_eta < 0.0_dp) then
      zh_eta_eff = 0.85_dp
    else
      zh_eta_eff = opts%zh_eta
    end if

    Q_zh           = 1.0_dp
    C_zh           = f_k
    m_cur          = 0_i32
    ring_head      = 0_i32
    has_prev       = .false.
    tiny_df_streak = 0_i32
    stag_streak    = 0_i32
    best_kkt       = kkt_res

    is_default_ftol = (opts%gtol > 0.0_dp .and. opts%ftol <= 1.0e-13_dp)
    if (is_default_ftol) then
      ftol_eff = 2.0_dp * FC_EPS
    else
      ftol_eff = opts%ftol
    end if

    is_default_xtol = (opts%gtol > 0.0_dp .and. opts%xtol <= 1.0e-13_dp)
    if (is_default_xtol) then
      xtol_eff = 2.0_dp * FC_EPS
    else
      xtol_eff = opts%xtol
    end if

    do r_idx = 1_i32, R + 1_i32
      gamma_blk(r_idx) = 1.0_dp
      abb_ss(r_idx)    = 0.0_dp
      abb_sy(r_idx)    = 0.0_dp
      abb_yy(r_idx)    = 0.0_dp
    end do
    do i = 1_i32, n_tot_pad
      s_cur(i) = 0.0_dp
      y_cur(i) = 0.0_dp
    end do

    ! 5. Main Coupled Quasi-Newton Loop
    do k_iter = 0_i32, opts%max_iter - 1_i32
      ! 5a. Save fresh Aq_buf copy if needed for ABB fallback: note fc_stiefel_rgrad
      ! can rapidly refresh Aq_buf on fallback, so zero permanent overhead on QN path!

      ! 5b. Compute candidate search direction (dz, eta_X)
      if (m_cur >= 1_i32 .and. opts%memory >= 1_i32) then
        call fc_lbfgs_direction( &
          d, R, n_st_tot, n_skew_tot, opts%metric, &
          opts%memory, m_cur, n_tot_pad, n_skew_pad, &
          x_off, skew_off, &
          z, l, u, gz, mask_box, &
          rgrad_X, Aq_buf, &
          gamma_blk, has_prev, s_cur, y_cur, &
          s_hist, y_hist, As_hist, Ay_hist, &
          st_sy, st_ss, st_yy, rho_m, alpha_m, &
          active_pair_flag, ring_order, &
          q_buf, dz, eta_X)
        used_qn = .true.
      else
        call fc_abb_direction( &
          k_iter, has_prev, d, R, n_arr, p_arr, opts%metric, &
          x_off, skew_off, &
          z, l, u, gz, mask_box, &
          rgrad_X, Aq_buf, &
          abb_ss, abb_sy, abb_yy, &
          dz, eta_X)
        used_qn = .false.
      end if

      ! 5c. Check strict directional descent phi'(0+) <= -1e-12 * v_0 * g_0
      call eval_dir_slope_and_norms( &
        d, R, n_arr, p_arr, opts%metric, x_off, &
        z, l, u, gz, dz, X_packed, rgrad_X, eta_X, scratch, &
        slope_box, slope_st, phi_prime0, v_st_sq, v_0, g_0, max_eta_can)

      descent_ok = .false.
      if (v_0 > 0.0_dp .and. ieee_is_finite(phi_prime0) .and. phi_prime0 < 0.0_dp) then
        if (v_0 <= 1.0e12_dp) then
          descent_ok = (phi_prime0 <= (-1.0e-12_dp * v_0) * g_0)
        else if (g_0 <= 1.0e12_dp) then
          descent_ok = (phi_prime0 <= v_0 * (-1.0e-12_dp * g_0))
        else
          descent_ok = .true.
        end if
      end if

      if (.not. descent_ok .and. used_qn) then
        m_cur           = 0_i32
        ring_head       = 0_i32
        res%n_fallbacks = res%n_fallbacks + 1_i32
        used_qn         = .false.

        do r_idx = 1_i32, R
          nr   = n_arr(r_idx)
          pr   = p_arr(r_idx)
          p2   = pr * pr
          x_s  = x_off(r_idx)
          x_e  = x_off(r_idx + 1_i32) - 1_i32
          sk_s = skew_off(r_idx)
          sk_e = skew_off(r_idx + 1_i32) - 1_i32
          call fc_stiefel_rgrad(nr, pr, X_packed(x_s:x_e), GX(x_s:x_e), opts%metric, &
            rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), scratch(1:p2))
        end do

        call fc_abb_direction( &
          k_iter, has_prev, d, R, n_arr, p_arr, opts%metric, &
          x_off, skew_off, &
          z, l, u, gz, mask_box, &
          rgrad_X, Aq_buf, &
          abb_ss, abb_sy, abb_yy, &
          dz, eta_X)

        call eval_dir_slope_and_norms( &
          d, R, n_arr, p_arr, opts%metric, x_off, &
          z, l, u, gz, dz, X_packed, rgrad_X, eta_X, scratch, &
          slope_box, slope_st, phi_prime0, v_st_sq, v_0, g_0, max_eta_can)
      end if

      ! 5d. Two-chance Zhang-Hager / Generalized Arc-Armijo backtracking line search
      ls_accepted    = .false.
      polished_step  = .false.
      max_drift_step = 0.0_dp
      tau            = 0.0_dp

      attempt_loop: do attempt = 1_i32, 2_i32
        do r_idx = 1_i32, R
          nr  = n_arr(r_idx)
          pr  = p_arr(r_idx)
          x_s = x_off(r_idx)
          x_e = x_off(r_idx + 1_i32) - 1_i32
          c_s = cay_off(r_idx)
          call fc_cayley_setup(nr, pr, X_packed(x_s:x_e), eta_X(x_s:x_e), &
            cayley_persist(c_s), scratch)
        end do

        tau_init = 1.0_dp
        if (R > 0_i32 .and. max_eta_can > opts%max_step_vel) then
          tau_init = min(tau_init, opts%max_step_vel / max_eta_can)
        end if
        tau = tau_init

        ls_loop: do ls_step = 1_i32, opts%max_ls_steps
          if (d > 0_i32) then
            call fc_box_trial_step(d, z, l, u, gz, dz, tau, z_trial, chord_gz, box_step_sq)
          else
            chord_gz    = 0.0_dp
            box_step_sq = 0.0_dp
          end if

          delta_tau = chord_gz + tau * slope_st
          if (tau <= 1.0e150_dp .and. v_st_sq <= 0.45_dp * FC_HUGE) then
            tot_step_sq = box_step_sq + (tau * tau) * v_st_sq
          else
            tot_step_sq = FC_HUGE
          end if

          if (.not. (tot_step_sq > 0.0_dp) .or. .not. (delta_tau < 0.0_dp) .or. &
              delta_tau > -1.0e-12_dp * tot_step_sq) then
            tau = 0.5_dp * tau
            cycle ls_loop
          end if

          lu_failed = .false.
          do r_idx = 1_i32, R
            nr  = n_arr(r_idx)
            pr  = p_arr(r_idx)
            x_s = x_off(r_idx)
            x_e = x_off(r_idx + 1_i32) - 1_i32
            c_s = cay_off(r_idx)
            p_s = ipiv_off(r_idx)
            call fc_cayley_retract(nr, pr, X_packed(x_s:x_e), eta_X(x_s:x_e), tau, &
              cayley_persist(c_s), ipiv_all(p_s), X_trial(x_s:x_e), scratch, info_lu)
            if (info_lu /= 0_i32) then
              lu_failed = .true.
              exit
            end if
          end do
          if (lu_failed) then
            tau = 0.5_dp * tau
            cycle ls_loop
          end if

          ! Strict hard evaluation budget check before polishing / invoking callback
          if (n_eval >= opts%max_eval) then
            res%n_eval = n_eval
            res%status = FC_MAX_EVAL_REACHED
            return
          end if

          ! Pre-callback positive-diagonal Householder-QR drift polish
          polished_step  = .false.
          max_drift_step = 0.0_dp
          qr_failed      = .false.
          do r_idx = 1_i32, R
            nr      = n_arr(r_idx)
            pr      = p_arr(r_idx)
            x_s     = x_off(r_idx)
            x_e     = x_off(r_idx + 1_i32) - 1_i32
            drift_r = fc_stiefel_orth_err(nr, pr, X_trial(x_s:x_e))
            if (drift_r > max_drift_step) max_drift_step = drift_r
            if (drift_r > opts%orth_tol) then
              call fc_geqrf_pos(nr, pr, X_trial(x_s:x_e), &
                scratch(1:pr), scratch(pr + 1_i32:2_i32 * pr), info_qr)
              if (info_qr /= 0_i32) then
                qr_failed = .true.
                exit
              end if
              polished_step = .true.
              res%n_reorth  = res%n_reorth + 1_i32
            end if
          end do
          if (qr_failed) then
            tau = 0.5_dp * tau
            cycle ls_loop
          end if

          do i = 1_i32, d
            gz_trial(i) = 0.0_dp
          end do
          do idx = 1_i32, n_st_tot
            GX_trial(idx) = 0.0_dp
          end do
          f_trial    = 0.0_dp
          rc         = eval_fg(d, R, n_arr, p_arr, z_trial, X_trial, &
                         f_trial, gz_trial, GX_trial, udata)
          n_eval     = n_eval + 1_i32
          res%n_eval = n_eval

          if (rc < 0_i32) then
            res%status = FC_ERR_CALLBACK_ABORT
            return
          end if
          if (rc > 0_i32 .or. .not. is_trial_finite(d, n_st_tot, f_trial, gz_trial, GX_trial)) then
            tau = 0.5_dp * tau
            cycle ls_loop
          end if

          if (f_trial <= C_zh + opts%c1_armijo * delta_tau) then
            ls_accepted = .true.
            exit attempt_loop
          else
            c_scale = max(1.0_dp, abs(C_zh))
            if (abs(delta_tau) <= 256.0_dp * FC_EPS * c_scale .and. &
                f_trial <= C_zh + 16.0_dp * FC_EPS * c_scale) then
              call eval_trial_slope_fast( &
                d, R, n_arr, p_arr, x_off, cay_off, ipiv_off, &
                z, z_trial, gz_trial, X_packed, X_trial, GX_trial, eta_X, tau, &
                cayley_persist, ipiv_all, polished_step, scratch, slope_trial)
              if (0.5_dp * (delta_tau + slope_trial) <= opts%c1_armijo * delta_tau) then
                ls_accepted = .true.
                exit attempt_loop
              end if
            end if
          end if

          tau = 0.5_dp * tau
        end do ls_loop

        ! If QN backtracking failed, flush history and retry once with ABB steepest descent
        if (used_qn) then
          m_cur           = 0_i32
          ring_head       = 0_i32
          res%n_fallbacks = res%n_fallbacks + 1_i32
          used_qn         = .false.

          do r_idx = 1_i32, R
            nr   = n_arr(r_idx)
            pr   = p_arr(r_idx)
            p2   = pr * pr
            x_s  = x_off(r_idx)
            x_e  = x_off(r_idx + 1_i32) - 1_i32
            sk_s = skew_off(r_idx)
            sk_e = skew_off(r_idx + 1_i32) - 1_i32
            call fc_stiefel_rgrad(nr, pr, X_packed(x_s:x_e), GX(x_s:x_e), opts%metric, &
              rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), scratch(1:p2))
          end do

          call fc_abb_direction( &
            k_iter, has_prev, d, R, n_arr, p_arr, opts%metric, &
            x_off, skew_off, &
            z, l, u, gz, mask_box, &
            rgrad_X, Aq_buf, &
            abb_ss, abb_sy, abb_yy, &
            dz, eta_X)

          call eval_dir_slope_and_norms( &
            d, R, n_arr, p_arr, opts%metric, x_off, &
            z, l, u, gz, dz, X_packed, rgrad_X, eta_X, scratch, &
            slope_box, slope_st, phi_prime0, v_st_sq, v_0, g_0, max_eta_can)
        else
          exit attempt_loop
        end if
      end do attempt_loop

      if (.not. ls_accepted) then
        res%status = FC_ERR_LS_FAILED
        return
      end if

      ! 5e. Compute step & objective deltas and update Zhang-Hager reference C_zh
      f_scale = max(1.0_dp, abs(f_trial), abs(f_k))
      f_diff  = safe_abs_diff(f_trial, f_k)
      if (opts%ftol > 0.0_dp .and. f_diff <= ftol_eff * f_scale) then
        tiny_df_streak = tiny_df_streak + 1_i32
      else
        tiny_df_streak = 0_i32
      end if

      z_diff_inf = 0.0_dp
      z_norm_inf = 0.0_dp
      do i = 1_i32, d
        if (safe_abs_diff(z_trial(i), z(i)) > z_diff_inf) then
          z_diff_inf = safe_abs_diff(z_trial(i), z(i))
        end if
        if (abs(z_trial(i)) > z_norm_inf) z_norm_inf = abs(z_trial(i))
      end do

      X_diff_max = 0.0_dp
      do r_idx = 1_i32, R
        nr  = n_arr(r_idx)
        pr  = p_arr(r_idx)
        np  = nr * pr
        x_s = x_off(r_idx)
        do i = 0_i32, np - 1_i32
          scratch(i + 1_i32) = X_trial(x_s + i) - X_packed(x_s + i)
        end do
        x_diff_r = fc_frob_norm(nr, pr, scratch(1:np))
        if (x_diff_r > X_diff_max) X_diff_max = x_diff_r
      end do

      step_diff = max(z_diff_inf, X_diff_max)
      x_scale   = max(1.0_dp, z_norm_inf, sqrt_p_max)

      Q_next = zh_eta_eff * Q_zh + 1.0_dp
      C_zh   = (zh_eta_eff * Q_zh * C_zh + f_trial) / Q_next
      Q_zh   = Q_next

      ! 5f-5h. Evaluate new KKT/rgrad, transport history, update BB2, and push (s_k, y_k)
      kkt_old = kkt_res
      do idx = 1_i32, n_st_tot
        rgrad_old(idx) = rgrad_X(idx)
      end do

      if (d > 0_i32) then
        call fc_box_kkt_residual(d, z_trial, l, u, gz_trial, rbox, box_res, box_l2)
        call fc_box_partition(d, z_trial, l, u, gz_trial, box_res, opts%eps_band, &
          mask_box, n_active_box)
      else
        box_res      = 0.0_dp
        n_active_box = 0_i32
      end if

      st_res   = 0.0_dp
      orth_max = 0.0_dp
      do r_idx = 1_i32, R
        nr   = n_arr(r_idx)
        pr   = p_arr(r_idx)
        p2   = pr * pr
        x_s  = x_off(r_idx)
        x_e  = x_off(r_idx + 1_i32) - 1_i32
        sk_s = skew_off(r_idx)
        sk_e = skew_off(r_idx + 1_i32) - 1_i32

        orth_r = fc_stiefel_orth_err(nr, pr, X_trial(x_s:x_e))
        if (orth_r > orth_max) orth_max = orth_r

        call fc_stiefel_rgrad(nr, pr, X_trial(x_s:x_e), GX_trial(x_s:x_e), opts%metric, &
          rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), scratch(1:p2))
        rg_nrm = stiefel_norm_from_skew(nr, pr, rgrad_X(x_s:x_e), Aq_buf(sk_s:sk_e), opts%metric)
        if (rg_nrm > st_res) st_res = rg_nrm
      end do

      call fc_lbfgs_transport_and_push( &
        d, R, n_arr, p_arr, n_st_tot, opts%metric, opts%damping_mode, &
        opts%memory, m_cur, ring_head, n_tot_pad, n_skew_pad, &
        x_off, skew_off, cay_off, ipiv_off, &
        l, u, z, z_trial, gz, gz_trial, mask_box, &
        X_packed, X_trial, eta_X, tau, cayley_persist, ipiv_all, &
        rgrad_old, rgrad_X, &
        kkt_old, polished_step, max_drift_step, &
        gamma_blk, abb_ss, abb_sy, abb_yy, &
        s_cur, y_cur, &
        s_hist, y_hist, As_hist, Ay_hist, &
        st_sy, st_ss, st_yy, ring_order, scratch)

      has_prev = .true.

      ! Commit new iterate
      do i = 1_i32, d
        z(i)  = z_trial(i)
        gz(i) = gz_trial(i)
      end do
      do idx = 1_i32, n_st_tot
        X_packed(idx) = X_trial(idx)
        GX(idx)       = GX_trial(idx)
      end do

      f_k     = f_trial
      kkt_res = max(box_res, st_res)

      res%f_val            = f_k
      res%kkt_residual     = kkt_res
      res%box_residual     = box_res
      res%stiefel_residual = st_res
      res%orth_error       = orth_max
      res%last_step_size   = tau
      res%n_iter           = k_iter + 1_i32
      res%n_eval           = n_eval
      res%n_active_box     = n_active_box

      ! 5i. Check deterministic stopping criteria in strict priority order
      if (kkt_res <= opts%gtol) then
        res%status = FC_CONVERGED_GRAD
        return
      end if

      if (kkt_res < 0.99_dp * best_kkt) then
        best_kkt    = kkt_res
        stag_streak = 0_i32
      else
        stag_streak = stag_streak + 1_i32
      end if

      if (opts%ftol > 0.0_dp .and. f_diff <= ftol_eff * f_scale) then
        if (is_default_ftol) then
          if (tiny_df_streak >= 16_i32 .and. stag_streak >= 16_i32 .and. &
              abs(delta_tau) <= 64.0_dp * FC_EPS * f_scale) then
            res%status = FC_CONVERGED_FTOL
            return
          end if
        else if (opts%gtol <= 0.0_dp .or. tiny_df_streak >= 2_i32) then
          res%status = FC_CONVERGED_FTOL
          return
        end if
      end if

      if (opts%xtol > 0.0_dp .and. step_diff <= xtol_eff * x_scale) then
        if (is_default_xtol) then
          if (stag_streak >= 16_i32) then
            res%status = FC_CONVERGED_XTOL
            return
          end if
        else
          res%status = FC_CONVERGED_XTOL
          return
        end if
      end if

      if (k_iter + 1_i32 < opts%max_iter .and. n_eval >= opts%max_eval) then
        res%status = FC_MAX_EVAL_REACHED
        return
      end if
    end do

    res%status = FC_MAX_ITER_REACHED
  end subroutine fc_solve_core

  !---------------------------------------------------------------------------
  ! Public Native Fortran 2018 Master Solver Entry Point (0 heap allocations).
  !---------------------------------------------------------------------------
  subroutine fc_solve( &
      d, R, n_arr, p_arr, z, l, u, X_packed, eval_fg, udata, &
      opts, work, lwork, iwork, liwork, res)
    integer(i32), intent(in)       :: d, R
    integer(i32), intent(in)       :: n_arr(*), p_arr(*)
    real(dp), intent(inout)        :: z(*), X_packed(*)
    real(dp), intent(in)           :: l(*), u(*)
    procedure(fc_eval_fg_cb)       :: eval_fg
    type(c_ptr), value, intent(in) :: udata
    type(fc_options_t), intent(in) :: opts
    real(dp), intent(inout)        :: work(*)
    integer(i64), intent(in)       :: lwork
    integer(i32), intent(inout)    :: iwork(*)
    integer(i64), intent(in)       :: liwork
    type(fc_result_t), intent(out) :: res

    integer(i64) :: d_pad, n_st_pad, n_tot_pad, n_skew_pad
    integer(i64) :: m_pad, n_cay_pad, n_piv_pad, n_scr
    integer(i64) :: lwork_min, liwork_min, m64, rp1_64
    integer(i64) :: w_z_trial, w_gz, w_gz_trial, w_dz, w_rbox
    integer(i64) :: w_X_trial, w_GX, w_GX_trial, w_rgrad_X, w_rgrad_old, w_eta_X
    integer(i64) :: w_q_buf, w_s_cur, w_y_cur, w_Aq_buf
    integer(i64) :: w_gamma_blk, w_abb_ss, w_abb_sy, w_abb_yy
    integer(i64) :: w_s_hist, w_y_hist, w_As_hist, w_Ay_hist
    integer(i64) :: w_st_sy, w_st_ss, w_st_yy, w_rho_m, w_alpha_m
    integer(i64) :: w_cayley, w_scratch
    integer(i64) :: iw_mask_box, iw_active_pair, iw_ring_order
    integer(i64) :: iw_x_off, iw_skew_off, iw_cay_off, iw_ipiv_off, iw_ipiv_all
    integer(i32) :: st

    call fc_init_result(res)

    call compute_workspace_dims(d, R, n_arr, p_arr, opts%memory, &
      d_pad, n_st_pad, n_tot_pad, n_skew_pad, m_pad, n_cay_pad, n_piv_pad, n_scr, &
      lwork_min, liwork_min, st)
    if (st /= 0_i32) then
      res%status = st
      return
    end if

    call fc_validate_options(opts, st)
    if (st /= 0_i32) then
      res%status = st
      return
    end if

    res%lwork_min  = lwork_min
    res%liwork_min = liwork_min

    if (lwork == -1_i64 .or. liwork == -1_i64) then
      work(1)    = real(lwork_min, dp)
      iwork(1)   = int(min(liwork_min, int(huge(0_i32), i64)), i32)
      res%status = FC_SUCCESS_QUERY
      return
    end if

    if (lwork < lwork_min .or. liwork < liwork_min) then
      res%status = FC_ERR_WORK_TOO_SMALL
      return
    end if

    ! Exact 10-segment work(1:lwork) and 4-segment iwork(1:liwork) layout (ARCHITECTURE_SPEC 3.4)
    m64    = int(opts%memory, i64)
    rp1_64 = int(R, i64) + 1_i64

    w_z_trial   = 1_i64
    w_gz        = w_z_trial   + d_pad
    w_gz_trial  = w_gz        + d_pad
    w_dz        = w_gz_trial  + d_pad
    w_rbox      = w_dz        + d_pad

    w_X_trial   = w_rbox      + d_pad
    w_GX        = w_X_trial   + n_st_pad
    w_GX_trial  = w_GX        + n_st_pad
    w_rgrad_X   = w_GX_trial  + n_st_pad
    w_rgrad_old = w_rgrad_X   + n_st_pad
    w_eta_X     = w_rgrad_old + n_st_pad

    w_q_buf     = w_eta_X     + n_st_pad
    w_s_cur     = w_q_buf     + n_tot_pad
    w_y_cur     = w_s_cur     + n_tot_pad

    w_Aq_buf    = w_y_cur     + n_tot_pad

    w_gamma_blk = w_Aq_buf    + n_skew_pad
    w_abb_ss    = w_gamma_blk + rp1_64
    w_abb_sy    = w_abb_ss    + rp1_64
    w_abb_yy    = w_abb_sy    + rp1_64

    w_s_hist    = w_abb_yy    + rp1_64
    w_y_hist    = w_s_hist    + m64 * n_tot_pad

    w_As_hist   = w_y_hist    + m64 * n_tot_pad
    w_Ay_hist   = w_As_hist   + m64 * n_skew_pad

    w_st_sy     = w_Ay_hist   + m64 * n_skew_pad
    w_st_ss     = w_st_sy     + m_pad
    w_st_yy     = w_st_ss     + m_pad
    w_rho_m     = w_st_yy     + m_pad
    w_alpha_m   = w_rho_m     + m_pad

    w_cayley    = w_alpha_m   + m_pad
    w_scratch   = w_cayley    + n_cay_pad

    iw_mask_box    = 1_i64
    iw_active_pair = iw_mask_box    + d_pad
    iw_ring_order  = iw_active_pair + m_pad
    iw_x_off       = iw_ring_order  + m_pad
    iw_skew_off    = iw_x_off       + rp1_64
    iw_cay_off     = iw_skew_off    + rp1_64
    iw_ipiv_off    = iw_cay_off     + rp1_64
    iw_ipiv_all    = iw_ipiv_off    + rp1_64

    call fc_solve_core( &
      d, R, n_arr, p_arr, z, l, u, X_packed, eval_fg, udata, opts, res, &
      int(d_pad, i32), int(n_st_pad, i32), int(n_tot_pad, i32), int(n_skew_pad, i32), &
      int(m_pad, i32), int(n_cay_pad, i32), int(n_piv_pad, i32), int(n_scr, i32), &
      work(w_z_trial:w_gz - 1_i64), &
      work(w_gz:w_gz_trial - 1_i64), &
      work(w_gz_trial:w_dz - 1_i64), &
      work(w_dz:w_rbox - 1_i64), &
      work(w_rbox:w_X_trial - 1_i64), &
      work(w_X_trial:w_GX - 1_i64), &
      work(w_GX:w_GX_trial - 1_i64), &
      work(w_GX_trial:w_rgrad_X - 1_i64), &
      work(w_rgrad_X:w_rgrad_old - 1_i64), &
      work(w_rgrad_old:w_eta_X - 1_i64), &
      work(w_eta_X:w_q_buf - 1_i64), &
      work(w_q_buf:w_s_cur - 1_i64), &
      work(w_s_cur:w_y_cur - 1_i64), &
      work(w_y_cur:w_Aq_buf - 1_i64), &
      work(w_Aq_buf:w_gamma_blk - 1_i64), &
      work(w_gamma_blk:w_abb_ss - 1_i64), &
      work(w_abb_ss:w_abb_sy - 1_i64), &
      work(w_abb_sy:w_abb_yy - 1_i64), &
      work(w_abb_yy:w_s_hist - 1_i64), &
      work(w_s_hist:w_y_hist - 1_i64), &
      work(w_y_hist:w_As_hist - 1_i64), &
      work(w_As_hist:w_Ay_hist - 1_i64), &
      work(w_Ay_hist:w_st_sy - 1_i64), &
      work(w_st_sy:w_st_ss - 1_i64), &
      work(w_st_ss:w_st_yy - 1_i64), &
      work(w_st_yy:w_rho_m - 1_i64), &
      work(w_rho_m:w_alpha_m - 1_i64), &
      work(w_alpha_m:w_cayley - 1_i64), &
      work(w_cayley:w_scratch - 1_i64), &
      work(w_scratch:w_scratch + n_scr - 1_i64), &
      iwork(iw_mask_box:iw_active_pair - 1_i64), &
      iwork(iw_active_pair:iw_ring_order - 1_i64), &
      iwork(iw_ring_order:iw_x_off - 1_i64), &
      iwork(iw_x_off:iw_skew_off - 1_i64), &
      iwork(iw_skew_off:iw_cay_off - 1_i64), &
      iwork(iw_cay_off:iw_ipiv_off - 1_i64), &
      iwork(iw_ipiv_off:iw_ipiv_all - 1_i64), &
      iwork(iw_ipiv_all:iw_ipiv_all + n_piv_pad - 1_i64))
  end subroutine fc_solve

end module fortchuck_solver
