module fortchuck_box
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: &
    dp, i32, FC_EPS, FC_HUGE, &
    FC_ERR_INVALID_DIM, FC_ERR_INVALID_BOUNDS, FC_ERR_INFEASIBLE_LIN, &
    FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED
  implicit none (type, external)
  private

  real(dp), parameter :: GAMMA_MIN   = 1.0e-10_dp
  real(dp), parameter :: GAMMA_MAX   = 1.0e+10_dp
  real(dp), parameter :: SAFE_SQ_LIM = 1.0e+153_dp

  public :: fc_box_validate
  public :: fc_box_project
  public :: fc_box_kkt_residual
  public :: fc_box_partition
  public :: fc_box_active_step
  public :: fc_box_active_step_diag
  public :: fc_box_breakpoint_cap
  public :: fc_box_dir_deriv0
  public :: fc_box_trial_step
  public :: fc_box_lin_validate
  public :: fc_box_lin_project
  public :: fc_box_lin_kkt_residual
  public :: fc_box_lin_partition
  public :: fc_box_lin_trial_step

contains

  !---------------------------------------------------------------------------
  ! Overflow-safe clamped step min(max(zi + tau * dzi, li), ui).
  !---------------------------------------------------------------------------
  pure function safe_clamp_step(zi, li, ui, dzi, tau) result(zt)
    real(dp), intent(in) :: zi, li, ui, dzi, tau
    real(dp)             :: zt

    real(dp) :: zc, ceil_u, floor_l, half_gap, half_dz, half_step

    if (.not. (li < ui)) then
      zt = li
      return
    end if

    zc = min(max(zi, li), ui)
    if (.not. (abs(tau) > 0.0_dp) .or. .not. (abs(dzi) > 0.0_dp)) then
      zt = zc
      return
    end if

    ! Fast path when operands cannot approach real64 overflow limits
    if (abs(dzi) <= 1.0e150_dp .and. abs(tau) <= 1.0e150_dp .and. abs(zc) <= 0.45_dp * FC_HUGE) then
      zt = min(max(zc + tau * dzi, li), ui)
      return
    end if

    ! Overflow-safe path near +-FC_HUGE or extreme steps (assuming tau >= 0; handle sign via tau*dzi)
    if ((tau > 0.0_dp .and. dzi > 0.0_dp) .or. (tau < 0.0_dp .and. dzi < 0.0_dp)) then
      if (ieee_is_finite(ui)) then
        ceil_u = ui
      else
        ceil_u = FC_HUGE
      end if
      half_gap = 0.5_dp * ceil_u - 0.5_dp * zc
      if (half_gap <= 0.0_dp) then
        zt = ceil_u
        return
      end if
      half_dz = 0.5_dp * abs(dzi)
      if (half_dz >= 1.0_dp) then
        if (abs(tau) >= half_gap / half_dz) then
          zt = ceil_u
        else
          half_step = half_dz * abs(tau)
          zt = min((zc + half_step) + half_step, ceil_u)
        end if
      else
        half_step = half_dz * abs(tau)
        if (half_step >= half_gap) then
          zt = ceil_u
        else
          zt = min((zc + half_step) + half_step, ceil_u)
        end if
      end if
    else
      if (ieee_is_finite(li)) then
        floor_l = li
      else
        floor_l = -FC_HUGE
      end if
      half_gap = 0.5_dp * zc - 0.5_dp * floor_l
      if (half_gap <= 0.0_dp) then
        zt = floor_l
        return
      end if
      half_dz = 0.5_dp * abs(dzi)
      if (half_dz >= 1.0_dp) then
        if (abs(tau) >= half_gap / half_dz) then
          zt = floor_l
        else
          half_step = half_dz * abs(tau)
          zt = max((zc - half_step) - half_step, floor_l)
        end if
      else
        half_step = half_dz * abs(tau)
        if (half_step >= half_gap) then
          zt = floor_l
        else
          zt = max((zc - half_step) - half_step, floor_l)
        end if
      end if
    end if
  end function safe_clamp_step

  !---------------------------------------------------------------------------
  ! Overflow-safe accumulate acc := min(FC_HUGE, acc + x^2).
  !---------------------------------------------------------------------------
  pure subroutine accum_sq_safe(acc, x)
    real(dp), intent(inout) :: acc
    real(dp), intent(in)    :: x
    real(dp)                :: ax, term

    ax = abs(x)
    if (ax <= SAFE_SQ_LIM) then
      term = ax * ax
      if (0.5_dp * acc + 0.5_dp * term >= 0.5_dp * FC_HUGE) then
        acc = FC_HUGE
      else
        acc = acc + term
      end if
    else
      acc = FC_HUGE
    end if
  end subroutine accum_sq_safe

  !---------------------------------------------------------------------------
  ! Validate box bounds: reject d < 0, NaN, l_i = +Inf, u_i = -Inf, or l_i > u_i.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_validate(d, l, u, status)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: l(*), u(*)
    integer(i32), intent(out) :: status

    integer(i32) :: i

    if (d < 0_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    status = 0_i32
    do i = 1_i32, d
      if (ieee_is_nan(l(i)) .or. ieee_is_nan(u(i))) then
        status = FC_ERR_INVALID_BOUNDS
        return
      end if
      if (.not. ieee_is_finite(l(i))) then
        if (l(i) > 0.0_dp) then
          status = FC_ERR_INVALID_BOUNDS
          return
        end if
      end if
      if (.not. ieee_is_finite(u(i))) then
        if (u(i) < 0.0_dp) then
          status = FC_ERR_INVALID_BOUNDS
          return
        end if
      end if
      if (l(i) > u(i)) then
        status = FC_ERR_INVALID_BOUNDS
        return
      end if
    end do
  end subroutine fc_box_validate

  !---------------------------------------------------------------------------
  ! Coordinatewise Euclidean projection z_i := min(max(z_i, l_i), u_i).
  !---------------------------------------------------------------------------
  pure subroutine fc_box_project(d, l, u, z)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: l(*), u(*)
    real(dp), intent(inout)  :: z(*)

    integer(i32) :: i

    do i = 1_i32, d
      z(i) = min(max(z(i), l(i)), u(i))
    end do
  end subroutine fc_box_project

  !---------------------------------------------------------------------------
  ! Continuous 1-Lipschitz box KKT residual:
  !   rbox(i) = z(i) - min(max(z(i) - gz(i), l(i)), u(i))
  ! Safe against FE_OVERFLOW even when z, l, u, gz approach +-FC_HUGE.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_kkt_residual(d, z, l, u, gz, rbox, box_inf, box_l2)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*)
    real(dp), intent(out)    :: rbox(*)
    real(dp), intent(out)    :: box_inf, box_l2

    integer(i32) :: i
    real(dp)     :: proj, ri, absri, scale, ssq, ratio, half_diff

    box_inf = 0.0_dp
    box_l2  = 0.0_dp
    if (d <= 0_i32) return

    scale = 0.0_dp
    ssq   = 1.0_dp

    do i = 1_i32, d
      proj = safe_clamp_step(z(i), l(i), u(i), -gz(i), 1.0_dp)
      if (abs(z(i)) <= 0.45_dp * FC_HUGE .and. abs(proj) <= 0.45_dp * FC_HUGE) then
        ri = z(i) - proj
      else
        half_diff = 0.5_dp * z(i) - 0.5_dp * proj
        if (half_diff >= 0.5_dp * FC_HUGE) then
          ri = FC_HUGE
        else if (half_diff <= -0.5_dp * FC_HUGE) then
          ri = -FC_HUGE
        else
          ri = z(i) - proj
        end if
      end if

      rbox(i) = ri
      absri   = abs(ri)
      if (absri > box_inf) box_inf = absri

      if (absri > 0.0_dp) then
        if (scale < absri) then
          ratio = scale / absri
          ssq   = 1.0_dp + ssq * (ratio * ratio)
          scale = absri
        else
          ratio = absri / scale
          ssq   = ssq + ratio * ratio
        end if
      end if
    end do

    box_l2 = scale * sqrt(ssq)
  end subroutine fc_box_kkt_residual

  !---------------------------------------------------------------------------
  ! Overflow-safe Bertsekas active-band partition:
  !   mask_box(i) = FC_BOX_FIXED  (2) if l(i) == u(i)
  !   mask_box(i) = FC_BOX_ACTIVE (1) if near bound with outward -gz(i)
  !   mask_box(i) = FC_BOX_FREE   (0) otherwise
  !---------------------------------------------------------------------------
  pure subroutine fc_box_partition(d, z, l, u, gz, box_inf, eps_band, mask_box, n_active)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: z(*), l(*), u(*), gz(*)
    real(dp), intent(in)      :: box_inf, eps_band
    integer(i32), intent(out) :: mask_box(*)
    integer(i32), intent(out) :: n_active

    integer(i32) :: i
    real(dp)     :: eps_k, eps_ki, q_width
    logical      :: near_lower, near_upper

    n_active = 0_i32
    if (d <= 0_i32) return

    eps_k = min(max(0.0_dp, eps_band), max(0.0_dp, box_inf))

    do i = 1_i32, d
      if (.not. (l(i) < u(i))) then
        mask_box(i) = FC_BOX_FIXED
        n_active    = n_active + 1_i32
      else
        if (ieee_is_finite(l(i)) .and. ieee_is_finite(u(i))) then
          q_width = 0.25_dp * u(i) - 0.25_dp * l(i)
          eps_ki  = min(eps_k, max(0.0_dp, q_width))
        else
          eps_ki = eps_k
        end if

        near_lower = .false.
        if (gz(i) > 0.0_dp .and. ieee_is_finite(l(i))) then
          if (l(i) > FC_HUGE - eps_ki) then
            near_lower = .true.
          else if (z(i) <= l(i) + eps_ki) then
            near_lower = .true.
          end if
        end if

        near_upper = .false.
        if (gz(i) < 0.0_dp .and. ieee_is_finite(u(i))) then
          if (u(i) < -FC_HUGE + eps_ki) then
            near_upper = .true.
          else if (z(i) >= u(i) - eps_ki) then
            near_upper = .true.
          end if
        end if

        if (near_lower .or. near_upper) then
          mask_box(i) = FC_BOX_ACTIVE
          n_active    = n_active + 1_i32
        else
          mask_box(i) = FC_BOX_FREE
        end if
      end if
    end do
  end subroutine fc_box_partition

  !---------------------------------------------------------------------------
  ! Populate dz(i) on FC_BOX_FIXED (0) and FC_BOX_ACTIVE (-gamma_act * gz(i)),
  ! leaving FC_BOX_FREE coordinates of dz untouched.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_active_step(d, z, l, u, gz, mask_box, gamma_box, &
      has_prev, s_prev, y_prev, dz)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*)
    integer(i32), intent(in) :: mask_box(*)
    real(dp), intent(in)     :: gamma_box
    logical, intent(in)      :: has_prev
    real(dp), intent(in)     :: s_prev(*), y_prev(*)
    real(dp), intent(inout)  :: dz(*)

    integer(i32) :: i
    real(dp)     :: gamma_fallback, gamma_act, si, yi, gzi

    if (d <= 0_i32) return
    if (.false. .and. z(1) > 0.0_dp) continue

    gamma_fallback = min(max(gamma_box, GAMMA_MIN), GAMMA_MAX)

    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FIXED .or. .not. (l(i) < u(i))) then
        dz(i) = 0.0_dp
      else if (mask_box(i) == FC_BOX_ACTIVE) then
        gamma_act = gamma_fallback
        if (has_prev) then
          si = s_prev(i)
          yi = y_prev(i)
          if (abs(si) > 1.0e-150_dp .and. abs(si) <= SAFE_SQ_LIM .and. &
              abs(yi) > 1.0e-150_dp .and. abs(yi) <= SAFE_SQ_LIM) then
            if (si * yi > 1.0e-16_dp * (si * si)) then
              gamma_act = min(max(si / yi, GAMMA_MIN), GAMMA_MAX)
            end if
          end if
        end if

        gzi = gz(i)
        if (abs(gzi) > 1.0_dp) then
          if (gamma_act >= 0.5_dp * FC_HUGE / abs(gzi)) then
            dz(i) = -sign(0.5_dp * FC_HUGE, gzi)
          else
            dz(i) = -gamma_act * gzi
          end if
        else
          dz(i) = -gamma_act * gzi
        end if
      end if
    end do
  end subroutine fc_box_active_step

  !---------------------------------------------------------------------------
  ! Right-hand directional derivative at tau = 0+:
  !   dot_z_i(0+) = 0 if (z_i <= l_i and dz_i <= 0) or (z_i >= u_i and dz_i >= 0),
  !                 else dz_i.
  !   slope0   = sum_i gz(i) * dot_z_i(0+)
  !   v_box_sq = sum_i dot_z_i(0+)^2
  !   g_box_sq = sum_{i : dot_z_i(0+) /= 0} gz(i)^2
  !---------------------------------------------------------------------------
  pure subroutine fc_box_dir_deriv0(d, z, l, u, gz, dz, slope0, v_box_sq, g_box_sq)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*), dz(*)
    real(dp), intent(out)    :: slope0, v_box_sq, g_box_sq

    integer(i32) :: i
    real(dp)     :: dot_zi, gzi, prod

    slope0   = 0.0_dp
    v_box_sq = 0.0_dp
    g_box_sq = 0.0_dp
    if (d <= 0_i32) return

    do i = 1_i32, d
      if ((z(i) <= l(i) .and. dz(i) <= 0.0_dp) .or. &
          (z(i) >= u(i) .and. dz(i) >= 0.0_dp) .or. &
          .not. (l(i) < u(i))) then
        dot_zi = 0.0_dp
      else
        dot_zi = dz(i)
      end if

      if (abs(dot_zi) > 0.0_dp) then
        gzi = gz(i)
        if (abs(gzi) <= SAFE_SQ_LIM .and. abs(dot_zi) <= SAFE_SQ_LIM) then
          prod   = gzi * dot_zi
          slope0 = slope0 + prod
        else if ((gzi > 0.0_dp .and. dot_zi > 0.0_dp) .or. &
                 (gzi < 0.0_dp .and. dot_zi < 0.0_dp)) then
          slope0 = FC_HUGE
        else
          slope0 = -FC_HUGE
        end if
        call accum_sq_safe(v_box_sq, dot_zi)
        call accum_sq_safe(g_box_sq, gzi)
      end if
    end do
  end subroutine fc_box_dir_deriv0

  !---------------------------------------------------------------------------
  ! Evaluate projected trial point z_trial(i) = P_[l,u](z(i) + tau * dz(i)),
  ! realized chord <gz, z_trial - z>, and squared displacement ||z_trial - z||_2^2.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_trial_step(d, z, l, u, gz, dz, tau, z_trial, chord_gz, step_norm_sq)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*), dz(*)
    real(dp), intent(in)     :: tau
    real(dp), intent(out)    :: z_trial(*)
    real(dp), intent(out)    :: chord_gz, step_norm_sq

    integer(i32) :: i
    real(dp)     :: zti, diff_i, half_diff, gzi

    chord_gz     = 0.0_dp
    step_norm_sq = 0.0_dp
    if (d <= 0_i32) return

    do i = 1_i32, d
      zti        = safe_clamp_step(z(i), l(i), u(i), dz(i), tau)
      z_trial(i) = zti

      if (abs(zti) <= 0.45_dp * FC_HUGE .and. abs(z(i)) <= 0.45_dp * FC_HUGE) then
        diff_i = zti - z(i)
      else
        half_diff = 0.5_dp * zti - 0.5_dp * z(i)
        if (half_diff >= 0.5_dp * FC_HUGE) then
          diff_i = FC_HUGE
        else if (half_diff <= -0.5_dp * FC_HUGE) then
          diff_i = -FC_HUGE
        else
          diff_i = zti - z(i)
        end if
      end if

      if (abs(diff_i) > 0.0_dp) then
        gzi = gz(i)
        if (abs(gzi) <= SAFE_SQ_LIM .and. abs(diff_i) <= SAFE_SQ_LIM) then
          chord_gz = chord_gz + gzi * diff_i
        else if ((gzi > 0.0_dp .and. diff_i > 0.0_dp) .or. &
                 (gzi < 0.0_dp .and. diff_i < 0.0_dp)) then
          chord_gz = FC_HUGE
        else
          chord_gz = -FC_HUGE
        end if
        call accum_sq_safe(step_norm_sq, diff_i)
      end if
    end do
  end subroutine fc_box_trial_step

  !===========================================================================
  ! v0.2.0 Coordinatewise Diagonal Active Step, Breakpoint Cap, and
  ! Overflow-Safe Dai-Fletcher / Cominetti-Mascarenhas Linear-Box Engine
  !===========================================================================

  !---------------------------------------------------------------------------
  ! Overflow-safe Neumaier compensated summation: (s, c) += term, clamped to
  ! [-FC_HUGE, +FC_HUGE] without triggering FE_OVERFLOW.
  !---------------------------------------------------------------------------
  pure subroutine neumaier_add_safe(s, c, term)
    real(dp), intent(inout) :: s, c
    real(dp), intent(in)    :: term

    real(dp) :: hs, tmp

    if (.not. (abs(term) > 0.0_dp)) return

    if (abs(s) <= 0.45_dp * FC_HUGE .and. abs(term) <= 0.45_dp * FC_HUGE) then
      tmp = s + term
      if (abs(s) >= abs(term)) then
        c = c + ((s - tmp) + term)
      else
        c = c + ((term - tmp) + s)
      end if
      s = tmp
    else
      hs = 0.5_dp * s + 0.5_dp * term
      if (hs >= 0.5_dp * FC_HUGE) then
        s = FC_HUGE
        c = 0.0_dp
      else if (hs <= -0.5_dp * FC_HUGE) then
        s = -FC_HUGE
        c = 0.0_dp
      else
        s = 2.0_dp * hs
        c = 0.0_dp
      end if
    end if
  end subroutine neumaier_add_safe

  !---------------------------------------------------------------------------
  ! Finalize Neumaier sum s + c safely without FE_OVERFLOW.
  !---------------------------------------------------------------------------
  pure function neumaier_finalize_safe(s, c) result(res_val)
    real(dp), intent(in) :: s, c
    real(dp)             :: res_val
    real(dp)             :: hs

    if (abs(s) <= 0.45_dp * FC_HUGE .and. abs(c) <= 0.45_dp * FC_HUGE) then
      res_val = s + c
    else
      hs = 0.5_dp * s + 0.5_dp * c
      if (hs >= 0.5_dp * FC_HUGE) then
        res_val = FC_HUGE
      else if (hs <= -0.5_dp * FC_HUGE) then
        res_val = -FC_HUGE
      else
        res_val = 2.0_dp * hs
      end if
    end if
  end function neumaier_finalize_safe

  !---------------------------------------------------------------------------
  ! Overflow-safe difference a_val - b_val without FE_OVERFLOW near +-FC_HUGE.
  !---------------------------------------------------------------------------
  pure function safe_sub_dp(a_val, b_val) result(diff_val)
    real(dp), intent(in) :: a_val, b_val
    real(dp)             :: diff_val
    real(dp)             :: hd

    if (abs(a_val) <= 0.45_dp * FC_HUGE .and. abs(b_val) <= 0.45_dp * FC_HUGE) then
      diff_val = a_val - b_val
    else
      hd = 0.5_dp * a_val - 0.5_dp * b_val
      if (hd >= 0.5_dp * FC_HUGE) then
        diff_val = FC_HUGE
      else if (hd <= -0.5_dp * FC_HUGE) then
        diff_val = -FC_HUGE
      else
        diff_val = a_val - b_val
      end if
    end if
  end function safe_sub_dp

  !---------------------------------------------------------------------------
  ! Overflow-safe evaluation of min(ui, max(li, yi - nu_hat * a_hat_i))
  ! where |a_hat_i| <= 1.0 and yi may lie outside [li, ui].
  !---------------------------------------------------------------------------
  pure subroutine clamp_shifted_coord(yi, li, ui, a_hat_i, nu_hat, zi, is_free)
    real(dp), intent(in)  :: yi, li, ui, a_hat_i, nu_hat
    real(dp), intent(out) :: zi
    logical, intent(out)  :: is_free

    real(dp) :: shift, val, half_val

    if (.not. (li < ui)) then
      zi      = li
      is_free = .false.
      return
    end if

    if (.not. (abs(a_hat_i) > 0.0_dp) .or. .not. (abs(nu_hat) > 0.0_dp)) then
      if (yi <= li) then
        zi      = li
        is_free = .false.
      else if (yi >= ui) then
        zi      = ui
        is_free = .false.
      else
        zi      = yi
        is_free = .true.
      end if
      return
    end if

    if (abs(yi) <= 0.45_dp * FC_HUGE .and. abs(nu_hat) <= 0.45_dp * FC_HUGE) then
      val = yi - nu_hat * a_hat_i
      if (val <= li) then
        zi      = li
        is_free = .false.
      else if (val >= ui) then
        zi      = ui
        is_free = .false.
      else
        zi      = val
        is_free = .true.
      end if
      return
    end if

    ! Since |a_hat_i| <= 1.0 and |nu_hat| <= FC_HUGE, shift never overflows
    shift    = nu_hat * a_hat_i
    half_val = 0.5_dp * yi - 0.5_dp * shift

    if (ieee_is_finite(li)) then
      if (half_val <= 0.5_dp * li) then
        zi      = li
        is_free = .false.
        return
      end if
    end if

    if (ieee_is_finite(ui)) then
      if (half_val >= 0.5_dp * ui) then
        zi      = ui
        is_free = .false.
        return
      end if
    end if

    if (abs(half_val) < 0.5_dp * FC_HUGE) then
      zi      = 2.0_dp * half_val
      is_free = .true.
    else if (half_val > 0.0_dp) then
      zi      = min(FC_HUGE, ui)
      is_free = (zi < ui)
    else
      zi      = max(-FC_HUGE, li)
      is_free = (zi > li)
    end if
  end subroutine clamp_shifted_coord

  !---------------------------------------------------------------------------
  ! Populate dz(i) on FC_BOX_FIXED (0) and FC_BOX_ACTIVE (-gamma_act * gz(i))
  ! using coordinatewise persistent diag_box(i) fallback clamped to
  ! [d_min, d_max], leaving FC_BOX_FREE coordinates of dz untouched.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_active_step_diag(d, z, l, u, gz, mask_box, diag_box, &
      d_min, d_max, has_prev, s_prev, y_prev, dz)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*)
    integer(i32), intent(in) :: mask_box(*)
    real(dp), intent(in)     :: diag_box(*)
    real(dp), intent(in)     :: d_min, d_max
    logical, intent(in)      :: has_prev
    real(dp), intent(in)     :: s_prev(*), y_prev(*)
    real(dp), intent(inout)  :: dz(*)

    integer(i32) :: i
    real(dp)     :: lo_eff, hi_eff, gamma_act, si, yi, gzi

    if (d <= 0_i32) return
    if (.false. .and. z(1) > 0.0_dp) continue

    if (ieee_is_nan(d_min)) then
      lo_eff = GAMMA_MIN
    else
      lo_eff = min(max(d_min, GAMMA_MIN), GAMMA_MAX)
    end if

    if (ieee_is_nan(d_max)) then
      hi_eff = GAMMA_MAX
    else
      hi_eff = min(max(d_max, lo_eff), GAMMA_MAX)
    end if

    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FIXED .or. .not. (l(i) < u(i))) then
        dz(i) = 0.0_dp
      else if (mask_box(i) == FC_BOX_ACTIVE) then
        if (ieee_is_nan(diag_box(i)) .or. .not. ieee_is_finite(diag_box(i))) then
          gamma_act = lo_eff
        else
          gamma_act = min(max(diag_box(i), lo_eff), hi_eff)
        end if

        if (has_prev) then
          si = s_prev(i)
          yi = y_prev(i)
          if (abs(si) > 1.0e-150_dp .and. abs(si) <= SAFE_SQ_LIM .and. &
              abs(yi) > 1.0e-30_dp  .and. abs(yi) <= SAFE_SQ_LIM) then
            if (si * yi >= 1.0e-12_dp * (si * si)) then
              gamma_act = min(max(si / yi, lo_eff), hi_eff)
            end if
          end if
        end if

        gzi = gz(i)
        if (abs(gzi) > 1.0_dp) then
          if (gamma_act >= 0.5_dp * FC_HUGE / abs(gzi)) then
            dz(i) = -sign(0.5_dp * FC_HUGE, gzi)
          else
            dz(i) = -gamma_act * gzi
          end if
        else
          dz(i) = -gamma_act * gzi
        end if
      end if
    end do
  end subroutine fc_box_active_step_diag

  !---------------------------------------------------------------------------
  ! Overflow-safe scan of earliest and latest positive free-coordinate
  ! bound-hit breakpoints along dz (skipping unbounded >= 0.5 * FC_HUGE walls).
  !---------------------------------------------------------------------------
  pure subroutine fc_box_breakpoint_cap(d, z, l, u, mask_box, dz, &
      tau_brk_min, tau_brk_max, n_free_hit)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: z(*), l(*), u(*), dz(*)
    integer(i32), intent(in)  :: mask_box(*)
    real(dp), intent(out)     :: tau_brk_min, tau_brk_max
    integer(i32), intent(out) :: n_free_hit

    integer(i32) :: i
    real(dp)     :: zc, dzi, half_gap, half_dz, tau_i

    tau_brk_min = FC_HUGE
    tau_brk_max = 0.0_dp
    n_free_hit  = 0_i32
    if (d <= 0_i32) return

    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FREE .and. (l(i) < u(i))) then
        dzi = dz(i)
        if (dzi > 0.0_dp) then
          if (ieee_is_finite(u(i)) .and. u(i) < 0.5_dp * FC_HUGE) then
            zc       = min(max(z(i), l(i)), u(i))
            half_gap = 0.5_dp * u(i) - 0.5_dp * zc
            if (half_gap > 0.0_dp) then
              half_dz = 0.5_dp * dzi
              if (half_dz >= 1.0_dp) then
                tau_i = half_gap / half_dz
              else if (half_gap < 0.5_dp * FC_HUGE * half_dz) then
                tau_i = half_gap / half_dz
              else
                tau_i = 0.5_dp * FC_HUGE
              end if
              if (tau_i > 0.0_dp) then
                n_free_hit = n_free_hit + 1_i32
                if (tau_i < tau_brk_min) tau_brk_min = tau_i
                if (tau_i > tau_brk_max) tau_brk_max = tau_i
              end if
            end if
          end if
        else if (dzi < 0.0_dp) then
          if (ieee_is_finite(l(i)) .and. l(i) > -0.5_dp * FC_HUGE) then
            zc       = min(max(z(i), l(i)), u(i))
            half_gap = 0.5_dp * zc - 0.5_dp * l(i)
            if (half_gap > 0.0_dp) then
              half_dz = 0.5_dp * abs(dzi)
              if (half_dz >= 1.0_dp) then
                tau_i = half_gap / half_dz
              else if (half_gap < 0.5_dp * FC_HUGE * half_dz) then
                tau_i = half_gap / half_dz
              else
                tau_i = 0.5_dp * FC_HUGE
              end if
              if (tau_i > 0.0_dp) then
                n_free_hit = n_free_hit + 1_i32
                if (tau_i < tau_brk_min) tau_brk_min = tau_i
                if (tau_i > tau_brk_max) tau_brk_max = tau_i
              end if
            end if
          end if
        end if
      end if
    end do
  end subroutine fc_box_breakpoint_cap

  !---------------------------------------------------------------------------
  ! Shared overflow-safe linear-box validator & normalized range analyzer:
  !   alpha_max  = max_i |a(i)| > 0
  !   a_hat_l1   = sum_i |a(i)| / alpha_max
  !   b_hat      = b / alpha_max
  !   b_hat_min  = sum_{a_i>0} (a_i/alpha_max)*l_i + sum_{a_i<0} (a_i/alpha_max)*u_i
  !   b_hat_max  = sum_{a_i>0} (a_i/alpha_max)*u_i + sum_{a_i<0} (a_i/alpha_max)*l_i
  ! Skips a(i) == 0 so 0 * FC_HUGE / 0 * Inf is never evaluated.
  !---------------------------------------------------------------------------
  pure subroutine analyze_lin_box(d, l, u, a, b, alpha_max, a_hat_l1, b_hat, &
      b_hat_min, b_hat_max, tau_hat_feas, has_unbounded_lo, has_unbounded_hi, status)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: l(*), u(*), a(*)
    real(dp), intent(in)      :: b
    real(dp), intent(out)     :: alpha_max, a_hat_l1, b_hat
    real(dp), intent(out)     :: b_hat_min, b_hat_max, tau_hat_feas
    logical, intent(out)      :: has_unbounded_lo, has_unbounded_hi
    integer(i32), intent(out) :: status

    integer(i32) :: i
    real(dp)     :: ai, abs_ai, a_hat_i, inv_alpha
    real(dp)     :: s_lo, c_lo, s_hi, c_hi

    alpha_max        = 0.0_dp
    a_hat_l1         = 0.0_dp
    b_hat            = 0.0_dp
    b_hat_min        = 0.0_dp
    b_hat_max        = 0.0_dp
    tau_hat_feas     = 0.0_dp
    has_unbounded_lo = .false.
    has_unbounded_hi = .false.

    if (d < 1_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    call fc_box_validate(d, l, u, status)
    if (status /= 0_i32) return

    if (ieee_is_nan(b) .or. .not. ieee_is_finite(b)) then
      status = FC_ERR_INFEASIBLE_LIN
      return
    end if

    do i = 1_i32, d
      ai = a(i)
      if (ieee_is_nan(ai) .or. .not. ieee_is_finite(ai)) then
        status = FC_ERR_INFEASIBLE_LIN
        return
      end if
      abs_ai = abs(ai)
      if (abs_ai > alpha_max) alpha_max = abs_ai
    end do

    if (.not. (alpha_max > 0.0_dp)) then
      status = FC_ERR_INFEASIBLE_LIN
      return
    end if

    if (alpha_max < 1.0_dp) then
      if (abs(b) / max(1.0_dp, real(d, dp)) > 0.45_dp * FC_HUGE * alpha_max) then
        status = FC_ERR_INFEASIBLE_LIN
        return
      end if
      if (abs(b) >= 0.45_dp * FC_HUGE * alpha_max) then
        b_hat = sign(0.45_dp * FC_HUGE, b)
      else
        b_hat = b / alpha_max
      end if
      if (1.0e-14_dp >= 0.45_dp * FC_HUGE * alpha_max) then
        tau_hat_feas = 0.45_dp * FC_HUGE
      else
        tau_hat_feas = (1.0e-14_dp / alpha_max) + 1.0e-14_dp * abs(b_hat)
      end if
    else
      inv_alpha    = 1.0_dp / alpha_max
      b_hat        = b * inv_alpha
      tau_hat_feas = 1.0e-14_dp * inv_alpha + 1.0e-14_dp * abs(b_hat)
    end if

    s_lo = 0.0_dp; c_lo = 0.0_dp
    s_hi = 0.0_dp; c_hi = 0.0_dp

    do i = 1_i32, d
      ai = a(i)
      if (abs(ai) > 0.0_dp) then
        a_hat_i  = ai / alpha_max
        a_hat_l1 = a_hat_l1 + abs(a_hat_i)

        if (a_hat_i > 0.0_dp) then
          if (.not. ieee_is_finite(l(i)) .or. l(i) <= -0.5_dp * FC_HUGE) then
            has_unbounded_lo = .true.
          else
            call neumaier_add_safe(s_lo, c_lo, a_hat_i * l(i))
          end if

          if (.not. ieee_is_finite(u(i)) .or. u(i) >= 0.5_dp * FC_HUGE) then
            has_unbounded_hi = .true.
          else
            call neumaier_add_safe(s_hi, c_hi, a_hat_i * u(i))
          end if
        else
          if (.not. ieee_is_finite(u(i)) .or. u(i) >= 0.5_dp * FC_HUGE) then
            has_unbounded_lo = .true.
          else
            call neumaier_add_safe(s_lo, c_lo, a_hat_i * u(i))
          end if

          if (.not. ieee_is_finite(l(i)) .or. l(i) <= -0.5_dp * FC_HUGE) then
            has_unbounded_hi = .true.
          else
            call neumaier_add_safe(s_hi, c_hi, a_hat_i * l(i))
          end if
        end if
      end if
    end do

    b_hat_min = neumaier_finalize_safe(s_lo, c_lo)
    b_hat_max = neumaier_finalize_safe(s_hi, c_hi)

    if ((.not. has_unbounded_lo) .and. (safe_sub_dp(b_hat_min, b_hat) > tau_hat_feas)) then
      status = FC_ERR_INFEASIBLE_LIN
      return
    end if
    if ((.not. has_unbounded_hi) .and. (safe_sub_dp(b_hat, b_hat_max) > tau_hat_feas)) then
      status = FC_ERR_INFEASIBLE_LIN
      return
    end if

    status = 0_i32
  end subroutine analyze_lin_box

  !---------------------------------------------------------------------------
  ! Validate linear-box polytope {z in [l, u]^d : a^T z = b}:
  ! Returns 0 if feasible, FC_ERR_INVALID_DIM (-1) if d < 1,
  ! FC_ERR_INVALID_BOUNDS (-3) if [l, u] invalid, or
  ! FC_ERR_INFEASIBLE_LIN (-10) if a/b non-finite, ||a||_inf == 0, or infeasible.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_lin_validate(d, l, u, a, b, status)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: l(*), u(*), a(*)
    real(dp), intent(in)      :: b
    integer(i32), intent(out) :: status

    real(dp) :: alpha_max, a_hat_l1, b_hat, b_hat_min, b_hat_max, tau_hat_feas
    logical  :: has_unbounded_lo, has_unbounded_hi

    call analyze_lin_box(d, l, u, a, b, alpha_max, a_hat_l1, b_hat, &
      b_hat_min, b_hat_max, tau_hat_feas, has_unbounded_lo, has_unbounded_hi, status)
  end subroutine fc_box_lin_validate

  !---------------------------------------------------------------------------
  ! Evaluate normalized psi(nu_hat) = sum_i a_hat_i * P_[l_i,u_i](y_i - nu_hat*a_hat_i) - b_hat,
  ! active derivative dpsi(nu_hat) = -sum_{free} a_hat_i^2 <= 0, and exact
  ! stopping criterion |a^T z - b| <= 1e-14 * (1 + |b| + ||a||_1 ||z||_inf)
  ! in one O(d) scalar pass with Neumaier compensated summation.
  !---------------------------------------------------------------------------
  pure subroutine eval_psi_pass(d, y, l, u, a, alpha_max, a_hat_l1, b_hat, &
      nu_hat, psi_hat, dpsi_hat, is_conv)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: y(*), l(*), u(*), a(*)
    real(dp), intent(in)     :: alpha_max, a_hat_l1, b_hat, nu_hat
    real(dp), intent(out)    :: psi_hat, dpsi_hat
    logical, intent(out)     :: is_conv

    integer(i32) :: i
    real(dp)     :: ai, a_hat_i, zi, s_psi, c_psi, z_inf_a, base_hat_tol, az_term, tol_hat
    logical      :: is_free

    s_psi   = 0.0_dp
    c_psi   = 0.0_dp
    dpsi_hat = 0.0_dp
    z_inf_a = 0.0_dp

    do i = 1_i32, d
      ai = a(i)
      if (abs(ai) > 0.0_dp) then
        a_hat_i = ai / alpha_max
        call clamp_shifted_coord(y(i), l(i), u(i), a_hat_i, nu_hat, zi, is_free)
        call neumaier_add_safe(s_psi, c_psi, a_hat_i * zi)
        if (is_free) then
          dpsi_hat = dpsi_hat - a_hat_i * a_hat_i
        end if
        if (abs(zi) > z_inf_a) z_inf_a = abs(zi)
      end if
    end do

    call neumaier_add_safe(s_psi, c_psi, -b_hat)
    psi_hat = neumaier_finalize_safe(s_psi, c_psi)

    if (alpha_max >= 1.0_dp) then
      base_hat_tol = (1.0_dp / alpha_max) + abs(b_hat)
    else if (1.0_dp < 0.2_dp * FC_HUGE * alpha_max) then
      base_hat_tol = (1.0_dp / alpha_max) + abs(b_hat)
    else
      base_hat_tol = 0.2_dp * FC_HUGE
    end if

    if (z_inf_a <= 1.0_dp .or. a_hat_l1 <= 0.2_dp * FC_HUGE / z_inf_a) then
      az_term = a_hat_l1 * z_inf_a
    else
      az_term = 0.2_dp * FC_HUGE
    end if

    tol_hat = 1.0e-14_dp * base_hat_tol + 1.0e-14_dp * az_term
    is_conv = (abs(psi_hat) <= tol_hat)
  end subroutine eval_psi_pass

  !---------------------------------------------------------------------------
  ! Helper converting normalized nu_hat to unnormalized nu_out = nu_hat / alpha_max
  ! without FE_OVERFLOW.
  !---------------------------------------------------------------------------
  pure function unscale_nu_safe(nu_hat, alpha_max) result(nu_val)
    real(dp), intent(in) :: nu_hat, alpha_max
    real(dp)             :: nu_val

    if (.not. (abs(nu_hat) > 0.0_dp)) then
      nu_val = 0.0_dp
    else if (alpha_max >= 1.0_dp) then
      nu_val = nu_hat / alpha_max
    else if (abs(nu_hat) < 0.5_dp * FC_HUGE * alpha_max) then
      nu_val = nu_hat / alpha_max
    else
      nu_val = sign(0.5_dp * FC_HUGE, nu_hat)
    end if
  end function unscale_nu_safe

  !---------------------------------------------------------------------------
  ! Overflow-safe check that |num / den| < cap_frac * FC_HUGE (cap_frac <= 0.5)
  ! without overflowing cap_frac * FC_HUGE * |den| when |den| > 1.
  !---------------------------------------------------------------------------
  pure function safe_can_div(num, den, cap_frac) result(ok)
    real(dp), intent(in) :: num, den, cap_frac
    logical              :: ok
    real(dp)             :: ad

    ad = abs(den)
    if (.not. (ad > 0.0_dp)) then
      ok = .false.
    else if (ad >= 1.0_dp) then
      ok = (abs(num) / ad < cap_frac * FC_HUGE)
    else
      ok = (abs(num) < (cap_frac * FC_HUGE) * ad)
    end if
  end function safe_can_div

  !---------------------------------------------------------------------------
  ! Scalar-memory Dai-Fletcher / Cominetti-Mascarenhas semismooth Newton-secant
  ! + safeguarded bisection Euclidean projection onto {z in [l, u]^d : a^T z = b}.
  ! Overwrites z(1:d) in-place with P_C(z_in) and outputs dual multiplier nu_out.
  ! Zero array workspace; converges in <= 60 passes to <= 1e-14 or 2 ULPs.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_lin_project(d, l, u, a, b, z, nu_out, status)
    integer(i32), intent(in)    :: d
    real(dp), intent(in)        :: l(*), u(*), a(*)
    real(dp), intent(in)        :: b
    real(dp), intent(inout)     :: z(*)
    real(dp), intent(out)       :: nu_out
    integer(i32), intent(out)   :: status

    integer(i32), parameter :: MAX_TOTAL_PASSES = 60_i32
    integer(i32) :: i, n_passes
    real(dp)     :: alpha_max, a_hat_l1, b_hat, b_hat_min, b_hat_max, tau_hat_feas
    real(dp)     :: ai, a_hat_i, yi, zi, diff_yl, diff_yu, brk_val
    real(dp)     :: nu_sing, s_ay, c_ay, s_aa, kink_lo_min, kink_hi_max
    real(dp)     :: nu_0, psi_0, dpsi_0, nu_1, psi_1, dpsi_1
    real(dp)     :: nu_lo, psi_lo, dpsi_lo, nu_hi, psi_hi, dpsi_hi
    real(dp)     :: nu_cur, psi_cur, dpsi_cur, abs_psi_prev
    real(dp)     :: nu_best, abs_psi_best, nu_try, psi_try, dpsi_try
    real(dp)     :: delta, step_n, width, half_w, nu_next, sec_frac
    logical      :: has_unbounded_lo, has_unbounded_hi, is_free, is_conv
    logical      :: has_kink_lo, has_kink_hi, has_lo, has_hi, have_sing, have_newton

    nu_out = 0.0_dp
    call analyze_lin_box(d, l, u, a, b, alpha_max, a_hat_l1, b_hat, &
      b_hat_min, b_hat_max, tau_hat_feas, has_unbounded_lo, has_unbounded_hi, status)
    if (status /= 0_i32) return

    do i = 1_i32, d
      if (ieee_is_nan(z(i)) .or. .not. ieee_is_finite(z(i))) then
        status = FC_ERR_INFEASIBLE_LIN
        return
      end if
    end do

    ! 1. Exact O(d) boundary resolution when b == b_min (singleton lower face)
    if ((.not. has_unbounded_lo) .and. (safe_sub_dp(b_hat, b_hat_min) <= tau_hat_feas)) then
      nu_sing   = 0.0_dp
      have_sing = .false.
      do i = 1_i32, d
        ai = a(i)
        yi = z(i)
        if (abs(ai) > 0.0_dp) then
          a_hat_i = ai / alpha_max
          if (a_hat_i > 0.0_dp) then
            z(i)    = l(i)
            diff_yl = safe_sub_dp(yi, l(i))
            if ((l(i) < u(i)) .and. abs(diff_yl) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
              brk_val = diff_yl / a_hat_i
              if (.not. have_sing .or. brk_val > nu_sing) then
                nu_sing   = brk_val
                have_sing = .true.
              end if
            end if
          else
            z(i)    = u(i)
            diff_yu = safe_sub_dp(yi, u(i))
            if ((l(i) < u(i)) .and. abs(diff_yu) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
              brk_val = diff_yu / a_hat_i
              if (.not. have_sing .or. brk_val > nu_sing) then
                nu_sing   = brk_val
                have_sing = .true.
              end if
            end if
          end if
        else
          z(i) = min(max(yi, l(i)), u(i))
        end if
      end do
      nu_out = unscale_nu_safe(nu_sing, alpha_max)
      status = 0_i32
      return
    end if

    ! 2. Exact O(d) boundary resolution when b == b_max (singleton upper face)
    if ((.not. has_unbounded_hi) .and. (safe_sub_dp(b_hat_max, b_hat) <= tau_hat_feas)) then
      nu_sing   = 0.0_dp
      have_sing = .false.
      do i = 1_i32, d
        ai = a(i)
        yi = z(i)
        if (abs(ai) > 0.0_dp) then
          a_hat_i = ai / alpha_max
          if (a_hat_i > 0.0_dp) then
            z(i)    = u(i)
            diff_yu = safe_sub_dp(yi, u(i))
            if ((l(i) < u(i)) .and. abs(diff_yu) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
              brk_val = diff_yu / a_hat_i
              if (.not. have_sing .or. brk_val < nu_sing) then
                nu_sing   = brk_val
                have_sing = .true.
              end if
            end if
          else
            z(i)    = l(i)
            diff_yl = safe_sub_dp(yi, l(i))
            if ((l(i) < u(i)) .and. abs(diff_yl) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
              brk_val = diff_yl / a_hat_i
              if (.not. have_sing .or. brk_val < nu_sing) then
                nu_sing   = brk_val
                have_sing = .true.
              end if
            end if
          end if
        else
          z(i) = min(max(yi, l(i)), u(i))
        end if
      end do
      nu_out = unscale_nu_safe(nu_sing, alpha_max)
      status = 0_i32
      return
    end if

    ! 3. Scan unconstrained hyperplane estimate and finite kink range in O(d)
    s_ay        = 0.0_dp
    c_ay        = 0.0_dp
    s_aa        = 0.0_dp
    kink_lo_min =  FC_HUGE
    kink_hi_max = -FC_HUGE
    has_kink_lo = (.not. has_unbounded_hi)
    has_kink_hi = (.not. has_unbounded_lo)

    do i = 1_i32, d
      ai = a(i)
      if (abs(ai) > 0.0_dp) then
        a_hat_i = ai / alpha_max
        yi      = z(i)
        if (l(i) < u(i)) then
          call neumaier_add_safe(s_ay, c_ay, a_hat_i * yi)
          s_aa = s_aa + a_hat_i * a_hat_i

          if (a_hat_i > 0.0_dp) then
            if (has_kink_lo) then
              diff_yu = safe_sub_dp(yi, u(i))
              if (abs(diff_yu) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
                brk_val = diff_yu / a_hat_i
                if (brk_val < kink_lo_min) kink_lo_min = brk_val
              else
                has_kink_lo = .false.
              end if
            end if
            if (has_kink_hi) then
              diff_yl = safe_sub_dp(yi, l(i))
              if (abs(diff_yl) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
                brk_val = diff_yl / a_hat_i
                if (brk_val > kink_hi_max) kink_hi_max = brk_val
              else
                has_kink_hi = .false.
              end if
            end if
          else
            if (has_kink_lo) then
              diff_yl = safe_sub_dp(yi, l(i))
              if (abs(diff_yl) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
                brk_val = diff_yl / a_hat_i
                if (brk_val < kink_lo_min) kink_lo_min = brk_val
              else
                has_kink_lo = .false.
              end if
            end if
            if (has_kink_hi) then
              diff_yu = safe_sub_dp(yi, u(i))
              if (abs(diff_yu) < 0.2_dp * FC_HUGE * abs(a_hat_i)) then
                brk_val = diff_yu / a_hat_i
                if (brk_val > kink_hi_max) kink_hi_max = brk_val
              else
                has_kink_hi = .false.
              end if
            end if
          end if
        else
          call neumaier_add_safe(s_ay, c_ay, a_hat_i * l(i))
        end if
      end if
    end do
    call neumaier_add_safe(s_ay, c_ay, -b_hat)

    ! 4. Pass 1: Test nu_hat = 0 (fast exit if y already satisfies a^T P_[l,u](y) == b)
    nu_0     = 0.0_dp
    n_passes = 1_i32
    call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
      nu_0, psi_0, dpsi_0, is_conv)
    if (is_conv) then
      nu_best = nu_0
    else
      nu_best      = nu_0
      abs_psi_best = abs(psi_0)
      has_lo       = .false.
      has_hi       = .false.
      nu_lo        = 0.0_dp; psi_lo = 0.0_dp; dpsi_lo = 0.0_dp
      nu_hi        = 0.0_dp; psi_hi = 0.0_dp; dpsi_hi = 0.0_dp

      if (psi_0 > 0.0_dp) then
        has_lo  = .true.
        nu_lo   = nu_0
        psi_lo  = psi_0
        dpsi_lo = dpsi_0
      else
        has_hi  = .true.
        nu_hi   = nu_0
        psi_hi  = psi_0
        dpsi_hi = dpsi_0
      end if

      ! Pass 2: Test affine Newton step from 0 (or unconstrained hyperplane root)
      if (dpsi_0 < -1.0e-30_dp .and. safe_can_div(psi_0, dpsi_0, 0.2_dp)) then
        nu_1 = -psi_0 / dpsi_0
      else if (s_aa > 1.0e-30_dp) then
        brk_val = neumaier_finalize_safe(s_ay, c_ay)
        if (safe_can_div(brk_val, s_aa, 0.2_dp)) then
          nu_1 = brk_val / s_aa
        else
          nu_1 = sign(0.2_dp * FC_HUGE, brk_val)
        end if
      else
        nu_1 = sign(1.0_dp, psi_0)
      end if

      if (has_kink_lo .and. nu_1 < kink_lo_min) nu_1 = kink_lo_min
      if (has_kink_hi .and. nu_1 > kink_hi_max) nu_1 = kink_hi_max

      n_passes = n_passes + 1_i32
      call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
        nu_1, psi_1, dpsi_1, is_conv)
      if (abs(psi_1) < abs_psi_best) then
        nu_best      = nu_1
        abs_psi_best = abs(psi_1)
      end if

      if (.not. is_conv) then
        if (psi_1 > 0.0_dp) then
          if (.not. has_lo .or. nu_1 > nu_lo) then
            has_lo  = .true.
            nu_lo   = nu_1
            psi_lo  = psi_1
            dpsi_lo = dpsi_1
          end if
        else
          if (.not. has_hi .or. nu_1 < nu_hi) then
            has_hi  = .true.
            nu_hi   = nu_1
            psi_hi  = psi_1
            dpsi_hi = dpsi_1
          end if
        end if

        ! If unconstrained hyperplane shift was skipped on Pass 2 and still one-sided,
        ! jump straight toward the unconstrained hyperplane / finite kink window!
        if (.not. has_lo .and. s_aa > 1.0e-30_dp .and. n_passes < MAX_TOTAL_PASSES) then
          brk_val = neumaier_finalize_safe(s_ay, c_ay)
          if (safe_can_div(brk_val, s_aa, 0.2_dp)) then
            nu_try = brk_val / s_aa
          else
            nu_try = -0.2_dp * FC_HUGE
          end if
          if (has_kink_lo .and. nu_try < kink_lo_min) nu_try = kink_lo_min
          if (nu_try < nu_hi) then
            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_try, psi_try, dpsi_try, is_conv)
            if (abs(psi_try) < abs_psi_best) then
              nu_best      = nu_try
              abs_psi_best = abs(psi_try)
            end if
            if (psi_try >= 0.0_dp) then
              has_lo  = .true.
              nu_lo   = nu_try
              psi_lo  = psi_try
              dpsi_lo = dpsi_try
            else
              nu_hi   = nu_try
              psi_hi  = psi_try
              dpsi_hi = dpsi_try
            end if
          end if
        else if (.not. has_hi .and. s_aa > 1.0e-30_dp .and. n_passes < MAX_TOTAL_PASSES) then
          brk_val = neumaier_finalize_safe(s_ay, c_ay)
          if (safe_can_div(brk_val, s_aa, 0.2_dp)) then
            nu_try = brk_val / s_aa
          else
            nu_try = 0.2_dp * FC_HUGE
          end if
          if (has_kink_hi .and. nu_try > kink_hi_max) nu_try = kink_hi_max
          if (nu_try > nu_lo) then
            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_try, psi_try, dpsi_try, is_conv)
            if (abs(psi_try) < abs_psi_best) then
              nu_best      = nu_try
              abs_psi_best = abs(psi_try)
            end if
            if (psi_try <= 0.0_dp) then
              has_hi  = .true.
              nu_hi   = nu_try
              psi_hi  = psi_try
              dpsi_hi = dpsi_try
            else
              nu_lo   = nu_try
              psi_lo  = psi_try
              dpsi_lo = dpsi_try
            end if
          end if
        end if

        ! Bracket left (find nu_lo with psi(nu_lo) >= 0) if still needed
        if (.not. is_conv .and. .not. has_lo) then
          delta = max(1.0_dp, abs(nu_hi))
          do while (.not. has_lo .and. .not. is_conv .and. n_passes < 12_i32)
            if (dpsi_hi < -1.0e-30_dp .and. safe_can_div(psi_hi, dpsi_hi, 0.1_dp)) then
              step_n = 2.0_dp * abs(psi_hi / dpsi_hi)
              if (step_n > delta) delta = step_n
            end if
            if (nu_hi <= -0.45_dp * FC_HUGE + delta) then
              nu_try = -0.45_dp * FC_HUGE
            else
              nu_try = nu_hi - delta
            end if
            if (has_kink_lo .and. nu_try < kink_lo_min) nu_try = kink_lo_min

            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_try, psi_try, dpsi_try, is_conv)
            if (abs(psi_try) < abs_psi_best) then
              nu_best      = nu_try
              abs_psi_best = abs(psi_try)
            end if
            if (psi_try >= 0.0_dp .or. (has_kink_lo .and. .not. (nu_try > kink_lo_min))) then
              has_lo  = .true.
              nu_lo   = nu_try
              psi_lo  = max(0.0_dp, psi_try)
              dpsi_lo = dpsi_try
            else
              nu_hi   = nu_try
              psi_hi  = psi_try
              dpsi_hi = dpsi_try
              if (delta < 0.02_dp * FC_HUGE) then
                delta = 16.0_dp * delta
              else
                delta = 0.45_dp * FC_HUGE
              end if
              if (.not. (nu_try > -0.45_dp * FC_HUGE)) exit
            end if
          end do
          if (.not. has_lo .and. has_kink_lo .and. n_passes < MAX_TOTAL_PASSES) then
            nu_lo   = kink_lo_min
            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_lo, psi_lo, dpsi_lo, is_conv)
            if (abs(psi_lo) < abs_psi_best) then
              nu_best      = nu_lo
              abs_psi_best = abs(psi_lo)
            end if
            has_lo = .true.
          end if
        end if

        ! Bracket right (find nu_hi with psi(nu_hi) <= 0) if still needed
        if (.not. is_conv .and. .not. has_hi) then
          delta = max(1.0_dp, abs(nu_lo))
          do while (.not. has_hi .and. .not. is_conv .and. n_passes < 12_i32)
            if (dpsi_lo < -1.0e-30_dp .and. safe_can_div(psi_lo, dpsi_lo, 0.1_dp)) then
              step_n = 2.0_dp * abs(psi_lo / dpsi_lo)
              if (step_n > delta) delta = step_n
            end if
            if (nu_lo >= 0.45_dp * FC_HUGE - delta) then
              nu_try = 0.45_dp * FC_HUGE
            else
              nu_try = nu_lo + delta
            end if
            if (has_kink_hi .and. nu_try > kink_hi_max) nu_try = kink_hi_max

            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_try, psi_try, dpsi_try, is_conv)
            if (abs(psi_try) < abs_psi_best) then
              nu_best      = nu_try
              abs_psi_best = abs(psi_try)
            end if
            if (psi_try <= 0.0_dp .or. (has_kink_hi .and. .not. (nu_try < kink_hi_max))) then
              has_hi  = .true.
              nu_hi   = nu_try
              psi_hi  = min(0.0_dp, psi_try)
              dpsi_hi = dpsi_try
            else
              nu_lo   = nu_try
              psi_lo  = psi_try
              dpsi_lo = dpsi_try
              if (delta < 0.02_dp * FC_HUGE) then
                delta = 16.0_dp * delta
              else
                delta = 0.45_dp * FC_HUGE
              end if
              if (.not. (nu_try < 0.45_dp * FC_HUGE)) exit
            end if
          end do
          if (.not. has_hi .and. has_kink_hi .and. n_passes < MAX_TOTAL_PASSES) then
            nu_hi   = kink_hi_max
            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_hi, psi_hi, dpsi_hi, is_conv)
            if (abs(psi_hi) < abs_psi_best) then
              nu_best      = nu_hi
              abs_psi_best = abs(psi_hi)
            end if
            has_hi = .true.
          end if
        end if

        ! 5. Safeguarded Dai-Fletcher / Cominetti-Mascarenhas loop on [nu_lo, nu_hi]
        if (.not. is_conv .and. has_lo .and. has_hi) then
          if (abs(psi_lo) <= abs(psi_hi)) then
            nu_cur   = nu_lo
            psi_cur  = psi_lo
            dpsi_cur = dpsi_lo
          else
            nu_cur   = nu_hi
            psi_cur  = psi_hi
            dpsi_cur = dpsi_hi
          end if
          abs_psi_prev = max(abs(psi_lo), abs(psi_hi))

          do while (.not. is_conv .and. n_passes < MAX_TOTAL_PASSES)
            half_w = 0.5_dp * nu_hi - 0.5_dp * nu_lo
            if (half_w <= FC_EPS * (1.0_dp + min(abs(nu_lo), 0.2_dp * FC_HUGE) + &
                                             min(abs(nu_hi), 0.2_dp * FC_HUGE))) then
              exit
            end if

            if (abs(psi_cur) <= 0.5_dp * abs_psi_prev) then
              ! Try exact affine semismooth Newton step first
              have_newton = .false.
              nu_next     = 0.5_dp * nu_lo + 0.5_dp * nu_hi
              if (dpsi_cur < -1.0e-30_dp .and. safe_can_div(psi_cur, dpsi_cur, 0.2_dp)) then
                nu_next     = safe_sub_dp(nu_cur, psi_cur / dpsi_cur)
                have_newton = (nu_next > nu_lo .and. nu_next < nu_hi)
              end if

              if (.not. have_newton) then
                ! Fall back to safeguarded secant across [nu_lo, nu_hi]
                if (half_w < 0.45_dp * FC_HUGE .and. &
                    safe_sub_dp(psi_lo, psi_hi) > 1.0e-60_dp) then
                  width    = 2.0_dp * half_w
                  sec_frac = psi_lo / safe_sub_dp(psi_lo, psi_hi)
                  sec_frac = min(max(sec_frac, 0.05_dp), 0.95_dp)
                  nu_next  = nu_lo + sec_frac * width
                else
                  nu_next = 0.5_dp * nu_lo + 0.5_dp * nu_hi
                end if
              end if
            else
              ! Residual failed to halve -> enforce exact midpoint bisection
              nu_next = 0.5_dp * nu_lo + 0.5_dp * nu_hi
            end if

            if (.not. (nu_next > nu_lo .and. nu_next < nu_hi)) then
              nu_next = 0.5_dp * nu_lo + 0.5_dp * nu_hi
              if (.not. (nu_next > nu_lo .and. nu_next < nu_hi)) exit
            end if

            n_passes = n_passes + 1_i32
            call eval_psi_pass(d, z, l, u, a, alpha_max, a_hat_l1, b_hat, &
              nu_next, psi_try, dpsi_try, is_conv)
            if (abs(psi_try) < abs_psi_best) then
              nu_best      = nu_next
              abs_psi_best = abs(psi_try)
            end if

            if (psi_try > 0.0_dp) then
              nu_lo   = nu_next
              psi_lo  = psi_try
              dpsi_lo = dpsi_try
            else if (psi_try < 0.0_dp) then
              nu_hi   = nu_next
              psi_hi  = psi_try
              dpsi_hi = dpsi_try
            else
              nu_best = nu_next
              exit
            end if

            abs_psi_prev = abs(psi_cur)
            nu_cur       = nu_next
            psi_cur      = psi_try
            dpsi_cur     = dpsi_try
          end do
        end if
      end if
    end if

    ! 6. Materialize optimal projection z* = P_[l,u](y - nu_best * a_hat) in-place
    do i = 1_i32, d
      ai = a(i)
      if (abs(ai) > 0.0_dp) then
        a_hat_i = ai / alpha_max
        call clamp_shifted_coord(z(i), l(i), u(i), a_hat_i, nu_best, zi, is_free)
        z(i) = zi
      else
        z(i) = min(max(z(i), l(i)), u(i))
      end if
    end do

    nu_out = unscale_nu_safe(nu_best, alpha_max)
    status = 0_i32
  end subroutine fc_box_lin_project

  !---------------------------------------------------------------------------
  ! Continuous 1-Lipschitz constrained KKT residual on {z in [l,u]^d : a^T z = b}:
  !   z_proj_tmp = P_C(z - gz), rbox = z - z_proj_tmp, box_inf = ||rbox||_inf,
  !   box_l2 = ||rbox||_2, and multiplier estimate nu_out.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_lin_kkt_residual(d, z, l, u, a, b, gz, &
      z_proj_tmp, rbox, nu_out, box_inf, box_l2, status)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: z(*), l(*), u(*), a(*), gz(*)
    real(dp), intent(in)      :: b
    real(dp), intent(out)     :: z_proj_tmp(*), rbox(*)
    real(dp), intent(out)     :: nu_out, box_inf, box_l2
    integer(i32), intent(out) :: status

    integer(i32) :: i
    real(dp)     :: ri, absri, scale, ssq, ratio

    nu_out  = 0.0_dp
    box_inf = 0.0_dp
    box_l2  = 0.0_dp
    status  = 0_i32

    if (d < 1_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    do i = 1_i32, d
      if (ieee_is_nan(z(i)) .or. ieee_is_nan(gz(i)) .or. &
          .not. ieee_is_finite(z(i)) .or. .not. ieee_is_finite(gz(i))) then
        status = FC_ERR_INFEASIBLE_LIN
        return
      end if
      z_proj_tmp(i) = safe_sub_dp(z(i), gz(i))
    end do

    call fc_box_lin_project(d, l, u, a, b, z_proj_tmp, nu_out, status)
    if (status /= 0_i32) return

    scale = 0.0_dp
    ssq   = 1.0_dp
    do i = 1_i32, d
      ri      = safe_sub_dp(z(i), z_proj_tmp(i))
      rbox(i) = ri
      absri   = abs(ri)
      if (absri > box_inf) box_inf = absri

      if (absri > 0.0_dp) then
        if (scale < absri) then
          ratio = scale / absri
          ssq   = 1.0_dp + ssq * (ratio * ratio)
          scale = absri
        else
          ratio = absri / scale
          ssq   = ssq + ratio * ratio
        end if
      end if
    end do

    box_l2 = scale * sqrt(ssq)
  end subroutine fc_box_lin_kkt_residual

  !---------------------------------------------------------------------------
  ! Boundary active-face partition relative to reduced Lagrangian gradient
  ! ghat_z = gz + nu_k * a on {z in [l, u]^d : a^T z = b}:
  !   mask_box(i) = FC_BOX_FIXED  (2) if l(i) == u(i)  -> dz(i) = 0
  !   mask_box(i) = FC_BOX_ACTIVE (1) if on bound with outward -ghat_z(i),
  !                 snapping tiny <= 1e-12*(1+|z_i|) boundary gaps and
  !                 accumulating delta_b = -sum_{i in A_k} a(i) * dz(i)
  !   mask_box(i) = FC_BOX_FREE   (0) otherwise        -> dz(i) = 0
  !---------------------------------------------------------------------------
  pure subroutine fc_box_lin_partition(d, z, l, u, a, gz, nu_k, &
      mask_box, dz, delta_b, n_active)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: z(*), l(*), u(*), a(*), gz(*)
    real(dp), intent(in)      :: nu_k
    integer(i32), intent(out) :: mask_box(*)
    real(dp), intent(out)     :: dz(*)
    real(dp), intent(out)     :: delta_b
    integer(i32), intent(out) :: n_active

    integer(i32) :: i
    real(dp)     :: ai, zi, gzi, nu_ai, ghat_i, tau_bnd_i, q_width, dzi, s_db, c_db
    logical      :: near_lower, near_upper

    delta_b  = 0.0_dp
    n_active = 0_i32
    if (d <= 0_i32) return

    s_db = 0.0_dp
    c_db = 0.0_dp

    do i = 1_i32, d
      dz(i) = 0.0_dp
      if (.not. (l(i) < u(i))) then
        mask_box(i) = FC_BOX_FIXED
        n_active    = n_active + 1_i32
      else
        ai  = a(i)
        zi  = z(i)
        gzi = gz(i)

        if (.not. (abs(ai) > 0.0_dp) .or. .not. (abs(nu_k) > 0.0_dp) .or. &
            ieee_is_nan(nu_k) .or. .not. ieee_is_finite(nu_k)) then
          ghat_i = gzi
        else if (abs(nu_k) <= SAFE_SQ_LIM .and. abs(ai) <= SAFE_SQ_LIM) then
          nu_ai  = nu_k * ai
          ghat_i = safe_sub_dp(gzi, -nu_ai)
        else if (abs(ai) >= 1.0_dp .and. abs(nu_k) >= 0.45_dp * FC_HUGE / abs(ai)) then
          ghat_i = sign(FC_HUGE, nu_k) * sign(1.0_dp, ai)
        else
          nu_ai  = nu_k * ai
          ghat_i = safe_sub_dp(gzi, -nu_ai)
        end if

        tau_bnd_i = 1.0e-12_dp + 1.0e-12_dp * min(abs(zi), 0.5_dp * FC_HUGE)
        if (ieee_is_finite(l(i)) .and. ieee_is_finite(u(i))) then
          q_width   = 0.25_dp * u(i) - 0.25_dp * l(i)
          tau_bnd_i = min(tau_bnd_i, max(0.0_dp, q_width))
        end if

        near_lower = .false.
        if (ghat_i >= 0.0_dp .and. ieee_is_finite(l(i))) then
          if (l(i) > FC_HUGE - tau_bnd_i) then
            near_lower = .true.
          else if (zi <= l(i) + tau_bnd_i) then
            near_lower = .true.
          end if
        end if

        near_upper = .false.
        if (.not. near_lower .and. ghat_i <= 0.0_dp .and. ieee_is_finite(u(i))) then
          if (u(i) < -FC_HUGE + tau_bnd_i) then
            near_upper = .true.
          else if (zi >= u(i) - tau_bnd_i) then
            near_upper = .true.
          end if
        end if

        if (near_lower) then
          mask_box(i) = FC_BOX_ACTIVE
          dzi         = safe_sub_dp(l(i), zi)
          dz(i)       = dzi
          n_active    = n_active + 1_i32
          if (abs(ai) > 0.0_dp .and. abs(dzi) > 0.0_dp) then
            if (abs(ai) <= SAFE_SQ_LIM .and. abs(dzi) <= SAFE_SQ_LIM) then
              call neumaier_add_safe(s_db, c_db, -ai * dzi)
            end if
          end if
        else if (near_upper) then
          mask_box(i) = FC_BOX_ACTIVE
          dzi         = safe_sub_dp(u(i), zi)
          dz(i)       = dzi
          n_active    = n_active + 1_i32
          if (abs(ai) > 0.0_dp .and. abs(dzi) > 0.0_dp) then
            if (abs(ai) <= SAFE_SQ_LIM .and. abs(dzi) <= SAFE_SQ_LIM) then
              call neumaier_add_safe(s_db, c_db, -ai * dzi)
            end if
          end if
        else
          mask_box(i) = FC_BOX_FREE
        end if
      end if
    end do

    delta_b = neumaier_finalize_safe(s_db, c_db)
  end subroutine fc_box_lin_partition

  !---------------------------------------------------------------------------
  ! Evaluate projected constrained trial step z_trial = P_C(z + tau * dz),
  ! realized chord <gz, z_trial - z>, and squared displacement ||z_trial - z||_2^2.
  !---------------------------------------------------------------------------
  pure subroutine fc_box_lin_trial_step(d, z, l, u, a, b, gz, dz, tau, &
      z_trial, chord_gz, step_norm_sq, status)
    integer(i32), intent(in)  :: d
    real(dp), intent(in)      :: z(*), l(*), u(*), a(*), gz(*), dz(*)
    real(dp), intent(in)      :: b, tau
    real(dp), intent(out)     :: z_trial(*)
    real(dp), intent(out)     :: chord_gz, step_norm_sq
    integer(i32), intent(out) :: status

    integer(i32) :: i
    real(dp)     :: zi, dzi, step_i, hs, nu_dummy, diff_i, gzi

    chord_gz     = 0.0_dp
    step_norm_sq = 0.0_dp
    status       = 0_i32

    if (d < 1_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    if (ieee_is_nan(tau) .or. .not. ieee_is_finite(tau)) then
      status = FC_ERR_INFEASIBLE_LIN
      return
    end if

    do i = 1_i32, d
      zi  = z(i)
      dzi = dz(i)
      if (ieee_is_nan(zi) .or. ieee_is_nan(dzi) .or. &
          .not. ieee_is_finite(zi) .or. .not. ieee_is_finite(dzi)) then
        status = FC_ERR_INFEASIBLE_LIN
        return
      end if

      if (.not. (l(i) < u(i))) then
        z_trial(i) = l(i)
      else if (.not. (abs(tau) > 0.0_dp) .or. .not. (abs(dzi) > 0.0_dp)) then
        z_trial(i) = zi
      else if (abs(zi) <= 0.45_dp * FC_HUGE .and. &
               abs(tau) <= 1.0e150_dp .and. abs(dzi) <= 1.0e150_dp) then
        z_trial(i) = zi + tau * dzi
      else
        if (abs(dzi) >= 1.0_dp .and. abs(tau) >= 0.45_dp * FC_HUGE / abs(dzi)) then
          step_i = sign(0.45_dp * FC_HUGE, tau) * sign(1.0_dp, dzi)
        else if (abs(tau) >= 1.0_dp .and. abs(dzi) >= 0.45_dp * FC_HUGE / abs(tau)) then
          step_i = sign(0.45_dp * FC_HUGE, tau) * sign(1.0_dp, dzi)
        else
          step_i = tau * dzi
        end if
        hs = 0.5_dp * zi + 0.5_dp * step_i
        if (hs >= 0.45_dp * FC_HUGE) then
          z_trial(i) = 0.9_dp * FC_HUGE
        else if (hs <= -0.45_dp * FC_HUGE) then
          z_trial(i) = -0.9_dp * FC_HUGE
        else
          z_trial(i) = 2.0_dp * hs
        end if
      end if
    end do

    call fc_box_lin_project(d, l, u, a, b, z_trial, nu_dummy, status)
    if (status /= 0_i32) return

    do i = 1_i32, d
      diff_i = safe_sub_dp(z_trial(i), z(i))
      if (abs(diff_i) > 0.0_dp) then
        gzi = gz(i)
        if (abs(gzi) <= SAFE_SQ_LIM .and. abs(diff_i) <= SAFE_SQ_LIM) then
          chord_gz = chord_gz + gzi * diff_i
        else if ((gzi > 0.0_dp .and. diff_i > 0.0_dp) .or. &
                 (gzi < 0.0_dp .and. diff_i < 0.0_dp)) then
          chord_gz = FC_HUGE
        else
          chord_gz = -FC_HUGE
        end if
        call accum_sq_safe(step_norm_sq, diff_i)
      end if
    end do
  end subroutine fc_box_lin_trial_step

end module fortchuck_box
