module fortchuck_box
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: &
    dp, i32, FC_HUGE, FC_ERR_INVALID_DIM, FC_ERR_INVALID_BOUNDS, &
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
  public :: fc_box_dir_deriv0
  public :: fc_box_trial_step

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

end module fortchuck_box
