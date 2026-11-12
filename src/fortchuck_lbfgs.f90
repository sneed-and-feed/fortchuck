module fortchuck_lbfgs
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan
  use fortchuck_kinds, only: &
    dp, i32, FC_HUGE, &
    FC_METRIC_EUCLIDEAN, FC_DAMP_POWELL, &
    FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED
  use fortchuck_linalg, only: fc_frob_inner, fc_frob_norm
  use fortchuck_box, only: fc_box_active_step
  use fortchuck_stiefel, only: &
    fc_stiefel_proj_tangent, fc_cayley_transport_vec
  implicit none (type, external)
  private

  real(dp), parameter :: GAMMA_MIN   = 1.0e-10_dp
  real(dp), parameter :: GAMMA_MAX   = 1.0e+10_dp
  real(dp), parameter :: SAFE_SQ_LIM = 1.0e+153_dp

  public :: fc_lbfgs_direction
  public :: fc_abb_direction
  public :: fc_lbfgs_transport_and_push

contains

  !---------------------------------------------------------------------------
  ! Overflow-safe clipped positive ratio clip(num / den, lo, hi) for num, den > 0.
  !---------------------------------------------------------------------------
  pure function safe_clipped_ratio(num, den, lo, hi) result(val)
    real(dp), intent(in) :: num, den, lo, hi
    real(dp)             :: val

    if (.not. (den > 0.0_dp) .or. .not. (num > 0.0_dp)) then
      val = lo
    else if (.not. ieee_is_finite(num) .or. .not. ieee_is_finite(den)) then
      val = lo
    else if (den >= 1.0_dp) then
      if (num / den <= lo) then
        val = lo
      else if (num / den >= hi) then
        val = hi
      else
        val = num / den
      end if
    else
      if (num >= hi * den) then
        val = hi
      else
        val = max(lo, num / den)
      end if
    end if
  end function safe_clipped_ratio

  !---------------------------------------------------------------------------
  ! Overflow-safe difference a - b avoiding FE_OVERFLOW near +-FC_HUGE.
  !---------------------------------------------------------------------------
  pure function safe_diff(a, b) result(dval)
    real(dp), intent(in) :: a, b
    real(dp)             :: dval
    real(dp)             :: hd

    if (abs(a) <= 0.45_dp * FC_HUGE .and. abs(b) <= 0.45_dp * FC_HUGE) then
      dval = a - b
    else
      hd = 0.5_dp * a - 0.5_dp * b
      if (hd >= 0.5_dp * FC_HUGE) then
        dval = FC_HUGE
      else if (hd <= -0.5_dp * FC_HUGE) then
        dval = -FC_HUGE
      else
        dval = a - b
      end if
    end if
  end function safe_diff

  !---------------------------------------------------------------------------
  ! Overflow-safe scaled accumulation for x^2 and x*y over box free set.
  !---------------------------------------------------------------------------
  pure subroutine accum_pair_stats(n_len, mask_box, x_vec, y_vec, s_xx, s_xy, s_yy)
    integer(i32), intent(in) :: n_len
    integer(i32), intent(in) :: mask_box(*)
    real(dp), intent(in)     :: x_vec(*), y_vec(*)
    real(dp), intent(out)    :: s_xx, s_xy, s_yy

    integer(i32) :: i
    real(dp)     :: xi, yi, term

    s_xx = 0.0_dp
    s_xy = 0.0_dp
    s_yy = 0.0_dp

    do i = 1_i32, n_len
      if (mask_box(i) == FC_BOX_FREE) then
        xi = x_vec(i)
        yi = y_vec(i)
        if (abs(xi) <= SAFE_SQ_LIM .and. abs(yi) <= SAFE_SQ_LIM) then
          term = xi * xi
          if (0.5_dp * s_xx + 0.5_dp * term >= 0.5_dp * FC_HUGE) then
            s_xx = FC_HUGE
          else
            s_xx = s_xx + term
          end if

          term = yi * yi
          if (0.5_dp * s_yy + 0.5_dp * term >= 0.5_dp * FC_HUGE) then
            s_yy = FC_HUGE
          else
            s_yy = s_yy + term
          end if

          term = xi * yi
          if (abs(0.5_dp * s_xy + 0.5_dp * term) >= 0.5_dp * FC_HUGE) then
            s_xy = sign(FC_HUGE, term)
          else
            s_xy = s_xy + term
          end if
        else
          if (abs(xi) > SAFE_SQ_LIM) s_xx = FC_HUGE
          if (abs(yi) > SAFE_SQ_LIM) s_yy = FC_HUGE
          s_xy = 0.0_dp
        end if
      end if
    end do
  end subroutine accum_pair_stats

  !---------------------------------------------------------------------------
  ! Overflow-safe Stiefel metric norm ||U||_{X, metric} given U(n,p) and
  ! A_u = X^T * U in R^{p x p}.
  !---------------------------------------------------------------------------
  pure function stiefel_metric_norm_from_skew(n, p, U, A_u, metric) result(nrm)
    integer(i32), intent(in) :: n, p, metric
    real(dp), intent(in)     :: U(n, p), A_u(p, p)
    real(dp)                 :: nrm

    real(dp) :: nrm_f, nrm_a, q_rat

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
  end function stiefel_metric_norm_from_skew

  !---------------------------------------------------------------------------
  ! Joint Product-Space Masked L-BFGS Two-Loop Direction:
  !   1. Dynamic free-subspace curvature filter (FM-2 & FM-3 guard)
  !   2. GEMM-free backward, block-BB2 center, and forward product loops
  !   3. Safeguarded secant active-band box step via fc_box_active_step
  !---------------------------------------------------------------------------
  pure subroutine fc_lbfgs_direction( &
      d, R, n_st_tot, n_skew_tot, metric, &
      m, m_cur, n_tot_pad, n_skew_pad, &
      x_off, skew_off, &
      z, l, u, gz, mask_box, &
      rgrad_X, Aq_buf, &
      gamma_blk, has_prev, s_prev, y_prev, &
      s_hist, y_hist, As_hist, Ay_hist, &
      st_sy, st_ss, st_yy, rho_m, alpha_m, &
      active_pair_flag, ring_order, &
      q_buf, dz, eta_X)
    integer(i32), intent(in)    :: d, R, n_st_tot, n_skew_tot, metric
    integer(i32), intent(in)    :: m, m_cur, n_tot_pad, n_skew_pad
    integer(i32), intent(in)    :: x_off(*), skew_off(*)
    real(dp), intent(in)        :: z(*), l(*), u(*), gz(*)
    integer(i32), intent(in)    :: mask_box(*)
    real(dp), intent(in)        :: rgrad_X(*)
    real(dp), intent(inout)     :: Aq_buf(*)
    real(dp), intent(in)        :: gamma_blk(*)
    logical, intent(in)         :: has_prev
    real(dp), intent(in)        :: s_prev(*), y_prev(*)
    real(dp), intent(in)        :: s_hist(n_tot_pad, m), y_hist(n_tot_pad, m)
    real(dp), intent(in)        :: As_hist(n_skew_pad, m), Ay_hist(n_skew_pad, m)
    real(dp), intent(in)        :: st_sy(*), st_ss(*), st_yy(*)
    real(dp), intent(inout)     :: rho_m(*), alpha_m(*)
    integer(i32), intent(inout) :: active_pair_flag(*)
    integer(i32), intent(in)    :: ring_order(*)
    real(dp), intent(inout)     :: q_buf(*)
    real(dp), intent(out)       :: dz(*), eta_X(*)

    integer(i32) :: ord, slot, i, idx, r_idx, x_s, x_e, sk_s, sk_e
    real(dp)     :: one_minus_beta, box_ss, box_sy, box_yy
    real(dp)     :: nu_s, nu_y, sig_tilde, s_nrm, y_nrm
    real(dp)     :: dot_sq, frob_sq, frob_asaq, alpha_j
    real(dp)     :: dot_yq, frob_yq, frob_ayaq, beta_loop, diff_ab
    real(dp)     :: gam_box, gam_r

    if (metric == FC_METRIC_EUCLIDEAN) then
      one_minus_beta = 0.0_dp
    else
      one_minus_beta = 0.5_dp
    end if

    ! Step 1: Dynamic free-subspace curvature filter across stored slots
    do ord = 1_i32, m_cur
      slot = ring_order(ord)
      if (d > 0_i32) then
        call accum_pair_stats(d, mask_box, s_hist(1, slot), y_hist(1, slot), &
          box_ss, box_sy, box_yy)
      else
        box_ss = 0.0_dp
        box_sy = 0.0_dp
        box_yy = 0.0_dp
      end if

      if (R > 0_i32) then
        nu_s      = max(0.0_dp, box_ss + st_ss(slot))
        nu_y      = max(0.0_dp, box_yy + st_yy(slot))
        sig_tilde = box_sy + st_sy(slot)
      else
        nu_s      = max(0.0_dp, box_ss)
        nu_y      = max(0.0_dp, box_yy)
        sig_tilde = box_sy
      end if

      if (nu_s > 1.0e-60_dp .and. nu_y > 1.0e-60_dp .and. &
          ieee_is_finite(nu_s) .and. ieee_is_finite(nu_y) .and. &
          ieee_is_finite(sig_tilde) .and. sig_tilde > 1.0e-60_dp) then
        s_nrm = sqrt(nu_s)
        y_nrm = sqrt(nu_y)
        if (sig_tilde >= 1.0e-10_dp * s_nrm * y_nrm) then
          active_pair_flag(slot) = 1_i32
          rho_m(slot)            = 1.0_dp / sig_tilde
        else
          active_pair_flag(slot) = 0_i32
          rho_m(slot)            = 0.0_dp
        end if
      else
        active_pair_flag(slot) = 0_i32
        rho_m(slot)            = 0.0_dp
      end if
      alpha_m(slot) = 0.0_dp
    end do

    ! Step 2a: Initialize free-subspace accumulator q_buf (Aq_buf already holds X^T * rgrad)
    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FREE) then
        q_buf(i) = gz(i)
      else
        q_buf(i) = 0.0_dp
      end if
    end do
    do idx = 1_i32, n_st_tot
      q_buf(d + idx) = rgrad_X(idx)
    end do

    ! Step 2b: Backward loop (newest ord = m_cur down to oldest ord = 1)
    do ord = m_cur, 1_i32, -1_i32
      slot = ring_order(ord)
      if (active_pair_flag(slot) == 1_i32) then
        dot_sq = 0.0_dp
        do i = 1_i32, d
          if (mask_box(i) == FC_BOX_FREE) then
            dot_sq = dot_sq + s_hist(i, slot) * q_buf(i)
          end if
        end do

        if (R > 0_i32) then
          frob_sq = 0.0_dp
          do idx = 1_i32, n_st_tot
            frob_sq = frob_sq + s_hist(d + idx, slot) * q_buf(d + idx)
          end do
          if (one_minus_beta > 0.0_dp) then
            frob_asaq = 0.0_dp
            do idx = 1_i32, n_skew_tot
              frob_asaq = frob_asaq + As_hist(idx, slot) * Aq_buf(idx)
            end do
            dot_sq = dot_sq + (frob_sq - one_minus_beta * frob_asaq)
          else
            dot_sq = dot_sq + frob_sq
          end if
        end if

        alpha_j       = rho_m(slot) * dot_sq
        alpha_m(slot) = alpha_j

        do i = 1_i32, d
          if (mask_box(i) == FC_BOX_FREE) then
            q_buf(i) = q_buf(i) - alpha_j * y_hist(i, slot)
          end if
        end do
        do idx = 1_i32, n_st_tot
          q_buf(d + idx) = q_buf(d + idx) - alpha_j * y_hist(d + idx, slot)
        end do
        if (R > 0_i32 .and. one_minus_beta > 0.0_dp) then
          do idx = 1_i32, n_skew_tot
            Aq_buf(idx) = Aq_buf(idx) - alpha_j * Ay_hist(idx, slot)
          end do
        end if
      end if
    end do

    ! Step 2c: Cosine-guarded persistent per-block BB2 center scaling
    gam_box = min(max(gamma_blk(1), GAMMA_MIN), GAMMA_MAX)
    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FREE) then
        if (abs(q_buf(i)) > 1.0_dp .and. gam_box >= 0.5_dp * FC_HUGE / abs(q_buf(i))) then
          q_buf(i) = sign(0.5_dp * FC_HUGE, q_buf(i))
        else
          q_buf(i) = gam_box * q_buf(i)
        end if
      end if
    end do

    do r_idx = 1_i32, R
      gam_r = min(max(gamma_blk(r_idx + 1_i32), GAMMA_MIN), GAMMA_MAX)
      x_s   = x_off(r_idx)
      x_e   = x_off(r_idx + 1_i32) - 1_i32
      sk_s  = skew_off(r_idx)
      sk_e  = skew_off(r_idx + 1_i32) - 1_i32
      do idx = x_s, x_e
        q_buf(d + idx) = gam_r * q_buf(d + idx)
      end do
      if (one_minus_beta > 0.0_dp) then
        do idx = sk_s, sk_e
          Aq_buf(idx) = gam_r * Aq_buf(idx)
        end do
      end if
    end do

    ! Step 2d: Forward loop (oldest ord = 1 up to newest ord = m_cur)
    do ord = 1_i32, m_cur
      slot = ring_order(ord)
      if (active_pair_flag(slot) == 1_i32) then
        dot_yq = 0.0_dp
        do i = 1_i32, d
          if (mask_box(i) == FC_BOX_FREE) then
            dot_yq = dot_yq + y_hist(i, slot) * q_buf(i)
          end if
        end do

        if (R > 0_i32) then
          frob_yq = 0.0_dp
          do idx = 1_i32, n_st_tot
            frob_yq = frob_yq + y_hist(d + idx, slot) * q_buf(d + idx)
          end do
          if (one_minus_beta > 0.0_dp) then
            frob_ayaq = 0.0_dp
            do idx = 1_i32, n_skew_tot
              frob_ayaq = frob_ayaq + Ay_hist(idx, slot) * Aq_buf(idx)
            end do
            dot_yq = dot_yq + (frob_yq - one_minus_beta * frob_ayaq)
          else
            dot_yq = dot_yq + frob_yq
          end if
        end if

        beta_loop = rho_m(slot) * dot_yq
        diff_ab   = alpha_m(slot) - beta_loop

        do i = 1_i32, d
          if (mask_box(i) == FC_BOX_FREE) then
            q_buf(i) = q_buf(i) + diff_ab * s_hist(i, slot)
          end if
        end do
        do idx = 1_i32, n_st_tot
          q_buf(d + idx) = q_buf(d + idx) + diff_ab * s_hist(d + idx, slot)
        end do
        if (R > 0_i32 .and. one_minus_beta > 0.0_dp) then
          do idx = 1_i32, n_skew_tot
            Aq_buf(idx) = Aq_buf(idx) + diff_ab * As_hist(idx, slot)
          end do
        end if
      end if
    end do

    ! Step 2e: Negate free-subspace direction and populate active/fixed box entries
    do i = 1_i32, d
      if (mask_box(i) == FC_BOX_FREE) then
        dz(i) = -q_buf(i)
      else
        dz(i) = 0.0_dp
      end if
    end do
    if (d > 0_i32) then
      call fc_box_active_step(d, z, l, u, gz, mask_box, gam_box, &
        has_prev, s_prev, y_prev, dz)
    end if

    do idx = 1_i32, n_st_tot
      eta_X(idx) = -q_buf(d + idx)
    end do
  end subroutine fc_lbfgs_direction

  !---------------------------------------------------------------------------
  ! Safeguarded Per-Block Alternating Barzilai-Borwein (ABB) Steepest Descent:
  !   Odd k  -> BB1: sigma_ss / |sigma_sy|
  !   Even k -> BB2: |sigma_sy| / sigma_yy
  !   Fallback / k = 0 -> 1 / max(1, ||g_blk||), clipped to [1e-10, 1e10].
  !---------------------------------------------------------------------------
  pure subroutine fc_abb_direction( &
      k_iter, has_prev, d, R, n_arr, p_arr, metric, &
      x_off, skew_off, &
      z, l, u, gz, mask_box, &
      rgrad_X, Aq_fresh, &
      abb_ss, abb_sy, abb_yy, &
      dz, eta_X)
    integer(i32), intent(in) :: k_iter
    logical, intent(in)      :: has_prev
    integer(i32), intent(in) :: d, R, metric
    integer(i32), intent(in) :: n_arr(*), p_arr(*)
    integer(i32), intent(in) :: x_off(*), skew_off(*)
    real(dp), intent(in)     :: z(*), l(*), u(*), gz(*)
    integer(i32), intent(in) :: mask_box(*)
    real(dp), intent(in)     :: rgrad_X(*), Aq_fresh(*)
    real(dp), intent(in)     :: abb_ss(*), abb_sy(*), abb_yy(*)
    real(dp), intent(out)    :: dz(*), eta_X(*)

    integer(i32) :: i, r_idx, nr, pr, idx, x_s, x_e, sk_s
    real(dp)     :: scale, ssq, agz, ratio, g_box_nrm, alpha_box, gzi
    real(dp)     :: g_st_nrm, alpha_r, ss_b, sy_abs, yy_b
    logical      :: use_bb1

    use_bb1 = (mod(max(0_i32, k_iter), 2_i32) == 1_i32)

    ! 1. Box block (b = 1)
    if (d > 0_i32) then
      scale = 0.0_dp
      ssq   = 1.0_dp
      do i = 1_i32, d
        if (mask_box(i) /= FC_BOX_FIXED .and. (l(i) < u(i))) then
          if (.not. ((z(i) <= l(i) .and. gz(i) >= 0.0_dp) .or. &
                     (z(i) >= u(i) .and. gz(i) <= 0.0_dp))) then
            agz = abs(gz(i))
            if (agz > 0.0_dp .and. ieee_is_finite(agz)) then
              if (scale < agz) then
                ratio = scale / agz
                ssq   = 1.0_dp + ssq * (ratio * ratio)
                scale = agz
              else
                ratio = agz / scale
                ssq   = ssq + ratio * ratio
              end if
            end if
          end if
        end if
      end do
      g_box_nrm = scale * sqrt(ssq)

      ss_b   = abb_ss(1)
      sy_abs = abs(abb_sy(1))
      yy_b   = abb_yy(1)
      if (has_prev .and. k_iter > 0_i32 .and. ss_b > 1.0e-30_dp .and. sy_abs > 1.0e-30_dp) then
        if (use_bb1) then
          alpha_box = safe_clipped_ratio(ss_b, sy_abs, GAMMA_MIN, GAMMA_MAX)
        else if (yy_b > 1.0e-30_dp) then
          alpha_box = safe_clipped_ratio(sy_abs, yy_b, GAMMA_MIN, GAMMA_MAX)
        else
          alpha_box = safe_clipped_ratio(1.0_dp, max(1.0_dp, g_box_nrm), GAMMA_MIN, GAMMA_MAX)
        end if
      else
        alpha_box = safe_clipped_ratio(1.0_dp, max(1.0_dp, g_box_nrm), GAMMA_MIN, GAMMA_MAX)
      end if

      do i = 1_i32, d
        if (mask_box(i) == FC_BOX_FIXED .or. .not. (l(i) < u(i))) then
          dz(i) = 0.0_dp
        else
          gzi = gz(i)
          if (abs(gzi) > 1.0_dp .and. alpha_box >= 0.5_dp * FC_HUGE / abs(gzi)) then
            dz(i) = -sign(0.5_dp * FC_HUGE, gzi)
          else
            dz(i) = -alpha_box * gzi
          end if
        end if
      end do
    end if

    ! 2. Stiefel blocks (r = 1 .. R, b = r + 1)
    do r_idx = 1_i32, R
      nr   = n_arr(r_idx)
      pr   = p_arr(r_idx)
      x_s  = x_off(r_idx)
      x_e  = x_off(r_idx + 1_i32) - 1_i32
      sk_s = skew_off(r_idx)

      g_st_nrm = stiefel_metric_norm_from_skew(nr, pr, &
        rgrad_X(x_s:x_e), Aq_fresh(sk_s:sk_s + pr * pr - 1_i32), metric)

      ss_b   = abb_ss(r_idx + 1_i32)
      sy_abs = abs(abb_sy(r_idx + 1_i32))
      yy_b   = abb_yy(r_idx + 1_i32)
      if (has_prev .and. k_iter > 0_i32 .and. ss_b > 1.0e-30_dp .and. sy_abs > 1.0e-30_dp) then
        if (use_bb1) then
          alpha_r = safe_clipped_ratio(ss_b, sy_abs, GAMMA_MIN, GAMMA_MAX)
        else if (yy_b > 1.0e-30_dp) then
          alpha_r = safe_clipped_ratio(sy_abs, yy_b, GAMMA_MIN, GAMMA_MAX)
        else
          alpha_r = safe_clipped_ratio(1.0_dp, max(1.0_dp, g_st_nrm), GAMMA_MIN, GAMMA_MAX)
        end if
      else
        alpha_r = safe_clipped_ratio(1.0_dp, max(1.0_dp, g_st_nrm), GAMMA_MIN, GAMMA_MAX)
      end if

      do idx = x_s, x_e
        eta_X(idx) = -alpha_r * rgrad_X(idx)
      end do
    end do
  end subroutine fc_abb_direction

  !---------------------------------------------------------------------------
  ! Post-Step Isometric Cayley Transport, Block-BB2 Scale Update, and
  ! Cautious / Powell Curvature-Guarded Ring-Buffer Insertion.
  !---------------------------------------------------------------------------
  pure subroutine fc_lbfgs_transport_and_push( &
      d, R, n_arr, p_arr, n_st_tot, metric, damping_mode, &
      m, m_cur, ring_head, n_tot_pad, n_skew_pad, &
      x_off, skew_off, cay_off, ipiv_off, &
      l, u, z_old, z_new, gz_old, gz_new, mask_box_new, &
      X_old, X_new, eta_X, tau, cayley_persist, ipiv_all, &
      rgrad_old, rgrad_new, &
      kkt_old, polished_step, max_drift_step, &
      gamma_blk, abb_ss, abb_sy, abb_yy, &
      s_cur, y_cur, &
      s_hist, y_hist, As_hist, Ay_hist, &
      st_sy, st_ss, st_yy, ring_order, scratch)
    integer(i32), intent(in)    :: d, R, n_st_tot, metric, damping_mode
    integer(i32), intent(in)    :: n_arr(*), p_arr(*)
    integer(i32), intent(in)    :: m, n_tot_pad, n_skew_pad
    integer(i32), intent(inout) :: m_cur, ring_head
    integer(i32), intent(in)    :: x_off(*), skew_off(*), cay_off(*), ipiv_off(*)
    real(dp), intent(in)        :: l(*), u(*), z_old(*), z_new(*), gz_old(*), gz_new(*)
    integer(i32), intent(in)    :: mask_box_new(*)
    real(dp), intent(in)        :: X_old(*), X_new(*), eta_X(*)
    real(dp), intent(in)        :: tau, cayley_persist(*)
    integer(i32), intent(in)    :: ipiv_all(*)
    real(dp), intent(inout)     :: rgrad_old(*)
    real(dp), intent(in)        :: rgrad_new(*)
    real(dp), intent(in)        :: kkt_old, max_drift_step
    logical, intent(in)         :: polished_step
    real(dp), intent(inout)     :: gamma_blk(*), abb_ss(*), abb_sy(*), abb_yy(*)
    real(dp), intent(out)       :: s_cur(*), y_cur(*)
    real(dp), intent(inout)     :: s_hist(n_tot_pad, m), y_hist(n_tot_pad, m)
    real(dp), intent(inout)     :: As_hist(n_skew_pad, m), Ay_hist(n_skew_pad, m)
    real(dp), intent(inout)     :: st_sy(*), st_ss(*), st_yy(*)
    integer(i32), intent(inout) :: ring_order(*)
    real(dp), intent(inout)     :: scratch(*)

    integer(i32) :: ord, slot, i, idx, r_idx, nr, pr, p2
    integer(i32) :: x_s, x_e, sk_s, sk_e, c_s, p_s
    real(dp)     :: one_minus_beta, acc_sy, acc_ss, acc_yy
    real(dp)     :: f_ss, f_sy, f_yy, a_ss, a_sy, a_yy
    real(dp)     :: box_free_ss, box_free_sy, box_free_yy
    real(dp)     :: int_ss, int_sy, int_yy
    real(dp)     :: st_cur_ss, st_cur_sy, st_cur_yy
    real(dp)     :: clamped_box_ss, st_gnorm_sq, rg_frob_sq
    real(dp)     :: sig_ss, sig_sy, sig_yy
    real(dp)     :: tot_free_ss, tot_free_sy, tot_free_yy
    real(dp)     :: norm_s_free, norm_y_free, thresh_cautious
    real(dp)     :: mu_k, omega_p, theta_k, inv_gam
    logical      :: do_push, did_damp, clamp_impulse

    if (metric == FC_METRIC_EUCLIDEAN) then
      one_minus_beta = 0.0_dp
    else
      one_minus_beta = 0.5_dp
    end if

    ! 1. Flush ring buffer on extreme pre-polish manifold drift (> 1e-4)
    if (max_drift_step > 1.0e-4_dp) then
      m_cur     = 0_i32
      ring_head = 0_i32
    end if

    ! 2. Transport stored Stiefel history slots 1..m_cur to T_{X_new}
    if (R > 0_i32 .and. m > 0_i32 .and. m_cur > 0_i32) then
      do ord = 1_i32, m_cur
        slot = ring_order(ord)
        acc_sy = 0.0_dp
        acc_ss = 0.0_dp
        acc_yy = 0.0_dp

        do r_idx = 1_i32, R
          nr   = n_arr(r_idx)
          pr   = p_arr(r_idx)
          p2   = pr * pr
          x_s  = x_off(r_idx)
          x_e  = x_off(r_idx + 1_i32) - 1_i32
          sk_s = skew_off(r_idx)
          sk_e = skew_off(r_idx + 1_i32) - 1_i32
          c_s  = cay_off(r_idx)
          p_s  = ipiv_off(r_idx)

          call fc_cayley_transport_vec(nr, pr, X_old(x_s:x_e), eta_X(x_s:x_e), tau, &
            cayley_persist(c_s), ipiv_all(p_s), s_hist(d + x_s:d + x_e, slot), scratch)
          call fc_cayley_transport_vec(nr, pr, X_old(x_s:x_e), eta_X(x_s:x_e), tau, &
            cayley_persist(c_s), ipiv_all(p_s), y_hist(d + x_s:d + x_e, slot), scratch)

          if (polished_step) then
            call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
              s_hist(d + x_s:d + x_e, slot), As_hist(sk_s:sk_e, slot), scratch(1:p2))
            call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
              y_hist(d + x_s:d + x_e, slot), Ay_hist(sk_s:sk_e, slot), scratch(1:p2))

            f_ss = fc_frob_inner(nr, pr, s_hist(d + x_s:d + x_e, slot), &
              s_hist(d + x_s:d + x_e, slot))
            f_sy = fc_frob_inner(nr, pr, s_hist(d + x_s:d + x_e, slot), &
              y_hist(d + x_s:d + x_e, slot))
            f_yy = fc_frob_inner(nr, pr, y_hist(d + x_s:d + x_e, slot), &
              y_hist(d + x_s:d + x_e, slot))
            if (one_minus_beta > 0.0_dp) then
              a_ss = fc_frob_inner(pr, pr, As_hist(sk_s:sk_e, slot), As_hist(sk_s:sk_e, slot))
              a_sy = fc_frob_inner(pr, pr, As_hist(sk_s:sk_e, slot), Ay_hist(sk_s:sk_e, slot))
              a_yy = fc_frob_inner(pr, pr, Ay_hist(sk_s:sk_e, slot), Ay_hist(sk_s:sk_e, slot))
              acc_ss = acc_ss + max(0.0_dp, f_ss - one_minus_beta * a_ss)
              acc_sy = acc_sy + (f_sy - one_minus_beta * a_sy)
              acc_yy = acc_yy + max(0.0_dp, f_yy - one_minus_beta * a_yy)
            else
              acc_ss = acc_ss + max(0.0_dp, f_ss)
              acc_sy = acc_sy + f_sy
              acc_yy = acc_yy + max(0.0_dp, f_yy)
            end if
          end if
        end do

        if (polished_step) then
          st_ss(slot) = acc_ss
          st_sy(slot) = acc_sy
          st_yy(slot) = acc_yy
        end if
      end do
    end if

    ! 3. Form current step pair (s_cur, y_cur) in T_{p_{k+1}}
    clamped_box_ss = 0.0_dp
    do i = 1_i32, d
      s_cur(i) = safe_diff(z_new(i), z_old(i))
      y_cur(i) = safe_diff(gz_new(i), gz_old(i))
      if (mask_box_new(i) == FC_BOX_ACTIVE .and. abs(s_cur(i)) <= SAFE_SQ_LIM) then
        if (0.5_dp * clamped_box_ss + 0.5_dp * (s_cur(i) * s_cur(i)) < 0.5_dp * FC_HUGE) then
          clamped_box_ss = clamped_box_ss + s_cur(i) * s_cur(i)
        else
          clamped_box_ss = FC_HUGE
        end if
      end if
    end do

    if (d > 0_i32) then
      call accum_pair_stats(d, mask_box_new, s_cur, y_cur, &
        box_free_ss, box_free_sy, box_free_yy)
      if (box_free_ss > 1.0e-60_dp) then
        abb_ss(1) = box_free_ss
        abb_sy(1) = box_free_sy
        abb_yy(1) = box_free_yy
      else
        int_ss = 0.0_dp
        int_sy = 0.0_dp
        int_yy = 0.0_dp
        do i = 1_i32, d
          if (z_new(i) > l(i) .and. z_new(i) < u(i) .and. &
              abs(s_cur(i)) <= SAFE_SQ_LIM .and. abs(y_cur(i)) <= SAFE_SQ_LIM) then
            int_ss = int_ss + s_cur(i) * s_cur(i)
            int_sy = int_sy + s_cur(i) * y_cur(i)
            int_yy = int_yy + y_cur(i) * y_cur(i)
          end if
        end do
        abb_ss(1) = int_ss
        abb_sy(1) = int_sy
        abb_yy(1) = int_yy
      end if
    else
      box_free_ss = 0.0_dp
      box_free_sy = 0.0_dp
      box_free_yy = 0.0_dp
      abb_ss(1)   = 0.0_dp
      abb_sy(1)   = 0.0_dp
      abb_yy(1)   = 0.0_dp
    end if

    st_cur_ss   = 0.0_dp
    st_cur_sy   = 0.0_dp
    st_cur_yy   = 0.0_dp
    st_gnorm_sq = 0.0_dp

    do r_idx = 1_i32, R
      nr  = n_arr(r_idx)
      pr  = p_arr(r_idx)
      p2  = pr * pr
      x_s = x_off(r_idx)
      x_e = x_off(r_idx + 1_i32) - 1_i32
      c_s = cay_off(r_idx)
      p_s = ipiv_off(r_idx)

      do idx = x_s, x_e
        s_cur(d + idx) = tau * eta_X(idx)
      end do
      call fc_cayley_transport_vec(nr, pr, X_old(x_s:x_e), eta_X(x_s:x_e), tau, &
        cayley_persist(c_s), ipiv_all(p_s), s_cur(d + x_s:d + x_e), scratch)

      call fc_cayley_transport_vec(nr, pr, X_old(x_s:x_e), eta_X(x_s:x_e), tau, &
        cayley_persist(c_s), ipiv_all(p_s), rgrad_old(x_s:x_e), scratch)
      do idx = x_s, x_e
        y_cur(d + idx) = safe_diff(rgrad_new(idx), rgrad_old(idx))
      end do

      ! Project onto T_{X_new, r} and compute exact Stiefel block secant scalars
      ! scratch(1:p2) = As_r, scratch(p2+1:2*p2) = Ay_r, scratch(2*p2+1:3*p2) = scr_p2
      call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
        s_cur(d + x_s:d + x_e), scratch(1:p2), scratch(2_i32 * p2 + 1_i32:3_i32 * p2))
      call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
        y_cur(d + x_s:d + x_e), scratch(p2 + 1_i32:2_i32 * p2), &
        scratch(2_i32 * p2 + 1_i32:3_i32 * p2))

      f_ss = fc_frob_inner(nr, pr, s_cur(d + x_s:d + x_e), s_cur(d + x_s:d + x_e))
      f_sy = fc_frob_inner(nr, pr, s_cur(d + x_s:d + x_e), y_cur(d + x_s:d + x_e))
      f_yy = fc_frob_inner(nr, pr, y_cur(d + x_s:d + x_e), y_cur(d + x_s:d + x_e))
      if (one_minus_beta > 0.0_dp) then
        a_ss   = fc_frob_inner(pr, pr, scratch(1:p2), scratch(1:p2))
        a_sy   = fc_frob_inner(pr, pr, scratch(1:p2), scratch(p2 + 1_i32:2_i32 * p2))
        a_yy   = fc_frob_inner(pr, pr, scratch(p2 + 1_i32:2_i32 * p2), &
                   scratch(p2 + 1_i32:2_i32 * p2))
        sig_ss = max(0.0_dp, f_ss - one_minus_beta * a_ss)
        sig_sy = f_sy - one_minus_beta * a_sy
        sig_yy = max(0.0_dp, f_yy - one_minus_beta * a_yy)
      else
        sig_ss = max(0.0_dp, f_ss)
        sig_sy = f_sy
        sig_yy = max(0.0_dp, f_yy)
      end if

      abb_ss(r_idx + 1_i32) = sig_ss
      abb_sy(r_idx + 1_i32) = sig_sy
      abb_yy(r_idx + 1_i32) = sig_yy

      rg_frob_sq = fc_frob_inner(nr, pr, rgrad_new(x_s:x_e), rgrad_new(x_s:x_e))
      if (ieee_is_finite(rg_frob_sq)) st_gnorm_sq = st_gnorm_sq + max(0.0_dp, rg_frob_sq)

      st_cur_ss = st_cur_ss + sig_ss
      st_cur_sy = st_cur_sy + sig_sy
      st_cur_yy = st_cur_yy + sig_yy
    end do

    ! Guard against cross-Hessian clamping impulse: when box variables actively
    ! slam into bounds on step k and pre-clamp cross-gradient dominates y_X by >10x,
    ! skip polluting Stiefel BB2 / ring-buffer curvature with the pre-clamp impulse.
    clamp_impulse = (d > 0_i32 .and. R > 0_i32 .and. &
      clamped_box_ss > 1.0e-16_dp * max(1.0e-30_dp, st_cur_ss) .and. &
      st_cur_yy > 100.0_dp * max(1.0e-30_dp, st_gnorm_sq))
    if (clamp_impulse) then
      do r_idx = 1_i32, R
        abb_ss(r_idx + 1_i32) = 0.0_dp
        abb_sy(r_idx + 1_i32) = 0.0_dp
        abb_yy(r_idx + 1_i32) = 0.0_dp
      end do
    end if

    ! 4. Update persistent cosine-guarded per-block BB2 scales gamma_blk(1:R+1)
    do r_idx = 1_i32, R + 1_i32
      sig_ss = abb_ss(r_idx)
      sig_sy = abb_sy(r_idx)
      sig_yy = abb_yy(r_idx)
      if (sig_yy > 1.0e-30_dp .and. sig_ss > 1.0e-60_dp .and. &
          ieee_is_finite(sig_ss) .and. ieee_is_finite(sig_sy) .and. ieee_is_finite(sig_yy)) then
        if (sig_sy >= 1.0e-3_dp * (sqrt(sig_ss) * sqrt(sig_yy))) then
          gamma_blk(r_idx) = safe_clipped_ratio(sig_sy, sig_yy, GAMMA_MIN, GAMMA_MAX)
        end if
      end if
    end do

    if (m <= 0_i32 .or. clamp_impulse) return

    ! 5. Curvature positivity check / damping on updated free subspace at p_{k+1}
    tot_free_ss = max(0.0_dp, box_free_ss + st_cur_ss)
    tot_free_sy = box_free_sy + st_cur_sy
    tot_free_yy = max(0.0_dp, box_free_yy + st_cur_yy)

    if (.not. ieee_is_finite(tot_free_ss) .or. .not. ieee_is_finite(tot_free_sy) .or. &
        .not. ieee_is_finite(tot_free_yy)) return

    norm_s_free = sqrt(tot_free_ss)
    norm_y_free = sqrt(tot_free_yy)
    if (.not. (norm_s_free > 1.0e-30_dp)) return

    do_push  = .false.
    did_damp = .false.
    theta_k  = 1.0_dp

    if (damping_mode == FC_DAMP_POWELL) then
      omega_p = 0.2_dp
      mu_k    = (1.0_dp / min(max(gamma_blk(1), GAMMA_MIN), GAMMA_MAX)) * box_free_ss
      do r_idx = 1_i32, R
        inv_gam = 1.0_dp / min(max(gamma_blk(r_idx + 1_i32), GAMMA_MIN), GAMMA_MAX)
        mu_k    = mu_k + inv_gam * abb_ss(r_idx + 1_i32)
      end do

      if (mu_k > 1.0e-60_dp .and. ieee_is_finite(mu_k)) then
        do_push = .true.
        if (tot_free_sy < omega_p * mu_k) then
          theta_k  = ((1.0_dp - omega_p) * mu_k) / (mu_k - tot_free_sy)
          theta_k  = min(max(theta_k, 0.0_dp), 1.0_dp)
          did_damp = .true.
        end if
      end if
    else
      ! Default FC_DAMP_CAUTIOUS (Huang-Absil-Gallivan 2018)
      thresh_cautious = max( &
        1.0e-8_dp * tot_free_ss * min(1.0_dp, max(0.0_dp, kkt_old)), &
        1.0e-12_dp * norm_s_free * norm_y_free)
      if (tot_free_sy >= thresh_cautious .and. tot_free_sy > 1.0e-60_dp) then
        do_push = .true.
      end if
    end if

    if (.not. do_push) return

    ! 6. Insert (s_k, y_k) into circular ring buffer and cache Stiefel invariants
    slot      = mod(ring_head, m) + 1_i32
    ring_head = slot
    if (m_cur < m) m_cur = m_cur + 1_i32

    if (m_cur < m) then
      do ord = 1_i32, m_cur
        ring_order(ord) = ord
      end do
    else
      do ord = 1_i32, m
        ring_order(ord) = mod(ring_head + ord - 1_i32, m) + 1_i32
      end do
    end if

    inv_gam = 1.0_dp / min(max(gamma_blk(1), GAMMA_MIN), GAMMA_MAX)
    do i = 1_i32, d
      s_hist(i, slot) = s_cur(i)
      if (did_damp) then
        y_hist(i, slot) = theta_k * y_cur(i) + (1.0_dp - theta_k) * inv_gam * s_cur(i)
      else
        y_hist(i, slot) = y_cur(i)
      end if
    end do

    if (.not. did_damp) then
      do idx = 1_i32, n_st_tot
        s_hist(d + idx, slot) = s_cur(d + idx)
        y_hist(d + idx, slot) = y_cur(d + idx)
      end do
    else
      do r_idx = 1_i32, R
        inv_gam = 1.0_dp / min(max(gamma_blk(r_idx + 1_i32), GAMMA_MIN), GAMMA_MAX)
        x_s     = x_off(r_idx)
        x_e     = x_off(r_idx + 1_i32) - 1_i32
        do idx = x_s, x_e
          s_hist(d + idx, slot) = s_cur(d + idx)
          y_hist(d + idx, slot) = theta_k * y_cur(d + idx) + &
            (1.0_dp - theta_k) * inv_gam * s_cur(d + idx)
        end do
      end do
    end if

    acc_ss = 0.0_dp
    acc_sy = 0.0_dp
    acc_yy = 0.0_dp
    do r_idx = 1_i32, R
      nr   = n_arr(r_idx)
      pr   = p_arr(r_idx)
      p2   = pr * pr
      x_s  = x_off(r_idx)
      x_e  = x_off(r_idx + 1_i32) - 1_i32
      sk_s = skew_off(r_idx)
      sk_e = skew_off(r_idx + 1_i32) - 1_i32

      call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
        s_hist(d + x_s:d + x_e, slot), As_hist(sk_s:sk_e, slot), scratch(1:p2))
      call fc_stiefel_proj_tangent(nr, pr, X_new(x_s:x_e), &
        y_hist(d + x_s:d + x_e, slot), Ay_hist(sk_s:sk_e, slot), scratch(1:p2))

      f_ss = fc_frob_inner(nr, pr, s_hist(d + x_s:d + x_e, slot), &
        s_hist(d + x_s:d + x_e, slot))
      f_sy = fc_frob_inner(nr, pr, s_hist(d + x_s:d + x_e, slot), &
        y_hist(d + x_s:d + x_e, slot))
      f_yy = fc_frob_inner(nr, pr, y_hist(d + x_s:d + x_e, slot), &
        y_hist(d + x_s:d + x_e, slot))
      if (one_minus_beta > 0.0_dp) then
        a_ss   = fc_frob_inner(pr, pr, As_hist(sk_s:sk_e, slot), As_hist(sk_s:sk_e, slot))
        a_sy   = fc_frob_inner(pr, pr, As_hist(sk_s:sk_e, slot), Ay_hist(sk_s:sk_e, slot))
        a_yy   = fc_frob_inner(pr, pr, Ay_hist(sk_s:sk_e, slot), Ay_hist(sk_s:sk_e, slot))
        acc_ss = acc_ss + max(0.0_dp, f_ss - one_minus_beta * a_ss)
        acc_sy = acc_sy + (f_sy - one_minus_beta * a_sy)
        acc_yy = acc_yy + max(0.0_dp, f_yy - one_minus_beta * a_yy)
      else
        acc_ss = acc_ss + max(0.0_dp, f_ss)
        acc_sy = acc_sy + f_sy
        acc_yy = acc_yy + max(0.0_dp, f_yy)
      end if
    end do

    st_ss(slot) = acc_ss
    st_sy(slot) = acc_sy
    st_yy(slot) = acc_yy
  end subroutine fc_lbfgs_transport_and_push

end module fortchuck_lbfgs
