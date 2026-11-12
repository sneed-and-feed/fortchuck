program test_linalg_and_stiefel
  use, intrinsic :: iso_c_binding, only: c_sizeof
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_positive_inf, ieee_quiet_nan
  use fortchuck_kinds, only: &
    dp, i32, FC_HUGE, FC_PI, &
    FC_SUCCESS_QUERY, FC_ERR_INVALID_BOUNDS, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, FC_DAMP_CAUTIOUS, &
    FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED, &
    fc_options_t, fc_result_t, fc_init_default_options, fc_init_result
  use fortchuck_linalg, only: &
    fc_gemm_nn, fc_gemm_tn, fc_gemm_nt, fc_syrk_tn, &
    fc_frob_inner, fc_frob_norm, &
    fc_lu_factor, fc_lu_solve, fc_lu_solve_trans, fc_lu_solve_right, fc_geqrf_pos
  use fortchuck_box, only: &
    fc_box_validate, fc_box_project, fc_box_kkt_residual, &
    fc_box_partition, fc_box_active_step, fc_box_dir_deriv0, fc_box_trial_step
  use fortchuck_stiefel, only: &
    fc_stiefel_offsets, fc_stiefel_rgrad, fc_stiefel_proj_tangent, &
    fc_stiefel_inner, fc_stiefel_orth_err, fc_stiefel_orth_mat, &
    fc_cayley_setup, fc_cayley_retract, fc_cayley_transport_vec, &
    fc_cayley_setup_branch_a, fc_cayley_retract_branch_a, fc_cayley_transport_branch_a, &
    fc_cayley_setup_branch_b, fc_cayley_retract_branch_b, fc_cayley_transport_branch_b
  implicit none (type, external)

  integer(i32) :: rng_state

  rng_state = 133742_i32

  call test_structs_and_kinds()
  call test_gemm_syrk_frob()
  call test_lu_and_right_solve()
  call test_geqrf_pos_properties()
  call test_box_overflow_and_partition()
  call test_stiefel_geometry_and_cayley_parity()

  write (*, '(A)') '[PASS] test_linalg_and_stiefel: all assertions passed.'

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

  subroutine fill_rand_mat(m, n, A, state)
    integer(i32), intent(in)    :: m, n
    real(dp), intent(out)       :: A(m, n)
    integer(i32), intent(inout) :: state
    integer(i32)                :: i, j
    do j = 1_i32, n
      do i = 1_i32, m
        A(i, j) = next_rand(state)
      end do
    end do
  end subroutine fill_rand_mat

  function mat_diff_frob(m, n, A, B) result(err)
    integer(i32), intent(in) :: m, n
    real(dp), intent(in)     :: A(m, n), B(m, n)
    real(dp)                 :: err
    integer(i32)             :: i, j
    real(dp)                 :: acc, d
    acc = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        d   = A(i, j) - B(i, j)
        acc = acc + d * d
      end do
    end do
    err = sqrt(acc)
  end function mat_diff_frob

  function det_6x6(A_in) result(det_val)
    real(dp), intent(in) :: A_in(6, 6)
    real(dp)             :: det_val
    real(dp)             :: LU(6, 6)
    integer(i32)         :: ipiv(6), info, i
    LU = A_in
    call fc_lu_factor(6_i32, LU, ipiv, 1.0e-12_dp, info)
    if (info /= 0_i32) then
      det_val = 0.0_dp
      return
    end if
    det_val = 1.0_dp
    do i = 1_i32, 6_i32
      if (ipiv(i) /= i) det_val = -det_val
      det_val = det_val * LU(i, i)
    end do
  end function det_6x6

  !---------------------------------------------------------------------------
  ! 1. Verify C-ABI struct sizes and default initialization
  !---------------------------------------------------------------------------
  subroutine test_structs_and_kinds()
    type(fc_options_t) :: opts
    type(fc_result_t)  :: res

    call fc_init_default_options(opts)
    call fc_init_result(res)

    call check_true(int(c_sizeof(opts), i32) == 96_i32, 'sizeof(fc_options_t) == 96')
    call check_true(int(c_sizeof(res),  i32) == 88_i32, 'sizeof(fc_result_t) == 88')
    call check_true(abs(opts%gtol - 1.0e-6_dp) <= 1.0e-16_dp, 'default gtol')
    call check_true(abs(opts%max_step_vel - 2.0_dp * FC_PI) <= 1.0e-15_dp, 'default max_step_vel')
    call check_true(opts%memory == 7_i32, 'default memory == 7')
    call check_true(opts%metric == FC_METRIC_CANONICAL, 'default canonical metric')
    call check_true(opts%damping_mode == FC_DAMP_CAUTIOUS, 'default cautious damping')
    call check_true(res%status == FC_SUCCESS_QUERY, 'default result status')
  end subroutine test_structs_and_kinds

  !---------------------------------------------------------------------------
  ! 2. Verify GEMM (NN, TN, NT), SYRK, and overflow-safe Frobenius norm
  !---------------------------------------------------------------------------
  subroutine test_gemm_syrk_frob()
    real(dp) :: A(7, 5), B(5, 6), C1(7, 6), C2(7, 6)
    real(dp) :: AT(5, 7), BT(6, 5), S1(5, 5), S2(5, 5)
    real(dp) :: AL(40, 32), BL(32, 32), CL1(40, 32), CL2(40, 32)
    real(dp) :: big_vec(2, 1), nrm, acc
    integer(i32) :: i, j, l

    call fill_rand_mat(7_i32, 5_i32, A, rng_state)
    call fill_rand_mat(5_i32, 6_i32, B, rng_state)
    do j = 1_i32, 5_i32
      do i = 1_i32, 7_i32
        AT(j, i) = A(i, j)
      end do
    end do
    do j = 1_i32, 6_i32
      do i = 1_i32, 5_i32
        BT(j, i) = B(i, j)
      end do
    end do

    ! Reference C2 = 1.5 * A * B
    C1 = 999.0_dp
    call fc_gemm_nn(7_i32, 6_i32, 5_i32, 1.5_dp, A, 7_i32, B, 5_i32, 0.0_dp, C1, 7_i32)
    do j = 1_i32, 6_i32
      do i = 1_i32, 7_i32
        acc = 0.0_dp
        do l = 1_i32, 5_i32
          acc = acc + A(i, l) * B(l, j)
        end do
        C2(i, j) = 1.5_dp * acc
      end do
    end do
    call check_true(mat_diff_frob(7_i32, 6_i32, C1, C2) <= 1.0e-14_dp, 'fc_gemm_nn beta=0')

    ! Nonzero beta accumulation across TN and NT
    call fill_rand_mat(7_i32, 6_i32, C1, rng_state)
    do j = 1_i32, 6_i32
      do i = 1_i32, 7_i32
        acc = 0.0_dp
        do l = 1_i32, 5_i32
          acc = acc + AT(l, i) * B(l, j)
        end do
        C2(i, j) = -0.7_dp * C1(i, j) + 2.25_dp * acc
      end do
    end do
    call fc_gemm_tn(7_i32, 6_i32, 5_i32, 2.25_dp, AT, 5_i32, B, 5_i32, -0.7_dp, C1, 7_i32)
    call check_true(mat_diff_frob(7_i32, 6_i32, C1, C2) <= 1.0e-14_dp, 'fc_gemm_tn beta/=0')

    call fill_rand_mat(7_i32, 6_i32, C1, rng_state)
    do j = 1_i32, 6_i32
      do i = 1_i32, 7_i32
        acc = 0.0_dp
        do l = 1_i32, 5_i32
          acc = acc + A(i, l) * BT(j, l)
        end do
        C2(i, j) = 0.35_dp * C1(i, j) - 1.1_dp * acc
      end do
    end do
    call fc_gemm_nt(7_i32, 6_i32, 5_i32, -1.1_dp, A, 7_i32, BT, 6_i32, 0.35_dp, C1, 7_i32)
    call check_true(mat_diff_frob(7_i32, 6_i32, C1, C2) <= 1.0e-14_dp, 'fc_gemm_nt beta/=0')

    ! SYRK exact symmetry & match with A^T * A
    call fc_syrk_tn(7_i32, 5_i32, A, S1)
    call fc_gemm_tn(5_i32, 5_i32, 7_i32, 1.0_dp, A, 7_i32, A, 7_i32, 0.0_dp, S2, 5_i32)
    call check_true(mat_diff_frob(5_i32, 5_i32, S1, S2) <= 1.0e-14_dp, 'fc_syrk_tn accuracy')
    do j = 1_i32, 5_i32
      do i = 1_i32, j - 1_i32
        call check_true(.not. (abs(S1(i, j) - S1(j, i)) > 0.0_dp), 'fc_syrk_tn exact symmetry')
      end do
    end do

    ! Large GEMM check (40 * 32 * 32 = 40960 > 32768 BLAS dispatch threshold)
    call fill_rand_mat(40_i32, 32_i32, AL, rng_state)
    call fill_rand_mat(32_i32, 32_i32, BL, rng_state)
    call fc_gemm_nn(40_i32, 32_i32, 32_i32, 1.25_dp, AL, 40_i32, BL, 32_i32, 0.0_dp, CL1, 40_i32)
    do j = 1_i32, 32_i32
      do i = 1_i32, 40_i32
        acc = 0.0_dp
        do l = 1_i32, 32_i32
          acc = acc + AL(i, l) * BL(l, j)
        end do
        CL2(i, j) = 1.25_dp * acc
      end do
    end do
    call check_true(mat_diff_frob(40_i32, 32_i32, CL1, CL2) <= 1.0e-13_dp, 'fc_gemm_nn large m*n*k>32768')

    ! Overflow-safe Frobenius norm near FC_HUGE (0.6^2 + 0.8^2 = 1.0)
    big_vec(1, 1) = 0.6_dp * FC_HUGE
    big_vec(2, 1) = 0.8_dp * FC_HUGE
    nrm = fc_frob_norm(2_i32, 1_i32, big_vec)
    call check_true(abs(nrm / FC_HUGE - 1.0_dp) <= 1.0e-14_dp, 'fc_frob_norm overflow-safe')
  end subroutine test_gemm_syrk_frob

  !---------------------------------------------------------------------------
  ! 3. Verify LU factorization, solve, transpose solve, right-panel solve,
  !    and ill-conditioned / NaN pivot guards
  !---------------------------------------------------------------------------
  subroutine test_lu_and_right_solve()
    integer(i32), parameter :: n = 8_i32, nrhs = 5_i32, m_pan = 11_i32, p_pan = 6_i32
    real(dp)     :: A(n, n), LU(n, n), B(n, nrhs), X_sol(n, nrhs), Prod(n, nrhs)
    real(dp)     :: J_mat(p_pan, p_pan), LU_Jt(p_pan, p_pan)
    real(dp)     :: Panel(m_pan, p_pan), Z_pan(m_pan, p_pan), Prod_pan(m_pan, p_pan)
    real(dp)     :: Bad(4, 4)
    integer(i32) :: ipiv(n), ipiv_p(p_pan), ipiv_bad(4), info, i, j

    call fill_rand_mat(n, n, A, rng_state)
    do i = 1_i32, n
      A(i, i) = A(i, i) + 0.5_dp
    end do
    call fill_rand_mat(n, nrhs, B, rng_state)

    LU = A
    call fc_lu_factor(n, LU, ipiv, 1.0e-12_dp, info)
    call check_true(info == 0_i32, 'fc_lu_factor nonsingular info==0')

    X_sol = B
    call fc_lu_solve(n, nrhs, LU, ipiv, X_sol)
    call fc_gemm_nn(n, nrhs, n, 1.0_dp, A, n, X_sol, n, 0.0_dp, Prod, n)
    call check_true(mat_diff_frob(n, nrhs, Prod, B) <= &
      1.0e-15_dp * (fc_frob_norm(n, n, A) * fc_frob_norm(n, nrhs, X_sol) + fc_frob_norm(n, nrhs, B)), &
      'fc_lu_solve backward residual <= 1e-15')

    X_sol = B
    call fc_lu_solve_trans(n, nrhs, LU, ipiv, X_sol)
    call fc_gemm_tn(n, nrhs, n, 1.0_dp, A, n, X_sol, n, 0.0_dp, Prod, n)
    call check_true(mat_diff_frob(n, nrhs, Prod, B) <= &
      1.0e-15_dp * (fc_frob_norm(n, n, A) * fc_frob_norm(n, nrhs, X_sol) + fc_frob_norm(n, nrhs, B)), &
      'fc_lu_solve_trans backward residual <= 1e-15')

    ! Right-panel solve Z * J = Panel via LU of J^T
    call fill_rand_mat(p_pan, p_pan, J_mat, rng_state)
    do i = 1_i32, p_pan
      J_mat(i, i) = J_mat(i, i) + 1.5_dp
    end do
    call fill_rand_mat(m_pan, p_pan, Panel, rng_state)

    do j = 1_i32, p_pan
      do i = 1_i32, p_pan
        LU_Jt(i, j) = J_mat(j, i)
      end do
    end do
    call fc_lu_factor(p_pan, LU_Jt, ipiv_p, 1.0e-12_dp, info)
    call check_true(info == 0_i32, 'fc_lu_factor(J^T) info==0')

    Z_pan = Panel
    call fc_lu_solve_right(m_pan, p_pan, LU_Jt, ipiv_p, Z_pan)
    call fc_gemm_nn(m_pan, p_pan, p_pan, 1.0_dp, Z_pan, m_pan, J_mat, p_pan, 0.0_dp, Prod_pan, m_pan)
    call check_true(mat_diff_frob(m_pan, p_pan, Prod_pan, Panel) <= &
      1.0e-15_dp * (fc_frob_norm(m_pan, p_pan, Z_pan) * fc_frob_norm(p_pan, p_pan, J_mat) + &
                    fc_frob_norm(m_pan, p_pan, Panel)), 'fc_lu_solve_right backward residual <= 1e-15')

    ! Pivot ratio guard: U(1,1) = 1e-13, U(4,4) = 1.0 -> must reject!
    Bad = 0.0_dp
    Bad(1, 1) = 1.0e-13_dp
    Bad(2, 2) = 1.0_dp
    Bad(3, 3) = 1.0_dp
    Bad(4, 4) = 1.0_dp
    call fc_lu_factor(4_i32, Bad, ipiv_bad, 1.0e-12_dp, info)
    call check_true(info > 0_i32, 'fc_lu_factor rejects small leading pivot ratio')

    ! Pivot ratio guard: U(1,1) = 1.0, U(4,4) = 1e-13 -> must reject!
    Bad = 0.0_dp
    Bad(1, 1) = 1.0_dp
    Bad(2, 2) = 1.0_dp
    Bad(3, 3) = 1.0_dp
    Bad(4, 4) = 1.0e-13_dp
    call fc_lu_factor(4_i32, Bad, ipiv_bad, 1.0e-12_dp, info)
    call check_true(info == 4_i32, 'fc_lu_factor rejects small trailing pivot ratio')

    ! NaN guard
    Bad = 0.0_dp
    do i = 1_i32, 4_i32
      Bad(i, i) = 1.0_dp
    end do
    Bad(2, 3) = ieee_value(1.0_dp, ieee_quiet_nan)
    call fc_lu_factor(4_i32, Bad, ipiv_bad, 1.0e-12_dp, info)
    call check_true(info > 0_i32, 'fc_lu_factor rejects NaN')
  end subroutine test_lu_and_right_solve

  !---------------------------------------------------------------------------
  ! 4. Verify fc_geqrf_pos: orthonormality, R_ii > 0, det(Q)=+1 on SO(n),
  !    small-drift bound ||Q - X||_F <= delta, and rank-deficiency detection
  !---------------------------------------------------------------------------
  subroutine test_geqrf_pos_properties()
    real(dp)     :: X0(12, 6), Q(12, 6), QR_prod(12, 6), R_rec(6, 6), tau_qr(6), work_qr(6)
    real(dp)     :: Q_so(6, 6), Pert(6, 6), X_so(6, 6)
    real(dp)     :: X_drift(12, 4), Q_drift(12, 4), E_pert(12, 4)
    real(dp)     :: X_rank(8, 4)
    real(dp)     :: det0, det1, delta, q_diff
    integer(i32) :: info, i, j

    ! Rectangular St(12, 6)
    call fill_rand_mat(12_i32, 6_i32, X0, rng_state)
    Q = X0
    call fc_geqrf_pos(12_i32, 6_i32, Q, tau_qr, work_qr, info)
    call check_true(info == 0_i32, 'fc_geqrf_pos (12,6) info==0')
    call check_true(fc_stiefel_orth_err(12_i32, 6_i32, Q) <= 1.0e-15_dp, &
      'fc_geqrf_pos ||Q^T Q - I||_F <= 1e-15')

    call fc_gemm_tn(6_i32, 6_i32, 12_i32, 1.0_dp, Q, 12_i32, X0, 12_i32, 0.0_dp, R_rec, 6_i32)
    call fc_gemm_nn(12_i32, 6_i32, 6_i32, 1.0_dp, Q, 12_i32, R_rec, 6_i32, 0.0_dp, QR_prod, 12_i32)
    call check_true(mat_diff_frob(12_i32, 6_i32, QR_prod, X0) <= &
      1.0e-14_dp * fc_frob_norm(12_i32, 6_i32, X0), 'fc_geqrf_pos Q*R == X0')
    do j = 1_i32, 6_i32
      call check_true(R_rec(j, j) > 0.0_dp, 'fc_geqrf_pos R(j,j) > 0')
      do i = j + 1_i32, 6_i32
        call check_true(abs(R_rec(i, j)) <= 1.0e-14_dp, 'fc_geqrf_pos R upper triangular')
      end do
    end do

    ! Negative identity -I_6 must return -I_6 with R = +I_6 (zero sign flip!)
    Q_so = 0.0_dp
    do i = 1_i32, 6_i32
      Q_so(i, i) = -1.0_dp
    end do
    X_so = Q_so
    call fc_geqrf_pos(6_i32, 6_i32, X_so, tau_qr, work_qr, info)
    call check_true(info == 0_i32, 'fc_geqrf_pos(-I_6) info==0')
    call check_true(mat_diff_frob(6_i32, 6_i32, X_so, Q_so) <= 1.0e-15_dp, &
      'fc_geqrf_pos(-I_6) preserves -I_6 without sign flip')

    ! Perturbed SO(6) preserves det(Q) = +1
    call fill_rand_mat(6_i32, 6_i32, Q_so, rng_state)
    call fc_geqrf_pos(6_i32, 6_i32, Q_so, tau_qr, work_qr, info)
    det0 = det_6x6(Q_so)
    if (det0 < 0.0_dp) then
      do i = 1_i32, 6_i32
        Q_so(i, 1) = -Q_so(i, 1)
      end do
    end if
    det0 = det_6x6(Q_so)
    call check_true(abs(det0 - 1.0_dp) <= 1.0e-14_dp, 'SO(6) base det == +1')

    call fill_rand_mat(6_i32, 6_i32, Pert, rng_state)
    do j = 1_i32, 6_i32
      do i = 1_i32, 6_i32
        X_so(i, j) = Q_so(i, j) + 1.0e-3_dp * Pert(i, j)
      end do
    end do
    call fc_geqrf_pos(6_i32, 6_i32, X_so, tau_qr, work_qr, info)
    call check_true(info == 0_i32, 'fc_geqrf_pos perturbed SO(6) info==0')
    call check_true(fc_stiefel_orth_err(6_i32, 6_i32, X_so) <= 1.0e-15_dp, &
      'fc_geqrf_pos perturbed SO(6) orth <= 1e-15')
    det1 = det_6x6(X_so)
    call check_true(abs(det1 - 1.0_dp) <= 1.0e-14_dp, 'fc_geqrf_pos preserves det(Q) == +1 on SO(6)')

    ! Small drift delta ~ 1e-12 satisfies ||Q - X||_F <= delta
    call fill_rand_mat(12_i32, 4_i32, X_drift, rng_state)
    call fc_geqrf_pos(12_i32, 4_i32, X_drift, tau_qr(1:4), work_qr(1:4), info)
    call fill_rand_mat(12_i32, 4_i32, E_pert, rng_state)
    do j = 1_i32, 4_i32
      do i = 1_i32, 12_i32
        X_drift(i, j) = X_drift(i, j) + 3.0e-13_dp * E_pert(i, j)
      end do
    end do
    delta   = fc_stiefel_orth_err(12_i32, 4_i32, X_drift)
    Q_drift = X_drift
    call fc_geqrf_pos(12_i32, 4_i32, Q_drift, tau_qr(1:4), work_qr(1:4), info)
    q_diff  = mat_diff_frob(12_i32, 4_i32, Q_drift, X_drift)
    call check_true(delta > 1.0e-13_dp .and. delta < 1.0e-11_dp, 'drift test setup delta ~ 1e-12')
    call check_true(q_diff <= delta, 'fc_geqrf_pos ||Q - X||_F <= delta')
    call check_true(fc_stiefel_orth_err(12_i32, 4_i32, Q_drift) <= 1.0e-15_dp, &
      'fc_geqrf_pos polished drift <= 1e-15')

    ! Rank-deficient column detection
    call fill_rand_mat(8_i32, 4_i32, X_rank, rng_state)
    do i = 1_i32, 8_i32
      X_rank(i, 3) = 2.0_dp * X_rank(i, 1) - 0.5_dp * X_rank(i, 2)
    end do
    call fc_geqrf_pos(8_i32, 4_i32, X_rank, tau_qr(1:4), work_qr(1:4), info)
    call check_true(info == 3_i32, 'fc_geqrf_pos detects rank deficiency at col 3')
  end subroutine test_geqrf_pos_properties

  !---------------------------------------------------------------------------
  ! 5. Verify overflow-safe box routines on +-huge(1.0_dp), +-Inf, and fixed
  !---------------------------------------------------------------------------
  subroutine test_box_overflow_and_partition()
    integer(i32), parameter :: d = 6_i32
    real(dp)     :: l(d), u(d), z(d), gz(d), rbox(d), dz(d), z_trial(d)
    real(dp)     :: s_prev(d), y_prev(d), l_bad(2), u_bad(2)
    real(dp)     :: box_inf, box_l2, slope0, v_sq, g_sq, chord_gz, step_sq, pinf
    integer(i32) :: mask_box(d), status, n_act

    pinf = ieee_value(1.0_dp, ieee_positive_inf)

    l(1) = -FC_HUGE; l(2) = -FC_HUGE; l(3) = 2.5_dp; l(4) = -1.0_dp; l(5) = -pinf;    l(6) = FC_HUGE
    u(1) =  FC_HUGE; u(2) =  FC_HUGE; u(3) = 2.5_dp; u(4) =  1.0_dp; u(5) = -FC_HUGE; u(6) = pinf
    z  = [ 0.5_dp,  -FC_HUGE, 9.0_dp,  1.0_dp, -FC_HUGE, FC_HUGE]
    gz = [ 3.0_dp,   4.0_dp, -7.0_dp, -2.0_dp, -5.0_dp,  6.0_dp ]

    call fc_box_validate(d, l, u, status)
    call check_true(status == 0_i32, 'fc_box_validate accepts +-FC_HUGE and one-sided Inf')

    call fc_box_project(d, l, u, z)
    call check_true(abs(z(3) - 2.5_dp) <= 1.0e-16_dp, 'fc_box_project clamps fixed coord')

    call fc_box_kkt_residual(d, z, l, u, gz, rbox, box_inf, box_l2)
    call check_true(abs(rbox(1) - 3.0_dp) <= 1.0e-15_dp, 'rbox(1) free == gz(1)')
    call check_true(abs(rbox(2)) <= 1.0e-15_dp, 'rbox(2) clamped lower == 0')
    call check_true(abs(rbox(3)) <= 1.0e-15_dp, 'rbox(3) fixed == 0')
    call check_true(abs(rbox(4)) <= 1.0e-15_dp, 'rbox(4) clamped upper == 0')
    call check_true(abs(box_inf - 3.0_dp) <= 1.0e-15_dp, 'box_inf == 3.0')
    call check_true(abs(box_l2 - 3.0_dp) <= 1.0e-15_dp,  'box_l2 == 3.0')

    call fc_box_partition(d, z, l, u, gz, box_inf, 1.0e-4_dp, mask_box, n_act)
    call check_true(mask_box(1) == FC_BOX_FREE,   'mask_box(1) FREE')
    call check_true(mask_box(2) == FC_BOX_ACTIVE, 'mask_box(2) ACTIVE')
    call check_true(mask_box(3) == FC_BOX_FIXED,  'mask_box(3) FIXED')
    call check_true(mask_box(4) == FC_BOX_ACTIVE, 'mask_box(4) ACTIVE')
    call check_true(mask_box(5) == FC_BOX_ACTIVE, 'mask_box(5) ACTIVE (-Inf, -FC_HUGE)')
    call check_true(mask_box(6) == FC_BOX_ACTIVE, 'mask_box(6) ACTIVE (+FC_HUGE, +Inf)')
    call check_true(n_act == 5_i32, 'n_active == 5')

    dz     = [-1.5_dp, 99.0_dp, 99.0_dp, 99.0_dp, 99.0_dp, 99.0_dp]
    s_prev = [0.1_dp, 0.2_dp, 0.0_dp, 0.4_dp, 0.1_dp, 0.1_dp]
    y_prev = [0.2_dp, 0.1_dp, 0.0_dp, 0.8_dp, 0.2_dp, 0.2_dp]
    call fc_box_active_step(d, z, l, u, gz, mask_box, 0.5_dp, .true., s_prev, y_prev, dz)
    call check_true(abs(dz(1) + 1.5_dp) <= 1.0e-15_dp, 'fc_box_active_step preserves FREE dz(1)')
    call check_true(abs(dz(3)) <= 1.0e-16_dp,          'fc_box_active_step zeroes FIXED dz(3)')
    call check_true(abs(dz(4) - 1.0_dp) <= 1.0e-15_dp, 'fc_box_active_step secant scale dz(4)')

    call fc_box_dir_deriv0(d, z, l, u, gz, dz, slope0, v_sq, g_sq)
    call check_true(abs(slope0 + 4.5_dp) <= 1.0e-15_dp, 'fc_box_dir_deriv0 exact slope0')
    call check_true(abs(v_sq - 2.25_dp) <= 1.0e-15_dp,  'fc_box_dir_deriv0 exact v_sq')
    call check_true(abs(g_sq - 9.0_dp) <= 1.0e-15_dp,   'fc_box_dir_deriv0 exact g_sq')

    call fc_box_trial_step(d, z, l, u, gz, dz, 1.0_dp, z_trial, chord_gz, step_sq)
    call check_true(abs(z_trial(1) + 1.0_dp) <= 1.0e-15_dp, 'fc_box_trial_step z_trial(1)')
    call check_true(abs(chord_gz + 4.5_dp) <= 1.0e-15_dp,   'fc_box_trial_step chord_gz')
    call check_true(abs(step_sq - 2.25_dp) <= 1.0e-15_dp,   'fc_box_trial_step step_sq')

    l_bad = [1.0_dp, 0.0_dp]
    u_bad = [0.5_dp, 1.0_dp]
    call fc_box_validate(2_i32, l_bad, u_bad, status)
    call check_true(status == FC_ERR_INVALID_BOUNDS, 'fc_box_validate rejects l > u')

    l_bad(1) = pinf; l_bad(2) = 0.0_dp
    u_bad(1) = pinf; u_bad(2) = 1.0_dp
    call fc_box_validate(2_i32, l_bad, u_bad, status)
    call check_true(status == FC_ERR_INVALID_BOUNDS, 'fc_box_validate rejects l = +Inf')
  end subroutine test_box_overflow_and_partition

  !---------------------------------------------------------------------------
  ! Helper running a single (n, p) Stiefel & Branch A == B parity check using
  ! contiguous explicit-shape buffers sliced from caller arena (0 temporaries).
  !---------------------------------------------------------------------------
  subroutine verify_single_stiefel_case(n, p, X, GX, rgrad, eta, &
      z1, z2, z1_a, z1_b, z1_auto, z2_a, z2_b, Y_a, Y_b, Y_auto, Y_p, Y_m, &
      A_sk, scr_p2, E_mat, XTrg, XTz, YTza)
    integer(i32), intent(in) :: n, p
    real(dp), intent(inout)  :: X(n, p), GX(n, p), rgrad(n, p), eta(n, p)
    real(dp), intent(inout)  :: z1(n, p), z2(n, p), z1_a(n, p), z1_b(n, p), z1_auto(n, p)
    real(dp), intent(inout)  :: z2_a(n, p), z2_b(n, p)
    real(dp), intent(inout)  :: Y_a(n, p), Y_b(n, p), Y_auto(n, p), Y_p(n, p), Y_m(n, p)
    real(dp), intent(inout)  :: A_sk(p, p), scr_p2(p, p), E_mat(p, p)
    real(dp), intent(inout)  :: XTrg(p, p), XTz(p, p), YTza(p, p)

    real(dp), parameter :: taus(4) = [0.0_dp, 0.05_dp, 0.7_dp, 2.3_dp]
    real(dp)            :: cay_a(1024), cay_b(1024), cay_auto(1024), scr(1024)
    real(dp)            :: tau_qr(12), work_qr(12)
    integer(i32)        :: ipiv_a(12), ipiv_b(12), ipiv_auto(12)
    integer(i32)        :: t_idx, info_a, info_b, info_auto, i, j
    real(dp)            :: tau, ip_riesz, ip_amb, in_can_x, in_can_y, in_euc_x, in_euc_y
    real(dp)            :: err_mat, h_fd, vel_err, d_ij

    call fill_rand_mat(n, p, X, rng_state)
    call fc_geqrf_pos(n, p, X, tau_qr(1:p), work_qr(1:p), info_a)
    call check_true(info_a == 0_i32, 'Stiefel base frame QR')

    call fc_stiefel_orth_mat(n, p, X, E_mat, err_mat)
    call check_true(err_mat <= 1.0e-15_dp, 'fc_stiefel_orth_mat err <= 1e-15')

    call fill_rand_mat(n, p, eta, rng_state)
    call fc_stiefel_proj_tangent(n, p, X, eta, A_sk, scr_p2)
    call fill_rand_mat(n, p, z1, rng_state)
    call fc_stiefel_proj_tangent(n, p, X, z1, A_sk, scr_p2)
    call fill_rand_mat(n, p, z2, rng_state)
    call fc_stiefel_proj_tangent(n, p, X, z2, A_sk, scr_p2)

    ! Riesz identity & A_skew == X^T * rgrad under both metrics
    call fill_rand_mat(n, p, GX, rng_state)
    call fc_stiefel_rgrad(n, p, X, GX, FC_METRIC_CANONICAL, rgrad, A_sk, scr_p2)
    ip_riesz = fc_stiefel_inner(n, p, X, rgrad, eta, FC_METRIC_CANONICAL)
    ip_amb   = fc_frob_inner(n, p, GX, eta)
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, rgrad, n, 0.0_dp, XTrg, p)
    call check_true(abs(ip_riesz - ip_amb) <= 1.0e-14_dp, 'Canonical rgrad Riesz identity')
    call check_true(mat_diff_frob(p, p, XTrg, A_sk) <= 1.0e-14_dp, 'Canonical A_skew == X^T*rgrad')

    call fc_stiefel_rgrad(n, p, X, GX, FC_METRIC_EUCLIDEAN, rgrad, A_sk, scr_p2)
    ip_riesz = fc_stiefel_inner(n, p, X, rgrad, eta, FC_METRIC_EUCLIDEAN)
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, rgrad, n, 0.0_dp, XTrg, p)
    call check_true(abs(ip_riesz - ip_amb) <= 1.0e-14_dp, 'Euclidean rgrad Riesz identity')
    call check_true(mat_diff_frob(p, p, XTrg, A_sk) <= 1.0e-14_dp, 'Euclidean A_skew == X^T*rgrad')

    ! Setup Branch A, Branch B, and Auto-Dispatch on identical (X, eta)
    call fc_cayley_setup_branch_a(n, p, X, eta, cay_a, scr)
    call fc_cayley_setup_branch_b(n, p, X, eta, cay_b, scr)
    call fc_cayley_setup(n, p, X, eta, cay_auto, scr)

    ! Central FD check of initial curve velocity d/dtau R_X^Cay(tau*eta)|_0 == eta
    h_fd = 1.0e-6_dp
    call fc_cayley_retract(n, p, X, eta,  h_fd, cay_auto, ipiv_auto, Y_p, scr, info_auto)
    call fc_cayley_retract(n, p, X, eta, -h_fd, cay_auto, ipiv_auto, Y_m, scr, info_auto)
    vel_err = 0.0_dp
    do j = 1_i32, p
      do i = 1_i32, n
        d_ij    = (Y_p(i, j) - Y_m(i, j)) / (2.0_dp * h_fd) - eta(i, j)
        vel_err = vel_err + d_ij * d_ij
      end do
    end do
    call check_true(sqrt(vel_err) <= 1.0e-9_dp, 'Cayley curve initial velocity == eta')

    in_can_x = fc_stiefel_inner(n, p, X, z1, z2, FC_METRIC_CANONICAL)
    in_euc_x = fc_stiefel_inner(n, p, X, z1, z2, FC_METRIC_EUCLIDEAN)
    call fc_gemm_tn(p, p, n, 1.0_dp, X, n, z1, n, 0.0_dp, XTz, p)

    do t_idx = 1_i32, 4_i32
      tau = taus(t_idx)

      call fc_cayley_retract_branch_a(n, p, X, eta, tau, cay_a, ipiv_a(1:p), Y_a, scr, info_a)
      call fc_cayley_retract_branch_b(n, p, X, eta, tau, cay_b, ipiv_b(1:n), Y_b, scr, info_b)
      call fc_cayley_retract(n, p, X, eta, tau, cay_auto, ipiv_auto, Y_auto, scr, info_auto)

      call check_true(info_a == 0_i32 .and. info_b == 0_i32 .and. info_auto == 0_i32, &
        'Cayley retract info == 0 across Branch A, B, Auto')
      call check_true(mat_diff_frob(n, p, Y_a, Y_b) <= 1.0e-14_dp, &
        'Branch A == Branch B retraction parity <= 1e-14')
      call check_true(mat_diff_frob(n, p, Y_auto, Y_b) <= 1.0e-14_dp, &
        'Auto == Branch B retraction parity <= 1e-14')
      call check_true(fc_stiefel_orth_err(n, p, Y_a) <= 1.0e-14_dp, &
        'Branch A manifold feasibility <= 1e-14')
      call check_true(fc_stiefel_orth_err(n, p, Y_b) <= 1.0e-14_dp, &
        'Branch B manifold feasibility <= 1e-14')

      z1_a    = z1
      z1_b    = z1
      z1_auto = z1
      z2_a    = z2
      z2_b    = z2

      call fc_cayley_transport_branch_a(n, p, X, eta, tau, cay_a, ipiv_a(1:p), z1_a, scr)
      call fc_cayley_transport_branch_b(n, p, X, eta, tau, cay_b, ipiv_b(1:n), z1_b, scr)
      call fc_cayley_transport_vec(n, p, X, eta, tau, cay_auto, ipiv_auto, z1_auto, scr)

      call fc_cayley_transport_branch_a(n, p, X, eta, tau, cay_a, ipiv_a(1:p), z2_a, scr)
      call fc_cayley_transport_branch_b(n, p, X, eta, tau, cay_b, ipiv_b(1:n), z2_b, scr)

      call check_true(mat_diff_frob(n, p, z1_a, z1_b) <= 1.0e-14_dp, &
        'Branch A == Branch B transport parity <= 1e-14')
      call check_true(mat_diff_frob(n, p, z1_auto, z1_b) <= 1.0e-14_dp, &
        'Auto == Branch B transport parity <= 1e-14')

      call fc_gemm_tn(p, p, n, 1.0_dp, Y_a, n, z1_a, n, 0.0_dp, YTza, p)
      call check_true(mat_diff_frob(p, p, YTza, XTz) <= 1.0e-14_dp, &
        'Isometric Cayley transport preserves Y^T * T(zeta) == X^T * zeta')

      in_can_y = fc_stiefel_inner(n, p, Y_a, z1_a, z2_a, FC_METRIC_CANONICAL)
      in_euc_y = fc_stiefel_inner(n, p, Y_a, z1_a, z2_a, FC_METRIC_EUCLIDEAN)
      call check_true(abs(in_can_y - in_can_x) <= 1.0e-14_dp, &
        'Isometric Cayley transport preserves canonical inner product <= 1e-14')
      call check_true(abs(in_euc_y - in_euc_x) <= 1.0e-14_dp, &
        'Isometric Cayley transport preserves Euclidean inner product <= 1e-14')
    end do
  end subroutine verify_single_stiefel_case

  !---------------------------------------------------------------------------
  ! 6. Verify Stiefel offsets and Branch A == B across:
  !      (n, p) in {(10, 1), (12, 3), (8, 4), (7, 5), (6, 6)}
  !---------------------------------------------------------------------------
  subroutine test_stiefel_geometry_and_cayley_parity()
    integer(i32), parameter :: n_cases = 5_i32
    integer(i32), parameter :: ns(n_cases) = [10_i32, 12_i32, 8_i32, 7_i32, 6_i32]
    integer(i32), parameter :: ps(n_cases) = [ 1_i32,  3_i32, 4_i32, 5_i32, 6_i32]
    integer(i32), parameter :: n_arr_off(3) = [10_i32, 12_i32, 6_i32]
    integer(i32), parameter :: p_arr_off(3) = [ 1_i32,  3_i32, 6_i32]

    real(dp)     :: arena_np(144, 16), arena_p2(144, 6)
    integer(i32) :: x_off(4), sk_off(4), c_off(4), p_off(4)
    integer(i32) :: n_st_t, n_sk_t, n_cay_t, n_piv_t, st_code
    integer(i32) :: c_idx, n, p, np, p2

    call fc_stiefel_offsets(3_i32, n_arr_off, p_arr_off, &
      x_off, sk_off, c_off, p_off, n_st_t, n_sk_t, n_cay_t, n_piv_t, st_code)
    call check_true(st_code == 0_i32, 'fc_stiefel_offsets valid')
    call check_true(n_st_t == 82_i32,  'fc_stiefel_offsets n_st_tot')
    call check_true(n_sk_t == 46_i32,  'fc_stiefel_offsets n_skew_tot')
    call check_true(n_cay_t == 194_i32, 'fc_stiefel_offsets n_cayley_tot')
    call check_true(n_piv_t == 10_i32, 'fc_stiefel_offsets n_ipiv_tot')

    do c_idx = 1_i32, n_cases
      n  = ns(c_idx)
      p  = ps(c_idx)
      np = n * p
      p2 = p * p
      call verify_single_stiefel_case(n, p, &
        arena_np(1:np,  1), arena_np(1:np,  2), arena_np(1:np,  3), arena_np(1:np,  4), &
        arena_np(1:np,  5), arena_np(1:np,  6), arena_np(1:np,  7), arena_np(1:np,  8), &
        arena_np(1:np,  9), arena_np(1:np, 10), arena_np(1:np, 11), arena_np(1:np, 12), &
        arena_np(1:np, 13), arena_np(1:np, 14), arena_np(1:np, 15), arena_np(1:np, 16), &
        arena_p2(1:p2,  1), arena_p2(1:p2,  2), arena_p2(1:p2,  3), &
        arena_p2(1:p2,  4), arena_p2(1:p2,  5), arena_p2(1:p2,  6))
    end do
  end subroutine test_stiefel_geometry_and_cayley_parity

end program test_linalg_and_stiefel
