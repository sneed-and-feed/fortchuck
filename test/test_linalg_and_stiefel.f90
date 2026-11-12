program test_linalg_and_stiefel
  use, intrinsic :: iso_c_binding, only: c_sizeof
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_positive_inf, ieee_quiet_nan
  use fortchuck_kinds, only: &
    dp, i32, FC_HUGE, FC_PI, &
    FC_SUCCESS_QUERY, FC_ERR_INVALID_DIM, FC_ERR_INVALID_BOUNDS, FC_ERR_INFEASIBLE_LIN, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, FC_DAMP_CAUTIOUS, &
    FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED, &
    fc_options_t, fc_result_t, fc_init_default_options, fc_init_result
  use fortchuck_linalg, only: &
    fc_gemm_nn, fc_gemm_tn, fc_gemm_nt, fc_syrk_tn, &
    fc_frob_inner, fc_frob_norm, &
    fc_lu_factor, fc_lu_solve, fc_lu_solve_trans, fc_lu_solve_right, fc_geqrf_pos
  use fortchuck_box, only: &
    fc_box_validate, fc_box_project, fc_box_kkt_residual, &
    fc_box_partition, fc_box_active_step, fc_box_active_step_diag, &
    fc_box_breakpoint_cap, fc_box_dir_deriv0, fc_box_trial_step, &
    fc_box_lin_validate, fc_box_lin_project, fc_box_lin_kkt_residual, &
    fc_box_lin_partition, fc_box_lin_trial_step
  use fortchuck_stiefel, only: &
    fc_stiefel_offsets, fc_stiefel_rgrad, fc_stiefel_proj_tangent, &
    fc_stiefel_inner, fc_stiefel_orth_err, fc_stiefel_orth_mat, &
    fc_cayley_setup, fc_cayley_retract, fc_cayley_transport_vec, fc_cayley_transport_batch, &
    fc_cayley_setup_branch_a, fc_cayley_retract_branch_a, &
    fc_cayley_transport_branch_a, fc_cayley_transport_batch_branch_a, &
    fc_cayley_setup_branch_b, fc_cayley_retract_branch_b, &
    fc_cayley_transport_branch_b, fc_cayley_transport_batch_branch_b
  implicit none (type, external)

  integer(i32) :: rng_state

  rng_state = 133742_i32

  call test_structs_and_kinds()
  call test_gemm_syrk_frob()
  call test_lu_and_right_solve()
  call test_gemm_and_lu_register_tiles()
  call test_geqrf_pos_properties()
  call test_box_overflow_and_partition()
  call test_box_diag_and_breakpoints()
  call test_box_lin_suite()
  call test_stiefel_geometry_and_cayley_parity()
  call test_cayley_transport_batch_parity()

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
    call check_true(FC_ERR_INFEASIBLE_LIN == -10_i32, 'FC_ERR_INFEASIBLE_LIN == -10')
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

  !---------------------------------------------------------------------------
  ! 7. v0.2.0 Odd/prime cache-blocked & 4x4/1x4 register-tiled GEMM and
  !    4-RHS division-hoisted LU / LU-transpose accuracy across
  !    m, n, k, nrhs in {1, 2, 3, 4, 5, 7, 16, 65, 131}
  !---------------------------------------------------------------------------
  subroutine check_gemm_case_slices(m, n, k, lda, ldb, ldc, A, ka, B, kb, C1, C2)
    integer(i32), intent(in) :: m, n, k, lda, ldb, ldc, ka, kb
    real(dp), intent(inout)  :: A(lda, ka), B(ldb, kb), C1(ldc, n), C2(ldc, n)

    integer(i32) :: i, j, l
    real(dp)     :: acc, diff_sq, ref_sq, d

    ! (a) fc_gemm_nn: beta = 0 and beta /= 0
    call fill_rand_mat(lda, ka, A, rng_state)
    call fill_rand_mat(ldb, kb, B, rng_state)
    C1 = -77.0_dp
    call fc_gemm_nn(m, n, k, 1.35_dp, A, lda, B, ldb, 0.0_dp, C1, ldc)
    diff_sq = 0.0_dp
    ref_sq  = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        acc = 0.0_dp
        do l = 1_i32, k
          acc = acc + A(i, l) * B(l, j)
        end do
        C2(i, j) = 1.35_dp * acc
        d        = C1(i, j) - C2(i, j)
        diff_sq  = diff_sq + d * d
        ref_sq   = ref_sq + C2(i, j) * C2(i, j)
      end do
    end do
    call check_true(sqrt(diff_sq) <= 1.0e-14_dp * (1.0_dp + sqrt(ref_sq)), &
      'fc_gemm_nn tiled odd/prime beta=0')

    call fill_rand_mat(ldc, n, C1, rng_state)
    C2 = C1
    call fc_gemm_nn(m, n, k, -0.85_dp, A, lda, B, ldb, 0.45_dp, C1, ldc)
    diff_sq = 0.0_dp
    ref_sq  = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        acc = 0.0_dp
        do l = 1_i32, k
          acc = acc + A(i, l) * B(l, j)
        end do
        C2(i, j) = 0.45_dp * C2(i, j) - 0.85_dp * acc
        d        = C1(i, j) - C2(i, j)
        diff_sq  = diff_sq + d * d
        ref_sq   = ref_sq + C2(i, j) * C2(i, j)
      end do
    end do
    call check_true(sqrt(diff_sq) <= 1.0e-14_dp * (1.0_dp + sqrt(ref_sq)), &
      'fc_gemm_nn tiled odd/prime beta/=0')

    ! (b) fc_gemm_tn: A is k x m (lda >= k), B is k x n (ldb >= k)
    call fill_rand_mat(lda, ka, A, rng_state)
    call fill_rand_mat(ldb, kb, B, rng_state)
    C1 = 42.0_dp
    call fc_gemm_tn(m, n, k, 1.75_dp, A, lda, B, ldb, 0.0_dp, C1, ldc)
    diff_sq = 0.0_dp
    ref_sq  = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        acc = 0.0_dp
        do l = 1_i32, k
          acc = acc + A(l, i) * B(l, j)
        end do
        C2(i, j) = 1.75_dp * acc
        d        = C1(i, j) - C2(i, j)
        diff_sq  = diff_sq + d * d
        ref_sq   = ref_sq + C2(i, j) * C2(i, j)
      end do
    end do
    call check_true(sqrt(diff_sq) <= 1.0e-14_dp * (1.0_dp + sqrt(ref_sq)), &
      'fc_gemm_tn 4x4 micro-kernel odd/prime beta=0')

    call fill_rand_mat(ldc, n, C1, rng_state)
    C2 = C1
    call fc_gemm_tn(m, n, k, -1.15_dp, A, lda, B, ldb, -0.6_dp, C1, ldc)
    diff_sq = 0.0_dp
    ref_sq  = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        acc = 0.0_dp
        do l = 1_i32, k
          acc = acc + A(l, i) * B(l, j)
        end do
        C2(i, j) = -0.6_dp * C2(i, j) - 1.15_dp * acc
        d        = C1(i, j) - C2(i, j)
        diff_sq  = diff_sq + d * d
        ref_sq   = ref_sq + C2(i, j) * C2(i, j)
      end do
    end do
    call check_true(sqrt(diff_sq) <= 1.0e-14_dp * (1.0_dp + sqrt(ref_sq)), &
      'fc_gemm_tn 4x4 micro-kernel odd/prime beta/=0')

    ! (c) fc_gemm_nt: A is m x k (lda >= m), B is n x k (ldb >= n)
    call fill_rand_mat(lda, ka, A, rng_state)
    call fill_rand_mat(ldb, kb, B, rng_state)
    call fill_rand_mat(ldc, n, C1, rng_state)
    C2 = C1
    call fc_gemm_nt(m, n, k, 0.95_dp, A, lda, B, ldb, -0.35_dp, C1, ldc)
    diff_sq = 0.0_dp
    ref_sq  = 0.0_dp
    do j = 1_i32, n
      do i = 1_i32, m
        acc = 0.0_dp
        do l = 1_i32, k
          acc = acc + A(i, l) * B(j, l)
        end do
        C2(i, j) = -0.35_dp * C2(i, j) + 0.95_dp * acc
        d        = C1(i, j) - C2(i, j)
        diff_sq  = diff_sq + d * d
        ref_sq   = ref_sq + C2(i, j) * C2(i, j)
      end do
    end do
    call check_true(sqrt(diff_sq) <= 1.0e-14_dp * (1.0_dp + sqrt(ref_sq)), &
      'fc_gemm_nt 1x4 micro-kernel odd/prime beta/=0')
  end subroutine check_gemm_case_slices

  subroutine check_lu_panel_case(n, nrhs, A, LU, B, X_sol, Prod, ipiv)
    integer(i32), intent(in)  :: n, nrhs
    real(dp), intent(inout)   :: A(n, n), LU(n, n), B(n, nrhs), X_sol(n, nrhs), Prod(n, nrhs)
    integer(i32), intent(out) :: ipiv(n)

    integer(i32) :: i, j, r1, r2, info
    real(dp)     :: tmp_val, scale_ref

    call fill_rand_mat(n, n, A, rng_state)
    do i = 1_i32, n
      A(i, i) = A(i, i) + 3.0_dp + 0.25_dp * sqrt(real(n, dp))
    end do
    ! Cyclically swap several row pairs to force nontrivial partial-pivoting ipiv swaps
    if (n >= 2_i32) then
      do i = 1_i32, min(n / 2_i32, 8_i32)
        r1 = i
        r2 = n - i + 1_i32
        do j = 1_i32, n
          tmp_val   = A(r1, j)
          A(r1, j)  = A(r2, j)
          A(r2, j)  = tmp_val
        end do
      end do
    end if

    LU = A
    call fc_lu_factor(n, LU, ipiv, 1.0e-12_dp, info)
    call check_true(info == 0_i32, 'fc_lu_factor nonsingular across {1..131}')

    call fill_rand_mat(n, nrhs, B, rng_state)

    X_sol = B
    call fc_lu_solve(n, nrhs, LU, ipiv, X_sol)
    call fc_gemm_nn(n, nrhs, n, 1.0_dp, A, n, X_sol, n, 0.0_dp, Prod, n)
    scale_ref = fc_frob_norm(n, n, A) * fc_frob_norm(n, nrhs, X_sol) + fc_frob_norm(n, nrhs, B)
    call check_true(mat_diff_frob(n, nrhs, Prod, B) <= 1.0e-14_dp * scale_ref, &
      'fc_lu_solve 4-RHS panel residual <= 1e-14')

    X_sol = B
    call fc_lu_solve_trans(n, nrhs, LU, ipiv, X_sol)
    call fc_gemm_tn(n, nrhs, n, 1.0_dp, A, n, X_sol, n, 0.0_dp, Prod, n)
    scale_ref = fc_frob_norm(n, n, A) * fc_frob_norm(n, nrhs, X_sol) + fc_frob_norm(n, nrhs, B)
    call check_true(mat_diff_frob(n, nrhs, Prod, B) <= 1.0e-14_dp * scale_ref, &
      'fc_lu_solve_trans 4-RHS panel residual <= 1e-14')
  end subroutine check_lu_panel_case

  subroutine test_gemm_and_lu_register_tiles()
    integer(i32), parameter :: dims_all(9) = [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, &
                                              7_i32, 16_i32, 65_i32, 131_i32]
    integer(i32), parameter :: n_big = 14_i32
    integer(i32), parameter :: bm(n_big) = [65_i32, 65_i32, 5_i32, 131_i32, 7_i32, &
      65_i32, 131_i32, 131_i32, 1_i32, 131_i32, 131_i32, 4_i32, 16_i32, 65_i32]
    integer(i32), parameter :: bn(n_big) = [65_i32, 5_i32, 65_i32, 7_i32, 131_i32, &
      131_i32, 65_i32, 131_i32, 131_i32, 1_i32, 131_i32, 131_i32, 65_i32, 16_i32]
    integer(i32), parameter :: bk(n_big) = [65_i32, 131_i32, 131_i32, 65_i32, 65_i32, &
      7_i32, 5_i32, 131_i32, 131_i32, 131_i32, 1_i32, 65_i32, 131_i32, 131_i32]
    integer(i32), parameter :: max_buf = 140_i32 * 140_i32

    real(dp), allocatable :: A_buf(:), B_buf(:), C1_buf(:), C2_buf(:), P_buf(:)
    integer(i32)          :: ipiv(140)
    integer(i32)          :: im, in_idx, ik, idx, m, n, k, nrhs
    integer(i32)          :: ld_max, col_max, sz_ab, sz_c, n2, n_rhs_sz

    allocate(A_buf(max_buf), B_buf(max_buf), C1_buf(max_buf), C2_buf(max_buf), P_buf(max_buf))

    ! Exhaustive small/odd/prime/4x4 register-tile sweep over {1, 2, 3, 4, 5, 7, 16}
    do im = 1_i32, 7_i32
      m = dims_all(im)
      do in_idx = 1_i32, 7_i32
        n = dims_all(in_idx)
        do ik = 1_i32, 7_i32
          k       = dims_all(ik)
          ld_max  = max(m, max(n, k)) + mod(im + ik, 3_i32)
          col_max = max(m, max(n, k))
          sz_ab   = ld_max * col_max
          sz_c    = (m + mod(in_idx, 2_i32)) * n
          call check_gemm_case_slices(m, n, k, ld_max, ld_max, m + mod(in_idx, 2_i32), &
            A_buf(1:sz_ab), col_max, B_buf(1:sz_ab), col_max, C1_buf(1:sz_c), C2_buf(1:sz_c))
        end do
      end do
    end do

    ! L1/L2 cache-block boundary & large prime combinations across 65 and 131
    do idx = 1_i32, n_big
      m       = bm(idx)
      n       = bn(idx)
      k       = bk(idx)
      ld_max  = max(m, max(n, k)) + 3_i32
      col_max = max(m, max(n, k))
      sz_ab   = ld_max * col_max
      sz_c    = (m + 2_i32) * n
      call check_gemm_case_slices(m, n, k, ld_max, ld_max, m + 2_i32, &
        A_buf(1:sz_ab), col_max, B_buf(1:sz_ab), col_max, C1_buf(1:sz_c), C2_buf(1:sz_c))
    end do

    ! Multi-RHS LU and LU-transpose across all (n, nrhs) in {1, 2, 3, 4, 5, 7, 16, 65, 131}
    do im = 1_i32, 9_i32
      n  = dims_all(im)
      n2 = n * n
      do in_idx = 1_i32, 9_i32
        nrhs     = dims_all(in_idx)
        n_rhs_sz = n * nrhs
        call check_lu_panel_case(n, nrhs, &
          A_buf(1:n2), B_buf(1:n2), C1_buf(1:n_rhs_sz), C2_buf(1:n_rhs_sz), &
          P_buf(1:n_rhs_sz), ipiv(1:n))
      end do
    end do

    deallocate(A_buf, B_buf, C1_buf, C2_buf, P_buf)
  end subroutine test_gemm_and_lu_register_tiles

  !---------------------------------------------------------------------------
  ! 8. v0.2.0 Diagonal active-band step and overflow-safe free breakpoint cap
  !---------------------------------------------------------------------------
  subroutine test_box_diag_and_breakpoints()
    integer(i32), parameter :: d = 7_i32
    real(dp)     :: z(d), l(d), u(d), gz(d), diag_box(d), s_prev(d), y_prev(d), dz(d)
    real(dp)     :: tau_min, tau_max, pinf
    integer(i32) :: mask_box(d), n_hit

    pinf = ieee_value(1.0_dp, ieee_positive_inf)

    l        = [0.0_dp,  0.0_dp,  2.0_dp, -FC_HUGE,  0.0_dp, -1.0_dp,  0.0_dp]
    l(5)     = -pinf
    u        = [10.0_dp, 10.0_dp, 2.0_dp,  FC_HUGE,  5.0_dp,  FC_HUGE, 4.0_dp]
    z        = [1.0_dp,  0.0_dp,  2.0_dp,  0.0_dp,   1.0_dp,  0.5_dp,  1.0_dp]
    gz       = [2.0_dp,  3.0_dp, -5.0_dp,  1.0_dp,  -2.0_dp,  4.0_dp, -2.0_dp]
    diag_box = [0.25_dp, 50.0_dp, 1.0_dp,  1.0_dp,   0.001_dp, 0.5_dp, 0.8_dp]
    s_prev   = [0.2_dp,  0.0_dp,  0.0_dp,  0.0_dp,   0.0_dp,   0.3_dp, 0.4_dp]
    y_prev   = [0.1_dp,  0.0_dp,  0.0_dp,  0.0_dp,   0.0_dp,   0.6_dp, 0.2_dp]
    mask_box = [FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED, FC_BOX_FREE, &
                FC_BOX_ACTIVE, FC_BOX_ACTIVE, FC_BOX_FREE]
    dz       = [-0.5_dp, 99.0_dp, 99.0_dp, 3.0_dp, 99.0_dp, 99.0_dp, 1.5_dp]

    call fc_box_active_step_diag(d, z, l, u, gz, mask_box, diag_box, &
      0.05_dp, 10.0_dp, .true., s_prev, y_prev, dz)

    ! Coord 1 FREE: preserved (-0.5)
    call check_true(abs(dz(1) + 0.5_dp) <= 1.0e-15_dp, 'active_step_diag preserves FREE')
    ! Coord 2 ACTIVE, s_prev=0 -> uses clip(diag_box(2)=50, 0.05, 10.0) = 10.0 -> dz = -30.0
    call check_true(abs(dz(2) + 30.0_dp) <= 1.0e-15_dp, 'active_step_diag clips diag_box upper')
    ! Coord 3 FIXED -> 0.0
    call check_true(abs(dz(3)) <= 1.0e-16_dp, 'active_step_diag zeroes FIXED')
    ! Coord 5 ACTIVE, s_prev=0 -> uses clip(diag_box(5)=0.001, 0.05, 10.0) = 0.05 -> dz = +0.1
    call check_true(abs(dz(5) - 0.1_dp) <= 1.0e-15_dp, 'active_step_diag clips diag_box lower')
    ! Coord 6 ACTIVE, valid secant s/y = 0.3/0.6 = 0.5 -> dz = -0.5 * 4 = -2.0
    call check_true(abs(dz(6) + 2.0_dp) <= 1.0e-15_dp, 'active_step_diag uses secant ratio')

    ! Breakpoints across FREE coords {1, 4, 7}:
    !   Coord 1: z=1, l=0, dz=-0.5 -> tau = (0 - 1)/(-0.5) = 2.0
    !   Coord 4: z=0, u=+FC_HUGE, dz=+3 -> unbounded wall skipped!
    !   Coord 7: z=1, u=4, dz=+1.5 -> tau = (4 - 1)/1.5 = 2.0? Adjust dz(7)=0.5 -> tau = 6.0!
    dz(7) = 0.5_dp
    call fc_box_breakpoint_cap(d, z, l, u, mask_box, dz, tau_min, tau_max, n_hit)
    call check_true(n_hit == 2_i32, 'fc_box_breakpoint_cap counts finite free walls only')
    call check_true(abs(tau_min - 2.0_dp) <= 1.0e-15_dp, 'fc_box_breakpoint_cap tau_brk_min == 2')
    call check_true(abs(tau_max - 6.0_dp) <= 1.0e-15_dp, 'fc_box_breakpoint_cap tau_brk_max == 6')
  end subroutine test_box_diag_and_breakpoints

  !---------------------------------------------------------------------------
  ! 9. v0.2.0 Singly-linearly-constrained box & simplex projection, KKT,
  !    active-face partition, trial step, overflow guards, & variational ineq.
  !---------------------------------------------------------------------------
  subroutine assert_lin_proj_vi(d, l, u, a, b, y, z_star)
    integer(i32), intent(in) :: d
    real(dp), intent(in)     :: l(d), u(d), a(d), b, y(d), z_star(d)

    integer(i32) :: i, s_idx
    real(dp)     :: az, a_l1, z_inf, tol_eq, b_min, b_max
    real(dp)     :: z_lo(80), z_hi(80), w(80), z_feas(80)
    real(dp)     :: li_eff, ui_eff, u01, aw, theta, vi_dot, nrm_yz, nrm_dz

    az    = 0.0_dp
    a_l1  = 0.0_dp
    z_inf = 0.0_dp
    b_min = 0.0_dp
    b_max = 0.0_dp

    do i = 1_i32, d
      call check_true(z_star(i) >= l(i) - 1.0e-15_dp .and. z_star(i) <= u(i) + 1.0e-15_dp, &
        'fc_box_lin_project box feasibility l <= z* <= u')
      az    = az + a(i) * z_star(i)
      a_l1  = a_l1 + abs(a(i))
      z_inf = max(z_inf, abs(z_star(i)))

      ! Finite sampling box around z_star for independent geometric feasible test points
      li_eff = max(l(i), z_star(i) - 10.0_dp)
      ui_eff = min(u(i), z_star(i) + 10.0_dp)
      if (a(i) > 0.0_dp) then
        z_lo(i) = li_eff
        z_hi(i) = ui_eff
        b_min   = b_min + a(i) * li_eff
        b_max   = b_max + a(i) * ui_eff
      else if (a(i) < 0.0_dp) then
        z_lo(i) = ui_eff
        z_hi(i) = li_eff
        b_min   = b_min + a(i) * ui_eff
        b_max   = b_max + a(i) * li_eff
      else
        z_lo(i) = 0.5_dp * (li_eff + ui_eff)
        z_hi(i) = z_lo(i)
      end if
    end do

    tol_eq = 1.0e-14_dp * (1.0_dp + abs(b) + a_l1 * z_inf)
    call check_true(abs(az - b) <= tol_eq, &
      'fc_box_lin_project equality |a^T z* - b| <= 1e-14*(1+|b|+||a||_1*||z*||_inf)')

    ! Independent geometric convex-combination feasible points z_feas in C (0 calls to projector!)
    do s_idx = 1_i32, 20_i32
      aw = 0.0_dp
      do i = 1_i32, d
        u01  = 0.5_dp * (next_rand(rng_state) + 1.0_dp)
        w(i) = min(z_lo(i), z_hi(i)) + u01 * abs(z_hi(i) - z_lo(i))
        aw   = aw + a(i) * w(i)
      end do
      if (aw >= b .and. aw > b_min + 1.0e-15_dp) then
        theta = min(1.0_dp, max(0.0_dp, (b - b_min) / (aw - b_min)))
        do i = 1_i32, d
          z_feas(i) = theta * w(i) + (1.0_dp - theta) * z_lo(i)
        end do
      else if (aw < b .and. b_max > aw + 1.0e-15_dp) then
        theta = min(1.0_dp, max(0.0_dp, (b_max - b) / (b_max - aw)))
        do i = 1_i32, d
          z_feas(i) = theta * w(i) + (1.0_dp - theta) * z_hi(i)
        end do
      else
        do i = 1_i32, d
          z_feas(i) = z_lo(i)
        end do
      end if

      vi_dot = 0.0_dp
      nrm_yz = 0.0_dp
      nrm_dz = 0.0_dp
      do i = 1_i32, d
        vi_dot = vi_dot + (y(i) - z_star(i)) * (z_feas(i) - z_star(i))
        nrm_yz = nrm_yz + (y(i) - z_star(i)) * (y(i) - z_star(i))
        nrm_dz = nrm_dz + (z_feas(i) - z_star(i)) * (z_feas(i) - z_star(i))
      end do
      call check_true(vi_dot <= 1.0e-14_dp * (1.0_dp + sqrt(nrm_yz * nrm_dz)), &
        'fc_box_lin_project variational inequality <y - z*, z - z*> <= 1e-14')
    end do
  end subroutine assert_lin_proj_vi

  subroutine test_box_lin_suite()
    real(dp)     :: l75(75), u75(75), a75(75), y75(75), z75(75)
    real(dp)     :: l8(8), u8(8), a8(8), y8(8), z8(8), gz8(8), zp8(8), rb8(8), dz8(8), zt8(8)
    real(dp)     :: nu_out, b_val, b_lo, b_hi, pinf, qnan, aa, ay
    real(dp)     :: box_inf, box_l2, delta_b, chord_gz, step_sq, chk_sum
    integer(i32) :: mask8(8), status, n_act, i, inst

    pinf = ieee_value(1.0_dp, ieee_positive_inf)
    qnan = ieee_value(1.0_dp, ieee_quiet_nan)

    ! 1. Unit simplex d = 75 (interior identity + exterior sparse projection)
    l75 = 0.0_dp
    u75 = 1.0_dp
    a75 = 1.0_dp
    chk_sum = 0.0_dp
    do i = 1_i32, 75_i32
      y75(i)  = real(i, dp)
      chk_sum = chk_sum + y75(i)
    end do
    y75 = y75 / chk_sum
    z75 = y75
    call fc_box_lin_validate(75_i32, l75, u75, a75, 1.0_dp, status)
    call check_true(status == 0_i32, 'fc_box_lin_validate unit simplex d=75')
    call fc_box_lin_project(75_i32, l75, u75, a75, 1.0_dp, z75, nu_out, status)
    call check_true(status == 0_i32, 'fc_box_lin_project interior simplex status==0')
    call check_true(mat_diff_frob(75_i32, 1_i32, z75, y75) <= 1.0e-14_dp, &
      'fc_box_lin_project interior simplex fixed point')

    do i = 1_i32, 75_i32
      y75(i) = 4.0_dp * next_rand(rng_state)
    end do
    z75 = y75
    call fc_box_lin_project(75_i32, l75, u75, a75, 1.0_dp, z75, nu_out, status)
    call check_true(status == 0_i32, 'fc_box_lin_project exterior simplex status==0')
    call assert_lin_proj_vi(75_i32, l75, u75, a75, 1.0_dp, y75, z75)

    ! 2. Unbounded & one-sided +-FC_HUGE / +-Inf with a_i == 0 under -ffpe-trap=overflow
    !    (a) Full unbounded hyperplane: exact orthogonal projection y - ((a^T y - b)/||a||^2)*a
    l8 = -FC_HUGE
    u8 =  FC_HUGE
    a8 = [2.0_dp, -1.5_dp, 0.0_dp, 4.0_dp, -0.5_dp, 0.0_dp, 1.25_dp, -3.0_dp]
    y8 = [5.0_dp, -3.0_dp, 9.0_dp, 1.0_dp,  2.0_dp, -8.0_dp, 0.5_dp,  4.0_dp]
    b_val = 3.75_dp
    z8 = y8
    call fc_box_lin_validate(8_i32, l8, u8, a8, b_val, status)
    call check_true(status == 0_i32, 'fc_box_lin_validate unbounded +-FC_HUGE with a_i=0')
    call fc_box_lin_project(8_i32, l8, u8, a8, b_val, z8, nu_out, status)
    call check_true(status == 0_i32, 'fc_box_lin_project unbounded hyperplane status==0')
    aa = 0.0_dp
    ay = 0.0_dp
    do i = 1_i32, 8_i32
      aa = aa + a8(i) * a8(i)
      ay = ay + a8(i) * y8(i)
    end do
    do i = 1_i32, 8_i32
      zp8(i) = y8(i) - ((ay - b_val) / aa) * a8(i)
    end do
    call check_true(mat_diff_frob(8_i32, 1_i32, z8, zp8) <= 1.0e-14_dp, &
      'fc_box_lin_project exact orthogonal hyperplane projection')

    !    (b) One-sided [0, +FC_HUGE) & (-Inf, u] with a_i == 0 and +-FC_HUGE bounds
    l8 = [0.0_dp, -FC_HUGE, -FC_HUGE,  0.0_dp,  0.0_dp,  1.5_dp,  0.0_dp, -2.0_dp]
    l8(5) = -pinf
    u8 = [FC_HUGE, 2.0_dp,   FC_HUGE,  0.0_dp,  3.0_dp,  1.5_dp,  4.0_dp,  2.0_dp]
    u8(4) = pinf
    a8 = [1.5_dp, -2.0_dp,   0.0_dp,   0.75_dp, -1.0_dp, 2.0_dp, 0.5_dp, -1.25_dp]
    y8 = [-4.0_dp, 7.0_dp,   3.5_dp,  -2.0_dp,  8.0_dp, 9.0_dp,  6.0_dp, -5.0_dp]
    b_val = 6.5_dp
    z8 = y8
    call fc_box_lin_validate(8_i32, l8, u8, a8, b_val, status)
    call check_true(status == 0_i32, 'fc_box_lin_validate one-sided +-FC_HUGE/Inf')
    call fc_box_lin_project(8_i32, l8, u8, a8, b_val, z8, nu_out, status)
    call check_true(status == 0_i32, 'fc_box_lin_project one-sided +-FC_HUGE/Inf')
    call check_true(abs(z8(3) - 3.5_dp) <= 1.0e-15_dp, 'a_i==0 coord keeps clamped y_i')
    call check_true(abs(z8(6) - 1.5_dp) <= 1.0e-15_dp, 'fixed l_i==u_i coord exact')
    call assert_lin_proj_vi(8_i32, l8, u8, a8, b_val, y8, z8)

    ! 3. Singleton boundary faces b == b_min and b == b_max with signed a_i and a_i == 0
    l8 = [-1.0_dp, 0.0_dp, -2.0_dp,  0.5_dp, -3.0_dp, 1.0_dp, -0.5_dp, 0.0_dp]
    u8 = [ 2.0_dp, 3.0_dp,  4.0_dp,  2.5_dp,  1.0_dp, 1.0_dp,  1.5_dp, 2.0_dp]
    a8 = [ 2.0_dp, -1.5_dp, 0.0_dp,  3.0_dp, -0.5_dp, 4.0_dp,  1.0_dp, -2.0_dp]
    b_lo = 0.0_dp
    b_hi = 0.0_dp
    do i = 1_i32, 8_i32
      if (a8(i) > 0.0_dp) then
        b_lo = b_lo + a8(i) * l8(i)
        b_hi = b_hi + a8(i) * u8(i)
      else if (a8(i) < 0.0_dp) then
        b_lo = b_lo + a8(i) * u8(i)
        b_hi = b_hi + a8(i) * l8(i)
      end if
    end do
    y8 = [0.7_dp, 1.1_dp, 9.0_dp, 1.2_dp, -0.4_dp, 1.0_dp, 0.2_dp, 0.9_dp]
    z8 = y8
    call fc_box_lin_project(8_i32, l8, u8, a8, b_lo, z8, nu_out, status)
    call check_true(status == 0_i32, 'singleton b == b_min status==0')
    call check_true(abs(z8(1) - l8(1)) <= 1.0e-15_dp .and. abs(z8(2) - u8(2)) <= 1.0e-15_dp, &
      'singleton b == b_min hits signed lower/upper vertex')
    call check_true(abs(z8(3) - u8(3)) <= 1.0e-15_dp, &
      'singleton b == b_min clamps a_i==0 coord to [l_i, u_i]')
    call assert_lin_proj_vi(8_i32, l8, u8, a8, b_lo, y8, z8)

    z8 = y8
    call fc_box_lin_project(8_i32, l8, u8, a8, b_hi, z8, nu_out, status)
    call check_true(status == 0_i32, 'singleton b == b_max status==0')
    call check_true(abs(z8(1) - u8(1)) <= 1.0e-15_dp .and. abs(z8(2) - l8(2)) <= 1.0e-15_dp, &
      'singleton b == b_max hits signed upper/lower vertex')
    call assert_lin_proj_vi(8_i32, l8, u8, a8, b_hi, y8, z8)

    ! 4. Infeasible and invalid error returns (FC_ERR_INFEASIBLE_LIN = -10)
    call fc_box_lin_validate(8_i32, l8, u8, a8, b_lo - 1.0e-4_dp, status)
    call check_true(status == FC_ERR_INFEASIBLE_LIN, 'rejects b < b_min with FC_ERR_INFEASIBLE_LIN')
    z8 = y8
    call fc_box_lin_project(8_i32, l8, u8, a8, b_hi + 1.0e-4_dp, z8, nu_out, status)
    call check_true(status == FC_ERR_INFEASIBLE_LIN, 'project rejects b > b_max')
    zp8 = 0.0_dp
    call fc_box_lin_validate(8_i32, l8, u8, zp8, 0.0_dp, status)
    call check_true(status == FC_ERR_INFEASIBLE_LIN, 'rejects zero normal vector a == 0')
    call fc_box_lin_validate(8_i32, l8, u8, a8, qnan, status)
    call check_true(status == FC_ERR_INFEASIBLE_LIN, 'rejects NaN rhs b')
    call fc_box_lin_validate(8_i32, l8, u8, a8, pinf, status)
    call check_true(status == FC_ERR_INFEASIBLE_LIN, 'rejects +Inf rhs b')
    call fc_box_lin_validate(0_i32, l8, u8, a8, 0.0_dp, status)
    call check_true(status == FC_ERR_INVALID_DIM, 'rejects d == 0 with FC_ERR_INVALID_DIM')

    ! 5. Randomized 30-instance signed knapsack & extreme-scale variational inequality sweep
    do inst = 1_i32, 30_i32
      b_lo = 0.0_dp
      b_hi = 0.0_dp
      do i = 1_i32, 25_i32
        l75(i) = 2.0_dp * next_rand(rng_state) - 1.0_dp
        u75(i) = l75(i) + 0.2_dp + 3.0_dp * abs(next_rand(rng_state))
        a75(i) = next_rand(rng_state) * (10.0_dp ** (2.0_dp * next_rand(rng_state)))
        if (mod(inst + i, 9_i32) == 0_i32) a75(i) = 0.0_dp
        y75(i) = 6.0_dp * next_rand(rng_state)
        if (a75(i) > 0.0_dp) then
          b_lo = b_lo + a75(i) * l75(i)
          b_hi = b_hi + a75(i) * u75(i)
        else if (a75(i) < 0.0_dp) then
          b_lo = b_lo + a75(i) * u75(i)
          b_hi = b_hi + a75(i) * l75(i)
        end if
      end do
      b_val = b_lo + (0.05_dp + 0.90_dp * 0.5_dp * (next_rand(rng_state) + 1.0_dp)) * (b_hi - b_lo)
      z75(1:25) = y75(1:25)
      call fc_box_lin_project(25_i32, l75(1:25), u75(1:25), a75(1:25), b_val, &
        z75(1:25), nu_out, status)
      call check_true(status == 0_i32, 'random signed knapsack status == 0')
      call assert_lin_proj_vi(25_i32, l75(1:25), u75(1:25), a75(1:25), b_val, &
        y75(1:25), z75(1:25))
    end do

    ! 6. fc_box_lin_kkt_residual, fc_box_lin_partition, & fc_box_lin_trial_step
    !    At z* = P_C(y), minimizing 0.5*||z - y||^2 has gradient gz = z* - y -> KKT residual == 0!
    z8 = y8
    b_val = 0.5_dp * (b_lo + b_hi)
    ! Recompute b_lo, b_hi for l8, u8, a8
    b_lo = 0.0_dp
    b_hi = 0.0_dp
    do i = 1_i32, 8_i32
      if (a8(i) > 0.0_dp) then
        b_lo = b_lo + a8(i) * l8(i)
        b_hi = b_hi + a8(i) * u8(i)
      else if (a8(i) < 0.0_dp) then
        b_lo = b_lo + a8(i) * u8(i)
        b_hi = b_hi + a8(i) * l8(i)
      end if
    end do
    b_val = 0.4_dp * b_lo + 0.6_dp * b_hi
    z8 = [4.0_dp, -3.0_dp, 1.5_dp, 5.0_dp, -4.0_dp, 1.0_dp, 2.5_dp, -1.0_dp]
    zp8 = z8
    call fc_box_lin_project(8_i32, l8, u8, a8, b_val, z8, nu_out, status)
    call check_true(status == 0_i32, 'KKT setup projection status==0')
    do i = 1_i32, 8_i32
      gz8(i) = z8(i) - zp8(i)
    end do
    call fc_box_lin_kkt_residual(8_i32, z8, l8, u8, a8, b_val, gz8, &
      zt8, rb8, nu_out, box_inf, box_l2, status)
    call check_true(status == 0_i32, 'fc_box_lin_kkt_residual status==0')
    call check_true(box_inf <= 1.0e-14_dp .and. box_l2 <= 1.0e-14_dp, &
      'fc_box_lin_kkt_residual vanishes at exact constrained optimizer z*')

    ! Partition with tiny boundary gap snap (<= 1e-12) and SPG trial step verification
    l8  = 0.0_dp
    u8  = 1.0_dp
    a8  = 1.0_dp
    b_val = 1.0_dp
    z8  = [1.0e-13_dp, 0.5_dp - 1.0e-13_dp, 0.5_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp]
    gz8 = [5.0_dp, -2.0_dp, 1.0_dp, 4.0_dp, 3.0_dp, 6.0_dp, 2.0_dp, 2.5_dp]
    call fc_box_lin_kkt_residual(8_i32, z8, l8, u8, a8, b_val, gz8, &
      zp8, rb8, nu_out, box_inf, box_l2, status)
    call check_true(status == 0_i32 .and. box_inf > 0.1_dp, &
      'fc_box_lin_kkt_residual positive at non-stationary simplex point')
    call fc_box_lin_partition(8_i32, z8, l8, u8, a8, gz8, nu_out, mask8, dz8, delta_b, n_act)
    call check_true(mask8(1) == FC_BOX_ACTIVE, 'fc_box_lin_partition snaps 1e-13 lower gap ACTIVE')
    call check_true(abs(dz8(1) + 1.0e-13_dp) <= 1.0e-25_dp, &
      'fc_box_lin_partition sets exact snap step l_i - z_i')
    call check_true(abs(delta_b - 1.0e-13_dp) <= 1.0e-25_dp, &
      'fc_box_lin_partition delta_b == -sum_{active} a_i * dz_i')
    call check_true(mask8(2) == FC_BOX_FREE .and. mask8(3) == FC_BOX_FREE, &
      'fc_box_lin_partition keeps interior simplex coords FREE')

    dz8 = -gz8
    call fc_box_lin_trial_step(8_i32, z8, l8, u8, a8, b_val, gz8, dz8, 0.25_dp, &
      zt8, chord_gz, step_sq, status)
    call check_true(status == 0_i32, 'fc_box_lin_trial_step status==0')
    call check_true(chord_gz < -1.0e-3_dp .and. step_sq > 1.0e-4_dp, &
      'fc_box_lin_trial_step strict descent along SPG trial curve')
    chk_sum = 0.0_dp
    do i = 1_i32, 8_i32
      chk_sum = chk_sum + zt8(i)
    end do
    call check_true(abs(chk_sum - 1.0_dp) <= 1.0e-14_dp, &
      'fc_box_lin_trial_step preserves simplex equality sum(z_trial)==1')
  end subroutine test_box_lin_suite

  !---------------------------------------------------------------------------
  ! 10. v0.2.0 Batched Level-3 Isometric Cayley Panel Transport Parity & Isometry
  !     across (n, p) in {(10,1),(12,3),(8,4),(7,5),(6,6),(40,6),(16,16)}
  !     and k_batch in {1, 5, 16} (+ k_batch = 0 and tau = 0 no-ops)
  !---------------------------------------------------------------------------
  subroutine verify_batch_transport_case(n, p, k_batch, X, eta, Y_a, Y_b, Y_auto, &
      A_sk, scr_p2, XTz, YTza, Panel_init, Panel_ref, Panel_a, Panel_b, Panel_auto, &
      cay_a, cay_b, cay_auto, scr_big)
    integer(i32), intent(in) :: n, p, k_batch
    real(dp), intent(inout)  :: X(n, p), eta(n, p), Y_a(n, p), Y_b(n, p), Y_auto(n, p)
    real(dp), intent(inout)  :: A_sk(p, p), scr_p2(p, p), XTz(p, p), YTza(p, p)
    real(dp), intent(inout)  :: Panel_init(n, p * k_batch), Panel_ref(n, p * k_batch)
    real(dp), intent(inout)  :: Panel_a(n, p * k_batch), Panel_b(n, p * k_batch)
    real(dp), intent(inout)  :: Panel_auto(n, p * k_batch)
    real(dp), intent(inout)  :: cay_a(4096), cay_b(4096), cay_auto(4096), scr_big(*)

    real(dp), parameter :: taus(4) = [0.0_dp, 0.05_dp, 0.7_dp, 2.3_dp]
    real(dp)            :: tau_qr(20), work_qr(20), scr_p1(1)
    integer(i32)        :: ipiv_a(40), ipiv_b(40), ipiv_auto(40)
    integer(i32)        :: ncol, info_a, info_b, info_auto, t_idx, b_idx, b_next, c0, d0
    real(dp)            :: tau, scale_z, in_can_x, in_can_y, in_euc_x, in_euc_y

    ncol = p * k_batch
    call fill_rand_mat(n, p, X, rng_state)
    call fc_geqrf_pos(n, p, X, tau_qr(1:p), work_qr(1:p), info_a)
    call check_true(info_a == 0_i32, 'batch transport base frame QR')

    call fill_rand_mat(n, p, eta, rng_state)
    call fc_stiefel_proj_tangent(n, p, X, eta, A_sk, scr_p2)

    ! Generate k_batch normalized tangent blocks in Panel_init
    do b_idx = 1_i32, k_batch
      c0 = (b_idx - 1_i32) * p + 1_i32
      call fill_rand_mat(n, p, Panel_init(1:n, c0:c0 + p - 1_i32), rng_state)
      call fc_stiefel_proj_tangent(n, p, X, Panel_init(1:n, c0:c0 + p - 1_i32), A_sk, scr_p2)
      scale_z = max(1.0_dp, fc_frob_norm(n, p, Panel_init(1:n, c0:c0 + p - 1_i32)))
      Panel_init(1:n, c0:c0 + p - 1_i32) = Panel_init(1:n, c0:c0 + p - 1_i32) / &
        (scale_z * sqrt(real(k_batch, dp)))
    end do

    call fc_cayley_setup_branch_a(n, p, X, eta, cay_a, scr_big)
    call fc_cayley_setup_branch_b(n, p, X, eta, cay_b, scr_big)
    call fc_cayley_setup(n, p, X, eta, cay_auto, scr_big)

    ! Verify k_batch == 0 is a strict no-op
    Panel_auto = Panel_init
    call fc_cayley_transport_batch(n, p, 0_i32, X, eta, 0.7_dp, &
      cay_auto, ipiv_auto, Panel_auto, scr_big)
    call check_true(mat_diff_frob(n, ncol, Panel_auto, Panel_init) <= 0.0_dp, &
      'fc_cayley_transport_batch k_batch==0 exact no-op')

    do t_idx = 1_i32, 4_i32
      tau = taus(t_idx)
      call fc_cayley_retract_branch_a(n, p, X, eta, tau, cay_a, ipiv_a(1:p), Y_a, scr_big, info_a)
      call fc_cayley_retract_branch_b(n, p, X, eta, tau, cay_b, ipiv_b(1:n), Y_b, scr_big, info_b)
      call fc_cayley_retract(n, p, X, eta, tau, cay_auto, ipiv_auto, Y_auto, scr_big, info_auto)
      call check_true(info_a == 0_i32 .and. info_b == 0_i32 .and. info_auto == 0_i32, &
        'batch transport retract info == 0')
      call check_true(mat_diff_frob(n, p, Y_a, Y_b) <= 1.0e-14_dp, &
        'batch transport retract Branch A == Branch B <= 1e-14')

      ! Reference: k_batch sequential calls to fc_cayley_transport_vec
      Panel_ref = Panel_init
      do b_idx = 1_i32, k_batch
        c0 = (b_idx - 1_i32) * p + 1_i32
        call fc_cayley_transport_vec(n, p, X, eta, tau, cay_auto, ipiv_auto, &
          Panel_ref(1:n, c0:c0 + p - 1_i32), scr_big)
      end do

      Panel_a    = Panel_init
      Panel_b    = Panel_init
      Panel_auto = Panel_init

      if (p == 1_i32) then
        scr_p1(1) = 777.0_dp
        call fc_cayley_transport_batch_branch_a(n, p, k_batch, X, eta, tau, &
          cay_a, ipiv_a(1:p), Panel_a, scr_p1)
        call check_true(.not. (abs(scr_p1(1) - 777.0_dp) > 0.0_dp), &
          'fc_cayley_transport_batch_branch_a p==1 uses 0 scratch words')
      else
        call fc_cayley_transport_batch_branch_a(n, p, k_batch, X, eta, tau, &
          cay_a, ipiv_a(1:p), Panel_a, scr_big)
      end if

      call fc_cayley_transport_batch_branch_b(n, p, k_batch, X, eta, tau, &
        cay_b, ipiv_b(1:n), Panel_b, scr_big)
      call fc_cayley_transport_batch(n, p, k_batch, X, eta, tau, &
        cay_auto, ipiv_auto, Panel_auto, scr_big)

      call check_true(mat_diff_frob(n, ncol, Panel_a, Panel_ref) <= 1.0e-14_dp, &
        'fc_cayley_transport_batch_branch_a == k_batch x single <= 1e-14')
      call check_true(mat_diff_frob(n, ncol, Panel_b, Panel_ref) <= 1.0e-14_dp, &
        'fc_cayley_transport_batch_branch_b == k_batch x single <= 1e-14')
      call check_true(mat_diff_frob(n, ncol, Panel_auto, Panel_ref) <= 1.0e-14_dp, &
        'fc_cayley_transport_batch auto == k_batch x single <= 1e-14')

      if (.not. (abs(tau) > 0.0_dp)) then
        call check_true(mat_diff_frob(n, ncol, Panel_auto, Panel_init) <= 0.0_dp, &
          'fc_cayley_transport_batch tau==0 exact no-op')
      end if

      ! Verify canonical & Euclidean inner-product isometry + skew invariance across batch
      do b_idx = 1_i32, k_batch
        b_next = mod(b_idx, k_batch) + 1_i32
        c0     = (b_idx - 1_i32) * p + 1_i32
        d0     = (b_next - 1_i32) * p + 1_i32

        call fc_gemm_tn(p, p, n, 1.0_dp, X, n, Panel_init(1:n, c0:c0 + p - 1_i32), n, &
          0.0_dp, XTz, p)
        call fc_gemm_tn(p, p, n, 1.0_dp, Y_auto, n, Panel_auto(1:n, c0:c0 + p - 1_i32), n, &
          0.0_dp, YTza, p)
        call check_true(mat_diff_frob(p, p, YTza, XTz) <= 1.0e-14_dp, &
          'batched Cayley transport preserves Y^T*T(zeta) == X^T*zeta')

        in_can_x = fc_stiefel_inner(n, p, X, &
          Panel_init(1:n, c0:c0 + p - 1_i32), Panel_init(1:n, d0:d0 + p - 1_i32), &
          FC_METRIC_CANONICAL)
        in_can_y = fc_stiefel_inner(n, p, Y_auto, &
          Panel_auto(1:n, c0:c0 + p - 1_i32), Panel_auto(1:n, d0:d0 + p - 1_i32), &
          FC_METRIC_CANONICAL)
        call check_true(abs(in_can_y - in_can_x) <= 1.0e-14_dp, &
          'batched Cayley transport canonical inner-product isometry <= 1e-14')

        in_euc_x = fc_stiefel_inner(n, p, X, &
          Panel_init(1:n, c0:c0 + p - 1_i32), Panel_init(1:n, d0:d0 + p - 1_i32), &
          FC_METRIC_EUCLIDEAN)
        in_euc_y = fc_stiefel_inner(n, p, Y_auto, &
          Panel_auto(1:n, c0:c0 + p - 1_i32), Panel_auto(1:n, d0:d0 + p - 1_i32), &
          FC_METRIC_EUCLIDEAN)
        call check_true(abs(in_euc_y - in_euc_x) <= 1.0e-14_dp, &
          'batched Cayley transport Euclidean inner-product isometry <= 1e-14')
      end do
    end do
  end subroutine verify_batch_transport_case

  subroutine test_cayley_transport_batch_parity()
    integer(i32), parameter :: n_cases = 7_i32
    integer(i32), parameter :: ns(n_cases) = [10_i32, 12_i32, 8_i32, 7_i32, 6_i32, 40_i32, 16_i32]
    integer(i32), parameter :: ps(n_cases) = [ 1_i32,  3_i32, 4_i32, 5_i32, 6_i32,  6_i32, 16_i32]
    integer(i32), parameter :: k_list(3)   = [ 1_i32,  5_i32, 16_i32]

    real(dp), allocatable :: np_buf(:,:), p2_buf(:,:), pan_buf(:,:)
    real(dp), allocatable :: cay_a(:), cay_b(:), cay_auto(:), scr_big(:)
    integer(i32)          :: c_idx, k_idx, n, p, k_batch, np, p2, n_pan

    ! Max np = 40*6 = 240 or 16*16 = 256; max p2 = 256; max n_pan = max(40*6*16=3840, 16*16*16=4096)
    ! Max scr_big = max(2*16*256 + 256 = 8448, 16*256 = 4096, 40*96 = 3840) -> 16384
    allocate(np_buf(256, 5), p2_buf(256, 4), pan_buf(4096, 5))
    allocate(cay_a(4096), cay_b(4096), cay_auto(4096), scr_big(16384))

    do c_idx = 1_i32, n_cases
      n  = ns(c_idx)
      p  = ps(c_idx)
      np = n * p
      p2 = p * p
      do k_idx = 1_i32, 3_i32
        k_batch = k_list(k_idx)
        n_pan   = np * k_batch
        call verify_batch_transport_case(n, p, k_batch, &
          np_buf(1:np, 1), np_buf(1:np, 2), np_buf(1:np, 3), np_buf(1:np, 4), np_buf(1:np, 5), &
          p2_buf(1:p2, 1), p2_buf(1:p2, 2), p2_buf(1:p2, 3), p2_buf(1:p2, 4), &
          pan_buf(1:n_pan, 1), pan_buf(1:n_pan, 2), pan_buf(1:n_pan, 3), &
          pan_buf(1:n_pan, 4), pan_buf(1:n_pan, 5), &
          cay_a, cay_b, cay_auto, scr_big)
      end do
    end do

    deallocate(np_buf, p2_buf, pan_buf, cay_a, cay_b, cay_auto, scr_big)
  end subroutine test_cayley_transport_batch_parity

end program test_linalg_and_stiefel
