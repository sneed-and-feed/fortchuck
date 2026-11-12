module omp_reentrancy_callbacks
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_double, c_ptr, c_f_pointer
  use fortchuck, only: dp, i32
  implicit none (type, external)
  public

  ! Per-instance problem parameters for [l, u]^8 x St(12, 3) x St(4, 4)
  type :: omp_prob_ctx_t
    real(dp) :: z_targ(8)
    real(dp) :: z_w(8)
    real(dp) :: st1_diag(12, 3)
    real(dp) :: st2_targ(4, 4)
    real(dp) :: cpl_alpha
  end type omp_prob_ctx_t

contains

  function cb_omp_coupled(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(omp_prob_ctx_t), pointer :: ctx
    integer(i32)                  :: i, j, idx
    real(dp)                      :: diff_i, xij, z_mod, s_x1, s_x2, alpha

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue

    call c_f_pointer(udata, ctx)

    f_out = 0.0_dp
    do i = 1_i32, 8_i32
      diff_i    = z(i) - ctx%z_targ(i)
      f_out     = f_out + 0.5_dp * ctx%z_w(i) * diff_i * diff_i
      gz_out(i) = ctx%z_w(i) * diff_i
    end do

    ! Factor 1: St(12, 3) (Branch A: 2*3 < 12) coupled with z(1..3)
    idx  = 0_i32
    s_x1 = 0.0_dp
    do j = 1_i32, 3_i32
      z_mod = 0.5_dp + z(j)
      do i = 1_i32, 12_i32
        idx                = idx + 1_i32
        xij                = X_packed(idx)
        f_out              = f_out + 0.5_dp * z_mod * ctx%st1_diag(i, j) * xij * xij
        gz_out(j)          = gz_out(j) + 0.5_dp * ctx%st1_diag(i, j) * xij * xij
        GX_packed_out(idx) = z_mod * ctx%st1_diag(i, j) * xij
      end do
      s_x1 = s_x1 + X_packed((j - 1_i32) * 12_i32 + j)
    end do

    ! Factor 2: St(4, 4) (Branch B: 2*4 >= 4) coupled with z(5..8)
    idx  = 36_i32
    s_x2 = 0.0_dp
    do j = 1_i32, 4_i32
      z_mod = 0.75_dp + 0.5_dp * z(4_i32 + j)
      do i = 1_i32, 4_i32
        idx                = idx + 1_i32
        diff_i             = X_packed(idx) - ctx%st2_targ(i, j)
        f_out              = f_out + 0.5_dp * z_mod * diff_i * diff_i
        gz_out(4_i32 + j)  = gz_out(4_i32 + j) + 0.25_dp * diff_i * diff_i
        GX_packed_out(idx) = z_mod * diff_i
      end do
      s_x2 = s_x2 + X_packed(36_i32 + (j - 1_i32) * 4_i32 + j)
    end do

    ! Trilinear coupling across z(4), St(12, 3), and St(4, 4)
    alpha     = ctx%cpl_alpha
    f_out     = f_out + alpha * z(4) * s_x1 * s_x2
    gz_out(4) = gz_out(4) + alpha * s_x1 * s_x2
    do j = 1_i32, 3_i32
      idx                = (j - 1_i32) * 12_i32 + j
      GX_packed_out(idx) = GX_packed_out(idx) + alpha * z(4) * s_x2
    end do
    do j = 1_i32, 4_i32
      idx                = 36_i32 + (j - 1_i32) * 4_i32 + j
      GX_packed_out(idx) = GX_packed_out(idx) + alpha * z(4) * s_x1
    end do

    rc = 0_i32
  end function cb_omp_coupled

end module omp_reentrancy_callbacks

program test_openmp_reentrancy
  use, intrinsic :: iso_c_binding, only: c_loc
  use omp_lib, only: omp_get_thread_num
  use fortchuck, only: &
    dp, i32, i64, &
    FC_SUCCESS_QUERY, FC_CONVERGED_GRAD, &
    fc_options_t, fc_result_t, fc_init_default_options, &
    fc_workspace_query, fc_solve
  use omp_reentrancy_callbacks, only: omp_prob_ctx_t, cb_omp_coupled
  implicit none (type, external)

  integer(i32), parameter :: N_INST    = 64_i32
  integer(i32), parameter :: N_THREADS = 16_i32
  integer(i32), parameter :: D_BOX     = 8_i32
  integer(i32), parameter :: R_ST      = 2_i32
  integer(i32), parameter :: N_ST_TOT  = 52_i32  ! 12*3 + 4*4
  integer(i32), parameter :: MAX_LWORK = 8192_i32
  integer(i32), parameter :: MAX_LIWRK = 512_i32

  integer(i32)                :: n_arr(R_ST), p_arr(R_ST)
  type(omp_prob_ctx_t), target :: ctxs(N_INST)
  real(dp)                    :: l_all(D_BOX, N_INST), u_all(D_BOX, N_INST)
  real(dp)                    :: z0_all(D_BOX, N_INST), X0_all(N_ST_TOT, N_INST)
  real(dp)                    :: z_seq(D_BOX, N_INST), X_seq(N_ST_TOT, N_INST)
  real(dp)                    :: z_par(D_BOX, N_INST), X_par(N_ST_TOT, N_INST)
  type(fc_result_t)           :: res_seq(N_INST), res_par(N_INST)
  real(dp)                    :: work_seq(MAX_LWORK)
  integer(i32)                :: iwork_seq(MAX_LIWRK)
  real(dp)                    :: work_thr(MAX_LWORK, N_THREADS)
  integer(i32)                :: iwork_thr(MAX_LIWRK, N_THREADS)
  type(fc_options_t)          :: opts
  integer(i64)                :: lwork_q, liwork_q
  integer(i32)                :: st, inst, tid, i, j, seed
  real(dp)                    :: max_f_diff, max_z_diff, max_X_diff, d_val

  n_arr = [12_i32, 4_i32]
  p_arr = [ 3_i32, 4_i32]

  call fc_init_default_options(opts)
  opts%gtol     = 1.0e-9_dp
  opts%max_iter = 1000_i32
  opts%max_eval = 5000_i32

  call fc_workspace_query(D_BOX, R_ST, n_arr, p_arr, opts%memory, lwork_q, liwork_q, st)
  if (st /= FC_SUCCESS_QUERY .or. lwork_q > int(MAX_LWORK, i64) .or. &
      liwork_q > int(MAX_LIWRK, i64)) then
    write (*, '(A)') '[FAIL] OpenMP workspace query out of static arena bounds'
    error stop 1
  end if

  ! Deterministic per-instance problem & initialization synthesis across 64 seeds
  do inst = 1_i32, N_INST
    seed = 10007_i32 * inst + 31337_i32
    do i = 1_i32, D_BOX
      l_all(i, inst)        = 0.1_dp
      u_all(i, inst)        = 2.0_dp
      ctxs(inst)%z_targ(i)  = 1.05_dp + 1.4_dp * next_u(seed)
      ctxs(inst)%z_w(i)     = 2.0_dp + real(i, dp) + next_u(seed)
      z0_all(i, inst)       = 0.2_dp + 1.6_dp * (0.5_dp * next_u(seed) + 0.5_dp)
    end do
    ! Coordinate 7 is fixed l_7 == u_7 on every instance
    l_all(7, inst)  = 0.65_dp
    u_all(7, inst)  = 0.65_dp
    z0_all(7, inst) = 0.65_dp

    do j = 1_i32, 3_i32
      do i = 1_i32, 12_i32
        ctxs(inst)%st1_diag(i, j) = 0.5_dp * real(i, dp) + 0.2_dp * real(j, dp) + &
          0.1_dp * next_u(seed)
      end do
    end do

    do j = 1_i32, 4_i32
      do i = 1_i32, 4_i32
        if (i == j) then
          ctxs(inst)%st2_targ(i, j) = 1.0_dp + 0.15_dp * next_u(seed)
        else
          ctxs(inst)%st2_targ(i, j) = 0.2_dp * next_u(seed)
        end if
      end do
    end do
    ctxs(inst)%cpl_alpha = 0.25_dp + 0.1_dp * next_u(seed)

    do i = 1_i32, N_ST_TOT
      X0_all(i, inst) = next_u(seed)
    end do
  end do

  z_seq     = z0_all
  X_seq     = X0_all
  z_par     = z0_all
  X_par     = X0_all
  work_seq  = -999.0_dp
  iwork_seq = -999_i32
  work_thr  =  777.0_dp
  iwork_thr =  777_i32

  ! 1. Sequential reference baseline across all 64 instances
  do inst = 1_i32, N_INST
    call fc_solve(D_BOX, R_ST, n_arr, p_arr, &
      z_seq(:, inst), l_all(:, inst), u_all(:, inst), X_seq(:, inst), &
      cb_omp_coupled, c_loc(ctxs(inst)), opts, &
      work_seq, lwork_q, iwork_seq, liwork_q, res_seq(inst))
  end do

  ! 2. Concurrent 16-thread OpenMP execution with schedule(dynamic, 1)
  !$omp parallel num_threads(N_THREADS) default(shared) private(inst, tid)
  tid = int(omp_get_thread_num(), i32) + 1_i32
  !$omp do schedule(dynamic, 1)
  do inst = 1_i32, N_INST
    call fc_solve(D_BOX, R_ST, n_arr, p_arr, &
      z_par(:, inst), l_all(:, inst), u_all(:, inst), X_par(:, inst), &
      cb_omp_coupled, c_loc(ctxs(inst)), opts, &
      work_thr(:, tid), lwork_q, iwork_thr(:, tid), liwork_q, res_par(inst))
  end do
  !$omp end do
  !$omp end parallel

  ! 3. Assert 100% convergence AND 0.0_dp bit-for-bit identical reproducibility
  max_f_diff = 0.0_dp
  max_z_diff = 0.0_dp
  max_X_diff = 0.0_dp

  do inst = 1_i32, N_INST
    if (res_seq(inst)%status /= FC_CONVERGED_GRAD) then
      write (*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,ES11.3,A,ES11.3,A,ES11.3)') &
        '[FAIL] inst=', inst, ' st=', res_seq(inst)%status, &
        ' it=', res_seq(inst)%n_iter, ' ev=', res_seq(inst)%n_eval, &
        ' fb=', res_seq(inst)%n_fallbacks, ' box=', res_seq(inst)%box_residual, &
        ' stf=', res_seq(inst)%stiefel_residual, ' tau=', res_seq(inst)%last_step_size
      error stop 1
    end if
    if (res_par(inst)%status /= res_seq(inst)%status .or. &
        res_par(inst)%n_iter /= res_seq(inst)%n_iter .or. &
        res_par(inst)%n_eval /= res_seq(inst)%n_eval) then
      write (*, '(A,I0)') '[FAIL] OpenMP count/status mismatch on instance ', inst
      error stop 1
    end if
    if (res_par(inst)%kkt_residual > 1.0e-9_dp .or. res_par(inst)%orth_error > 1.0e-13_dp) then
      write (*, '(A,I0)') '[FAIL] OpenMP KKT/orth tolerance exceeded on instance ', inst
      error stop 1
    end if

    d_val = abs(res_par(inst)%f_val - res_seq(inst)%f_val)
    if (d_val > max_f_diff) max_f_diff = d_val

    do i = 1_i32, D_BOX
      d_val = abs(z_par(i, inst) - z_seq(i, inst))
      if (d_val > max_z_diff) max_z_diff = d_val
    end do

    do i = 1_i32, N_ST_TOT
      d_val = abs(X_par(i, inst) - X_seq(i, inst))
      if (d_val > max_X_diff) max_X_diff = d_val
    end do
  end do

  if (max_f_diff > 0.0_dp .or. max_z_diff > 0.0_dp .or. max_X_diff > 0.0_dp) then
    write (*, '(A,3ES14.6)') '[FAIL] Non-zero OpenMP vs sequential diff: ', &
      max_f_diff, max_z_diff, max_X_diff
    error stop 1
  end if

  write (*, '(A,ES10.3)') '[PASS] test_openmp_reentrancy: 64 instances x 16 threads max_diff = ', &
    max(max_f_diff, max(max_z_diff, max_X_diff))

contains

  function next_u(state) result(u)
    integer(i32), intent(inout) :: state
    real(dp)                    :: u
    integer(8)                  :: s64
    s64   = mod(int(state, 8) * 48271_8, 2147483647_8)
    state = int(s64, i32)
    u     = 2.0_dp * (real(s64, dp) / 2147483647.0_dp) - 1.0_dp
  end function next_u

end program test_openmp_reentrancy
