module fortchuck_bench_callbacks
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_double, c_ptr, c_f_pointer
  use fortchuck, only: dp, i32
  use fortchuck_linalg, only: fc_gemm_nn
  implicit none (type, external)
  public

  !---------------------------------------------------------------------------
  ! B6: Ill-Conditioned Box-Coupled Brockett [0.1, 5.0]^10 x St(150, 10)
  !     f(z, X) = tr(X^T A X diag(z)) + 0.5 * (z - z_ref)^T M (z - z_ref)
  !     with kappa(M) = 10^4 and analytically certified Ky Fan f_star.
  !---------------------------------------------------------------------------
  type :: b6_ctx_t
    real(dp) :: A(150, 150)
    real(dp) :: M(10, 10)
    real(dp) :: z_ref(10)
  end type b6_ctx_t

  !---------------------------------------------------------------------------
  ! B7B: Mixed-Regime Multi-Stiefel St(200, 8) [Branch A] x St(24, 24) [Branch B]
  !      f(X_1, X_2) = 0.5 * tr(X_1^T A_1 X_1 diag(mu_1)) + 0.5 * ||X_2 B_2 - C_2||_F^2
  !---------------------------------------------------------------------------
  type :: b7b_ctx_t
    real(dp) :: A1(200, 200)
    real(dp) :: mu1(8)
    real(dp) :: B2(24, 24)
    real(dp) :: C2(24, 24)
  end type b7b_ctx_t

  !---------------------------------------------------------------------------
  ! B8A: 1D Periodic Metallic Kohn-Sham DFT with Fractional Fermi-Dirac Occs
  !      {z in [0, 1]^12 : 1^T z = 5.0} x St(120, 12)
  !---------------------------------------------------------------------------
  type :: b8a_ctx_t
    real(dp) :: H0(120, 120)
    real(dp) :: K_coul(120, 120)
    real(dp) :: alpha_H
    real(dp) :: kB_T
    real(dp) :: delta_FD
  end type b8a_ctx_t

contains

  function cb_bench_b6(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(b6_ctx_t), pointer :: ctx
    real(dp)                :: AX(150, 10), dz(10), Mdz(10), ray_j, two_zj
    integer(i32)            :: i, j, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue

    call c_f_pointer(udata, ctx)
    call fc_gemm_nn(150_i32, 10_i32, 150_i32, 1.0_dp, ctx%A, 150_i32, &
      X_packed, 150_i32, 0.0_dp, AX, 150_i32)

    do j = 1_i32, 10_i32
      dz(j) = z(j) - ctx%z_ref(j)
    end do

    f_out = 0.0_dp
    do i = 1_i32, 10_i32
      Mdz(i) = 0.0_dp
      do j = 1_i32, 10_i32
        Mdz(i) = Mdz(i) + ctx%M(i, j) * dz(j)
      end do
      f_out = f_out + 0.5_dp * dz(i) * Mdz(i)
    end do

    idx = 0_i32
    do j = 1_i32, 10_i32
      ray_j  = 0.0_dp
      two_zj = 2.0_dp * z(j)
      do i = 1_i32, 150_i32
        idx                = idx + 1_i32
        ray_j              = ray_j + X_packed(idx) * AX(i, j)
        GX_packed_out(idx) = two_zj * AX(i, j)
      end do
      f_out     = f_out + z(j) * ray_j
      gz_out(j) = ray_j + Mdz(j)
    end do

    rc = 0_i32
  end function cb_bench_b6

  function cb_bench_b7b(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(b7b_ctx_t), pointer :: ctx
    real(dp)                 :: AX1(200, 8), Res2(24, 24), GX2(24, 24), muj, rij
    integer(i32)             :: i, j, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue
    if (.false. .and. z(1) > 0.0_dp .and. gz_out(1) > 0.0_dp) continue

    call c_f_pointer(udata, ctx)

    ! Factor 1: St(200, 8) Brockett term
    call fc_gemm_nn(200_i32, 8_i32, 200_i32, 1.0_dp, ctx%A1, 200_i32, &
      X_packed, 200_i32, 0.0_dp, AX1, 200_i32)

    f_out = 0.0_dp
    idx   = 0_i32
    do j = 1_i32, 8_i32
      muj = ctx%mu1(j)
      do i = 1_i32, 200_i32
        idx                = idx + 1_i32
        f_out              = f_out + 0.5_dp * muj * X_packed(idx) * AX1(i, j)
        GX_packed_out(idx) = muj * AX1(i, j)
      end do
    end do

    ! Factor 2: St(24, 24) Orthogonal Wahba / Procrustes starting at offset 1601
    call fc_gemm_nn(24_i32, 24_i32, 24_i32, 1.0_dp, X_packed(1601), 24_i32, &
      ctx%B2, 24_i32, 0.0_dp, Res2, 24_i32)

    do j = 1_i32, 24_i32
      do i = 1_i32, 24_i32
        rij        = Res2(i, j) - ctx%C2(i, j)
        Res2(i, j) = rij
        f_out      = f_out + 0.5_dp * rij * rij
      end do
    end do

    ! GX2 = Res2 * B2^T via unit-stride inner accumulation
    do j = 1_i32, 24_i32
      do i = 1_i32, 24_i32
        GX2(i, j) = 0.0_dp
      end do
    end do
    do j = 1_i32, 24_i32
      do idx = 1_i32, 24_i32
        muj = ctx%B2(j, idx)
        do i = 1_i32, 24_i32
          GX2(i, j) = GX2(i, j) + Res2(i, idx) * muj
        end do
      end do
    end do

    idx = 1600_i32
    do j = 1_i32, 24_i32
      do i = 1_i32, 24_i32
        idx                = idx + 1_i32
        GX_packed_out(idx) = GX2(i, j)
      end do
    end do

    rc = 0_i32
  end function cb_bench_b7b

  function cb_bench_b8a(d, R, n_arr, p_arr, z, X_packed, &
      f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d, R
    integer(c_int32_t), intent(in)        :: n_arr(*), p_arr(*)
    real(c_double), intent(in)            :: z(*), X_packed(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_packed_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc

    type(b8a_ctx_t), pointer :: ctx
    real(dp)                 :: HX(120, 12), rho(120), Krho(120), v_H(120)
    real(dp)                 :: zj, two_zj, xij, fx_ij, e_kin_ext, e_hartree, e_ent
    real(dp)                 :: zp, zm, acc_j
    integer(i32)             :: i, j, idx

    if (.false. .and. d == 0_i32 .and. R == 0_i32) continue
    if (.false. .and. n_arr(1) == 0_i32 .and. p_arr(1) == 0_i32) continue

    call c_f_pointer(udata, ctx)

    ! 1. Single-particle core Hamiltonian action H0 * X (120x120 * 120x12)
    call fc_gemm_nn(120_i32, 12_i32, 120_i32, 1.0_dp, ctx%H0, 120_i32, &
      X_packed, 120_i32, 0.0_dp, HX, 120_i32)

    ! 2. Occupation-weighted electron density rho_i = sum_j z_j * X_{i,j}^2
    do i = 1_i32, 120_i32
      rho(i) = 0.0_dp
    end do
    idx = 0_i32
    do j = 1_i32, 12_i32
      zj = z(j)
      do i = 1_i32, 120_i32
        idx    = idx + 1_i32
        xij    = X_packed(idx)
        rho(i) = rho(i) + zj * xij * xij
      end do
    end do

    ! 3. Hartree potential v_H = alpha_H * (K_coul * rho)
    call fc_gemm_nn(120_i32, 1_i32, 120_i32, 1.0_dp, ctx%K_coul, 120_i32, &
      rho, 120_i32, 0.0_dp, Krho, 120_i32)

    e_hartree = 0.0_dp
    do i = 1_i32, 120_i32
      e_hartree = e_hartree + rho(i) * Krho(i)
      v_H(i)    = ctx%alpha_H * Krho(i)
    end do
    e_hartree = 0.5_dp * ctx%alpha_H * e_hartree

    ! 4. Fock matrix F_X = H0*X + diag(v_H)*X, gradients gz and GX, and energies
    e_kin_ext = 0.0_dp
    e_ent     = 0.0_dp
    idx       = 0_i32
    do j = 1_i32, 12_i32
      zj     = z(j)
      two_zj = 2.0_dp * zj
      acc_j  = 0.0_dp
      do i = 1_i32, 120_i32
        idx                = idx + 1_i32
        xij                = X_packed(idx)
        fx_ij              = HX(i, j) + v_H(i) * xij
        e_kin_ext          = e_kin_ext + zj * xij * HX(i, j)
        acc_j              = acc_j + xij * fx_ij
        GX_packed_out(idx) = two_zj * fx_ij
      end do
      zp        = max(zj + ctx%delta_FD, 1.0e-30_dp)
      zm        = max(1.0_dp - zj + ctx%delta_FD, 1.0e-30_dp)
      e_ent     = e_ent + ctx%kB_T * (zp * log(zp) + zm * log(zm))
      gz_out(j) = acc_j + ctx%kB_T * (log(zp) - log(zm))
    end do

    f_out = e_kin_ext + e_hartree + e_ent
    rc    = 0_i32
  end function cb_bench_b8a

end module fortchuck_bench_callbacks

program fortchuck_bench
  use, intrinsic :: iso_c_binding, only: c_loc
  use fortchuck, only: &
    dp, i32, i64, &
    FC_SUCCESS_QUERY, FC_CONVERGED_GRAD, &
    fc_options_t, fc_result_t, fc_init_default_options, &
    fc_workspace_query, fc_solve, fc_solve_constrained
  use fortchuck_linalg, only: &
    fc_gemm_nn, fc_gemm_nt, fc_lu_factor, fc_lu_solve, fc_geqrf_pos
  use fortchuck_stiefel, only: &
    fc_stiefel_proj_tangent, fc_cayley_setup, fc_cayley_retract, &
    fc_cayley_transport_vec, fc_cayley_transport_batch
  use fortchuck_bench_callbacks, only: &
    b6_ctx_t, b7b_ctx_t, b8a_ctx_t, &
    cb_bench_b6, cb_bench_b7b, cb_bench_b8a
  implicit none (type, external)

  write (*, '(A)') '================================================================================'
  write (*, '(A)') ' fortchuck v0.2.0 Native Fortran 2018 Benchmark Suite (0 FFI Callback Overhead)'
  write (*, '(A)') '================================================================================'
  write (*, '(A)') ''

  call bench_cayley_transport_throughput()
  write (*, '(A)') ''
  call bench_b6_coupled_brockett()
  write (*, '(A)') ''
  call bench_b7b_multistiefel()
  write (*, '(A)') ''
  call bench_b8a_kohn_sham_dft()
  write (*, '(A)') ''
  write (*, '(A)') '================================================================================'

contains

  function next_rand_u(state) result(u)
    integer(i32), intent(inout) :: state
    real(dp)                    :: u
    integer(i64)                :: s64
    s64   = mod(int(state, i64) * 48271_i64, 2147483647_i64)
    state = int(s64, i32)
    u     = 2.0_dp * (real(s64, dp) / 2147483647.0_dp) - 1.0_dp
  end function next_rand_u

  subroutine fill_rand(n, arr, state)
    integer(i32), intent(in)    :: n
    real(dp), intent(out)       :: arr(n)
    integer(i32), intent(inout) :: state
    integer(i32)                :: i
    do i = 1_i32, n
      arr(i) = next_rand_u(state)
    end do
  end subroutine fill_rand

  subroutine fill_randn(n, arr, state)
    integer(i32), intent(in)    :: n
    real(dp), intent(out)       :: arr(n)
    integer(i32), intent(inout) :: state
    real(dp), parameter         :: TWO_PI = 6.28318530717958647692_dp
    integer(i32)                :: i
    real(dp)                    :: u1, u2, r_bm, theta
    i = 1_i32
    do while (i <= n)
      u1    = max(1.0e-12_dp, 0.5_dp * next_rand_u(state) + 0.5_dp)
      u2    = 0.5_dp * next_rand_u(state) + 0.5_dp
      r_bm  = sqrt(-2.0_dp * log(u1))
      theta = TWO_PI * u2
      arr(i) = r_bm * cos(theta)
      if (i + 1_i32 <= n) arr(i + 1_i32) = r_bm * sin(theta)
      i = i + 2_i32
    end do
  end subroutine fill_randn

  subroutine sort_real_asc(n, a)
    integer(i32), intent(in) :: n
    real(dp), intent(inout)  :: a(n)
    integer(i32)             :: i, j
    real(dp)                 :: key
    do i = 2_i32, n
      key = a(i)
      j   = i - 1_i32
      do while (j >= 1_i32)
        if (a(j) <= key) exit
        a(j + 1_i32) = a(j)
        j            = j - 1_i32
      end do
      a(j + 1_i32) = key
    end do
  end subroutine sort_real_asc

  function median_10(vals) result(med)
    real(dp), intent(in) :: vals(10)
    real(dp)             :: med
    real(dp)             :: tmp(10)
    tmp = vals
    call sort_real_asc(10_i32, tmp)
    med = 0.5_dp * (tmp(5) + tmp(6))
  end function median_10

  !---------------------------------------------------------------------------
  ! (a) Batched vs Per-Vector Cayley Transport Throughput Microbenchmark
  !     K_batch = 16 (m = 7 -> 2*m + 2 = 16) on:
  !       - St(200, 8) [Branch A: 2*p < n, Schur-Woodbury]
  !       - St(24, 24) [Branch B: 2*p >= n, Resolvent Doubling]
  !---------------------------------------------------------------------------
  subroutine bench_cayley_transport_throughput()
    integer(i32), parameter :: K_BATCH     = 16_i32
    integer(i32), parameter :: N_WARM      = 250_i32
    integer(i32), parameter :: N_CHUNKS    = 5_i32
    integer(i32), parameter :: N_CHUNK_REP = 800_i32
    integer(i32), parameter :: N_REP       = N_CHUNKS * N_CHUNK_REP

    ! Branch A: St(200, 8)
    real(dp)                  :: XA(200, 8), YA(200, 8), etaA(200, 8), tau_qrA(8), w_qrA(200)
    real(dp)                  :: AskwA(8, 8), scr_p2A(8, 8)
    real(dp), allocatable     :: cayA(:), scrA(:)
    real(dp), allocatable     :: PanelA_init(:, :), PanelA_vec(:, :), PanelA_bat(:, :)
    integer(i32)              :: ipivA(8)

    ! Branch B: St(24, 24)
    real(dp)                  :: XB(24, 24), YB(24, 24), etaB(24, 24), tau_qrB(24), w_qrB(24)
    real(dp)                  :: AskwB(24, 24), scr_p2B(24, 24)
    real(dp), allocatable     :: cayB(:), scrB(:)
    real(dp), allocatable     :: PanelB_init(:, :), PanelB_vec(:, :), PanelB_bat(:, :)
    integer(i32)              :: ipivB(24)

    integer(i64) :: t0, t1, t_rate
    integer(i32) :: rng, info, k, chk_idx, rep, col0
    real(dp)     :: tau, c_ms, ms_vec_a, ms_bat_a, sp_a, err_a1, err_an
    real(dp)     :: ms_vec_b, ms_bat_b, sp_b, err_b1, err_bn, chk

    allocate (cayA(4096), scrA(8192))
    allocate (PanelA_init(200, 128), PanelA_vec(200, 128), PanelA_bat(200, 128))
    allocate (cayB(2048), scrB(16384))
    allocate (PanelB_init(24, 384), PanelB_vec(24, 384), PanelB_bat(24, 384))

    rng = 135791_i32
    tau = 0.25_dp

    ! --- Setup Branch A: St(200, 8) ---
    call fill_rand(1600_i32, XA, rng)
    call fc_geqrf_pos(200_i32, 8_i32, XA, tau_qrA, w_qrA, info)
    call fill_rand(1600_i32, etaA, rng)
    call fc_stiefel_proj_tangent(200_i32, 8_i32, XA, etaA, AskwA, scr_p2A)
    call fc_cayley_setup(200_i32, 8_i32, XA, etaA, cayA, scrA)
    call fc_cayley_retract(200_i32, 8_i32, XA, etaA, tau, cayA, ipivA, YA, scrA, info)
    if (info /= 0_i32) error stop 'Cayley retract failed on St(200,8)'

    call fill_rand(25600_i32, PanelA_init, rng)
    do k = 1_i32, K_BATCH
      col0 = (k - 1_i32) * 8_i32 + 1_i32
      call fc_stiefel_proj_tangent(200_i32, 8_i32, XA, &
        PanelA_init(1, col0), AskwA, scr_p2A)
    end do

    ! 1-step parity check on St(200, 8)
    PanelA_vec = PanelA_init
    PanelA_bat = PanelA_init
    do k = 1_i32, K_BATCH
      col0 = (k - 1_i32) * 8_i32 + 1_i32
      call fc_cayley_transport_vec(200_i32, 8_i32, XA, etaA, tau, &
        cayA, ipivA, PanelA_vec(1, col0), scrA)
    end do
    call fc_cayley_transport_batch(200_i32, 8_i32, K_BATCH, XA, etaA, tau, &
      cayA, ipivA, PanelA_bat, scrA)
    err_a1 = maxval(abs(PanelA_bat - PanelA_vec))
    if (err_a1 > 1.0e-14_dp) error stop 'Branch A batch transport parity > 1e-14'

    ! Warmup & interleaved timing on St(200, 8)
    PanelA_vec = PanelA_init
    PanelA_bat = PanelA_init
    do rep = 1_i32, N_WARM
      do k = 1_i32, K_BATCH
        col0 = (k - 1_i32) * 8_i32 + 1_i32
        call fc_cayley_transport_vec(200_i32, 8_i32, XA, etaA, tau, &
          cayA, ipivA, PanelA_vec(1, col0), scrA)
      end do
      call fc_cayley_transport_batch(200_i32, 8_i32, K_BATCH, XA, etaA, tau, &
        cayA, ipivA, PanelA_bat, scrA)
    end do

    PanelA_vec = PanelA_init
    PanelA_bat = PanelA_init
    ms_vec_a   = 1.0e30_dp
    ms_bat_a   = 1.0e30_dp
    do chk_idx = 1_i32, N_CHUNKS
      call system_clock(t0, t_rate)
      do rep = 1_i32, N_CHUNK_REP
        do k = 1_i32, K_BATCH
          col0 = (k - 1_i32) * 8_i32 + 1_i32
          call fc_cayley_transport_vec(200_i32, 8_i32, XA, etaA, tau, &
            cayA, ipivA, PanelA_vec(1, col0), scrA)
        end do
      end do
      call system_clock(t1)
      c_ms = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      if (c_ms < ms_vec_a) ms_vec_a = c_ms

      call system_clock(t0, t_rate)
      do rep = 1_i32, N_CHUNK_REP
        call fc_cayley_transport_batch(200_i32, 8_i32, K_BATCH, XA, etaA, tau, &
          cayA, ipivA, PanelA_bat, scrA)
      end do
      call system_clock(t1)
      c_ms = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      if (c_ms < ms_bat_a) ms_bat_a = c_ms
    end do
    ms_vec_a = real(N_CHUNKS, dp) * ms_vec_a
    ms_bat_a = real(N_CHUNKS, dp) * ms_bat_a
    sp_a     = ms_vec_a / max(ms_bat_a, 1.0e-9_dp)
    err_an   = maxval(abs(PanelA_bat - PanelA_vec))

    ! --- Setup Branch B: St(24, 24) ---
    call fill_rand(576_i32, XB, rng)
    call fc_geqrf_pos(24_i32, 24_i32, XB, tau_qrB, w_qrB, info)
    call fill_rand(576_i32, etaB, rng)
    call fc_stiefel_proj_tangent(24_i32, 24_i32, XB, etaB, AskwB, scr_p2B)
    call fc_cayley_setup(24_i32, 24_i32, XB, etaB, cayB, scrB)
    call fc_cayley_retract(24_i32, 24_i32, XB, etaB, tau, cayB, ipivB, YB, scrB, info)
    if (info /= 0_i32) error stop 'Cayley retract failed on St(24,24)'

    call fill_rand(9216_i32, PanelB_init, rng)
    do k = 1_i32, K_BATCH
      col0 = (k - 1_i32) * 24_i32 + 1_i32
      call fc_stiefel_proj_tangent(24_i32, 24_i32, XB, &
        PanelB_init(1, col0), AskwB, scr_p2B)
    end do

    ! 1-step parity check on St(24, 24)
    PanelB_vec = PanelB_init
    PanelB_bat = PanelB_init
    do k = 1_i32, K_BATCH
      col0 = (k - 1_i32) * 24_i32 + 1_i32
      call fc_cayley_transport_vec(24_i32, 24_i32, XB, etaB, tau, &
        cayB, ipivB, PanelB_vec(1, col0), scrB)
    end do
    call fc_cayley_transport_batch(24_i32, 24_i32, K_BATCH, XB, etaB, tau, &
      cayB, ipivB, PanelB_bat, scrB)
    err_b1 = maxval(abs(PanelB_bat - PanelB_vec))
    if (err_b1 > 1.0e-14_dp) error stop 'Branch B batch transport parity > 1e-14'

    ! Warmup & interleaved timing on St(24, 24)
    PanelB_vec = PanelB_init
    PanelB_bat = PanelB_init
    do rep = 1_i32, N_WARM
      do k = 1_i32, K_BATCH
        col0 = (k - 1_i32) * 24_i32 + 1_i32
        call fc_cayley_transport_vec(24_i32, 24_i32, XB, etaB, tau, &
          cayB, ipivB, PanelB_vec(1, col0), scrB)
      end do
      call fc_cayley_transport_batch(24_i32, 24_i32, K_BATCH, XB, etaB, tau, &
        cayB, ipivB, PanelB_bat, scrB)
    end do

    PanelB_vec = PanelB_init
    PanelB_bat = PanelB_init
    ms_vec_b   = 1.0e30_dp
    ms_bat_b   = 1.0e30_dp
    do chk_idx = 1_i32, N_CHUNKS
      call system_clock(t0, t_rate)
      do rep = 1_i32, N_CHUNK_REP
        do k = 1_i32, K_BATCH
          col0 = (k - 1_i32) * 24_i32 + 1_i32
          call fc_cayley_transport_vec(24_i32, 24_i32, XB, etaB, tau, &
            cayB, ipivB, PanelB_vec(1, col0), scrB)
        end do
      end do
      call system_clock(t1)
      c_ms = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      if (c_ms < ms_vec_b) ms_vec_b = c_ms

      call system_clock(t0, t_rate)
      do rep = 1_i32, N_CHUNK_REP
        call fc_cayley_transport_batch(24_i32, 24_i32, K_BATCH, XB, etaB, tau, &
          cayB, ipivB, PanelB_bat, scrB)
      end do
      call system_clock(t1)
      c_ms = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      if (c_ms < ms_bat_b) ms_bat_b = c_ms
    end do
    ms_vec_b = real(N_CHUNKS, dp) * ms_vec_b
    ms_bat_b = real(N_CHUNKS, dp) * ms_bat_b
    sp_b     = ms_vec_b / max(ms_bat_b, 1.0e-9_dp)
    err_bn   = maxval(abs(PanelB_bat - PanelB_vec))
    chk      = YA(1, 1) + YB(1, 1) + PanelA_bat(1, 1) + PanelB_bat(1, 1)

    write (*, '(A,I0,A,I0,A)') &
      '[1] Batched Cayley Vector Transport Throughput (K_batch = ', K_BATCH, &
      ', N_rep = ', N_REP, ')'
    write (*, '(A,F9.3,A,F7.3,A,F9.3,A,F7.3,A,F6.2,A,ES9.2,A,ES9.2)') &
      '  St(200, 8) [Branch A]: 16x vec = ', ms_vec_a, ' ms (', &
      (ms_vec_a * 1000.0_dp) / real(N_REP, dp), ' us/step) | batch = ', &
      ms_bat_a, ' ms (', (ms_bat_a * 1000.0_dp) / real(N_REP, dp), &
      ' us/step) | speedup = ', sp_a, 'x | 1-step err = ', err_a1, &
      ' | N-step err = ', err_an
    write (*, '(A,F9.3,A,F7.3,A,F9.3,A,F7.3,A,F6.2,A,ES9.2,A,ES9.2,A,ES9.2)') &
      '  St(24, 24) [Branch B]: 16x vec = ', ms_vec_b, ' ms (', &
      (ms_vec_b * 1000.0_dp) / real(N_REP, dp), ' us/step) | batch = ', &
      ms_bat_b, ' ms (', (ms_bat_b * 1000.0_dp) / real(N_REP, dp), &
      ' us/step) | speedup = ', sp_b, 'x | 1-step err = ', err_b1, &
      ' | N-step err = ', err_bn, ' | chk=', chk
  end subroutine bench_cayley_transport_throughput

  function det_sign_24(A_in) result(sgn)
    real(dp), intent(in) :: A_in(24, 24)
    real(dp)             :: sgn
    real(dp)             :: LU(24, 24)
    integer(i32)         :: ipiv(24), info, i

    LU = A_in
    call fc_lu_factor(24_i32, LU, ipiv, 1.0e-12_dp, info)
    if (info /= 0_i32) then
      sgn = 1.0_dp
      return
    end if
    sgn = 1.0_dp
    do i = 1_i32, 24_i32
      if (ipiv(i) /= i) sgn = -sgn
      if (LU(i, i) < 0.0_dp) sgn = -sgn
    end do
  end function det_sign_24

  !---------------------------------------------------------------------------
  ! (b.1) B6: [0.1, 5.0]^10 x St(150, 10), kappa(M) = 10^4 (10 seeds)
  !       Exact canonical formulation from benchmarks/problems.py:
  !         alphas(1:10)   = -0.80 + 0.004*(i-1)  (delta_alpha < mu_min*delta_z)
  !         alphas(11:150) = alphas(10) + 0.12*(i-10)
  !         m_eigs         = geomspace(0.01, 100.0, 10)  (kappa(M) = 10^4)
  !         z_star         = linspace(5.0, 0.1, 10), active u_1=5.0 & l_10=0.1
  !---------------------------------------------------------------------------
  subroutine bench_b6_coupled_brockett()
    integer(i32), parameter             :: D_BOX = 10_i32, R_ST = 1_i32, N_SEEDS = 10_i32
    integer(i32)                        :: n_arr(1), p_arr(1)
    type(b6_ctx_t), allocatable, target :: ctx
    type(fc_options_t)                  :: opts
    type(fc_result_t)                   :: res
    real(dp), allocatable               :: Q150(:, :), AQ(:, :), work(:)
    integer(i32), allocatable           :: iwork(:)
    real(dp)                            :: tau_qr(150), w_qr(150)
    real(dp)                            :: Q10(10, 10), MQ(10, 10), tau10(10), w10(10), m_eigs(10)
    real(dp)                            :: alphas(150), z_star(10), lam_lo(10), lam_hi(10)
    real(dp)                            :: M_lu(10, 10), rhs(10, 1), dz_star(10), M_dz(10)
    real(dp)                            :: l(10), u(10), z(10), X(1500)
    integer(i32)                        :: ipiv10(10)
    real(dp)                            :: ms_arr(10), it_arr(10), ev_arr(10)
    real(dp)                            :: f_star, f_err, max_f_err, max_kkt, max_orth, tot_ms
    integer(i64)                        :: lwork_q, liwork_q, t0, t1, t_rate
    integer(i32)                        :: s, i, j, rng_prob, rng_init, info, st, n_pass

    allocate (ctx, Q150(150, 150), AQ(150, 150))

    n_arr(1) = 150_i32
    p_arr(1) = 10_i32
    l        = 0.1_dp
    u        = 5.0_dp

    call fc_init_default_options(opts)
    opts%gtol = 1.0e-6_dp

    call fc_workspace_query(D_BOX, R_ST, n_arr, p_arr, opts%memory, lwork_q, liwork_q, st)
    if (st /= FC_SUCCESS_QUERY) then
      error stop 'B6 workspace query failed'
    end if
    allocate (work(lwork_q), iwork(liwork_q))

    ! Build canonical B6 problem instance (kappa(M) = 10^4, unique Ky Fan global minimum)
    rng_prob = 202607_i32
    do i = 1_i32, 10_i32
      alphas(i) = -0.80_dp + 0.004_dp * real(i - 1_i32, dp)
      z_star(i) = 5.0_dp - 4.9_dp * real(i - 1_i32, dp) / 9.0_dp
      m_eigs(i) = 0.01_dp * (10000.0_dp**(real(i - 1_i32, dp) / 9.0_dp))
    end do
    do i = 11_i32, 150_i32
      alphas(i) = alphas(10) + 0.12_dp * real(i - 10_i32, dp)
    end do

    call fill_randn(22500_i32, Q150, rng_prob)
    call fc_geqrf_pos(150_i32, 150_i32, Q150, tau_qr, w_qr, info)
    do j = 1_i32, 150_i32
      do i = 1_i32, 150_i32
        AQ(i, j) = Q150(i, j) * alphas(j)
      end do
    end do
    call fc_gemm_nt(150_i32, 150_i32, 150_i32, 1.0_dp, AQ, 150_i32, &
      Q150, 150_i32, 0.0_dp, ctx%A, 150_i32)
    do j = 1_i32, 150_i32
      do i = 1_i32, j - 1_i32
        ctx%A(i, j) = 0.5_dp * (ctx%A(i, j) + ctx%A(j, i))
        ctx%A(j, i) = ctx%A(i, j)
      end do
    end do

    call fill_randn(100_i32, Q10, rng_prob)
    call fc_geqrf_pos(10_i32, 10_i32, Q10, tau10, w10, info)
    do j = 1_i32, 10_i32
      do i = 1_i32, 10_i32
        MQ(i, j) = Q10(i, j) * m_eigs(j)
      end do
    end do
    call fc_gemm_nt(10_i32, 10_i32, 10_i32, 1.0_dp, MQ, 10_i32, &
      Q10, 10_i32, 0.0_dp, ctx%M, 10_i32)
    do j = 1_i32, 10_i32
      do i = 1_i32, j - 1_i32
        ctx%M(i, j) = 0.5_dp * (ctx%M(i, j) + ctx%M(j, i))
        ctx%M(j, i) = ctx%M(i, j)
      end do
    end do

    lam_lo     = 0.0_dp
    lam_hi     = 0.0_dp
    lam_hi(1)  = 0.25_dp
    lam_lo(10) = 0.25_dp
    do i = 1_i32, 10_i32
      rhs(i, 1) = lam_lo(i) - lam_hi(i) - alphas(i)
    end do
    M_lu = ctx%M
    call fc_lu_factor(10_i32, M_lu, ipiv10, 1.0e-12_dp, info)
    call fc_lu_solve(10_i32, 1_i32, M_lu, ipiv10, rhs)

    do i = 1_i32, 10_i32
      ctx%z_ref(i) = z_star(i) - rhs(i, 1)
      dz_star(i)   = z_star(i) - ctx%z_ref(i)
    end do
    f_star = 0.0_dp
    do i = 1_i32, 10_i32
      M_dz(i) = 0.0_dp
      do j = 1_i32, 10_i32
        M_dz(i) = M_dz(i) + ctx%M(i, j) * dz_star(j)
      end do
      f_star = f_star + alphas(i) * z_star(i) + 0.5_dp * dz_star(i) * M_dz(i)
    end do

    tot_ms    = 0.0_dp
    max_f_err = 0.0_dp
    max_kkt   = 0.0_dp
    max_orth  = 0.0_dp
    n_pass    = 0_i32

    do s = 1_i32, N_SEEDS
      rng_init = 90000_i32 + 7919_i32 * s
      do i = 1_i32, 10_i32
        z(i) = 0.3_dp + 4.4_dp * (0.5_dp * next_rand_u(rng_init) + 0.5_dp)
      end do
      call fill_randn(1500_i32, X, rng_init)
      call fc_geqrf_pos(150_i32, 10_i32, X, tau10, w_qr, info)

      call system_clock(t0, t_rate)
      call fc_solve(D_BOX, R_ST, n_arr, p_arr, z, l, u, X, &
        cb_bench_b6, c_loc(ctx), opts, work, lwork_q, iwork, liwork_q, res)
      call system_clock(t1)

      ms_arr(s) = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      it_arr(s) = real(res%n_iter, dp)
      ev_arr(s) = real(res%n_eval, dp)
      tot_ms    = tot_ms + ms_arr(s)

      if (res%status == FC_CONVERGED_GRAD) n_pass = n_pass + 1_i32
      f_err = abs(res%f_val - f_star)
      if (f_err > max_f_err) max_f_err = f_err
      if (res%kkt_residual > max_kkt) max_kkt = res%kkt_residual
      if (res%orth_error > max_orth) max_orth = res%orth_error
    end do

    write (*, '(A,I0,A,I0,A)') &
      '[2] B6: [0.1, 5.0]^10 x St(150, 10) Coupled Brockett (kappa=1e4, ', &
      n_pass, '/', N_SEEDS, ' converged)'
    write (*, '(A,F10.3,A,F9.3,A,F9.3,A,F7.1,A,F7.1,A,ES9.2,A,ES9.2,A,ES9.2)') &
      '  LRBFGS (m=7): total = ', tot_ms, ' ms | mean = ', tot_ms / 10.0_dp, &
      ' ms | med = ', median_10(ms_arr), ' ms | med_it = ', median_10(it_arr), &
      ' | med_ev = ', median_10(ev_arr), ' | max|f-f*| = ', max_f_err, &
      ' | max_KKT = ', max_kkt, ' | max_orth = ', max_orth
  end subroutine bench_b6_coupled_brockett

  !---------------------------------------------------------------------------
  ! (b.2) B7B: St(200, 8) x St(24, 24) Mixed-Regime Multi-Stiefel (SO(24))
  !            10 seeds comparing LRBFGS (m = 7) vs ABB (m = 0)
  !---------------------------------------------------------------------------
  subroutine bench_b7b_multistiefel()
    integer(i32), parameter              :: R_ST = 2_i32, N_SEEDS = 10_i32
    integer(i32)                         :: n_arr(2), p_arr(2)
    integer(i32), allocatable            :: iwork(:)
    type(b7b_ctx_t), allocatable, target :: ctx
    type(fc_options_t)                   :: opts
    type(fc_result_t)                    :: res
    real(dp), allocatable                :: Q200(:, :), AQ200(:, :), work(:)
    real(dp)                             :: tau200(200), w200(200)
    real(dp)                             :: U24(24, 24), V24(24, 24), R24(24, 24), UB24(24, 24)
    real(dp)                             :: Noise24(24, 24), X2_init(24, 24)
    real(dp)                             :: tau24(24), w24(24), sv24(24), eigs200(200)
    real(dp)                             :: z_dum(1), l_dum(1), u_dum(1)
    real(dp)                             :: X0(2176), X(2176)
    real(dp)                             :: ms_m7(10), it_m7(10), ev_m7(10)
    real(dp)                             :: ms_m0(10), it_m0(10), ev_m0(10)
    real(dp)                             :: tot_m7, tot_m0, kkt_m7, kkt_m0, orth_m7, orth_m0
    integer(i64)                         :: lwork_m7, liwork_m7, lwork_m0, liwork_m0, t0, t1, t_rate
    integer(i32)                         :: s, i, j, idx, rng_prob, rng_init, info, st, pass_m7, pass_m0

    allocate (ctx, Q200(200, 200), AQ200(200, 200))

    n_arr = [200_i32, 24_i32]
    p_arr = [  8_i32, 24_i32]
    z_dum = 0.0_dp
    l_dum = 0.0_dp
    u_dum = 0.0_dp

    do j = 1_i32, 8_i32
      ctx%mu1(j) = real(9_i32 - j, dp)
    end do
    do i = 1_i32, 200_i32
      eigs200(i) = -4.0_dp + 22.0_dp * real(i - 1_i32, dp) / 199.0_dp
    end do
    do j = 1_i32, 24_i32
      sv24(j) = 0.25_dp * (60.0_dp**(real(j - 1_i32, dp) / 23.0_dp))
    end do

    ! Build canonical B7B problem instance once (matching benchmarks/problems.py)
    rng_prob = 202609_i32
    call fill_rand(40000_i32, Q200, rng_prob)
    call fc_geqrf_pos(200_i32, 200_i32, Q200, tau200, w200, info)
    do j = 1_i32, 200_i32
      do i = 1_i32, 200_i32
        AQ200(i, j) = Q200(i, j) * eigs200(j)
      end do
    end do
    call fc_gemm_nt(200_i32, 200_i32, 200_i32, 1.0_dp, AQ200, 200_i32, &
      Q200, 200_i32, 0.0_dp, ctx%A1, 200_i32)
    do j = 1_i32, 200_i32
      do i = 1_i32, j - 1_i32
        ctx%A1(i, j) = 0.5_dp * (ctx%A1(i, j) + ctx%A1(j, i))
        ctx%A1(j, i) = ctx%A1(i, j)
      end do
    end do

    call fill_rand(576_i32, U24, rng_prob)
    call fc_geqrf_pos(24_i32, 24_i32, U24, tau24, w24, info)
    call fill_rand(576_i32, V24, rng_prob)
    call fc_geqrf_pos(24_i32, 24_i32, V24, tau24, w24, info)
    do j = 1_i32, 24_i32
      do i = 1_i32, 24_i32
        UB24(i, j) = U24(i, j) * sv24(j)
      end do
    end do
    call fc_gemm_nt(24_i32, 24_i32, 24_i32, 1.0_dp, UB24, 24_i32, &
      V24, 24_i32, 0.0_dp, ctx%B2, 24_i32)

    call fill_rand(576_i32, R24, rng_prob)
    call fc_geqrf_pos(24_i32, 24_i32, R24, tau24, w24, info)
    if (det_sign_24(R24) < 0.0_dp) then
      do i = 1_i32, 24_i32
        R24(i, 1) = -R24(i, 1)
      end do
    end if
    call fill_rand(576_i32, Noise24, rng_prob)
    call fc_gemm_nn(24_i32, 24_i32, 24_i32, 1.0_dp, R24, 24_i32, &
      ctx%B2, 24_i32, 0.0_dp, ctx%C2, 24_i32)
    do j = 1_i32, 24_i32
      do i = 1_i32, 24_i32
        ctx%C2(i, j) = ctx%C2(i, j) + 0.05_dp * Noise24(i, j)
      end do
    end do

    call fc_workspace_query(0_i32, R_ST, n_arr, p_arr, 7_i32, lwork_m7, liwork_m7, st)
    call fc_workspace_query(0_i32, R_ST, n_arr, p_arr, 0_i32, lwork_m0, liwork_m0, st)
    if (st /= FC_SUCCESS_QUERY) error stop 'B7B workspace query failed'
    allocate (work(max(lwork_m7, lwork_m0)), iwork(max(liwork_m7, liwork_m0)))

    tot_m7  = 0.0_dp; tot_m0  = 0.0_dp
    kkt_m7  = 0.0_dp; kkt_m0  = 0.0_dp
    orth_m7 = 0.0_dp; orth_m0 = 0.0_dp
    pass_m7 = 0_i32;  pass_m0 = 0_i32

    do s = 1_i32, N_SEEDS
      rng_init = 88000_i32 + 4999_i32 * s

      call fill_rand(1600_i32, X0(1:1600), rng_init)
      call fc_geqrf_pos(200_i32, 8_i32, X0(1:1600), tau200(1:8), w200, info)

      call fill_rand(576_i32, X2_init, rng_init)
      call fc_geqrf_pos(24_i32, 24_i32, X2_init, tau24, w24, info)
      if (det_sign_24(X2_init) < 0.0_dp) then
        do i = 1_i32, 24_i32
          X2_init(i, 1) = -X2_init(i, 1)
        end do
      end if
      idx = 1600_i32
      do j = 1_i32, 24_i32
        do i = 1_i32, 24_i32
          idx     = idx + 1_i32
          X0(idx) = X2_init(i, j)
        end do
      end do

      ! 1. LRBFGS (m = 7, batched Cayley vector transport, default 15000 cap)
      call fc_init_default_options(opts)
      opts%memory = 7_i32
      opts%gtol   = 1.0e-6_dp
      X = X0

      call system_clock(t0, t_rate)
      call fc_solve(0_i32, R_ST, n_arr, p_arr, z_dum, l_dum, u_dum, X, &
        cb_bench_b7b, c_loc(ctx), opts, work, lwork_m7, iwork, liwork_m7, res)
      call system_clock(t1)

      ms_m7(s) = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      it_m7(s) = real(res%n_iter, dp)
      ev_m7(s) = real(res%n_eval, dp)
      tot_m7   = tot_m7 + ms_m7(s)
      if (res%status == FC_CONVERGED_GRAD) pass_m7 = pass_m7 + 1_i32
      if (res%kkt_residual > kkt_m7) kkt_m7 = res%kkt_residual
      if (res%orth_error > orth_m7) orth_m7 = res%orth_error

      ! 2. ABB (m = 0, first-order baseline, default 15000 cap)
      opts%memory = 0_i32
      X           = X0

      call system_clock(t0, t_rate)
      call fc_solve(0_i32, R_ST, n_arr, p_arr, z_dum, l_dum, u_dum, X, &
        cb_bench_b7b, c_loc(ctx), opts, work, lwork_m0, iwork, liwork_m0, res)
      call system_clock(t1)

      ms_m0(s) = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      it_m0(s) = real(res%n_iter, dp)
      ev_m0(s) = real(res%n_eval, dp)
      tot_m0   = tot_m0 + ms_m0(s)
      if (res%status == FC_CONVERGED_GRAD) pass_m0 = pass_m0 + 1_i32
      if (res%kkt_residual > kkt_m0) kkt_m0 = res%kkt_residual
      if (res%orth_error > orth_m0) orth_m0 = res%orth_error
    end do

    write (*, '(A,F6.2,A)') &
      '[3] B7B: St(200, 8) x St(24, 24) Multi-Stiefel — m=7 vs m=0 Speedup = ', &
      tot_m0 / max(tot_m7, 1.0e-9_dp), 'x'
    write (*, '(A,I0,A,I0,A,F10.3,A,F9.3,A,F9.3,A,F7.1,A,F7.1,A,ES9.2,A,ES9.2)') &
      '  LRBFGS (m=7) [', pass_m7, '/', N_SEEDS, ']: total = ', tot_m7, &
      ' ms | mean = ', tot_m7 / 10.0_dp, ' ms | med = ', median_10(ms_m7), &
      ' ms | med_it = ', median_10(it_m7), ' | med_ev = ', median_10(ev_m7), &
      ' | max_KKT = ', kkt_m7, ' | max_orth = ', orth_m7
    write (*, '(A,I0,A,I0,A,F10.3,A,F9.3,A,F9.3,A,F7.1,A,F7.1,A,ES9.2,A,ES9.2)') &
      '  ABB    (m=0) [', pass_m0, '/', N_SEEDS, ']: total = ', tot_m0, &
      ' ms | mean = ', tot_m0 / 10.0_dp, ' ms | med = ', median_10(ms_m0), &
      ' ms | med_it = ', median_10(it_m0), ' | med_ev = ', median_10(ev_m0), &
      ' | max_KKT = ', kkt_m0, ' | max_orth = ', orth_m0
  end subroutine bench_b7b_multistiefel

  !---------------------------------------------------------------------------
  ! (b.3) B8A: 1D Periodic Metallic Kohn-Sham DFT with Fractional Fermi-Dirac
  !            Occupations on {z in [0, 1]^12 : 1^T z = 5.0} x St(120, 12)
  !            via fc_solve_constrained (10 seeds)
  !---------------------------------------------------------------------------
  subroutine bench_b8a_kohn_sham_dft()
    integer(i32), parameter              :: D_BOX = 12_i32, R_ST = 1_i32, N_SEEDS = 10_i32
    integer(i32)                         :: n_arr(1), p_arr(1)
    integer(i32), allocatable            :: iwork(:)
    type(b8a_ctx_t), allocatable, target :: ctx
    type(fc_options_t)                   :: opts
    type(fc_result_t)                    :: res
    real(dp)                             :: l(12), u(12), a_ones(12), z(12), X(1440)
    real(dp)                             :: tau12(12), w120(120)
    real(dp), allocatable                :: work(:)
    real(dp)                             :: ms_arr(10), it_arr(10), ev_arr(10)
    real(dp)                             :: dom_len, dx, inv_dx2, x_i, x_j, dist_ij, b_nelec
    real(dp)                             :: d1, d2, v_ext_i, tot_ms, lin_err, max_lin, max_kkt, max_orth
    integer(i64)                         :: lwork_q, liwork_q, t0, t1, t_rate
    integer(i32)                         :: s, i, j, ip1, im1, rng_init, info, st, n_pass

    allocate (ctx)

    n_arr(1) = 120_i32
    p_arr(1) = 12_i32
    l        = 0.0_dp
    u        = 1.0_dp
    a_ones   = 1.0_dp
    b_nelec  = 5.0_dp

    ! Build 1D periodic metallic dimer Hamiltonian H0 and Yukawa-Coulomb K_coul
    dom_len      = 24.0_dp
    dx           = dom_len / 120.0_dp
    inv_dx2      = 1.0_dp / (dx * dx)
    ctx%alpha_H  = 0.35_dp
    ctx%kB_T     = 0.02_dp
    ctx%delta_FD = 1.0e-12_dp

    ctx%H0     = 0.0_dp
    ctx%K_coul = 0.0_dp
    do i = 1_i32, 120_i32
      x_i = real(i - 1_i32, dp) * dx
      ip1 = mod(i, 120_i32) + 1_i32
      im1 = mod(i + 118_i32, 120_i32) + 1_i32

      ! Periodic metallic dimer external potential (2 attractive wells + weak asymmetry)
      d1      = min(abs(x_i - 8.5_dp), dom_len - abs(x_i - 8.5_dp))
      d2      = min(abs(x_i - 15.5_dp), dom_len - abs(x_i - 15.5_dp))
      v_ext_i = -2.6_dp * exp(-0.45_dp * d1 * d1) - 2.55_dp * exp(-0.45_dp * d2 * d2) + &
                0.04_dp * cos(2.0_dp * 3.14159265358979323846_dp * x_i / dom_len)

      ctx%H0(i, i)   = inv_dx2 + v_ext_i
      ctx%H0(i, ip1) = -0.5_dp * inv_dx2
      ctx%H0(i, im1) = -0.5_dp * inv_dx2

      do j = 1_i32, 120_i32
        x_j     = real(j - 1_i32, dp) * dx
        dist_ij = min(abs(x_i - x_j), dom_len - abs(x_i - x_j))
        ctx%K_coul(i, j) = dx * exp(-0.15_dp * dist_ij) / sqrt(dist_ij * dist_ij + 1.0_dp)
      end do
    end do

    call fc_init_default_options(opts)
    opts%gtol = 1.0e-6_dp

    call fc_workspace_query(D_BOX, R_ST, n_arr, p_arr, opts%memory, lwork_q, liwork_q, st)
    if (st /= FC_SUCCESS_QUERY) then
      error stop 'B8A workspace query failed'
    end if
    allocate (work(lwork_q), iwork(liwork_q))

    tot_ms   = 0.0_dp
    max_lin  = 0.0_dp
    max_kkt  = 0.0_dp
    max_orth = 0.0_dp
    n_pass   = 0_i32

    do s = 1_i32, N_SEEDS
      rng_init = 55000_i32 + 6151_i32 * s
      do i = 1_i32, D_BOX
        z(i) = 0.5_dp * next_rand_u(rng_init) + 0.5_dp
      end do
      call fill_rand(1440_i32, X, rng_init)
      call fc_geqrf_pos(120_i32, 12_i32, X, tau12, w120, info)

      call system_clock(t0, t_rate)
      call fc_solve_constrained(D_BOX, R_ST, n_arr, p_arr, z, l, u, a_ones, b_nelec, X, &
        cb_bench_b8a, c_loc(ctx), opts, work, lwork_q, iwork, liwork_q, res)
      call system_clock(t1)

      ms_arr(s) = real(t1 - t0, dp) * 1000.0_dp / real(t_rate, dp)
      it_arr(s) = real(res%n_iter, dp)
      ev_arr(s) = real(res%n_eval, dp)
      tot_ms    = tot_ms + ms_arr(s)

      if (res%status == FC_CONVERGED_GRAD) n_pass = n_pass + 1_i32
      lin_err = abs(sum(z) - b_nelec)
      if (lin_err > max_lin) max_lin = lin_err
      if (res%kkt_residual > max_kkt) max_kkt = res%kkt_residual
      if (res%orth_error > max_orth) max_orth = res%orth_error
    end do

    write (*, '(A,I0,A,I0,A)') &
      '[4] B8A: {z in [0,1]^12 : 1^T z = 5} x St(120, 12) Kohn-Sham DFT (', &
      n_pass, '/', N_SEEDS, ' converged)'
    write (*, '(A,F10.3,A,F9.3,A,F9.3,A,F7.1,A,F7.1,A,ES9.2,A,ES9.2,A,ES9.2)') &
      '  LRBFGS-KKT:   total = ', tot_ms, ' ms | mean = ', tot_ms / 10.0_dp, &
      ' ms | med = ', median_10(ms_arr), ' ms | med_it = ', median_10(it_arr), &
      ' | med_ev = ', median_10(ev_arr), ' | max|1^Tz-5| = ', max_lin, &
      ' | max_KKT = ', max_kkt, ' | max_orth = ', max_orth
  end subroutine bench_b8a_kohn_sham_dft

end program fortchuck_bench
