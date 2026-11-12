module fortchuck_kinds
  use, intrinsic :: iso_c_binding, only: &
    c_double, c_int, c_int32_t, c_int64_t, c_ptr, c_funptr, c_null_ptr, c_null_funptr
  implicit none (type, external)
  private

  ! Precision and integer kind parameters
  integer, parameter, public :: dp  = c_double
  integer, parameter, public :: i32 = c_int
  integer, parameter, public :: i64 = c_int64_t

  ! Fundamental floating-point constants
  real(dp), parameter, public :: FC_EPS  = epsilon(1.0_dp)
  real(dp), parameter, public :: FC_HUGE = huge(1.0_dp)
  real(dp), parameter, public :: FC_PI   = 3.14159265358979323846264338327950288_dp

  ! Deterministic solver status and error codes
  integer(i32), parameter, public :: FC_SUCCESS_QUERY      =  0_i32
  integer(i32), parameter, public :: FC_CONVERGED_GRAD     =  1_i32
  integer(i32), parameter, public :: FC_CONVERGED_FTOL     =  2_i32
  integer(i32), parameter, public :: FC_CONVERGED_XTOL     =  3_i32
  integer(i32), parameter, public :: FC_MAX_ITER_REACHED   =  4_i32
  integer(i32), parameter, public :: FC_MAX_EVAL_REACHED   =  5_i32
  integer(i32), parameter, public :: FC_ERR_INVALID_DIM    = -1_i32
  integer(i32), parameter, public :: FC_ERR_NULL_PTR       = -2_i32
  integer(i32), parameter, public :: FC_ERR_INVALID_BOUNDS = -3_i32
  integer(i32), parameter, public :: FC_ERR_INVALID_X0     = -4_i32
  integer(i32), parameter, public :: FC_ERR_WORK_TOO_SMALL = -5_i32
  integer(i32), parameter, public :: FC_ERR_CALLBACK_NAN   = -6_i32
  integer(i32), parameter, public :: FC_ERR_CALLBACK_ABORT = -7_i32
  integer(i32), parameter, public :: FC_ERR_LS_FAILED      = -8_i32
  integer(i32), parameter, public :: FC_ERR_INVALID_OPTS   = -9_i32

  ! Stiefel Riemannian metric family selectors
  integer(i32), parameter, public :: FC_METRIC_CANONICAL = 0_i32
  integer(i32), parameter, public :: FC_METRIC_EUCLIDEAN = 1_i32

  ! Quasi-Newton curvature positivity guard selectors
  integer(i32), parameter, public :: FC_DAMP_CAUTIOUS = 0_i32
  integer(i32), parameter, public :: FC_DAMP_POWELL   = 1_i32

  ! Bertsekas box active-band partition tags
  integer(i32), parameter, public :: FC_BOX_FREE   = 0_i32
  integer(i32), parameter, public :: FC_BOX_ACTIVE = 1_i32
  integer(i32), parameter, public :: FC_BOX_FIXED  = 2_i32

  ! Re-export C pointer types/constants for downstream solver & C-ABI modules
  public :: c_ptr, c_funptr, c_null_ptr, c_null_funptr

  ! 96-byte naturally aligned options struct (8 doubles + 8 int32s, 0 padding)
  type, bind(C), public :: fc_options_t
    real(c_double)     :: gtol            = 1.0e-6_dp
    real(c_double)     :: ftol            = 1.0e-14_dp
    real(c_double)     :: xtol            = 1.0e-14_dp
    real(c_double)     :: c1_armijo       = 1.0e-4_dp
    real(c_double)     :: zh_eta          = -1.0_dp
    real(c_double)     :: eps_band        = 1.0e-4_dp
    real(c_double)     :: orth_tol        = 1.0e-13_dp
    real(c_double)     :: max_step_vel    = 2.0_dp * FC_PI
    integer(c_int32_t) :: max_iter        = 15000_i32
    integer(c_int32_t) :: max_eval        = 15000_i32
    integer(c_int32_t) :: memory          = 7_i32
    integer(c_int32_t) :: metric          = FC_METRIC_CANONICAL
    integer(c_int32_t) :: damping_mode    = FC_DAMP_CAUTIOUS
    integer(c_int32_t) :: project_initial = 1_i32
    integer(c_int32_t) :: max_ls_steps    = 40_i32
    integer(c_int32_t) :: reserved_flags  = 0_i32
  end type fc_options_t

  ! 88-byte naturally aligned result struct (6 doubles + 2 int64s + 6 int32s, 0 padding)
  type, bind(C), public :: fc_result_t
    real(c_double)     :: f_val            = 0.0_dp
    real(c_double)     :: kkt_residual     = 0.0_dp
    real(c_double)     :: box_residual     = 0.0_dp
    real(c_double)     :: stiefel_residual = 0.0_dp
    real(c_double)     :: orth_error       = 0.0_dp
    real(c_double)     :: last_step_size   = 0.0_dp
    integer(c_int64_t) :: lwork_min        = 0_i64
    integer(c_int64_t) :: liwork_min       = 0_i64
    integer(c_int32_t) :: status           = FC_SUCCESS_QUERY
    integer(c_int32_t) :: n_iter           = 0_i32
    integer(c_int32_t) :: n_eval           = 0_i32
    integer(c_int32_t) :: n_reorth         = 0_i32
    integer(c_int32_t) :: n_fallbacks      = 0_i32
    integer(c_int32_t) :: n_active_box     = 0_i32
  end type fc_result_t

  ! Objective + Euclidean partial gradient callback interface matching fortchuck.h
  abstract interface
    function fc_eval_fg_cb(d, R, n_arr, p_arr, z, X_packed, &
        f_out, gz_out, GX_packed_out, udata) bind(C) result(rc)
      import :: c_int32_t, c_double, c_ptr
      implicit none (type, external)
      integer(c_int32_t), value, intent(in) :: d
      integer(c_int32_t), value, intent(in) :: R
      integer(c_int32_t), intent(in)        :: n_arr(*)
      integer(c_int32_t), intent(in)        :: p_arr(*)
      real(c_double), intent(in)            :: z(*)
      real(c_double), intent(in)            :: X_packed(*)
      real(c_double), intent(out)           :: f_out
      real(c_double), intent(inout)         :: gz_out(*)
      real(c_double), intent(inout)         :: GX_packed_out(*)
      type(c_ptr), value, intent(in)        :: udata
      integer(c_int32_t)                    :: rc
    end function fc_eval_fg_cb
  end interface

  public :: fc_eval_fg_cb
  public :: fc_init_default_options
  public :: fc_init_result

contains

  pure subroutine fc_init_default_options(opts)
    type(fc_options_t), intent(out) :: opts

    opts%gtol            = 1.0e-6_dp
    opts%ftol            = 1.0e-14_dp
    opts%xtol            = 1.0e-14_dp
    opts%c1_armijo       = 1.0e-4_dp
    opts%zh_eta          = -1.0_dp
    opts%eps_band        = 1.0e-4_dp
    opts%orth_tol        = 1.0e-13_dp
    opts%max_step_vel    = 2.0_dp * FC_PI
    opts%max_iter        = 15000_i32
    opts%max_eval        = 15000_i32
    opts%memory          = 7_i32
    opts%metric          = FC_METRIC_CANONICAL
    opts%damping_mode    = FC_DAMP_CAUTIOUS
    opts%project_initial = 1_i32
    opts%max_ls_steps    = 40_i32
    opts%reserved_flags  = 0_i32
  end subroutine fc_init_default_options

  pure subroutine fc_init_result(res)
    type(fc_result_t), intent(out) :: res

    res%f_val            = 0.0_dp
    res%kkt_residual     = 0.0_dp
    res%box_residual     = 0.0_dp
    res%stiefel_residual = 0.0_dp
    res%orth_error       = 0.0_dp
    res%last_step_size   = 0.0_dp
    res%lwork_min        = 0_i64
    res%liwork_min       = 0_i64
    res%status           = FC_SUCCESS_QUERY
    res%n_iter           = 0_i32
    res%n_eval           = 0_i32
    res%n_reorth         = 0_i32
    res%n_fallbacks      = 0_i32
    res%n_active_box     = 0_i32
  end subroutine fc_init_result

end module fortchuck_kinds
