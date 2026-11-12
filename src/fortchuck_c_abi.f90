module fortchuck_c_abi
  use, intrinsic :: iso_c_binding, only: &
    c_int32_t, c_int64_t, c_ptr, c_funptr, &
    c_associated, c_f_pointer, c_f_procpointer
  use fortchuck_kinds, only: &
    dp, i32, i64, &
    FC_SUCCESS_QUERY, FC_ERR_INVALID_DIM, FC_ERR_NULL_PTR, FC_ERR_WORK_TOO_SMALL, &
    fc_options_t, fc_result_t, fc_eval_fg_cb, &
    fc_init_default_options, fc_init_result
  use fortchuck_solver, only: &
    fc_validate_options, fc_workspace_query, fc_solve, &
    fc_project_stiefel, fc_kkt_residual
  implicit none (type, external)
  private

  public :: fortchuck_default_options
  public :: fortchuck_workspace_query
  public :: fortchuck_solve
  public :: fortchuck_project_stiefel
  public :: fortchuck_kkt_residual

contains

  !---------------------------------------------------------------------------
  ! void fortchuck_default_options(fc_options_t *opts)
  !---------------------------------------------------------------------------
  subroutine fortchuck_default_options(opts_ptr) bind(C, name="fortchuck_default_options")
    type(c_ptr), value, intent(in) :: opts_ptr
    type(fc_options_t), pointer    :: opts_f

    if (.not. c_associated(opts_ptr)) return
    call c_f_pointer(opts_ptr, opts_f)
    call fc_init_default_options(opts_f)
  end subroutine fortchuck_default_options

  !---------------------------------------------------------------------------
  ! int32_t fortchuck_workspace_query(...)
  ! Null-safe when R == 0 and n_arr / p_arr == NULL.
  !---------------------------------------------------------------------------
  function fortchuck_workspace_query( &
      d, R, n_arr_ptr, p_arr_ptr, memory, lwork_ptr, liwork_ptr) &
      bind(C, name="fortchuck_workspace_query") result(status)
    integer(c_int32_t), value, intent(in) :: d, R, memory
    type(c_ptr), value, intent(in)        :: n_arr_ptr, p_arr_ptr
    type(c_ptr), value, intent(in)        :: lwork_ptr, liwork_ptr
    integer(c_int32_t)                    :: status

    integer(i64), pointer             :: lwork_f, liwork_f
    integer(i32), pointer, contiguous :: n_arr_f(:), p_arr_f(:)
    integer(i32)                      :: dummy_i32(1)
    integer(i64)                      :: shp1(1)

    if (.not. c_associated(lwork_ptr) .or. .not. c_associated(liwork_ptr)) then
      status = FC_ERR_NULL_PTR
      return
    end if

    call c_f_pointer(lwork_ptr, lwork_f)
    call c_f_pointer(liwork_ptr, liwork_f)
    lwork_f  = 0_i64
    liwork_f = 0_i64

    if (d < 0_i32 .or. R < 0_i32 .or. (d == 0_i32 .and. R == 0_i32) .or. memory < 0_i32) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    if (R > 0_i32) then
      if (.not. c_associated(n_arr_ptr) .or. .not. c_associated(p_arr_ptr)) then
        status = FC_ERR_NULL_PTR
        return
      end if
      shp1(1) = int(R, i64)
      call c_f_pointer(n_arr_ptr, n_arr_f, shp1)
      call c_f_pointer(p_arr_ptr, p_arr_f, shp1)
      call fc_workspace_query(d, R, n_arr_f, p_arr_f, memory, lwork_f, liwork_f, status)
    else
      dummy_i32(1) = 1_i32
      call fc_workspace_query(d, 0_i32, dummy_i32, dummy_i32, memory, lwork_f, liwork_f, status)
    end if
  end function fortchuck_workspace_query

  !---------------------------------------------------------------------------
  ! int32_t fortchuck_solve(...)
  ! Null-safe when d == 0 (z/l/u == NULL), R == 0 (n_arr/p_arr/X_packed == NULL),
  ! opts == NULL (uses defaults), or lwork == -1 / liwork == -1 (query mode).
  !---------------------------------------------------------------------------
  function fortchuck_solve( &
      d, R, n_arr_ptr, p_arr_ptr, z_ptr, l_ptr, u_ptr, X_ptr, &
      eval_fg_c, udata, opts_ptr, work_ptr, lwork, iwork_ptr, liwork, res_ptr) &
      bind(C, name="fortchuck_solve") result(status)
    integer(c_int32_t), value, intent(in) :: d, R
    type(c_ptr), value, intent(in)        :: n_arr_ptr, p_arr_ptr
    type(c_ptr), value, intent(in)        :: z_ptr, l_ptr, u_ptr, X_ptr
    type(c_funptr), value, intent(in)     :: eval_fg_c
    type(c_ptr), value, intent(in)        :: udata, opts_ptr, work_ptr, iwork_ptr, res_ptr
    integer(c_int64_t), value, intent(in) :: lwork, liwork
    integer(c_int32_t)                    :: status

    type(fc_result_t), pointer        :: res_f
    type(fc_options_t), pointer       :: opts_in
    type(fc_options_t)                :: opts_local
    procedure(fc_eval_fg_cb), pointer :: eval_fg_f
    integer(i32), pointer, contiguous :: n_arr_f(:), p_arr_f(:), iwork_f(:)
    real(dp), pointer, contiguous     :: z_f(:), l_f(:), u_f(:), X_f(:), work_f(:)
    integer(i32), target              :: dummy_n(1), dummy_p(1)
    real(dp), target                  :: dummy_z(1), dummy_l(1), dummy_u(1), dummy_X(1)
    integer(i64)                      :: lwork_min, liwork_min, n_st_tot, shp1(1)
    integer(i32)                      :: r_idx

    if (.not. c_associated(res_ptr)) then
      status = FC_ERR_NULL_PTR
      return
    end if
    call c_f_pointer(res_ptr, res_f)
    call fc_init_result(res_f)

    if (c_associated(opts_ptr)) then
      call c_f_pointer(opts_ptr, opts_in)
      opts_local = opts_in
    else
      call fc_init_default_options(opts_local)
    end if

    if (d < 0_i32 .or. R < 0_i32 .or. (d == 0_i32 .and. R == 0_i32) .or. &
        opts_local%memory < 0_i32) then
      status       = FC_ERR_INVALID_DIM
      res_f%status = status
      return
    end if

    dummy_n(1) = 1_i32
    dummy_p(1) = 1_i32
    if (R > 0_i32) then
      if (.not. c_associated(n_arr_ptr) .or. .not. c_associated(p_arr_ptr)) then
        status       = FC_ERR_NULL_PTR
        res_f%status = status
        return
      end if
      shp1(1) = int(R, i64)
      call c_f_pointer(n_arr_ptr, n_arr_f, shp1)
      call c_f_pointer(p_arr_ptr, p_arr_f, shp1)
    else
      n_arr_f => dummy_n
      p_arr_f => dummy_p
    end if

    call fc_workspace_query(d, R, n_arr_f, p_arr_f, opts_local%memory, lwork_min, liwork_min, status)
    if (status /= 0_i32) then
      res_f%status = status
      return
    end if

    call fc_validate_options(opts_local, status)
    if (status /= 0_i32) then
      res_f%status = status
      return
    end if

    res_f%lwork_min  = lwork_min
    res_f%liwork_min = liwork_min

    ! Workspace query mode (lwork == -1 or liwork == -1)
    if (lwork == -1_i64 .or. liwork == -1_i64) then
      shp1(1) = 1_i64
      if (c_associated(work_ptr)) then
        call c_f_pointer(work_ptr, work_f, shp1)
        work_f(1) = real(lwork_min, dp)
      end if
      if (c_associated(iwork_ptr)) then
        call c_f_pointer(iwork_ptr, iwork_f, shp1)
        iwork_f(1) = int(min(liwork_min, int(huge(0_i32), i64)), i32)
      end if
      status       = FC_SUCCESS_QUERY
      res_f%status = status
      return
    end if

    ! Required pointer validation for actual solve
    dummy_z(1) = 0.0_dp
    dummy_l(1) = 0.0_dp
    dummy_u(1) = 0.0_dp
    if (d > 0_i32) then
      if (.not. c_associated(z_ptr) .or. .not. c_associated(l_ptr) .or. &
          .not. c_associated(u_ptr)) then
        status       = FC_ERR_NULL_PTR
        res_f%status = status
        return
      end if
      shp1(1) = int(d, i64)
      call c_f_pointer(z_ptr, z_f, shp1)
      call c_f_pointer(l_ptr, l_f, shp1)
      call c_f_pointer(u_ptr, u_f, shp1)
    else
      z_f => dummy_z
      l_f => dummy_l
      u_f => dummy_u
    end if

    dummy_X(1) = 0.0_dp
    n_st_tot   = 0_i64
    if (R > 0_i32) then
      if (.not. c_associated(X_ptr)) then
        status       = FC_ERR_NULL_PTR
        res_f%status = status
        return
      end if
      do r_idx = 1_i32, R
        n_st_tot = n_st_tot + int(n_arr_f(r_idx), i64) * int(p_arr_f(r_idx), i64)
      end do
      shp1(1) = n_st_tot
      call c_f_pointer(X_ptr, X_f, shp1)
    else
      X_f => dummy_X
    end if

    if (.not. c_associated(eval_fg_c)) then
      status       = FC_ERR_NULL_PTR
      res_f%status = status
      return
    end if
    if (.not. c_associated(work_ptr) .or. .not. c_associated(iwork_ptr)) then
      status       = FC_ERR_NULL_PTR
      res_f%status = status
      return
    end if

    if (lwork < lwork_min .or. liwork < liwork_min) then
      status       = FC_ERR_WORK_TOO_SMALL
      res_f%status = status
      return
    end if

    call c_f_procpointer(eval_fg_c, eval_fg_f)
    shp1(1) = lwork
    call c_f_pointer(work_ptr, work_f, shp1)
    shp1(1) = liwork
    call c_f_pointer(iwork_ptr, iwork_f, shp1)

    call fc_solve(d, R, n_arr_f, p_arr_f, z_f, l_f, u_f, X_f, &
      eval_fg_f, udata, opts_local, work_f, lwork, iwork_f, liwork, res_f)
    status = res_f%status
  end function fortchuck_solve

  !---------------------------------------------------------------------------
  ! int32_t fortchuck_project_stiefel(int32_t n, int32_t p, double *X, double *work, int64_t lwork)
  !---------------------------------------------------------------------------
  function fortchuck_project_stiefel(n, p, X_ptr, work_ptr, lwork) &
      bind(C, name="fortchuck_project_stiefel") result(status)
    integer(c_int32_t), value, intent(in) :: n, p
    type(c_ptr), value, intent(in)        :: X_ptr, work_ptr
    integer(c_int64_t), value, intent(in) :: lwork
    integer(c_int32_t)                    :: status

    real(dp), pointer, contiguous :: X_f(:), work_f(:)
    integer(i64)                  :: lwork_min, shp1(1)

    if (n < 1_i32 .or. p < 1_i32 .or. p > n) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    lwork_min = 2_i64 * int(p, i64)
    if (lwork == -1_i64) then
      if (c_associated(work_ptr)) then
        shp1(1) = 1_i64
        call c_f_pointer(work_ptr, work_f, shp1)
        work_f(1) = real(lwork_min, dp)
      end if
      status = FC_SUCCESS_QUERY
      return
    end if

    if (.not. c_associated(X_ptr) .or. .not. c_associated(work_ptr)) then
      status = FC_ERR_NULL_PTR
      return
    end if
    if (lwork < lwork_min) then
      status = FC_ERR_WORK_TOO_SMALL
      return
    end if

    shp1(1) = int(n, i64) * int(p, i64)
    call c_f_pointer(X_ptr, X_f, shp1)
    shp1(1) = lwork
    call c_f_pointer(work_ptr, work_f, shp1)
    call fc_project_stiefel(n, p, X_f, work_f, lwork, status)
  end function fortchuck_project_stiefel

  !---------------------------------------------------------------------------
  ! int32_t fortchuck_kkt_residual(...)
  !---------------------------------------------------------------------------
  function fortchuck_kkt_residual( &
      d, R, n_arr_ptr, p_arr_ptr, z_ptr, l_ptr, u_ptr, X_ptr, &
      gz_ptr, GX_ptr, metric, work_ptr, lwork, kkt_inf_ptr, orth_err_ptr) &
      bind(C, name="fortchuck_kkt_residual") result(status)
    integer(c_int32_t), value, intent(in) :: d, R, metric
    type(c_ptr), value, intent(in)        :: n_arr_ptr, p_arr_ptr
    type(c_ptr), value, intent(in)        :: z_ptr, l_ptr, u_ptr, X_ptr
    type(c_ptr), value, intent(in)        :: gz_ptr, GX_ptr, work_ptr
    integer(c_int64_t), value, intent(in) :: lwork
    type(c_ptr), value, intent(in)        :: kkt_inf_ptr, orth_err_ptr
    integer(c_int32_t)                    :: status

    integer(i32), pointer, contiguous :: n_arr_f(:), p_arr_f(:)
    real(dp), pointer, contiguous     :: z_f(:), l_f(:), u_f(:), X_f(:)
    real(dp), pointer, contiguous     :: gz_f(:), GX_f(:), work_f(:)
    real(dp), pointer                 :: kkt_f, orth_f
    integer(i32), target              :: dummy_n(1), dummy_p(1)
    real(dp), target                  :: dummy_r(1), dummy_w(1)
    real(dp)                          :: kkt_tmp, orth_tmp
    integer(i64)                      :: n_st_tot, shp1(1)
    integer(i32)                      :: r_idx, nr, pr

    if (d < 0_i32 .or. R < 0_i32 .or. (d == 0_i32 .and. R == 0_i32)) then
      status = FC_ERR_INVALID_DIM
      return
    end if

    dummy_n(1) = 1_i32
    dummy_p(1) = 1_i32
    n_st_tot   = 0_i64
    if (R > 0_i32) then
      if (.not. c_associated(n_arr_ptr) .or. .not. c_associated(p_arr_ptr)) then
        status = FC_ERR_NULL_PTR
        return
      end if
      shp1(1) = int(R, i64)
      call c_f_pointer(n_arr_ptr, n_arr_f, shp1)
      call c_f_pointer(p_arr_ptr, p_arr_f, shp1)
      do r_idx = 1_i32, R
        nr = n_arr_f(r_idx)
        pr = p_arr_f(r_idx)
        if (nr < 1_i32 .or. pr < 1_i32 .or. pr > nr) then
          status = FC_ERR_INVALID_DIM
          return
        end if
        n_st_tot = n_st_tot + int(nr, i64) * int(pr, i64)
      end do
    else
      n_arr_f => dummy_n
      p_arr_f => dummy_p
    end if

    dummy_r(1) = 0.0_dp
    dummy_w(1) = 0.0_dp

    ! Workspace query mode (lwork == -1)
    if (lwork == -1_i64) then
      call fc_kkt_residual(d, R, n_arr_f, p_arr_f, dummy_r, dummy_r, dummy_r, dummy_r, &
        dummy_r, dummy_r, metric, dummy_w, -1_i64, kkt_tmp, orth_tmp, status)
      if (status == FC_SUCCESS_QUERY .and. c_associated(work_ptr)) then
        shp1(1) = 1_i64
        call c_f_pointer(work_ptr, work_f, shp1)
        work_f(1) = dummy_w(1)
      end if
      return
    end if

    if (.not. c_associated(kkt_inf_ptr) .or. .not. c_associated(orth_err_ptr) .or. &
        .not. c_associated(work_ptr)) then
      status = FC_ERR_NULL_PTR
      return
    end if

    if (d > 0_i32) then
      if (.not. c_associated(z_ptr) .or. .not. c_associated(l_ptr) .or. &
          .not. c_associated(u_ptr) .or. .not. c_associated(gz_ptr)) then
        status = FC_ERR_NULL_PTR
        return
      end if
      shp1(1) = int(d, i64)
      call c_f_pointer(z_ptr,  z_f,  shp1)
      call c_f_pointer(l_ptr,  l_f,  shp1)
      call c_f_pointer(u_ptr,  u_f,  shp1)
      call c_f_pointer(gz_ptr, gz_f, shp1)
    else
      z_f  => dummy_r
      l_f  => dummy_r
      u_f  => dummy_r
      gz_f => dummy_r
    end if

    if (R > 0_i32) then
      if (.not. c_associated(X_ptr) .or. .not. c_associated(GX_ptr)) then
        status = FC_ERR_NULL_PTR
        return
      end if
      shp1(1) = n_st_tot
      call c_f_pointer(X_ptr,  X_f,  shp1)
      call c_f_pointer(GX_ptr, GX_f, shp1)
    else
      X_f  => dummy_r
      GX_f => dummy_r
    end if

    if (lwork < 1_i64) then
      status = FC_ERR_WORK_TOO_SMALL
      return
    end if

    shp1(1) = lwork
    call c_f_pointer(work_ptr, work_f, shp1)
    call c_f_pointer(kkt_inf_ptr, kkt_f)
    call c_f_pointer(orth_err_ptr, orth_f)

    call fc_kkt_residual(d, R, n_arr_f, p_arr_f, z_f, l_f, u_f, X_f, &
      gz_f, GX_f, metric, work_f, lwork, kkt_f, orth_f, status)
  end function fortchuck_kkt_residual

end module fortchuck_c_abi
