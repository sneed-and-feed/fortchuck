module fortchuck
  use fortchuck_kinds, only: &
    dp, i32, i64, FC_EPS, FC_HUGE, FC_PI, &
    c_ptr, c_funptr, c_null_ptr, c_null_funptr, &
    FC_SUCCESS_QUERY, FC_CONVERGED_GRAD, FC_CONVERGED_FTOL, FC_CONVERGED_XTOL, &
    FC_MAX_ITER_REACHED, FC_MAX_EVAL_REACHED, &
    FC_ERR_INVALID_DIM, FC_ERR_NULL_PTR, FC_ERR_INVALID_BOUNDS, FC_ERR_INVALID_X0, &
    FC_ERR_WORK_TOO_SMALL, FC_ERR_CALLBACK_NAN, FC_ERR_CALLBACK_ABORT, &
    FC_ERR_LS_FAILED, FC_ERR_INVALID_OPTS, &
    FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN, &
    FC_DAMP_CAUTIOUS, FC_DAMP_POWELL, &
    FC_BOX_FREE, FC_BOX_ACTIVE, FC_BOX_FIXED, &
    fc_options_t, fc_result_t, fc_eval_fg_cb, &
    fc_init_default_options, fc_init_result
  use fortchuck_solver, only: &
    fc_validate_options, fc_workspace_query, fc_solve, &
    fc_project_stiefel, fc_kkt_residual
  use fortchuck_c_abi, only: &
    fortchuck_default_options, fortchuck_workspace_query, fortchuck_solve, &
    fortchuck_project_stiefel, fortchuck_kkt_residual
  implicit none (type, external)
  public
end module fortchuck
