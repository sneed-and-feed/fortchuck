#ifndef FORTCHUCK_H
#define FORTCHUCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status and error codes */
enum fc_status_e {
    FC_SUCCESS_QUERY       =   0,
    FC_CONVERGED_GRAD      =   1,
    FC_CONVERGED_FTOL      =   2,
    FC_CONVERGED_XTOL      =   3,
    FC_MAX_ITER_REACHED    =   4,
    FC_MAX_EVAL_REACHED    =   5,
    FC_ERR_INVALID_DIM     =  -1,
    FC_ERR_NULL_PTR        =  -2,
    FC_ERR_INVALID_BOUNDS  =  -3,
    FC_ERR_INVALID_X0      =  -4,
    FC_ERR_WORK_TOO_SMALL  =  -5,
    FC_ERR_CALLBACK_NAN    =  -6,
    FC_ERR_CALLBACK_ABORT  =  -7,
    FC_ERR_LS_FAILED       =  -8,
    FC_ERR_INVALID_OPTS    =  -9,
    FC_ERR_INFEASIBLE_LIN  = -10
};

enum fc_metric_e {
    FC_METRIC_CANONICAL = 0,  /* beta = 1/2 (default) */
    FC_METRIC_EUCLIDEAN = 1   /* beta = 1             */
};

enum fc_damping_e {
    FC_DAMP_CAUTIOUS = 0,     /* Huang-Absil-Gallivan cautious skip (default) */
    FC_DAMP_POWELL   = 1      /* Block-prior Riemannian Powell damping        */
};

/* 96-byte naturally aligned options struct (8 doubles + 8 int32s, 0 padding) */
typedef struct fc_options_t {
    double  gtol;             /* Default: 1.0e-6  */
    double  ftol;             /* Default: 1.0e-14 */
    double  xtol;             /* Default: 1.0e-14 */
    double  c1_armijo;        /* Default: 1.0e-4  */
    double  zh_eta;           /* Default: -1.0 (auto: 0.85 Zhang-Hager) */
    double  eps_band;         /* Default: 1.0e-4  */
    double  orth_tol;         /* Default: 1.0e-13 */
    double  max_step_vel;     /* Default: 2*pi    */
    int32_t max_iter;         /* Default: 15000   */
    int32_t max_eval;         /* Default: 15000   */
    int32_t memory;           /* Default: 7 (0 = pure ABB/SPG steepest descent) */
    int32_t metric;           /* Default: FC_METRIC_CANONICAL (0) */
    int32_t damping_mode;     /* Default: FC_DAMP_CAUTIOUS (0)    */
    int32_t project_initial;  /* Default: 1 (project z0, polish full-rank X0) */
    int32_t max_ls_steps;     /* Default: 40      */
    int32_t reserved_flags;   /* Default: 0       */
} fc_options_t;

/* 88-byte naturally aligned result struct (6 doubles + 2 int64s + 6 int32s, 0 padding) */
typedef struct fc_result_t {
    double  f_val;
    double  kkt_residual;
    double  box_residual;
    double  stiefel_residual;
    double  orth_error;
    double  last_step_size;
    int64_t lwork_min;
    int64_t liwork_min;
    int32_t status;
    int32_t n_iter;
    int32_t n_eval;
    int32_t n_reorth;
    int32_t n_fallbacks;
    int32_t n_active_box;
} fc_result_t;

/* Objective + Euclidean partial gradient callback:
 * Return 0 on success, >0 on recoverable domain error (triggers step halving),
 * <0 to abort optimization immediately with FC_ERR_CALLBACK_ABORT. */
typedef int32_t (*fc_eval_fg_cb)(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *X_packed,
    double *f_out,
    double *gz_out,
    double *GX_packed_out,
    void *udata
);

void fortchuck_default_options(fc_options_t *opts);

int32_t fortchuck_workspace_query(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    int32_t memory,
    int64_t *lwork_out,
    int64_t *liwork_out
);

int32_t fortchuck_solve(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    double *z,
    const double *l,
    const double *u,
    double *X_packed,
    fc_eval_fg_cb eval_fg,
    void *udata,
    const fc_options_t *opts,
    double *work,
    int64_t lwork,
    int32_t *iwork,
    int64_t liwork,
    fc_result_t *res
);

/* Coupled singly-linearly-constrained Box / Simplex x Multi-Stiefel solver (v0.2.0):
 * Minimizes f(z, X_1..X_R) subject to l_i <= z_i <= u_i, a^T z = b, and X_r^T X_r = I_{p_r}. */
int32_t fortchuck_solve_constrained(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    double *z,
    const double *l,
    const double *u,
    const double *a,
    double b,
    double *X_packed,
    fc_eval_fg_cb eval_fg,
    void *udata,
    const fc_options_t *opts,
    double *work,
    int64_t lwork,
    int32_t *iwork,
    int64_t liwork,
    fc_result_t *res
);

int32_t fortchuck_project_stiefel(
    int32_t n,
    int32_t p,
    double *X,
    double *work,
    int64_t lwork
);

/* Zero-workspace Euclidean projection onto {z in [l, u]^d : a^T z = b} (v0.2.0):
 * Projects z[0..d-1] in-place; writes dual multiplier to *nu_out if nu_out != NULL. */
int32_t fortchuck_project_linear_box(
    int32_t d,
    const double *l,
    const double *u,
    const double *a,
    double b,
    double *z,
    double *nu_out
);

int32_t fortchuck_kkt_residual(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *l,
    const double *u,
    const double *X_packed,
    const double *gz,
    const double *GX_packed,
    int32_t metric,
    double *work,
    int64_t lwork,
    double *kkt_inf_out,
    double *orth_err_out
);

/* Constrained KKT & linear/orthogonality feasibility certificate (v0.2.0):
 * Supports lwork == -1 workspace query (minimum lwork = max(1, d + max_r(n_r*p_r + 3*p_r^2))). */
int32_t fortchuck_kkt_residual_constrained(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *l,
    const double *u,
    const double *a,
    double b,
    const double *X_packed,
    const double *gz,
    const double *GX_packed,
    int32_t metric,
    double *work,
    int64_t lwork,
    double *kkt_inf_out,
    double *lin_err_out,
    double *orth_err_out
);

#ifdef __cplusplus
}
#endif

#endif /* FORTCHUCK_H */
