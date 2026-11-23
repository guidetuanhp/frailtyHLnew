#ifndef FHL_LINALG_H
#define FHL_LINALG_H

#include "fhl_core.h"

static inline int fhl_safe_add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static inline int fhl_safe_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static inline int fhl_safe_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab;
    if (!fhl_safe_mul_size(a, b, &ab)) return 0;
    return fhl_safe_mul_size(ab, c, out);
}

void fhl_c_linalg_begin_malloc_scope(void);
int fhl_c_linalg_end_malloc_scope(void);
int fhl_c_linalg_had_oom(void);

/* ==========================================================================
 * Risk-set operator descriptor for O(n + m) survival cumulative sums
 * Supports both right-censored nested risk sets (L0 == 0) and
 * general left-truncated interval risk sets (L0 >= 0, (L_i, y_i]).
 * ========================================================================== */
typedef struct {
    int n;              /* Number of subjects (rows of Mi) */
    int m;              /* Number of unique event time points (cols of Mi, idx2) */
    int nested;         /* 1 if L0 == 0 everywhere (nested right-censored), 0 if general left-truncated */
    const int *lo;      /* Length n: 0-based count of sorted event times <= L_i */
    const int *hi;      /* Length n: 0-based count of sorted event times <= y_i */
    const int *ord;     /* Length m: 1-based permutation sorting unique event times */
    const int *inv;     /* Length m: 1-based inverse permutation of ord */
    const int *obs;     /* Length n: 1-based subject order sorted by decreasing hi (when nested == 1) */
} fhl_c_risk_t;

/* Unpack an R fhl_build_risk_set() descriptor list into fhl_c_risk_t in O(1) */
void fhl_c_unpack_risk(SEXP s_Mi, fhl_c_risk_t *risk);

/* Compute out = Mi %*% b (length n) from b (length m) in O(n + m) */
void fhl_c_risk_forward(const fhl_c_risk_t *risk, const double *b, double *out, double *work_m1);

/* Compute out = t(Mi) %*% b (length m) from b (length n) in O(n + m) */
void fhl_c_risk_backward(const fhl_c_risk_t *risk, const double *b, double *out, double *work_3m3);

/* ==========================================================================
 * Cholesky SPD, Block Schur-Complement, and Log-Determinant Solvers
 * ========================================================================== */

/* Strict SPD matrix inverse (n x n) via single-pass Cholesky factorization
 * (AVX2/OpenMP blocked Cholesky for n >= 64, LAPACK dpotrf + dpotri for n < 64).
 * Never falls back to LU/dgesv; returns 1 on strict SPD success, 0 otherwise. */
int fhl_c_solve_inv(const double *A, double *inv, int n);

/* Strict SPD linear solve A * X = B (n x nrhs) via LAPACK dpotrf + dpotrs.
 * Never falls back to LU/dgesv; returns 1 on strict SPD success, 0 otherwise. */
int fhl_c_solve_spd(const double *A, const double *B, double *X, int n, int nrhs);

/* Solve general non-symmetric/indefinite system A * X = B (n x nrhs) via LAPACK dgesv_ */
int fhl_c_solve_gesv(const double *A, const double *B, double *X, int n, int nrhs);

/* Invert general non-SPD n x n matrix A^{-1} via LAPACK dgesv_ on identity RHS */
int fhl_c_solve_gesv_inv(const double *A, double *inv, int n);

/* Block Schur-complement inversion of strictly SPD (p + Q) x (p + Q) Hessian H.
 * Both H_zz (when Dinv_in == NULL) and Schur complement S are inverted via
 * strict SPD Cholesky fhl_c_solve_inv(). */
int fhl_c_lognormal_inverse(
    const double *H, int p, int Q,
    const double *Dinv_in,
    double *Hinv_out,
    double *Dinv_out
);

/* Block Schur-complement linear solve H * step = rhs for strictly SPD H.
 * Uses Cholesky dpotrf + dpotrs on H_zz (when Dinv_in == NULL) and on S;
 * never falls back to indefinite dgesv. */
int fhl_c_schur_solve_step(
    const double *H, int p, int Q,
    const double *Dinv_in,
    const double *rhs,
    double *step_out
);

/* Exact scaled Schur-Sylvester factorization/solve/inverse/logdet for
 * H_act = [ H0_bb, H0_ba ; H0_ab, H0_aa + D_act^{-1} ],
 * where d_diag[r] > 0 marks active RE rows/cols (<= 0 -> inactive: v_r = 0, Hinv row/col = 0). */
int fhl_c_schur_scaled_eval(
    int p, int Q,
    const double *H0,
    const double *d_diag,
    const double *rhs,
    double pi_const,
    double *Hinv_out,
    double *H22_inv_out,
    double *step_out,
    double *logdet_W_out,
    double *logdet_S2pi_out,
    double *R_diag_out
);

/* Fast exact trace tr((Hinv %*% Hd1) %*% (Hinv %*% Hd2)):
 * Skips inactive (zero) rows/columns of Hd1 and Hd2, and uses
 * OpenMP + AVX2/FMA symmetric multiplication when Hd is dense and k >= 64. */
double fhl_c_trace_hinv_hd_prod(
    const double *Hinv,
    const double *Hd1,
    const double *Hd2,
    int k
);

/* Exact matrix trace tr(A %*% B) = sum_{i,j} A[i,j] * B[j,i] in long double */
double fhl_c_trace_ab(const double *A, const double *B, int n);

/* Compute log|M| for n x n SPD matrix M via Cholesky (or bounded SVD when explicit bounds are set). */
double fhl_c_logdet_spd(
    const double *M, int n,
    int has_lower, double lower_clip,
    int has_upper, double upper_clip
);

/* Helper: lookup list element by name */
SEXP fhl_c_get_list_elt(SEXP list, const char *name);

/* Helper: obtain const double* from REALSXP or INTSXP/LGLSXP vector/matrix */
const double *fhl_c_real_ptr(SEXP s);

/* ==========================================================================
 * 4. Sparse Design Matrix [X, Z] (CSR + CSC for O(nnz) Z*v, t(Z)*r, Gram)
 * ========================================================================== */

typedef struct {
    int n;
    int p;
    int Q;
    int k_tot;        /* p + Q */
    int nnz;
    const double *X;  /* n x p dense col-major (NULL if p == 0) */
    int *row_ptr;     /* CSR row pointers, length n + 1 */
    int *col_idx;     /* CSR 0-based col indices in 0..Q-1, length nnz */
    double *row_val;  /* CSR values, length nnz */
    int *col_ptr;     /* CSC col pointers, length Q + 1 */
    int *row_idx;     /* CSC 0-based row indices in 0..n-1, length nnz */
    double *col_val;  /* CSC values, length nnz */
} fhl_c_xz_t;

/* Build sparse [X, Z] operator using R_alloc (leak-safe across Rf_error).
 * s_x may be R_NilValue (p = 0) or an n x p numeric matrix.
 * s_z_input may be a VECSXP list of matrices/dgCMatrix/fhl_group placeholders,
 * or a single matrix/dgCMatrix/fhl_group placeholder.
 * q_out / qcum_out may be NULL or arrays of length nrand / nrand + 1. */
void fhl_c_xz_init(
    fhl_c_xz_t *xz,
    SEXP s_x,
    SEXP s_z_input,
    int n_expected,
    int *q_out,
    int *qcum_out
);

/* Build sparse [X, I_n] operator (Z = I_n, nnz = n) using R_alloc in O(n) */
void fhl_c_xz_init_identity(
    fhl_c_xz_t *xz,
    const double *X,
    int n,
    int p
);

/* Compute out = Z %*% v (length n) in O(nnz) */
void fhl_c_z_mv(const fhl_c_xz_t *xz, const double *v, double *out);

/* Compute out = t(Z) %*% r (length Q) in O(nnz) */
void fhl_c_z_tmv(const fhl_c_xz_t *xz, const double *r, double *out);

/* Compute out = t(Z) %*% diag(w) %*% Z (Q x Q col-major) using sparse rows */
void fhl_c_z_diag_quad(const fhl_c_xz_t *xz, const double *w, double *out);

/* Sparse [X, Z] (include_x=1, k=p+Q) or Z-only (include_x=0, k=Q) Gram kernels */
void fhl_c_xz_gram0(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    double *out
);

void fhl_c_xz_gram1(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw, const double *db, const double *da,
    double *out
);

/* Direct exact batch trace tr_out[k] = tr(P_sym %*% G^(1)_k) for k = 0..K-1
 * without materializing intermediate Gram matrices when risk->nested == 1. */
void fhl_c_xz_gram1_trace_batch(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *P_sym,
    const double *w, const double *b, const double *a,
    int K,
    const double * const *dw_list,
    const double * const *db_list,
    const double * const *da_list,
    double *tr_out
);

void fhl_c_xz_gram2(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw, const double *db, const double *da,
    const double *ddw, const double *ddb, const double *dda,
    double *out
);

void fhl_c_xz_gram2_cross(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw1, const double *db1, const double *da1,
    const double *dw2, const double *db2, const double *da2,
    const double *ddw12, const double *ddb12, const double *dda12,
    double *out
);

/* UCRT-aligned exp/log wrappers on Windows for consistent floating-point evaluation with R. */
double fhl_c_r_exp(double x);
double fhl_c_r_log(double x);

#endif /* FHL_LINALG_H */
