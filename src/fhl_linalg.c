#include "fhl_linalg.h"
#include "fhl_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif
double fhl_c_r_exp(double x) {
    return (double)expl((long double)x);
}

double fhl_c_r_log(double x) {
    return (double)logl((long double)x);
}

static int fhl_linalg_malloc_scope_depth = 0;
static int fhl_linalg_oom_flag = 0;

void fhl_c_linalg_begin_malloc_scope(void) {
    if (fhl_linalg_malloc_scope_depth == 0) {
        fhl_linalg_oom_flag = 0;
    }
    ++fhl_linalg_malloc_scope_depth;
}

int fhl_c_linalg_end_malloc_scope(void) {
    if (fhl_linalg_malloc_scope_depth > 0) {
        --fhl_linalg_malloc_scope_depth;
    }
    return fhl_linalg_oom_flag;
}

int fhl_c_linalg_had_oom(void) {
    return fhl_linalg_oom_flag;
}

static void fhl_c_linalg_signal_oom(const char *msg) {
    fhl_linalg_oom_flag = 1;
    if (fhl_linalg_malloc_scope_depth <= 0) {
        Rf_error("%s", msg);
    }
}

static void *fhl_c_malloc_chk(size_t count, size_t elem_size, int *ok) {
    if (!*ok) return NULL;
    size_t total = 0;
    if (!fhl_safe_mul_size(count, elem_size, &total)) {
        *ok = 0;
        return NULL;
    }
    void *p = malloc(total > 0 ? total : 1);
    if (!p) *ok = 0;
    return p;
}

static void *fhl_c_calloc_chk(size_t count, size_t elem_size, int *ok) {
    if (!*ok) return NULL;
    size_t probe = 0;
    if (!fhl_safe_mul_size(count, elem_size, &probe)) {
        *ok = 0;
        return NULL;
    }
    void *p = (probe > 0) ? calloc(count, elem_size) : calloc(1, elem_size ? elem_size : 1);
    if (!p) *ok = 0;
    return p;
}

/* ==========================================================================
 * 1. R List / Environment Helpers & Risk-Set Unpacking
 * ========================================================================== */

SEXP fhl_c_get_list_elt(SEXP list, const char *name) {
    if (Rf_isNull(list) || TYPEOF(list) != VECSXP) return R_NilValue;
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    if (Rf_isNull(names)) return R_NilValue;
    int len = Rf_length(list);
    for (int i = 0; i < len; ++i) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            return VECTOR_ELT(list, i);
        }
    }
    return R_NilValue;
}

const double *fhl_c_real_ptr(SEXP s) {
    if (TYPEOF(s) == REALSXP) return REAL(s);
    R_xlen_t len = Rf_xlength(s);
    double *out = (double *)R_alloc((size_t)len, sizeof(double));
    if (TYPEOF(s) == INTSXP || TYPEOF(s) == LGLSXP) {
        const int *ip = INTEGER(s);
        for (R_xlen_t i = 0; i < len; ++i) {
            out[i] = (ip[i] == NA_INTEGER) ? NA_REAL : (double)ip[i];
        }
    } else {
        SEXP s_r = PROTECT(Rf_coerceVector(s, REALSXP));
        memcpy(out, REAL(s_r), (size_t)len * sizeof(double));
        UNPROTECT(1);
    }
    return out;
}

void fhl_c_unpack_risk(SEXP s_Mi, fhl_c_risk_t *risk) {
    memset(risk, 0, sizeof(fhl_c_risk_t));
    if (TYPEOF(s_Mi) != VECSXP) {
        Rf_error("Risk-set descriptor Mi must be a list from fhl_build_risk_set()");
    }
    SEXP s_nr = fhl_c_get_list_elt(s_Mi, "nr");
    SEXP s_nc = fhl_c_get_list_elt(s_Mi, "nc");
    SEXP s_nested = fhl_c_get_list_elt(s_Mi, "nested");
    SEXP s_obs = fhl_c_get_list_elt(s_Mi, "obs");
    SEXP s_lo = fhl_c_get_list_elt(s_Mi, "lo");
    SEXP s_hi = fhl_c_get_list_elt(s_Mi, "hi");
    SEXP s_ord = fhl_c_get_list_elt(s_Mi, "ord");
    SEXP s_inv = fhl_c_get_list_elt(s_Mi, "inv");

    risk->n = (!Rf_isNull(s_nr)) ? Rf_asInteger(s_nr) : Rf_length(s_hi);
    risk->m = (!Rf_isNull(s_nc)) ? Rf_asInteger(s_nc) : Rf_length(s_ord);
    risk->nested = (!Rf_isNull(s_nested) && Rf_asLogical(s_nested) == 1) ? 1 : 0;
    risk->lo  = (!Rf_isNull(s_lo)  && TYPEOF(s_lo)  == INTSXP) ? INTEGER(s_lo)  : NULL;
    risk->hi  = (!Rf_isNull(s_hi)  && TYPEOF(s_hi)  == INTSXP) ? INTEGER(s_hi)  : NULL;
    risk->ord = (!Rf_isNull(s_ord) && TYPEOF(s_ord) == INTSXP) ? INTEGER(s_ord) : NULL;
    risk->inv = (!Rf_isNull(s_inv) && TYPEOF(s_inv) == INTSXP) ? INTEGER(s_inv) : NULL;
    risk->obs = (!Rf_isNull(s_obs) && TYPEOF(s_obs) == INTSXP) ? INTEGER(s_obs) : NULL;
    if (risk->n <= 0 || risk->m <= 0 ||
        risk->lo == NULL || Rf_length(s_lo) != risk->n ||
        risk->hi == NULL || Rf_length(s_hi) != risk->n ||
        risk->ord == NULL || Rf_length(s_ord) != risk->m ||
        risk->inv == NULL || Rf_length(s_inv) != risk->m ||
        (risk->nested && (risk->obs == NULL || Rf_length(s_obs) != risk->n))) {
        Rf_error("Invalid risk-set descriptor Mi");
    }
}

/* ==========================================================================
 * 2. Risk-Set Forward (Mi %*% b) and Backward (t(Mi) %*% b) Operations
 * ========================================================================== */

void fhl_c_risk_forward(const fhl_c_risk_t *risk, const double *b, double *out, double *work_m1) {
    (void)work_m1;
    const int n = risk->n;
    const int m = risk->m;
    if (n < 0 || m < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_risk_forward");
        return;
    }
    size_t mp1 = 0;
    int ok = fhl_safe_add_size((size_t)m, 1, &mp1);
    long double *cs = (long double *)fhl_c_malloc_chk(mp1, sizeof(long double), &ok);
    if (!ok) {
        free(cs);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_risk_forward");
        return;
    }
    cs[0] = 0.0L;
    for (int r = 0; r < m; ++r) {
        cs[r + 1] = cs[r] + b[risk->ord[r] - 1];
    }
    for (int i = 0; i < n; ++i) {
        out[i] = (double)cs[risk->hi[i]] - (double)cs[risk->lo[i]];
    }
    free(cs);
}

void fhl_c_risk_backward(const fhl_c_risk_t *risk, const double *b, double *out, double *work_3m3) {
    const int n = risk->n;
    const int m = risk->m;
    if (n < 0 || m < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_risk_backward");
        return;
    }
    size_t mp1 = 0, n3 = 0, bytes = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size(n3, sizeof(double), &bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_risk_backward");
        return;
    }
    int free_work = 0;
    double *buf = work_3m3;
    if (buf == NULL) {
        int ok = 1;
        buf = (double *)fhl_c_calloc_chk(n3, sizeof(double), &ok);
        if (!ok) {
            free(buf);
            fhl_c_linalg_signal_oom("Out of memory in fhl_c_risk_backward");
            return;
        }
        free_work = 1;
    } else {
        memset(buf, 0, bytes);
    }
    double *ends   = buf;
    double *starts = buf + (m + 1);
    double *cs     = buf + 2 * (m + 1);

    for (int i = 0; i < n; ++i) {
        ends[risk->hi[i]]   += b[i];
        starts[risk->lo[i]] += b[i];
    }
    long double se = 0.0L, ss = 0.0L;
    for (int r = m; r >= 1; --r) {
        se += ends[r];
        ss += starts[r];
        cs[r] = (double)se - (double)ss;
    }
    for (int r = 0; r < m; ++r) {
        out[r] = cs[risk->inv[r]];
    }
    if (free_work) free(buf);
}

/* ==========================================================================
 * 3. Cholesky SPD, Block Schur-Complement, and Log-Determinant Solvers
 * ========================================================================== */

static int fhl_c_all_finite(const double *arr, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (!R_FINITE(arr[i])) return 0;
    }
    return 1;
}

int fhl_c_solve_gesv(const double *A, const double *B, double *X, int n, int nrhs) {
    if (n <= 0 || nrhs <= 0 || !A || !B || !X) return 0;
    size_t nn_len = 0, nrhs_len = 0, nn_bytes = 0, rhs_bytes = 0;
    if (!fhl_safe_mul_size((size_t)n, (size_t)n, &nn_len) ||
        !fhl_safe_mul_size((size_t)n, (size_t)nrhs, &nrhs_len) ||
        !fhl_safe_mul_size(nn_len, sizeof(double), &nn_bytes) ||
        !fhl_safe_mul_size(nrhs_len, sizeof(double), &rhs_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_solve_gesv");
        return 0;
    }
    if (!fhl_c_all_finite(A, nn_len) || !fhl_c_all_finite(B, nrhs_len)) {
        return 0;
    }
    int ok = 1;
    double *lu = (double *)fhl_c_malloc_chk((size_t)n * (size_t)n, sizeof(double), &ok);
    int *ipiv  = (int *)fhl_c_malloc_chk((size_t)n, sizeof(int), &ok);
    if (!ok) {
        free(lu);
        free(ipiv);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_solve_gesv");
        return 0;
    }
    memcpy(lu, A, nn_bytes);
    if (X != B) {
        memcpy(X, B, rhs_bytes);
    }
    int info = 0;
    F77_CALL(dgesv)(&n, &nrhs, lu, &n, ipiv, X, &n, &info);
    free(lu);
    free(ipiv);
    return (info == 0) ? 1 : 0;
}

int fhl_c_solve_inv(const double *A, double *inv, int n) {
    if (n <= 0 || !A || !inv) return 0;
    size_t nn_len = 0, nn_bytes = 0;
    if (!fhl_safe_mul_size((size_t)n, (size_t)n, &nn_len) ||
        !fhl_safe_mul_size(nn_len, sizeof(double), &nn_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_solve_inv");
        return 0;
    }
    return fhl_blas_inverse_c(n, A, inv);
}

int fhl_c_solve_spd(const double *A, const double *B, double *X, int n, int nrhs) {
    if (n <= 0 || nrhs <= 0 || !A || !B || !X) return 0;
    size_t nn_len = 0, nrhs_len = 0, nn_bytes = 0, rhs_bytes = 0;
    if (!fhl_safe_mul_size((size_t)n, (size_t)n, &nn_len) ||
        !fhl_safe_mul_size((size_t)n, (size_t)nrhs, &nrhs_len) ||
        !fhl_safe_mul_size(nn_len, sizeof(double), &nn_bytes) ||
        !fhl_safe_mul_size(nrhs_len, sizeof(double), &rhs_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_solve_spd");
        return 0;
    }
    if (!fhl_c_all_finite(A, nn_len) || !fhl_c_all_finite(B, nrhs_len)) {
        return 0;
    }
    for (int j = 0; j < n; ++j) {
        if (A[j + (size_t)j * (size_t)n] <= 0.0) return 0;
        for (int i = j + 1; i < n; ++i) {
            double aij = A[i + (size_t)j * (size_t)n];
            double aji = A[j + (size_t)i * (size_t)n];
            double scale = 1.0;
            if (fabs(aij) > scale) scale = fabs(aij);
            if (fabs(aji) > scale) scale = fabs(aji);
            if (fabs(aij - aji) > 1e-10 * scale) return 0;
        }
    }
    int ok = 1;
    double *chol = (double *)fhl_c_malloc_chk(nn_len, sizeof(double), &ok);
    if (!ok) {
        free(chol);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_solve_spd");
        return 0;
    }
    memcpy(chol, A, nn_bytes);
    if (X != B) {
        memcpy(X, B, rhs_bytes);
    }
    int info = 0;
    const char *uplo = "U";
    if (n >= 64) {
        int nth = fhl_blas_detect_threads(n);
        info = fhl_portable_potrf_upper(n, chol, nth);
    } else {
        F77_CALL(dpotrf)(uplo, &n, chol, &n, &info FCONE);
    }
    if (info == 0) {
        F77_CALL(dpotrs)(uplo, &n, &nrhs, chol, &n, X, &n, &info FCONE);
    }
    free(chol);
    return (info == 0 && fhl_c_all_finite(X, nrhs_len)) ? 1 : 0;
}

int fhl_c_solve_gesv_inv(const double *A, double *inv, int n) {
    if (n <= 0 || !A || !inv) return 0;
    if (n == 1) {
        if (!R_FINITE(A[0]) || A[0] == 0.0) return 0;
        inv[0] = 1.0 / A[0];
        return R_FINITE(inv[0]) ? 1 : 0;
    }
    size_t nn_bytes = 0;
    if (!fhl_safe_mul3_size((size_t)n, (size_t)n, sizeof(double), &nn_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_solve_gesv_inv");
        return 0;
    }
    memset(inv, 0, nn_bytes);
    for (int i = 0; i < n; ++i) {
        inv[i + (size_t)i * (size_t)n] = 1.0;
    }
    return fhl_c_solve_gesv(A, inv, inv, n, n);
}

int fhl_c_lognormal_inverse(
    const double *H, int p, int Q,
    const double *Dinv_in,
    double *Hinv_out,
    double *Dinv_out
) {
    if (p <= 0 || Q <= 0 || !H || !Hinv_out) return 0;
    if (p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Dimension overflow in fhl_c_lognormal_inverse");
        return 0;
    }
    const int k = p + Q;
    size_t pp_len = 0, pq_len = 0, qq_len = 0, kk_len = 0, kk_bytes = 0;
    if (!fhl_safe_mul_size((size_t)p, (size_t)p, &pp_len) ||
        !fhl_safe_mul_size((size_t)p, (size_t)Q, &pq_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)k, (size_t)k, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_lognormal_inverse");
        return 0;
    }
    if (!fhl_c_all_finite(H, kk_len) || (Dinv_in != NULL && !fhl_c_all_finite(Dinv_in, qq_len))) {
        return 0;
    }
    (void)kk_bytes;
    int alloc_ok = 1;
    double *H_xx     = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *H_xz     = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *H_zx     = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *H_zz     = (double *)fhl_c_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    double *Dinv     = (Dinv_out != NULL) ? Dinv_out : (double *)fhl_c_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    double *BD       = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *S        = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *Si       = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *topright = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *DC       = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *botleft  = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *botright = (double *)fhl_c_malloc_chk(qq_len, sizeof(double), &alloc_ok);

    if (!alloc_ok || !H_xx || !H_xz || !H_zx || !H_zz || !Dinv || !BD || !S || !Si || !topright || !DC || !botleft || !botright) {
        free(H_xx); free(H_xz); free(H_zx); free(H_zz);
        if (Dinv_out == NULL) free(Dinv);
        free(BD); free(S); free(Si); free(topright); free(DC); free(botleft); free(botright);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_lognormal_inverse");
        return 0;
    }

    /* Extract subblocks H_xx, H_xz, H_zx, H_zz */
    for (int c = 0; c < p; ++c) {
        memcpy(H_xx + (size_t)c * (size_t)p, H + (size_t)c * (size_t)k, (size_t)p * sizeof(double));
        memcpy(H_zx + (size_t)c * (size_t)Q, H + (size_t)c * (size_t)k + p, (size_t)Q * sizeof(double));
    }
    for (int c = 0; c < Q; ++c) {
        memcpy(H_xz + (size_t)c * (size_t)p, H + (size_t)(p + c) * (size_t)k, (size_t)p * sizeof(double));
        memcpy(H_zz + (size_t)c * (size_t)Q, H + (size_t)(p + c) * (size_t)k + p, (size_t)Q * sizeof(double));
    }

    int ok = 1;
    int local_oom = 0;
    fhl_c_linalg_begin_malloc_scope();
    if (Dinv_in == NULL) {
        ok = fhl_c_solve_inv(H_zz, Dinv, Q);
    } else if (Dinv != Dinv_in) {
        memcpy(Dinv, Dinv_in, qq_len * sizeof(double));
    }

    if (ok) {
        const int nth_inv = fhl_blas_detect_threads(Q);
        /* BD <- H_xz %*% Dinv (p x Q) and DC <- Dinv %*% H_zx = BD^T (Q x p) via unit-stride columns */
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth_inv) schedule(static) if (Q >= 128)
#endif
        for (int c = 0; c < Q; ++c) {
            const double *dcol = Dinv + (size_t)c * (size_t)Q;
            for (int j = 0; j < p; ++j) {
                const double *hz_j = H_zx + (size_t)j * (size_t)Q;
                double s = 0.0;
                for (int l = 0; l < Q; ++l) {
                    s += hz_j[l] * dcol[l];
                }
                BD[j + (size_t)c * (size_t)p] = s;
                DC[c + (size_t)j * (size_t)Q] = s;
            }
        }

        /* S <- H_xx - BD %*% H_zx = H_xx - DC^T %*% H_zx (p x p) */
        for (int j = 0; j < p; ++j) {
            const double *hz_j = H_zx + (size_t)j * (size_t)Q;
            for (int i = 0; i < p; ++i) {
                const double *dc_i = DC + (size_t)i * (size_t)Q;
                double s = 0.0;
                for (int l = 0; l < Q; ++l) {
                    s += dc_i[l] * hz_j[l];
                }
                S[i + (size_t)j * (size_t)p] = H_xx[i + (size_t)j * (size_t)p] - s;
            }
        }

        /* Invert Schur complement: Si = S^{-1} */
        ok = fhl_c_solve_inv(S, Si, p);
        if (ok) {
            /* topright <- -Si %*% BD (p x Q) and botleft <- -DC %*% Si = topright^T (Q x p) */
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth_inv) schedule(static) if (Q >= 128)
#endif
            for (int c = 0; c < Q; ++c) {
                const double *bd_c = BD + (size_t)c * (size_t)p;
                for (int j = 0; j < p; ++j) {
                    double s = 0.0;
                    for (int u = 0; u < p; ++u) {
                        s += Si[j + (size_t)u * (size_t)p] * bd_c[u];
                    }
                    double val = -s;
                    topright[j + (size_t)c * (size_t)p] = val;
                    botleft[c + (size_t)j * (size_t)Q] = val;
                }
            }

            /* botright <- Dinv - DC %*% topright (Q x Q, column-parallel unit-stride rank-p update) */
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth_inv) schedule(static) if (Q >= 128)
#endif
            for (int c = 0; c < Q; ++c) {
                double *br_c = botright + (size_t)c * (size_t)Q;
                memcpy(br_c, Dinv + (size_t)c * (size_t)Q, (size_t)Q * sizeof(double));
                for (int u = 0; u < p; ++u) {
                    const double tr_uc = topright[u + (size_t)c * (size_t)p];
                    const double *dc_u = DC + (size_t)u * (size_t)Q;
                    for (int r = 0; r < Q; ++r) {
                        br_c[r] -= dc_u[r] * tr_uc;
                    }
                }
            }

            /* Assemble rbind(cbind(Si, topright), cbind(botleft, botright)) */
            for (int c = 0; c < p; ++c) {
                memcpy(Hinv_out + (size_t)c * (size_t)k, Si + (size_t)c * (size_t)p, (size_t)p * sizeof(double));
                memcpy(Hinv_out + (size_t)c * (size_t)k + p, botleft + (size_t)c * (size_t)Q, (size_t)Q * sizeof(double));
            }
            for (int c = 0; c < Q; ++c) {
                memcpy(Hinv_out + (size_t)(p + c) * (size_t)k, topright + (size_t)c * (size_t)p, (size_t)p * sizeof(double));
                memcpy(Hinv_out + (size_t)(p + c) * (size_t)k + p, botright + (size_t)c * (size_t)Q, (size_t)Q * sizeof(double));
            }
        }
    }

    free(H_xx);
    free(H_xz);
    free(H_zx);
    free(H_zz);
    if (Dinv_out == NULL) free(Dinv);
    free(BD);
    free(S);
    free(Si);
    free(topright);
    free(DC);
    free(botleft);
    free(botright);
    int scope_oom = fhl_c_linalg_end_malloc_scope();
    if (local_oom || scope_oom) {
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_lognormal_inverse");
        return 0;
    }
    return ok;
}

int fhl_c_schur_solve_step(
    const double *H, int p, int Q,
    const double *Dinv_in,
    const double *rhs,
    double *step_out
) {
    if (p < 0 || Q < 0 || p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Dimension overflow in fhl_c_schur_solve_step");
        return 0;
    }
    const int k = p + Q;
    if (k <= 0 || !H || !rhs || !step_out) return 0;
    if (p <= 0 || Q <= 0) {
        return fhl_c_solve_spd(H, rhs, step_out, k, 1);
    }

    size_t pp_len = 0, pq_len = 0, qq_len = 0, qp1_len = 0, kk_len = 0, kk_bytes = 0;
    if (!fhl_safe_mul_size((size_t)p, (size_t)p, &pp_len) ||
        !fhl_safe_mul_size((size_t)p, (size_t)Q, &pq_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)p + 1, &qp1_len) ||
        !fhl_safe_mul_size((size_t)k, (size_t)k, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_schur_solve_step");
        return 0;
    }
    if (!fhl_c_all_finite(H, kk_len) ||
        (Dinv_in != NULL && !fhl_c_all_finite(Dinv_in, qq_len)) ||
        !fhl_c_all_finite(rhs, (size_t)k)) {
        return 0;
    }
    (void)kk_bytes;
    int alloc_ok = 1;
    double *H_xx  = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *H_xz  = (double *)fhl_c_malloc_chk(pq_len, sizeof(double), &alloc_ok);
    double *W_u   = (double *)fhl_c_malloc_chk(qp1_len, sizeof(double), &alloc_ok);
    double *H_zz  = (Dinv_in == NULL) ? (double *)fhl_c_malloc_chk(qq_len, sizeof(double), &alloc_ok) : NULL;
    double *W_sol = (Dinv_in != NULL) ? (double *)fhl_c_malloc_chk(qp1_len, sizeof(double), &alloc_ok) : NULL;
    double *S     = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *S_tmp = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
    double *r_s   = (double *)fhl_c_malloc_chk((size_t)p, sizeof(double), &alloc_ok);
    double *s_x   = (double *)fhl_c_malloc_chk((size_t)p, sizeof(double), &alloc_ok);

    if (!alloc_ok || !H_xx || !H_xz || !W_u || (Dinv_in == NULL && !H_zz) ||
        (Dinv_in != NULL && !W_sol) || !S || !S_tmp || !r_s || !s_x) {
        free(H_xx); free(H_xz); free(W_u); free(H_zz); free(W_sol);
        free(S); free(S_tmp); free(r_s); free(s_x);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_schur_solve_step");
        return 0;
    }

    /* Extract blocks of symmetric H and pack [H_zx | r_z] into W_u (Q x (p + 1)) */
    for (int c = 0; c < p; ++c) {
        memcpy(H_xx + (size_t)c * (size_t)p, H + (size_t)c * (size_t)k, (size_t)p * sizeof(double));
        memcpy(W_u  + (size_t)c * (size_t)Q, H + (size_t)c * (size_t)k + p, (size_t)Q * sizeof(double));
    }
    memcpy(W_u + (size_t)p * (size_t)Q, rhs + p, (size_t)Q * sizeof(double));
    for (int c = 0; c < Q; ++c) {
        memcpy(H_xz + (size_t)c * (size_t)p, H + (size_t)(p + c) * (size_t)k, (size_t)p * sizeof(double));
        if (H_zz != NULL) {
            memcpy(H_zz + (size_t)c * (size_t)Q, H + (size_t)(p + c) * (size_t)k + p, (size_t)Q * sizeof(double));
        }
    }

    int ok = 1;
    double *sol_ptr = NULL;
    const char *transN = "N";
    const char *uplo = "U";
    double one = 1.0, neg_one = -1.0, zero = 0.0;
    int inc = 1;
    int nrhs = p + 1;

    fhl_c_linalg_begin_malloc_scope();
    if (Dinv_in == NULL) {
        int info = 0;
        if (Q >= 64) {
            int nth = fhl_blas_detect_threads(Q);
            info = fhl_portable_potrf_upper(Q, H_zz, nth);
        } else {
            F77_CALL(dpotrf)(uplo, &Q, H_zz, &Q, &info FCONE);
        }
        if (info == 0) {
            F77_CALL(dpotrs)(uplo, &Q, &nrhs, H_zz, &Q, W_u, &Q, &info FCONE);
        }
        if (info != 0) {
            ok = 0;
        } else {
            sol_ptr = W_u;
        }
    } else {
        F77_CALL(dgemm)(transN, transN, &Q, &nrhs, &Q,
                        &one, Dinv_in, &Q, W_u, &Q, &zero, W_sol, &Q FCONE FCONE);
        sol_ptr = W_sol;
    }

    if (ok) {
        const double *W   = sol_ptr;
        const double *u_z = sol_ptr + (size_t)p * (size_t)Q;

        /* S = H_xx - H_xz %*% W */
        F77_CALL(dgemm)(transN, transN, &p, &p, &Q,
                        &one, H_xz, &p, W, &Q, &zero, S_tmp, &p FCONE FCONE);
        for (size_t idx = 0; idx < pp_len; ++idx) S[idx] = H_xx[idx] - S_tmp[idx];

        /* r_s = r_x - H_xz %*% u_z */
        memcpy(r_s, rhs, (size_t)p * sizeof(double));
        F77_CALL(dgemv)(transN, &p, &Q, &neg_one, H_xz, &p, u_z, &inc, &one, r_s, &inc FCONE);

        ok = fhl_c_solve_spd(S, r_s, s_x, p, 1);
        if (ok) {
            memcpy(step_out, s_x, (size_t)p * sizeof(double));
            memcpy(step_out + p, u_z, (size_t)Q * sizeof(double));
            F77_CALL(dgemv)(transN, &Q, &p, &neg_one, W, &Q, s_x, &inc, &one, step_out + p, &inc FCONE);
        }
    }

    free(H_xx); free(H_xz); free(W_u); free(H_zz); free(W_sol);
    free(S); free(S_tmp); free(r_s); free(s_x);
    int scope_oom = fhl_c_linalg_end_malloc_scope();
    if (scope_oom) {
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_schur_solve_step");
        return 0;
    }

    return ok;
}

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
) {
    if (p < 0 || Q < 0 || p > INT_MAX - Q) return 0;
    const int k = p + Q;
    size_t kk_len = 0, qq_len = 0, pp_len = 0;
    if (!fhl_safe_mul_size((size_t)k, (size_t)k, &kk_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)p, (size_t)p, &pp_len)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_schur_scaled_eval");
        return 0;
    }
    if ((kk_len > 0 && (!H0 || !fhl_c_all_finite(H0, kk_len))) ||
        (Q > 0 && (!d_diag || !fhl_c_all_finite(d_diag, (size_t)Q))) ||
        (rhs && k > 0 && !fhl_c_all_finite(rhs, (size_t)k))) {
        return 0;
    }

    int Q_act = 0;
    for (int r = 0; r < Q; ++r) {
        if (d_diag[r] > 0.0) ++Q_act;
    }

    if (Hinv_out && kk_len > 0) memset(Hinv_out, 0, kk_len * sizeof(double));
    if (H22_inv_out && qq_len > 0) memset(H22_inv_out, 0, qq_len * sizeof(double));
    if (step_out && k > 0) memset(step_out, 0, (size_t)k * sizeof(double));
    if (R_diag_out && Q > 0) memset(R_diag_out, 0, (size_t)Q * sizeof(double));
    if (logdet_W_out) *logdet_W_out = 0.0;
    if (logdet_S2pi_out) *logdet_S2pi_out = 0.0;

    if (k == 0) return 1;

    int alloc_ok = 1;
    int *act = (Q_act > 0) ? (int *)fhl_c_malloc_chk((size_t)Q_act, sizeof(int), &alloc_ok) : NULL;
    double *sd = (Q_act > 0) ? (double *)fhl_c_malloc_chk((size_t)Q_act, sizeof(double), &alloc_ok) : NULL;
    size_t qa2 = (size_t)Q_act * (size_t)Q_act;
    size_t qap = (size_t)Q_act * (size_t)p;
    double *H0_sc = (Q_act > 0) ? (double *)fhl_c_malloc_chk(qa2, sizeof(double), &alloc_ok) : NULL;
    double *W = (Q_act > 0) ? (double *)fhl_c_malloc_chk(qa2, sizeof(double), &alloc_ok) : NULL;
    double *W_inv = (Q_act > 0) ? (double *)fhl_c_malloc_chk(qa2, sizeof(double), &alloc_ok) : NULL;
    double *B_sc = (Q_act > 0 && p > 0) ? (double *)fhl_c_malloc_chk(qap, sizeof(double), &alloc_ok) : NULL;
    double *WB_sc = (Q_act > 0 && p > 0) ? (double *)fhl_c_malloc_chk(qap, sizeof(double), &alloc_ok) : NULL;
    double *S_bb = (p > 0) ? (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok) : NULL;
    double *S_inv = (p > 0) ? (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok) : NULL;
    double *S_2pi = (p > 0 && logdet_S2pi_out) ? (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok) : NULL;
    double *K_bp = (Q_act > 0 && p > 0 && (Hinv_out || R_diag_out)) ? (double *)fhl_c_malloc_chk(qap, sizeof(double), &alloc_ok) : NULL;
    double *Adj = (Q_act > 0 && p > 0 && (Hinv_out || R_diag_out)) ? (double *)fhl_c_malloc_chk(qa2, sizeof(double), &alloc_ok) : NULL;
    double *rhs_sc = (Q_act > 0 && rhs && step_out) ? (double *)fhl_c_malloc_chk((size_t)Q_act, sizeof(double), &alloc_ok) : NULL;
    double *vtmp = (Q_act > 0 && rhs && step_out) ? (double *)fhl_c_malloc_chk((size_t)Q_act, sizeof(double), &alloc_ok) : NULL;
    double *rhs_b = (p > 0 && rhs && step_out) ? (double *)fhl_c_malloc_chk((size_t)p, sizeof(double), &alloc_ok) : NULL;
    double *step_b = (p > 0 && rhs && step_out) ? (double *)fhl_c_malloc_chk((size_t)p, sizeof(double), &alloc_ok) : NULL;
    if (!alloc_ok) {
        free(act); free(sd); free(H0_sc); free(W); free(W_inv);
        free(B_sc); free(WB_sc); free(S_bb); free(S_inv); free(S_2pi);
        free(K_bp); free(Adj); free(rhs_sc); free(vtmp); free(rhs_b); free(step_b);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_schur_scaled_eval");
        return 0;
    }

    const char *transN = "N", *transT = "T";
    double one = 1.0, neg_one = -1.0, zero = 0.0;
    int inc = 1, ok = 1;

    fhl_c_linalg_begin_malloc_scope();

    if (Q_act > 0) {
        int a_idx = 0;
        for (int r = 0; r < Q; ++r) {
            if (d_diag[r] > 0.0) {
                act[a_idx] = r;
                sd[a_idx] = sqrt(d_diag[r]);
                ++a_idx;
            }
        }
        for (int b = 0; b < Q_act; ++b) {
            const int cb = p + act[b];
            const double sdb = sd[b];
            for (int a = 0; a < Q_act; ++a) {
                double h_ab = sd[a] * H0[(p + act[a]) + (size_t)cb * (size_t)k] * sdb;
                H0_sc[a + (size_t)b * (size_t)Q_act] = h_ab;
                W[a + (size_t)b * (size_t)Q_act] = ((a == b) ? 1.0 : 0.0) + h_ab;
            }
        }
        ok = fhl_c_solve_inv(W, W_inv, Q_act);
        if (ok && logdet_W_out) {
            *logdet_W_out = fhl_c_logdet_spd(W, Q_act, 0, 0.0, 0, 0.0);
            if (!R_FINITE(*logdet_W_out)) ok = 0;
        }
        if (ok && H22_inv_out) {
            for (int b = 0; b < Q_act; ++b) {
                const int cb = act[b];
                const double sdb = sd[b];
                for (int a = 0; a < Q_act; ++a) {
                    H22_inv_out[act[a] + (size_t)cb * (size_t)Q] = sd[a] * W_inv[a + (size_t)b * (size_t)Q_act] * sdb;
                }
            }
        }
    }

    if (ok && p > 0) {
        for (int j = 0; j < p; ++j) {
            memcpy(S_bb + (size_t)j * (size_t)p, H0 + (size_t)j * (size_t)k, (size_t)p * sizeof(double));
        }
        if (Q_act > 0) {
            for (int j = 0; j < p; ++j) {
                for (int a = 0; a < Q_act; ++a) {
                    B_sc[a + (size_t)j * (size_t)Q_act] = sd[a] * H0[(p + act[a]) + (size_t)j * (size_t)k];
                }
            }
            F77_CALL(dgemm)(transN, transN, &Q_act, &p, &Q_act,
                            &one, W_inv, &Q_act, B_sc, &Q_act, &zero, WB_sc, &Q_act FCONE FCONE);
            double *BtW = (double *)fhl_c_malloc_chk(pp_len, sizeof(double), &alloc_ok);
            if (!alloc_ok || !BtW) {
                free(BtW);
                ok = 0;
            } else {
                F77_CALL(dgemm)(transT, transN, &p, &p, &Q_act,
                                &one, B_sc, &Q_act, WB_sc, &Q_act, &zero, BtW, &p FCONE FCONE);
                for (size_t idx = 0; idx < pp_len; ++idx) S_bb[idx] -= BtW[idx];
                free(BtW);
            }
        }
        if (ok) ok = fhl_c_solve_inv(S_bb, S_inv, p);
        if (ok && logdet_S2pi_out) {
            double inv_2pi = 1.0 / (2.0 * pi_const);
            for (size_t idx = 0; idx < pp_len; ++idx) S_2pi[idx] = S_bb[idx] * inv_2pi;
            *logdet_S2pi_out = fhl_c_logdet_spd(S_2pi, p, 0, 0.0, 0, 0.0);
            if (!R_FINITE(*logdet_S2pi_out)) ok = 0;
        }
    }

    if (ok && step_out && rhs) {
        if (Q_act > 0) {
            for (int a = 0; a < Q_act; ++a) rhs_sc[a] = sd[a] * rhs[p + act[a]];
            F77_CALL(dgemv)(transN, &Q_act, &Q_act, &one, W_inv, &Q_act, rhs_sc, &inc, &zero, vtmp, &inc FCONE);
        }
        if (p > 0) {
            memcpy(rhs_b, rhs, (size_t)p * sizeof(double));
            if (Q_act > 0) {
                F77_CALL(dgemv)(transT, &Q_act, &p, &neg_one, B_sc, &Q_act, vtmp, &inc, &one, rhs_b, &inc FCONE);
            }
            F77_CALL(dgemv)(transN, &p, &p, &one, S_inv, &p, rhs_b, &inc, &zero, step_b, &inc FCONE);
            memcpy(step_out, step_b, (size_t)p * sizeof(double));
            if (Q_act > 0) {
                F77_CALL(dgemv)(transN, &Q_act, &p, &neg_one, WB_sc, &Q_act, step_b, &inc, &one, vtmp, &inc FCONE);
            }
        }
        for (int a = 0; a < Q_act; ++a) {
            step_out[p + act[a]] = sd[a] * vtmp[a];
        }
    }

    if (ok && (Hinv_out || R_diag_out)) {
        if (Q_act > 0 && p > 0) {
            F77_CALL(dgemm)(transN, transT, &p, &Q_act, &p,
                            &one, S_inv, &p, WB_sc, &Q_act, &zero, K_bp, &p FCONE FCONE);
            F77_CALL(dgemm)(transN, transN, &Q_act, &Q_act, &p,
                            &one, WB_sc, &Q_act, K_bp, &p, &zero, Adj, &Q_act FCONE FCONE);
        }
        if (Hinv_out) {
            if (p > 0) {
                for (int j = 0; j < p; ++j) {
                    memcpy(Hinv_out + (size_t)j * (size_t)k, S_inv + (size_t)j * (size_t)p, (size_t)p * sizeof(double));
                }
            }
            if (Q_act > 0) {
                if (p > 0) {
                    for (int b = 0; b < Q_act; ++b) {
                        const int col_r = p + act[b];
                        const double sdb = sd[b];
                        for (int j = 0; j < p; ++j) {
                            double val = -K_bp[j + (size_t)b * (size_t)p] * sdb;
                            Hinv_out[j + (size_t)col_r * (size_t)k] = val;
                            Hinv_out[col_r + (size_t)j * (size_t)k] = val;
                        }
                    }
                }
                for (int b = 0; b < Q_act; ++b) {
                    const int col_r = p + act[b];
                    const double sdb = sd[b];
                    for (int a = 0; a < Q_act; ++a) {
                        double w_plus = W_inv[a + (size_t)b * (size_t)Q_act] + ((p > 0) ? Adj[a + (size_t)b * (size_t)Q_act] : 0.0);
                        Hinv_out[(p + act[a]) + (size_t)col_r * (size_t)k] = sd[a] * w_plus * sdb;
                    }
                }
            }
        }
        if (R_diag_out && Q_act > 0) {
            for (int a = 0; a < Q_act; ++a) {
                long double wh = 0.0L;
                for (int c = 0; c < Q_act; ++c) {
                    wh += (long double)W_inv[a + (size_t)c * (size_t)Q_act] *
                          (long double)H0_sc[c + (size_t)a * (size_t)Q_act];
                }
                double sub_p = (p > 0) ? Adj[a + (size_t)a * (size_t)Q_act] : 0.0;
                R_diag_out[act[a]] = (double)(wh - (long double)sub_p);
            }
        }
    }

    int scope_oom = fhl_c_linalg_end_malloc_scope();
    if (scope_oom) {
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_schur_scaled_eval");
        ok = 0;
    }

    free(act); free(sd); free(H0_sc); free(W); free(W_inv);
    free(B_sc); free(WB_sc); free(S_bb); free(S_inv); free(S_2pi);
    free(K_bp); free(Adj); free(rhs_sc); free(vtmp); free(rhs_b); free(step_b);
    return ok;
}

double fhl_c_trace_ab(const double *A, const double *B, int n) {
    long double s = 0.0L;
    for (int j = 0; j < n; ++j) {
        const size_t jn = (size_t)j * (size_t)n;
        for (int i = 0; i < n; ++i) {
            s += A[i + jn] * B[j + (size_t)i * (size_t)n];
        }
    }
    return (double)s;
}

double fhl_c_trace_hinv_hd_prod(
    const double *Hinv,
    const double *Hd1,
    const double *Hd2,
    int k
) {
    if (k <= 0 || !Hinv || !Hd1 || !Hd2) return 0.0;
    size_t kk_len = 0, kk_bytes = 0;
    if (!fhl_safe_mul_size((size_t)k, (size_t)k, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_trace_hinv_hd_prod");
        return 0.0;
    }
    (void)kk_bytes;
    const int same = (Hd1 == Hd2);

    /* Identify active row/column indices where Hd1 or Hd2 has non-zero entries */
    int alloc_ok = 1;
    int *mask1 = (int *)fhl_c_calloc_chk((size_t)k, sizeof(int), &alloc_ok);
    int *mask2 = same ? mask1 : (int *)fhl_c_calloc_chk((size_t)k, sizeof(int), &alloc_ok);
    int *idx1  = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &alloc_ok);
    int *idx2  = same ? idx1 : (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &alloc_ok);
    if (!alloc_ok || !mask1 || !mask2 || !idx1 || !idx2) {
        free(mask1); if (!same) free(mask2);
        free(idx1);  if (!same) free(idx2);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_trace_hinv_hd_prod");
        return 0.0;
    }

    for (int c = 0; c < k; ++c) {
        const double *col = Hd1 + (size_t)c * (size_t)k;
        for (int r = 0; r < k; ++r) {
            if (col[r] != 0.0) {
                mask1[c] = 1;
                mask1[r] = 1;
            }
        }
    }
    int n1 = 0;
    for (int i = 0; i < k; ++i) {
        if (mask1[i]) idx1[n1++] = i;
    }
    if (n1 == 0) {
        free(mask1); if (!same) free(mask2);
        free(idx1);  if (!same) free(idx2);
        return 0.0;
    }

    int n2 = n1;
    if (!same) {
        for (int c = 0; c < k; ++c) {
            const double *col = Hd2 + (size_t)c * (size_t)k;
            for (int r = 0; r < k; ++r) {
                if (col[r] != 0.0) {
                    mask2[c] = 1;
                    mask2[r] = 1;
                }
            }
        }
        n2 = 0;
        for (int i = 0; i < k; ++i) {
            if (mask2[i]) idx2[n2++] = i;
        }
        if (n2 == 0) {
            free(mask1); free(mask2); free(idx1); free(idx2);
            return 0.0;
        }
    }

    double tr = 0.0;
    const char *transN = "N";
    double one = 1.0, zero = 0.0;

    if (n1 <= (3 * k) / 4 || n2 <= (3 * k) / 4) {
        /* Compressed active-block multiply: saves O(k^3) -> O(n1*n2*(n1+n2)) */
        size_t n21 = (size_t)n2 * (size_t)n1;
        size_t n11 = (size_t)n1 * (size_t)n1;
        int blk1_ok = 1;
        double *H21   = (double *)fhl_c_malloc_chk(n21, sizeof(double), &blk1_ok);
        double *Hd1_s = (double *)fhl_c_malloc_chk(n11, sizeof(double), &blk1_ok);
        double *P1    = (double *)fhl_c_malloc_chk(n21, sizeof(double), &blk1_ok);
        if (!blk1_ok) {
            free(H21); free(Hd1_s); free(P1);
            free(mask1); if (!same) free(mask2);
            free(idx1);  if (!same) free(idx2);
            fhl_c_linalg_signal_oom("Out of memory in fhl_c_trace_hinv_hd_prod");
            return 0.0;
        }
        for (int j = 0; j < n1; ++j) {
            const int cj = idx1[j];
            const double *hcol = Hinv + (size_t)cj * (size_t)k;
            const double *dcol = Hd1  + (size_t)cj * (size_t)k;
            double *h_dst = H21   + (size_t)j * (size_t)n2;
            double *d_dst = Hd1_s + (size_t)j * (size_t)n1;
            for (int i = 0; i < n2; ++i) h_dst[i] = hcol[idx2[i]];
            for (int i = 0; i < n1; ++i) d_dst[i] = dcol[idx1[i]];
        }
        F77_CALL(dgemm)(transN, transN, &n2, &n1, &n1,
                        &one, H21, &n2, Hd1_s, &n1, &zero, P1, &n2 FCONE FCONE);
        free(H21);
        free(Hd1_s);

        if (same) {
            long double s = 0.0L;
            for (int j = 0; j < n1; ++j) {
                for (int i = 0; i < n1; ++i) {
                    s += P1[i + (size_t)j * (size_t)n1] * P1[j + (size_t)i * (size_t)n1];
                }
            }
            tr = (double)s;
        } else {
            size_t n22 = (size_t)n2 * (size_t)n2;
            int blk2_ok = 1;
            double *H12   = (double *)fhl_c_malloc_chk(n21, sizeof(double), &blk2_ok);
            double *Hd2_s = (double *)fhl_c_malloc_chk(n22, sizeof(double), &blk2_ok);
            double *P2    = (double *)fhl_c_malloc_chk(n21, sizeof(double), &blk2_ok);
            if (!blk2_ok) {
                free(H12); free(Hd2_s); free(P2);
                free(P1);
                free(mask1); free(mask2);
                free(idx1);  free(idx2);
                fhl_c_linalg_signal_oom("Out of memory in fhl_c_trace_hinv_hd_prod");
                return 0.0;
            }
            for (int j = 0; j < n2; ++j) {
                const int cj = idx2[j];
                const double *hcol = Hinv + (size_t)cj * (size_t)k;
                const double *dcol = Hd2  + (size_t)cj * (size_t)k;
                double *h_dst = H12   + (size_t)j * (size_t)n1;
                double *d_dst = Hd2_s + (size_t)j * (size_t)n2;
                for (int i = 0; i < n1; ++i) h_dst[i] = hcol[idx1[i]];
                for (int i = 0; i < n2; ++i) d_dst[i] = dcol[idx2[i]];
            }
            F77_CALL(dgemm)(transN, transN, &n1, &n2, &n2,
                            &one, H12, &n1, Hd2_s, &n2, &zero, P2, &n1 FCONE FCONE);
            long double s = 0.0L;
            for (int j = 0; j < n1; ++j) {
                for (int i = 0; i < n2; ++i) {
                    s += P1[i + (size_t)j * (size_t)n2] * P2[j + (size_t)i * (size_t)n1];
                }
            }
            tr = (double)s;
            free(H12); free(Hd2_s); free(P2);
        }
        free(P1);
    } else {
        /* Full dense path: use OpenMP + AVX2/FMA symmetric multiplication when k >= 64 */
        int dense_ok = 1;
        double *P1 = (double *)fhl_c_malloc_chk(kk_len, sizeof(double), &dense_ok);
        double *P2 = same ? P1 : (double *)fhl_c_malloc_chk(kk_len, sizeof(double), &dense_ok);
        double *col_sums = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &dense_ok);
        if (!dense_ok) {
            free(col_sums);
            if (!same) free(P2);
            free(P1);
            free(mask1); if (!same) free(mask2);
            free(idx1);  if (!same) free(idx2);
            fhl_c_linalg_signal_oom("Out of memory in fhl_c_trace_hinv_hd_prod");
            return 0.0;
        }
        int nth = (k >= 64) ? fhl_blas_detect_threads(k) : 1;
        if (k >= 64) {
            fhl_portable_symm_left(k, Hinv, Hd1, P1, nth);
            if (!same) {
                fhl_portable_symm_left(k, Hinv, Hd2, P2, nth);
            }
        } else {
            F77_CALL(dgemm)(transN, transN, &k, &k, &k,
                            &one, Hinv, &k, Hd1, &k, &zero, P1, &k FCONE FCONE);
            if (!same) {
                F77_CALL(dgemm)(transN, transN, &k, &k, &k,
                                &one, Hinv, &k, Hd2, &k, &zero, P2, &k FCONE FCONE);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static) if (k >= 128 && nth > 1)
#endif
        for (int j = 0; j < k; ++j) {
            const size_t jk = (size_t)j * (size_t)k;
            double col_sum = 0.0;
            for (int i = 0; i < k; ++i) {
                col_sum += P1[i + jk] * P2[j + (size_t)i * (size_t)k];
            }
            col_sums[j] = col_sum;
        }
        long double s_tot = 0.0L;
        for (int j = 0; j < k; ++j) {
            s_tot += (long double)col_sums[j];
        }
        tr = (double)s_tot;
        free(col_sums);
        if (!same) free(P2);
        free(P1);
    }

    free(mask1); if (!same) free(mask2);
    free(idx1);  if (!same) free(idx2);
    return tr;
}

double fhl_c_logdet_spd(
    const double *M, int n,
    int has_lower, double lower_clip,
    int has_upper, double upper_clip
) {
    if (n <= 0 || !M) return 0.0;
    size_t nn_len = 0, nn_bytes = 0;
    if (!fhl_safe_mul_size((size_t)n, (size_t)n, &nn_len) ||
        !fhl_safe_mul_size(nn_len, sizeof(double), &nn_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_logdet_spd");
        return 0.0;
    }
    if (!fhl_c_all_finite(M, nn_len)) return NA_REAL;
    int ok = 1;
    double *Rmat = (double *)fhl_c_malloc_chk(nn_len, sizeof(double), &ok);
    if (!ok) {
        free(Rmat);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_logdet_spd");
        return 0.0;
    }
    /* Strict SPD Cholesky path when no eigenvalue clipping bounds are active;
       never falls back to SVD on non-SPD matrices. */
    int info = 0;
    if (!has_lower && !has_upper) {
        for (int j = 0; j < n; ++j) {
            if (M[j + (size_t)j * (size_t)n] <= 0.0) {
                free(Rmat);
                return NA_REAL;
            }
            for (int i = j + 1; i < n; ++i) {
                double aij = M[i + (size_t)j * (size_t)n];
                double aji = M[j + (size_t)i * (size_t)n];
                double scale = 1.0;
                if (fabs(aij) > scale) scale = fabs(aij);
                if (fabs(aji) > scale) scale = fabs(aji);
                if (fabs(aij - aji) > 1e-10 * scale) {
                    free(Rmat);
                    return NA_REAL;
                }
            }
        }
        memcpy(Rmat, M, nn_bytes);
        const char *uplo = "U";
        F77_CALL(dpotrf)(uplo, &n, Rmat, &n, &info FCONE);
        if (info != 0) {
            free(Rmat);
            return NA_REAL;
        }
        double sum_log = 0.0;
        for (int i = 0; i < n; ++i) {
            double dii = Rmat[i + (size_t)i * (size_t)n];
            if (!R_FINITE(dii) || dii <= 0.0) {
                free(Rmat);
                return NA_REAL;
            }
            sum_log += log(dii);
        }
        free(Rmat);
        return 2.0 * sum_log;
    }

    /* Eigenvalue-clipping SVD path (used only by AR1 when has_lower || has_upper) */
    memcpy(Rmat, M, nn_bytes);
    int svd_ok = 1;
    double *svals = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &svd_ok);
    if (!svd_ok) {
        free(svals);
        free(Rmat);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_logdet_spd");
        return 0.0;
    }
    double wk_opt = 0.0;
    int lwork = -1;
    const char *jobu = "N", *jobvt = "N";
    double dummy_u = 0.0, dummy_vt = 0.0;
    int one = 1;
    F77_CALL(dgesvd)(jobu, jobvt, &n, &n, Rmat, &n, svals, &dummy_u, &one, &dummy_vt, &one, &wk_opt, &lwork, &info FCONE FCONE);
    size_t min_lw = 0;
    if (info != 0 || !fhl_safe_mul_size((size_t)n, 5, &min_lw)) {
        free(svals);
        free(Rmat);
        fhl_c_linalg_signal_oom("Invalid SVD workspace in fhl_c_logdet_spd");
        return 0.0;
    }
    size_t lw_sz = (wk_opt >= 0.0 && wk_opt <= (double)INT_MAX) ? (size_t)(int)wk_opt : min_lw;
    if (lw_sz < min_lw) lw_sz = min_lw;
    if (lw_sz > (size_t)INT_MAX) {
        free(svals);
        free(Rmat);
        fhl_c_linalg_signal_oom("SVD workspace overflow in fhl_c_logdet_spd");
        return 0.0;
    }
    lwork = (int)lw_sz;
    int wk_ok = 1;
    double *work = (double *)fhl_c_malloc_chk(lw_sz, sizeof(double), &wk_ok);
    if (!wk_ok) {
        free(work);
        free(svals);
        free(Rmat);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_logdet_spd");
        return 0.0;
    }
    F77_CALL(dgesvd)(jobu, jobvt, &n, &n, Rmat, &n, svals, &dummy_u, &one, &dummy_vt, &one, work, &lwork, &info FCONE FCONE);
    if (info != 0) {
        free(work);
        free(svals);
        free(Rmat);
        return NA_REAL;
    }
    double sum_log = 0.0;
    for (int i = 0; i < n; ++i) {
        double di = svals[i];
        if (has_lower && di < lower_clip) di = 1.0;
        if (has_upper && di > upper_clip) di = 1.0;
        sum_log += log(di);
    }
    free(work);
    free(svals);
    free(Rmat);
    return sum_log;
}

/* ==========================================================================
 * 4. Sparse Design Matrix [X, Z] (CSR + CSC) and Gram Kernels
 * ========================================================================== */

static int fhl_c_is_sparse_z_item(SEXP s_zi) {
    return Rf_isS4(s_zi) && Rf_inherits(s_zi, "dgCMatrix");
}

static void fhl_c_inspect_z_item(SEXP s_zi, int n_expected, int *q_out, size_t *nnz_out, int *kind_out) {
    if (fhl_c_is_sparse_z_item(s_zi)) {
        SEXP s_dim = R_do_slot(s_zi, Rf_install("Dim"));
        if (TYPEOF(s_dim) != INTSXP || Rf_length(s_dim) != 2) {
            Rf_error("Invalid dgCMatrix Dim slot in Z");
        }
        int nr = INTEGER(s_dim)[0];
        int nc = INTEGER(s_dim)[1];
        if (nr != n_expected || nc < 0) {
            Rf_error("Z block dimension (%d x %d) does not match n = %d", nr, nc, n_expected);
        }
        SEXP s_p = R_do_slot(s_zi, Rf_install("p"));
        SEXP s_i = R_do_slot(s_zi, Rf_install("i"));
        SEXP s_x = R_do_slot(s_zi, Rf_install("x"));
        if (TYPEOF(s_p) != INTSXP || Rf_length(s_p) != nc + 1 ||
            TYPEOF(s_i) != INTSXP || TYPEOF(s_x) != REALSXP) {
            Rf_error("Invalid dgCMatrix slot types or lengths in Z");
        }
        const int *pp = INTEGER(s_p);
        if (pp[0] != 0) Rf_error("Invalid dgCMatrix p[0] != 0 in Z");
        for (int c = 0; c < nc; ++c) {
            if (pp[c] == NA_INTEGER || pp[c + 1] == NA_INTEGER ||
                pp[c + 1] < pp[c] || pp[c] < 0) {
                Rf_error("Invalid dgCMatrix column pointer monotonicity in Z");
            }
        }
        int nnz = pp[nc];
        if (Rf_length(s_i) < nnz || Rf_length(s_x) < nnz) {
            Rf_error("Truncated dgCMatrix @i or @x slot in Z");
        }
        const int *ip = INTEGER(s_i);
        for (int t = 0; t < nnz; ++t) {
            if (ip[t] == NA_INTEGER || ip[t] < 0 || ip[t] >= n_expected) {
                Rf_error("dgCMatrix row index %d out of range [0, %d)", ip[t], n_expected);
            }
        }
        *q_out = nc;
        *nnz_out = (size_t)nnz;
        *kind_out = 1;
        return;
    }

    SEXP s_grp = Rf_getAttrib(s_zi, Rf_install("fhl_group"));
    if (!Rf_isNull(s_grp) && Rf_isMatrix(s_zi) && Rf_nrows(s_zi) == 1) {
        int nc = Rf_ncols(s_zi);
        if (nc < 0 || TYPEOF(s_grp) != INTSXP || Rf_length(s_grp) != n_expected) {
            Rf_error("Invalid fhl_group placeholder dimension in Z");
        }
        const int *gp = INTEGER(s_grp);
        for (int r = 0; r < n_expected; ++r) {
            if (gp[r] == NA_INTEGER || gp[r] < 1 || gp[r] > nc) {
                Rf_error("fhl_group index out of range [1, %d] at row %d", nc, r + 1);
            }
        }
        *q_out = nc;
        *nnz_out = (size_t)n_expected;
        *kind_out = 2;
        return;
    }

    if (!Rf_isMatrix(s_zi)) {
        Rf_error("Each element of Z must be a matrix or dgCMatrix");
    }
    int nr = Rf_nrows(s_zi);
    int nc = Rf_ncols(s_zi);
    if (nr != n_expected || nc < 0) {
        Rf_error("Dense Z block dimension (%d x %d) does not match n = %d", nr, nc, n_expected);
    }
    const double *zp = fhl_c_real_ptr(s_zi);
    size_t nnz = 0;
    size_t tot = (size_t)nr * (size_t)nc;
    for (size_t k = 0; k < tot; ++k) {
        if (zp[k] != 0.0) ++nnz;
    }
    *q_out = nc;
    *nnz_out = nnz;
    *kind_out = 0;
}

void fhl_c_xz_init(
    fhl_c_xz_t *xz,
    SEXP s_x,
    SEXP s_z_input,
    int n_expected,
    int *q_out,
    int *qcum_out
) {
    memset(xz, 0, sizeof(fhl_c_xz_t));
    int n = n_expected;
    int p = 0;
    const double *X_ptr = NULL;
    if (!Rf_isNull(s_x)) {
        if (!Rf_isMatrix(s_x)) Rf_error("X must be a numeric matrix");
        int nx = Rf_nrows(s_x);
        p = Rf_ncols(s_x);
        if (n < 0) n = nx;
        if (nx != n) Rf_error("X row count (%d) does not match n (%d)", nx, n);
        if (p > 0) X_ptr = fhl_c_real_ptr(s_x);
    }
    if (n < 0) Rf_error("Unable to determine row count n in fhl_c_xz_init");

    int is_list = (TYPEOF(s_z_input) == VECSXP && !Rf_isS4(s_z_input));
    int nrand = is_list ? Rf_length(s_z_input) : 1;
    if (nrand <= 0) Rf_error("Z must contain at least one random-effect block");

    int *kinds = (int *)R_alloc((size_t)nrand, sizeof(int));
    int *qs    = (int *)R_alloc((size_t)nrand, sizeof(int));
    int *qcum  = (int *)R_alloc((size_t)(nrand + 1), sizeof(int));
    qcum[0] = 0;

    size_t total_nnz_sz = 0;
    for (int i = 0; i < nrand; ++i) {
        SEXP s_zi = is_list ? VECTOR_ELT(s_z_input, i) : s_z_input;
        int qi = 0, kind = 0;
        size_t nnz_i = 0;
        fhl_c_inspect_z_item(s_zi, n, &qi, &nnz_i, &kind);
        kinds[i] = kind;
        qs[i] = qi;
        if ((size_t)qcum[i] + (size_t)qi > (size_t)INT_MAX) {
            Rf_error("Total random-effect columns Q overflows int");
        }
        qcum[i + 1] = qcum[i] + qi;
        if (total_nnz_sz + nnz_i > (size_t)INT_MAX) {
            Rf_error("Total sparse Z nonzeros overflows int");
        }
        total_nnz_sz += nnz_i;
        if (q_out) q_out[i] = qi;
    }
    if (qcum_out) {
        for (int i = 0; i <= nrand; ++i) qcum_out[i] = qcum[i];
    }

    int Q = qcum[nrand];
    int nnz = (int)total_nnz_sz;
    size_t alloc_nnz = (nnz > 0) ? (size_t)nnz : 1;

    xz->n = n;
    xz->p = p;
    xz->Q = Q;
    xz->k_tot = p + Q;
    xz->nnz = nnz;
    xz->X = X_ptr;

    xz->col_ptr = (int *)R_alloc((size_t)(Q + 1), sizeof(int));
    xz->row_idx = (int *)R_alloc(alloc_nnz, sizeof(int));
    xz->col_val = (double *)R_alloc(alloc_nnz, sizeof(double));
    xz->row_ptr = (int *)R_alloc((size_t)(n + 1), sizeof(int));
    xz->col_idx = (int *)R_alloc(alloc_nnz, sizeof(int));
    xz->row_val = (double *)R_alloc(alloc_nnz, sizeof(double));

    int *row_counts = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    memset(row_counts, 0, (size_t)(n > 0 ? n : 1) * sizeof(int));

    int pos = 0;
    xz->col_ptr[0] = 0;
    for (int i = 0; i < nrand; ++i) {
        SEXP s_zi = is_list ? VECTOR_ELT(s_z_input, i) : s_z_input;
        int qi = qs[i];
        int col_off = qcum[i];
        if (kinds[i] == 1) {
            const int *pp = INTEGER(R_do_slot(s_zi, Rf_install("p")));
            const int *ip = INTEGER(R_do_slot(s_zi, Rf_install("i")));
            const double *xp = REAL(R_do_slot(s_zi, Rf_install("x")));
            for (int c = 0; c < qi; ++c) {
                int start = pp[c], end = pp[c + 1];
                for (int t = start; t < end; ++t) {
                    int r = ip[t];
                    xz->row_idx[pos] = r;
                    xz->col_val[pos] = xp[t];
                    row_counts[r]++;
                    pos++;
                }
                xz->col_ptr[col_off + c + 1] = pos;
            }
        } else if (kinds[i] == 2) {
            const int *gp = INTEGER(Rf_getAttrib(s_zi, Rf_install("fhl_group")));
            int *ccount = (int *)R_alloc((size_t)(qi > 0 ? qi : 1), sizeof(int));
            memset(ccount, 0, (size_t)(qi > 0 ? qi : 1) * sizeof(int));
            for (int r = 0; r < n; ++r) {
                ccount[gp[r] - 1]++;
                row_counts[r]++;
            }
            int *cwrite = (int *)R_alloc((size_t)(qi > 0 ? qi : 1), sizeof(int));
            int cur = pos;
            for (int c = 0; c < qi; ++c) {
                cwrite[c] = cur;
                cur += ccount[c];
                xz->col_ptr[col_off + c + 1] = cur;
            }
            for (int r = 0; r < n; ++r) {
                int c = gp[r] - 1;
                int wpos = cwrite[c]++;
                xz->row_idx[wpos] = r;
                xz->col_val[wpos] = 1.0;
            }
            pos = cur;
        } else {
            const double *zp = fhl_c_real_ptr(s_zi);
            for (int c = 0; c < qi; ++c) {
                const double *z_col = zp + (size_t)c * (size_t)n;
                for (int r = 0; r < n; ++r) {
                    double v = z_col[r];
                    if (v != 0.0) {
                        xz->row_idx[pos] = r;
                        xz->col_val[pos] = v;
                        row_counts[r]++;
                        pos++;
                    }
                }
                xz->col_ptr[col_off + c + 1] = pos;
            }
        }
    }

    /* Transpose CSC -> CSR in ascending column order per row */
    xz->row_ptr[0] = 0;
    for (int r = 0; r < n; ++r) {
        xz->row_ptr[r + 1] = xz->row_ptr[r] + row_counts[r];
    }
    int *rwrite = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    memcpy(rwrite, xz->row_ptr, (size_t)n * sizeof(int));

    for (int c = 0; c < Q; ++c) {
        int start = xz->col_ptr[c], end = xz->col_ptr[c + 1];
        for (int t = start; t < end; ++t) {
            int r = xz->row_idx[t];
            int wpos = rwrite[r]++;
            xz->col_idx[wpos] = c;
            xz->row_val[wpos] = xz->col_val[t];
        }
    }
}

void fhl_c_xz_init_identity(fhl_c_xz_t *xz, const double *X, int n, int p) {
    xz->n = n;
    xz->p = p;
    xz->Q = n;
    xz->k_tot = p + n;
    xz->nnz = n;
    xz->X = (p > 0) ? X : NULL;

    size_t n_alloc = (n > 0) ? (size_t)n : 1;
    xz->row_ptr = (int *)R_alloc((size_t)(n + 1), sizeof(int));
    xz->col_idx = (int *)R_alloc(n_alloc, sizeof(int));
    xz->row_val = (double *)R_alloc(n_alloc, sizeof(double));
    xz->col_ptr = (int *)R_alloc((size_t)(n + 1), sizeof(int));
    xz->row_idx = (int *)R_alloc(n_alloc, sizeof(int));
    xz->col_val = (double *)R_alloc(n_alloc, sizeof(double));

    for (int i = 0; i < n; ++i) {
        xz->row_ptr[i] = i;
        xz->col_idx[i] = i;
        xz->row_val[i] = 1.0;
        xz->col_ptr[i] = i;
        xz->row_idx[i] = i;
        xz->col_val[i] = 1.0;
    }
    xz->row_ptr[n] = n;
    xz->col_ptr[n] = n;
}

void fhl_c_z_mv(const fhl_c_xz_t *xz, const double *v, double *out) {
    const int n = xz->n;
    const int Q = xz->Q;
    memset(out, 0, (size_t)n * sizeof(double));
    for (int c = 0; c < Q; ++c) {
        double vc = v[c];
        if (vc == 0.0) continue;
        int start = xz->col_ptr[c], end = xz->col_ptr[c + 1];
        for (int t = start; t < end; ++t) {
            out[xz->row_idx[t]] += xz->col_val[t] * vc;
        }
    }
}

void fhl_c_z_tmv(const fhl_c_xz_t *xz, const double *r, double *out) {
    const int Q = xz->Q;
    for (int c = 0; c < Q; ++c) {
        double s = 0.0;
        int start = xz->col_ptr[c], end = xz->col_ptr[c + 1];
        for (int t = start; t < end; ++t) {
            s += xz->col_val[t] * r[xz->row_idx[t]];
        }
        out[c] = s;
    }
}

void fhl_c_z_diag_quad(const fhl_c_xz_t *xz, const double *w, double *out) {
    const int n = xz->n;
    const int Q = xz->Q;
    memset(out, 0, (size_t)Q * (size_t)Q * sizeof(double));
    for (int r = 0; r < n; ++r) {
        double wr = w[r];
        if (wr == 0.0) continue;
        int start = xz->row_ptr[r], end = xz->row_ptr[r + 1];
        for (int jj = start; jj < end; ++jj) {
            int c = xz->col_idx[jj];
            double w_zc = wr * xz->row_val[jj];
            for (int ii = start; ii <= jj; ++ii) {
                int a = xz->col_idx[ii];
                out[a + (size_t)c * (size_t)Q] += w_zc * xz->row_val[ii];
            }
        }
    }
    for (int c = 0; c < Q; ++c) {
        for (int r = c + 1; r < Q; ++r) {
            out[r + (size_t)c * (size_t)Q] = out[c + (size_t)r * (size_t)Q];
        }
    }
}

static void fhl_c_risk_backward_sparse(
    const fhl_c_risk_t *risk,
    int start, int end,
    const int *r_idx, const double *r_val,
    const double *w_vec,
    double *out, double *work_3m3
) {
    const int m = risk->m;
    memset(work_3m3, 0, (size_t)3 * (size_t)(m + 1) * sizeof(double));
    double *ends   = work_3m3;
    double *starts = work_3m3 + (m + 1);
    double *cs     = work_3m3 + 2 * (m + 1);
    for (int t = start; t < end; ++t) {
        int r = r_idx[t];
        double wrv = w_vec[r] * r_val[t];
        ends[risk->hi[r]]   += wrv;
        starts[risk->lo[r]] += wrv;
    }
    long double se = 0.0L, ss = 0.0L;
    for (int r = m; r >= 1; --r) {
        se += ends[r];
        ss += starts[r];
        cs[r] = (double)se - (double)ss;
    }
    for (int r = 0; r < m; ++r) {
        out[r] = cs[risk->inv[r]];
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__amd64__)) && !defined(FHL_NO_AVX2)
#include <immintrin.h>
#define FHL_LA_HAVE_AVX2 1
#if defined(__clang__)
#define FHL_LA_AVX2_ATTR __attribute__((target("avx2,fma")))
#else
#define FHL_LA_AVX2_ATTR __attribute__((target("avx2,fma"), optimize("O3")))
#endif

static FHL_LA_AVX2_ATTR void fhl_la_axpy_avx2(int len, double a, const double *x, double *y) {
    int u = 0;
    __m256d va = _mm256_set1_pd(a);
    for (; u + 3 < len; u += 4) {
        __m256d vy = _mm256_loadu_pd(y + u);
        __m256d vx = _mm256_loadu_pd(x + u);
        _mm256_storeu_pd(y + u, _mm256_fmadd_pd(va, vx, vy));
    }
    for (; u < len; ++u) y[u] += a * x[u];
}

static FHL_LA_AVX2_ATTR double fhl_la_dot_avx2(int len, const double *x, const double *y) {
    int u = 0;
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    for (; u + 7 < len; u += 8) {
        a0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + u),     _mm256_loadu_pd(y + u),     a0);
        a1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + u + 4), _mm256_loadu_pd(y + u + 4), a1);
    }
    a0 = _mm256_add_pd(a0, a1);
    double buf[4];
    _mm256_storeu_pd(buf, a0);
    double s = (buf[0] + buf[1]) + (buf[2] + buf[3]);
    for (; u < len; ++u) s += x[u] * y[u];
    return s;
}
#endif

static inline void fhl_la_axpy(int len, double a, const double *x, double *y) {
    if (a == 0.0 || len <= 0) return;
#ifdef FHL_LA_HAVE_AVX2
    if (fhl_cpu_has_avx2) {
        fhl_la_axpy_avx2(len, a, x, y);
        return;
    }
#endif
    for (int u = 0; u < len; ++u) y[u] += a * x[u];
}

static inline double fhl_la_dot(int len, const double *x, const double *y) {
    if (len <= 0) return 0.0;
#ifdef FHL_LA_HAVE_AVX2
    if (fhl_cpu_has_avx2) {
        return fhl_la_dot_avx2(len, x, y);
    }
#endif
    double s = 0.0;
    for (int u = 0; u < len; ++u) s += x[u] * y[u];
    return s;
}

static void fhl_c_xz_nested_gram_core(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *h, const double *w, const double *dw, const double *ddw,
    const double *a, const double *da, const double *dda,
    int lev, double *ans
) {
    const int n = xz->n;
    const int p = include_x ? xz->p : 0;
    const int Q = xz->Q;
    if (p < 0 || Q < 0 || p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Dimension overflow in fhl_c_xz_nested_gram_core");
        return;
    }
    const int k = p + Q;
    const int m = risk->m;
    const double *X = xz->X;
    if (k <= 0) return;
    if (m < 0 || n < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_nested_gram_core");
        return;
    }
    size_t mp1 = 0, kk_len = 0, kk_bytes = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size((size_t)k, (size_t)k, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_xz_nested_gram_core");
        return;
    }

    int ok = 1;
    double *ca   = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
    double *cd   = (lev >= 1) ? (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok) : NULL;
    double *ce   = (lev == 2) ? (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok) : NULL;
    double *Mmat = (double *)fhl_c_calloc_chk(kk_len, sizeof(double), &ok);
    if (!ok) {
        free(ca); free(cd); free(ce); free(Mmat);
        memset(ans, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_nested_gram_core");
        return;
    }
    ca[0] = 0.0;
    if (cd) cd[0] = 0.0;
    if (ce) ce[0] = 0.0;
    long double sa = 0.0L, sd = 0.0L, se = 0.0L;
    for (int r = 0; r < m; ++r) {
        int j = risk->ord[r] - 1;
        sa += a[j];
        ca[r + 1] = (double)sa;
        if (lev >= 1) { sd += da[j]; cd[r + 1] = (double)sd; }
        if (lev == 2) { se += dda[j]; ce[r + 1] = (double)se; }
    }

    memset(ans, 0, kk_bytes);

    /* Diagonal / within-row contribution (O(n * count^2)) */
    int *nz_idx0 = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &ok);
    double *nz_val0 = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &ok);
    if (!ok) {
        free(nz_idx0); free(nz_val0); free(ca); free(cd); free(ce); free(Mmat);
        memset(ans, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_nested_gram_core");
        return;
    }
    for (int ii = 0; ii < n; ++ii) {
        int r = risk->obs[ii] - 1;
        int idx = risk->hi[r];
        double W = w[r];
        double V = (lev >= 1) ? dw[r] : 0.0;
        double Qv = (lev == 2) ? ddw[r] : 0.0;
        double c = ca[idx];
        double d = (lev >= 1) ? cd[idx] : 0.0;
        double e = (lev == 2) ? ce[idx] : 0.0;
        double diagcoef;
        if (lev == 0) {
            diagcoef = c * W * W;
        } else if (lev == 1) {
            diagcoef = 2.0 * c * W * V + d * W * W;
        } else {
            diagcoef = c * (2.0 * W * Qv + 2.0 * V * V) + 4.0 * d * W * V + e * W * W;
        }
        int count = 0;
        if (include_x) {
            for (int j = 0; j < p; ++j) {
                double xv = X[r + (size_t)n * (size_t)j];
                if (xv != 0.0) { nz_idx0[count] = j; nz_val0[count] = xv; count++; }
            }
        }
        int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
        for (int t = rstart; t < rend; ++t) {
            double zv = xz->row_val[t];
            if (zv != 0.0) { nz_idx0[count] = p + xz->col_idx[t]; nz_val0[count] = zv; count++; }
        }
        double h_sub = h[r] - diagcoef;
        if (h_sub != 0.0) {
            for (int i = 0; i < count; ++i) {
                int j = nz_idx0[i];
                double h_bj = h_sub * nz_val0[i];
                for (int t = 0; t < count; ++t) {
                    int l = nz_idx0[t];
                    ans[j + (size_t)k * (size_t)l] += h_bj * nz_val0[t];
                }
            }
        }
    }
    free(nz_idx0);
    free(nz_val0);

    /* Strip-parallel unit-stride accumulation into Mmat[t_lo..t_hi-1, j] (4-aligned strips for 0 ULP OMP determinism) */
    int nth = (k >= 128) ? fhl_blas_detect_threads(k) : 1;
    int n_blocks4 = (k + 3) / 4;
    if (nth > n_blocks4) nth = (n_blocks4 >= 1) ? n_blocks4 : 1;
    int blocks_per_th = (n_blocks4 + nth - 1) / nth;
    int strip = blocks_per_th * 4;
    int strip_oom = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) if (nth > 1)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        int t_lo = tid * strip;
        int t_hi = t_lo + strip;
        if (t_hi > k) t_hi = k;
        int len = t_hi - t_lo;
        if (len > 0) {
            int loc_ok = 1;
            double *S_loc = (double *)fhl_c_calloc_chk((size_t)len, sizeof(double), &loc_ok);
            double *D_loc = (lev >= 1) ? (double *)fhl_c_calloc_chk((size_t)len, sizeof(double), &loc_ok) : NULL;
            double *E_loc = (lev == 2) ? (double *)fhl_c_calloc_chk((size_t)len, sizeof(double), &loc_ok) : NULL;
            double *cr_loc = (lev >= 1) ? (double *)fhl_c_malloc_chk((size_t)len, sizeof(double), &loc_ok) : NULL;
            int *nz_i = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &loc_ok);
            double *nz_v = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &loc_ok);
            if (!loc_ok) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
                strip_oom = 1;
            } else {
                for (int ii = 0; ii < n; ++ii) {
                    int r = risk->obs[ii] - 1;
                    int idx = risk->hi[r];
                    double W = w[r];
                    double c = ca[idx];
                    int count = 0;
                    if (include_x) {
                        for (int j = 0; j < p; ++j) {
                            double xv = X[r + (size_t)n * (size_t)j];
                            if (xv != 0.0) { nz_i[count] = j; nz_v[count] = xv; count++; }
                        }
                    }
                    int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
                    for (int t = rstart; t < rend; ++t) {
                        double zv = xz->row_val[t];
                        if (zv != 0.0) { nz_i[count] = p + xz->col_idx[t]; nz_v[count] = zv; count++; }
                    }

                    if (lev == 0) {
                        double cw = c * W;
                        if (cw != 0.0) {
                            for (int i = 0; i < count; ++i) {
                                int j = nz_i[i];
                                fhl_la_axpy(len, cw * nz_v[i], S_loc, Mmat + (size_t)k * (size_t)j + t_lo);
                            }
                        }
                        for (int i = 0; i < count; ++i) {
                            int j = nz_i[i];
                            if (j >= t_lo && j < t_hi) S_loc[j - t_lo] += W * nz_v[i];
                        }
                    } else if (lev == 1) {
                        double V = dw[r], d = cd[idx];
                        double a_s = c * V + d * W, a_d = c * W;
                        for (int u = 0; u < len; ++u) cr_loc[u] = a_s * S_loc[u] + a_d * D_loc[u];
                        for (int i = 0; i < count; ++i) {
                            int j = nz_i[i];
                            fhl_la_axpy(len, nz_v[i], cr_loc, Mmat + (size_t)k * (size_t)j + t_lo);
                        }
                        for (int i = 0; i < count; ++i) {
                            int j = nz_i[i];
                            if (j >= t_lo && j < t_hi) {
                                double bj = nz_v[i];
                                S_loc[j - t_lo] += W * bj;
                                D_loc[j - t_lo] += V * bj;
                            }
                        }
                    } else {
                        double V = dw[r], Qv = ddw[r], d = cd[idx], e = ce[idx];
                        double a_s = c * Qv + 2.0 * d * V + e * W;
                        double a_d = 2.0 * (c * V + d * W);
                        double a_e = c * W;
                        for (int u = 0; u < len; ++u) cr_loc[u] = a_s * S_loc[u] + a_d * D_loc[u] + a_e * E_loc[u];
                        for (int i = 0; i < count; ++i) {
                            int j = nz_i[i];
                            fhl_la_axpy(len, nz_v[i], cr_loc, Mmat + (size_t)k * (size_t)j + t_lo);
                        }
                        for (int i = 0; i < count; ++i) {
                            int j = nz_i[i];
                            if (j >= t_lo && j < t_hi) {
                                double bj = nz_v[i];
                                S_loc[j - t_lo] += W * bj;
                                D_loc[j - t_lo] += V * bj;
                                E_loc[j - t_lo] += Qv * bj;
                            }
                        }
                    }
                }
            }
            free(S_loc); free(D_loc); free(E_loc); free(cr_loc); free(nz_i); free(nz_v);
        }
    }

    if (strip_oom) {
        free(ca); free(cd); free(ce); free(Mmat);
        memset(ans, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_nested_gram_core");
        return;
    }

    /* Single-pass symmetric subtraction: ans[i, j] -= (Mmat[i, j] + Mmat[j, i]) */
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth) schedule(static) if (nth > 1)
#endif
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i <= j; ++i) {
            double v = ans[i + (size_t)k * (size_t)j] -
                       (Mmat[i + (size_t)k * (size_t)j] + Mmat[j + (size_t)k * (size_t)i]);
            ans[i + (size_t)k * (size_t)j] = v;
            ans[j + (size_t)k * (size_t)i] = v;
        }
    }

    free(ca); free(cd); free(ce); free(Mmat);
}

static void fhl_c_xz_general_gram_core(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *h, const double *w, const double *dw, const double *ddw,
    const double *a, const double *da, const double *dda,
    int lev, double *result
) {
    const int n = xz->n;
    const int p = include_x ? xz->p : 0;
    const int Q = xz->Q;
    if (p < 0 || Q < 0 || p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Dimension overflow in fhl_c_xz_general_gram_core");
        return;
    }
    const int k = p + Q;
    const int m = risk->m;
    const double *X = xz->X;
    if (k <= 0) return;
    if (m < 0 || n < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_general_gram_core");
        return;
    }
    size_t mp1 = 0, n3 = 0, mk = 0, kk_bytes = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size((size_t)m, (size_t)k, &mk) ||
        !fhl_safe_mul3_size((size_t)k, (size_t)k, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_xz_general_gram_core");
        return;
    }

    int ok = 1;
    double *wk     = (double *)fhl_c_malloc_chk(n3, sizeof(double), &ok);
    double *col    = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    double *F      = (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok);
    double *D      = (lev >= 1) ? (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok) : NULL;
    double *E      = (lev == 2) ? (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok) : NULL;
    int *nz_idx    = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &ok);
    double *nz_val = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &ok);
    if (!ok) {
        free(nz_idx); free(nz_val); free(wk); free(col); free(F); free(D); free(E);
        memset(result, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_general_gram_core");
        return;
    }

    for (int c = 0; c < p; ++c) {
        const double *b_col = X + (size_t)c * (size_t)n;
        for (int i = 0; i < n; ++i) col[i] = w[i] * b_col[i];
        fhl_c_risk_backward(risk, col, F + (size_t)c * (size_t)m, wk);
        if (lev >= 1) {
            for (int i = 0; i < n; ++i) col[i] = dw[i] * b_col[i];
            fhl_c_risk_backward(risk, col, D + (size_t)c * (size_t)m, wk);
        }
        if (lev == 2) {
            for (int i = 0; i < n; ++i) col[i] = ddw[i] * b_col[i];
            fhl_c_risk_backward(risk, col, E + (size_t)c * (size_t)m, wk);
        }
    }

    for (int cz = 0; cz < Q; ++cz) {
        int c = p + cz;
        int cs = xz->col_ptr[cz], ce = xz->col_ptr[cz + 1];
        fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, w,
                                   F + (size_t)c * (size_t)m, wk);
        if (lev >= 1) {
            fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, dw,
                                       D + (size_t)c * (size_t)m, wk);
        }
        if (lev == 2) {
            fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, ddw,
                                       E + (size_t)c * (size_t)m, wk);
        }
    }

    memset(result, 0, kk_bytes);
    for (int r = 0; r < n; ++r) {
        int count = 0;
        if (include_x) {
            for (int j = 0; j < p; ++j) {
                double xv = X[r + (size_t)n * (size_t)j];
                if (xv != 0.0) {
                    nz_idx[count] = j;
                    nz_val[count] = xv;
                    count++;
                }
            }
        }
        int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
        for (int t = rstart; t < rend; ++t) {
            double zv = xz->row_val[t];
            if (zv != 0.0) {
                nz_idx[count] = p + xz->col_idx[t];
                nz_val[count] = zv;
                count++;
            }
        }
        for (int jj = 0; jj < count; ++jj) {
            int j = nz_idx[jj];
            double hr_vj = h[r] * nz_val[jj];
            for (int ii = 0; ii <= jj; ++ii) {
                int i = nz_idx[ii];
                result[i + (size_t)k * (size_t)j] += hr_vj * nz_val[ii];
            }
        }
    }

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i <= j; ++i) {
            double s = result[i + (size_t)k * (size_t)j];
            const double *fi = F + (size_t)m * (size_t)i;
            const double *fj = F + (size_t)m * (size_t)j;
            if (lev == 0) {
                for (int r = 0; r < m; ++r) s -= a[r] * fi[r] * fj[r];
            } else {
                const double *di = D + (size_t)m * (size_t)i;
                const double *dj = D + (size_t)m * (size_t)j;
                if (lev == 1) {
                    for (int r = 0; r < m; ++r) {
                        s -= a[r] * (di[r] * fj[r] + fi[r] * dj[r]) + da[r] * fi[r] * fj[r];
                    }
                } else {
                    const double *ei = E + (size_t)m * (size_t)i;
                    const double *ej = E + (size_t)m * (size_t)j;
                    for (int r = 0; r < m; ++r) {
                        s -= a[r] * (ei[r] * fj[r] + fi[r] * ej[r] + 2.0 * di[r] * dj[r]) +
                             2.0 * da[r] * (di[r] * fj[r] + fi[r] * dj[r]) + dda[r] * fi[r] * fj[r];
                    }
                }
            }
            result[i + (size_t)k * (size_t)j] = result[j + (size_t)k * (size_t)i] = s;
        }
    }

    free(nz_idx); free(nz_val); free(wk); free(col); free(F);
    if (D) free(D);
    if (E) free(E);
}

void fhl_c_xz_gram0(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    double *out
) {
    const int n = xz->n;
    if (n < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_gram0");
        return;
    }
    int ok = 1;
    double *h = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    if (!ok) {
        free(h);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram0");
        return;
    }
    for (int i = 0; i < n; ++i) h[i] = w[i] * b[i];
    fhl_c_linalg_begin_malloc_scope();
    if (risk->nested && risk->obs != NULL && risk->hi != NULL && risk->ord != NULL) {
        fhl_c_xz_nested_gram_core(xz, include_x, risk, h, w, NULL, NULL, a, NULL, NULL, 0, out);
    } else {
        fhl_c_xz_general_gram_core(xz, include_x, risk, h, w, NULL, NULL, a, NULL, NULL, 0, out);
    }
    free(h);
    if (fhl_c_linalg_end_malloc_scope() && fhl_linalg_malloc_scope_depth == 0) {
        Rf_error("Out of memory in fhl_c_xz_gram0");
    }
}

void fhl_c_xz_gram1(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw, const double *db, const double *da,
    double *out
) {
    const int n = xz->n;
    if (n < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_gram1");
        return;
    }
    int ok = 1;
    double *h = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    if (!ok) {
        free(h);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram1");
        return;
    }
    for (int i = 0; i < n; ++i) h[i] = dw[i] * b[i] + w[i] * db[i];
    fhl_c_linalg_begin_malloc_scope();
    if (risk->nested && risk->obs != NULL && risk->hi != NULL && risk->ord != NULL) {
        fhl_c_xz_nested_gram_core(xz, include_x, risk, h, w, dw, NULL, a, da, NULL, 1, out);
    } else {
        fhl_c_xz_general_gram_core(xz, include_x, risk, h, w, dw, NULL, a, da, NULL, 1, out);
    }
    free(h);
    if (fhl_c_linalg_end_malloc_scope() && fhl_linalg_malloc_scope_depth == 0) {
        Rf_error("Out of memory in fhl_c_xz_gram1");
    }
}

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
) {
    if (K <= 0 || !tr_out) return;
    for (int kk = 0; kk < K; ++kk) tr_out[kk] = 0.0;
    const int n = xz->n;
    const int p = include_x ? xz->p : 0;
    const int Q = xz->Q;
    if (n < 0 || p < 0 || Q < 0 || p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_gram1_trace_batch");
        return;
    }
    const int d = p + Q;
    const int m = risk->m;
    if (d <= 0) return;

    /* General left-truncated fallback: exact fhl_c_xz_gram1 + fhl_c_trace_ab */
    if (!(risk->nested && risk->obs != NULL && risk->hi != NULL && risk->ord != NULL)) {
        size_t dd_len = 0;
        int ok = fhl_safe_mul_size((size_t)d, (size_t)d, &dd_len);
        double *Gtmp = (double *)fhl_c_malloc_chk(dd_len, sizeof(double), &ok);
        if (!ok) {
            free(Gtmp);
            fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram1_trace_batch");
            return;
        }
        for (int kk = 0; kk < K; ++kk) {
            fhl_c_xz_gram1(xz, include_x, risk, w, b, a, dw_list[kk], db_list[kk], da_list[kk], Gtmp);
            if (fhl_c_linalg_had_oom()) { free(Gtmp); return; }
            tr_out[kk] = fhl_c_trace_ab(P_sym, Gtmp, d);
        }
        free(Gtmp);
        return;
    }

    size_t mp1 = 0, km_len = 0, kd_len = 0;
    int ok = fhl_safe_add_size((size_t)m, 1, &mp1) &&
             fhl_safe_mul_size((size_t)K, mp1, &km_len) &&
             fhl_safe_mul_size((size_t)K, (size_t)d, &kd_len);
    double *ca      = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
    double *cd_all  = (double *)fhl_c_malloc_chk(km_len, sizeof(double), &ok);
    double *S       = (double *)fhl_c_calloc_chk((size_t)d, sizeof(double), &ok);
    double *D_all   = (double *)fhl_c_calloc_chk(kd_len, sizeof(double), &ok);
    double *p_row   = (double *)fhl_c_malloc_chk((size_t)d, sizeof(double), &ok);
    double *dot_D   = (double *)fhl_c_malloc_chk((size_t)K, sizeof(double), &ok);
    long double *acc = (long double *)fhl_c_calloc_chk((size_t)K, sizeof(long double), &ok);
    int *nz_idx     = (int *)fhl_c_malloc_chk((size_t)d, sizeof(int), &ok);
    double *nz_val  = (double *)fhl_c_malloc_chk((size_t)d, sizeof(double), &ok);
    if (!ok) {
        free(ca); free(cd_all); free(S); free(D_all); free(p_row);
        free(dot_D); free(acc); free(nz_idx); free(nz_val);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram1_trace_batch");
        return;
    }

    ca[0] = 0.0;
    long double sa = 0.0L;
    for (int r = 0; r < m; ++r) {
        sa += a[risk->ord[r] - 1];
        ca[r + 1] = (double)sa;
    }
    for (int kk = 0; kk < K; ++kk) {
        double *cd_k = cd_all + (size_t)kk * mp1;
        const double *da_k = da_list[kk];
        cd_k[0] = 0.0;
        long double sdk = 0.0L;
        for (int r = 0; r < m; ++r) {
            sdk += da_k[risk->ord[r] - 1];
            cd_k[r + 1] = (double)sdk;
        }
    }

    const double *X = xz->X;
    for (int ii = 0; ii < n; ++ii) {
        int r = risk->obs[ii] - 1;
        int idx = risk->hi[r];
        double W = w[r];
        double br = b[r];
        double c = ca[idx];

        int count = 0;
        if (include_x) {
            for (int j = 0; j < p; ++j) {
                double xv = X[r + (size_t)n * (size_t)j];
                if (xv != 0.0) { nz_idx[count] = j; nz_val[count] = xv; count++; }
            }
        }
        int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
        for (int t = rstart; t < rend; ++t) {
            double zv = xz->row_val[t];
            if (zv != 0.0) { nz_idx[count] = p + xz->col_idx[t]; nz_val[count] = zv; count++; }
        }
        if (count == 0) continue;

        double bPb = 0.0;
        double dot_S = 0.0;
        if (count == 1) {
            int j0 = nz_idx[0];
            double bj0 = nz_val[0];
            const double *p_col = P_sym + (size_t)d * (size_t)j0;
            bPb = (bj0 * bj0) * p_col[j0];
            dot_S = bj0 * fhl_la_dot(d, p_col, S);
            for (int kk = 0; kk < K; ++kk) {
                dot_D[kk] = bj0 * fhl_la_dot(d, p_col, D_all + (size_t)kk * (size_t)d);
            }
        } else {
            memset(p_row, 0, (size_t)d * sizeof(double));
            for (int u = 0; u < count; ++u) {
                int ju = nz_idx[u];
                double bju = nz_val[u];
                const double *p_col = P_sym + (size_t)d * (size_t)ju;
                for (int v = 0; v < count; ++v) {
                    bPb += bju * nz_val[v] * p_col[nz_idx[v]];
                }
                fhl_la_axpy(d, bju, p_col, p_row);
            }
            dot_S = fhl_la_dot(d, p_row, S);
            for (int kk = 0; kk < K; ++kk) {
                dot_D[kk] = fhl_la_dot(d, p_row, D_all + (size_t)kk * (size_t)d);
            }
        }

        for (int kk = 0; kk < K; ++kk) {
            double V = dw_list[kk][r];
            double dk = cd_all[(size_t)kk * mp1 + idx];
            double hk = V * br + W * db_list[kk][r];
            double diagcoef = 2.0 * c * W * V + dk * W * W;
            acc[kk] += (long double)(hk - diagcoef) * (long double)bPb -
                       2.0L * ((long double)(c * V + dk * W) * (long double)dot_S +
                               (long double)(c * W) * (long double)dot_D[kk]);
        }

        for (int u = 0; u < count; ++u) {
            int ju = nz_idx[u];
            double bju = nz_val[u];
            S[ju] += W * bju;
            for (int kk = 0; kk < K; ++kk) {
                D_all[(size_t)kk * (size_t)d + ju] += dw_list[kk][r] * bju;
            }
        }
    }

    for (int kk = 0; kk < K; ++kk) {
        tr_out[kk] = (double)acc[kk];
    }

    free(ca); free(cd_all); free(S); free(D_all); free(p_row);
    free(dot_D); free(acc); free(nz_idx); free(nz_val);
}

void fhl_c_xz_gram2(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw, const double *db, const double *da,
    const double *ddw, const double *ddb, const double *dda,
    double *out
) {
    const int n = xz->n;
    if (n < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_gram2");
        return;
    }
    int ok = 1;
    double *h = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    if (!ok) {
        free(h);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram2");
        return;
    }
    for (int i = 0; i < n; ++i) h[i] = ddw[i] * b[i] + 2.0 * dw[i] * db[i] + w[i] * ddb[i];
    fhl_c_linalg_begin_malloc_scope();
    if (risk->nested && risk->obs != NULL && risk->hi != NULL && risk->ord != NULL) {
        fhl_c_xz_nested_gram_core(xz, include_x, risk, h, w, dw, ddw, a, da, dda, 2, out);
    } else {
        fhl_c_xz_general_gram_core(xz, include_x, risk, h, w, dw, ddw, a, da, dda, 2, out);
    }
    free(h);
    if (fhl_c_linalg_end_malloc_scope() && fhl_linalg_malloc_scope_depth == 0) {
        Rf_error("Out of memory in fhl_c_xz_gram2");
    }
}

void fhl_c_xz_gram2_cross(
    const fhl_c_xz_t *xz, int include_x,
    const fhl_c_risk_t *risk,
    const double *w, const double *b, const double *a,
    const double *dw1, const double *db1, const double *da1,
    const double *dw2, const double *db2, const double *da2,
    const double *ddw12, const double *ddb12, const double *dda12,
    double *out
) {
    const int n = xz->n;
    const int p = include_x ? xz->p : 0;
    const int Q = xz->Q;
    if (p < 0 || Q < 0 || p > INT_MAX - Q) {
        fhl_c_linalg_signal_oom("Dimension overflow in fhl_c_xz_gram2_cross");
        return;
    }
    const int k = p + Q;
    const int m = risk->m;
    const double *X = xz->X;
    if (k <= 0) return;
    if (n < 0 || m < 0) {
        fhl_c_linalg_signal_oom("Invalid dimensions in fhl_c_xz_gram2_cross");
        return;
    }
    size_t mp1 = 0, n3 = 0, mk = 0, kk_bytes = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size((size_t)m, (size_t)k, &mk) ||
        !fhl_safe_mul3_size((size_t)k, (size_t)k, sizeof(double), &kk_bytes)) {
        fhl_c_linalg_signal_oom("Allocation size overflow in fhl_c_xz_gram2_cross");
        return;
    }

    int ok = 1;
    double *h = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    if (!ok) {
        free(h);
        memset(out, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram2_cross");
        return;
    }
    for (int i = 0; i < n; ++i) {
        h[i] = ddw12[i] * b[i] + dw1[i] * db2[i] + dw2[i] * db1[i] + w[i] * ddb12[i];
    }

    if (risk->nested && risk->obs != NULL && risk->hi != NULL && risk->ord != NULL) {
        double *ca     = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
        double *cd1    = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
        double *cd2    = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
        double *ce     = (double *)fhl_c_malloc_chk(mp1, sizeof(double), &ok);
        double *S      = (double *)fhl_c_calloc_chk((size_t)k, sizeof(double), &ok);
        double *D1     = (double *)fhl_c_calloc_chk((size_t)k, sizeof(double), &ok);
        double *D2     = (double *)fhl_c_calloc_chk((size_t)k, sizeof(double), &ok);
        double *E      = (double *)fhl_c_calloc_chk((size_t)k, sizeof(double), &ok);
        double *cross  = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &ok);
        int *nz_idx    = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &ok);
        double *nz_val = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &ok);
        if (!ok) {
            free(ca); free(cd1); free(cd2); free(ce);
            free(S); free(D1); free(D2); free(E); free(cross); free(nz_idx); free(nz_val);
            free(h);
            memset(out, 0, kk_bytes);
            fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram2_cross");
            return;
        }
        ca[0] = cd1[0] = cd2[0] = ce[0] = 0.0;
        long double sa = 0.0L, sd1 = 0.0L, sd2 = 0.0L, se = 0.0L;
        for (int r = 0; r < m; ++r) {
            int j = risk->ord[r] - 1;
            sa  += a[j];
            sd1 += da1[j];
            sd2 += da2[j];
            se  += dda12[j];
            ca[r + 1]  = (double)sa;
            cd1[r + 1] = (double)sd1;
            cd2[r + 1] = (double)sd2;
            ce[r + 1]  = (double)se;
        }

        memset(out, 0, kk_bytes);

        for (int ii = 0; ii < n; ++ii) {
            int r = risk->obs[ii] - 1;
            int idx = risk->hi[r];
            double W = w[r], V1 = dw1[r], V2 = dw2[r], Qv = ddw12[r];
            double c = ca[idx], d1 = cd1[idx], d2 = cd2[idx], e = ce[idx];
            double diagcoef = c * (2.0 * W * Qv + 2.0 * V1 * V2) + 2.0 * d1 * W * V2 + 2.0 * d2 * W * V1 + e * W * W;
            for (int j = 0; j < k; ++j) {
                cross[j] = c * (Qv * S[j] + W * E[j] + V1 * D2[j] + V2 * D1[j]) +
                           d1 * (V2 * S[j] + W * D2[j]) +
                           d2 * (V1 * S[j] + W * D1[j]) +
                           e * W * S[j];
            }
            int count = 0;
            if (include_x) {
                for (int j = 0; j < p; ++j) {
                    double xv = X[r + (size_t)n * (size_t)j];
                    if (xv != 0.0) {
                        nz_idx[count] = j;
                        nz_val[count] = xv;
                        count++;
                    }
                }
            }
            int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
            for (int t = rstart; t < rend; ++t) {
                double zv = xz->row_val[t];
                if (zv != 0.0) {
                    nz_idx[count] = p + xz->col_idx[t];
                    nz_val[count] = zv;
                    count++;
                }
            }
            for (int i = 0; i < count; ++i) {
                int j = nz_idx[i];
                double bj = nz_val[i];
                for (int t = 0; t < k; ++t) {
                    double v = bj * cross[t];
                    out[j + (size_t)k * (size_t)t] -= v;
                    out[t + (size_t)k * (size_t)j] -= v;
                }
                for (int t = 0; t < count; ++t) {
                    int l = nz_idx[t];
                    out[j + (size_t)k * (size_t)l] += (h[r] - diagcoef) * bj * nz_val[t];
                }
                S[j]  += W * bj;
                D1[j] += V1 * bj;
                D2[j] += V2 * bj;
                E[j]  += Qv * bj;
            }
        }

        free(ca); free(cd1); free(cd2); free(ce);
        free(S); free(D1); free(D2); free(E); free(cross); free(nz_idx); free(nz_val);
        free(h);
        return;
    }

    double *wk     = (double *)fhl_c_malloc_chk(n3, sizeof(double), &ok);
    double *col    = (double *)fhl_c_malloc_chk((size_t)n, sizeof(double), &ok);
    double *F      = (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok);
    double *D1     = (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok);
    double *D2     = (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok);
    double *E      = (double *)fhl_c_malloc_chk(mk, sizeof(double), &ok);
    int *nz_idx    = (int *)fhl_c_malloc_chk((size_t)k, sizeof(int), &ok);
    double *nz_val = (double *)fhl_c_malloc_chk((size_t)k, sizeof(double), &ok);
    if (!ok) {
        free(nz_idx); free(nz_val); free(wk); free(col); free(F); free(D1); free(D2); free(E);
        free(h);
        memset(out, 0, kk_bytes);
        fhl_c_linalg_signal_oom("Out of memory in fhl_c_xz_gram2_cross");
        return;
    }

    for (int c = 0; c < p; ++c) {
        const double *b_col = X + (size_t)c * (size_t)n;
        for (int i = 0; i < n; ++i) col[i] = w[i] * b_col[i];
        fhl_c_risk_backward(risk, col, F + (size_t)c * (size_t)m, wk);
        for (int i = 0; i < n; ++i) col[i] = dw1[i] * b_col[i];
        fhl_c_risk_backward(risk, col, D1 + (size_t)c * (size_t)m, wk);
        for (int i = 0; i < n; ++i) col[i] = dw2[i] * b_col[i];
        fhl_c_risk_backward(risk, col, D2 + (size_t)c * (size_t)m, wk);
        for (int i = 0; i < n; ++i) col[i] = ddw12[i] * b_col[i];
        fhl_c_risk_backward(risk, col, E + (size_t)c * (size_t)m, wk);
    }

    for (int cz = 0; cz < Q; ++cz) {
        int c = p + cz;
        int cs = xz->col_ptr[cz], ce = xz->col_ptr[cz + 1];
        fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, w,
                                   F + (size_t)c * (size_t)m, wk);
        fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, dw1,
                                   D1 + (size_t)c * (size_t)m, wk);
        fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, dw2,
                                   D2 + (size_t)c * (size_t)m, wk);
        fhl_c_risk_backward_sparse(risk, cs, ce, xz->row_idx, xz->col_val, ddw12,
                                   E + (size_t)c * (size_t)m, wk);
    }

    memset(out, 0, kk_bytes);
    for (int r = 0; r < n; ++r) {
        int count = 0;
        if (include_x) {
            for (int j = 0; j < p; ++j) {
                double xv = X[r + (size_t)n * (size_t)j];
                if (xv != 0.0) {
                    nz_idx[count] = j;
                    nz_val[count] = xv;
                    count++;
                }
            }
        }
        int rstart = xz->row_ptr[r], rend = xz->row_ptr[r + 1];
        for (int t = rstart; t < rend; ++t) {
            double zv = xz->row_val[t];
            if (zv != 0.0) {
                nz_idx[count] = p + xz->col_idx[t];
                nz_val[count] = zv;
                count++;
            }
        }
        for (int jj = 0; jj < count; ++jj) {
            int j = nz_idx[jj];
            double hr_vj = h[r] * nz_val[jj];
            for (int ii = 0; ii <= jj; ++ii) {
                int i = nz_idx[ii];
                out[i + (size_t)k * (size_t)j] += hr_vj * nz_val[ii];
            }
        }
    }

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i <= j; ++i) {
            double s = out[i + (size_t)k * (size_t)j];
            const double *fi  = F  + (size_t)m * (size_t)i;
            const double *fj  = F  + (size_t)m * (size_t)j;
            const double *d1i = D1 + (size_t)m * (size_t)i;
            const double *d1j = D1 + (size_t)m * (size_t)j;
            const double *d2i = D2 + (size_t)m * (size_t)i;
            const double *d2j = D2 + (size_t)m * (size_t)j;
            const double *ei  = E  + (size_t)m * (size_t)i;
            const double *ej  = E  + (size_t)m * (size_t)j;
            for (int r = 0; r < m; ++r) {
                s -= a[r]     * (ei[r] * fj[r] + fi[r] * ej[r] + d1i[r] * d2j[r] + d2i[r] * d1j[r]) +
                     da1[r]   * (d2i[r] * fj[r] + fi[r] * d2j[r]) +
                     da2[r]   * (d1i[r] * fj[r] + fi[r] * d1j[r]) +
                     dda12[r] * fi[r] * fj[r];
            }
            out[i + (size_t)k * (size_t)j] = out[j + (size_t)k * (size_t)i] = s;
        }
    }

    free(nz_idx); free(nz_val); free(wk); free(col); free(F); free(D1); free(D2); free(E);
    free(h);
}
