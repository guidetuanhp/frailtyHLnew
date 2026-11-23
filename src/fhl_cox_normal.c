#include "fhl_linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#undef exp
#undef log
#define exp(x) fhl_c_r_exp(x)
#define log(x) fhl_c_r_log(x)

/* ==========================================================================
 * Cox Log-Normal Frailty Model Engine (1-RE & Multi-RE, Right & Left-Truncated)
 *   - fhl_c_normal_step   : Single penalized h-likelihood & REML Newton step
 *   - fhl_c_normal_se     : Post-fit dispersion Hessian, standard errors & AIC
 *   - fhl_fit_cox_normal  : Complete outer Newton solver and post-fit summary
 * ========================================================================== */

static void *fhl_n_malloc_chk(size_t count, size_t elem_size, int *ok) {
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

static void *fhl_n_calloc_chk(size_t count, size_t elem_size, int *ok) {
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

/* Evaluate tr((Hinv * Hd1)^2) via symmetric left-product blocks */
static double fhl_c_trace_prod2(const double *Hinv, const double *Hd1, const double *Hd2, int k) {
    return fhl_c_trace_hinv_hd_prod(Hinv, Hd1, Hd2, k);
}

/* Single Newton iteration for log-normal Cox frailty models */
static int fhl_c_lognormal_pn_core(
    int n, int p, int nrand, const int *q, const int *qcum, int Q,
    const double *x, const fhl_c_xz_t *xz,
    const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h0, const double *v_h0, const double *alpha_h0_in,
    int mord, int dord, int varfixed, int varnonneg,
    double *alpha_h0_out, double *beta_h, double *v_h, double *alpha_h,
    double *dft, double *Hinv, double *clam0, double *H,
    double *se_beta_h, double *iD_diag, double *H0
) {
    const int m = risk->m;
    if (n < 0 || p < 0 || Q < 0 || m < 0 || nrand <= 0 || p > INT_MAX - Q) return -1;
    const int k_tot = p + Q;
    size_t mp1 = 0, n3 = 0, qq_len = 0, kk_len = 0, kk_bytes = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)k_tot, (size_t)k_tot, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes)) {
        return -1;
    }
    const char *transN = "N", *transT = "T";
    double one = 1.0, zero = 0.0;
    int inc = 1;

    /* 1. Active component mask & alpha floor:
     *    - varfixed && alpha_h0_in[i] <= 0 : strictly inactive Cox component (v = 0)
     *    - varfixed && alpha_h0_in[i] >  0 : keep positive alpha down to 1e-30 (no 1e-6 clamp)
     *    - !varfixed                       : probe floor 1e-6 for optimization */
    int rc = 1;
    int alloc_ok = 1;
    int *act_comp = NULL, *act_q_flag = NULL;
    double *d_diag = NULL, *v_h0_eff = NULL, *R_diag = NULL;
    double *xb = NULL, *zv = NULL, *expeta = NULL, *tM_exp = NULL, *cla0 = NULL;
    double *as_vec = NULL, *res_v = NULL, *risk_wk = NULL, *dinv = NULL, *H22 = NULL;
    double *T_zz = NULL, *neg_dinv = NULL, *dv_db_k = NULL, *xz_k = NULL, *dmu0_k = NULL;
    double *tM_dmu0 = NULL, *dcla0b_k = NULL, *dc_k = NULL, *da_k = NULL;
    double *step = NULL;
    double *c_vh1 = NULL, *dv1 = NULL, *z_dv1 = NULL, *dexpeta1 = NULL, *tM_dexp1 = NULL;
    double *dcla01 = NULL, *dBi1 = NULL, *dAs1 = NULL;
    double *muu = NULL, *zmuu = NULL;

    fhl_c_linalg_begin_malloc_scope();

    /* Allocate step workspace */
    act_comp   = (int *)fhl_n_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
    act_q_flag = (int *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(int), &alloc_ok);
    d_diag     = (double *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    v_h0_eff   = (double *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    R_diag     = (double *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    xb      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    zv      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    expeta  = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    tM_exp  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    cla0    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    as_vec  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    res_v   = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    risk_wk = (double *)fhl_n_malloc_chk(n3, sizeof(double), &alloc_ok);
    dinv    = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    H22     = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }

    double min_act_a = R_PosInf;
    int any_inact = 0;
    for (int i = 0; i < nrand; ++i) {
        double a0 = alpha_h0_in[i];
        if (!R_FINITE(a0)) { rc = 0; goto cleanup; }
        if (varfixed && a0 <= 0.0) {
            act_comp[i] = 0;
            any_inact = 1;
            alpha_h0_out[i] = 0.0;
            alpha_h[i] = 0.0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                act_q_flag[r] = 0;
                iD_diag[r] = 0.0;
                d_diag[r] = 0.0;
                v_h0_eff[r] = 0.0;
            }
        } else {
            act_comp[i] = 1;
            if (!varfixed && a0 < 9.9999999999999995e-07) a0 = 9.9999999999999995e-07;
            else if (a0 < 1e-30) a0 = 1e-30;
            if (a0 < min_act_a) min_act_a = a0;
            alpha_h0_out[i] = a0;
            alpha_h[i] = a0;
            double ia0 = 1.0 / a0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                act_q_flag[r] = 1;
                iD_diag[r] = ia0;
                d_diag[r] = a0;
                v_h0_eff[r] = v_h0[r];
            }
        }
    }
    const int use_scaled = (any_inact || (min_act_a <= 1e-3));

    /* 2. muh = x %*% beta_h0 + z %*% v_h0_eff; expeta = exp(muh) */
    if (p > 0) {
        F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h0, &inc, &zero, xb, &inc FCONE);
    } else {
        memset(xb, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v_h0_eff, zv);
    for (int i = 0; i < n; ++i) {
        expeta[i] = exp(xb[i] + zv[i]);
    }

    /* 3. cla0 = di / (t(Mi) %*% expeta); clam0 = Mi %*% cla0; As = cla0^2 / di */
    fhl_c_risk_backward(risk, expeta, tM_exp, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    for (int r = 0; r < m; ++r) {
        cla0[r] = di[r] / tM_exp[r];
        as_vec[r] = (cla0[r] * cla0[r]) / di[r];
    }
    fhl_c_risk_forward(risk, cla0, clam0, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

    /* 4. dft1 = t(x) %*% (del - expeta * clam0) */
    for (int i = 0; i < n; ++i) {
        res_v[i] = del[i] - expeta[i] * clam0[i];
    }
    if (p > 0) {
        F77_CALL(dgemv)(transT, &n, &p, &one, x, &n, res_v, &inc, &zero, dft, &inc FCONE);
    }

    /* 5. Base Gram matrix H0 = .G_mat = fhl_gram0(B_xz, Mi, expeta, clam0, as_vec) */
    fhl_c_xz_gram0(xz, 1, risk, expeta, clam0, as_vec, H0);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

    /* 6. First-order Laplace correction for fixed effects (mord == 1) */
    if (mord == 1 && p > 0) {
        T_zz = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        int dinv_ok = 1;
        if (use_scaled) {
            for (int c = 0; c < Q; ++c) {
                memcpy(T_zz + (size_t)c * (size_t)Q, H0 + (size_t)(p + c) * (size_t)k_tot + p, (size_t)Q * sizeof(double));
            }
            dinv_ok = fhl_c_schur_scaled_eval(0, Q, T_zz, d_diag, NULL, 3.1415926535900001,
                                              dinv, NULL, NULL, NULL, NULL, NULL);
        } else {
            for (int c = 0; c < Q; ++c) {
                memcpy(T_zz + (size_t)c * (size_t)Q, H0 + (size_t)(p + c) * (size_t)k_tot + p, (size_t)Q * sizeof(double));
                T_zz[c + (size_t)c * (size_t)Q] += iD_diag[c];
            }
            dinv_ok = fhl_c_solve_inv(T_zz, dinv, Q);
        }
        free(T_zz); T_zz = NULL;
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (!dinv_ok) { rc = 0; goto cleanup; }

        size_t pn_len = 0, pm_len = 0;
        if (!fhl_safe_mul_size((size_t)p, (size_t)n, &pn_len) ||
            !fhl_safe_mul_size((size_t)p, (size_t)m, &pm_len)) {
            rc = -1; goto cleanup;
        }
        neg_dinv = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        dv_db_k  = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        xz_k     = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dmu0_k   = (double *)fhl_n_malloc_chk(pn_len, sizeof(double), &alloc_ok);
        tM_dmu0  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dcla0b_k = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dc_k     = (double *)fhl_n_malloc_chk(pn_len, sizeof(double), &alloc_ok);
        da_k     = (double *)fhl_n_malloc_chk(pm_len, sizeof(double), &alloc_ok);
        const double **dw_ptrs = (const double **)fhl_n_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        const double **db_ptrs = (const double **)fhl_n_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        const double **da_ptrs = (const double **)fhl_n_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        double *ad1_vec        = (double *)fhl_n_malloc_chk((size_t)p, sizeof(double), &alloc_ok);
        if (!alloc_ok) {
            free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
            rc = -1; goto cleanup;
        }

        for (size_t idx = 0; idx < qq_len; ++idx) neg_dinv[idx] = -dinv[idx];

        for (int k = 0; k < p; ++k) {
            double *dmu0_col = dmu0_k + (size_t)k * (size_t)n;
            double *dc_col   = dc_k   + (size_t)k * (size_t)n;
            double *da_col   = da_k   + (size_t)k * (size_t)m;
            dw_ptrs[k] = dmu0_col;
            db_ptrs[k] = dc_col;
            da_ptrs[k] = da_col;

            const double *g_zk = H0 + (size_t)k * (size_t)k_tot + p;
            /* dv_db[, k] = -dinv %*% .G_mat[p + 1:Q, k] */
            F77_CALL(dgemv)(transN, &Q, &Q, &one, neg_dinv, &Q, g_zk, &inc, &zero, dv_db_k, &inc FCONE);
            /* xz[, k] = x[, k] + z %*% dv_db[, k] */
            fhl_c_z_mv(xz, dv_db_k, xz_k);
            const double *x_col = x + (size_t)k * (size_t)n;
            for (int i = 0; i < n; ++i) {
                xz_k[i] = x_col[i] + xz_k[i];
                dmu0_col[i] = expeta[i] * xz_k[i];
            }
            /* dcla0b[, k] = -As * (t(Mi) %*% (Wi * xz[, k])) */
            fhl_c_risk_backward(risk, dmu0_col, tM_dmu0, risk_wk);
            if (fhl_c_linalg_had_oom()) {
                free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
                rc = -1; goto cleanup;
            }
            for (int r = 0; r < m; ++r) {
                dcla0b_k[r] = -as_vec[r] * tM_dmu0[r];
                da_col[r] = (2.0 * cla0[r] / di[r]) * dcla0b_k[r];
            }
            /* dc = Mi %*% dcla0b[, k] */
            fhl_c_risk_forward(risk, dcla0b_k, dc_col, risk_wk);
            if (fhl_c_linalg_had_oom()) {
                free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
                rc = -1; goto cleanup;
            }
        }

        fhl_c_xz_gram1_trace_batch(xz, 0, risk, dinv, expeta, clam0, as_vec,
                                   p, dw_ptrs, db_ptrs, da_ptrs, ad1_vec);
        if (fhl_c_linalg_had_oom()) {
            free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
            rc = -1; goto cleanup;
        }
        for (int k = 0; k < p; ++k) {
            dft[k] -= 0.5 * ad1_vec[k];
        }

        free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
        free(neg_dinv);  neg_dinv  = NULL;
        free(dv_db_k);   dv_db_k   = NULL;
        free(xz_k);      xz_k      = NULL;
        free(dmu0_k);    dmu0_k    = NULL;
        free(tM_dmu0);   tM_dmu0   = NULL;
        free(dcla0b_k);  dcla0b_k  = NULL;
        free(dc_k);      dc_k      = NULL;
        free(da_k);      da_k      = NULL;
    }

    /* 7. dft2 = t(z) %*% (del - expeta * clam0) - iD * v_h0_eff */
    fhl_c_z_tmv(xz, res_v, dft + p);
    for (int r = 0; r < Q; ++r) {
        dft[p + r] = act_q_flag[r] ? (dft[p + r] - iD_diag[r] * v_h0_eff[r]) : 0.0;
    }

    /* 8. H = H0 with + iD on bottom-right Q x Q block */
    memcpy(H, H0, kk_bytes);
    for (int r = 0; r < Q; ++r) {
        H[(p + r) + (size_t)(p + r) * (size_t)k_tot] += iD_diag[r];
    }

    /* 9. Block Schur-complement inverse of H (scaled Schur-Sylvester for small/inactive RE) */
    step = (double *)fhl_n_malloc_chk((size_t)k_tot, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }
    int inv_ok = 1;
    if (use_scaled) {
        inv_ok = fhl_c_schur_scaled_eval(
            p, Q, H0, d_diag, dft, 3.1415926535900001,
            Hinv, H22, step, NULL, NULL, R_diag
        );
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    } else {
        inv_ok = (p > 0)
            ? fhl_c_lognormal_inverse(H, p, Q, (mord == 1) ? dinv : NULL, Hinv, H22)
            : fhl_c_solve_inv(H, Hinv, k_tot);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (inv_ok && p == 0) memcpy(H22, Hinv, qq_len * sizeof(double));
        if (inv_ok) {
            F77_CALL(dgemv)(transN, &k_tot, &k_tot, &one, Hinv, &k_tot, dft, &inc, &zero, step, &inc FCONE);
        }
    }
    if (!inv_ok) { rc = 0; goto cleanup; }

    /* 10. Joint (beta, v) Newton update: be_h = be_h0 + step */
    for (int j = 0; j < p; ++j) {
        beta_h[j] = beta_h0[j] + step[j];
        se_beta_h[j] = sqrt(Hinv[j + (size_t)j * (size_t)k_tot]);
    }
    for (int r = 0; r < Q; ++r) {
        v_h[r] = act_q_flag[r] ? (v_h0_eff[r] + step[p + r]) : 0.0;
    }
    free(step); step = NULL;

    /* 11. Dispersion parameter updates for each variance component i = 0 .. nrand-1 */
    double nonneg_adj = varnonneg ? 2.0 : 0.0;
    if (dord == 0) {
        for (int i = 0; i < nrand; ++i) {
            if (!act_comp[i]) {
                alpha_h[i] = 0.0;
                continue;
            }
            double q_minus_gamma = 0.0;
            if (use_scaled) {
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    q_minus_gamma += R_diag[r];
                }
            } else {
                double tr_hinv = 0.0;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    tr_hinv += Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                }
                q_minus_gamma = (double)q[i] - (tr_hinv / alpha_h[i]);
            }
            if (!varfixed) {
                double ss_v = 0.0;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    ss_v += v_h[r] * v_h[r];
                }
                double denom = q_minus_gamma - nonneg_adj;
                double a_old = alpha_h[i];
                if (!R_FINITE(ss_v) || !R_FINITE(denom)) { rc = 0; goto cleanup; }
                if (denom <= 0.0) {
                    alpha_h[i] = (ss_v > 0.0) ? fmin(a_old * 2.0, a_old + 5.0) : (0.5 * a_old);
                } else {
                    double a_raw = ss_v / denom;
                    double max_up = (a_old < 0.25) ? 1.0 : (5.0 * a_old);
                    alpha_h[i] = (a_raw - a_old > max_up) ? (a_old + max_up) : a_raw;
                }
            }
        }
    } else if (dord == 1 || dord == 2) {
        c_vh1    = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        dv1      = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        z_dv1    = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dexpeta1 = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        tM_dexp1 = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dcla01   = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dBi1     = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dAs1     = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }

        for (int i = 0; i < nrand; ++i) {
            if (!act_comp[i]) {
                alpha_h[i] = 0.0;
                continue;
            }
            memset(c_vh1, 0, (size_t)Q * sizeof(double));
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                double ia_r = iD_diag[r];
                c_vh1[r] = (ia_r * ia_r) * v_h[r];
            }
            /* dv1 = H22 %*% c_vh1 */
            F77_CALL(dgemv)(transN, &Q, &Q, &one, H22, &Q, c_vh1, &inc, &zero, dv1, &inc FCONE);
            /* dexpeta1 = expeta * (z %*% dv1) */
            fhl_c_z_mv(xz, dv1, z_dv1);
            for (int j = 0; j < n; ++j) {
                dexpeta1[j] = expeta[j] * z_dv1[j];
            }
            fhl_c_risk_backward(risk, dexpeta1, tM_dexp1, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            for (int r = 0; r < m; ++r) {
                dcla01[r] = -(di[r] / (tM_exp[r] * tM_exp[r])) * tM_dexp1[r];
                dAs1[r] = (2.0 * (cla0[r] * dcla01[r])) / di[r];
            }
            fhl_c_risk_forward(risk, dcla01, dBi1, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            const double *dw_p = dexpeta1, *db_p = dBi1, *da_p = dAs1;
            double tr_g1 = 0.0;
            fhl_c_xz_gram1_trace_batch(xz, 1, risk, Hinv, expeta, clam0, as_vec,
                                       1, &dw_p, &db_p, &da_p, &tr_g1);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

            double q_minus_gamma1 = 0.0;
            if (use_scaled) {
                double sum_r = 0.0;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) sum_r += R_diag[r];
                q_minus_gamma1 = sum_r + alpha_h[i] * tr_g1;
            } else {
                long double diag_sub = 0.0L;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    double ia_r = iD_diag[r];
                    diag_sub += (long double)(ia_r * ia_r) *
                                (long double)Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                }
                double gamma1 = -alpha_h[i] * (tr_g1 - (double)diag_sub);
                q_minus_gamma1 = (double)q[i] - gamma1;
            }

            double gamma21 = 0.0;
            if (dord == 2) {
                if (p > 0) {
                    F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h, &inc, &zero, xb, &inc FCONE);
                } else {
                    memset(xb, 0, (size_t)n * sizeof(double));
                }
                muu  = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                zmuu = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
                if (!alloc_ok) { rc = -1; goto cleanup; }
                for (int j = 0; j < n; ++j) {
                    muu[j] = exp(xb[j]) * clam0[j];
                }
                fhl_c_z_tmv(xz, muu, zmuu);
                double a_i = alpha_h[i];
                double a2_i = a_i * a_i;
                double a3_i = a2_i * a_i;
                double sum_term1 = 0.0;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    double u_hr = exp(v_h[r]);
                    double ude1_r = u_hr * dv1[r];
                    double zm_r = zmuu[r];
                    double zm_u = zm_r * u_hr;
                    double a_tilde = a_i * zm_u + 1.0;
                    double b_tilde = a2_i * zm_r * ude1_r - 1.0;
                    double at3 = a_tilde * a_tilde * a_tilde;
                    double at4 = at3 * a_tilde;
                    double term11 = ((a2_i * a_tilde * zm_r * ude1_r) - (2.0 * a_i * zm_u * b_tilde)) / at3;
                    double term21 = ((2.0 * a3_i * a_tilde * zm_r * zm_r * u_hr * ude1_r) - (3.0 * a2_i * (zm_u * zm_u) * b_tilde)) / at4;
                    sum_term1 += (3.0 * term11) - (5.0 * term21);
                }
                gamma21 = -(a_i / 12.0) * sum_term1;
                free(muu);  muu  = NULL;
                free(zmuu); zmuu = NULL;
            }
            if (!varfixed) {
                double ss_v = 0.0;
                for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                    ss_v += v_h[r] * v_h[r];
                }
                double denom = q_minus_gamma1 - gamma21 - nonneg_adj;
                double a_old = alpha_h[i];
                if (!R_FINITE(ss_v) || !R_FINITE(denom)) { rc = 0; goto cleanup; }
                if (denom <= 0.0) {
                    alpha_h[i] = (ss_v > 0.0) ? fmin(a_old * 2.0, a_old + 5.0) : (0.5 * a_old);
                } else {
                    double a_raw = ss_v / denom;
                    double max_up = (a_old < 0.25) ? 1.0 : (5.0 * a_old);
                    alpha_h[i] = (a_raw - a_old > max_up) ? (a_old + max_up) : a_raw;
                }
            }
        }

        free(c_vh1);    c_vh1    = NULL;
        free(dv1);      dv1      = NULL;
        free(z_dv1);    z_dv1    = NULL;
        free(dexpeta1); dexpeta1 = NULL;
        free(tM_dexp1); tM_dexp1 = NULL;
        free(dcla01);   dcla01   = NULL;
        free(dBi1);     dBi1     = NULL;
        free(dAs1);     dAs1     = NULL;
    }

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(act_comp); free(act_q_flag); free(d_diag); free(v_h0_eff); free(R_diag);
    free(muu); free(zmuu);
    free(c_vh1); free(dv1); free(z_dv1); free(dexpeta1);
    free(tM_dexp1); free(dcla01); free(dBi1); free(dAs1);
    free(step);
    free(T_zz); free(neg_dinv); free(dv_db_k); free(xz_k); free(dmu0_k);
    free(tM_dmu0); free(dcla0b_k); free(dc_k); free(da_k);
    free(xb); free(zv); free(expeta); free(tM_exp);
    free(cla0); free(as_vec); free(res_v); free(risk_wk); free(dinv); free(H22);
    return rc;
}

/* Post-fit dispersion standard errors, profile/adjusted likelihoods & AIC for log-normal frailty */
static int fhl_c_lognormal_se_core(
    int n, int p, int nrand, const int *q, const int *qcum, int Q,
    const double *x, const fhl_c_xz_t *xz,
    const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h1, const double *v_h1,
    const double *alpha_h, const double *alpha_h1,
    const double *Hinv, const double *clam0, const double *iD_prev, const double *H0,
    int varfixed, int varnonneg,
    double *se_lam_out, double *scalars_out /* [hliken, hpn1, hpn2, hpn3, hlike1, df1] */
) {
    (void)Hinv;
    (void)clam0;
    const double pi_const = 3.1415926535900001;
    const int m = risk->m;
    if (n < 0 || p < 0 || Q < 0 || m < 0 || nrand <= 0 || p > INT_MAX - Q) return -1;
    const int k_tot = p + Q;
    size_t mp1 = 0, n3 = 0, qq_len = 0, kk_len = 0, kk_bytes = 0, rr_len = 0;
    if (!fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)k_tot, (size_t)k_tot, &kk_len) ||
        !fhl_safe_mul_size(kk_len, sizeof(double), &kk_bytes) ||
        !fhl_safe_mul_size((size_t)nrand, (size_t)nrand, &rr_len)) {
        return -1;
    }
    const char *transN = "N";
    double one = 1.0, zero = 0.0;
    int inc = 1;

    int rc = 1;
    int alloc_ok = 1;
    int *act_comp = NULL;
    double *a_se = NULL, *d_diag_se = NULL, *Hinv_se = NULL, *H0_zz_dv1 = NULL;
    double *R_block = NULL, *P_i = NULL, *P_i_G1 = NULL;
    double *iA_diag = NULL, *xb = NULL, *zv = NULL, *eta = NULL, *expeta = NULL;
    double *tM_exp = NULL, *cla0 = NULL, *as_vec = NULL, *risk_wk = NULL;
    double *T_zz = NULL, *H22 = NULL;
    double *Hessian = NULL, *c_vh1 = NULL, *dv1 = NULL, *z_dv1 = NULL, *dexpeta1 = NULL;
    double *tM_dexp1 = NULL, *dcla01 = NULL, *dBi1 = NULL, *dAs1 = NULL, *G_dmat1 = NULL;
    double *Hd1 = NULL, *rhs11 = NULL, *dv11 = NULL, *z_dv11 = NULL, *ddW11 = NULL;
    double *w_zdv11 = NULL, *dw1_zdv1 = NULL, *tM_w_zdv11 = NULL, *tM_dw1_zdv1 = NULL;
    double *ddcla011 = NULL, *ddB11 = NULL, *ddAs11 = NULL, *Hd11 = NULL;
    double *c_vh2 = NULL, *dv2 = NULL, *z_dv2 = NULL, *dexpeta2 = NULL, *tM_dexp2 = NULL;
    double *dcla02 = NULL, *dBi2 = NULL, *dAs2 = NULL, *Hd2 = NULL, *rhs12 = NULL;
    double *dv12 = NULL, *z_dv12 = NULL, *ddW12 = NULL, *dw2_zdv1 = NULL, *w_zdv12 = NULL;
    double *tM_dw2_zdv1 = NULL, *tM_w_zdv12 = NULL, *ddcla012 = NULL, *ddB12 = NULL;
    double *ddAs12 = NULL, *Hd12 = NULL, *iAp = NULL;
    double *two_pi_Hinv = NULL, *muu = NULL, *zmu = NULL;

    fhl_c_linalg_begin_malloc_scope();

    act_comp  = (int *)fhl_n_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
    a_se      = (double *)fhl_n_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
    d_diag_se = (double *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    Hinv_se   = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    iA_diag = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    xb      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    zv      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    eta     = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    expeta  = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    tM_exp  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    cla0    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    as_vec  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    risk_wk = (double *)fhl_n_malloc_chk(n3, sizeof(double), &alloc_ok);
    T_zz    = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    H22     = (double *)fhl_n_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }

    double min_act_a = R_PosInf;
    int any_inact = 0;
    for (int i = 0; i < nrand; ++i) {
        double a1_cand = alpha_h1[i];
        double a0_cand = (alpha_h != NULL) ? alpha_h[i] : a1_cand;
        int boundary_zero = (!varfixed && alpha_h != NULL &&
            !varnonneg && a0_cand <= 1.0000001e-06 && a1_cand <= 1.0000001e-06);
        double a_orig = boundary_zero ? a0_cand : a1_cand;
        if (varfixed && (!R_FINITE(a_orig) || a_orig <= 0.0)) {
            act_comp[i] = 0;
            any_inact = 1;
            a_se[i] = 0.0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                iA_diag[r] = 0.0;
                d_diag_se[r] = 0.0;
            }
        } else {
            act_comp[i] = 1;
            double a1 = (R_FINITE(a_orig) && a_orig > 1e-30) ? a_orig : 1e-30;
            if (a1 < min_act_a) min_act_a = a1;
            a_se[i] = a1;
            double ia = 1.0 / a1;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                iA_diag[r] = ia;
                d_diag_se[r] = a1;
            }
        }
    }
    const int use_scaled_se = (any_inact || (min_act_a <= 1e-3));

    if (p > 0) {
        F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h1, &inc, &zero, xb, &inc FCONE);
    } else {
        memset(xb, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v_h1, zv);
    for (int i = 0; i < n; ++i) {
        eta[i] = xb[i] + zv[i];
        expeta[i] = exp(eta[i]);
    }
    fhl_c_risk_backward(risk, expeta, tM_exp, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    for (int r = 0; r < m; ++r) {
        cla0[r] = di[r] / tM_exp[r];
        as_vec[r] = (cla0[r] * cla0[r]) / di[r];
    }

    (void)iD_prev;
    double logdet_W = 0.0, logdet_S2pi = 0.0;
    const double *Hinv_use = Hinv_se;
    if (use_scaled_se) {
        int sc_ok = fhl_c_schur_scaled_eval(
            p, Q, H0, d_diag_se, NULL, pi_const,
            Hinv_se, H22, NULL, &logdet_W, &logdet_S2pi, NULL
        );
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (!sc_ok) { rc = 0; goto cleanup; }
    } else {
        double *H_se = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        if (!alloc_ok || !H_se) { free(H_se); rc = -1; goto cleanup; }
        memcpy(H_se, H0, kk_bytes);
        for (int c = 0; c < Q; ++c) {
            H_se[(p + c) + (size_t)(p + c) * (size_t)k_tot] += iA_diag[c];
        }
        int inv_ok = (p > 0)
            ? fhl_c_lognormal_inverse(H_se, p, Q, NULL, Hinv_se, H22)
            : fhl_c_solve_inv(H_se, Hinv_se, k_tot);
        if (fhl_c_linalg_had_oom()) { free(H_se); rc = -1; goto cleanup; }
        if (inv_ok && p == 0 && Q > 0) {
            memcpy(H22, Hinv_se, qq_len * sizeof(double));
        }
        free(H_se);
        if (!inv_ok) { rc = 0; goto cleanup; }
    }

    if (!varfixed) {
        Hessian     = (double *)fhl_n_calloc_chk(rr_len, sizeof(double), &alloc_ok);
        H0_zz_dv1   = (double *)fhl_n_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
        c_vh1       = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        dv1         = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        z_dv1       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dexpeta1    = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        tM_dexp1    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dcla01      = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dBi1        = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dAs1        = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        G_dmat1     = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        Hd1         = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        rhs11       = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        dv11        = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        z_dv11      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        ddW11       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        w_zdv11     = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dw1_zdv1    = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        tM_w_zdv11  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        tM_dw1_zdv1 = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        ddcla011    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        ddB11       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        ddAs11      = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        Hd11        = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }

        for (int i = 0; i < nrand; ++i) {
            if (!act_comp[i]) {
                se_lam_out[i] = NA_REAL;
                continue;
            }
            const int s1 = qcum[i];
            const int s2 = qcum[i + 1];
            const int q_i = s2 - s1;
            const double alpha_i = a_se[i];
            const double ia_i  = 1.0 / alpha_i;
            const double ib1_i = ia_i * ia_i;          /* iB1 diagonal on S_i */
            const double dia11_i = 2.0 * ib1_i * ia_i; /* dia11 diagonal on S_i */
            const int use_small_ddk = (alpha_i <= 1e-3);

            memset(c_vh1, 0, (size_t)Q * sizeof(double));
            for (int r = s1; r < s2; ++r) c_vh1[r] = ib1_i * v_h1[r];

            /* dv1 = H22 %*% c_vh1 */
            F77_CALL(dgemv)(transN, &Q, &Q, &one, H22, &Q, c_vh1, &inc, &zero, dv1, &inc FCONE);
            /* dexpeta1 = expeta * (z %*% dv1) */
            fhl_c_z_mv(xz, dv1, z_dv1);
            for (int j = 0; j < n; ++j) dexpeta1[j] = expeta[j] * z_dv1[j];

            fhl_c_risk_backward(risk, dexpeta1, tM_dexp1, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            for (int r = 0; r < m; ++r) {
                dcla01[r] = -(di[r] / (tM_exp[r] * tM_exp[r])) * tM_dexp1[r];
                dAs1[r]   = (2.0 * (cla0[r] * dcla01[r])) / di[r];
            }
            fhl_c_risk_forward(risk, dcla01, dBi1, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            fhl_c_xz_gram1(xz, 1, risk, expeta, clam0, as_vec, dexpeta1, dBi1, dAs1, G_dmat1);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

            memcpy(Hd1, G_dmat1, kk_bytes);
            for (int r = s1; r < s2; ++r) {
                Hd1[(p + r) + (size_t)(p + r) * (size_t)k_tot] -= ib1_i;
            }

            if (use_small_ddk) {
                for (int r = 0; r < Q; ++r) {
                    long double hdv = 0.0L;
                    for (int c = 0; c < Q; ++c) {
                        hdv += (long double)H0[(p + r) + (size_t)(p + c) * (size_t)k_tot] * (long double)dv1[c];
                    }
                    H0_zz_dv1[r] = (double)hdv;
                }
                for (int r = 0; r < Q; ++r) {
                    long double acc = 0.0L;
                    for (int c = 0; c < Q; ++c) {
                        acc += (long double)G_dmat1[(p + r) + (size_t)(p + c) * (size_t)k_tot] * (long double)dv1[c];
                    }
                    if (r >= s1 && r < s2) {
                        acc += 2.0L * (long double)ia_i * (long double)H0_zz_dv1[r];
                    }
                    rhs11[r] = -(double)acc;
                }
            } else {
                /* dv11 = -H22 %*% ((G_dmat1_zz + dia1) %*% dv1 - iB1 * dv1 + dia11 * v_h1) */
                for (int r = 0; r < Q; ++r) {
                    double acc = 0.0;
                    for (int c = 0; c < Q; ++c) {
                        double g_rc = G_dmat1[(p + r) + (size_t)(p + c) * (size_t)k_tot];
                        if (r == c && r >= s1 && r < s2) g_rc -= ib1_i;
                        acc += g_rc * dv1[c];
                    }
                    if (r >= s1 && r < s2) {
                        acc = acc - ib1_i * dv1[r] + dia11_i * v_h1[r];
                    }
                    rhs11[r] = -acc;
                }
            }
            F77_CALL(dgemv)(transN, &Q, &Q, &one, H22, &Q, rhs11, &inc, &zero, dv11, &inc FCONE);

            /* ddW11 = (z %*% dv1)^2 * expeta + (z %*% dv11) * expeta */
            fhl_c_z_mv(xz, dv11, z_dv11);
            for (int j = 0; j < n; ++j) {
                dw1_zdv1[j] = dexpeta1[j] * z_dv1[j];
                w_zdv11[j]  = expeta[j] * z_dv11[j];
                ddW11[j]    = dw1_zdv1[j] + w_zdv11[j];
            }
            fhl_c_risk_backward(risk, dw1_zdv1, tM_dw1_zdv1, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            fhl_c_risk_backward(risk, w_zdv11,  tM_w_zdv11,  risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            for (int r = 0; r < m; ++r) {
                ddcla011[r] = -(dAs1[r] * tM_dexp1[r] + as_vec[r] * tM_dw1_zdv1[r] + as_vec[r] * tM_w_zdv11[r]);
                ddAs11[r]   = (2.0 * (dcla01[r] * dcla01[r]) + 2.0 * (cla0[r] * ddcla011[r])) / di[r];
            }
            fhl_c_risk_forward(risk, ddcla011, ddB11, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            fhl_c_xz_gram2(xz, 1, risk, expeta, clam0, as_vec, dexpeta1, dBi1, dAs1, ddW11, ddB11, ddAs11, Hd11);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

            double ddk1 = 0.0;
            if (use_small_ddk) {
                size_t qi2 = (size_t)q_i * (size_t)q_i;
                size_t qik = (size_t)q_i * (size_t)k_tot;
                R_block = (double *)fhl_n_malloc_chk((qi2 > 0 ? qi2 : 1), sizeof(double), &alloc_ok);
                P_i     = (double *)fhl_n_malloc_chk((qik > 0 ? qik : 1), sizeof(double), &alloc_ok);
                P_i_G1  = (double *)fhl_n_malloc_chk((qik > 0 ? qik : 1), sizeof(double), &alloc_ok);
                if (!alloc_ok) { rc = -1; goto cleanup; }

                for (int b = 0; b < q_i; ++b) {
                    const int col_c = p + s1 + b;
                    for (int a = 0; a < q_i; ++a) {
                        long double rr = 0.0L;
                        for (int k = 0; k < k_tot; ++k) {
                            rr += (long double)H0[(p + s1 + a) + (size_t)k * (size_t)k_tot] *
                                  (long double)(Hinv_use[k + (size_t)col_c * (size_t)k_tot] * ia_i);
                        }
                        R_block[a + (size_t)b * (size_t)q_i] = (double)rr;
                    }
                }
                for (int k = 0; k < k_tot; ++k) {
                    for (int a = 0; a < q_i; ++a) {
                        P_i[a + (size_t)k * (size_t)q_i] = Hinv_use[(p + s1 + a) + (size_t)k * (size_t)k_tot] * ia_i;
                    }
                }
                if (q_i > 0 && k_tot > 0) {
                    F77_CALL(dgemm)(transN, transN, &q_i, &k_tot, &k_tot,
                                    &one, P_i, &q_i, G_dmat1, &k_tot,
                                    &zero, P_i_G1, &q_i FCONE FCONE);
                }
                double tr_r2 = fhl_c_trace_ab(R_block, R_block, q_i);
                long double tr_pg1p = 0.0L;
                for (size_t idx = 0; idx < qik; ++idx) {
                    tr_pg1p += (long double)P_i_G1[idx] * (long double)P_i[idx];
                }
                double tr_sg1 = fhl_c_trace_prod2(Hinv_use, G_dmat1, G_dmat1, k_tot);
                double tr_hg2 = fhl_c_trace_ab(Hinv_use, Hd11, k_tot);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

                long double vh_h0dv = 0.0L;
                for (int r = s1; r < s2; ++r) {
                    vh_h0dv += (long double)(v_h1[r] * ia_i) * (long double)H0_zz_dv1[r];
                }
                ddk1 = -0.5 * tr_r2 + (double)tr_pg1p - 0.5 * tr_sg1 + 0.5 * tr_hg2 +
                       (double)vh_h0dv + (varnonneg ? ib1_i : 0.0);

                free(R_block); R_block = NULL;
                free(P_i);     P_i     = NULL;
                free(P_i_G1);  P_i_G1  = NULL;
            } else {
                double ss_v_i = 0.0, dv1_v_i = 0.0;
                for (int r = s1; r < s2; ++r) {
                    ss_v_i  += v_h1[r] * v_h1[r];
                    dv1_v_i += dv1[r] * v_h1[r];
                }
                ddk1 = -0.5 * (double)q_i * ib1_i + (ia_i * ib1_i) * ss_v_i - ib1_i * dv1_v_i;
                for (int r = s1; r < s2; ++r) {
                    Hd11[(p + r) + (size_t)(p + r) * (size_t)k_tot] += dia11_i;
                }
                double trp1 = fhl_c_trace_prod2(Hinv_use, Hd1, Hd1, k_tot);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                ddk1 += 0.5 * (-trp1 + fhl_c_trace_ab(Hinv_use, Hd11, k_tot));
                if (varnonneg) ddk1 += ib1_i;
            }
            Hessian[i + (size_t)i * (size_t)nrand] = ddk1;

            /* Off-diagonal dispersion Hessian entries kk > i */
            for (int kk = i + 1; kk < nrand; ++kk) {
                if (!act_comp[kk]) continue;
                const int k1 = qcum[kk];
                const int k2 = qcum[kk + 1];
                const double ia_k  = 1.0 / a_se[kk];
                const double ib2_k = ia_k * ia_k;

                c_vh2       = (double *)fhl_n_calloc_chk((size_t)Q, sizeof(double), &alloc_ok);
                dv2         = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
                z_dv2       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                dexpeta2    = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                tM_dexp2    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                dcla02      = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                dBi2        = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                dAs2        = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                Hd2         = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
                rhs12       = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
                dv12        = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
                z_dv12      = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                ddW12       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                dw2_zdv1    = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                w_zdv12     = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                tM_dw2_zdv1 = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                tM_w_zdv12  = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                ddcla012    = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                ddB12       = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
                ddAs12      = (double *)fhl_n_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
                Hd12        = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
                if (!alloc_ok) { rc = -1; goto cleanup; }

                for (int r = k1; r < k2; ++r) c_vh2[r] = ib2_k * v_h1[r];
                F77_CALL(dgemv)(transN, &Q, &Q, &one, H22, &Q, c_vh2, &inc, &zero, dv2, &inc FCONE);
                fhl_c_z_mv(xz, dv2, z_dv2);
                for (int j = 0; j < n; ++j) dexpeta2[j] = expeta[j] * z_dv2[j];

                fhl_c_risk_backward(risk, dexpeta2, tM_dexp2, risk_wk);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                for (int r = 0; r < m; ++r) {
                    dcla02[r] = -(di[r] / (tM_exp[r] * tM_exp[r])) * tM_dexp2[r];
                    dAs2[r]   = (2.0 * (cla0[r] * dcla02[r])) / di[r];
                }
                fhl_c_risk_forward(risk, dcla02, dBi2, risk_wk);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

                /* Note: since S_i and S_kk are disjoint, C * D = 0, iB1 * D = 0, dia12 = 0 */
                double dv1_ib2_v = 0.0;
                for (int r = k1; r < k2; ++r) {
                    dv1_ib2_v += dv1[r] * ib2_k * v_h1[r];
                }
                double dd12 = -dv1_ib2_v;

                /* dv12 = -H22 %*% ((G_dmat1_zz + dia1) %*% dv2 - iB2 * dv1 + dia12 * v_h1) */
                for (int r = 0; r < Q; ++r) {
                    double acc = 0.0;
                    for (int c = 0; c < Q; ++c) {
                        double g_rc = G_dmat1[(p + r) + (size_t)(p + c) * (size_t)k_tot];
                        if (r == c && r >= s1 && r < s2) g_rc -= ib1_i;
                        acc += g_rc * dv2[c];
                    }
                    if (r >= k1 && r < k2) {
                        acc -= ib2_k * dv1[r];
                    }
                    rhs12[r] = -acc;
                }
                F77_CALL(dgemv)(transN, &Q, &Q, &one, H22, &Q, rhs12, &inc, &zero, dv12, &inc FCONE);

                fhl_c_z_mv(xz, dv12, z_dv12);
                for (int j = 0; j < n; ++j) {
                    dw2_zdv1[j] = dexpeta2[j] * z_dv1[j];
                    w_zdv12[j]  = expeta[j] * z_dv12[j];
                    ddW12[j]    = dw2_zdv1[j] + w_zdv12[j];
                }
                fhl_c_risk_backward(risk, dw2_zdv1, tM_dw2_zdv1, risk_wk);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                fhl_c_risk_backward(risk, w_zdv12,  tM_w_zdv12,  risk_wk);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                for (int r = 0; r < m; ++r) {
                    ddcla012[r] = -(dAs2[r] * tM_dexp1[r] + as_vec[r] * tM_dw2_zdv1[r] + as_vec[r] * tM_w_zdv12[r]);
                    ddAs12[r]   = (2.0 * (dcla02[r] * dcla01[r]) + 2.0 * (cla0[r] * ddcla012[r])) / di[r];
                }
                fhl_c_risk_forward(risk, ddcla012, ddB12, risk_wk);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

                fhl_c_xz_gram1(xz, 1, risk, expeta, clam0, as_vec, dexpeta2, dBi2, dAs2, Hd2);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                for (int r = k1; r < k2; ++r) {
                    Hd2[(p + r) + (size_t)(p + r) * (size_t)k_tot] -= ib2_k;
                }
                fhl_c_xz_gram2_cross(
                    xz, 1, risk, expeta, clam0, as_vec,
                    dexpeta1, dBi1, dAs1, dexpeta2, dBi2, dAs2,
                    ddW12, ddB12, ddAs12, Hd12
                );
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                double trp2 = fhl_c_trace_prod2(Hinv_use, Hd1, Hd2, k_tot);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                dd12 += 0.5 * (-trp2 + fhl_c_trace_ab(Hinv_use, Hd12, k_tot));
                Hessian[i + (size_t)kk * (size_t)nrand] = dd12;
                Hessian[kk + (size_t)i * (size_t)nrand] = dd12;

                free(c_vh2);       c_vh2       = NULL;
                free(dv2);         dv2         = NULL;
                free(z_dv2);       z_dv2       = NULL;
                free(dexpeta2);    dexpeta2    = NULL;
                free(tM_dexp2);    tM_dexp2    = NULL;
                free(dcla02);      dcla02      = NULL;
                free(dBi2);        dBi2        = NULL;
                free(dAs2);        dAs2        = NULL;
                free(Hd2);         Hd2         = NULL;
                free(rhs12);       rhs12       = NULL;
                free(dv12);        dv12        = NULL;
                free(z_dv12);      z_dv12      = NULL;
                free(ddW12);       ddW12       = NULL;
                free(dw2_zdv1);    dw2_zdv1    = NULL;
                free(w_zdv12);     w_zdv12     = NULL;
                free(tM_dw2_zdv1); tM_dw2_zdv1 = NULL;
                free(tM_w_zdv12);  tM_w_zdv12  = NULL;
                free(ddcla012);    ddcla012    = NULL;
                free(ddB12);       ddB12       = NULL;
                free(ddAs12);      ddAs12      = NULL;
                free(Hd12);        Hd12        = NULL;
            }
        }

        /* Post-fit nrand x nrand dispersion observed information may be nonsingular
         * indefinite when a variance component lies on the alpha = 0 boundary
         * (Ha, Jeong and Lee, 2017, Sec. 5.4.2); invert via LU and return NaN on negative diag. */
        iAp = (double *)fhl_n_malloc_chk(rr_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        int iap_ok = fhl_c_solve_gesv_inv(Hessian, iAp, nrand);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (!iap_ok) {
            for (int i = 0; i < nrand; ++i) se_lam_out[i] = NA_REAL;
        } else {
            for (int i = 0; i < nrand; ++i) {
                double d_ii = iAp[i + (size_t)i * (size_t)nrand];
                se_lam_out[i] = (d_ii < 0.0) ? R_NaN : (R_FINITE(d_ii) ? sqrt(d_ii) : NA_REAL);
            }
        }
        free(iAp);         iAp         = NULL;
        free(Hessian);     Hessian     = NULL;
        free(c_vh1);       c_vh1       = NULL;
        free(dv1);         dv1         = NULL;
        free(z_dv1);       z_dv1       = NULL;
        free(dexpeta1);    dexpeta1    = NULL;
        free(tM_dexp1);    tM_dexp1    = NULL;
        free(dcla01);      dcla01      = NULL;
        free(dBi1);        dBi1        = NULL;
        free(dAs1);        dAs1        = NULL;
        free(G_dmat1);     G_dmat1     = NULL;
        free(Hd1);         Hd1         = NULL;
        free(rhs11);       rhs11       = NULL;
        free(dv11);        dv11        = NULL;
        free(z_dv11);      z_dv11      = NULL;
        free(ddW11);       ddW11       = NULL;
        free(w_zdv11);     w_zdv11     = NULL;
        free(dw1_zdv1);    dw1_zdv1    = NULL;
        free(tM_w_zdv11);  tM_w_zdv11  = NULL;
        free(tM_dw1_zdv1); tM_dw1_zdv1 = NULL;
        free(ddcla011);    ddcla011    = NULL;
        free(ddB11);       ddB11       = NULL;
        free(ddAs11);      ddAs11      = NULL;
        free(Hd11);        Hd11        = NULL;
    }

    /* Evaluate h-likelihoods, Laplace marginal likelihoods, and effective degrees of freedom */
    double hlike1_p1 = 0.0, hlike1_p2 = 0.0;
    for (int i = 0; i < n; ++i) hlike1_p1 += del[i] * eta[i];
    for (int r = 0; r < m; ++r) hlike1_p2 += di[r] * log(tM_exp[r]);
    double hlike1 = hlike1_p1 - hlike1_p2;

    double hlike2 = 0.0, hlike3 = 0.0;
    for (int i = 0; i < nrand; ++i) {
        if (act_comp[i]) {
            hlike2 = hlike2 - ((double)q[i] / 2.0) * log(2.0 * pi_const) - (((double)q[i] / 2.0) * log(a_se[i]));
            double ss_v = 0.0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                ss_v += v_h1[r] * v_h1[r];
            }
            hlike3 = hlike3 - ss_v / (2.0 * a_se[i]);
        }
    }
    double hliken = hlike1 + hlike2 + hlike3;

    muu = (double *)fhl_n_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    zmu = (double *)fhl_n_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }
    for (int j = 0; j < n; ++j) muu[j] = exp(xb[j]) * clam0[j];
    fhl_c_z_tmv(xz, muu, zmu);

    double second = 0.0;
    for (int i = 0; i < nrand; ++i) {
        if (!act_comp[i]) continue;
        double a1 = a_se[i];
        double sum_t4 = 0.0;
        for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
            double b31 = zmu[r] * exp(v_h1[r]);
            double inv_d = a1 / (1.0 + a1 * b31);
            double S11 = 3.0 * (b31 * (inv_d * inv_d));
            double S21 = 5.0 * ((b31 * b31) * (inv_d * inv_d * inv_d));
            sum_t4 += (S11 - S21);
        }
        second -= sum_t4 / 24.0;
    }
    free(muu); muu = NULL;
    free(zmu); zmu = NULL;

    double hpn1 = hlike1, hpn2 = hlike1;
    if (use_scaled_se) {
        hpn2 = hlike1 + hlike3 - 0.5 * logdet_W;
        hpn1 = hpn2 - 0.5 * logdet_S2pi;
    } else {
        two_pi_Hinv = (double *)fhl_n_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        for (size_t idx = 0; idx < kk_len; ++idx) {
            two_pi_Hinv[idx] = 2.0 * pi_const * Hinv_use[idx];
        }
        double logdet1 = fhl_c_logdet_spd(two_pi_Hinv, k_tot, 0, 0.0, 0, 0.0);
        free(two_pi_Hinv); two_pi_Hinv = NULL;
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        hpn1 = hliken + 0.5 * logdet1;

        /* H22_new = .G_mat[p+1:p+Q, p+1:p+Q] + iD(alpha_h1), scaled by 1 / (2 * pi) */
        for (int c = 0; c < Q; ++c) {
            for (int r = 0; r < Q; ++r) {
                double val = H0[(p + r) + (size_t)(p + c) * (size_t)k_tot];
                if (r == c) val += iA_diag[r];
                T_zz[r + (size_t)c * (size_t)Q] = val / (2.0 * pi_const);
            }
        }
        double logdet2 = fhl_c_logdet_spd(T_zz, Q, 0, 0.0, 0, 0.0);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        hpn2 = hliken - 0.5 * logdet2;
    }
    double hpn3 = hpn1 + second;
    double df1  = fhl_c_trace_ab(Hinv_use, H0, k_tot);

    scalars_out[0] = hliken;
    scalars_out[1] = hpn1;
    scalars_out[2] = hpn2;
    scalars_out[3] = hpn3;
    scalars_out[4] = hlike1;
    scalars_out[5] = df1;

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(act_comp); free(a_se); free(d_diag_se); free(Hinv_se);
    free(H0_zz_dv1); free(R_block); free(P_i); free(P_i_G1);
    free(two_pi_Hinv); free(muu); free(zmu); free(iAp);
    free(c_vh2); free(dv2); free(z_dv2); free(dexpeta2); free(tM_dexp2);
    free(dcla02); free(dBi2); free(dAs2); free(Hd2); free(rhs12);
    free(dv12); free(z_dv12); free(ddW12); free(dw2_zdv1); free(w_zdv12);
    free(tM_dw2_zdv1); free(tM_w_zdv12); free(ddcla012); free(ddB12); free(ddAs12); free(Hd12);
    free(Hessian); free(c_vh1); free(dv1); free(z_dv1); free(dexpeta1);
    free(tM_dexp1); free(dcla01); free(dBi1); free(dAs1); free(G_dmat1);
    free(Hd1); free(rhs11); free(dv11); free(z_dv11); free(ddW11);
    free(w_zdv11); free(dw1_zdv1); free(tM_w_zdv11); free(tM_dw1_zdv1);
    free(ddcla011); free(ddB11); free(ddAs11); free(Hd11);
    free(T_zz); free(H22); free(iA_diag);
    free(xb); free(zv); free(eta); free(expeta); free(tM_exp); free(cla0); free(as_vec); free(risk_wk);
    return rc;
}

/* ==========================================================================
 * R .Call Interface: Cox Normal Frailty Model (fhl_fit_cox_normal)
 * ========================================================================== */
SEXP fhl_fit_cox_normal(
    SEXP s_x, SEXP s_z_list, SEXP s_y, SEXP s_del, SEXP s_Mi,
    SEXP s_idx2, SEXP s_t2, SEXP s_di,
    SEXP s_beta_init, SEXP s_v_init, SEXP s_alpha_init,
    SEXP s_mord, SEXP s_dord, SEXP s_varfixed, SEXP s_varnonneg,
    SEXP s_max_iter, SEXP s_convergence
) {
    const int n = Rf_nrows(s_x);
    const int p = Rf_ncols(s_x);
    const int is_z_list = (TYPEOF(s_z_list) == VECSXP && !Rf_isS4(s_z_list));
    const int nrand = is_z_list ? Rf_length(s_z_list) : 1;
    int *q = (int *)R_alloc((size_t)nrand, sizeof(int));
    int *qcum = (int *)R_alloc((size_t)(nrand + 1), sizeof(int));

    fhl_c_xz_t xz;
    fhl_c_xz_init(&xz, s_x, s_z_list, n, q, qcum);
    const int Q = xz.Q;
    const int k_tot = p + Q;
    const int mord = Rf_asInteger(s_mord);
    const int dord = Rf_asInteger(s_dord);
    const int varfixed = Rf_asLogical(s_varfixed);
    const int varnonneg = Rf_asLogical(s_varnonneg);
    const int max_iter = Rf_asInteger(s_max_iter);
    const double convergence = Rf_asReal(s_convergence);

    const double *x_ptr   = xz.X;
    const double *del_ptr = fhl_c_real_ptr(s_del);
    const double *di_ptr  = fhl_c_real_ptr(s_di);

    fhl_c_risk_t risk;
    fhl_c_unpack_risk(s_Mi, &risk);

    SEXP s_beta_h0      = PROTECT(Rf_allocMatrix(REALSXP, p, 1));
    SEXP s_v_h0         = PROTECT(Rf_allocMatrix(REALSXP, Q, 1));
    SEXP s_alpha_h0_out = PROTECT(Rf_allocVector(REALSXP, nrand));
    SEXP s_beta_h       = PROTECT(Rf_allocMatrix(REALSXP, p, 1));
    SEXP s_v_h          = PROTECT(Rf_allocMatrix(REALSXP, Q, 1));
    SEXP s_alpha_h      = PROTECT(Rf_allocVector(REALSXP, nrand));
    SEXP s_dft          = PROTECT(Rf_allocMatrix(REALSXP, k_tot, 1));
    SEXP s_Hinv         = PROTECT(Rf_allocMatrix(REALSXP, k_tot, k_tot));
    SEXP s_clam0        = PROTECT(Rf_allocMatrix(REALSXP, n, 1));
    SEXP s_H            = PROTECT(Rf_allocMatrix(REALSXP, k_tot, k_tot));
    SEXP s_se_beta_h    = PROTECT(Rf_allocMatrix(REALSXP, p, 1));
    SEXP s_U            = PROTECT(Rf_allocMatrix(REALSXP, Q, Q));
    SEXP s_H0           = PROTECT(Rf_allocMatrix(REALSXP, k_tot, k_tot));

    double *beta_curr  = (double *)R_alloc((size_t)(p > 0 ? p : 1), sizeof(double));
    double *v_curr     = (double *)R_alloc((size_t)(Q > 0 ? Q : 1), sizeof(double));
    double *alpha_curr = (double *)R_alloc((size_t)nrand, sizeof(double));
    double *iD_diag    = (double *)R_alloc((size_t)(Q > 0 ? Q : 1), sizeof(double));

    if (p > 0) memcpy(beta_curr, fhl_c_real_ptr(s_beta_init), (size_t)p * sizeof(double));
    if (Q > 0) memcpy(v_curr,    fhl_c_real_ptr(s_v_init),    (size_t)Q * sizeof(double));
    memcpy(alpha_curr, fhl_c_real_ptr(s_alpha_init), (size_t)nrand * sizeof(double));

    double err = 1.0;
    double print_err = 1.0;
    int print_i = 1;

    for (int ii = 1; ii <= max_iter; ++ii) {
        if (err >= convergence) {
            if (p > 0) memcpy(REAL(s_beta_h0), beta_curr, (size_t)p * sizeof(double));
            if (Q > 0) memcpy(REAL(s_v_h0),    v_curr,    (size_t)Q * sizeof(double));

            int ok = fhl_c_lognormal_pn_core(
                n, p, nrand, q, qcum, Q,
                x_ptr, &xz, del_ptr, &risk, di_ptr,
                REAL(s_beta_h0), REAL(s_v_h0), alpha_curr,
                mord, dord, varfixed, varnonneg,
                REAL(s_alpha_h0_out), REAL(s_beta_h), REAL(s_v_h), REAL(s_alpha_h),
                REAL(s_dft), REAL(s_Hinv), REAL(s_clam0), REAL(s_H),
                REAL(s_se_beta_h), iD_diag, REAL(s_H0)
            );
            if (ok < 0) {
                UNPROTECT(13);
                Rf_error("Out of memory in fhl_fit_cox_normal at iteration %d", ii);
            }
            if (!ok) {
                UNPROTECT(13);
                Rf_error("Singular matrix in fhl_fit_cox_normal at iteration %d", ii);
            }

            double step_err = 0.0;
            for (int i = 0; i < nrand; ++i) {
                step_err += fabs(REAL(s_alpha_h0_out)[i] - REAL(s_alpha_h)[i]);
            }
            for (int r = 0; r < Q; ++r) {
                step_err += fabs(REAL(s_v_h)[r] - REAL(s_v_h0)[r]);
            }
            for (int j = 0; j < p; ++j) {
                step_err += fabs(REAL(s_beta_h)[j] - REAL(s_beta_h0)[j]);
            }

            if (p > 0) memcpy(beta_curr, REAL(s_beta_h), (size_t)p * sizeof(double));
            if (Q > 0) memcpy(v_curr,    REAL(s_v_h),    (size_t)Q * sizeof(double));
            memcpy(alpha_curr, REAL(s_alpha_h), (size_t)nrand * sizeof(double));

            err = step_err;
            print_err = err;
            print_i = ii;
        }
    }

    memset(REAL(s_U), 0, (size_t)Q * (size_t)Q * sizeof(double));
    for (int r = 0; r < Q; ++r) {
        REAL(s_U)[r + (size_t)r * (size_t)Q] = iD_diag[r];
    }

    SEXP res2 = PROTECT(Rf_allocVector(VECSXP, 22));
    SET_VECTOR_ELT(res2, 0,  s_x);
    SET_VECTOR_ELT(res2, 1,  s_z_list);
    SET_VECTOR_ELT(res2, 2,  s_y);
    SET_VECTOR_ELT(res2, 3,  s_del);
    SET_VECTOR_ELT(res2, 4,  s_Mi);
    SET_VECTOR_ELT(res2, 5,  s_idx2);
    SET_VECTOR_ELT(res2, 6,  s_t2);
    SET_VECTOR_ELT(res2, 7,  s_di);
    SET_VECTOR_ELT(res2, 8,  s_beta_h0);
    SET_VECTOR_ELT(res2, 9,  s_v_h0);
    SET_VECTOR_ELT(res2, 10, s_beta_h);
    SET_VECTOR_ELT(res2, 11, s_v_h);
    SET_VECTOR_ELT(res2, 12, s_alpha_h0_out);
    SET_VECTOR_ELT(res2, 13, s_alpha_h);
    SET_VECTOR_ELT(res2, 14, s_dft);
    SET_VECTOR_ELT(res2, 15, s_Hinv);
    SET_VECTOR_ELT(res2, 16, s_clam0);
    SET_VECTOR_ELT(res2, 17, s_H);
    SET_VECTOR_ELT(res2, 18, R_NilValue);
    SET_VECTOR_ELT(res2, 19, s_se_beta_h);
    SET_VECTOR_ELT(res2, 20, s_U);
    SET_VECTOR_ELT(res2, 21, s_H0);

    double *se_lam_num = (double *)R_alloc((size_t)nrand, sizeof(double));
    double scalars[6];
    int ok_se = fhl_c_lognormal_se_core(
        n, p, nrand, q, qcum, Q,
        x_ptr, &xz, del_ptr, &risk, di_ptr,
        REAL(s_beta_h), REAL(s_v_h), REAL(s_alpha_h0_out), REAL(s_alpha_h),
        REAL(s_Hinv), REAL(s_clam0), iD_diag, REAL(s_H0),
        varfixed, varnonneg, se_lam_num, scalars
    );
    if (ok_se < 0) {
        UNPROTECT(14);
        Rf_error("Out of memory in fhl_fit_cox_normal post-fit SE");
    }
    if (!ok_se) {
        UNPROTECT(14);
        Rf_error("Singular matrix in fhl_fit_cox_normal post-fit SE");
    }

    SEXP s_se_lam;
    if (varfixed) {
        s_se_lam = PROTECT(Rf_allocVector(STRSXP, nrand));
        for (int i = 0; i < nrand; ++i) SET_STRING_ELT(s_se_lam, i, Rf_mkChar("NULL"));
    } else {
        s_se_lam = PROTECT(Rf_allocVector(REALSXP, nrand));
        memcpy(REAL(s_se_lam), se_lam_num, (size_t)nrand * sizeof(double));
    }

    SEXP res3 = PROTECT(Rf_allocVector(VECSXP, 7));
    SET_VECTOR_ELT(res3, 0, s_se_lam);
    for (int j = 0; j < 6; ++j) {
        SEXP s_val = PROTECT(Rf_allocMatrix(REALSXP, 1, 1));
        REAL(s_val)[0] = scalars[j];
        SET_VECTOR_ELT(res3, j + 1, s_val);
        UNPROTECT(1);
    }
    SET_VECTOR_ELT(res3, 6, Rf_ScalarReal(scalars[5]));

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, res2);
    SET_VECTOR_ELT(out, 1, res3);
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger(print_i));
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal(print_err));

    UNPROTECT(17);
    return out;
}
