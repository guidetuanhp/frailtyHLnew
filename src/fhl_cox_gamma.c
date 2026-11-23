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
 * Cox Gamma Frailty Model Engine (1-RE & Multi-RE, Right & Left-Truncated)
 *   - fhl_fit_cox_gamma : Complete Newton estimation, Hinv_u, SE & likelihoods
 * ========================================================================== */

static void *fhl_g_malloc_chk(size_t count, size_t elem_size, int *ok) {
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

static double fhl_gamma_v_minus_expm1(double v) {
    if (fabs(v) <= 1e-2) {
        return -v * v * (0.5 + v * ((1.0 / 6.0) + v * ((1.0 / 24.0) +
               v * ((1.0 / 120.0) + v * (1.0 / 720.0)))));
    }
    return v - expm1(v);
}

/* Cancellation-free evaluation of (e^{2v} - 2 v e^v - 1) / alpha^3 */
static double fhl_gamma_pair1_over_a3(double v, double a) {
    if (fabs(v) <= 5e-2) {
        double r = v / a;
        double poly = (1.0 / 3.0) + v * ((1.0 / 3.0) + v * ((11.0 / 60.0) +
                      v * ((13.0 / 180.0) + v * ((19.0 / 840.0) + v * (1.0 / 168.0)))));
        return (r * r * r) * poly;
    }
    return (expm1(2.0 * v) - 2.0 * v * exp(v)) / (a * a * a);
}

/* Cancellation-free dispersion score base (k2_base) and 2nd derivative scalar (k21) */
static void fhl_gamma_small_alpha_k2_k21(
    int q_i, double alpha_i, int varnonneg,
    const double *v_seg,
    double *k2_base_ex_out, double *k2_base_out,
    double *k21_rem_out, double *k21_out
) {
    double nonneg_adj = varnonneg ? alpha_i : 0.0;
    double ial = 1.0 / alpha_i;
    double ial2 = ial * ial;
    double ial3 = ial2 * ial;

    if (alpha_i <= 1e-3) {
        double sum_vv_ex = 0.0;
        for (int r = 0; r < q_i; ++r) {
            sum_vv_ex += fhl_gamma_v_minus_expm1(v_seg[r]);
        }
        double a2 = alpha_i * alpha_i;
        double a4 = a2 * a2;
        double a6 = a4 * a2;
        double a8 = a4 * a4;
        double k2_base_ex = sum_vv_ex + (double)q_i * (a2 / 12.0 - a4 / 120.0 + a6 / 252.0 - a8 / 240.0) - nonneg_adj;
        double k2_base = k2_base_ex + 0.5 * (double)q_i * alpha_i;
        double k21_rem = ial3 * (2.0 * sum_vv_ex + (double)q_i * (a4 / 60.0 - a6 / 63.0 + a8 / 40.0));
        double k21 = 0.5 * (double)q_i * ial2 + k21_rem;
        if (k2_base_ex_out) *k2_base_ex_out = k2_base_ex;
        if (k2_base_out) *k2_base_out = k2_base;
        if (k21_rem_out) *k21_rem_out = k21_rem;
        if (k21_out) *k21_out = k21;
    } else {
        double dp = Rf_digamma(ial);
        double ddp = Rf_trigamma(ial);
        double sum_vv_minus_uu = 0.0;
        for (int r = 0; r < q_i; ++r) {
            sum_vv_minus_uu += (v_seg[r] - exp(v_seg[r]));
        }
        double k2_base = sum_vv_minus_uu + ((double)q_i * (-log(alpha_i) + 1.0 - dp)) - nonneg_adj;
        double k21 = ial3 * (2.0 * sum_vv_minus_uu + ((double)q_i * ((-2.0 * log(alpha_i)) + 3.0 - (2.0 * dp) - (ial * ddp))));
        if (k2_base_ex_out) *k2_base_ex_out = k2_base - 0.5 * (double)q_i * alpha_i;
        if (k2_base_out) *k2_base_out = k2_base;
        if (k21_rem_out) *k21_rem_out = k21 - 0.5 * (double)q_i * ial2;
        if (k21_out) *k21_out = k21;
    }
}

/* Cancellation-free random-effect h-likelihood contribution for active Gamma RE */
static double fhl_gamma_se_hlike2_comp(
    double a1, int s1, int s2,
    const double *v_h1, const double *u_h1, double pi_const
) {
    double acc = 0.0;
    if (a1 <= 1e-3) {
        double a2 = a1 * a1;
        double a3 = a2 * a1;
        double a5 = a3 * a2;
        double a7 = a5 * a2;
        double c_rem = -0.5 * log(2.0 * pi_const * a1) - a1 / 12.0 + a3 / 360.0 - a5 / 1260.0 + a7 / 1680.0;
        for (int r = s1; r < s2; ++r) {
            acc += (fhl_gamma_v_minus_expm1(v_h1[r]) / a1) + c_rem;
        }
    } else {
        double i_alp1 = 1.0 / a1;
        double c_alp1 = -Rf_lgammafn(i_alp1) - (i_alp1 * log(a1));
        for (int r = s1; r < s2; ++r) {
            acc += ((v_h1[r] - u_h1[r]) / a1) + c_alp1;
        }
    }
    return acc;
}

/* Combined Gamma hlike2 + 0.5*log(2*pi*alpha_i) - 0.5*v_r with log(2*pi*alpha) pre-cancelled */
static double fhl_gamma_se_hlike2_cancel_comp(
    double a1, int s1, int s2,
    const double *v_h1, const double *u_h1, double pi_const
) {
    double acc = 0.0;
    if (a1 <= 1e-3) {
        double a2 = a1 * a1;
        double a3 = a2 * a1;
        double a5 = a3 * a2;
        double a7 = a5 * a2;
        double stir_rem = -a1 / 12.0 + a3 / 360.0 - a5 / 1260.0 + a7 / 1680.0;
        for (int r = s1; r < s2; ++r) {
            acc += (fhl_gamma_v_minus_expm1(v_h1[r]) / a1) - 0.5 * v_h1[r] + stir_rem;
        }
    } else {
        double i_alp1 = 1.0 / a1;
        double c_alp1 = -Rf_lgammafn(i_alp1) - (i_alp1 * log(a1)) + 0.5 * log(2.0 * pi_const * a1);
        for (int r = s1; r < s2; ++r) {
            acc += ((v_h1[r] - u_h1[r]) / a1) - 0.5 * v_h1[r] + c_alp1;
        }
    }
    return acc;
}

/* Dense symmetric Cholesky PD check & solve: S * x = b (m x m, col-major).
 * Returns 1 if strictly SPD and solve succeeds with finite x, 0 otherwise. */
static int fhl_gamma_spd_solve(const double *A, const double *b, double *x, int m) {
    if (m <= 0) return 1;
    double *L = (double *)malloc((size_t)m * (size_t)m * sizeof(double));
    double *y = (double *)malloc((size_t)m * sizeof(double));
    if (!L || !y) {
        free(L); free(y);
        return 0;
    }
    double max_diag = 0.0;
    for (int j = 0; j < m; ++j) {
        for (int i = j; i < m; ++i) {
            double s = 0.5 * (A[i + (size_t)j * (size_t)m] + A[j + (size_t)i * (size_t)m]);
            if (!R_FINITE(s)) { free(L); free(y); return 0; }
            L[i + (size_t)j * (size_t)m] = s;
            if (i == j && fabs(s) > max_diag) max_diag = fabs(s);
        }
    }
    double tol_diag = 1e-12 * max_diag;
    for (int j = 0; j < m; ++j) {
        double sum = L[j + (size_t)j * (size_t)m];
        for (int k = 0; k < j; ++k) {
            double ljk = L[j + (size_t)k * (size_t)m];
            sum -= ljk * ljk;
        }
        if (!R_FINITE(sum) || sum <= 0.0 || sum <= tol_diag) {
            free(L); free(y);
            return 0;
        }
        double diag = sqrt(sum);
        L[j + (size_t)j * (size_t)m] = diag;
        double inv_diag = 1.0 / diag;
        for (int i = j + 1; i < m; ++i) {
            double s = L[i + (size_t)j * (size_t)m];
            for (int k = 0; k < j; ++k) {
                s -= L[i + (size_t)k * (size_t)m] * L[j + (size_t)k * (size_t)m];
            }
            L[i + (size_t)j * (size_t)m] = s * inv_diag;
        }
    }
    for (int i = 0; i < m; ++i) {
        double s = b[i];
        if (!R_FINITE(s)) { free(L); free(y); return 0; }
        for (int k = 0; k < i; ++k) s -= L[i + (size_t)k * (size_t)m] * y[k];
        y[i] = s / L[i + (size_t)i * (size_t)m];
    }
    for (int i = m - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < m; ++k) s -= L[k + (size_t)i * (size_t)m] * x[k];
        x[i] = s / L[i + (size_t)i * (size_t)m];
        if (!R_FINITE(x[i])) { free(L); free(y); return 0; }
    }
    free(L);
    free(y);
    return 1;
}

/* Shared exact Multi-RE Gamma dispersion score (k2_out) and full nrand x nrand
 * fixed-beta, profile-v Hessian / Jacobian (Hess_out) for dord in {0, 1, 2}.
 * For dord = 0, solves the PQL-type dispersion estimating equation k_{2,d0} = 0
 * using exact -d(k_{2,d0})/dalpha_j under fixed beta and profile v(alpha).
 * Handles arbitrary active masks, blockwise alpha_i, cross-derivatives via
 * fhl_c_xz_gram2_cross, and exact small-alpha cancellation across mixed scales.
 * Returns 1 on success, -1 on OOM. */
static int fhl_gamma_eval_multire_derivs(
    int n, int p, int nrand, const int *q, const int *qcum, int Q,
    const fhl_c_xz_t *xz, const fhl_c_risk_t *risk, const double *di,
    const double *expeta, const double *tM_exp, const double *cla0, const double *as_vec,
    const double *clam0, const double *v_h, const double *u_h,
    const double *alpha_h, const double *iD_diag, const int *act_comp,
    const double *H0, const double *Hinv, const double *invt, const double *zd,
    int dord, int varnonneg,
    double *k2_out, double *Hess_out
) {
    const int k_tot = p + Q;
    const int m = risk->m;
    if (k2_out) {
        for (int i = 0; i < nrand; ++i) k2_out[i] = 0.0;
    }
    if (Hess_out) {
        for (int idx = 0; idx < nrand * nrand; ++idx) Hess_out[idx] = 0.0;
    }
    if (nrand <= 0 || Q <= 0) return 1;

    size_t kk_len = 0, qq_len = 0, qk_len = 0, mp1 = 0, n3 = 0;
    size_t nr_q = 0, nr_n = 0, nr_m = 0, nr_kk = 0;
    if (!fhl_safe_mul_size((size_t)k_tot, (size_t)k_tot, &kk_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)k_tot, &qk_len) ||
        !fhl_safe_add_size((size_t)m, 1, &mp1) ||
        !fhl_safe_mul_size(3, mp1, &n3) ||
        !fhl_safe_mul_size((size_t)nrand, (size_t)Q, &nr_q) ||
        !fhl_safe_mul_size((size_t)nrand, (size_t)n, &nr_n) ||
        !fhl_safe_mul_size((size_t)nrand, (size_t)m, &nr_m) ||
        !fhl_safe_mul_size((size_t)nrand, kk_len, &nr_kk)) {
        return -1;
    }

    int *comp_of_r = NULL, *act_q = NULL, *is_small = NULL;
    double *k2_base_arr = NULL, *k2_base_ex_arr = NULL, *k21_arr = NULL;
    double *c_vh_all = NULL, *dv_all = NULL, *z_dv_all = NULL;
    double *dexpeta_all = NULL, *tM_dexp_all = NULL, *dcla0_all = NULL, *dBi_all = NULL, *dAs_all = NULL;
    double *G1_all = NULL, *Hd_all = NULL;
    double *H0_zz_dv_all = NULL, *R_over_a = NULL, *Q_mat = NULL, *QG1_tmp = NULL;
    double *diag_QQ_all = NULL, *diag_HQ_all = NULL;
    double *G1_j_zz_dvi = NULL, *rhs_total = NULL, *ddv = NULL, *z_ddv = NULL;
    double *dWi_zdv = NULL, *Wi_zddv = NULL, *tM_dWi_zdv = NULL, *tM_Wi_zddv = NULL;
    double *ddcla0 = NULL, *ddBi = NULL, *ddWi = NULL, *ddAs = NULL, *G2_ij = NULL, *risk_wk = NULL;
    double *d0_B_j = NULL, *d0_M_jr = NULL;

    int rc = 1;
    int alloc_ok = 1;
    const char *transN = "N";
    double one = 1.0, zero = 0.0;
    int inc = 1;

    fhl_c_linalg_begin_malloc_scope();

    comp_of_r     = (int *)fhl_g_malloc_chk((size_t)Q, sizeof(int), &alloc_ok);
    act_q         = (int *)fhl_g_malloc_chk((size_t)Q, sizeof(int), &alloc_ok);
    is_small      = (int *)fhl_g_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
    k2_base_arr   = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
    k2_base_ex_arr = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
    k21_arr       = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
    c_vh_all      = (double *)fhl_g_malloc_chk(nr_q, sizeof(double), &alloc_ok);
    dv_all        = (double *)fhl_g_malloc_chk(nr_q, sizeof(double), &alloc_ok);
    z_dv_all      = (double *)fhl_g_malloc_chk(nr_n, sizeof(double), &alloc_ok);
    dexpeta_all   = (double *)fhl_g_malloc_chk(nr_n, sizeof(double), &alloc_ok);
    tM_dexp_all   = (double *)fhl_g_malloc_chk(nr_m > 0 ? nr_m : 1, sizeof(double), &alloc_ok);
    dcla0_all     = (double *)fhl_g_malloc_chk(nr_m > 0 ? nr_m : 1, sizeof(double), &alloc_ok);
    dBi_all       = (double *)fhl_g_malloc_chk(nr_n, sizeof(double), &alloc_ok);
    dAs_all       = (double *)fhl_g_malloc_chk(nr_m > 0 ? nr_m : 1, sizeof(double), &alloc_ok);
    G1_all        = (double *)fhl_g_malloc_chk(nr_kk, sizeof(double), &alloc_ok);
    risk_wk       = (double *)fhl_g_malloc_chk(n3, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }

    double min_act_a = R_PosInf;
    for (int i = 0; i < nrand; ++i) {
        is_small[i] = (act_comp[i] && alpha_h[i] <= 1e-3) ? 1 : 0;
        if (act_comp[i] && alpha_h[i] < min_act_a) min_act_a = alpha_h[i];
        for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
            comp_of_r[r] = i;
            act_q[r] = act_comp[i] ? 1 : 0;
        }
    }
    const int use_small_path = (min_act_a <= 1e-3);

    if (!use_small_path) {
        Hd_all = (double *)fhl_g_malloc_chk(nr_kk, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
    } else {
        H0_zz_dv_all = (double *)fhl_g_malloc_chk(nr_q, sizeof(double), &alloc_ok);
        R_over_a     = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        if (Hess_out) {
            Q_mat       = (double *)fhl_g_malloc_chk(qk_len, sizeof(double), &alloc_ok);
            QG1_tmp     = (double *)fhl_g_malloc_chk(qk_len, sizeof(double), &alloc_ok);
            diag_QQ_all = (double *)fhl_g_malloc_chk(nr_q, sizeof(double), &alloc_ok);
            diag_HQ_all = (double *)fhl_g_malloc_chk(nr_q, sizeof(double), &alloc_ok);
        }
        if (!alloc_ok) { rc = -1; goto cleanup; }

        memset(R_over_a, 0, qq_len * sizeof(double));
        for (int c = 0; c < Q; ++c) {
            if (!act_q[c]) continue;
            double ic = iD_diag[c];
            for (int r = 0; r < Q; ++r) {
                if (!act_q[r]) continue;
                long double rr = 0.0L;
                for (int k = 0; k < k_tot; ++k) {
                    rr += (long double)H0[(p + r) + (size_t)k * (size_t)k_tot] *
                          (long double)(Hinv[k + (size_t)(p + c) * (size_t)k_tot] * ic);
                }
                R_over_a[r + (size_t)c * (size_t)Q] = (double)rr;
            }
        }
        if (Hess_out) {
            for (int k = 0; k < k_tot; ++k) {
                for (int r = 0; r < Q; ++r) {
                    Q_mat[r + (size_t)k * (size_t)Q] =
                        act_q[r] ? ((u_h[r] * iD_diag[r]) * Hinv[(p + r) + (size_t)k * (size_t)k_tot]) : 0.0;
                }
            }
        }
    }

    if (Hess_out) {
        if (dord == 0 && !use_small_path) {
            size_t kq_len = 0;
            if (!fhl_safe_mul_size((size_t)k_tot, (size_t)Q, &kq_len)) { rc = -1; goto cleanup; }
            d0_B_j  = (double *)fhl_g_malloc_chk(kq_len > 0 ? kq_len : 1, sizeof(double), &alloc_ok);
            d0_M_jr = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
            if (!alloc_ok) { rc = -1; goto cleanup; }
        } else if (dord != 0) {
            G1_j_zz_dvi = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
            rhs_total   = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
            ddv         = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
            z_ddv       = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
            dWi_zdv     = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
            Wi_zddv     = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
            tM_dWi_zdv  = (double *)fhl_g_malloc_chk((size_t)(m > 0 ? m : 1), sizeof(double), &alloc_ok);
            tM_Wi_zddv  = (double *)fhl_g_malloc_chk((size_t)(m > 0 ? m : 1), sizeof(double), &alloc_ok);
            ddcla0      = (double *)fhl_g_malloc_chk((size_t)(m > 0 ? m : 1), sizeof(double), &alloc_ok);
            ddBi        = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
            ddWi        = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
            ddAs        = (double *)fhl_g_malloc_chk((size_t)(m > 0 ? m : 1), sizeof(double), &alloc_ok);
            G2_ij       = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
            if (!alloc_ok) { rc = -1; goto cleanup; }
        }
    }

    /* Pass 1: per-component 1st derivatives & dispersion score k2 */
    for (int i = 0; i < nrand; ++i) {
        if (!act_comp[i]) continue;
        const int s1 = qcum[i];
        const int s2 = qcum[i + 1];
        const double alpha_i = alpha_h[i];
        const double ial  = 1.0 / alpha_i;
        const double ial2 = ial * ial;

        double k2_b_ex = 0.0, k2_b = 0.0, k21_r = 0.0, k21_v = 0.0;
        fhl_gamma_small_alpha_k2_k21(q[i], alpha_i, varnonneg, v_h + s1, &k2_b_ex, &k2_b, &k21_r, &k21_v);
        k2_base_ex_arr[i] = k2_b_ex;
        k2_base_arr[i]    = k2_b;
        k21_arr[i]        = k21_v;
        (void)k21_r;

        if (dord == 0 && !Hess_out) {
            if (k2_out) {
                if (use_small_path && is_small[i]) {
                    double r_sum_ial = 0.0;
                    for (int r = s1; r < s2; ++r) r_sum_ial += R_over_a[r + (size_t)r * (size_t)Q];
                    k2_out[i] = -(ial2 * k2_b_ex) - 0.5 * r_sum_ial;
                } else {
                    double sum_hu = 0.0;
                    for (int r = s1; r < s2; ++r) {
                        sum_hu += (u_h[r] * ial2) * Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                    }
                    k2_out[i] = -(ial2 * k2_b) + 0.5 * sum_hu;
                }
            }
            continue;
        }

        double *c_vh_i    = c_vh_all + (size_t)i * (size_t)Q;
        double *dv_i      = dv_all + (size_t)i * (size_t)Q;
        double *z_dv_i    = z_dv_all + (size_t)i * (size_t)n;
        double *dexpeta_i = dexpeta_all + (size_t)i * (size_t)n;
        double *tM_dexp_i = tM_dexp_all + (size_t)i * (size_t)m;
        double *dcla0_i   = dcla0_all + (size_t)i * (size_t)m;
        double *dBi_i     = dBi_all + (size_t)i * (size_t)n;
        double *dAs_i     = dAs_all + (size_t)i * (size_t)m;
        double *G1_i      = G1_all + (size_t)i * kk_len;

        memset(c_vh_i, 0, (size_t)Q * sizeof(double));
        for (int r = s1; r < s2; ++r) {
            c_vh_i[r] = ial2 * expm1(v_h[r]);
        }
        F77_CALL(dgemv)(transN, &Q, &Q, &one, invt, &Q, c_vh_i, &inc, &zero, dv_i, &inc FCONE);
        fhl_c_z_mv(xz, dv_i, z_dv_i);
        for (int j = 0; j < n; ++j) dexpeta_i[j] = expeta[j] * z_dv_i[j];

        fhl_c_risk_backward(risk, dexpeta_i, tM_dexp_i, risk_wk);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        for (int r = 0; r < m; ++r) {
            dcla0_i[r] = -(di[r] / (tM_exp[r] * tM_exp[r])) * tM_dexp_i[r];
            dAs_i[r]   = (2.0 * (cla0[r] * dcla0_i[r])) / di[r];
        }
        fhl_c_risk_forward(risk, dcla0_i, dBi_i, risk_wk);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

        fhl_c_xz_gram1(xz, 1, risk, expeta, clam0, as_vec, dexpeta_i, dBi_i, dAs_i, G1_i);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

        double dk3 = 0.0;
        if (dord == 2) {
            long double sum_cor1 = 0.0L;
            for (int r = s1; r < s2; ++r) {
                double base_r = 1.0 + alpha_i * zd[r];
                sum_cor1 += 1.0 / (base_r * base_r);
            }
            dk3 = (double)sum_cor1 / 12.0;
        }

        if (!use_small_path) {
            double *Hd_i = Hd_all + (size_t)i * kk_len;
            memcpy(Hd_i, G1_i, kk_len * sizeof(double));
            for (int r = 0; r < Q; ++r) {
                if (!act_q[r]) continue;
                double d1 = iD_diag[r] * (u_h[r] * dv_i[r]);
                if (comp_of_r[r] == i) d1 -= ial2 * u_h[r];
                Hd_i[(p + r) + (size_t)(p + r) * (size_t)k_tot] += d1;
            }
            if (k2_out) {
                if (dord == 0) {
                    double sum_hu = 0.0;
                    for (int r = s1; r < s2; ++r) {
                        sum_hu += (u_h[r] * ial2) * Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                    }
                    k2_out[i] = -(ial2 * k2_b) + 0.5 * sum_hu;
                } else {
                    double dk2 = -0.5 * fhl_c_trace_ab(Hinv, Hd_i, k_tot);
                    k2_out[i] = -(ial2 * k2_b) + dk2 + dk3;
                }
            }
        } else {
            double *H0_zz_dv_i = H0_zz_dv_all + (size_t)i * (size_t)Q;
            for (int r = 0; r < Q; ++r) {
                if (!act_q[r]) { H0_zz_dv_i[r] = 0.0; continue; }
                long double hdv = 0.0L;
                for (int c = 0; c < Q; ++c) {
                    hdv += (long double)H0[(p + r) + (size_t)(p + c) * (size_t)k_tot] * (long double)dv_i[c];
                }
                H0_zz_dv_i[r] = (double)hdv;
            }
            if (Hess_out) {
                double *diag_QQ_i = diag_QQ_all + (size_t)i * (size_t)Q;
                double *diag_HQ_i = diag_HQ_all + (size_t)i * (size_t)Q;
                if (Q > 0 && k_tot > 0) {
                    F77_CALL(dgemm)(transN, transN, &Q, &k_tot, &k_tot,
                                    &one, Q_mat, &Q, G1_i, &k_tot,
                                    &zero, QG1_tmp, &Q FCONE FCONE);
                }
                for (int r = 0; r < Q; ++r) {
                    if (!act_q[r]) { diag_QQ_i[r] = 0.0; diag_HQ_i[r] = 0.0; continue; }
                    long double s_qq = 0.0L, s_hq = 0.0L;
                    for (int k = 0; k < k_tot; ++k) {
                        double qg = QG1_tmp[r + (size_t)k * (size_t)Q];
                        s_qq += (long double)qg * (long double)Q_mat[r + (size_t)k * (size_t)Q];
                        s_hq += (long double)qg * (long double)Hinv[k + (size_t)(p + r) * (size_t)k_tot];
                    }
                    diag_QQ_i[r] = (double)s_qq;
                    diag_HQ_i[r] = (double)s_hq;
                }
            }
            if (k2_out) {
                double lead_term = 0.0;
                if (is_small[i]) {
                    double r_sum_ial = 0.0;
                    for (int r = s1; r < s2; ++r) r_sum_ial += R_over_a[r + (size_t)r * (size_t)Q];
                    lead_term = -(ial2 * k2_b_ex) - 0.5 * r_sum_ial;
                } else {
                    double sum_hu = 0.0;
                    for (int r = s1; r < s2; ++r) {
                        sum_hu += (u_h[r] * ial2) * Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                    }
                    lead_term = -(ial2 * k2_b) + 0.5 * sum_hu;
                }
                if (dord == 0) {
                    k2_out[i] = lead_term;
                } else {
                    double dv_r_adj = 0.0;
                    for (int r = 0; r < Q; ++r) {
                        if (!act_q[r]) continue;
                        int cr = comp_of_r[r];
                        double one_m_r = is_small[cr]
                            ? (1.0 - alpha_h[cr] * R_over_a[r + (size_t)r * (size_t)Q])
                            : ((u_h[r] * iD_diag[r]) * Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot]);
                        dv_r_adj += one_m_r * dv_i[r];
                    }
                    k2_out[i] = lead_term - 0.5 * fhl_c_trace_ab(Hinv, G1_i, k_tot) - 0.5 * dv_r_adj + dk3;
                }
            }
        }
    }

    if (!Hess_out) goto cleanup;

    /* Pass 2a (dord == 0): exact Jacobian J_{ij} = -d(k2_d0[i])/dalpha_j (generally asymmetric for nrand > 1) */
    if (dord == 0) {
        if (!use_small_path) {
            for (int j = 0; j < nrand; ++j) {
                if (!act_comp[j]) continue;
                const double *dv_j = dv_all + (size_t)j * (size_t)Q;
                const double *Hd_j = Hd_all + (size_t)j * kk_len;
                if (Q > 0 && k_tot > 0) {
                    F77_CALL(dgemm)(transN, transN, &k_tot, &Q, &k_tot,
                                    &one, Hd_j, &k_tot, Hinv + (size_t)p * (size_t)k_tot, &k_tot,
                                    &zero, d0_B_j, &k_tot FCONE FCONE);
                }
                for (int r = 0; r < Q; ++r) {
                    if (!act_q[r]) { d0_M_jr[r] = 0.0; continue; }
                    long double mr = 0.0L;
                    for (int k = 0; k < k_tot; ++k) {
                        mr += (long double)Hinv[k + (size_t)(p + r) * (size_t)k_tot] * (long double)d0_B_j[k + (size_t)r * (size_t)k_tot];
                    }
                    d0_M_jr[r] = (double)mr;
                }
                for (int i = 0; i < nrand; ++i) {
                    if (!act_comp[i]) continue;
                    const int s1 = qcum[i];
                    const int s2 = qcum[i + 1];
                    const double alpha_i = alpha_h[i];
                    const double ial2_i  = 1.0 / (alpha_i * alpha_i);
                    const double ial3_i  = ial2_i / alpha_i;
                    const double *c_vh_i = c_vh_all + (size_t)i * (size_t)Q;

                    long double cvh_dv_ld = 0.0L, tr_hk = 0.0L, tr_prod = 0.0L;
                    for (int r = s1; r < s2; ++r) {
                        double h_rr = Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                        cvh_dv_ld += (long double)c_vh_i[r] * (long double)dv_j[r];
                        tr_hk     += (long double)h_rr * (long double)(((i == j) ? (2.0 * ial3_i * u_h[r]) : 0.0) - ial2_i * u_h[r] * dv_j[r]);
                        tr_prod   += (long double)(ial2_i * u_h[r]) * (long double)d0_M_jr[r];
                    }
                    Hess_out[i + (size_t)j * (size_t)nrand] =
                        ((i == j) ? (-k21_arr[i] + (varnonneg ? ial2_i : 0.0)) : 0.0) -
                        (double)cvh_dv_ld + 0.5 * (double)tr_hk + 0.5 * (double)tr_prod;
                }
            }
        } else {
            for (int j = 0; j < nrand; ++j) {
                if (!act_comp[j]) continue;
                const double *dv_j      = dv_all + (size_t)j * (size_t)Q;
                const double *h0_j      = H0_zz_dv_all + (size_t)j * (size_t)Q;
                const double *diag_QQ_j = diag_QQ_all + (size_t)j * (size_t)Q;
                const double *diag_HQ_j = diag_HQ_all + (size_t)j * (size_t)Q;

                for (int i = 0; i < nrand; ++i) {
                    if (!act_comp[i]) continue;
                    const int s1 = qcum[i];
                    const int s2 = qcum[i + 1];
                    const double alpha_i = alpha_h[i];
                    const double ial2_i  = 1.0 / (alpha_i * alpha_i);
                    const double *c_vh_i = c_vh_all + (size_t)i * (size_t)Q;

                    double hess_val = 0.0;
                    if (i == j) {
                        if (is_small[i]) {
                            double a2 = alpha_i * alpha_i, a3 = a2 * alpha_i, a5 = a3 * a2;
                            hess_val += -(double)q[i] * (alpha_i / 60.0 - a3 / 63.0 + a5 / 40.0);
                        } else {
                            hess_val += -k21_arr[i];
                        }
                        if (varnonneg) hess_val += ial2_i;
                    }

                    for (int r = s1; r < s2; ++r) {
                        double ik = iD_diag[r];
                        if (is_small[i]) {
                            double ra = R_over_a[r + (size_t)r * (size_t)Q];
                            double one_m_r = 1.0 - alpha_i * ra;
                            hess_val += 0.5 * (diag_QQ_j[r] / u_h[r]);
                            hess_val += (expm1(v_h[r]) * ik) * (h0_j[r] / u_h[r]) - 0.5 * ra * one_m_r * dv_j[r];
                            if (i == j) {
                                hess_val += fhl_gamma_pair1_over_a3(v_h[r], alpha_i) / u_h[r] - 0.5 * ra * ra;
                            }
                        } else {
                            double h_rr = Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                            hess_val += 0.5 * (diag_HQ_j[r] / alpha_i);
                            hess_val += -c_vh_i[r] * dv_j[r];
                            hess_val += ((i == j) ? ((ik * ik * ik) * u_h[r] * h_rr) : 0.0) -
                                        0.5 * (ik * ik) * u_h[r] * h_rr * dv_j[r];
                            hess_val += 0.5 * (ik * ik * ik) * u_h[r] * u_h[r] * h_rr * h_rr *
                                        (((i == j) ? -ik : 0.0) + dv_j[r]);
                        }
                        for (int c = 0; c < Q; ++c) {
                            if (c == r || !act_q[c]) continue;
                            int cc = comp_of_r[c];
                            double fac_jc = ((cc == j) ? -1.0 : 0.0) + alpha_h[cc] * dv_j[c];
                            hess_val += 0.5 * R_over_a[c + (size_t)r * (size_t)Q] *
                                        R_over_a[r + (size_t)c * (size_t)Q] * fac_jc;
                        }
                    }
                    Hess_out[i + (size_t)j * (size_t)nrand] = hess_val;
                }
            }
        }
        goto cleanup;
    }

    /* Pass 2b (dord in {1, 2}): upper triangle (i <= j) 2nd derivatives & full symmetric Hessian */
    for (int i = 0; i < nrand; ++i) {
        if (!act_comp[i]) continue;
        const int s1 = qcum[i];
        const int s2 = qcum[i + 1];
        const double alpha_i = alpha_h[i];
        const double ial2_i  = 1.0 / (alpha_i * alpha_i);

        const double *c_vh_i    = c_vh_all + (size_t)i * (size_t)Q;
        const double *dv_i      = dv_all + (size_t)i * (size_t)Q;
        const double *z_dv_i    = z_dv_all + (size_t)i * (size_t)n;
        const double *dexpeta_i = dexpeta_all + (size_t)i * (size_t)n;
        const double *tM_dexp_i = tM_dexp_all + (size_t)i * (size_t)m;
        const double *dcla0_i   = dcla0_all + (size_t)i * (size_t)m;
        const double *dBi_i     = dBi_all + (size_t)i * (size_t)n;
        const double *dAs_i     = dAs_all + (size_t)i * (size_t)m;
        const double *G1_i      = G1_all + (size_t)i * kk_len;

        for (int j = i; j < nrand; ++j) {
            if (!act_comp[j]) continue;
            const double *dv_j      = dv_all + (size_t)j * (size_t)Q;
            const double *dexpeta_j = dexpeta_all + (size_t)j * (size_t)n;
            const double *dcla0_j   = dcla0_all + (size_t)j * (size_t)m;
            const double *dBi_j     = dBi_all + (size_t)j * (size_t)n;
            const double *dAs_j     = dAs_all + (size_t)j * (size_t)m;
            const double *G1_j      = G1_all + (size_t)j * kk_len;

            for (int r = 0; r < Q; ++r) {
                if (!act_q[r]) { G1_j_zz_dvi[r] = 0.0; rhs_total[r] = 0.0; continue; }
                long double gv = 0.0L;
                for (int c = 0; c < Q; ++c) {
                    gv += (long double)G1_j[(p + r) + (size_t)(p + c) * (size_t)k_tot] * (long double)dv_i[c];
                }
                G1_j_zz_dvi[r] = (double)gv;

                int cr = comp_of_r[r];
                double ik = iD_diag[r];
                double w_r = 0.0;
                if (!is_small[cr]) {
                    w_r = -(ik * u_h[r] * dv_j[r]) * dv_i[r];
                    if (cr == j) w_r += (ik * ik) * u_h[r] * dv_i[r];
                    if (cr == i) {
                        if (i == j) w_r += -2.0 * (ik * ik * ik) * expm1(v_h[r]) + (ik * ik) * u_h[r] * dv_j[r];
                        else        w_r += (ik * ik) * u_h[r] * dv_j[r];
                    }
                } else {
                    const double *h0_i = H0_zz_dv_all + (size_t)i * (size_t)Q;
                    const double *h0_j = H0_zz_dv_all + (size_t)j * (size_t)Q;
                    if (cr == i && i == j) {
                        w_r = -(2.0 * ik) * h0_i[r] - (u_h[r] * ik) * dv_i[r] * dv_i[r];
                    } else if (cr == i) {
                        w_r = -(h0_j[r] * ik) / u_h[r] + h0_i[r] * dv_j[r];
                    } else if (cr == j) {
                        w_r = -(h0_i[r] * ik) / u_h[r] + h0_j[r] * dv_i[r];
                    } else {
                        w_r = h0_i[r] * dv_j[r];
                    }
                }
                rhs_total[r] = -G1_j_zz_dvi[r] + w_r;
            }
            F77_CALL(dgemv)(transN, &Q, &Q, &one, invt, &Q, rhs_total, &inc, &zero, ddv, &inc FCONE);

            fhl_c_z_mv(xz, ddv, z_ddv);
            for (int row = 0; row < n; ++row) {
                dWi_zdv[row] = dexpeta_j[row] * z_dv_i[row];
                Wi_zddv[row] = expeta[row] * z_ddv[row];
                ddWi[row]    = dWi_zdv[row] + Wi_zddv[row];
            }
            fhl_c_risk_backward(risk, dWi_zdv, tM_dWi_zdv, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            fhl_c_risk_backward(risk, Wi_zddv, tM_Wi_zddv, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
            for (int r = 0; r < m; ++r) {
                ddcla0[r] = -dAs_j[r] * tM_dexp_i[r] - as_vec[r] * tM_dWi_zdv[r] - as_vec[r] * tM_Wi_zddv[r];
                ddAs[r]   = (2.0 * (dcla0_i[r] * dcla0_j[r]) + 2.0 * (cla0[r] * ddcla0[r])) / di[r];
            }
            fhl_c_risk_forward(risk, ddcla0, ddBi, risk_wk);
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

            if (i == j) {
                fhl_c_xz_gram2(xz, 1, risk, expeta, clam0, as_vec,
                               dexpeta_i, dBi_i, dAs_i, ddWi, ddBi, ddAs, G2_ij);
            } else {
                fhl_c_xz_gram2_cross(xz, 1, risk, expeta, clam0, as_vec,
                                     dexpeta_i, dBi_i, dAs_i,
                                     dexpeta_j, dBi_j, dAs_j,
                                     ddWi, ddBi, ddAs, G2_ij);
            }
            if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

            double kcor = 0.0;
            if (dord == 2 && i == j) {
                long double sum_cor22 = 0.0L;
                for (int r = s1; r < s2; ++r) {
                    double base_r = 1.0 + alpha_i * zd[r];
                    sum_cor22 += zd[r] / (base_r * base_r * base_r);
                }
                kcor = (double)sum_cor22 / 6.0;
            }

            double hess_val = 0.0;
            if (!use_small_path) {
                const double *Hd_i = Hd_all + (size_t)i * kk_len;
                const double *Hd_j = Hd_all + (size_t)j * kk_len;
                for (int r = 0; r < Q; ++r) {
                    if (!act_q[r]) continue;
                    int cr = comp_of_r[r];
                    double ik = iD_diag[r];
                    double uad2_r = u_h[r] * (dv_i[r] * dv_j[r] + ddv[r]);
                    double d2 = ik * uad2_r;
                    if (cr == i) d2 -= (ik * ik) * u_h[r] * dv_j[r];
                    if (cr == j) d2 -= (ik * ik) * u_h[r] * dv_i[r];
                    if (i == j && cr == i) d2 += 2.0 * (ik * ik * ik) * u_h[r];
                    G2_ij[(p + r) + (size_t)(p + r) * (size_t)k_tot] += d2;
                }
                long double cvh_dv_ld = 0.0L;
                for (int r = s1; r < s2; ++r) cvh_dv_ld += (long double)c_vh_i[r] * (long double)dv_j[r];
                double tr_hg2 = fhl_c_trace_ab(Hinv, G2_ij, k_tot);
                double tr_hd  = fhl_c_trace_hinv_hd_prod(Hinv, Hd_i, Hd_j, k_tot);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
                hess_val = ((i == j) ? -k21_arr[i] : 0.0) - (double)cvh_dv_ld +
                           0.5 * (tr_hg2 - tr_hd) + kcor +
                           ((varnonneg && i == j) ? ial2_i : 0.0);
            } else {
                const double *h0_i      = H0_zz_dv_all + (size_t)i * (size_t)Q;
                const double *h0_j      = H0_zz_dv_all + (size_t)j * (size_t)Q;
                const double *diag_QQ_i = diag_QQ_all + (size_t)i * (size_t)Q;
                const double *diag_QQ_j = diag_QQ_all + (size_t)j * (size_t)Q;
                const double *diag_HQ_i = diag_HQ_all + (size_t)i * (size_t)Q;
                const double *diag_HQ_j = diag_HQ_all + (size_t)j * (size_t)Q;

                double tr_hg2 = fhl_c_trace_ab(Hinv, G2_ij, k_tot);
                double tr_s2  = fhl_c_trace_hinv_hd_prod(Hinv, G1_i, G1_j, k_tot);
                if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

                double cross_G1_D1 = 0.0;
                double diag_sum = 0.0;
                if (i == j && is_small[i]) {
                    double a2 = alpha_i * alpha_i, a3 = a2 * alpha_i, a5 = a3 * a2;
                    diag_sum += -(double)q[i] * (alpha_i / 60.0 - a3 / 63.0 + a5 / 40.0);
                } else if (i == j && !is_small[i]) {
                    diag_sum += -k21_arr[i];
                }

                for (int r = 0; r < Q; ++r) {
                    if (!act_q[r]) continue;
                    int cr = comp_of_r[r];
                    double ik = iD_diag[r];

                    double cj = (cr == j && is_small[cr])
                        ? (diag_QQ_i[r] / u_h[r]) * (-1.0 + alpha_h[cr] * dv_j[r])
                        : diag_HQ_i[r] * ((cr == j ? -ik : 0.0) + dv_j[r]);
                    double ci = (cr == i && is_small[cr])
                        ? (diag_QQ_j[r] / u_h[r]) * (-1.0 + alpha_h[cr] * dv_i[r])
                        : diag_HQ_j[r] * ((cr == i ? -ik : 0.0) + dv_i[r]);
                    cross_G1_D1 += -0.5 * (cj + ci);

                    if (is_small[cr]) {
                        double ra = R_over_a[r + (size_t)r * (size_t)Q];
                        double r_diag = alpha_h[cr] * ra;
                        double one_m_r = 1.0 - r_diag;
                        diag_sum += 0.5 * one_m_r * ddv[r] + 0.5 * r_diag * one_m_r * dv_i[r] * dv_j[r];
                        if (cr == i && i == j) {
                            diag_sum += fhl_gamma_pair1_over_a3(v_h[r], alpha_i) / u_h[r] +
                                        (expm1(v_h[r]) * ik) * (h0_i[r] / u_h[r]) -
                                        0.5 * ra * ra - ra * one_m_r * dv_i[r];
                        } else if (cr == i) {
                            diag_sum += (expm1(v_h[r]) * ik) * (h0_j[r] / u_h[r]) -
                                        0.5 * ra * one_m_r * dv_j[r];
                        } else if (cr == j) {
                            diag_sum += -0.5 * ra * one_m_r * dv_i[r];
                        }
                    } else {
                        double h_rr = Hinv[(p + r) + (size_t)(p + r) * (size_t)k_tot];
                        double d_ir = ((cr == i) ? -(ik * ik) * u_h[r] : 0.0) + ik * u_h[r] * dv_i[r];
                        double d_jr = ((cr == j) ? -(ik * ik) * u_h[r] : 0.0) + ik * u_h[r] * dv_j[r];
                        double uad2_r = u_h[r] * (dv_i[r] * dv_j[r] + ddv[r]);
                        double d_ijr = ik * uad2_r - ((cr == i) ? (ik * ik) * u_h[r] * dv_j[r] : 0.0)
                                                   - ((cr == j) ? (ik * ik) * u_h[r] * dv_i[r] : 0.0)
                                                   + ((i == j && cr == i) ? 2.0 * (ik * ik * ik) * u_h[r] : 0.0);
                        diag_sum += 0.5 * h_rr * d_ijr - 0.5 * h_rr * h_rr * d_ir * d_jr -
                                    ((cr == i) ? (c_vh_i[r] * dv_j[r]) : 0.0);
                    }
                }

                double off_D1sq = 0.0;
                for (int c = 0; c < Q; ++c) {
                    if (!act_q[c]) continue;
                    int cc = comp_of_r[c];
                    double fac_ic = ((cc == i) ? -1.0 : 0.0) + alpha_h[cc] * dv_i[c];
                    for (int r = 0; r < Q; ++r) {
                        if (r == c || !act_q[r]) continue;
                        int cr = comp_of_r[r];
                        double fac_jr = ((cr == j) ? -1.0 : 0.0) + alpha_h[cr] * dv_j[r];
                        off_D1sq += -0.5 * R_over_a[r + (size_t)c * (size_t)Q] *
                                    R_over_a[c + (size_t)r * (size_t)Q] * fac_jr * fac_ic;
                    }
                }
                hess_val = 0.5 * (tr_hg2 - tr_s2) + cross_G1_D1 + diag_sum + off_D1sq + kcor +
                           ((varnonneg && i == j) ? ial2_i : 0.0);
            }
            Hess_out[i + (size_t)j * (size_t)nrand] = hess_val;
            Hess_out[j + (size_t)i * (size_t)nrand] = hess_val;
        }
    }

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(comp_of_r); free(act_q); free(is_small);
    free(k2_base_arr); free(k2_base_ex_arr); free(k21_arr);
    free(c_vh_all); free(dv_all); free(z_dv_all);
    free(dexpeta_all); free(tM_dexp_all); free(dcla0_all); free(dBi_all); free(dAs_all);
    free(G1_all); free(Hd_all);
    free(H0_zz_dv_all); free(R_over_a); free(Q_mat); free(QG1_tmp);
    free(diag_QQ_all); free(diag_HQ_all);
    free(G1_j_zz_dvi); free(rhs_total); free(ddv); free(z_ddv);
    free(dWi_zdv); free(Wi_zddv); free(tM_dWi_zdv); free(tM_Wi_zddv);
    free(ddcla0); free(ddBi); free(ddWi); free(ddAs); free(G2_ij); free(risk_wk);
    free(d0_B_j); free(d0_M_jr);
    return rc;
}

/* Evaluate u-scale inverse Hessian Hinv_u = Hu^{-1} at the converged solution */
static int fhl_c_gamma_finalize_u_core(
    int n, int p, int nrand, const int *qcum, int Q,
    const double *x, const fhl_c_xz_t *xz,
    const fhl_c_risk_t *risk,
    const double *beta_h, const double *v_h, const double *alpha_h,
    const double *clam0, const double *as_vec, const double *G_mat_curr,
    double *Hinv_u_out
) {
    if (n < 0 || p < 0 || Q < 0 || nrand <= 0 || p > INT_MAX - Q) return -1;
    const int k_tot = p + Q;
    size_t kk_len = 0, qq_len = 0;
    if (!fhl_safe_mul_size((size_t)k_tot, (size_t)k_tot, &kk_len) ||
        !fhl_safe_mul_size((size_t)Q, (size_t)Q, &qq_len)) {
        return -1;
    }
    const char *transN = "N";
    double one = 1.0, zero = 0.0;
    int inc = 1;

    int alloc_ok = 1;
    double *ialp_vec = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    double *xb      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    double *exp_xb  = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    double *mu1     = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    double *zmu     = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    double *G_mat1  = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    double *Hu      = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    double *Zt_WB_Z = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    if (!alloc_ok) {
        free(ialp_vec); free(xb); free(exp_xb); free(mu1); free(zmu); free(G_mat1); free(Hu); free(Zt_WB_Z);
        return -1;
    }

    double min_a = R_PosInf;
    int any_inact = 0;
    for (int i = 0; i < nrand; ++i) {
        double a_i = alpha_h[i];
        if (!R_FINITE(a_i) || a_i <= 0.0) {
            any_inact = 1;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) ialp_vec[r] = 0.0;
        } else {
            if (a_i < min_a) min_a = a_i;
            double ialp_i = 1.0 / a_i;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) ialp_vec[r] = ialp_i;
        }
    }
    const int use_scaled_u = (any_inact || (min_a <= 1e-3));

    int rc = 1;
    fhl_c_linalg_begin_malloc_scope();

    if (p > 0) {
        F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h, &inc, &zero, xb, &inc FCONE);
    } else {
        memset(xb, 0, (size_t)n * sizeof(double));
    }
    for (int i = 0; i < n; ++i) {
        exp_xb[i] = exp(xb[i]);
        mu1[i] = exp_xb[i] * clam0[i];
    }
    fhl_c_z_tmv(xz, mu1, zmu);

    fhl_c_xz_gram0(xz, 1, risk, exp_xb, clam0, as_vec, G_mat1);
    if (fhl_c_linalg_had_oom()) {
        rc = -1;
        goto cleanup;
    }

    /* Assemble Hu:
     * Top-left p x p from G_mat_curr
     * Top-right p x Q and bottom-left Q x p from G_mat1
     * Bottom-right Q x Q: U1 - U2 where U2 = crossprod(z, mu1 * z) - G_mat1_zz */
    for (int c = 0; c < p; ++c) {
        memcpy(Hu + (size_t)c * (size_t)k_tot, G_mat_curr + (size_t)c * (size_t)k_tot, (size_t)p * sizeof(double));
        memcpy(Hu + (size_t)c * (size_t)k_tot + p, G_mat1 + (size_t)c * (size_t)k_tot + p, (size_t)Q * sizeof(double));
    }
    for (int c = 0; c < Q; ++c) {
        memcpy(Hu + (size_t)(p + c) * (size_t)k_tot, G_mat1 + (size_t)(p + c) * (size_t)k_tot, (size_t)p * sizeof(double));
    }

    /* Compute Zt_WB_Z = crossprod(z, (W0 * Bi) * z) = t(z) %*% diag(mu1) %*% z */
    fhl_c_z_diag_quad(xz, mu1, Zt_WB_Z);

    if (use_scaled_u) {
        double *d_u_diag = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
        if (!alloc_ok || !d_u_diag) {
            free(d_u_diag);
            rc = -1;
            goto cleanup;
        }
        for (int i = 0; i < nrand; ++i) {
            double a_i = alpha_h[i];
            int act_i = (R_FINITE(a_i) && a_i > 0.0);
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                d_u_diag[r] = act_i ? (a_i * exp(v_h[r])) : 0.0;
            }
        }
        for (int c = 0; c < Q; ++c) {
            for (int r = 0; r < Q; ++r) {
                double u2_rc = Zt_WB_Z[r + (size_t)c * (size_t)Q] - G_mat1[(p + r) + (size_t)(p + c) * (size_t)k_tot];
                double u1_base = (r == c) ? (zmu[r] / exp(v_h[r])) : 0.0;
                Hu[(p + r) + (size_t)(p + c) * (size_t)k_tot] = u1_base - u2_rc;
            }
        }
        free(Zt_WB_Z);
        Zt_WB_Z = NULL;
        rc = fhl_c_schur_scaled_eval(p, Q, Hu, d_u_diag, NULL, 3.1415926535900001,
                                     Hinv_u_out, NULL, NULL, NULL, NULL, NULL);
        free(d_u_diag);
        if (fhl_c_linalg_had_oom()) {
            rc = -1;
            goto cleanup;
        }
    } else {
        for (int c = 0; c < Q; ++c) {
            for (int r = 0; r < Q; ++r) {
                double u2_rc = Zt_WB_Z[r + (size_t)c * (size_t)Q] - G_mat1[(p + r) + (size_t)(p + c) * (size_t)k_tot];
                double u1_rc = (r == c) ? ((ialp_vec[r] + zmu[r]) / exp(v_h[r])) : 0.0;
                Hu[(p + r) + (size_t)(p + c) * (size_t)k_tot] = u1_rc - u2_rc;
            }
        }
        free(Zt_WB_Z);
        Zt_WB_Z = NULL;

        rc = (p > 0 && Q > 0)
            ? fhl_c_lognormal_inverse(Hu, p, Q, NULL, Hinv_u_out, NULL)
            : fhl_c_solve_inv(Hu, Hinv_u_out, k_tot);
        if (fhl_c_linalg_had_oom()) {
            rc = -1;
            goto cleanup;
        }
        if (!rc) {
            /* Post-fit u-scale Hessian Hu may be nonsingular indefinite; invert via general LU */
            rc = fhl_c_solve_gesv_inv(Hu, Hinv_u_out, k_tot);
            if (fhl_c_linalg_had_oom()) {
                rc = -1;
                goto cleanup;
            }
        }
    }

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(Zt_WB_Z);
    free(ialp_vec); free(xb); free(exp_xb); free(mu1); free(zmu); free(G_mat1); free(Hu);
    return rc;
}

/* Single Newton iteration for Gamma Cox frailty models */
static int fhl_c_gamma_pg_core(
    int n, int p, int nrand, const int *q, const int *qcum, int Q,
    const double *x, const fhl_c_xz_t *xz,
    const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h0, const double *v_h0, const double *alpha_h0_in,
    int mord, int dord, int varfixed, int varnonneg,
    double *alpha_h0_out, double *beta_h, double *v_h, double *alpha_h,
    double *dft, double *Hinv, double *clam0, double *H,
    double *se_beta_h, double *U_diag_out, double *H0, double *u_h_out,
    double *as_vec_out, double *k2_out, double *k22_out, int *capped_out,
    int *hess_pd_out,
    double *expeta_cache, double *tM_exp_cache, double *cla0_cache, int *have_cache
) {
    for (int i = 0; i < nrand; ++i) {
        if (k2_out) k2_out[i] = 0.0;
        if (k22_out) k22_out[i] = 0.0;
        if (capped_out) capped_out[i] = 0;
    }
    if (hess_pd_out) *hess_pd_out = 1;
    const int m = risk->m;
    if (n < 0 || p < 0 || Q < 0 || m < 0 || p > INT_MAX - Q) return -1;
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

    int rc = 1;
    int alloc_ok = 1;
    double *iD_diag = NULL, *xb = NULL, *zv = NULL, *expeta = NULL, *tM_exp = NULL;
    double *cla0 = NULL, *as_vec_buf = NULL, *as_vec = NULL, *res_v = NULL;
    double *risk_wk = NULL, *iu_h0 = NULL, *U_diag0 = NULL;
    double *T_zz = NULL, *dinv = NULL;
    double *neg_dinv = NULL, *dv_db_k = NULL, *xz_k = NULL, *dmu0_k = NULL;
    double *tM_dmu0 = NULL, *dcla0b_k = NULL, *dc_k = NULL, *da_k = NULL;
    double *step = NULL, *invt = NULL, *term = NULL;
    double *zd = NULL;
    double *d_diag0 = NULL, *d_diag1 = NULL, *v_h0_eff = NULL;
    int *act_comp = NULL, *act_q_flag = NULL;

    fhl_c_linalg_begin_malloc_scope();

    act_comp   = (int *)fhl_g_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
    act_q_flag = (int *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(int), &alloc_ok);
    d_diag0    = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    d_diag1    = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    v_h0_eff   = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    iD_diag = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    xb      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    zv      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    expeta  = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    tM_exp  = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    cla0    = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    if (as_vec_out != NULL) {
        as_vec = as_vec_out;
    } else {
        as_vec_buf = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        as_vec = as_vec_buf;
    }
    res_v   = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    risk_wk = (double *)fhl_g_malloc_chk(n3, sizeof(double), &alloc_ok);
    iu_h0   = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    U_diag0 = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
    if (!alloc_ok) {
        rc = -1;
        goto cleanup;
    }

    /* 1. Active component mask & alpha floor:
     *    - varfixed && alpha_h0_in[i] <= 0 : strictly inactive Cox component (v = 0)
     *    - varfixed && alpha_h0_in[i] >  0 : keep positive alpha down to 1e-30 (no 1e-6 clamp)
     *    - !varfixed                       : probe floor 1e-6 for optimization */
    double min_act_a0 = R_PosInf;
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
                v_h0_eff[r] = 0.0;
                iu_h0[r] = 1.0;
                U_diag0[r] = 0.0;
                d_diag0[r] = 0.0;
            }
        } else {
            act_comp[i] = 1;
            if (!varfixed && a0 < 9.9999999999999995e-07) a0 = 9.9999999999999995e-07;
            else if (a0 < 1e-30) a0 = 1e-30;
            if (a0 < min_act_a0) min_act_a0 = a0;
            alpha_h0_out[i] = a0;
            alpha_h[i] = a0;
            double ia0 = 1.0 / a0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                act_q_flag[r] = 1;
                iD_diag[r] = ia0;
                v_h0_eff[r] = v_h0[r];
                iu_h0[r] = exp(v_h0_eff[r]);
                U_diag0[r] = ia0 * iu_h0[r];
                d_diag0[r] = a0 / iu_h0[r];
            }
        }
    }
    const int use_scaled = (any_inact || (min_act_a0 <= 1e-3));

    /* 2. muh = x %*% beta_h0 + z %*% v_h0_eff; expeta = exp(muh) (reuse post-step cache on iter >= 2) */
    if (have_cache != NULL && *have_cache) {
        memcpy(expeta, expeta_cache, (size_t)n * sizeof(double));
        memcpy(tM_exp, tM_exp_cache, (size_t)m * sizeof(double));
        memcpy(cla0,   cla0_cache,   (size_t)m * sizeof(double));
    } else {
        if (p > 0) {
            F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h0, &inc, &zero, xb, &inc FCONE);
        } else {
            memset(xb, 0, (size_t)n * sizeof(double));
        }
        fhl_c_z_mv(xz, v_h0_eff, zv);
        for (int i = 0; i < n; ++i) expeta[i] = exp(xb[i] + zv[i]);

        fhl_c_risk_backward(risk, expeta, tM_exp, risk_wk);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        for (int r = 0; r < m; ++r) {
            cla0[r] = di[r] / tM_exp[r];
            as_vec[r] = (cla0[r] * cla0[r]) / di[r];
        }
        fhl_c_risk_forward(risk, cla0, clam0, risk_wk);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

        /* Base Gram matrix at (beta_h0, v_h0) */
        fhl_c_xz_gram0(xz, 1, risk, expeta, clam0, as_vec, H0);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    }

    for (int i = 0; i < n; ++i) res_v[i] = del[i] - expeta[i] * clam0[i];
    if (p > 0) {
        F77_CALL(dgemv)(transT, &n, &p, &one, x, &n, res_v, &inc, &zero, dft, &inc FCONE);
    }

    /* 3. First-order Laplace adjustment for fixed effects (mord == 1) */
    if (mord == 1 && p > 0) {
        T_zz = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        dinv = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        int dinv_ok = 1;
        if (use_scaled) {
            for (int c = 0; c < Q; ++c) {
                memcpy(T_zz + (size_t)c * (size_t)Q, H0 + (size_t)(p + c) * (size_t)k_tot + p, (size_t)Q * sizeof(double));
            }
            dinv_ok = fhl_c_schur_scaled_eval(0, Q, T_zz, d_diag0, NULL, 3.1415926535900001,
                                              dinv, NULL, NULL, NULL, NULL, NULL);
        } else {
            for (int c = 0; c < Q; ++c) {
                memcpy(T_zz + (size_t)c * (size_t)Q, H0 + (size_t)(p + c) * (size_t)k_tot + p, (size_t)Q * sizeof(double));
                T_zz[c + (size_t)c * (size_t)Q] += U_diag0[c];
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
        neg_dinv  = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
        dv_db_k   = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
        xz_k      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
        dmu0_k    = (double *)fhl_g_malloc_chk(pn_len, sizeof(double), &alloc_ok);
        tM_dmu0   = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dcla0b_k  = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
        dc_k      = (double *)fhl_g_malloc_chk(pn_len, sizeof(double), &alloc_ok);
        da_k      = (double *)fhl_g_malloc_chk(pm_len, sizeof(double), &alloc_ok);
        const double **dw_ptrs = (const double **)fhl_g_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        const double **db_ptrs = (const double **)fhl_g_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        const double **da_ptrs = (const double **)fhl_g_malloc_chk((size_t)p, sizeof(const double *), &alloc_ok);
        double *ad1_vec        = (double *)fhl_g_malloc_chk((size_t)p, sizeof(double), &alloc_ok);
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
            F77_CALL(dgemv)(transN, &Q, &Q, &one, neg_dinv, &Q, g_zk, &inc, &zero, dv_db_k, &inc FCONE);
            fhl_c_z_mv(xz, dv_db_k, xz_k);
            const double *x_col = x + (size_t)k * (size_t)n;
            for (int i = 0; i < n; ++i) {
                xz_k[i] = x_col[i] + xz_k[i];
                dmu0_col[i] = expeta[i] * xz_k[i];
            }
            fhl_c_risk_backward(risk, dmu0_col, tM_dmu0, risk_wk);
            if (fhl_c_linalg_had_oom()) {
                free(dw_ptrs); free(db_ptrs); free(da_ptrs); free(ad1_vec);
                rc = -1; goto cleanup;
            }
            for (int r = 0; r < m; ++r) {
                dcla0b_k[r] = -as_vec[r] * tM_dmu0[r];
                da_col[r] = (2.0 * cla0[r] / di[r]) * dcla0b_k[r];
            }
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

    /* 4. dft2 = t(z) %*% (del - Wi * clam0) + iD * 1 - iD * iu_h0 */
    fhl_c_z_tmv(xz, res_v, dft + p);
    for (int r = 0; r < Q; ++r) {
        dft[p + r] = act_q_flag[r] ? ((dft[p + r] + iD_diag[r]) - (iD_diag[r] * iu_h0[r])) : 0.0;
    }

    /* 5. Pre-step H and Schur solve step = H^{-1} %*% dft */
    memcpy(H, H0, kk_bytes);
    for (int r = 0; r < Q; ++r) {
        H[(p + r) + (size_t)(p + r) * (size_t)k_tot] += U_diag0[r];
    }
    step = (double *)fhl_g_malloc_chk((size_t)k_tot, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }
    int step_ok = 1;
    if (use_scaled) {
        step_ok = fhl_c_schur_scaled_eval(p, Q, H0, d_diag0, dft, 3.1415926535900001,
                                          NULL, NULL, step, NULL, NULL, NULL);
    } else {
        step_ok = fhl_c_schur_solve_step(H, p, Q, dinv, dft, step);
    }
    free(dinv); dinv = NULL;
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    if (!step_ok) { rc = 0; goto cleanup; }

    /* 6. Update beta_h and v_h */
    for (int j = 0; j < p; ++j) beta_h[j] = beta_h0[j] + step[j];
    for (int r = 0; r < Q; ++r) {
        if (act_q_flag[r]) {
            v_h[r] = v_h0_eff[r] + step[p + r];
            u_h_out[r] = exp(v_h[r]);
            U_diag_out[r] = iD_diag[r] * u_h_out[r];
        } else {
            v_h[r] = 0.0;
            u_h_out[r] = 1.0;
            U_diag_out[r] = 0.0;
        }
    }
    for (int i = 0; i < nrand; ++i) {
        double a0 = alpha_h0_out[i];
        for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
            d_diag1[r] = act_comp[i] ? (a0 / u_h_out[r]) : 0.0;
        }
    }
    free(step); step = NULL;

    /* 7. Recompute eta, expeta, cla0, clam0, .G_mat, H, Hinv, invt at updated (beta_h, v_h) */
    if (p > 0) {
        F77_CALL(dgemv)(transN, &n, &p, &one, x, &n, beta_h, &inc, &zero, xb, &inc FCONE);
    } else {
        memset(xb, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v_h, zv);
    for (int i = 0; i < n; ++i) expeta[i] = exp(xb[i] + zv[i]);

    fhl_c_risk_backward(risk, expeta, tM_exp, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    for (int r = 0; r < m; ++r) {
        cla0[r] = di[r] / tM_exp[r];
        as_vec[r] = (cla0[r] * cla0[r]) / di[r];
    }
    fhl_c_risk_forward(risk, cla0, clam0, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

    fhl_c_xz_gram0(xz, 1, risk, expeta, clam0, as_vec, H0);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    if (have_cache != NULL) {
        memcpy(expeta_cache, expeta, (size_t)n * sizeof(double));
        memcpy(tM_exp_cache, tM_exp, (size_t)m * sizeof(double));
        memcpy(cla0_cache,   cla0,   (size_t)m * sizeof(double));
        *have_cache = 1;
    }
    memcpy(H, H0, kk_bytes);
    for (int r = 0; r < Q; ++r) {
        H[(p + r) + (size_t)(p + r) * (size_t)k_tot] += U_diag_out[r];
    }
    invt = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    if (!alloc_ok) { rc = -1; goto cleanup; }
    int inv_ok = 1;
    if (use_scaled) {
        inv_ok = fhl_c_schur_scaled_eval(p, Q, H0, d_diag1, NULL, 3.1415926535900001,
                                         Hinv, invt, NULL, NULL, NULL, NULL);
    } else {
        inv_ok = (p > 0)
            ? fhl_c_lognormal_inverse(H, p, Q, NULL, Hinv, invt)
            : fhl_c_solve_inv(H, Hinv, k_tot);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (inv_ok && p == 0) memcpy(invt, Hinv, qq_len * sizeof(double));
    }
    if (!inv_ok) { rc = 0; goto cleanup; }
    for (int j = 0; j < p; ++j) {
        se_beta_h[j] = sqrt(Hinv[j + (size_t)j * (size_t)k_tot]);
    }

    /* 8. Dispersion updates (unified across dord in {0, 1, 2}) */
    if (!varfixed) {
        if (dord == 2) {
            zd = (double *)fhl_g_malloc_chk((size_t)Q, sizeof(double), &alloc_ok);
            if (!alloc_ok) { rc = -1; goto cleanup; }
            fhl_c_z_tmv(xz, del, zd);
        }

        double *k2_vec    = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
        double *Hess_a    = (double *)fhl_g_malloc_chk((size_t)nrand * (size_t)nrand, sizeof(double), &alloc_ok);
        double *H_free    = (double *)fhl_g_malloc_chk((size_t)nrand * (size_t)nrand, sizeof(double), &alloc_ok);
        double *k2_free   = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
        double *step_free = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
        int    *free_idx  = (int *)fhl_g_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
        if (!alloc_ok) {
            free(k2_vec); free(Hess_a); free(H_free); free(k2_free); free(step_free); free(free_idx);
            rc = -1; goto cleanup;
        }

        int eval_ok = fhl_gamma_eval_multire_derivs(
            n, p, nrand, q, qcum, Q, xz, risk, di,
            expeta, tM_exp, cla0, as_vec, clam0,
            v_h, u_h_out, alpha_h0_out, iD_diag, act_comp,
            H0, Hinv, invt, zd, dord, varnonneg,
            k2_vec, Hess_a
        );
        if (eval_ok < 0) {
            free(k2_vec); free(Hess_a); free(H_free); free(k2_free); free(step_free); free(free_idx);
            rc = -1; goto cleanup;
        }

        for (int i = 0; i < nrand; ++i) {
            if (act_comp[i] && !R_FINITE(k2_vec[i])) {
                free(k2_vec); free(Hess_a); free(H_free); free(k2_free); free(step_free); free(free_idx);
                rc = 0; goto cleanup;
            }
            if (k2_out) k2_out[i] = k2_vec[i];
            if (k22_out) k22_out[i] = Hess_a[i + (size_t)i * (size_t)nrand];
        }

        if (nrand == 1) {
            int i = 0;
            double alpha_i = alpha_h0_out[0];
            double k2  = k2_vec[0];
            double k22 = Hess_a[0];
            int spd_ok = (R_FINITE(k22) && k22 > 0.0) ? 1 : 0;
            if (hess_pd_out) *hess_pd_out = spd_ok;
            int capped = 0;
            double a_new = alpha_i;
            if (spd_ok) {
                double step_a = k2 / k22;
                double max_up = (alpha_i < 0.25) ? 0.5 : (4.0 * alpha_i);
                a_new = alpha_i + step_a;
                if (a_new <= 0.0) {
                    a_new = alpha_i * 0.5;
                    capped = 1;
                } else if (step_a > max_up) {
                    a_new = alpha_i + max_up;
                    capped = 1;
                }
            } else {
                capped = 1;
                if (k2 > 0.0) {
                    double step_up = (alpha_i < 1e-4) ? fmax(9.0 * alpha_i, 1e-4) : fmin(0.5 * alpha_i, 0.25);
                    a_new = alpha_i + step_up;
                } else if (k2 < 0.0) {
                    a_new = alpha_i * 0.5;
                }
            }
            alpha_h[i] = a_new;
            if (capped_out) capped_out[i] = capped;
        } else {
            int n_free = 0;
            for (int i = 0; i < nrand; ++i) {
                if (!act_comp[i]) continue;
                double a0 = alpha_h0_out[i];
                int pinned = (!varnonneg && a0 <= 1.0000001e-06 && k2_vec[i] <= 0.0);
                if (pinned) {
                    alpha_h[i] = a0 * 0.5;
                    if (capped_out) capped_out[i] = 0;
                } else {
                    free_idx[n_free++] = i;
                }
            }
            for (int c = 0; c < n_free; ++c) {
                int fc = free_idx[c];
                k2_free[c] = k2_vec[fc];
                for (int r = 0; r < n_free; ++r) {
                    H_free[r + (size_t)c * (size_t)n_free] = Hess_a[free_idx[r] + (size_t)fc * (size_t)nrand];
                }
            }
            int spd_ok = 1;
            if (n_free > 0) {
                spd_ok = fhl_gamma_spd_solve(H_free, k2_free, step_free, n_free);
                if (dord == 0 && spd_ok) {
                    if (!fhl_c_solve_gesv(H_free, k2_free, step_free, n_free, 1)) {
                        spd_ok = 0;
                    }
                }
            }
            if (hess_pd_out) *hess_pd_out = spd_ok;

            if (spd_ok && n_free > 0) {
                double scale = 1.0;
                for (int f = 0; f < n_free; ++f) {
                    int idx = free_idx[f];
                    double ai = alpha_h0_out[idx];
                    double sa = step_free[f];
                    double max_up = (ai < 0.25) ? 0.5 : (4.0 * ai);
                    if (sa > max_up) {
                        double sc = max_up / sa;
                        if (sc < scale) scale = sc;
                        if (capped_out) capped_out[idx] = 1;
                    }
                }
                for (int f = 0; f < n_free; ++f) {
                    int idx = free_idx[f];
                    double ai = alpha_h0_out[idx];
                    double a_new = ai + scale * step_free[f];
                    if (!R_FINITE(a_new) || a_new <= 0.0) {
                        a_new = ai * 0.5;
                        if (capped_out) capped_out[idx] = 1;
                    }
                    alpha_h[idx] = a_new;
                }
            } else if (n_free > 0) {
                for (int f = 0; f < n_free; ++f) {
                    int idx = free_idx[f];
                    double ai = alpha_h0_out[idx];
                    double ki = k2_vec[idx];
                    double hii = Hess_a[idx + (size_t)idx * (size_t)nrand];
                    double a_new = ai;
                    if (capped_out) capped_out[idx] = 1;
                    if (R_FINITE(hii) && hii > 0.0) {
                        double sa = ki / hii;
                        double max_up = (ai < 0.25) ? 0.5 : (4.0 * ai);
                        if (sa > max_up) sa = max_up;
                        a_new = ai + sa;
                        if (a_new <= 0.0) a_new = ai * 0.5;
                    } else {
                        if (ki > 0.0) {
                            double step_up = (ai < 1e-4) ? fmax(9.0 * ai, 1e-4) : fmin(0.5 * ai, 0.25);
                            a_new = ai + step_up;
                        } else if (ki < 0.0) {
                            a_new = ai * 0.5;
                        }
                    }
                    alpha_h[idx] = a_new;
                }
            }
        }
        free(k2_vec); free(Hess_a); free(H_free); free(k2_free); free(step_free); free(free_idx);
    }

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(act_comp); free(act_q_flag); free(d_diag0); free(d_diag1); free(v_h0_eff);
    free(T_zz); free(dinv); free(term); free(step); free(invt);
    free(neg_dinv); free(dv_db_k); free(xz_k); free(dmu0_k); free(tM_dmu0);
    free(dcla0b_k); free(dc_k); free(da_k);
    free(zd);
    free(iD_diag); free(xb); free(zv); free(expeta); free(tM_exp); free(cla0);
    free(as_vec_buf);
    free(res_v); free(risk_wk); free(iu_h0); free(U_diag0);
    return rc;
}

/* Post-fit dispersion standard errors, profile/adjusted likelihoods & AIC for Gamma frailty */
static int fhl_c_gamma_se_core(
    int n, int p, int nrand, const int *q, const int *qcum, int Q,
    const double *x, const fhl_c_xz_t *xz,
    const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h1, const double *v_h1,
    const double *alpha_h, const double *alpha_h1,
    const double *H0, int dord, int varfixed, int varnonneg,
    double *se_lam_out, double *scalars_out /* [hlike, hpn1, hpn2, hpn3, hlike1, df1] */
) {
    (void)H0;
    const double pi_const = 3.1415926535900001;
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
    const char *transN = "N";
    double one = 1.0, zero = 0.0;
    int inc = 1;

    int rc = 1;
    int alloc_ok = 1;
    double *u_h1 = NULL, *iD_diag = NULL, *U_diag = NULL, *d_diag_se = NULL;
    double *xb = NULL, *zv = NULL, *eta = NULL, *expeta = NULL, *tM_exp = NULL;
    double *cla0 = NULL, *as_vec = NULL, *Bi = NULL, *risk_wk = NULL;
    double *G_mat = NULL, *H = NULL, *Hinv = NULL, *T_zz = NULL, *invt = NULL;
    double *zd = NULL, *two_pi_Hinv = NULL;
    double *a_se = NULL, *T_zz_act = NULL, *Hinv_act = NULL;
    int *act_flag = NULL, *act_q = NULL, *idx_act = NULL;

    fhl_c_linalg_begin_malloc_scope();

    a_se      = (double *)fhl_g_malloc_chk((size_t)nrand, sizeof(double), &alloc_ok);
    act_flag  = (int *)fhl_g_malloc_chk((size_t)nrand, sizeof(int), &alloc_ok);
    act_q     = (int *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(int), &alloc_ok);
    u_h1      = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    iD_diag   = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    U_diag    = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    d_diag_se = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
    xb      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    zv      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    eta     = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    expeta  = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    tM_exp  = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    cla0    = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    as_vec  = (double *)fhl_g_malloc_chk((size_t)m, sizeof(double), &alloc_ok);
    Bi      = (double *)fhl_g_malloc_chk((size_t)n, sizeof(double), &alloc_ok);
    risk_wk = (double *)fhl_g_malloc_chk(n3, sizeof(double), &alloc_ok);
    G_mat   = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    H       = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    Hinv    = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
    T_zz    = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
    invt    = (double *)fhl_g_malloc_chk(qq_len, sizeof(double), &alloc_ok);
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
            act_flag[i] = 0;
            any_inact = 1;
            a_se[i] = 0.0;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                u_h1[r]      = 1.0;
                iD_diag[r]   = 0.0;
                U_diag[r]    = 0.0;
                d_diag_se[r] = 0.0;
            }
        } else {
            act_flag[i] = 1;
            double a1 = (R_FINITE(a_orig) && a_orig > 1e-30) ? a_orig : 1e-30;
            if (a1 < min_act_a) min_act_a = a1;
            a_se[i] = a1;
            double ia = 1.0 / a1;
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                u_h1[r]      = exp(v_h1[r]);
                iD_diag[r]   = ia;
                U_diag[r]    = ia * u_h1[r];
                d_diag_se[r] = a1 / u_h1[r];
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
        eta[i]    = xb[i] + zv[i];
        expeta[i] = exp(eta[i]);
    }
    fhl_c_risk_backward(risk, expeta, tM_exp, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    for (int r = 0; r < m; ++r) {
        cla0[r]   = di[r] / tM_exp[r];
        as_vec[r] = (cla0[r] * cla0[r]) / di[r];
    }
    fhl_c_risk_forward(risk, cla0, Bi, risk_wk);
    if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

    if (H0 != NULL) {
        memcpy(G_mat, H0, kk_bytes);
    } else {
        fhl_c_xz_gram0(xz, 1, risk, expeta, Bi, as_vec, G_mat);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    }
    for (int c = 0; c < Q; ++c) {
        memcpy(T_zz + (size_t)c * (size_t)Q, G_mat + (size_t)(p + c) * (size_t)k_tot + p, (size_t)Q * sizeof(double));
        T_zz[c + (size_t)c * (size_t)Q] += U_diag[c];
    }
    memcpy(H, G_mat, kk_bytes);
    for (int r = 0; r < Q; ++r) {
        H[(p + r) + (size_t)(p + r) * (size_t)k_tot] += U_diag[r];
    }

    double logdet_W = 0.0, logdet_S2pi = 0.0;
    int se_inv_ok = 1;
    if (use_scaled_se) {
        se_inv_ok = fhl_c_schur_scaled_eval(
            p, Q, G_mat, d_diag_se, NULL, pi_const,
            Hinv, invt, NULL, &logdet_W, &logdet_S2pi, NULL
        );
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
    } else {
        se_inv_ok = (p > 0)
            ? fhl_c_lognormal_inverse(H, p, Q, NULL, Hinv, invt)
            : fhl_c_solve_inv(H, Hinv, k_tot);
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        if (se_inv_ok && p == 0 && Q > 0) {
            memcpy(invt, Hinv, qq_len * sizeof(double));
        }
    }
    if (!se_inv_ok) { rc = 0; goto cleanup; }

    if (dord == 2) {
        zd = (double *)fhl_g_malloc_chk((size_t)(Q > 0 ? Q : 1), sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        fhl_c_z_tmv(xz, del, zd);
    }

    if (!varfixed) {
        double *Hess_a   = (double *)fhl_g_malloc_chk((size_t)nrand * (size_t)nrand, sizeof(double), &alloc_ok);
        double *Hess_inv = (double *)fhl_g_malloc_chk((size_t)nrand * (size_t)nrand, sizeof(double), &alloc_ok);
        if (!alloc_ok) {
            free(Hess_a); free(Hess_inv);
            rc = -1; goto cleanup;
        }
        int eval_se_ok = fhl_gamma_eval_multire_derivs(
            n, p, nrand, q, qcum, Q, xz, risk, di,
            expeta, tM_exp, cla0, as_vec, Bi,
            v_h1, u_h1, a_se, iD_diag, act_flag,
            G_mat, Hinv, invt, zd, dord, varnonneg,
            NULL, Hess_a
        );
        if (eval_se_ok < 0) {
            free(Hess_a); free(Hess_inv);
            rc = -1; goto cleanup;
        }
        if (nrand == 1) {
            double dalp_2 = Hess_a[0];
            se_lam_out[0] = (act_flag[0] && dalp_2 > 0.0 && R_FINITE(dalp_2)) ? sqrt(1.0 / dalp_2) : NA_REAL;
        } else {
            /* Post-fit nrand x nrand dispersion observed information may be
             * nonsingular indefinite when a component lies on the alpha = 0 boundary */
            int inv_h = fhl_c_solve_gesv_inv(Hess_a, Hess_inv, nrand);
            if (fhl_c_linalg_had_oom()) {
                free(Hess_a); free(Hess_inv);
                rc = -1; goto cleanup;
            }
            for (int i = 0; i < nrand; ++i) {
                double hii = inv_h ? Hess_inv[i + (size_t)i * (size_t)nrand] : NA_REAL;
                se_lam_out[i] = (act_flag[i] && inv_h && R_FINITE(hii) && hii > 0.0) ? sqrt(hii) : NA_REAL;
            }
        }
        free(Hess_a);
        free(Hess_inv);
    }

    /* Evaluate h-likelihoods, Laplace marginal likelihoods, and effective degrees of freedom */
    double hlike1_p1 = 0.0, hlike1_p2 = 0.0;
    for (int i = 0; i < n; ++i) hlike1_p1 += del[i] * eta[i];
    for (int r = 0; r < m; ++r) hlike1_p2 += di[r] * log(tM_exp[r]);
    double hlike1 = hlike1_p1 - hlike1_p2;

    double hlike2 = 0.0, hlike2_cancel = 0.0;
    int Q_act = 0;
    for (int i = 0; i < nrand; ++i) {
        if (act_flag[i]) {
            hlike2 += fhl_gamma_se_hlike2_comp(a_se[i], qcum[i], qcum[i + 1], v_h1, u_h1, pi_const);
            if (use_scaled_se) {
                hlike2_cancel += fhl_gamma_se_hlike2_cancel_comp(a_se[i], qcum[i], qcum[i + 1], v_h1, u_h1, pi_const);
            }
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                act_q[Q_act++] = r;
            }
        }
    }
    double hlike = hlike1 + hlike2;

    double second = 0.0;
    if (dord == 2) {
        double sum_t4 = 0.0;
        for (int i = 0; i < nrand; ++i) {
            if (!act_flag[i]) continue;
            double a1 = a_se[i];
            for (int r = qcum[i]; r < qcum[i + 1]; ++r) {
                sum_t4 += a1 / (1.0 + a1 * zd[r]);
            }
        }
        second = sum_t4 / 12.0;
        free(zd); zd = NULL;
    }

    double hpn1 = hlike1, hpn2 = hlike1;
    if (use_scaled_se) {
        hpn2 = hlike1 + hlike2_cancel - 0.5 * logdet_W;
        hpn1 = hpn2 - 0.5 * logdet_S2pi;
    } else {
        for (size_t idx = 0; idx < qq_len; ++idx) {
            T_zz[idx] /= (2.0 * pi_const);
        }
        double logdet_h22 = 0.0;
        if (Q_act == Q && Q > 0) {
            logdet_h22 = fhl_c_logdet_spd(T_zz, Q, 0, 0.0, 0, 0.0);
        } else if (Q_act > 0) {
            size_t qa2 = (size_t)Q_act * (size_t)Q_act;
            T_zz_act = (double *)fhl_g_malloc_chk(qa2, sizeof(double), &alloc_ok);
            if (!alloc_ok) { rc = -1; goto cleanup; }
            for (int c = 0; c < Q_act; ++c) {
                int cc = act_q[c];
                for (int r = 0; r < Q_act; ++r) {
                    T_zz_act[r + (size_t)c * (size_t)Q_act] = T_zz[act_q[r] + (size_t)cc * (size_t)Q];
                }
            }
            logdet_h22 = fhl_c_logdet_spd(T_zz_act, Q_act, 0, 0.0, 0, 0.0);
            free(T_zz_act); T_zz_act = NULL;
        }
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }
        double pvhs = hlike - 0.5 * logdet_h22;

        two_pi_Hinv = (double *)fhl_g_malloc_chk(kk_len, sizeof(double), &alloc_ok);
        if (!alloc_ok) { rc = -1; goto cleanup; }
        for (size_t idx = 0; idx < kk_len; ++idx) {
            two_pi_Hinv[idx] = Hinv[idx] * 2.0 * pi_const;
        }
        const int k_act = p + Q_act;
        double logdet_hinv = 0.0;
        if (k_act == k_tot && k_tot > 0) {
            logdet_hinv = fhl_c_logdet_spd(two_pi_Hinv, k_tot, 0, 0.0, 0, 0.0);
        } else if (k_act > 0) {
            idx_act = (int *)fhl_g_malloc_chk((size_t)k_act, sizeof(int), &alloc_ok);
            Hinv_act = (double *)fhl_g_malloc_chk((size_t)k_act * (size_t)k_act, sizeof(double), &alloc_ok);
            if (!alloc_ok) { rc = -1; goto cleanup; }
            for (int j = 0; j < p; ++j) idx_act[j] = j;
            for (int j = 0; j < Q_act; ++j) idx_act[p + j] = p + act_q[j];
            for (int c = 0; c < k_act; ++c) {
                int cc = idx_act[c];
                for (int r = 0; r < k_act; ++r) {
                    Hinv_act[r + (size_t)c * (size_t)k_act] = two_pi_Hinv[idx_act[r] + (size_t)cc * (size_t)k_tot];
                }
            }
            logdet_hinv = fhl_c_logdet_spd(Hinv_act, k_act, 0, 0.0, 0, 0.0);
            free(idx_act); idx_act = NULL;
            free(Hinv_act); Hinv_act = NULL;
        }
        free(two_pi_Hinv); two_pi_Hinv = NULL;
        if (fhl_c_linalg_had_oom()) { rc = -1; goto cleanup; }

        double adj1 = 0.5 * logdet_hinv;
        hpn1 = hlike + adj1;
        hpn2 = pvhs;
    }
    double hpn3 = hpn1 + second;
    double df1  = fhl_c_trace_ab(Hinv, G_mat, k_tot);

    scalars_out[0] = hlike;
    scalars_out[1] = hpn1;
    scalars_out[2] = hpn2;
    scalars_out[3] = hpn3;
    scalars_out[4] = hlike1;
    scalars_out[5] = df1;

cleanup:
    fhl_c_linalg_end_malloc_scope();
    free(a_se); free(act_flag); free(act_q); free(idx_act); free(T_zz_act); free(Hinv_act);
    free(d_diag_se);
    free(zd); free(two_pi_Hinv);
    free(u_h1); free(iD_diag); free(U_diag); free(xb); free(zv); free(eta);
    free(expeta); free(tM_exp); free(cla0); free(as_vec); free(Bi); free(risk_wk);
    free(G_mat); free(H); free(Hinv); free(T_zz); free(invt);
    return rc;
}

/* ==========================================================================
 * R .Call Interface: Cox Gamma Frailty Model (fhl_fit_cox_gamma)
 * ========================================================================== */

SEXP fhl_fit_cox_gamma(
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
    SEXP s_u_h          = PROTECT(Rf_allocMatrix(REALSXP, Q, 1));
    SEXP s_Hinv_u       = PROTECT(Rf_allocMatrix(REALSXP, k_tot, k_tot));

    double *beta_curr  = (double *)R_alloc((size_t)(p > 0 ? p : 1), sizeof(double));
    double *v_curr     = (double *)R_alloc((size_t)(Q > 0 ? Q : 1), sizeof(double));
    double *alpha_curr = (double *)R_alloc((size_t)nrand, sizeof(double));
    double *U_diag     = (double *)R_alloc((size_t)(Q > 0 ? Q : 1), sizeof(double));
    double *as_vec       = (double *)R_alloc((size_t)(risk.m > 0 ? risk.m : 1), sizeof(double));
    double *expeta_cache = (double *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *tM_exp_cache = (double *)R_alloc((size_t)(risk.m > 0 ? risk.m : 1), sizeof(double));
    double *cla0_cache   = (double *)R_alloc((size_t)(risk.m > 0 ? risk.m : 1), sizeof(double));
    int have_cache       = 0;
    double *k2_curr    = (double *)R_alloc((size_t)nrand, sizeof(double));
    double *k22_curr   = (double *)R_alloc((size_t)nrand, sizeof(double));
    int    *cap_curr   = (int *)R_alloc((size_t)nrand, sizeof(int));
    int hess_pd_curr   = 1;

    if (p > 0) memcpy(beta_curr,  fhl_c_real_ptr(s_beta_init),  (size_t)p * sizeof(double));
    if (Q > 0) memcpy(v_curr,     fhl_c_real_ptr(s_v_init),     (size_t)Q * sizeof(double));
    memcpy(alpha_curr, fhl_c_real_ptr(s_alpha_init), (size_t)nrand * sizeof(double));

    double err = 1.0;
    double print_err = 1.0;
    int print_i = 1;

    for (int ii = 1; ii <= max_iter; ++ii) {
        if (err >= convergence) {
            if (p > 0) memcpy(REAL(s_beta_h0), beta_curr, (size_t)p * sizeof(double));
            if (Q > 0) memcpy(REAL(s_v_h0),    v_curr,    (size_t)Q * sizeof(double));

            int ok = fhl_c_gamma_pg_core(
                n, p, nrand, q, qcum, Q,
                x_ptr, &xz, del_ptr, &risk, di_ptr,
                REAL(s_beta_h0), REAL(s_v_h0), alpha_curr,
                mord, dord, varfixed, varnonneg,
                REAL(s_alpha_h0_out), REAL(s_beta_h), REAL(s_v_h), REAL(s_alpha_h),
                REAL(s_dft), REAL(s_Hinv), REAL(s_clam0), REAL(s_H),
                REAL(s_se_beta_h), U_diag, REAL(s_H0), REAL(s_u_h), as_vec,
                k2_curr, k22_curr, cap_curr, &hess_pd_curr,
                expeta_cache, tM_exp_cache, cla0_cache, &have_cache
            );
            if (ok < 0) {
                UNPROTECT(15);
                Rf_error("Out of memory in fhl_fit_cox_gamma at iteration %d", ii);
            }
            if (!ok) {
                UNPROTECT(15);
                Rf_error("Singular matrix in fhl_fit_cox_gamma at iteration %d", ii);
            }

            double step_bv = 0.0;
            for (int r = 0; r < Q; ++r) {
                step_bv += fabs(REAL(s_v_h)[r] - REAL(s_v_h0)[r]);
            }
            for (int j = 0; j < p; ++j) {
                step_bv += fabs(REAL(s_beta_h)[j] - REAL(s_beta_h0)[j]);
            }

            double step_err = step_bv;
            if (!varfixed) {
                int all_kkt_ok = 1;
                double kkt_viol = 0.0;
                for (int i = 0; i < nrand; ++i) {
                    double a0 = REAL(s_alpha_h0_out)[i];
                    double a1 = REAL(s_alpha_h)[i];
                    double k2i = k2_curr[i];
                    double k22i = k22_curr[i];
                    int boundary_opt = (!varnonneg && a0 <= 1.0000001e-06 && R_FINITE(k2i) && k2i <= convergence && step_bv < convergence);
                    if (boundary_opt) {
                        /* KKT optimal on boundary alpha = 0 (score <= 0) */
                        continue;
                    }
                    step_err += fabs(a0 - a1);
                    int interior_ok = (!cap_curr[i] && hess_pd_curr && R_FINITE(k22i) && k22i > 0.0 && R_FINITE(k2i) &&
                                       fabs(k2i) <= fmax(1e-4, convergence * fmax(1.0, 10.0 * k22i)));
                    if (!interior_ok) {
                        all_kkt_ok = 0;
                        double v_i = R_FINITE(k2i) ? fabs(k2i) : 1.0;
                        if (v_i > kkt_viol) kkt_viol = v_i;
                    }
                }
                if (!all_kkt_ok && step_err < convergence) {
                    step_err = fmax(convergence, kkt_viol);
                }
            }

            if (p > 0) memcpy(beta_curr,  REAL(s_beta_h),  (size_t)p * sizeof(double));
            if (Q > 0) memcpy(v_curr,     REAL(s_v_h),     (size_t)Q * sizeof(double));
            memcpy(alpha_curr, REAL(s_alpha_h), (size_t)nrand * sizeof(double));

            if (!R_FINITE(step_err)) {
                err = NA_REAL;
                print_err = NA_REAL;
                print_i = ii;
                break;
            }
            err = step_err;
            print_err = err;
            print_i = ii;
        }
    }

    int ok_fin = fhl_c_gamma_finalize_u_core(
        n, p, nrand, qcum, Q, x_ptr, &xz, &risk,
        REAL(s_beta_h), REAL(s_v_h), REAL(s_alpha_h),
        REAL(s_clam0), as_vec, REAL(s_H0), REAL(s_Hinv_u)
    );
    if (ok_fin < 0) {
        UNPROTECT(15);
        Rf_error("Out of memory in fhl_fit_cox_gamma finalize_u");
    }

    memset(REAL(s_U), 0, (size_t)Q * (size_t)Q * sizeof(double));
    for (int r = 0; r < Q; ++r) {
        REAL(s_U)[r + (size_t)r * (size_t)Q] = U_diag[r];
    }

    SEXP res2 = PROTECT(Rf_allocVector(VECSXP, 24));
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
    SET_VECTOR_ELT(res2, 22, s_u_h);
    SET_VECTOR_ELT(res2, 23, s_Hinv_u);

    double *se_lam_num = (double *)R_alloc((size_t)nrand, sizeof(double));
    double scalars[6];
    int ok_se = fhl_c_gamma_se_core(
        n, p, nrand, q, qcum, Q,
        x_ptr, &xz, del_ptr, &risk, di_ptr,
        REAL(s_beta_h), REAL(s_v_h), REAL(s_alpha_h0_out), REAL(s_alpha_h),
        have_cache ? REAL(s_H0) : NULL, dord, varfixed, varnonneg,
        se_lam_num, scalars
    );
    if (ok_se < 0) {
        UNPROTECT(16);
        Rf_error("Out of memory in fhl_fit_cox_gamma post-fit SE");
    }
    if (!ok_se) {
        UNPROTECT(16);
        Rf_error("Singular matrix in fhl_fit_cox_gamma post-fit SE");
    }

    SEXP s_se_lam;
    if (varfixed) {
        s_se_lam = PROTECT(Rf_allocMatrix(STRSXP, nrand, 1));
        for (int i = 0; i < nrand; ++i) SET_STRING_ELT(s_se_lam, i, Rf_mkChar("NULL"));
    } else {
        s_se_lam = PROTECT(Rf_allocMatrix(REALSXP, nrand, 1));
        memcpy(REAL(s_se_lam), se_lam_num, (size_t)nrand * sizeof(double));
    }

    SEXP res3 = PROTECT(Rf_allocVector(VECSXP, 7));
    SET_VECTOR_ELT(res3, 0, s_se_lam);
    for (int j = 0; j < 5; ++j) {
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

    UNPROTECT(19);
    return out;
}
