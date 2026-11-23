#include "fhl_linalg.h"
#include "fhl_native_api.h"
#include <Rmath.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#undef exp
#undef log
#define exp(x) fhl_c_r_exp(x)
#define log(x) fhl_c_r_log(x)

#ifndef FHL_INLINE
#define FHL_INLINE inline
#endif

/* ==========================================================================
 * Multilevel Accelerated Failure Time (AFT) Mixed Models (HL(0,1) & HL(1,1))
 * ========================================================================== */

static FHL_INLINE void fhl_aft_check_pos(double val, const char *what, const char *fn) {
    if (!R_FINITE(val) || val <= 1e-10) {
        Rf_error("%s: %s non-positive or non-finite (%g); variance component at boundary/degenerate", fn, what, val);
    }
}

static FHL_INLINE void fhl_aft_check_denom(double val, const char *what, const char *fn) {
    if (!R_FINITE(val) || val <= 0.0) {
        Rf_error("%s: %s denominator non-positive or non-finite (%g); REML/dispersion step degenerate", fn, what, val);
    }
}

static FHL_INLINE void fhl_mlm_mills_jet(double x, double *v_out, double *xi_out, double *dxi_out) {
    if (x <= 4.0) {
        if (x < -38.5) {
            *v_out = 0.0;
            *xi_out = 0.0;
            if (dxi_out) *dxi_out = 0.0;
            return;
        }
        double vm = exp(Rf_dnorm4(x, 0.0, 1.0, 1) - Rf_pnorm5(x, 0.0, 1.0, 0, 1));
        if (vm == 0.0 || !R_FINITE(vm)) {
            *v_out = 0.0;
            *xi_out = 0.0;
            if (dxi_out) *dxi_out = 0.0;
            return;
        }
        double xim = vm * (vm - x);
        *v_out = vm;
        *xi_out = xim;
        if (dxi_out) *dxi_out = (2.0 * vm - x) * xim - vm;
        return;
    }

    if (x <= 18.0) {
        /* Exact 60-stage Laplace continued fraction (cancellation-free r = V(x) - x) */
        double u_cf = 0.0;
        for (int k = 60; k >= 4; --k) {
            u_cf = (double)k / (x + u_cf);
        }
        double c3 = 3.0 / (x + u_cf);
        double c2 = 2.0 / (x + c3);
        double r = 1.0 / (x + c2);
        double vm = x + r;
        double xim = vm * r;
        *v_out = vm;
        *xi_out = xim;
        if (dxi_out) *dxi_out = r * (c2 * c3 - 3.0 * r * c2 + 2.0 * r * r);
        return;
    }

    /* Asymptotic Horner series in z = 1/x^2 (12 terms for V, 11 terms for xi & dxi) */
    static const double b_coef[12] = {
        1.0, -2.0, 10.0, -74.0, 706.0, -8162.0,
        110410.0, -1708394.0, 29752066.0, -576037442.0,
        12277827850.0, -285764591114.0
    };
    static const double xi_coef[11] = {
        -1.0, 6.0, -50.0, 518.0, -6354.0, 89782.0,
        -1435330.0, 25625910.0, -505785122.0,
        10944711398.0, -257834384850.0
    };
    double z = 1.0 / (x * x);
    double pv = b_coef[11];
    for (int k = 10; k >= 0; --k) pv = b_coef[k] + z * pv;
    double vm = x + pv / x;

    double pxi = xi_coef[10];
    for (int k = 9; k >= 0; --k) pxi = xi_coef[k] + z * pxi;
    double xim = 1.0 + z * pxi;

    *v_out = vm;
    *xi_out = xim;
    if (dxi_out) {
        double pxip = -22.0 * xi_coef[10];
        for (int k = 9; k >= 0; --k) pxip = (-2.0 * (double)(k + 1)) * xi_coef[k] + z * pxip;
        *dxi_out = pxip / (x * x * x);
    }
}

/* Compute sum(hinv * (B_xz^T diag(w) B_xz)) in C using BLAS dgemm */
static double fhl_mlm_weighted_gram_trace(const double *B_xz, int nn, int r_tot,
                                          const double *w, const double *hinv,
                                          double *W_B, double *H_star) {
    for (int c = 0; c < r_tot; c++) {
        const double *bcol = B_xz + (size_t)c * nn;
        double *wcol = W_B + (size_t)c * nn;
        for (int r = 0; r < nn; r++) wcol[r] = w[r] * bcol[r];
    }
    double one = 1.0, zero = 0.0;
    F77_CALL(dgemm)("T", "N", &r_tot, &r_tot, &nn, &one, B_xz, &nn, W_B, &nn, &zero, H_star, &r_tot FCONE FCONE);
    double tr = 0.0;
    for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
        tr += hinv[idx] * H_star[idx];
    }
    return tr;
}

/* Evaluate H^{-1} at final updated (beta_h0, v_h0, alpha1, alpha2, phi) for SE */
static void fhl_mlm_eval_dense_hinv(
    int nn, int pp, int qq, int qq1, int r_tot,
    const double *y, const double *x, const double *z, const double *B_xz, const double *delta,
    const double *beta_h0, const double *v_h0, double alpha1, double alpha2, double phi,
    double *muij, double *w1_vec, double *W_B, double *H_mlm, double *hinv
) {
    double one = 1.0, zero = 0.0;
    F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, &(int){1}, &zero, muij, &(int){1} FCONE);
    F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v_h0, &(int){1}, &one, muij, &(int){1} FCONE);
    double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi;
    for (int r = 0; r < nn; r++) {
        double mr = inv_sqrt_phi * (y[r] - muij[r]);
        double vr = 0.0, xir = 0.0;
        fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
        double w1r = delta[r] + (1.0 - delta[r]) * xir;
        w1_vec[r] = w1r * inv_phi;
    }
    for (int c = 0; c < r_tot; c++) {
        const double *bcol = B_xz + (size_t)c * nn;
        double *wcol = W_B + (size_t)c * nn;
        for (int r = 0; r < nn; r++) wcol[r] = w1_vec[r] * bcol[r];
    }
    F77_CALL(dgemm)("T", "N", &r_tot, &r_tot, &nn, &one, B_xz, &nn, W_B, &nn, &zero, H_mlm, &r_tot FCONE FCONE);
    double inv_a1 = 1.0 / alpha1, inv_a2 = 1.0 / alpha2;
    for (int j = 0; j < qq1; j++) H_mlm[(pp + j) + (size_t)r_tot * (pp + j)] += inv_a1;
    for (int j = qq1; j < qq; j++) H_mlm[(pp + j) + (size_t)r_tot * (pp + j)] += inv_a2;
    if (!fhl_c_solve_inv(H_mlm, hinv, r_tot)) {
        Rf_error("Singular matrix in mlmfit final SE evaluation (H_mlm) (not positive definite)");
    }
}

/* ==========================================================================
 * HL(1,1) (mord = 1, dord = 1) Native C Solvers for 1-RE and 2-RE AFT Models
 * ========================================================================== */

static SEXP fhl_aft_hl11_pack_result(
    int pp, int qq, int nrand,
    const double *beta, const double *se_beh, const double *v,
    const double *alpha, double phi, int iter_done, double last_pdiff,
    int converged_flag, const char *status_str, const char *reason_str
) {
    SEXP out = PROTECT(allocVector(VECSXP, 10));
    SEXP beta_s = PROTECT(allocVector(REALSXP, pp));
    SEXP se_s = PROTECT(allocVector(REALSXP, pp));
    SEXP v_s = PROTECT(allocVector(REALSXP, qq));
    SEXP alpha_s = PROTECT(allocVector(REALSXP, nrand));

    for (int j = 0; j < pp; j++) {
        REAL(beta_s)[j] = beta ? beta[j] : NA_REAL;
        REAL(se_s)[j] = (converged_flag && se_beh) ? se_beh[j] : NA_REAL;
    }
    for (int j = 0; j < qq; j++) {
        REAL(v_s)[j] = v ? v[j] : NA_REAL;
    }
    for (int r = 0; r < nrand; r++) {
        REAL(alpha_s)[r] = alpha ? alpha[r] : NA_REAL;
    }

    SET_VECTOR_ELT(out, 0, beta_s);
    SET_VECTOR_ELT(out, 1, se_s);
    SET_VECTOR_ELT(out, 2, v_s);
    SET_VECTOR_ELT(out, 3, alpha_s);
    SET_VECTOR_ELT(out, 4, ScalarReal(phi));
    SET_VECTOR_ELT(out, 5, ScalarInteger(iter_done));
    SET_VECTOR_ELT(out, 6, ScalarReal(last_pdiff));
    SET_VECTOR_ELT(out, 7, ScalarLogical(converged_flag ? 1 : 0));
    SET_VECTOR_ELT(out, 8, mkString(status_str));
    SET_VECTOR_ELT(out, 9, mkString(reason_str));

    UNPROTECT(5);
    return out;
}

static double fhl_hl11_1re_eval_h(
    int nn, int qq, const double *y, const double *delta,
    const double *mu_base, const double *v, const int *g,
    double phi, double alpha
) {
    const double two_pi = 6.283185307179586476925286766559;
    double inv_sqrt_phi = 1.0 / sqrt(phi);
    double c_norm = -0.5 * log(two_pi * phi);
    double l1 = 0.0;
    for (int r = 0; r < nn; r++) {
        double mr = (y[r] - (mu_base[r] + v[g[r]])) * inv_sqrt_phi;
        if (delta[r] > 0.5) {
            l1 += c_norm - 0.5 * mr * mr;
        } else {
            l1 += Rf_pnorm5(mr, 0.0, 1.0, 0, 1);
        }
    }
    double ss_v = 0.0;
    for (int j = 0; j < qq; j++) ss_v += v[j] * v[j];
    double l2 = -0.5 * (double)qq * log(two_pi * alpha) - 0.5 * ss_v / alpha;
    return l1 + l2;
}

static int fhl_hl11_1re_eval_pv(
    int nn, int qq, const double *y, const double *delta,
    const double *mu_base, const double *v, const int *g,
    double phi, double alpha, double *zz_w1, double *pv_out
) {
    const double two_pi = 6.283185307179586476925286766559;
    double inv_sqrt_phi = 1.0 / sqrt(phi);
    double c_norm = -0.5 * log(two_pi * phi);
    double l1 = 0.0;
    memset(zz_w1, 0, (size_t)qq * sizeof(double));
    for (int r = 0; r < nn; r++) {
        double mr = (y[r] - (mu_base[r] + v[g[r]])) * inv_sqrt_phi;
        double vr = 0.0, xir = 0.0;
        fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
        if (delta[r] > 0.5) {
            l1 += c_norm - 0.5 * mr * mr;
            zz_w1[g[r]] += 1.0;
        } else {
            l1 += Rf_pnorm5(mr, 0.0, 1.0, 0, 1);
            zz_w1[g[r]] += xir;
        }
    }
    double ss_v = 0.0, half_logdet_M = 0.0;
    double ratio = alpha / phi;
    for (int j = 0; j < qq; j++) {
        ss_v += v[j] * v[j];
        double mj = 1.0 + ratio * zz_w1[j];
        if (!R_FINITE(mj) || mj <= 0.0) return 0;
        half_logdet_M += 0.5 * log(mj);
    }
    double pv = l1 - 0.5 * (ss_v / alpha) - half_logdet_M;
    if (!R_FINITE(pv)) return 0;
    *pv_out = pv;
    return 1;
}

static int fhl_hl11_1re_solve_v(
    int nn, int qq, const double *y, const double *delta,
    const double *mu_base, const int *g, double phi, double alpha,
    double *v, double tol_v, int maxit,
    double *gv, double *zz_w1, double *step_v, double *v_try
) {
    double inv_sqrt_phi = 1.0 / sqrt(phi);
    double inv_phi = 1.0 / phi;
    double inv_alpha = 1.0 / alpha;

    for (int it = 1; it <= maxit; it++) {
        for (int j = 0; j < qq; j++) {
            gv[j] = -inv_alpha * v[j];
            zz_w1[j] = 0.0;
        }
        for (int r = 0; r < nn; r++) {
            double mr = (y[r] - (mu_base[r] + v[g[r]])) * inv_sqrt_phi;
            double vr = 0.0, xir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
            double ar = delta[r] * mr + (1.0 - delta[r]) * vr;
            double wr = delta[r] + (1.0 - delta[r]) * xir;
            int gr = g[r];
            gv[gr] += ar * inv_sqrt_phi;
            zz_w1[gr] += wr;
        }
        double max_gv = 0.0;
        for (int j = 0; j < qq; j++) {
            if (!R_FINITE(gv[j]) || !R_FINITE(zz_w1[j])) return 0;
            double ag = fabs(gv[j]);
            if (ag > max_gv) max_gv = ag;
        }
        if (max_gv < tol_v) return 1;

        for (int j = 0; j < qq; j++) {
            double dj = zz_w1[j] * inv_phi + inv_alpha;
            if (!R_FINITE(dj) || dj <= 0.0) return 0;
            step_v[j] = gv[j] / dj;
        }
        double h_cur = fhl_hl11_1re_eval_h(nn, qq, y, delta, mu_base, v, g, phi, alpha);
        double lam = 1.0;
        int moved = 0;
        while (lam > 1e-12) {
            for (int j = 0; j < qq; j++) v_try[j] = v[j] + lam * step_v[j];
            double h_try = fhl_hl11_1re_eval_h(nn, qq, y, delta, mu_base, v_try, g, phi, alpha);
            if (R_FINITE(h_try) && h_try >= h_cur - 1e-13 * fmax(1.0, fabs(h_cur))) {
                memcpy(v, v_try, (size_t)qq * sizeof(double));
                moved = 1;
                break;
            }
            lam *= 0.5;
        }
        if (!moved) {
            for (int j = 0; j < qq; j++) gv[j] = -inv_alpha * v[j];
            for (int r = 0; r < nn; r++) {
                double mr = (y[r] - (mu_base[r] + v[g[r]])) * inv_sqrt_phi;
                double vr = 0.0, xir = 0.0;
                fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
                double ar = delta[r] * mr + (1.0 - delta[r]) * vr;
                gv[g[r]] += ar * inv_sqrt_phi;
            }
            max_gv = 0.0;
            for (int j = 0; j < qq; j++) {
                double ag = fabs(gv[j]);
                if (ag > max_gv) max_gv = ag;
            }
            return (max_gv <= 1e-7) ? 1 : 0;
        }
    }
    return 1;
}

typedef struct {
    double alpha_new, alpha_unc, sc_a, rel_a, denom_a;
    double phi_new, phi_unc, sc_p, rel_p, denom_p;
} fhl_hl11_1re_reml_t;

static int fhl_hl11_1re_state_schur(
    int nn, int pp, int qq,
    const double *y, const double *x, const int *g, const double *delta,
    const double *mu_base, const double *v, double phi, double alpha,
    double *mij, double *a_vec, double *w_vec, double *xip_vec, double *resid_star,
    double *g_beta, double *max_gv_out, double *grad_pv,
    double *xx_w, double *xz_w, double *zz_w1, double *wx_buf,
    double *K_xz, double *schur_S, double *hinv_XX, double *hinv_XZ,
    double *hinv_ZZ_diag, double *iW_diag, double *sum_hinv_ZZ_out,
    double *c_vec, double *c_grp
) {
    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int inc1 = 1;
    double sqrt_phi = sqrt(phi), inv_sqrt_phi = 1.0 / sqrt_phi, inv_phi = 1.0 / phi;
    double inv_alpha = 1.0 / alpha;

    memset(zz_w1, 0, (size_t)qq * sizeof(double));
    memset(xz_w, 0, (size_t)pp * (size_t)qq * sizeof(double));
    memset(c_grp, 0, (size_t)qq * sizeof(double));

    /* Use c_grp temporarily to accumulate Z^T a / sqrt(phi) - v / alpha for max_gv */
    for (int j = 0; j < qq; j++) c_grp[j] = -inv_alpha * v[j];

    for (int r = 0; r < nn; r++) {
        double mu_r = mu_base[r] + v[g[r]];
        double mr = (y[r] - mu_r) * inv_sqrt_phi;
        double vr = 0.0, xir = 0.0, dxir = 0.0;
        fhl_mlm_mills_jet(mr, &vr, &xir, &dxir);
        double ar = delta[r] * mr + (1.0 - delta[r]) * vr;
        double wr = delta[r] + (1.0 - delta[r]) * xir;

        mij[r] = mr;
        a_vec[r] = ar;
        w_vec[r] = wr;
        xip_vec[r] = dxir;
        resid_star[r] = delta[r] * (y[r] - mu_r) + (1.0 - delta[r]) * (sqrt_phi * vr);

        int gr = g[r];
        zz_w1[gr] += wr;
        c_grp[gr] += ar * inv_sqrt_phi;
    }

    double max_gv = 0.0;
    for (int j = 0; j < qq; j++) {
        if (!R_FINITE(c_grp[j]) || !R_FINITE(zz_w1[j])) return 0;
        double ag = fabs(c_grp[j]);
        if (ag > max_gv) max_gv = ag;
    }
    *max_gv_out = max_gv;

    F77_CALL(dgemv)("T", &nn, &pp, &inv_sqrt_phi, x, &nn, a_vec, &inc1, &zero, g_beta, &inc1 FCONE);

    for (int a = 0; a < pp; a++) {
        const double *xcol = x + (size_t)a * (size_t)nn;
        double *wxcol = wx_buf + (size_t)a * (size_t)nn;
        for (int r = 0; r < nn; r++) {
            double wxr = (w_vec[r] * inv_phi) * xcol[r];
            wxcol[r] = wxr;
            xz_w[a + (size_t)pp * (size_t)g[r]] += wxr;
        }
    }
    F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);

    for (int j = 0; j < qq; j++) {
        double dj = zz_w1[j] * inv_phi + inv_alpha;
        double wj = zz_w1[j] + phi * inv_alpha;
        if (!R_FINITE(dj) || dj <= 0.0 || !R_FINITE(wj) || wj <= 0.0) return 0;
        double inv_dj = 1.0 / dj;
        iW_diag[j] = 1.0 / wj;
        double *kcol = K_xz + (size_t)pp * (size_t)j;
        const double *bcol = xz_w + (size_t)pp * (size_t)j;
        for (int a = 0; a < pp; a++) kcol[a] = bcol[a] * inv_dj;
        /* Store inv_dj temporarily in hinv_ZZ_diag[j] */
        hinv_ZZ_diag[j] = inv_dj;
    }

    memcpy(schur_S, xx_w, (size_t)pp * (size_t)pp * sizeof(double));
    F77_CALL(dgemm)("N", "T", &pp, &pp, &qq, &neg_one, xz_w, &pp, K_xz, &pp, &one, schur_S, &pp FCONE FCONE);
    if (!fhl_c_solve_inv(schur_S, hinv_XX, pp)) return 0;

    F77_CALL(dgemm)("N", "N", &pp, &qq, &pp, &neg_one, hinv_XX, &pp, K_xz, &pp, &zero, hinv_XZ, &pp FCONE FCONE);
    double sum_zz_diag = 0.0;
    for (int j = 0; j < qq; j++) {
        const double *hcol = hinv_XZ + (size_t)pp * (size_t)j;
        const double *kcol = K_xz + (size_t)pp * (size_t)j;
        double quad = 0.0;
        for (int a = 0; a < pp; a++) quad += hcol[a] * kcol[a];
        hinv_ZZ_diag[j] -= quad;
        sum_zz_diag += hinv_ZZ_diag[j];
    }
    *sum_hinv_ZZ_out = sum_zz_diag;

    /* HL(1,1) Laplace trace gradient a_d1 = (X - K_xz^T Z^T)^T c_vec */
    memset(c_grp, 0, (size_t)qq * sizeof(double));
    for (int r = 0; r < nn; r++) {
        int gr = g[r];
        double cr = -iW_diag[gr] * ((1.0 - delta[r]) * inv_sqrt_phi) * xip_vec[r];
        c_vec[r] = cr;
        c_grp[gr] += cr;
    }
    F77_CALL(dgemv)("T", &nn, &pp, &one, x, &nn, c_vec, &inc1, &zero, grad_pv, &inc1 FCONE);
    F77_CALL(dgemv)("N", &pp, &qq, &neg_one, K_xz, &pp, c_grp, &inc1, &one, grad_pv, &inc1 FCONE);
    for (int a = 0; a < pp; a++) {
        grad_pv[a] = g_beta[a] - 0.5 * grad_pv[a];
    }
    return 1;
}

static double fhl_mlm1_schur_weighted_trace(
    const double *x, const int *g, int nn, int pp, int qq,
    const double *w,
    const double *hinv_XX, const double *hinv_XZ, const double *hinv_ZZ_diag,
    double *xx_w, double *xz_w, double *zz_w, double *wx_buf
);

static void fhl_hl11_1re_reml(
    int nn, int pp, int qq,
    const double *x, const int *g, const double *delta,
    const double *v, double phi, double alpha,
    const double *mij, const double *a_vec, const double *w_vec,
    const double *xip_vec, const double *resid_star,
    const double *hinv_XX, const double *hinv_XZ, const double *hinv_ZZ_diag,
    const double *iW_diag, double sum_hinv_ZZ,
    double min_alpha, double min_phi,
    double *dva, double *term_q, double *dw_vec,
    double *xx_w, double *xz_w, double *zz_w, double *wx_buf,
    fhl_hl11_1re_reml_t *out
) {
    double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi, inv_phi2 = inv_phi * inv_phi;
    double scale_dva = phi / (alpha * alpha);
    for (int j = 0; j < qq; j++) dva[j] = scale_dva * iW_diag[j] * v[j];
    for (int r = 0; r < nn; r++) {
        dw_vec[r] = -inv_phi * inv_sqrt_phi * (1.0 - delta[r]) * dva[g[r]] * xip_vec[r];
    }
    double tr_a = fhl_mlm1_schur_weighted_trace(
        x, g, nn, pp, qq, dw_vec, hinv_XX, hinv_XZ, hinv_ZZ_diag,
        xx_w, xz_w, zz_w, wx_buf
    ) - (1.0 / (alpha * alpha)) * sum_hinv_ZZ;
    double gamma1 = -alpha * tr_a;
    double ss_v = 0.0;
    for (int j = 0; j < qq; j++) ss_v += v[j] * v[j];
    double denom_a = (double)qq - gamma1;
    double sc_a = ss_v / alpha - denom_a;
    out->denom_a = denom_a;
    out->sc_a = sc_a;
    if (R_FINITE(denom_a) && denom_a > 1e-12) {
        out->alpha_unc = ss_v / denom_a;
        out->rel_a = sc_a / denom_a;
        out->alpha_new = fmax(min_alpha, out->alpha_unc);
    } else {
        out->alpha_unc = NA_REAL;
        out->rel_a = NA_REAL;
        out->alpha_new = min_alpha;
    }

    memset(term_q, 0, (size_t)qq * sizeof(double));
    double sum_w1 = 0.0, rss = 0.0;
    for (int r = 0; r < nn; r++) {
        term_q[g[r]] += a_vec[r] + w_vec[r] * mij[r];
        sum_w1 += w_vec[r];
        rss += resid_star[r] * resid_star[r];
    }
    double scale_dvp = -0.5 * inv_sqrt_phi;
    for (int j = 0; j < qq; j++) dva[j] = scale_dvp * iW_diag[j] * term_q[j];
    for (int r = 0; r < nn; r++) {
        double dp_m = -0.5 * mij[r] * inv_phi - inv_sqrt_phi * dva[g[r]];
        dw_vec[r] = -w_vec[r] * inv_phi2 + inv_phi * (1.0 - delta[r]) * dp_m * xip_vec[r];
    }
    double tr_p = fhl_mlm1_schur_weighted_trace(
        x, g, nn, pp, qq, dw_vec, hinv_XX, hinv_XZ, hinv_ZZ_diag,
        xx_w, xz_w, zz_w, wx_buf
    );
    double gamma0 = phi * tr_p;
    double denom_p = sum_w1 + gamma0;
    double sc_p = rss / phi - denom_p;
    out->denom_p = denom_p;
    out->sc_p = sc_p;
    if (R_FINITE(denom_p) && denom_p > 1e-12) {
        out->phi_unc = rss / denom_p;
        out->rel_p = sc_p / denom_p;
        out->phi_new = fmax(min_phi, out->phi_unc);
    } else {
        out->phi_unc = NA_REAL;
        out->rel_p = NA_REAL;
        out->phi_new = fmax(min_phi, phi);
    }
}

/* Fast O(N * p^2 + K * p^2) C HL(1,1) solver for 1-RE shared cluster intercept */
static SEXP fhl_aft_1re_cluster_hl11(
    int nn, int pp, int qq,
    const double *y, const double *x, const int *g, const double *delta, int maxiter
) {
    const double tol_param = 1e-6, tol_gv = 1e-7, tol_gpv = 1e-5, tol_reml_sc = 1e-5;
    const double min_alpha = 1e-6, min_phi = 1e-8;
    double one = 1.0, zero = 0.0;
    int inc1 = 1;

    double *beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *beta_new = (double *)R_alloc((size_t)pp, sizeof(double));
    double *b_try = (double *)R_alloc((size_t)pp, sizeof(double));
    double *dbeta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *ndec3 = (double *)R_alloc((size_t)pp, sizeof(double));
    double *g_beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *grad_pv = (double *)R_alloc((size_t)pp, sizeof(double));
    double *se_beh = (double *)R_alloc((size_t)pp, sizeof(double));
    memset(beta, 0, (size_t)pp * sizeof(double));

    double *v = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_new = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_after = (double *)R_alloc((size_t)qq, sizeof(double));
    double *vt_try = (double *)R_alloc((size_t)qq, sizeof(double));
    double *gv_buf = (double *)R_alloc((size_t)qq, sizeof(double));
    double *step_v = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_try_inner = (double *)R_alloc((size_t)qq, sizeof(double));
    memset(v, 0, (size_t)qq * sizeof(double));

    double *mu_base = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mu_try = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *a_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *xip_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *resid_star = (double *)R_alloc((size_t)nn, sizeof(double));
    double *c_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *dw_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *wx_buf = (double *)R_alloc((size_t)nn * (size_t)pp, sizeof(double));

    double *xx_w = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *schur_S = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *hinv_XX = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *xz_w = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *K_xz = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *hinv_XZ = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *zz_w1 = (double *)R_alloc((size_t)qq, sizeof(double));
    double *zz_tmp = (double *)R_alloc((size_t)qq, sizeof(double));
    double *hinv_ZZ_diag = (double *)R_alloc((size_t)qq, sizeof(double));
    double *iW_diag = (double *)R_alloc((size_t)qq, sizeof(double));
    double *c_grp = (double *)R_alloc((size_t)qq, sizeof(double));
    double *dva = (double *)R_alloc((size_t)qq, sizeof(double));
    double *term_q = (double *)R_alloc((size_t)qq, sizeof(double));

    double alpha = 0.05, phi = 1.0;
    double pv_prev = NA_REAL, last_pdiff = R_PosInf;
    int converged_flag = 0, ls_rejects = 0, iter_done = 0;
    const char *status_str = "NOT_CONVERGED";
    const char *reason_str = "Did not converge within Maxiter";

    for (int it = 1; it <= maxiter; it++) {
        iter_done = it;
        if (alpha < min_alpha) alpha = min_alpha;
        if (phi < min_phi) phi = min_phi;

        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta, &inc1, &zero, mu_base, &inc1 FCONE);
        if (!fhl_hl11_1re_solve_v(nn, qq, y, delta, mu_base, g, phi, alpha, v, 1e-11, 50, gv_buf, zz_w1, step_v, v_try_inner)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Inner v solver failed");
        }

        double max_gv = 0.0, sum_hinv_ZZ = 0.0;
        if (!fhl_hl11_1re_state_schur(
                nn, pp, qq, y, x, g, delta, mu_base, v, phi, alpha,
                mij, a_vec, w_vec, xip_vec, resid_star, g_beta, &max_gv, grad_pv,
                xx_w, xz_w, zz_w1, wx_buf, K_xz, schur_S, hinv_XX, hinv_XZ,
                hinv_ZZ_diag, iW_diag, &sum_hinv_ZZ, c_vec, c_grp)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Cholesky of Schur Hessian failed");
        }

        F77_CALL(dgemv)("N", &pp, &pp, &one, hinv_XX, &pp, grad_pv, &inc1, &zero, dbeta, &inc1 FCONE);
        double pv_cur = 0.0;
        if (!fhl_hl11_1re_eval_pv(nn, qq, y, delta, mu_base, v, g, phi, alpha, zz_tmp, &pv_cur)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Failed evaluating p_v");
        }

        double max_gpv = 0.0, slope = 0.0;
        for (int a = 0; a < pp; a++) {
            double ag = fabs(grad_pv[a]);
            if (ag > max_gpv) max_gpv = ag;
            slope += grad_pv[a] * dbeta[a];
        }

        memcpy(beta_new, beta, (size_t)pp * sizeof(double));
        memcpy(v_new, v, (size_t)qq * sizeof(double));
        int accepted = 0;
        if (max_gpv == 0.0) {
            accepted = 1;
        } else if (!R_FINITE(slope) || slope <= 0.0) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Non-ascent Schur direction");
        } else {
            double lam = 1.0;
            while (lam > 1e-10) {
                for (int a = 0; a < pp; a++) b_try[a] = beta[a] + lam * dbeta[a];
                F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, b_try, &inc1, &zero, mu_try, &inc1 FCONE);
                memcpy(vt_try, v, (size_t)qq * sizeof(double));
                if (fhl_hl11_1re_solve_v(nn, qq, y, delta, mu_try, g, phi, alpha, vt_try, 1e-11, 50, gv_buf, zz_tmp, step_v, v_try_inner)) {
                    double pv_try = 0.0;
                    if (fhl_hl11_1re_eval_pv(nn, qq, y, delta, mu_try, vt_try, g, phi, alpha, zz_tmp, &pv_try) &&
                        pv_try >= pv_cur + 1e-4 * lam * slope) {
                        memcpy(beta_new, b_try, (size_t)pp * sizeof(double));
                        memcpy(v_new, vt_try, (size_t)qq * sizeof(double));
                        accepted = 1;
                        break;
                    }
                }
                lam *= 0.5;
            }
            if (!accepted) ls_rejects++;
        }

        /* Step 4: REML dispersion update at (beta_new, v_new, phi, alpha) */
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_new, &inc1, &zero, mu_base, &inc1 FCONE);
        if (!fhl_hl11_1re_state_schur(
                nn, pp, qq, y, x, g, delta, mu_base, v_new, phi, alpha,
                mij, a_vec, w_vec, xip_vec, resid_star, g_beta, &max_gv, grad_pv,
                xx_w, xz_w, zz_w1, wx_buf, K_xz, schur_S, hinv_XX, hinv_XZ,
                hinv_ZZ_diag, iW_diag, &sum_hinv_ZZ, c_vec, c_grp)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta_new, NULL, v_new, &alpha, phi, it, last_pdiff, 0, "NOT_CONVERGED", "Cholesky failed before REML step");
        }
        fhl_hl11_1re_reml_t reml_step;
        fhl_hl11_1re_reml(
            nn, pp, qq, x, g, delta, v_new, phi, alpha,
            mij, a_vec, w_vec, xip_vec, resid_star,
            hinv_XX, hinv_XZ, hinv_ZZ_diag, iW_diag, sum_hinv_ZZ,
            min_alpha, min_phi, dva, term_q, dw_vec, xx_w, xz_w, zz_tmp, wx_buf, &reml_step
        );
        double alpha_new = reml_step.alpha_new;
        double phi_new = reml_step.phi_new;

        /* Step 5: Sync v_after at (beta_new, phi_new, alpha_new) & check stationarity */
        memcpy(v_after, v_new, (size_t)qq * sizeof(double));
        if (!fhl_hl11_1re_solve_v(nn, qq, y, delta, mu_base, g, phi_new, alpha_new, v_after, 1e-11, 50, gv_buf, zz_tmp, step_v, v_try_inner)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta_new, NULL, v_new, &alpha_new, phi_new, it, last_pdiff, 0, "NOT_CONVERGED", "Inner v solve failed after REML update");
        }

        double pdiff = fabs(alpha_new - alpha) + fabs(phi_new - phi);
        for (int a = 0; a < pp; a++) pdiff += fabs(beta_new[a] - beta[a]);
        for (int j = 0; j < qq; j++) pdiff += fabs(v_after[j] - v[j]);

        double max_gv3 = 0.0, sum_hinv_ZZ3 = 0.0;
        if (!fhl_hl11_1re_state_schur(
                nn, pp, qq, y, x, g, delta, mu_base, v_after, phi_new, alpha_new,
                mij, a_vec, w_vec, xip_vec, resid_star, g_beta, &max_gv3, grad_pv,
                xx_w, xz_w, zz_w1, wx_buf, K_xz, schur_S, hinv_XX, hinv_XZ,
                hinv_ZZ_diag, iW_diag, &sum_hinv_ZZ3, c_vec, c_grp)) {
            return fhl_aft_hl11_pack_result(pp, qq, 1, beta_new, NULL, v_after, &alpha_new, phi_new, it, pdiff, 0, "NOT_CONVERGED", "Cholesky failed at post-update check");
        }
        F77_CALL(dgemv)("N", &pp, &pp, &one, hinv_XX, &pp, grad_pv, &inc1, &zero, ndec3, &inc1 FCONE);
        double pv3 = 0.0;
        int pv3_ok = fhl_hl11_1re_eval_pv(nn, qq, y, delta, mu_base, v_after, g, phi_new, alpha_new, zz_tmp, &pv3);
        double odiff = (!ISNAN(pv_prev) && pv3_ok) ? fabs(pv3 - pv_prev) : R_PosInf;

        fhl_hl11_1re_reml_t reml3;
        fhl_hl11_1re_reml(
            nn, pp, qq, x, g, delta, v_after, phi_new, alpha_new,
            mij, a_vec, w_vec, xip_vec, resid_star,
            hinv_XX, hinv_XZ, hinv_ZZ_diag, iW_diag, sum_hinv_ZZ3,
            min_alpha, min_phi, dva, term_q, dw_vec, xx_w, xz_w, zz_tmp, wx_buf, &reml3
        );

        int alpha_bound = (alpha_new <= min_alpha + 1e-12);
        int phi_bound = (phi_new <= min_phi + 1e-14) || (!ISNAN(reml_step.phi_unc) && reml_step.phi_unc <= min_phi);

        double max_ndec = 0.0, max_gpv3 = 0.0;
        for (int a = 0; a < pp; a++) {
            if (fabs(ndec3[a]) > max_ndec) max_ndec = fabs(ndec3[a]);
            if (fabs(grad_pv[a]) > max_gpv3) max_gpv3 = fabs(grad_pv[a]);
        }
        int gv_ok = (max_gv3 <= tol_gv);
        int ndec_ok = (max_ndec <= tol_param);
        int gpv_ok = (max_gpv3 <= tol_gpv);
        int reml_a_ok = 0;
        if (R_FINITE(reml3.denom_a) && reml3.denom_a > 1e-12 && R_FINITE(reml3.sc_a) && R_FINITE(reml3.rel_a)) {
            if (alpha_bound) {
                reml_a_ok = (reml3.sc_a <= 1e-13) && (reml3.rel_a <= tol_reml_sc);
            } else {
                reml_a_ok = (fabs(reml3.sc_a) <= tol_reml_sc) && (fabs(reml3.rel_a) <= tol_reml_sc);
            }
        }
        int phi_ok = (!phi_bound) && R_FINITE(reml3.denom_p) && (reml3.denom_p > 1e-12) &&
                     R_FINITE(reml3.sc_p) && R_FINITE(reml3.rel_p) &&
                     (fabs(reml3.sc_p) <= tol_reml_sc) && (fabs(reml3.rel_p) <= tol_reml_sc);

        memcpy(beta, beta_new, (size_t)pp * sizeof(double));
        memcpy(v, v_after, (size_t)qq * sizeof(double));
        alpha = alpha_new;
        phi = phi_new;
        last_pdiff = pdiff;
        if (pv3_ok) pv_prev = pv3;

        if (accepted && R_FINITE(pdiff) && pdiff <= tol_param &&
            R_FINITE(odiff) && odiff <= fmax(1e-6, 1e-8 * (1.0 + fabs(pv_prev))) &&
            gv_ok && ndec_ok && gpv_ok && reml_a_ok && phi_ok) {
            converged_flag = 1;
            status_str = alpha_bound ? "BOUNDARY_SOLUTION" : "CONVERGED_INTERIOR";
            reason_str = alpha_bound ? "Converged at lower variance-component boundary" : "Converged in interior";
            break;
        }
    }

    /* Final tight evaluation at terminal iterate (tol_v = 1e-12) */
    F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta, &inc1, &zero, mu_base, &inc1 FCONE);
    if (!fhl_hl11_1re_solve_v(nn, qq, y, delta, mu_base, g, phi, alpha, v, 1e-12, 50, gv_buf, zz_tmp, step_v, v_try_inner)) {
        return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, iter_done, last_pdiff, 0, "NOT_CONVERGED", "Final v_tilde solve failed");
    }
    double max_gv_f = 0.0, sum_hinv_ZZ_f = 0.0;
    if (!fhl_hl11_1re_state_schur(
            nn, pp, qq, y, x, g, delta, mu_base, v, phi, alpha,
            mij, a_vec, w_vec, xip_vec, resid_star, g_beta, &max_gv_f, grad_pv,
            xx_w, xz_w, zz_w1, wx_buf, K_xz, schur_S, hinv_XX, hinv_XZ,
            hinv_ZZ_diag, iW_diag, &sum_hinv_ZZ_f, c_vec, c_grp)) {
        return fhl_aft_hl11_pack_result(pp, qq, 1, beta, NULL, v, &alpha, phi, iter_done, last_pdiff, 0, "NOT_CONVERGED", "Final Hessian Cholesky failed");
    }
    for (int a = 0; a < pp; a++) {
        double vdiag = hinv_XX[a + (size_t)pp * (size_t)a];
        if (!R_FINITE(vdiag) || vdiag <= 0.0) {
            converged_flag = 0;
            status_str = "NOT_CONVERGED";
            reason_str = "Non-positive variance in terminal Schur Hessian";
            se_beh[a] = NA_REAL;
        } else {
            se_beh[a] = sqrt(vdiag);
        }
    }
    if (!converged_flag && ls_rejects > 0) {
        reason_str = "Armijo line search stalled before stationarity";
    }
    return fhl_aft_hl11_pack_result(pp, qq, 1, beta, se_beh, v, &alpha, phi, iter_done, last_pdiff, converged_flag, status_str, reason_str);
}

/* General dense C HL(1,1) solver for 1-RE random slopes (qq2 == 0) and all 2-RE models (qq2 > 0) */
static double fhl_hl11_dense_eval_h(
    int nn, int qq1, int qq2, const double *y, const double *z, const double *delta,
    const double *mu_base, const double *v, double phi, const double *alpha, double *mu_buf
) {
    const double two_pi = 6.283185307179586476925286766559;
    int qq = qq1 + qq2, inc1 = 1;
    double one = 1.0;
    memcpy(mu_buf, mu_base, (size_t)nn * sizeof(double));
    F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v, &inc1, &one, mu_buf, &inc1 FCONE);
    double inv_sqrt_phi = 1.0 / sqrt(phi), c_norm = -0.5 * log(two_pi * phi);
    double l1 = 0.0;
    for (int r = 0; r < nn; r++) {
        double mr = (y[r] - mu_buf[r]) * inv_sqrt_phi;
        if (delta[r] > 0.5) l1 += c_norm - 0.5 * mr * mr;
        else l1 += Rf_pnorm5(mr, 0.0, 1.0, 0, 1);
    }
    double ss1 = 0.0;
    for (int j = 0; j < qq1; j++) ss1 += v[j] * v[j];
    double l2 = -0.5 * (double)qq1 * log(two_pi * alpha[0]) - 0.5 * ss1 / alpha[0];
    if (qq2 > 0) {
        double ss2 = 0.0;
        for (int j = qq1; j < qq; j++) ss2 += v[j] * v[j];
        l2 += -0.5 * (double)qq2 * log(two_pi * alpha[1]) - 0.5 * ss2 / alpha[1];
    }
    return l1 + l2;
}

static int fhl_hl11_dense_eval_pv(
    int nn, int qq1, int qq2, const double *y, const double *z, const double *delta,
    const double *mu_base, const double *v, double phi, const double *alpha,
    double *mu_buf, double *wz_buf, double *M_mat, double *pv_out
) {
    const double two_pi = 6.283185307179586476925286766559;
    int qq = qq1 + qq2, inc1 = 1, info = 0;
    double one = 1.0, zero = 0.0;
    memcpy(mu_buf, mu_base, (size_t)nn * sizeof(double));
    F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v, &inc1, &one, mu_buf, &inc1 FCONE);
    double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi, c_norm = -0.5 * log(two_pi * phi);
    double l1 = 0.0;
    for (int r = 0; r < nn; r++) {
        double mr = (y[r] - mu_buf[r]) * inv_sqrt_phi;
        double vr = 0.0, xir = 0.0;
        fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
        double wr = delta[r] + (1.0 - delta[r]) * xir;
        if (delta[r] > 0.5) l1 += c_norm - 0.5 * mr * mr;
        else l1 += Rf_pnorm5(mr, 0.0, 1.0, 0, 1);
        mu_buf[r] = wr * inv_phi; /* reuse mu_buf as w / phi */
    }
    for (int c = 0; c < qq; c++) {
        double dhalf = (c < qq1) ? sqrt(alpha[0]) : sqrt(alpha[1]);
        const double *zcol = z + (size_t)c * (size_t)nn;
        double *wcol = wz_buf + (size_t)c * (size_t)nn;
        for (int r = 0; r < nn; r++) wcol[r] = (mu_buf[r] * dhalf) * zcol[r];
    }
    F77_CALL(dgemm)("T", "N", &qq, &qq, &nn, &one, z, &nn, wz_buf, &nn, &zero, M_mat, &qq FCONE FCONE);
    for (int r = 0; r < qq; r++) {
        double dhalf_r = (r < qq1) ? sqrt(alpha[0]) : sqrt(alpha[1]);
        for (int c = 0; c < qq; c++) M_mat[r + (size_t)qq * (size_t)c] *= dhalf_r;
        M_mat[r + (size_t)qq * (size_t)r] += 1.0;
    }
    F77_CALL(dpotrf)("U", &qq, M_mat, &qq, &info FCONE);
    if (info != 0) return 0;
    double sum_log_R = 0.0;
    for (int j = 0; j < qq; j++) {
        double rjj = M_mat[j + (size_t)qq * (size_t)j];
        if (!R_FINITE(rjj) || rjj <= 0.0) return 0;
        sum_log_R += log(rjj);
    }
    double quad_v = 0.0;
    for (int j = 0; j < qq1; j++) quad_v += (v[j] * v[j]) / alpha[0];
    for (int j = qq1; j < qq; j++) quad_v += (v[j] * v[j]) / alpha[1];
    double pv = l1 - 0.5 * quad_v - sum_log_R;
    if (!R_FINITE(pv)) return 0;
    *pv_out = pv;
    return 1;
}

static int fhl_hl11_dense_solve_v(
    int nn, int qq1, int qq2, const double *y, const double *z, const double *delta,
    const double *mu_base, double phi, const double *alpha,
    double *v, double tol_v, int maxit,
    double *mu_buf, double *a_buf, double *wz_buf, double *H_vv, double *gv, double *step_v, double *v_try
) {
    int qq = qq1 + qq2, inc1 = 1;
    double one = 1.0, zero = 0.0;
    double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi;

    for (int it = 1; it <= maxit; it++) {
        memcpy(mu_buf, mu_base, (size_t)nn * sizeof(double));
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v, &inc1, &one, mu_buf, &inc1 FCONE);
        for (int r = 0; r < nn; r++) {
            double mr = (y[r] - mu_buf[r]) * inv_sqrt_phi;
            double vr = 0.0, xir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
            a_buf[r] = (delta[r] * mr + (1.0 - delta[r]) * vr) * inv_sqrt_phi;
            mu_buf[r] = (delta[r] + (1.0 - delta[r]) * xir) * inv_phi;
        }
        F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, a_buf, &inc1, &zero, gv, &inc1 FCONE);
        double max_gv = 0.0;
        for (int j = 0; j < qq; j++) {
            double inv_aj = (j < qq1) ? (1.0 / alpha[0]) : (1.0 / alpha[1]);
            gv[j] -= inv_aj * v[j];
            if (!R_FINITE(gv[j])) return 0;
            if (fabs(gv[j]) > max_gv) max_gv = fabs(gv[j]);
        }
        if (max_gv < tol_v) return 1;

        for (int c = 0; c < qq; c++) {
            const double *zcol = z + (size_t)c * (size_t)nn;
            double *wcol = wz_buf + (size_t)c * (size_t)nn;
            for (int r = 0; r < nn; r++) wcol[r] = mu_buf[r] * zcol[r];
        }
        F77_CALL(dgemm)("T", "N", &qq, &qq, &nn, &one, z, &nn, wz_buf, &nn, &zero, H_vv, &qq FCONE FCONE);
        for (int j = 0; j < qq; j++) {
            H_vv[j + (size_t)qq * (size_t)j] += (j < qq1) ? (1.0 / alpha[0]) : (1.0 / alpha[1]);
        }
        if (!fhl_c_solve_spd(H_vv, gv, step_v, qq, 1)) return 0;

        double h_cur = fhl_hl11_dense_eval_h(nn, qq1, qq2, y, z, delta, mu_base, v, phi, alpha, mu_buf);
        double lam = 1.0;
        int moved = 0;
        while (lam > 1e-12) {
            for (int j = 0; j < qq; j++) v_try[j] = v[j] + lam * step_v[j];
            double h_try = fhl_hl11_dense_eval_h(nn, qq1, qq2, y, z, delta, mu_base, v_try, phi, alpha, mu_buf);
            if (R_FINITE(h_try) && h_try >= h_cur - 1e-13 * fmax(1.0, fabs(h_cur))) {
                memcpy(v, v_try, (size_t)qq * sizeof(double));
                moved = 1;
                break;
            }
            lam *= 0.5;
        }
        if (!moved) {
            memcpy(mu_buf, mu_base, (size_t)nn * sizeof(double));
            F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v, &inc1, &one, mu_buf, &inc1 FCONE);
            for (int r = 0; r < nn; r++) {
                double mr = (y[r] - mu_buf[r]) * inv_sqrt_phi;
                double vr = 0.0, xir = 0.0;
                fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
                a_buf[r] = (delta[r] * mr + (1.0 - delta[r]) * vr) * inv_sqrt_phi;
            }
            F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, a_buf, &inc1, &zero, gv, &inc1 FCONE);
            max_gv = 0.0;
            for (int j = 0; j < qq; j++) {
                double inv_aj = (j < qq1) ? (1.0 / alpha[0]) : (1.0 / alpha[1]);
                gv[j] -= inv_aj * v[j];
                if (fabs(gv[j]) > max_gv) max_gv = fabs(gv[j]);
            }
            return (max_gv <= 1e-7) ? 1 : 0;
        }
    }
    return 1;
}

static SEXP fhl_aft_dense_hl11(
    int nn, int pp, int qq1, int qq2,
    const double *y, const double *x, const double *z, const double *delta, int maxiter
) {
    const double tol_param = 1e-6, tol_gv = 1e-7, tol_gpv = 1e-5, tol_reml_sc = 1e-5;
    const double min_alpha = 1e-6, min_phi = 1e-8;
    int nrand = (qq2 > 0) ? 2 : 1;
    int qq = qq1 + qq2, r_tot = pp + qq, inc1 = 1;
    double one = 1.0, zero = 0.0, neg_one = -1.0;

    double *B_xz = (double *)R_alloc((size_t)nn * (size_t)r_tot, sizeof(double));
    memcpy(B_xz, x, (size_t)nn * (size_t)pp * sizeof(double));
    memcpy(B_xz + (size_t)nn * (size_t)pp, z, (size_t)nn * (size_t)qq * sizeof(double));

    double *beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *beta_new = (double *)R_alloc((size_t)pp, sizeof(double));
    double *b_try = (double *)R_alloc((size_t)pp, sizeof(double));
    double *dbeta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *ndec3 = (double *)R_alloc((size_t)pp, sizeof(double));
    double *g_beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *grad_pv = (double *)R_alloc((size_t)pp, sizeof(double));
    double *se_beh = (double *)R_alloc((size_t)pp, sizeof(double));
    memset(beta, 0, (size_t)pp * sizeof(double));

    double *v = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_new = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_after = (double *)R_alloc((size_t)qq, sizeof(double));
    double *vt_try = (double *)R_alloc((size_t)qq, sizeof(double));
    double *gv = (double *)R_alloc((size_t)qq, sizeof(double));
    double *step_v = (double *)R_alloc((size_t)qq, sizeof(double));
    double *v_try_inner = (double *)R_alloc((size_t)qq, sizeof(double));
    memset(v, 0, (size_t)qq * sizeof(double));

    double *mu_base = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mu_try = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mu_buf = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *a_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *xip_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *resid_star = (double *)R_alloc((size_t)nn, sizeof(double));
    double *c_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *dw_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *z_dva = (double *)R_alloc((size_t)nn, sizeof(double));

    double *W_B = (double *)R_alloc((size_t)nn * (size_t)r_tot, sizeof(double));
    double *H_full = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *H_tmp = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *Hinv = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *H_vv = (double *)R_alloc((size_t)qq * (size_t)qq, sizeof(double));
    double *Hvv_inv = (double *)R_alloc((size_t)qq * (size_t)qq, sizeof(double));
    double *H_bv_T = (double *)R_alloc((size_t)qq * (size_t)pp, sizeof(double));
    double *dv_db = (double *)R_alloc((size_t)qq * (size_t)pp, sizeof(double));
    double *Z_Hvv_inv = (double *)R_alloc((size_t)nn * (size_t)qq, sizeof(double));
    double *dva = (double *)R_alloc((size_t)qq, sizeof(double));
    double *term_q = (double *)R_alloc((size_t)qq, sizeof(double));

    double alpha[2] = {0.05, 0.1}, alpha_new[2] = {0.05, 0.1};
    double phi = 1.0, pv_prev = NA_REAL, last_pdiff = R_PosInf;
    int converged_flag = 0, ls_rejects = 0, iter_done = 0;
    const char *status_str = "NOT_CONVERGED";
    const char *reason_str = "Did not converge within Maxiter";

    /* Helper macro-like lambda via inline block for state + Hinv + Hvv_inv + grad_pv */
    #define EVAL_DENSE_STATE(BETA_P, V_P, PHI_V, ALPHA_P, MAX_GV_OUT, OK_OUT) do { \
        OK_OUT = 1; \
        double sq_p = sqrt(PHI_V), isq_p = 1.0 / sq_p, ip_v = 1.0 / (PHI_V); \
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, (BETA_P), &inc1, &zero, mu_base, &inc1 FCONE); \
        memcpy(mu_buf, mu_base, (size_t)nn * sizeof(double)); \
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, (V_P), &inc1, &one, mu_buf, &inc1 FCONE); \
        for (int r = 0; r < nn; r++) { \
            double mr = (y[r] - mu_buf[r]) * isq_p; \
            double vr = 0.0, xir = 0.0, dxir = 0.0; \
            fhl_mlm_mills_jet(mr, &vr, &xir, &dxir); \
            mij[r] = mr; \
            a_vec[r] = delta[r] * mr + (1.0 - delta[r]) * vr; \
            w_vec[r] = delta[r] + (1.0 - delta[r]) * xir; \
            xip_vec[r] = dxir; \
            resid_star[r] = delta[r] * (y[r] - mu_buf[r]) + (1.0 - delta[r]) * (sq_p * vr); \
            dw_vec[r] = w_vec[r] * ip_v; \
        } \
        F77_CALL(dgemv)("T", &nn, &pp, &isq_p, x, &nn, a_vec, &inc1, &zero, g_beta, &inc1 FCONE); \
        F77_CALL(dgemv)("T", &nn, &qq, &isq_p, z, &nn, a_vec, &inc1, &zero, gv, &inc1 FCONE); \
        (MAX_GV_OUT) = 0.0; \
        for (int j = 0; j < qq; j++) { \
            double iaj = (j < qq1) ? (1.0 / (ALPHA_P)[0]) : (1.0 / (ALPHA_P)[1]); \
            gv[j] -= iaj * (V_P)[j]; \
            if (fabs(gv[j]) > (MAX_GV_OUT)) (MAX_GV_OUT) = fabs(gv[j]); \
        } \
        for (int c = 0; c < r_tot; c++) { \
            const double *bcol = B_xz + (size_t)c * (size_t)nn; \
            double *wcol = W_B + (size_t)c * (size_t)nn; \
            for (int r = 0; r < nn; r++) wcol[r] = dw_vec[r] * bcol[r]; \
        } \
        F77_CALL(dgemm)("T", "N", &r_tot, &r_tot, &nn, &one, B_xz, &nn, W_B, &nn, &zero, H_full, &r_tot FCONE FCONE); \
        for (int j = 0; j < qq; j++) { \
            H_full[(pp + j) + (size_t)r_tot * (size_t)(pp + j)] += (j < qq1) ? (1.0 / (ALPHA_P)[0]) : (1.0 / (ALPHA_P)[1]); \
        } \
        for (int c = 0; c < qq; c++) { \
            for (int r = 0; r < qq; r++) H_vv[r + (size_t)qq * (size_t)c] = H_full[(pp + r) + (size_t)r_tot * (size_t)(pp + c)]; \
        } \
        for (int a = 0; a < pp; a++) { \
            for (int j = 0; j < qq; j++) H_bv_T[j + (size_t)qq * (size_t)a] = H_full[a + (size_t)r_tot * (size_t)(pp + j)]; \
        } \
        if (!fhl_c_solve_inv(H_vv, Hvv_inv, qq) || !fhl_c_solve_inv(H_full, Hinv, r_tot)) { \
            OK_OUT = 0; \
        } else { \
            F77_CALL(dgemm)("N", "N", &qq, &pp, &qq, &neg_one, Hvv_inv, &qq, H_bv_T, &qq, &zero, dv_db, &qq FCONE FCONE); \
            F77_CALL(dgemm)("N", "N", &nn, &qq, &qq, &ip_v, z, &nn, Hvv_inv, &qq, &zero, Z_Hvv_inv, &nn FCONE FCONE); \
            for (int r = 0; r < nn; r++) { \
                double pzz_r = 0.0; \
                for (int j = 0; j < qq; j++) pzz_r += Z_Hvv_inv[r + (size_t)nn * (size_t)j] * z[r + (size_t)nn * (size_t)j]; \
                c_vec[r] = -pzz_r * ((1.0 - delta[r]) * isq_p) * xip_vec[r]; \
            } \
            F77_CALL(dgemv)("T", &nn, &pp, &one, x, &nn, c_vec, &inc1, &zero, grad_pv, &inc1 FCONE); \
            F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, c_vec, &inc1, &zero, term_q, &inc1 FCONE); \
            F77_CALL(dgemv)("T", &qq, &pp, &one, dv_db, &qq, term_q, &inc1, &one, grad_pv, &inc1 FCONE); \
            for (int a = 0; a < pp; a++) grad_pv[a] = g_beta[a] - 0.5 * grad_pv[a]; \
        } \
    } while (0)

    #define EVAL_DENSE_REML(V_P, PHI_V, ALPHA_P, A_NEW_ARR, A_UNC_ARR, SC_A_ARR, REL_A_ARR, DEN_A_ARR, P_NEW_V, P_UNC_V, SC_P_V, REL_P_V, DEN_P_V) do { \
        double isq_p = 1.0 / sqrt(PHI_V), ip_v = 1.0 / (PHI_V), ip2_v = ip_v * ip_v; \
        int q_starts[2] = {0, qq1}, q_lens[2] = {qq1, qq2}; \
        for (int rc = 0; rc < nrand; rc++) { \
            int qs0 = q_starts[rc], qlen = q_lens[rc]; \
            double ar_cur = (ALPHA_P)[rc], inv_a2 = 1.0 / (ar_cur * ar_cur); \
            /* dva = (1 / alpha_r^2) * Hvv_inv[, qs0:(qs0+qlen-1)] * v_r (since iW = Hvv_inv / phi => -phi * iW * (-v/a^2) = Hvv_inv * v / a^2) */ \
            F77_CALL(dgemv)("N", &qq, &qlen, &inv_a2, Hvv_inv + (size_t)qq * (size_t)qs0, &qq, (V_P) + qs0, &inc1, &zero, dva, &inc1 FCONE); \
            F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, &inc1, &zero, z_dva, &inc1 FCONE); \
            for (int r = 0; r < nn; r++) dw_vec[r] = -ip_v * isq_p * (1.0 - delta[r]) * z_dva[r] * xip_vec[r]; \
            double tr_r = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, Hinv, W_B, H_tmp); \
            for (int j = qs0; j < qs0 + qlen; j++) tr_r -= inv_a2 * Hinv[(pp + j) + (size_t)r_tot * (size_t)(pp + j)]; \
            double gam_r = -ar_cur * tr_r; \
            double ss_vr = 0.0; \
            for (int j = qs0; j < qs0 + qlen; j++) ss_vr += (V_P)[j] * (V_P)[j]; \
            double den_r = (double)qlen - gam_r; \
            double sca = ss_vr / ar_cur - den_r; \
            (DEN_A_ARR)[rc] = den_r; (SC_A_ARR)[rc] = sca; \
            if (R_FINITE(den_r) && den_r > 1e-12) { \
                (A_UNC_ARR)[rc] = ss_vr / den_r; (REL_A_ARR)[rc] = sca / den_r; (A_NEW_ARR)[rc] = fmax(min_alpha, (A_UNC_ARR)[rc]); \
            } else { \
                (A_UNC_ARR)[rc] = NA_REAL; (REL_A_ARR)[rc] = NA_REAL; (A_NEW_ARR)[rc] = min_alpha; \
            } \
        } \
        double sum_w1 = 0.0, rss = 0.0; \
        for (int r = 0; r < nn; r++) { \
            dw_vec[r] = a_vec[r] + w_vec[r] * mij[r]; \
            sum_w1 += w_vec[r]; \
            rss += resid_star[r] * resid_star[r]; \
        } \
        F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, dw_vec, &inc1, &zero, term_q, &inc1 FCONE); \
        /* dvp = -0.5 / sqrt(phi) * iW * term_q = -0.5 / (phi * sqrt(phi)) * Hvv_inv * term_q */ \
        double sc_dvp = -0.5 * ip_v * isq_p; \
        F77_CALL(dgemv)("N", &qq, &qq, &sc_dvp, Hvv_inv, &qq, term_q, &inc1, &zero, dva, &inc1 FCONE); \
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, &inc1, &zero, z_dva, &inc1 FCONE); \
        for (int r = 0; r < nn; r++) { \
            double dp_m = -0.5 * mij[r] * ip_v - isq_p * z_dva[r]; \
            dw_vec[r] = -w_vec[r] * ip2_v + ip_v * (1.0 - delta[r]) * dp_m * xip_vec[r]; \
        } \
        double tr_p = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, Hinv, W_B, H_tmp); \
        double gam0 = (PHI_V) * tr_p; \
        (DEN_P_V) = sum_w1 + gam0; \
        (SC_P_V) = rss / (PHI_V) - (DEN_P_V); \
        if (R_FINITE(DEN_P_V) && (DEN_P_V) > 1e-12) { \
            (P_UNC_V) = rss / (DEN_P_V); (REL_P_V) = (SC_P_V) / (DEN_P_V); (P_NEW_V) = fmax(min_phi, (P_UNC_V)); \
        } else { \
            (P_UNC_V) = NA_REAL; (REL_P_V) = NA_REAL; (P_NEW_V) = fmax(min_phi, (PHI_V)); \
        } \
    } while (0)

    for (int it = 1; it <= maxiter; it++) {
        iter_done = it;
        for (int r = 0; r < nrand; r++) if (alpha[r] < min_alpha) alpha[r] = min_alpha;
        if (phi < min_phi) phi = min_phi;

        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta, &inc1, &zero, mu_base, &inc1 FCONE);
        if (!fhl_hl11_dense_solve_v(nn, qq1, qq2, y, z, delta, mu_base, phi, alpha, v, 1e-11, 50, mu_buf, a_vec, Z_Hvv_inv, H_vv, gv, step_v, v_try_inner)) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Inner v solver failed");
        }

        double max_gv = 0.0; int st_ok = 0;
        EVAL_DENSE_STATE(beta, v, phi, alpha, max_gv, st_ok);
        if (!st_ok) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Cholesky failed on H or H_vv");
        }

        for (int a = 0; a < pp; a++) {
            double acc = 0.0;
            for (int b = 0; b < pp; b++) acc += Hinv[a + (size_t)r_tot * (size_t)b] * grad_pv[b];
            dbeta[a] = acc;
        }
        double pv_cur = 0.0;
        if (!fhl_hl11_dense_eval_pv(nn, qq1, qq2, y, z, delta, mu_base, v, phi, alpha, mu_buf, Z_Hvv_inv, H_vv, &pv_cur)) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Cholesky of M failed evaluating p_v");
        }

        double max_gpv = 0.0, slope = 0.0;
        for (int a = 0; a < pp; a++) {
            if (fabs(grad_pv[a]) > max_gpv) max_gpv = fabs(grad_pv[a]);
            slope += grad_pv[a] * dbeta[a];
        }

        memcpy(beta_new, beta, (size_t)pp * sizeof(double));
        memcpy(v_new, v, (size_t)qq * sizeof(double));
        int accepted = 0;
        if (max_gpv == 0.0) {
            accepted = 1;
        } else if (!R_FINITE(slope) || slope <= 0.0) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, it - 1, last_pdiff, 0, "NOT_CONVERGED", "Non-ascent Schur direction");
        } else {
            double lam = 1.0;
            while (lam > 1e-10) {
                for (int a = 0; a < pp; a++) b_try[a] = beta[a] + lam * dbeta[a];
                F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, b_try, &inc1, &zero, mu_try, &inc1 FCONE);
                memcpy(vt_try, v, (size_t)qq * sizeof(double));
                if (fhl_hl11_dense_solve_v(nn, qq1, qq2, y, z, delta, mu_try, phi, alpha, vt_try, 1e-11, 50, mu_buf, a_vec, Z_Hvv_inv, H_vv, gv, step_v, v_try_inner)) {
                    double pv_try = 0.0;
                    if (fhl_hl11_dense_eval_pv(nn, qq1, qq2, y, z, delta, mu_try, vt_try, phi, alpha, mu_buf, Z_Hvv_inv, H_vv, &pv_try) &&
                        pv_try >= pv_cur + 1e-4 * lam * slope) {
                        memcpy(beta_new, b_try, (size_t)pp * sizeof(double));
                        memcpy(v_new, vt_try, (size_t)qq * sizeof(double));
                        accepted = 1;
                        break;
                    }
                }
                lam *= 0.5;
            }
            if (!accepted) ls_rejects++;
        }

        EVAL_DENSE_STATE(beta_new, v_new, phi, alpha, max_gv, st_ok);
        if (!st_ok) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta_new, NULL, v_new, alpha, phi, it, last_pdiff, 0, "NOT_CONVERGED", "Cholesky failed before REML step");
        }
        double a_unc[2], sc_a[2], rel_a[2], den_a[2];
        double phi_new = phi, phi_unc = NA_REAL, sc_p = 0.0, rel_p = 0.0, den_p = 0.0;
        EVAL_DENSE_REML(v_new, phi, alpha, alpha_new, a_unc, sc_a, rel_a, den_a, phi_new, phi_unc, sc_p, rel_p, den_p);
        (void)a_unc; (void)sc_a; (void)rel_a; (void)den_a; (void)sc_p; (void)rel_p; (void)den_p;

        memcpy(v_after, v_new, (size_t)qq * sizeof(double));
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_new, &inc1, &zero, mu_base, &inc1 FCONE);
        if (!fhl_hl11_dense_solve_v(nn, qq1, qq2, y, z, delta, mu_base, phi_new, alpha_new, v_after, 1e-11, 50, mu_buf, a_vec, Z_Hvv_inv, H_vv, gv, step_v, v_try_inner)) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta_new, NULL, v_new, alpha_new, phi_new, it, last_pdiff, 0, "NOT_CONVERGED", "Inner v solve failed after REML update");
        }

        double pdiff = fabs(phi_new - phi);
        for (int r = 0; r < nrand; r++) pdiff += fabs(alpha_new[r] - alpha[r]);
        for (int a = 0; a < pp; a++) pdiff += fabs(beta_new[a] - beta[a]);
        for (int j = 0; j < qq; j++) pdiff += fabs(v_after[j] - v[j]);

        double max_gv3 = 0.0;
        EVAL_DENSE_STATE(beta_new, v_after, phi_new, alpha_new, max_gv3, st_ok);
        if (!st_ok) {
            return fhl_aft_hl11_pack_result(pp, qq, nrand, beta_new, NULL, v_after, alpha_new, phi_new, it, pdiff, 0, "NOT_CONVERGED", "Cholesky failed at post-update check");
        }
        double max_ndec = 0.0, max_gpv3 = 0.0;
        for (int a = 0; a < pp; a++) {
            double acc = 0.0;
            for (int b = 0; b < pp; b++) acc += Hinv[a + (size_t)r_tot * (size_t)b] * grad_pv[b];
            ndec3[a] = acc;
            if (fabs(acc) > max_ndec) max_ndec = fabs(acc);
            if (fabs(grad_pv[a]) > max_gpv3) max_gpv3 = fabs(grad_pv[a]);
        }

        double pv3 = 0.0;
        int pv3_ok = fhl_hl11_dense_eval_pv(nn, qq1, qq2, y, z, delta, mu_base, v_after, phi_new, alpha_new, mu_buf, Z_Hvv_inv, H_vv, &pv3);
        double odiff = (!ISNAN(pv_prev) && pv3_ok) ? fabs(pv3 - pv_prev) : R_PosInf;

        double a_new3[2], a_unc3[2], sc_a3[2], rel_a3[2], den_a3[2];
        double p_new3 = phi_new, p_unc3 = NA_REAL, sc_p3 = 0.0, rel_p3 = 0.0, den_p3 = 0.0;
        EVAL_DENSE_REML(v_after, phi_new, alpha_new, a_new3, a_unc3, sc_a3, rel_a3, den_a3, p_new3, p_unc3, sc_p3, rel_p3, den_p3);
        (void)a_new3; (void)a_unc3; (void)p_new3; (void)p_unc3;

        int any_a_bound = 0, all_a_ok = 1;
        for (int r = 0; r < nrand; r++) {
            int ab = (alpha_new[r] <= min_alpha + 1e-12);
            if (ab) any_a_bound = 1;
            int ok_r = 0;
            if (R_FINITE(den_a3[r]) && den_a3[r] > 1e-12 && R_FINITE(sc_a3[r]) && R_FINITE(rel_a3[r])) {
                ok_r = ab ? ((sc_a3[r] <= 1e-13) && (rel_a3[r] <= tol_reml_sc))
                          : ((fabs(sc_a3[r]) <= tol_reml_sc) && (fabs(rel_a3[r]) <= tol_reml_sc));
            }
            if (!ok_r) all_a_ok = 0;
        }
        int phi_bound = (phi_new <= min_phi + 1e-14) || (!ISNAN(phi_unc) && phi_unc <= min_phi);
        int phi_ok = (!phi_bound) && R_FINITE(den_p3) && (den_p3 > 1e-12) &&
                     R_FINITE(sc_p3) && R_FINITE(rel_p3) &&
                     (fabs(sc_p3) <= tol_reml_sc) && (fabs(rel_p3) <= tol_reml_sc);

        memcpy(beta, beta_new, (size_t)pp * sizeof(double));
        memcpy(v, v_after, (size_t)qq * sizeof(double));
        for (int r = 0; r < nrand; r++) alpha[r] = alpha_new[r];
        phi = phi_new;
        last_pdiff = pdiff;
        if (pv3_ok) pv_prev = pv3;

        if (accepted && R_FINITE(pdiff) && pdiff <= tol_param &&
            R_FINITE(odiff) && odiff <= fmax(1e-6, 1e-8 * (1.0 + fabs(pv_prev))) &&
            (max_gv3 <= tol_gv) && (max_ndec <= tol_param) && (max_gpv3 <= tol_gpv) &&
            all_a_ok && phi_ok) {
            converged_flag = 1;
            status_str = any_a_bound ? "BOUNDARY_SOLUTION" : "CONVERGED_INTERIOR";
            reason_str = any_a_bound ? "Converged at lower variance-component boundary" : "Converged in interior";
            break;
        }
    }

    F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta, &inc1, &zero, mu_base, &inc1 FCONE);
    if (!fhl_hl11_dense_solve_v(nn, qq1, qq2, y, z, delta, mu_base, phi, alpha, v, 1e-12, 50, mu_buf, a_vec, Z_Hvv_inv, H_vv, gv, step_v, v_try_inner)) {
        return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, iter_done, last_pdiff, 0, "NOT_CONVERGED", "Final v_tilde solve failed");
    }
    double max_gv_f = 0.0; int st_ok_f = 0;
    EVAL_DENSE_STATE(beta, v, phi, alpha, max_gv_f, st_ok_f);
    if (!st_ok_f) {
        return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, NULL, v, alpha, phi, iter_done, last_pdiff, 0, "NOT_CONVERGED", "Final Hessian Cholesky failed");
    }
    for (int a = 0; a < pp; a++) {
        double vdiag = Hinv[a + (size_t)r_tot * (size_t)a];
        if (!R_FINITE(vdiag) || vdiag <= 0.0) {
            converged_flag = 0;
            status_str = "NOT_CONVERGED";
            reason_str = "Non-positive variance in terminal Hessian";
            se_beh[a] = NA_REAL;
        } else {
            se_beh[a] = sqrt(vdiag);
        }
    }
    #undef EVAL_DENSE_STATE
    #undef EVAL_DENSE_REML
    return fhl_aft_hl11_pack_result(pp, qq, nrand, beta, se_beh, v, alpha, phi, iter_done, last_pdiff, converged_flag, status_str, reason_str);
}

/* AFT 1-RE general random-effects solver (explicit Z matrix, e.g. random slopes) */
static SEXP fhl_aft_1re_general(SEXP y_s, SEXP x_s, SEXP z_s, int qq_expected, SEXP delta_s,
                                SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s) {
    int nn = nrows(x_s), pp = ncols(x_s), qq = ncols(z_s);
    int r_tot = pp + qq;
    int maxiter = asInteger(maxiter_s);
    int mord = (TYPEOF(maxiter_s) == INTSXP && LENGTH(maxiter_s) >= 2) ? INTEGER(maxiter_s)[1] : 0;
    if (nn <= 0 || pp <= 0 || qq <= 0 || qq != qq_expected || nrows(z_s) != nn || length(y_s) != nn ||
        length(delta_s) != nn || length(beta_h0_s) != pp || length(v_h0_s) != qq) {
        Rf_error("Invalid dimensions in fhl_fit_aft_1re");
    }
    if (mord == 1) {
        return fhl_aft_dense_hl11(nn, pp, qq, 0, REAL(y_s), REAL(x_s), REAL(z_s), REAL(delta_s), maxiter);
    }

    const double *y = REAL(y_s);
    const double *x = REAL(x_s);
    const double *z = REAL(z_s);
    const double *delta = REAL(delta_s);

    double *B_xz = (double *)R_alloc((size_t)nn * r_tot, sizeof(double));
    memcpy(B_xz, x, (size_t)nn * pp * sizeof(double));
    memcpy(B_xz + (size_t)nn * pp, z, (size_t)nn * qq * sizeof(double));

    double *beta_h0 = (double *)R_alloc(pp, sizeof(double));
    double *old_beta = (double *)R_alloc(pp, sizeof(double));
    double *v_h0 = (double *)R_alloc(qq, sizeof(double));
    memcpy(beta_h0, REAL(beta_h0_s), (size_t)pp * sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double phi = 1.0, alpha = 0.05;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *muij = (double *)R_alloc(nn, sizeof(double));
    double *mij = (double *)R_alloc(nn, sizeof(double));
    double *V_mij = (double *)R_alloc(nn, sizeof(double));
    double *xi_mij = (double *)R_alloc(nn, sizeof(double));
    double *weight1 = (double *)R_alloc(nn, sizeof(double));
    double *ystar = (double *)R_alloc(nn, sizeof(double));
    double *w_adj_yy = (double *)R_alloc(nn, sizeof(double));
    double *w1_vec = (double *)R_alloc(nn, sizeof(double));
    double *W_B = (double *)R_alloc((size_t)nn * r_tot, sizeof(double));
    double *H_mlm = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *hinv = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *rhs_mlm = (double *)R_alloc(r_tot, sizeof(double));
    double *beta_vv = (double *)R_alloc(r_tot, sizeof(double));
    double *G_zz_we = (double *)R_alloc((size_t)qq * qq, sizeof(double));
    double *iW = (double *)R_alloc((size_t)qq * qq, sizeof(double));
    double *dva = (double *)R_alloc(qq, sizeof(double));
    double *z_dva = (double *)R_alloc(nn, sizeof(double));
    double *da_weight2 = (double *)R_alloc(nn, sizeof(double));
    double *dw_vec = (double *)R_alloc(nn, sizeof(double));
    double *term_q = (double *)R_alloc(qq, sizeof(double));

    double one = 1.0, zero = 0.0;

    while (convergence > 0.000001 && iteration <= maxiter) {
        iteration++;

        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, & (int){1}, &zero, muij, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v_h0, & (int){1}, &one, muij, & (int){1} FCONE);

        double sqrt_phi = sqrt(phi);
        double inv_sqrt_phi = 1.0 / sqrt_phi;
        double inv_phi = 1.0 / phi;

        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0, dxir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, &dxir);
            double w1r = delta[r] + (1.0 - delta[r]) * xir;
            double resid_star = delta[r] * (y[r] - muij[r]) + (1.0 - delta[r]) * (sqrt_phi * vr);
            double w1_scaled = w1r * inv_phi;
            mij[r] = mr;
            V_mij[r] = vr;
            xi_mij[r] = xir;
            weight1[r] = w1r;
            ystar[r] = resid_star;
            da_weight2[r] = dxir;
            w1_vec[r] = w1_scaled;
            w_adj_yy[r] = w1_scaled * muij[r] + resid_star * inv_phi;
        }

        for (int c = 0; c < r_tot; c++) {
            const double *bcol = B_xz + (size_t)c * nn;
            double *wcol = W_B + (size_t)c * nn;
            for (int r = 0; r < nn; r++) wcol[r] = w1_vec[r] * bcol[r];
        }
        F77_CALL(dgemm)("T", "N", &r_tot, &r_tot, &nn, &one, B_xz, &nn, W_B, &nn, &zero, H_mlm, &r_tot FCONE FCONE);

        /* Save unpenalized ZZ block scaled by phi for G_zz_we = z^T diag(weight1) z */
        for (int c = 0; c < qq; c++) {
            for (int r = 0; r < qq; r++) {
                G_zz_we[r + (size_t)qq * c] = H_mlm[(pp + r) + (size_t)r_tot * (pp + c)] * phi;
            }
        }

        double inv_alpha = 1.0 / alpha;
        for (int j = 0; j < qq; j++) {
            H_mlm[(pp + j) + (size_t)r_tot * (pp + j)] += inv_alpha;
        }
        if (!fhl_c_solve_inv(H_mlm, hinv, r_tot)) {
            Rf_error("Singular matrix in mlmfit1 (H_mlm) (not positive definite)");
        }

        memcpy(old_beta, beta_h0, (size_t)pp * sizeof(double));
        F77_CALL(dgemv)("T", &nn, &r_tot, &one, B_xz, &nn, w_adj_yy, & (int){1}, &zero, rhs_mlm, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, hinv, &r_tot, rhs_mlm, & (int){1}, &zero, beta_vv, & (int){1} FCONE);
        memcpy(beta_h0, beta_vv, (size_t)pp * sizeof(double));
        memcpy(v_h0, beta_vv + pp, (size_t)qq * sizeof(double));

        /* alpha_h update */
        double old_alpha = alpha;
        double lam = phi / alpha;
        for (int j = 0; j < qq; j++) G_zz_we[j + (size_t)qq * j] += lam;
        if (!fhl_c_solve_inv(G_zz_we, iW, qq)) {
            Rf_error("Singular matrix in mlmfit1 (G_zz_we) (not positive definite)");
        }

        double scale_dva = phi / (alpha * alpha);
        F77_CALL(dgemv)("N", &qq, &qq, &scale_dva, iW, &qq, v_h0, & (int){1}, &zero, dva, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, & (int){1}, &zero, z_dva, & (int){1} FCONE);

        for (int r = 0; r < nn; r++) {
            double da_w1 = -inv_sqrt_phi * z_dva[r];
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1 * da_weight2[r]);
        }
        double tr_H_star2 = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, hinv, W_B, H_mlm);
        double inv_alpha2 = 1.0 / (alpha * alpha);
        for (int j = 0; j < qq; j++) {
            tr_H_star2 -= inv_alpha2 * hinv[(pp + j) + (size_t)r_tot * (pp + j)];
        }
        double gamma1 = -alpha * tr_H_star2;
        double ss_v = 0.0;
        for (int j = 0; j < qq; j++) ss_v += v_h0[j] * v_h0[j];
        double denom_a = (double)qq - gamma1;
        fhl_aft_check_denom(denom_a, "alpha_h", "mlmfit1");
        alpha = ss_v / denom_a;
        fhl_aft_check_pos(alpha, "alpha_h", "mlmfit1");

        /* phi_h update */
        double old_phi = phi;
        for (int r = 0; r < nn; r++) {
            double a_r = delta[r] * mij[r] + (1.0 - delta[r]) * V_mij[r];
            dw_vec[r] = a_r + weight1[r] * mij[r];
        }
        F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, dw_vec, & (int){1}, &zero, term_q, & (int){1} FCONE);
        double scale_dvp = -0.5 * inv_sqrt_phi;
        F77_CALL(dgemv)("N", &qq, &qq, &scale_dvp, iW, &qq, term_q, & (int){1}, &zero, dva, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, & (int){1}, &zero, z_dva, & (int){1} FCONE);

        double inv_phi2 = inv_phi * inv_phi;
        double sum_w1 = 0.0, rss_y = 0.0;
        for (int r = 0; r < nn; r++) {
            double dp_w1 = -0.5 * mij[r] * inv_phi - inv_sqrt_phi * z_dva[r];
            double d_w_phi = inv_phi * (1.0 - delta[r]) * (dp_w1 * da_weight2[r]);
            dw_vec[r] = -weight1[r] * inv_phi2 + d_w_phi;
            sum_w1 += weight1[r];
            double diff = ystar[r];
            rss_y += diff * diff;
        }
        double tr_H_star = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, hinv, W_B, H_mlm);
        double gamma0 = phi * tr_H_star;
        double denom_p = sum_w1 + gamma0;
        fhl_aft_check_denom(denom_p, "phi_h", "mlmfit1");
        phi = rss_y / denom_p;
        fhl_aft_check_pos(phi, "phi_h", "mlmfit1");

        double diff_beta = 0.0;
        for (int j = 0; j < pp; j++) diff_beta += fabs(old_beta[j] - beta_h0[j]);
        convergence = diff_beta + fabs(old_alpha - alpha) + fabs(old_phi - phi);
        i_count++;
    }

    /* Evaluate H^{-1} at final updated (beta_h0, v_h0, alpha, phi) for SE */
    fhl_mlm_eval_dense_hinv(
        nn, pp, qq, qq, r_tot,
        y, x, z, B_xz, delta,
        beta_h0, v_h0, alpha, alpha, phi,
        muij, w1_vec, W_B, H_mlm, hinv
    );

    SEXP out = PROTECT(allocVector(VECSXP, 7));
    SEXP beta_h_s = PROTECT(allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_h0[j];
        REAL(se_beh_s)[j] = sqrt(hinv[j + (size_t)r_tot * j]);
    }
    SEXP v_out_s = PROTECT(allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, ScalarReal(alpha));
    SET_VECTOR_ELT(out, 4, ScalarReal(phi));
    SET_VECTOR_ELT(out, 5, ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 6, ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}

/* Compute exact tr(hinv * B_xz^T diag(w) B_xz) in O(N * p^2 + K * p)
 * using Schur-complement blocks without materializing the K x K hinv_ZZ matrix. */
static double fhl_mlm1_schur_weighted_trace(
    const double *x, const int *g, int nn, int pp, int qq,
    const double *w,
    const double *hinv_XX, const double *hinv_XZ, const double *hinv_ZZ_diag,
    double *xx_w, double *xz_w, double *zz_w, double *wx_buf
) {
    memset(xz_w, 0, (size_t)pp * (size_t)qq * sizeof(double));
    memset(zz_w, 0, (size_t)qq * sizeof(double));

    for (int r = 0; r < nn; r++) {
        zz_w[g[r]] += w[r];
    }
    for (int a = 0; a < pp; a++) {
        const double *xcol = x + (size_t)a * nn;
        double *wxcol = wx_buf + (size_t)a * nn;
        for (int r = 0; r < nn; r++) {
            double v = w[r] * xcol[r];
            wxcol[r] = v;
            xz_w[a + (size_t)pp * (size_t)g[r]] += v;
        }
    }

    double one = 1.0, zero = 0.0;
    F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);

    double tr = 0.0;
    for (size_t idx = 0; idx < (size_t)pp * (size_t)pp; idx++) {
        tr += hinv_XX[idx] * xx_w[idx];
    }
    for (size_t idx = 0; idx < (size_t)pp * (size_t)qq; idx++) {
        tr += 2.0 * hinv_XZ[idx] * xz_w[idx];
    }
    for (int j = 0; j < qq; j++) {
        tr += hinv_ZZ_diag[j] * zz_w[j];
    }
    return tr;
}

/* Unified AFT 1-RE entrypoint: dispatches O(q) cluster Schur solver for integer
 * cluster indices or general Z solver for random-slope matrices. */
SEXP fhl_fit_aft_1re(
    SEXP y_s, SEXP x_s, SEXP group_s, SEXP qq_s, SEXP delta_s,
    SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s
) {
    if (TYPEOF(y_s) != REALSXP || TYPEOF(x_s) != REALSXP || !Rf_isMatrix(x_s) ||
        TYPEOF(delta_s) != REALSXP || TYPEOF(beta_h0_s) != REALSXP || TYPEOF(v_h0_s) != REALSXP) {
        Rf_error("Invalid SEXP types in fhl_fit_aft_1re");
    }
    if (TYPEOF(group_s) == REALSXP && Rf_isMatrix(group_s)) {
        return fhl_aft_1re_general(y_s, x_s, group_s, asInteger(qq_s), delta_s, beta_h0_s, v_h0_s, maxiter_s);
    }
    if (TYPEOF(group_s) != INTSXP) {
        Rf_error("group_s must be an integer cluster vector or numeric Z matrix in fhl_fit_aft_1re");
    }
    int nn = nrows(x_s), pp = ncols(x_s), qq = asInteger(qq_s);
    int maxiter = asInteger(maxiter_s);
    if (nn <= 0 || pp <= 0 || qq <= 0 || length(y_s) != nn || length(group_s) != nn ||
        length(delta_s) != nn || length(beta_h0_s) != pp || length(v_h0_s) != qq) {
        Rf_error("Invalid dimensions in fhl_fit_aft_1re");
    }

    const double *y = REAL(y_s);
    const double *x = REAL(x_s);
    const int *g1 = INTEGER(group_s);
    const double *delta = REAL(delta_s);

    /* Validate and convert 1-based cluster indices to 0-based once */
    int *g = (int *)R_alloc((size_t)nn, sizeof(int));
    for (int r = 0; r < nn; r++) {
        int gi = g1[r] - 1;
        if (gi < 0 || gi >= qq) Rf_error("Invalid cluster index in fhl_fit_aft_1re");
        g[r] = gi;
    }
    int mord = (TYPEOF(maxiter_s) == INTSXP && LENGTH(maxiter_s) >= 2) ? INTEGER(maxiter_s)[1] : 0;
    if (mord == 1) {
        return fhl_aft_1re_cluster_hl11(nn, pp, qq, y, x, g, delta, maxiter);
    }

    double *beta_h0 = (double *)R_alloc((size_t)pp, sizeof(double));
    double *old_beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *v_h0 = (double *)R_alloc((size_t)qq, sizeof(double));
    memcpy(beta_h0, REAL(beta_h0_s), (size_t)pp * sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double phi = 1.0, alpha = 0.05;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *muij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *V_mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *xi_mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *weight1 = (double *)R_alloc((size_t)nn, sizeof(double));
    double *ystar = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w1_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w_adj_yy = (double *)R_alloc((size_t)nn, sizeof(double));
    double *da_weight2 = (double *)R_alloc((size_t)nn, sizeof(double));
    double *dw_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *wx_buf = (double *)R_alloc((size_t)nn * (size_t)pp, sizeof(double));

    double *xx_w = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *schur_S = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *hinv_XX = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    memset(hinv_XX, 0, (size_t)pp * (size_t)pp * sizeof(double));
    double *xz_w = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *K_xz = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *hinv_XZ = (double *)R_alloc((size_t)pp * (size_t)qq, sizeof(double));
    double *zz_w = (double *)R_alloc((size_t)qq, sizeof(double));
    double *inv_D = (double *)R_alloc((size_t)qq, sizeof(double));
    double *hinv_ZZ_diag = (double *)R_alloc((size_t)qq, sizeof(double));
    double *iW_diag = (double *)R_alloc((size_t)qq, sizeof(double));
    double *rhs_X = (double *)R_alloc((size_t)pp, sizeof(double));
    double *rhs_Z = (double *)R_alloc((size_t)qq, sizeof(double));
    double *rhs_schur = (double *)R_alloc((size_t)pp, sizeof(double));
    double *dva = (double *)R_alloc((size_t)qq, sizeof(double));
    double *term_q = (double *)R_alloc((size_t)qq, sizeof(double));

    double one = 1.0, zero = 0.0, neg_one = -1.0;

    while (convergence > 0.000001 && iteration <= maxiter) {
        iteration++;

        /* muij = X %*% beta_h0 + v_h0[g] */
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, &(int){1}, &zero, muij, &(int){1} FCONE);
        for (int r = 0; r < nn; r++) {
            muij[r] += v_h0[g[r]];
        }

        double sqrt_phi = sqrt(phi);
        double inv_sqrt_phi = 1.0 / sqrt_phi;
        double inv_phi = 1.0 / phi;

        memset(zz_w, 0, (size_t)qq * sizeof(double));
        memset(xz_w, 0, (size_t)pp * (size_t)qq * sizeof(double));
        memset(rhs_Z, 0, (size_t)qq * sizeof(double));

        double sum_w1 = 0.0, rss_y = 0.0;
        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0, dxir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, &dxir);
            double w1r = delta[r] + (1.0 - delta[r]) * xir;
            double resid_star = delta[r] * (y[r] - muij[r]) + (1.0 - delta[r]) * (sqrt_phi * vr);
            double w1_scaled = w1r * inv_phi;
            double w_adj_r = w1_scaled * muij[r] + resid_star * inv_phi;

            mij[r] = mr;
            V_mij[r] = vr;
            xi_mij[r] = xir;
            weight1[r] = w1r;
            ystar[r] = resid_star;
            w1_vec[r] = w1_scaled;
            w_adj_yy[r] = w_adj_r;
            da_weight2[r] = dxir;

            sum_w1 += w1r;
            rss_y += resid_star * resid_star;

            int gr = g[r];
            zz_w[gr] += w1_scaled;
            rhs_Z[gr] += w_adj_r;
        }

        for (int a = 0; a < pp; a++) {
            const double *xcol = x + (size_t)a * nn;
            double *wxcol = wx_buf + (size_t)a * nn;
            for (int r = 0; r < nn; r++) {
                double wxr = w1_vec[r] * xcol[r];
                wxcol[r] = wxr;
                xz_w[a + (size_t)pp * (size_t)g[r]] += wxr;
            }
        }
        F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);
        F77_CALL(dgemv)("T", &nn, &pp, &one, x, &nn, w_adj_yy, &(int){1}, &zero, rhs_X, &(int){1} FCONE);

        /* Schur complement S = xx_w - xz_w %*% D^{-1} %*% xz_w^T (pp x pp) */
        double inv_alpha = 1.0 / alpha;
        for (int j = 0; j < qq; j++) {
            double d_j = zz_w[j] + inv_alpha;
            if (!R_FINITE(d_j) || d_j <= 0.0) {
                Rf_error("mlmfit1: diagonal entry D[%d] non-positive or non-finite (%g)", j + 1, d_j);
            }
            inv_D[j] = 1.0 / d_j;
            double *kcol = K_xz + (size_t)pp * (size_t)j;
            const double *bcol = xz_w + (size_t)pp * (size_t)j;
            double d_inv = inv_D[j];
            for (int a = 0; a < pp; a++) {
                kcol[a] = bcol[a] * d_inv;
            }
        }
        memcpy(schur_S, xx_w, (size_t)pp * (size_t)pp * sizeof(double));
        F77_CALL(dgemm)("N", "T", &pp, &pp, &qq, &neg_one, xz_w, &pp, K_xz, &pp, &one, schur_S, &pp FCONE FCONE);

        if (!fhl_c_solve_inv(schur_S, hinv_XX, pp)) {
            Rf_error("Singular Schur complement in mlmfit1 (not positive definite)");
        }

        /* hinv_XZ = - hinv_XX %*% K_xz (pp x qq) */
        F77_CALL(dgemm)("N", "N", &pp, &qq, &pp, &neg_one, hinv_XX, &pp, K_xz, &pp, &zero, hinv_XZ, &pp FCONE FCONE);

        /* diag(hinv_ZZ)[j] = inv_D[j] - sum_a hinv_XZ[a,j] * K_xz[a,j] */
        double sum_hinv_ZZ_diag = 0.0;
        for (int j = 0; j < qq; j++) {
            const double *hcol = hinv_XZ + (size_t)pp * (size_t)j;
            const double *kcol = K_xz + (size_t)pp * (size_t)j;
            double quad = 0.0;
            for (int a = 0; a < pp; a++) quad += hcol[a] * kcol[a];
            double diag_j = inv_D[j] - quad;
            hinv_ZZ_diag[j] = diag_j;
            sum_hinv_ZZ_diag += diag_j;
        }

        /* Update beta_h0 and v_h0 via Schur back-substitution */
        memcpy(old_beta, beta_h0, (size_t)pp * sizeof(double));
        memcpy(rhs_schur, rhs_X, (size_t)pp * sizeof(double));
        F77_CALL(dgemv)("N", &pp, &qq, &neg_one, K_xz, &pp, rhs_Z, &(int){1}, &one, rhs_schur, &(int){1} FCONE);
        F77_CALL(dgemv)("N", &pp, &pp, &one, hinv_XX, &pp, rhs_schur, &(int){1}, &zero, beta_h0, &(int){1} FCONE);

        for (int j = 0; j < qq; j++) {
            const double *bcol = xz_w + (size_t)pp * (size_t)j;
            double bt_beta = 0.0;
            for (int a = 0; a < pp; a++) bt_beta += bcol[a] * beta_h0[a];
            v_h0[j] = inv_D[j] * (rhs_Z[j] - bt_beta);
        }

        /* alpha_h update */
        double old_alpha = alpha;
        double lam = phi / alpha;
        double scale_dva = phi / (alpha * alpha);
        for (int j = 0; j < qq; j++) {
            double w_j = zz_w[j] * phi + lam;
            if (!R_FINITE(w_j) || w_j <= 0.0) {
                Rf_error("mlmfit1: W_zz diagonal[%d] non-positive or non-finite (%g)", j + 1, w_j);
            }
            double iw_j = 1.0 / w_j;
            iW_diag[j] = iw_j;
            dva[j] = scale_dva * iw_j * v_h0[j];
        }

        for (int r = 0; r < nn; r++) {
            double da_w1 = -inv_sqrt_phi * dva[g[r]];
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1 * da_weight2[r]);
        }
        double tr_H_star2 = fhl_mlm1_schur_weighted_trace(
            x, g, nn, pp, qq, dw_vec, hinv_XX, hinv_XZ, hinv_ZZ_diag,
            xx_w, xz_w, zz_w, wx_buf
        );
        double inv_alpha2 = 1.0 / (alpha * alpha);
        tr_H_star2 -= inv_alpha2 * sum_hinv_ZZ_diag;
        double gamma1 = -alpha * tr_H_star2;
        double ss_v = 0.0;
        for (int j = 0; j < qq; j++) ss_v += v_h0[j] * v_h0[j];
        double denom_a = (double)qq - gamma1;
        fhl_aft_check_denom(denom_a, "alpha_h", "mlmfit1");
        alpha = ss_v / denom_a;
        fhl_aft_check_pos(alpha, "alpha_h", "mlmfit1");

        /* phi_h update */
        double old_phi = phi;
        memset(term_q, 0, (size_t)qq * sizeof(double));
        for (int r = 0; r < nn; r++) {
            double a_r = delta[r] * mij[r] + (1.0 - delta[r]) * V_mij[r];
            term_q[g[r]] += a_r + weight1[r] * mij[r];
        }
        double scale_dvp = -0.5 * inv_sqrt_phi;
        for (int j = 0; j < qq; j++) {
            dva[j] = scale_dvp * iW_diag[j] * term_q[j];
        }

        double inv_phi2 = inv_phi * inv_phi;
        for (int r = 0; r < nn; r++) {
            double dp_w1 = -0.5 * mij[r] * inv_phi - inv_sqrt_phi * dva[g[r]];
            double d_w_phi = inv_phi * (1.0 - delta[r]) * (dp_w1 * da_weight2[r]);
            dw_vec[r] = -weight1[r] * inv_phi2 + d_w_phi;
        }
        double tr_H_star = fhl_mlm1_schur_weighted_trace(
            x, g, nn, pp, qq, dw_vec, hinv_XX, hinv_XZ, hinv_ZZ_diag,
            xx_w, xz_w, zz_w, wx_buf
        );
        double gamma0 = phi * tr_H_star;
        double denom_p = sum_w1 + gamma0;
        fhl_aft_check_denom(denom_p, "phi_h", "mlmfit1");
        phi = rss_y / denom_p;
        fhl_aft_check_pos(phi, "phi_h", "mlmfit1");

        double diff_beta = 0.0;
        for (int j = 0; j < pp; j++) diff_beta += fabs(old_beta[j] - beta_h0[j]);
        convergence = diff_beta + fabs(old_alpha - alpha) + fabs(old_phi - phi);
        i_count++;
    }

    /* Evaluate Schur Hessian H^{-1}_XX at final updated (beta_h0, v_h0, alpha, phi) */
    {
        double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi, inv_alpha = 1.0 / alpha;
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, &(int){1}, &zero, muij, &(int){1} FCONE);
        for (int r = 0; r < nn; r++) muij[r] += v_h0[g[r]];
        memset(zz_w, 0, (size_t)qq * sizeof(double));
        memset(xz_w, 0, (size_t)pp * (size_t)qq * sizeof(double));
        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
            double w1_scaled = (delta[r] + (1.0 - delta[r]) * xir) * inv_phi;
            w1_vec[r] = w1_scaled;
            zz_w[g[r]] += w1_scaled;
        }
        for (int a = 0; a < pp; a++) {
            const double *xcol = x + (size_t)a * nn;
            double *wxcol = wx_buf + (size_t)a * nn;
            for (int r = 0; r < nn; r++) {
                double wxr = w1_vec[r] * xcol[r];
                wxcol[r] = wxr;
                xz_w[a + (size_t)pp * (size_t)g[r]] += wxr;
            }
        }
        F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);
        for (int j = 0; j < qq; j++) {
            double d_j = zz_w[j] + inv_alpha;
            if (!R_FINITE(d_j) || d_j <= 0.0) {
                Rf_error("mlmfit1: final SE diagonal entry D[%d] non-positive or non-finite (%g)", j + 1, d_j);
            }
            double d_inv = 1.0 / d_j;
            double *kcol = K_xz + (size_t)pp * (size_t)j;
            const double *bcol = xz_w + (size_t)pp * (size_t)j;
            for (int a = 0; a < pp; a++) kcol[a] = bcol[a] * d_inv;
        }
        memcpy(schur_S, xx_w, (size_t)pp * (size_t)pp * sizeof(double));
        F77_CALL(dgemm)("N", "T", &pp, &pp, &qq, &neg_one, xz_w, &pp, K_xz, &pp, &one, schur_S, &pp FCONE FCONE);
        if (!fhl_c_solve_inv(schur_S, hinv_XX, pp)) {
            Rf_error("Singular Schur complement in mlmfit1 final SE evaluation (not positive definite)");
        }
    }

    SEXP out = PROTECT(allocVector(VECSXP, 7));
    SEXP beta_h_s = PROTECT(allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_h0[j];
        REAL(se_beh_s)[j] = sqrt(hinv_XX[j + (size_t)pp * j]);
    }
    SEXP v_out_s = PROTECT(allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, ScalarReal(alpha));
    SET_VECTOR_ELT(out, 4, ScalarReal(phi));
    SET_VECTOR_ELT(out, 5, ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 6, ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}

/* AFT 2-RE general random-effects solver (explicit Z1, Z2 matrices, e.g. random slopes) */
static SEXP fhl_aft_2re_general(SEXP y_s, SEXP x_s, SEXP z1_s, int qq1, SEXP z2_s, int qq2,
                                SEXP delta_s, SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s) {
    int nn = nrows(x_s), pp = ncols(x_s);
    int qq = qq1 + qq2;
    int r_tot = pp + qq;
    int maxiter = asInteger(maxiter_s);
    int mord = (TYPEOF(maxiter_s) == INTSXP && LENGTH(maxiter_s) >= 2) ? INTEGER(maxiter_s)[1] : 0;
    if (nn <= 0 || pp <= 0 || qq1 <= 0 || qq2 <= 0 ||
        nrows(z1_s) != nn || ncols(z1_s) != qq1 ||
        nrows(z2_s) != nn || ncols(z2_s) != qq2 ||
        length(y_s) != nn || length(delta_s) != nn || length(beta_h0_s) != pp || length(v_h0_s) != qq) {
        Rf_error("Invalid dimensions in fhl_fit_aft_2re");
    }

    const double *y = REAL(y_s);
    const double *x = REAL(x_s);
    const double *delta = REAL(delta_s);

    double *B_xz = (double *)R_alloc((size_t)nn * r_tot, sizeof(double));
    memcpy(B_xz, x, (size_t)nn * pp * sizeof(double));
    memcpy(B_xz + (size_t)nn * pp, REAL(z1_s), (size_t)nn * qq1 * sizeof(double));
    memcpy(B_xz + (size_t)nn * (pp + qq1), REAL(z2_s), (size_t)nn * qq2 * sizeof(double));
    const double *z = B_xz + (size_t)nn * pp;
    if (mord == 1) {
        return fhl_aft_dense_hl11(nn, pp, qq1, qq2, y, x, z, delta, maxiter);
    }

    double *beta_h0 = (double *)R_alloc(pp, sizeof(double));
    double *old_beta = (double *)R_alloc(pp, sizeof(double));
    double *v_h0 = (double *)R_alloc(qq, sizeof(double));
    memcpy(beta_h0, REAL(beta_h0_s), (size_t)pp * sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double phi = 1.0, alpha1 = 0.05, alpha2 = 0.1;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *muij = (double *)R_alloc(nn, sizeof(double));
    double *mij = (double *)R_alloc(nn, sizeof(double));
    double *V_mij = (double *)R_alloc(nn, sizeof(double));
    double *xi_mij = (double *)R_alloc(nn, sizeof(double));
    double *weight1 = (double *)R_alloc(nn, sizeof(double));
    double *ystar = (double *)R_alloc(nn, sizeof(double));
    double *w_adj_yy = (double *)R_alloc(nn, sizeof(double));
    double *w1_vec = (double *)R_alloc(nn, sizeof(double));
    double *W_B = (double *)R_alloc((size_t)nn * r_tot, sizeof(double));
    double *H_mlm = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *hinv = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *rhs_mlm = (double *)R_alloc(r_tot, sizeof(double));
    double *beta_vv = (double *)R_alloc(r_tot, sizeof(double));
    double *G_zz_we = (double *)R_alloc((size_t)qq * qq, sizeof(double));
    double *iW = (double *)R_alloc((size_t)qq * qq, sizeof(double));
    double *dva = (double *)R_alloc(qq, sizeof(double));
    double *z_dva = (double *)R_alloc(nn, sizeof(double));
    double *da_weight2 = (double *)R_alloc(nn, sizeof(double));
    double *dw_vec = (double *)R_alloc(nn, sizeof(double));
    double *term_q = (double *)R_alloc(qq, sizeof(double));

    double one = 1.0, zero = 0.0;

    while (convergence > 0.000001 && iteration <= maxiter) {
        iteration++;

        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, & (int){1}, &zero, muij, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, v_h0, & (int){1}, &one, muij, & (int){1} FCONE);

        double sqrt_phi = sqrt(phi);
        double inv_sqrt_phi = 1.0 / sqrt_phi;
        double inv_phi = 1.0 / phi;

        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0, dxir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, &dxir);
            double w1r = delta[r] + (1.0 - delta[r]) * xir;
            double resid_star = delta[r] * (y[r] - muij[r]) + (1.0 - delta[r]) * (sqrt_phi * vr);
            double w1_scaled = w1r * inv_phi;
            mij[r] = mr;
            V_mij[r] = vr;
            xi_mij[r] = xir;
            weight1[r] = w1r;
            ystar[r] = resid_star;
            da_weight2[r] = dxir;
            w1_vec[r] = w1_scaled;
            w_adj_yy[r] = w1_scaled * muij[r] + resid_star * inv_phi;
        }

        for (int c = 0; c < r_tot; c++) {
            const double *bcol = B_xz + (size_t)c * nn;
            double *wcol = W_B + (size_t)c * nn;
            for (int r = 0; r < nn; r++) wcol[r] = w1_vec[r] * bcol[r];
        }
        F77_CALL(dgemm)("T", "N", &r_tot, &r_tot, &nn, &one, B_xz, &nn, W_B, &nn, &zero, H_mlm, &r_tot FCONE FCONE);

        /* Save unpenalized ZZ block scaled by phi for G_zz_we = z^T diag(weight1) z */
        for (int c = 0; c < qq; c++) {
            for (int r = 0; r < qq; r++) {
                G_zz_we[r + (size_t)qq * c] = H_mlm[(pp + r) + (size_t)r_tot * (pp + c)] * phi;
            }
        }

        double inv_alpha1 = 1.0 / alpha1;
        double inv_alpha2 = 1.0 / alpha2;
        for (int j = 0; j < qq1; j++) {
            H_mlm[(pp + j) + (size_t)r_tot * (pp + j)] += inv_alpha1;
        }
        for (int j = qq1; j < qq; j++) {
            H_mlm[(pp + j) + (size_t)r_tot * (pp + j)] += inv_alpha2;
        }
        if (!fhl_c_solve_inv(H_mlm, hinv, r_tot)) {
            Rf_error("Singular matrix in mlmfit2 (H_mlm) (not positive definite)");
        }

        memcpy(old_beta, beta_h0, (size_t)pp * sizeof(double));
        F77_CALL(dgemv)("T", &nn, &r_tot, &one, B_xz, &nn, w_adj_yy, & (int){1}, &zero, rhs_mlm, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, hinv, &r_tot, rhs_mlm, & (int){1}, &zero, beta_vv, & (int){1} FCONE);
        memcpy(beta_h0, beta_vv, (size_t)pp * sizeof(double));
        memcpy(v_h0, beta_vv + pp, (size_t)qq * sizeof(double));

        /* Invert (z^T We z + U) once for alpha1, alpha2, phi */
        double lam1 = phi / alpha1;
        double lam2 = phi / alpha2;
        for (int j = 0; j < qq1; j++) G_zz_we[j + (size_t)qq * j] += lam1;
        for (int j = qq1; j < qq; j++) G_zz_we[j + (size_t)qq * j] += lam2;
        if (!fhl_c_solve_inv(G_zz_we, iW, qq)) {
            Rf_error("Singular matrix in mlmfit2 (G_zz_we) (not positive definite)");
        }

        /* da_weight2 already populated cleanly by fhl_mlm_mills_jet above */

        /* alpha1_h update */
        double old_alpha1 = alpha1;
        double scale_dva1 = phi / (alpha1 * alpha1);
        F77_CALL(dgemv)("N", &qq, &qq1, &scale_dva1, iW, &qq, v_h0, & (int){1}, &zero, dva, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, & (int){1}, &zero, z_dva, & (int){1} FCONE);
        for (int r = 0; r < nn; r++) {
            double da_w1 = -inv_sqrt_phi * z_dva[r];
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1 * da_weight2[r]);
        }
        double tr_H_star2_1 = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, hinv, W_B, H_mlm);
        double inv_alpha1_sq = 1.0 / (alpha1 * alpha1);
        for (int j = 0; j < qq1; j++) {
            tr_H_star2_1 -= inv_alpha1_sq * hinv[(pp + j) + (size_t)r_tot * (pp + j)];
        }
        double gamma1 = -alpha1 * tr_H_star2_1;
        double ss_v1 = 0.0;
        for (int j = 0; j < qq1; j++) ss_v1 += v_h0[j] * v_h0[j];
        double denom_a1 = (double)qq1 - gamma1;
        fhl_aft_check_denom(denom_a1, "alpha1_h", "mlmfit2");
        alpha1 = ss_v1 / denom_a1;
        fhl_aft_check_pos(alpha1, "alpha1_h", "mlmfit2");

        /* alpha2_h update */
        double old_alpha2 = alpha2;
        double scale_dva2 = phi / (alpha2 * alpha2);
        F77_CALL(dgemv)("N", &qq, &qq2, &scale_dva2, iW + (size_t)qq * qq1, &qq, v_h0 + qq1, & (int){1}, &zero, dva, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, & (int){1}, &zero, z_dva, & (int){1} FCONE);
        for (int r = 0; r < nn; r++) {
            double da_w1_2 = -inv_sqrt_phi * z_dva[r];
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1_2 * da_weight2[r]);
        }
        double tr_H_star2_2 = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, hinv, W_B, H_mlm);
        double inv_alpha2_sq = 1.0 / (alpha2 * alpha2);
        for (int j = qq1; j < qq; j++) {
            tr_H_star2_2 -= inv_alpha2_sq * hinv[(pp + j) + (size_t)r_tot * (pp + j)];
        }
        double gamma2 = -alpha2 * tr_H_star2_2;
        double ss_v2 = 0.0;
        for (int j = qq1; j < qq; j++) ss_v2 += v_h0[j] * v_h0[j];
        double denom_a2 = (double)qq2 - gamma2;
        fhl_aft_check_denom(denom_a2, "alpha2_h", "mlmfit2");
        alpha2 = ss_v2 / denom_a2;
        fhl_aft_check_pos(alpha2, "alpha2_h", "mlmfit2");

        /* phi_h update */
        double old_phi = phi;
        for (int r = 0; r < nn; r++) {
            double a_r = delta[r] * mij[r] + (1.0 - delta[r]) * V_mij[r];
            dw_vec[r] = a_r + weight1[r] * mij[r];
        }
        F77_CALL(dgemv)("T", &nn, &qq, &one, z, &nn, dw_vec, & (int){1}, &zero, term_q, & (int){1} FCONE);
        double scale_dvp = -0.5 * inv_sqrt_phi;
        F77_CALL(dgemv)("N", &qq, &qq, &scale_dvp, iW, &qq, term_q, & (int){1}, &zero, dva, & (int){1} FCONE);
        F77_CALL(dgemv)("N", &nn, &qq, &one, z, &nn, dva, & (int){1}, &zero, z_dva, & (int){1} FCONE);

        double inv_phi2 = inv_phi * inv_phi;
        double sum_w1 = 0.0, rss_y = 0.0;
        for (int r = 0; r < nn; r++) {
            double dp_w1 = -0.5 * mij[r] * inv_phi - inv_sqrt_phi * z_dva[r];
            double d_w_phi = inv_phi * (1.0 - delta[r]) * (dp_w1 * da_weight2[r]);
            dw_vec[r] = -weight1[r] * inv_phi2 + d_w_phi;
            sum_w1 += weight1[r];
            double diff = ystar[r];
            rss_y += diff * diff;
        }
        double tr_H_star = fhl_mlm_weighted_gram_trace(B_xz, nn, r_tot, dw_vec, hinv, W_B, H_mlm);
        double gamma0 = phi * tr_H_star;
        double denom_p = sum_w1 + gamma0;
        fhl_aft_check_denom(denom_p, "phi_h", "mlmfit2");
        phi = rss_y / denom_p;
        fhl_aft_check_pos(phi, "phi_h", "mlmfit2");

        double diff_beta = 0.0;
        for (int j = 0; j < pp; j++) diff_beta += fabs(old_beta[j] - beta_h0[j]);
        convergence = diff_beta + fabs(old_alpha1 - alpha1) + fabs(old_alpha2 - alpha2) + fabs(old_phi - phi);
        i_count++;
    }

    /* Evaluate H^{-1} at final updated (beta_h0, v_h0, alpha1, alpha2, phi) for SE */
    fhl_mlm_eval_dense_hinv(
        nn, pp, qq, qq1, r_tot,
        y, x, z, B_xz, delta,
        beta_h0, v_h0, alpha1, alpha2, phi,
        muij, w1_vec, W_B, H_mlm, hinv
    );

    SEXP out = PROTECT(allocVector(VECSXP, 8));
    SEXP beta_h_s = PROTECT(allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_h0[j];
        REAL(se_beh_s)[j] = sqrt(hinv[j + (size_t)r_tot * j]);
    }
    SEXP v_out_s = PROTECT(allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, ScalarReal(alpha1));
    SET_VECTOR_ELT(out, 4, ScalarReal(alpha2));
    SET_VECTOR_ELT(out, 5, ScalarReal(phi));
    SET_VECTOR_ELT(out, 6, ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 7, ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}

/* Solve 2-block Z-system W_zz * [out_s; out_b] = [r_s; r_b] using qs x qs Schur inverse */
static void fhl_mlm2_solve_W(
    const double *S_W_inv, const double *C_sb, const double *K_W_sb,
    const double *D_W_b_inv, int qs, int qb,
    const double *r_s, const double *r_b,
    double *out_s, double *out_b,
    double *tmp_s, double *tmp_b
) {
    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int inc1 = 1;
    memcpy(tmp_s, r_s, (size_t)qs * sizeof(double));
    F77_CALL(dgemv)("N", &qs, &qb, &neg_one, K_W_sb, &qs, r_b, &inc1, &one, tmp_s, &inc1 FCONE);
    F77_CALL(dgemv)("N", &qs, &qs, &one, S_W_inv, &qs, tmp_s, &inc1, &zero, out_s, &inc1 FCONE);

    memcpy(tmp_b, r_b, (size_t)qb * sizeof(double));
    F77_CALL(dgemv)("T", &qs, &qb, &neg_one, C_sb, &qs, out_s, &inc1, &one, tmp_b, &inc1 FCONE);
    for (int k = 0; k < qb; k++) {
        out_b[k] = D_W_b_inv[k] * tmp_b[k];
    }
}

/* Exact tr(H^{-1} B^T diag(w) B) for 2-RE mixed model via Schur blocks,
 * without materializing the dense (p+q1+q2)^2 matrices or B_xz. */
static double fhl_mlm2_schur_weighted_trace(
    const double *x, const int *gs, const int *gb,
    int nn, int pp, int qs, int qb,
    const double *w,
    const double *Hinv_AA, const double *Hinv_AB, const double *Hinv_BB_diag,
    double *wx_buf, double *tr_G_XX, double *tr_G_XZs, double *tr_G_Zs_diag,
    double *tr_G_AB, double *tr_G_Zb_diag
) {
    int mA = pp + qs;
    memset(tr_G_XZs, 0, (size_t)pp * (size_t)qs * sizeof(double));
    memset(tr_G_Zs_diag, 0, (size_t)qs * sizeof(double));
    memset(tr_G_AB, 0, (size_t)mA * (size_t)qb * sizeof(double));
    memset(tr_G_Zb_diag, 0, (size_t)qb * sizeof(double));

    for (int r = 0; r < nn; r++) {
        double wr = w[r];
        int gsr = gs[r];
        int gbr = gb[r];
        tr_G_Zs_diag[gsr] += wr;
        tr_G_Zb_diag[gbr] += wr;
        tr_G_AB[(pp + gsr) + (size_t)mA * (size_t)gbr] += wr;
    }
    for (int a = 0; a < pp; a++) {
        const double *xcol = x + (size_t)a * (size_t)nn;
        double *wxcol = wx_buf + (size_t)a * (size_t)nn;
        for (int r = 0; r < nn; r++) {
            double v = w[r] * xcol[r];
            wxcol[r] = v;
            tr_G_XZs[a + (size_t)pp * (size_t)gs[r]] += v;
            tr_G_AB[a + (size_t)mA * (size_t)gb[r]] += v;
        }
    }

    double one = 1.0, zero = 0.0;
    F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, tr_G_XX, &pp FCONE FCONE);

    double tr = 0.0;
    for (int c = 0; c < pp; c++) {
        for (int a = 0; a < pp; a++) {
            tr += Hinv_AA[a + (size_t)mA * (size_t)c] * tr_G_XX[a + (size_t)pp * (size_t)c];
        }
    }
    for (int j = 0; j < qs; j++) {
        for (int a = 0; a < pp; a++) {
            tr += 2.0 * Hinv_AA[a + (size_t)mA * (size_t)(pp + j)] * tr_G_XZs[a + (size_t)pp * (size_t)j];
        }
        tr += Hinv_AA[(pp + j) + (size_t)mA * (size_t)(pp + j)] * tr_G_Zs_diag[j];
    }
    size_t len_ab = (size_t)mA * (size_t)qb;
    for (size_t idx = 0; idx < len_ab; idx++) {
        tr += 2.0 * Hinv_AB[idx] * tr_G_AB[idx];
    }
    for (int k = 0; k < qb; k++) {
        tr += Hinv_BB_diag[k] * tr_G_Zb_diag[k];
    }
    return tr;
}

/* Unified AFT 2-RE entrypoint: dispatches O(q) 2-level Schur solver for integer
 * cluster vectors or general [Z1, Z2] solver for random-slope matrices. */
SEXP fhl_fit_aft_2re(
    SEXP y_s, SEXP x_s,
    SEXP group1_s, SEXP qq1_s,
    SEXP group2_s, SEXP qq2_s,
    SEXP delta_s, SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s
) {
    if (TYPEOF(y_s) != REALSXP || TYPEOF(x_s) != REALSXP || !Rf_isMatrix(x_s) ||
        TYPEOF(delta_s) != REALSXP || TYPEOF(beta_h0_s) != REALSXP || TYPEOF(v_h0_s) != REALSXP) {
        Rf_error("Invalid SEXP types in fhl_fit_aft_2re");
    }
    int qq1 = asInteger(qq1_s), qq2 = asInteger(qq2_s);
    if (TYPEOF(group1_s) == REALSXP && Rf_isMatrix(group1_s) &&
        TYPEOF(group2_s) == REALSXP && Rf_isMatrix(group2_s)) {
        return fhl_aft_2re_general(y_s, x_s, group1_s, qq1, group2_s, qq2, delta_s, beta_h0_s, v_h0_s, maxiter_s);
    }
    if (TYPEOF(group1_s) != INTSXP || TYPEOF(group2_s) != INTSXP) {
        Rf_error("group1_s and group2_s must both be integer cluster vectors or numeric Z matrices in fhl_fit_aft_2re");
    }
    int nn = nrows(x_s), pp = ncols(x_s);
    int maxiter = asInteger(maxiter_s);
    if (nn <= 0 || pp <= 0 || qq1 <= 0 || qq2 <= 0 ||
        length(y_s) != nn || length(group1_s) != nn || length(group2_s) != nn ||
        length(delta_s) != nn || length(beta_h0_s) != pp || length(v_h0_s) != qq1 + qq2) {
        Rf_error("Invalid dimensions in fhl_fit_aft_2re");
    }

    const double *y = REAL(y_s);
    const double *x = REAL(x_s);
    const int *g1_in = INTEGER(group1_s);
    const int *g2_in = INTEGER(group2_s);
    const double *delta = REAL(delta_s);

    int *g1 = (int *)R_alloc((size_t)nn, sizeof(int));
    int *g2 = (int *)R_alloc((size_t)nn, sizeof(int));
    for (int r = 0; r < nn; r++) {
        int gi1 = g1_in[r] - 1;
        int gi2 = g2_in[r] - 1;
        if (gi1 < 0 || gi1 >= qq1) Rf_error("Invalid group1 index in fhl_fit_aft_2re");
        if (gi2 < 0 || gi2 >= qq2) Rf_error("Invalid group2 index in fhl_fit_aft_2re");
        g1[r] = gi1;
        g2[r] = gi2;
    }
    int mord = (TYPEOF(maxiter_s) == INTSXP && LENGTH(maxiter_s) >= 2) ? INTEGER(maxiter_s)[1] : 0;
    if (mord == 1) {
        int qq_tot = qq1 + qq2;
        double *z_full = (double *)R_alloc((size_t)nn * (size_t)qq_tot, sizeof(double));
        memset(z_full, 0, (size_t)nn * (size_t)qq_tot * sizeof(double));
        for (int r = 0; r < nn; r++) {
            z_full[r + (size_t)nn * (size_t)g1[r]] = 1.0;
            z_full[r + (size_t)nn * (size_t)(qq1 + g2[r])] = 1.0;
        }
        return fhl_aft_dense_hl11(nn, pp, qq1, qq2, y, x, z_full, delta, maxiter);
    }

    /* Always eliminate the larger random-effect block so Schur inversion is (p + min(q1, q2)) */
    int swapped = (qq1 > qq2) ? 1 : 0;
    int qs = swapped ? qq2 : qq1;
    int qb = swapped ? qq1 : qq2;
    const int *gs = swapped ? g2 : g1;
    const int *gb = swapped ? g1 : g2;
    int mA = pp + qs;

    double *beta_h0 = (double *)R_alloc((size_t)pp, sizeof(double));
    double *old_beta = (double *)R_alloc((size_t)pp, sizeof(double));
    double *vs = (double *)R_alloc((size_t)qs, sizeof(double));
    double *vb = (double *)R_alloc((size_t)qb, sizeof(double));
    memcpy(beta_h0, REAL(beta_h0_s), (size_t)pp * sizeof(double));
    if (!swapped) {
        memcpy(vs, REAL(v_h0_s), (size_t)qs * sizeof(double));
        memcpy(vb, REAL(v_h0_s) + qs, (size_t)qb * sizeof(double));
    } else {
        memcpy(vs, REAL(v_h0_s) + qq1, (size_t)qs * sizeof(double));
        memcpy(vb, REAL(v_h0_s), (size_t)qb * sizeof(double));
    }

    double phi = 1.0;
    double alpha_s = swapped ? 0.1 : 0.05;
    double alpha_b = swapped ? 0.05 : 0.1;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *muij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *V_mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *xi_mij = (double *)R_alloc((size_t)nn, sizeof(double));
    double *weight1 = (double *)R_alloc((size_t)nn, sizeof(double));
    double *ystar = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w1_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *w_adj_yy = (double *)R_alloc((size_t)nn, sizeof(double));
    double *da_weight2 = (double *)R_alloc((size_t)nn, sizeof(double));
    double *dw_vec = (double *)R_alloc((size_t)nn, sizeof(double));
    double *wx_buf = (double *)R_alloc((size_t)nn * (size_t)pp, sizeof(double));

    double *xx_w = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *xz_s_w = (double *)R_alloc((size_t)pp * (size_t)qs, sizeof(double));
    double *zz_s_w1 = (double *)R_alloc((size_t)qs, sizeof(double));
    double *zz_b_w1 = (double *)R_alloc((size_t)qb, sizeof(double));
    double *inv_D_B = (double *)R_alloc((size_t)qb, sizeof(double));

    double *H_AA = (double *)R_alloc((size_t)mA * (size_t)mA, sizeof(double));
    double *Hinv_AA = (double *)R_alloc((size_t)mA * (size_t)mA, sizeof(double));
    memset(Hinv_AA, 0, (size_t)mA * (size_t)mA * sizeof(double));
    double *H_AB = (double *)R_alloc((size_t)mA * (size_t)qb, sizeof(double));
    double *K_AB = (double *)R_alloc((size_t)mA * (size_t)qb, sizeof(double));
    double *Hinv_AB = (double *)R_alloc((size_t)mA * (size_t)qb, sizeof(double));
    double *Hinv_BB_diag = (double *)R_alloc((size_t)qb, sizeof(double));

    double *C_sb = (double *)R_alloc((size_t)qs * (size_t)qb, sizeof(double));
    double *K_W_sb = (double *)R_alloc((size_t)qs * (size_t)qb, sizeof(double));
    double *D_W_b_inv = (double *)R_alloc((size_t)qb, sizeof(double));
    double *S_W = (double *)R_alloc((size_t)qs * (size_t)qs, sizeof(double));
    double *S_W_inv = (double *)R_alloc((size_t)qs * (size_t)qs, sizeof(double));

    double *rhs_A = (double *)R_alloc((size_t)mA, sizeof(double));
    double *rhs_Zb = (double *)R_alloc((size_t)qb, sizeof(double));
    double *sol_A = (double *)R_alloc((size_t)mA, sizeof(double));
    double *dva_s = (double *)R_alloc((size_t)qs, sizeof(double));
    double *dva_b = (double *)R_alloc((size_t)qb, sizeof(double));
    double *rhs_W_s = (double *)R_alloc((size_t)qs, sizeof(double));
    double *rhs_W_b = (double *)R_alloc((size_t)qb, sizeof(double));
    double *tmp_s = (double *)R_alloc((size_t)qs, sizeof(double));
    double *tmp_b = (double *)R_alloc((size_t)qb, sizeof(double));
    double *term_s = (double *)R_alloc((size_t)qs, sizeof(double));
    double *term_b = (double *)R_alloc((size_t)qb, sizeof(double));

    double *tr_G_XX = (double *)R_alloc((size_t)pp * (size_t)pp, sizeof(double));
    double *tr_G_XZs = (double *)R_alloc((size_t)pp * (size_t)qs, sizeof(double));
    double *tr_G_Zs_diag = (double *)R_alloc((size_t)qs, sizeof(double));
    double *tr_G_AB = (double *)R_alloc((size_t)mA * (size_t)qb, sizeof(double));
    double *tr_G_Zb_diag = (double *)R_alloc((size_t)qb, sizeof(double));

    double one = 1.0, zero = 0.0, neg_one = -1.0;
    int inc1 = 1;

    while (convergence > 0.000001 && iteration <= maxiter) {
        iteration++;

        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, &inc1, &zero, muij, &inc1 FCONE);
        for (int r = 0; r < nn; r++) {
            muij[r] += vs[gs[r]] + vb[gb[r]];
        }

        double sqrt_phi = sqrt(phi);
        double inv_sqrt_phi = 1.0 / sqrt_phi;
        double inv_phi = 1.0 / phi;

        memset(zz_s_w1, 0, (size_t)qs * sizeof(double));
        memset(zz_b_w1, 0, (size_t)qb * sizeof(double));
        memset(rhs_A, 0, (size_t)mA * sizeof(double));
        memset(rhs_Zb, 0, (size_t)qb * sizeof(double));
        memset(xz_s_w, 0, (size_t)pp * (size_t)qs * sizeof(double));
        memset(H_AB, 0, (size_t)mA * (size_t)qb * sizeof(double));

        double sum_w1 = 0.0, rss_y = 0.0;
        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0, dxir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, &dxir);
            double w1r = delta[r] + (1.0 - delta[r]) * xir;
            double resid_star = delta[r] * (y[r] - muij[r]) + (1.0 - delta[r]) * (sqrt_phi * vr);
            double w1_scaled = w1r * inv_phi;
            double w_adj_r = w1_scaled * muij[r] + resid_star * inv_phi;

            mij[r] = mr;
            V_mij[r] = vr;
            xi_mij[r] = xir;
            weight1[r] = w1r;
            ystar[r] = resid_star;
            w1_vec[r] = w1_scaled;
            w_adj_yy[r] = w_adj_r;
            da_weight2[r] = dxir;

            sum_w1 += w1r;
            rss_y += resid_star * resid_star;

            int gsr = gs[r];
            int gbr = gb[r];
            zz_s_w1[gsr] += w1r;
            zz_b_w1[gbr] += w1r;
            rhs_A[pp + gsr] += w_adj_r;
            rhs_Zb[gbr] += w_adj_r;
            H_AB[(pp + gsr) + (size_t)mA * (size_t)gbr] += w1_scaled;
        }

        for (int a = 0; a < pp; a++) {
            const double *xcol = x + (size_t)a * (size_t)nn;
            double *wxcol = wx_buf + (size_t)a * (size_t)nn;
            for (int r = 0; r < nn; r++) {
                double wxr = w1_vec[r] * xcol[r];
                wxcol[r] = wxr;
                xz_s_w[a + (size_t)pp * (size_t)gs[r]] += wxr;
                H_AB[a + (size_t)mA * (size_t)gb[r]] += wxr;
            }
        }
        F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);
        F77_CALL(dgemv)("T", &nn, &pp, &one, x, &nn, w_adj_yy, &inc1, &zero, rhs_A, &inc1 FCONE);

        /* Save unscaled C_sb = Z_s^T diag(weight1) Z_b for W_zz solve */
        for (int k = 0; k < qb; k++) {
            const double *hab_col = H_AB + (size_t)mA * (size_t)k + pp;
            double *csb_col = C_sb + (size_t)qs * (size_t)k;
            for (int j = 0; j < qs; j++) {
                csb_col[j] = hab_col[j] * phi;
            }
        }

        /* Assemble H_AA (mA x mA) */
        double inv_alpha_s = 1.0 / alpha_s;
        double inv_alpha_b = 1.0 / alpha_b;
        memset(H_AA, 0, (size_t)mA * (size_t)mA * sizeof(double));
        for (int c = 0; c < pp; c++) {
            for (int a = 0; a < pp; a++) {
                H_AA[a + (size_t)mA * (size_t)c] = xx_w[a + (size_t)pp * (size_t)c];
            }
        }
        for (int j = 0; j < qs; j++) {
            for (int a = 0; a < pp; a++) {
                double v = xz_s_w[a + (size_t)pp * (size_t)j];
                H_AA[a + (size_t)mA * (size_t)(pp + j)] = v;
                H_AA[(pp + j) + (size_t)mA * (size_t)a] = v;
            }
            H_AA[(pp + j) + (size_t)mA * (size_t)(pp + j)] = zz_s_w1[j] * inv_phi + inv_alpha_s;
        }

        /* Schur complement eliminating B = Z_b */
        for (int k = 0; k < qb; k++) {
            double db = zz_b_w1[k] * inv_phi + inv_alpha_b;
            if (!R_FINITE(db) || db <= 0.0) {
                Rf_error("mlmfit2: diagonal entry D_B[%d] non-positive or non-finite (%g)", k + 1, db);
            }
            double inv_d = 1.0 / db;
            inv_D_B[k] = inv_d;
            const double *hcol = H_AB + (size_t)mA * (size_t)k;
            double *kcol = K_AB + (size_t)mA * (size_t)k;
            for (int a = 0; a < mA; a++) {
                kcol[a] = hcol[a] * inv_d;
            }
        }
        F77_CALL(dgemm)("N", "T", &mA, &mA, &qb, &neg_one, H_AB, &mA, K_AB, &mA, &one, H_AA, &mA FCONE FCONE);
        if (!fhl_c_solve_inv(H_AA, Hinv_AA, mA)) {
            Rf_error("Singular matrix in mlmfit2 (Schur S_AA) (not positive definite)");
        }

        F77_CALL(dgemm)("N", "N", &mA, &qb, &mA, &neg_one, Hinv_AA, &mA, K_AB, &mA, &zero, Hinv_AB, &mA FCONE FCONE);

        for (int k = 0; k < qb; k++) {
            const double *hinv_col = Hinv_AB + (size_t)mA * (size_t)k;
            const double *k_col = K_AB + (size_t)mA * (size_t)k;
            double quad = 0.0;
            for (int a = 0; a < mA; a++) quad += hinv_col[a] * k_col[a];
            Hinv_BB_diag[k] = inv_D_B[k] - quad;
        }

        /* Update beta_h0, vs, vb via Schur back-substitution */
        memcpy(old_beta, beta_h0, (size_t)pp * sizeof(double));
        F77_CALL(dgemv)("N", &mA, &qb, &neg_one, K_AB, &mA, rhs_Zb, &inc1, &one, rhs_A, &inc1 FCONE);
        F77_CALL(dgemv)("N", &mA, &mA, &one, Hinv_AA, &mA, rhs_A, &inc1, &zero, sol_A, &inc1 FCONE);
        memcpy(beta_h0, sol_A, (size_t)pp * sizeof(double));
        memcpy(vs, sol_A + pp, (size_t)qs * sizeof(double));

        for (int k = 0; k < qb; k++) {
            const double *hcol = H_AB + (size_t)mA * (size_t)k;
            double ht_sol = 0.0;
            for (int a = 0; a < mA; a++) ht_sol += hcol[a] * sol_A[a];
            vb[k] = inv_D_B[k] * (rhs_Zb[k] - ht_sol);
        }

        /* Invert Schur complement of W_zz = Z^T We Z + diag(lam_s, lam_b) */
        double lam_s = phi / alpha_s;
        double lam_b = phi / alpha_b;
        for (int k = 0; k < qb; k++) {
            double dw_b = zz_b_w1[k] + lam_b;
            if (!R_FINITE(dw_b) || dw_b <= 0.0) {
                Rf_error("mlmfit2: W_bb diagonal[%d] non-positive or non-finite (%g)", k + 1, dw_b);
            }
            double dw_b_inv = 1.0 / dw_b;
            D_W_b_inv[k] = dw_b_inv;
            const double *csb_col = C_sb + (size_t)qs * (size_t)k;
            double *kw_col = K_W_sb + (size_t)qs * (size_t)k;
            for (int j = 0; j < qs; j++) {
                kw_col[j] = csb_col[j] * dw_b_inv;
            }
        }
        F77_CALL(dgemm)("N", "T", &qs, &qs, &qb, &neg_one, C_sb, &qs, K_W_sb, &qs, &zero, S_W, &qs FCONE FCONE);
        for (int j = 0; j < qs; j++) {
            S_W[j + (size_t)qs * (size_t)j] += zz_s_w1[j] + lam_s;
        }
        if (!fhl_c_solve_inv(S_W, S_W_inv, qs)) {
            Rf_error("Singular matrix in mlmfit2 (Schur S_W) (not positive definite)");
        }

        /* alpha_s update */
        double old_alpha_s = alpha_s;
        double scale_dva_s = phi / (alpha_s * alpha_s);
        for (int j = 0; j < qs; j++) rhs_W_s[j] = scale_dva_s * vs[j];
        memset(rhs_W_b, 0, (size_t)qb * sizeof(double));
        fhl_mlm2_solve_W(S_W_inv, C_sb, K_W_sb, D_W_b_inv, qs, qb, rhs_W_s, rhs_W_b, dva_s, dva_b, tmp_s, tmp_b);
        for (int r = 0; r < nn; r++) {
            double da_w1 = -inv_sqrt_phi * (dva_s[gs[r]] + dva_b[gb[r]]);
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1 * da_weight2[r]);
        }
        double tr_H_star2_s = fhl_mlm2_schur_weighted_trace(
            x, gs, gb, nn, pp, qs, qb, dw_vec,
            Hinv_AA, Hinv_AB, Hinv_BB_diag,
            wx_buf, tr_G_XX, tr_G_XZs, tr_G_Zs_diag, tr_G_AB, tr_G_Zb_diag
        );
        double inv_alpha_s_sq = 1.0 / (alpha_s * alpha_s);
        for (int j = 0; j < qs; j++) {
            tr_H_star2_s -= inv_alpha_s_sq * Hinv_AA[(pp + j) + (size_t)mA * (size_t)(pp + j)];
        }
        double gamma_s = -alpha_s * tr_H_star2_s;
        double ss_vs = 0.0;
        for (int j = 0; j < qs; j++) ss_vs += vs[j] * vs[j];
        double denom_as = (double)qs - gamma_s;
        fhl_aft_check_denom(denom_as, "alpha_s", "mlmfit2");
        alpha_s = ss_vs / denom_as;
        fhl_aft_check_pos(alpha_s, "alpha_s", "mlmfit2");

        /* alpha_b update */
        double old_alpha_b = alpha_b;
        double scale_dva_b = phi / (alpha_b * alpha_b);
        memset(rhs_W_s, 0, (size_t)qs * sizeof(double));
        for (int k = 0; k < qb; k++) rhs_W_b[k] = scale_dva_b * vb[k];
        fhl_mlm2_solve_W(S_W_inv, C_sb, K_W_sb, D_W_b_inv, qs, qb, rhs_W_s, rhs_W_b, dva_s, dva_b, tmp_s, tmp_b);
        for (int r = 0; r < nn; r++) {
            double da_w1_b = -inv_sqrt_phi * (dva_s[gs[r]] + dva_b[gb[r]]);
            dw_vec[r] = inv_phi * (1.0 - delta[r]) * (da_w1_b * da_weight2[r]);
        }
        double tr_H_star2_b = fhl_mlm2_schur_weighted_trace(
            x, gs, gb, nn, pp, qs, qb, dw_vec,
            Hinv_AA, Hinv_AB, Hinv_BB_diag,
            wx_buf, tr_G_XX, tr_G_XZs, tr_G_Zs_diag, tr_G_AB, tr_G_Zb_diag
        );
        double inv_alpha_b_sq = 1.0 / (alpha_b * alpha_b);
        for (int k = 0; k < qb; k++) {
            tr_H_star2_b -= inv_alpha_b_sq * Hinv_BB_diag[k];
        }
        double gamma_b = -alpha_b * tr_H_star2_b;
        double ss_vb = 0.0;
        for (int k = 0; k < qb; k++) ss_vb += vb[k] * vb[k];
        double denom_ab = (double)qb - gamma_b;
        fhl_aft_check_denom(denom_ab, "alpha_b", "mlmfit2");
        alpha_b = ss_vb / denom_ab;
        fhl_aft_check_pos(alpha_b, "alpha_b", "mlmfit2");

        /* phi_h update */
        double old_phi = phi;
        memset(term_s, 0, (size_t)qs * sizeof(double));
        memset(term_b, 0, (size_t)qb * sizeof(double));
        for (int r = 0; r < nn; r++) {
            double a_r = delta[r] * mij[r] + (1.0 - delta[r]) * V_mij[r];
            double tv = a_r + weight1[r] * mij[r];
            term_s[gs[r]] += tv;
            term_b[gb[r]] += tv;
        }
        double scale_dvp = -0.5 * inv_sqrt_phi;
        for (int j = 0; j < qs; j++) rhs_W_s[j] = scale_dvp * term_s[j];
        for (int k = 0; k < qb; k++) rhs_W_b[k] = scale_dvp * term_b[k];
        fhl_mlm2_solve_W(S_W_inv, C_sb, K_W_sb, D_W_b_inv, qs, qb, rhs_W_s, rhs_W_b, dva_s, dva_b, tmp_s, tmp_b);

        double inv_phi2 = inv_phi * inv_phi;
        for (int r = 0; r < nn; r++) {
            double z_dva_r = dva_s[gs[r]] + dva_b[gb[r]];
            double dp_w1 = -0.5 * mij[r] * inv_phi - inv_sqrt_phi * z_dva_r;
            double d_w_phi = inv_phi * (1.0 - delta[r]) * (dp_w1 * da_weight2[r]);
            dw_vec[r] = -weight1[r] * inv_phi2 + d_w_phi;
        }
        double tr_H_star = fhl_mlm2_schur_weighted_trace(
            x, gs, gb, nn, pp, qs, qb, dw_vec,
            Hinv_AA, Hinv_AB, Hinv_BB_diag,
            wx_buf, tr_G_XX, tr_G_XZs, tr_G_Zs_diag, tr_G_AB, tr_G_Zb_diag
        );
        double gamma0 = phi * tr_H_star;
        double denom_p = sum_w1 + gamma0;
        fhl_aft_check_denom(denom_p, "phi_h", "mlmfit2");
        phi = rss_y / denom_p;
        fhl_aft_check_pos(phi, "phi_h", "mlmfit2");

        double diff_beta = 0.0;
        for (int j = 0; j < pp; j++) diff_beta += fabs(old_beta[j] - beta_h0[j]);
        convergence = diff_beta + fabs(old_alpha_s - alpha_s) + fabs(old_alpha_b - alpha_b) + fabs(old_phi - phi);
        i_count++;
    }

    /* Evaluate 2-level Schur Hessian H^{-1}_AA at final updated (beta_h0, vs, vb, alpha_s, alpha_b, phi) */
    {
        double inv_sqrt_phi = 1.0 / sqrt(phi), inv_phi = 1.0 / phi;
        double inv_alpha_s = 1.0 / alpha_s, inv_alpha_b = 1.0 / alpha_b;
        F77_CALL(dgemv)("N", &nn, &pp, &one, x, &nn, beta_h0, &inc1, &zero, muij, &inc1 FCONE);
        for (int r = 0; r < nn; r++) muij[r] += vs[gs[r]] + vb[gb[r]];
        memset(zz_s_w1, 0, (size_t)qs * sizeof(double));
        memset(zz_b_w1, 0, (size_t)qb * sizeof(double));
        memset(xz_s_w, 0, (size_t)pp * (size_t)qs * sizeof(double));
        memset(H_AB, 0, (size_t)mA * (size_t)qb * sizeof(double));
        for (int r = 0; r < nn; r++) {
            double mr = inv_sqrt_phi * (y[r] - muij[r]);
            double vr = 0.0, xir = 0.0;
            fhl_mlm_mills_jet(mr, &vr, &xir, NULL);
            double w1r = delta[r] + (1.0 - delta[r]) * xir;
            double w1_scaled = w1r * inv_phi;
            w1_vec[r] = w1_scaled;
            zz_s_w1[gs[r]] += w1r;
            zz_b_w1[gb[r]] += w1r;
            H_AB[(pp + gs[r]) + (size_t)mA * (size_t)gb[r]] += w1_scaled;
        }
        for (int a = 0; a < pp; a++) {
            const double *xcol = x + (size_t)a * (size_t)nn;
            double *wxcol = wx_buf + (size_t)a * (size_t)nn;
            for (int r = 0; r < nn; r++) {
                double wxr = w1_vec[r] * xcol[r];
                wxcol[r] = wxr;
                xz_s_w[a + (size_t)pp * (size_t)gs[r]] += wxr;
                H_AB[a + (size_t)mA * (size_t)gb[r]] += wxr;
            }
        }
        F77_CALL(dgemm)("T", "N", &pp, &pp, &nn, &one, x, &nn, wx_buf, &nn, &zero, xx_w, &pp FCONE FCONE);
        memset(H_AA, 0, (size_t)mA * (size_t)mA * sizeof(double));
        for (int c = 0; c < pp; c++) {
            for (int a = 0; a < pp; a++) H_AA[a + (size_t)mA * (size_t)c] = xx_w[a + (size_t)pp * (size_t)c];
        }
        for (int j = 0; j < qs; j++) {
            for (int a = 0; a < pp; a++) {
                double v = xz_s_w[a + (size_t)pp * (size_t)j];
                H_AA[a + (size_t)mA * (size_t)(pp + j)] = v;
                H_AA[(pp + j) + (size_t)mA * (size_t)a] = v;
            }
            H_AA[(pp + j) + (size_t)mA * (size_t)(pp + j)] = zz_s_w1[j] * inv_phi + inv_alpha_s;
        }
        for (int k = 0; k < qb; k++) {
            double db = zz_b_w1[k] * inv_phi + inv_alpha_b;
            if (!R_FINITE(db) || db <= 0.0) {
                Rf_error("mlmfit2: final SE diagonal entry D_B[%d] non-positive or non-finite (%g)", k + 1, db);
            }
            double inv_d = 1.0 / db;
            const double *hcol = H_AB + (size_t)mA * (size_t)k;
            double *kcol = K_AB + (size_t)mA * (size_t)k;
            for (int a = 0; a < mA; a++) kcol[a] = hcol[a] * inv_d;
        }
        F77_CALL(dgemm)("N", "T", &mA, &mA, &qb, &neg_one, H_AB, &mA, K_AB, &mA, &one, H_AA, &mA FCONE FCONE);
        if (!fhl_c_solve_inv(H_AA, Hinv_AA, mA)) {
            Rf_error("Singular matrix in mlmfit2 final SE evaluation (Schur S_AA) (not positive definite)");
        }
    }

    SEXP out = PROTECT(allocVector(VECSXP, 8));
    SEXP beta_h_s = PROTECT(allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_h0[j];
        REAL(se_beh_s)[j] = sqrt(Hinv_AA[j + (size_t)mA * (size_t)j]);
    }
    SEXP v_out_s = PROTECT(allocVector(REALSXP, qq1 + qq2));
    if (!swapped) {
        memcpy(REAL(v_out_s), vs, (size_t)qs * sizeof(double));
        memcpy(REAL(v_out_s) + qs, vb, (size_t)qb * sizeof(double));
    } else {
        memcpy(REAL(v_out_s), vb, (size_t)qb * sizeof(double));
        memcpy(REAL(v_out_s) + qb, vs, (size_t)qs * sizeof(double));
    }

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, ScalarReal(swapped ? alpha_b : alpha_s));
    SET_VECTOR_ELT(out, 4, ScalarReal(swapped ? alpha_s : alpha_b));
    SET_VECTOR_ELT(out, 5, ScalarReal(phi));
    SET_VECTOR_ELT(out, 6, ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 7, ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}
