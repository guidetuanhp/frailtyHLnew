#include "fhl_linalg.h"
#include "fhl_native_api.h"
#include <float.h>

#undef exp
#undef log
#define exp(x) fhl_c_r_exp(x)
#define log(x) fhl_c_r_log(x)

#ifndef FHL_INLINE
#define FHL_INLINE inline
#endif

/* ==========================================================================
 * Joint Longitudinal-Survival & Bivariate Frailty Engine (jmfit2, jmfit3, jmfit_FM)
 * ========================================================================== */

static FHL_INLINE void fhl_jm_check_pos(double val, const char *what, const char *fn) {
    if (!R_FINITE(val) || val <= 1e-10) {
        Rf_error("%s: %s non-positive or non-finite (%g); variance component at boundary/degenerate", fn, what, val);
    }
}

static FHL_INLINE void fhl_jm_check_denom(double val, const char *what, const char *fn) {
    if (!R_FINITE(val) || val <= 0.0) {
        Rf_error("%s: %s denominator non-positive or non-finite (%g); REML/dispersion step degenerate", fn, what, val);
    }
}

static double fhl_jm_require_spd_logdet(const double *H, int n, const char *where) {
    if (n <= 0 || !H) {
        Rf_error("%s: invalid Hessian matrix dimensions (%d)", where, n);
    }
    size_t nn_len = (size_t)n * (size_t)n;
    double *buf = (double *)R_alloc(nn_len, sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double v = H[i + (size_t)n * (size_t)j];
            if (!R_FINITE(v)) {
                Rf_error("%s: joint Hessian H_jm is not positive definite (non-finite entry)", where);
            }
            buf[i + (size_t)n * (size_t)j] = v;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = j + 1; i < n; i++) {
            double aij = buf[i + (size_t)n * (size_t)j];
            double aji = buf[j + (size_t)n * (size_t)i];
            double scale = 1.0;
            if (fabs(aij) > scale) scale = fabs(aij);
            if (fabs(aji) > scale) scale = fabs(aji);
            if (fabs(aij - aji) > 1e-10 * scale) {
                Rf_error("%s: joint Hessian H_jm is not positive definite (asymmetric)", where);
            }
        }
    }
    int info = 0;
    F77_CALL(dpotrf)("U", &n, buf, &n, &info FCONE);
    if (info != 0) {
        Rf_error("%s: joint Hessian H_jm is not positive definite", where);
    }
    double sum_log = 0.0;
    for (int i = 0; i < n; i++) {
        double dii = buf[i + (size_t)n * (size_t)i];
        if (!R_FINITE(dii) || dii <= 0.0) {
            Rf_error("%s: joint Hessian H_jm is not positive definite", where);
        }
        sum_log += log(dii);
    }
    double logdet = 2.0 * sum_log;
    if (!R_FINITE(logdet)) {
        Rf_error("%s: joint Hessian H_jm log-determinant non-finite", where);
    }
    return logdet;
}

typedef struct {
    fhl_c_xz_t xz;
    fhl_c_risk_t risk;
    int n;
    int p;
    int qq;
    int idx2;
    const double *del;
    const double *di;
    double *beta_h;
    double *zv_vec;
    double *muh;
    double *Wi_vec;
    double *denom;
    double *cla0;
    double *Bi_vec;
    double *Adi_vec;
    double *G_surv;
    double *adj_zz;
    double *w_muh;
    double *tmp_m;
    double *tmp_n;
    double *rhs_z_tmp;
} fhl_jm_surv_ctx_t;

static void fhl_jm_init_surv_block(
    fhl_jm_surv_ctx_t *blk,
    SEXP del_s, SEXP X_s, SEXP Z_s,
    SEXP di_s, SEXP Mi_s, SEXP idx2_s,
    SEXP beta_h0_s, int qq_expected,
    const char *label
) {
    memset(blk, 0, sizeof(*blk));
    if (!Rf_isMatrix(X_s) || TYPEOF(X_s) != REALSXP) {
        Rf_error("X must be a numeric matrix in %s", label);
    }
    int n = Rf_nrows(X_s);
    int p = Rf_ncols(X_s);
    if (TYPEOF(del_s) != REALSXP || Rf_length(del_s) != n) {
        Rf_error("del length mismatch in %s (expected %d)", label, n);
    }
    int idx2 = Rf_asInteger(idx2_s);
    if (idx2 == NA_INTEGER || idx2 <= 0) {
        Rf_error("idx2 must be a positive integer in %s", label);
    }
    if (TYPEOF(di_s) != REALSXP || Rf_length(di_s) != idx2) {
        Rf_error("di length mismatch in %s (expected %d)", label, idx2);
    }
    if (TYPEOF(beta_h0_s) != REALSXP || Rf_length(beta_h0_s) != p) {
        Rf_error("beta_h0 length mismatch in %s (expected %d)", label, p);
    }

    fhl_c_xz_init(&blk->xz, X_s, Z_s, n, NULL, NULL);
    if (qq_expected > 0 && blk->xz.Q != qq_expected) {
        Rf_error("Random-effect dimension mismatch in %s (%d vs %d)", label, blk->xz.Q, qq_expected);
    }
    fhl_c_unpack_risk(Mi_s, &blk->risk);
    if (blk->risk.n != n || blk->risk.m != idx2) {
        Rf_error("Risk-set dimensions (%d x %d) mismatch (%d x %d) in %s",
                 blk->risk.n, blk->risk.m, n, idx2, label);
    }

    blk->n = n;
    blk->p = p;
    blk->qq = blk->xz.Q;
    blk->idx2 = idx2;
    blk->del = REAL(del_s);
    blk->di = REAL(di_s);

    int r_surv = p + blk->qq;
    blk->beta_h = (double *)R_alloc((size_t)(p > 0 ? p : 1), sizeof(double));
    if (p > 0) memcpy(blk->beta_h, REAL(beta_h0_s), (size_t)p * sizeof(double));
    blk->zv_vec = (double *)R_alloc((size_t)n, sizeof(double));
    blk->muh = (double *)R_alloc((size_t)n, sizeof(double));
    blk->Wi_vec = (double *)R_alloc((size_t)n, sizeof(double));
    blk->denom = (double *)R_alloc((size_t)idx2, sizeof(double));
    blk->cla0 = (double *)R_alloc((size_t)idx2, sizeof(double));
    blk->Bi_vec = (double *)R_alloc((size_t)n, sizeof(double));
    blk->Adi_vec = (double *)R_alloc((size_t)idx2, sizeof(double));
    blk->G_surv = (double *)R_alloc((size_t)r_surv * (size_t)r_surv, sizeof(double));
    blk->adj_zz = (double *)R_alloc((size_t)n, sizeof(double));
    blk->w_muh = (double *)R_alloc((size_t)n, sizeof(double));
    blk->tmp_m = (double *)R_alloc((size_t)idx2, sizeof(double));
    blk->tmp_n = (double *)R_alloc((size_t)n, sizeof(double));
    blk->rhs_z_tmp = (double *)R_alloc((size_t)blk->qq, sizeof(double));
}

static void fhl_jm_build_long_gram(
    const fhl_c_xz_t *xz1,
    const double *y1,
    double *G_long,
    double *rhs_long_base
) {
    int n1 = xz1->n;
    int p1 = xz1->p;
    int qq = xz1->Q;
    int r_long = p1 + qq;
    const double *X1 = xz1->X;
    double one = 1.0, zero = 0.0;
    int inc = 1;

    memset(G_long, 0, (size_t)r_long * (size_t)r_long * sizeof(double));

    if (p1 > 0) {
        double *G_xx = (double *)R_alloc((size_t)p1 * (size_t)p1, sizeof(double));
        F77_CALL(dgemm)("T", "N", &p1, &p1, &n1, &one, X1, &n1, X1, &n1, &zero, G_xx, &p1 FCONE FCONE);
        for (int c = 0; c < p1; c++) {
            memcpy(G_long + (size_t)c * (size_t)r_long, G_xx + (size_t)c * (size_t)p1, (size_t)p1 * sizeof(double));
        }
        for (int cz = 0; cz < qq; cz++) {
            int c_pos = p1 + cz;
            int cs = xz1->col_ptr[cz], ce = xz1->col_ptr[cz + 1];
            for (int t = cs; t < ce; t++) {
                int r = xz1->row_idx[t];
                double zval = xz1->col_val[t];
                for (int j = 0; j < p1; j++) {
                    double acc = X1[r + (size_t)n1 * (size_t)j] * zval;
                    G_long[j + (size_t)r_long * (size_t)c_pos] += acc;
                    G_long[c_pos + (size_t)r_long * (size_t)j] += acc;
                }
            }
        }
        F77_CALL(dgemv)("T", &n1, &p1, &one, X1, &n1, y1, &inc, &zero, rhs_long_base, &inc FCONE);
    }

    double *ones_n1 = (double *)R_alloc((size_t)n1, sizeof(double));
    for (int r = 0; r < n1; r++) ones_n1[r] = 1.0;
    double *G_zz = (double *)R_alloc((size_t)qq * (size_t)qq, sizeof(double));
    fhl_c_z_diag_quad(xz1, ones_n1, G_zz);
    for (int c = 0; c < qq; c++) {
        memcpy(G_long + (size_t)(p1 + c) * (size_t)r_long + p1,
               G_zz + (size_t)c * (size_t)qq, (size_t)qq * sizeof(double));
    }
    fhl_c_z_tmv(xz1, y1, rhs_long_base + p1);
}

static void fhl_jm_surv_step_prepare(
    fhl_jm_surv_ctx_t *blk,
    const double *v_h0,
    const double *zv_override,
    double rho,
    double *rhs_x_out,
    double *rhs_z_accum
) {
    int n = blk->n;
    int p = blk->p;
    int qq = blk->qq;
    int idx2 = blk->idx2;
    double one = 1.0, zero = 0.0;
    int inc = 1;

    if (zv_override != NULL) {
        memcpy(blk->zv_vec, zv_override, (size_t)n * sizeof(double));
    } else {
        fhl_c_z_mv(&blk->xz, v_h0, blk->zv_vec);
    }

    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, blk->xz.X, &n, blk->beta_h, &inc, &zero, blk->muh, &inc FCONE);
    } else {
        memset(blk->muh, 0, (size_t)n * sizeof(double));
    }
    for (int r = 0; r < n; r++) {
        blk->muh[r] += rho * blk->zv_vec[r];
        double wi = fhl_c_r_exp(blk->muh[r]);
        if (!R_FINITE(wi)) {
            Rf_error("survival linear predictor overflowed (non-finite exp(eta) in joint model)");
        }
        blk->Wi_vec[r] = wi;
    }

    fhl_c_risk_backward(&blk->risk, blk->Wi_vec, blk->denom, NULL);
    for (int k = 0; k < idx2; k++) {
        double dk = blk->denom[k];
        if (!R_FINITE(dk) || dk <= 0.0) {
            Rf_error("risk-set denominator degenerate in joint model");
        }
        double c_k = blk->di[k] / dk;
        blk->cla0[k] = c_k;
        blk->Adi_vec[k] = (c_k * c_k) / blk->di[k];
    }
    fhl_c_risk_forward(&blk->risk, blk->cla0, blk->Bi_vec, NULL);
    fhl_c_xz_gram0(&blk->xz, 1, &blk->risk, blk->Wi_vec, blk->Bi_vec, blk->Adi_vec, blk->G_surv);

    for (int r = 0; r < n; r++) blk->w_muh[r] = blk->Wi_vec[r] * blk->muh[r];
    fhl_c_risk_backward(&blk->risk, blk->w_muh, blk->tmp_m, NULL);
    for (int k = 0; k < idx2; k++) blk->tmp_m[k] *= blk->Adi_vec[k];
    fhl_c_risk_forward(&blk->risk, blk->tmp_m, blk->tmp_n, NULL);
    for (int r = 0; r < n; r++) {
        double w2_muh = blk->Wi_vec[r] * blk->Bi_vec[r] * blk->muh[r] - blk->Wi_vec[r] * blk->tmp_n[r];
        blk->adj_zz[r] = w2_muh + (blk->del[r] - blk->Bi_vec[r] * blk->Wi_vec[r]);
    }

    if (p > 0) {
        F77_CALL(dgemv)("T", &n, &p, &one, blk->xz.X, &n, blk->adj_zz, &inc, &zero, rhs_x_out, &inc FCONE);
    }
    fhl_c_z_tmv(&blk->xz, blk->adj_zz, blk->rhs_z_tmp);
    for (int j = 0; j < qq; j++) {
        rhs_z_accum[j] += rho * blk->rhs_z_tmp[j];
    }
}

static void fhl_jm_accum_surv_to_H(
    const fhl_jm_surv_ctx_t *blk,
    double rho,
    double *H_jm,
    int r_tot,
    int p_off,
    int pp
) {
    int p = blk->p;
    int qq = blk->qq;
    int r_surv = p + qq;
    const double *G = blk->G_surv;
    double rho2 = rho * rho;

    for (int c = 0; c < p; c++) {
        for (int r = 0; r < p; r++) {
            H_jm[(p_off + r) + (size_t)r_tot * (size_t)(p_off + c)] += G[r + (size_t)r_surv * (size_t)c];
        }
    }
    for (int c = 0; c < qq; c++) {
        for (int r = 0; r < p; r++) {
            double v1 = rho * G[r + (size_t)r_surv * (size_t)(p + c)];
            double v2 = rho * G[(p + c) + (size_t)r_surv * (size_t)r];
            H_jm[(p_off + r) + (size_t)r_tot * (size_t)(pp + c)] += v1;
            H_jm[(pp + c) + (size_t)r_tot * (size_t)(p_off + r)] += v2;
        }
    }
    for (int c = 0; c < qq; c++) {
        for (int r = 0; r < qq; r++) {
            H_jm[(pp + r) + (size_t)r_tot * (size_t)(pp + c)] +=
                rho2 * G[(p + r) + (size_t)r_surv * (size_t)(p + c)];
        }
    }
}

static double fhl_jm_surv_rho_update(
    fhl_jm_surv_ctx_t *blk,
    const double *v_h0,
    const double *hinv,
    int r_tot,
    int p_off,
    int pp,
    double rho
) {
    int n = blk->n;
    int p = blk->p;
    int qq = blk->qq;
    int r_surv = p + qq;
    const double *G = blk->G_surv;

    fhl_c_z_mv(&blk->xz, v_h0, blk->zv_vec);

    /* tr_Hda = sum(hinv * 2 * t(TT1) %*% weight %*% TT).
     * Note weight2_rho cancels algebraically to zero (cla0_rho = -cla0,
     * Adi_rho = -2 * Adi, Bi_rho = -Bi), giving tr_H_rho == 0 identically. */
    double tr_Hda = 0.0;
    for (int c = 0; c < p; c++) {
        for (int r = 0; r < qq; r++) {
            tr_Hda += hinv[(pp + r) + (size_t)r_tot * (size_t)(p_off + c)] *
                      (2.0 * G[(p + r) + (size_t)r_surv * (size_t)c]);
        }
    }
    for (int c = 0; c < qq; c++) {
        for (int r = 0; r < qq; r++) {
            tr_Hda += hinv[(pp + r) + (size_t)r_tot * (size_t)(pp + c)] *
                      (2.0 * rho * G[(p + r) + (size_t)r_surv * (size_t)(p + c)]);
        }
    }

    double dhdrho = 0.0, d2hdrho2 = 0.0;
    for (int r = 0; r < n; r++) {
        double zv = blk->zv_vec[r];
        double wb = blk->Wi_vec[r] * blk->Bi_vec[r];
        dhdrho += (blk->del[r] - wb) * zv;
        d2hdrho2 += (zv * zv) * wb;
    }
    dhdrho -= 0.5 * tr_Hda;
    if (!R_FINITE(d2hdrho2) || d2hdrho2 <= 0.0) {
        Rf_error("association update degenerate: d2h/drho2 = %g (zero frailty variance / identical clusters)", d2hdrho2);
    }
    double rho_new = rho + dhdrho / d2hdrho2;
    if (!R_FINITE(rho_new) || fabs(rho_new) > 1e3) {
        Rf_error("association parameter diverged (rho = %g); joint model not identifiable on these data", rho_new);
    }
    return rho_new;
}

SEXP fhl_fit_jm_cox1(SEXP y1_s, SEXP X1_s, SEXP Z1_s,
                      SEXP del2_s, SEXP X2_s, SEXP Z2_s,
                      SEXP di_s, SEXP Mi_s, SEXP idx2_s,
                      SEXP beta_h0_s, SEXP v_h0_s, SEXP maxiter_s) {
    if (!Rf_isMatrix(X1_s) || TYPEOF(X1_s) != REALSXP) {
        Rf_error("X1 must be a numeric matrix in jmfit2");
    }
    int n1 = Rf_nrows(X1_s), p1 = Rf_ncols(X1_s);
    if (TYPEOF(y1_s) != REALSXP || Rf_length(y1_s) != n1) {
        Rf_error("y1 length mismatch in jmfit2 (expected %d)", n1);
    }
    int maxiter = Rf_asInteger(maxiter_s);
    if (maxiter == NA_INTEGER || maxiter < 1) {
        Rf_error("Maxiter must be a positive integer (>= 1) in jmfit2");
    }

    const double *y1 = REAL(y1_s);
    fhl_c_xz_t xz1;
    fhl_c_xz_init(&xz1, X1_s, Z1_s, n1, NULL, NULL);
    int qq = xz1.Q;

    fhl_jm_surv_ctx_t surv2;
    fhl_jm_init_surv_block(&surv2, del2_s, X2_s, Z2_s, di_s, Mi_s, idx2_s, beta_h0_s, qq, "jmfit2");
    int p2 = surv2.p;
    if (TYPEOF(v_h0_s) != REALSXP || Rf_length(v_h0_s) != qq) {
        Rf_error("v_h0 length mismatch in jmfit2 (expected %d)", qq);
    }

    int pp = p1 + p2;
    int r_tot = pp + qq;
    int r_long = p1 + qq;

    double *G_long = (double *)R_alloc((size_t)r_long * (size_t)r_long, sizeof(double));
    double *rhs_long_base = (double *)R_alloc((size_t)r_long, sizeof(double));
    fhl_jm_build_long_gram(&xz1, y1, G_long, rhs_long_base);

    double *v_h0 = (double *)R_alloc((size_t)qq, sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double rho = 0.0, phi = 1.0, alpha = 0.1;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *H_jm = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *hinv = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *rhs_jm = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *beta_vv = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *old_beta_vv = (double *)R_alloc((size_t)r_tot, sizeof(double));
    memset(old_beta_vv, 0, (size_t)r_tot * sizeof(double));
    if (p2 > 0) memcpy(old_beta_vv + p1, surv2.beta_h, (size_t)p2 * sizeof(double));
    memcpy(old_beta_vv + pp, v_h0, (size_t)qq * sizeof(double));
    double *mu1 = (double *)R_alloc((size_t)n1, sizeof(double));

    double one = 1.0, zero = 0.0;
    int inc = 1;

    while (convergence > 0.0001 && iteration <= maxiter) {
        iteration++;

        memset(rhs_jm, 0, (size_t)r_tot * sizeof(double));
        double inv_phi = 1.0 / phi;
        for (int r = 0; r < r_long; r++) {
            int r_jm = (r < p1) ? r : (pp + (r - p1));
            rhs_jm[r_jm] = rhs_long_base[r] * inv_phi;
        }

        fhl_jm_surv_step_prepare(&surv2, v_h0, NULL, rho, rhs_jm + p1, rhs_jm + pp);

        /* Assemble H_jm ((pp + qq) x (pp + qq)) and invert via Schur complement */
        memset(H_jm, 0, (size_t)r_tot * (size_t)r_tot * sizeof(double));
        for (int c = 0; c < r_long; c++) {
            int c_jm = (c < p1) ? c : (pp + (c - p1));
            for (int r = 0; r < r_long; r++) {
                int r_jm = (r < p1) ? r : (pp + (r - p1));
                H_jm[r_jm + (size_t)r_tot * (size_t)c_jm] = G_long[r + (size_t)r_long * (size_t)c] * inv_phi;
            }
        }
        fhl_jm_accum_surv_to_H(&surv2, rho, H_jm, r_tot, p1, pp);
        double inv_alpha = 1.0 / alpha;
        for (int c = 0; c < qq; c++) {
            H_jm[(pp + c) + (size_t)r_tot * (size_t)(pp + c)] += inv_alpha;
        }

        (void)fhl_jm_require_spd_logdet(H_jm, r_tot, "jmfit2");
        if (!fhl_c_lognormal_inverse(H_jm, pp, qq, NULL, hinv, NULL)) {
            Rf_error("Singular matrix in jmfit2 (H_jm)");
        }

        F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, hinv, &r_tot, rhs_jm, &inc, &zero, beta_vv, &inc FCONE);
        if (p2 > 0) memcpy(surv2.beta_h, beta_vv + p1, (size_t)p2 * sizeof(double));
        memcpy(v_h0, beta_vv + pp, (size_t)qq * sizeof(double));

        double old_phi = phi;
        double old_alpha = alpha;

        /* Update phi */
        fhl_c_z_mv(&xz1, v_h0, mu1);
        if (p1 > 0) {
            F77_CALL(dgemv)("N", &n1, &p1, &one, xz1.X, &n1, beta_vv, &inc, &one, mu1, &inc FCONE);
        }
        double tr_long = 0.0;
        for (int c = 0; c < r_long; c++) {
            int c_jm = (c < p1) ? c : (pp + (c - p1));
            for (int r = 0; r < r_long; r++) {
                int r_jm = (r < p1) ? r : (pp + (r - p1));
                tr_long += hinv[r_jm + (size_t)r_tot * (size_t)c_jm] * G_long[r + (size_t)r_long * (size_t)c];
            }
        }
        double gamma0 = -(1.0 / phi) * tr_long;
        double rss1 = 0.0;
        for (int r = 0; r < n1; r++) {
            double diff = y1[r] - mu1[r];
            rss1 += diff * diff;
        }
        double denom_phi = (double)n1 + gamma0;
        fhl_jm_check_denom(denom_phi, "phi_h", "jmfit2");
        phi = rss1 / denom_phi;
        fhl_jm_check_pos(phi, "phi_h", "jmfit2");

        /* Update alpha */
        double tr_iz = 0.0, ss_v = 0.0;
        for (int j = 0; j < qq; j++) {
            tr_iz += hinv[(pp + j) + (size_t)r_tot * (size_t)(pp + j)];
            ss_v += v_h0[j] * v_h0[j];
        }
        double gamma1 = tr_iz / alpha;
        double denom_alpha = (double)qq - gamma1;
        fhl_jm_check_denom(denom_alpha, "alpha_h", "jmfit2");
        alpha = ss_v / denom_alpha;
        fhl_jm_check_pos(alpha, "alpha_h", "jmfit2");

        /* Update rho using newly updated v_h0 */
        double old_rho = rho;
        rho = fhl_jm_surv_rho_update(&surv2, v_h0, hinv, r_tot, p1, pp, rho);

        double max_bv = 0.0;
        for (int j = 0; j < r_tot; j++) {
            double d = fabs(beta_vv[j] - old_beta_vv[j]);
            if (d > max_bv) max_bv = d;
        }
        memcpy(old_beta_vv, beta_vv, (size_t)r_tot * sizeof(double));
        double conv_disp = fabs(old_phi - phi) + fabs(old_alpha - alpha) + fabs(old_rho - rho);
        convergence = (max_bv > conv_disp) ? max_bv : conv_disp;
        i_count++;
    }

    /* Return list(beta_h, se_beh, v_h0, phi, alpha, rho, i_count, convergence) */
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 8));
    SEXP beta_h_s = PROTECT(Rf_allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(Rf_allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_vv[j];
        REAL(se_beh_s)[j] = sqrt(hinv[j + (size_t)r_tot * (size_t)j]);
    }
    SEXP v_out_s = PROTECT(Rf_allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal(phi));
    SET_VECTOR_ELT(out, 4, Rf_ScalarReal(alpha));
    SET_VECTOR_ELT(out, 5, Rf_ScalarReal(rho));
    SET_VECTOR_ELT(out, 6, Rf_ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 7, Rf_ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}

SEXP fhl_fit_jm_cox2(SEXP y1_s, SEXP X1_s, SEXP Z1_s,
                      SEXP del2_s, SEXP X2_s, SEXP Z2_s,
                      SEXP di2_s, SEXP Mi2_s, SEXP idx2_2_s, SEXP beta_h0_s,
                      SEXP del3_s, SEXP X3_s, SEXP Z3_s,
                      SEXP di3_s, SEXP Mi3_s, SEXP idx2_3_s, SEXP beta_h02_s,
                      SEXP v_h0_s, SEXP maxiter_s) {
    if (!Rf_isMatrix(X1_s) || TYPEOF(X1_s) != REALSXP) {
        Rf_error("X1 must be a numeric matrix in jmfit3");
    }
    int n1 = Rf_nrows(X1_s), p1 = Rf_ncols(X1_s);
    if (TYPEOF(y1_s) != REALSXP || Rf_length(y1_s) != n1) {
        Rf_error("y1 length mismatch in jmfit3 (expected %d)", n1);
    }
    int maxiter = Rf_asInteger(maxiter_s);
    if (maxiter == NA_INTEGER || maxiter < 1) {
        Rf_error("Maxiter must be a positive integer (>= 1) in jmfit3");
    }

    const double *y1 = REAL(y1_s);
    fhl_c_xz_t xz1;
    fhl_c_xz_init(&xz1, X1_s, Z1_s, n1, NULL, NULL);
    int qq = xz1.Q;

    fhl_jm_surv_ctx_t surv2, surv3;
    fhl_jm_init_surv_block(&surv2, del2_s, X2_s, Z2_s, di2_s, Mi2_s, idx2_2_s, beta_h0_s, qq, "jmfit3(surv1)");
    fhl_jm_init_surv_block(&surv3, del3_s, X3_s, Z3_s, di3_s, Mi3_s, idx2_3_s, beta_h02_s, qq, "jmfit3(surv2)");
    if (TYPEOF(v_h0_s) != REALSXP || Rf_length(v_h0_s) != qq) {
        Rf_error("v_h0 length mismatch in jmfit3 (expected %d)", qq);
    }

    int p2 = surv2.p;
    int p3 = surv3.p;
    int pp = p1 + p2 + p3;
    int r_tot = pp + qq;
    int r_long = p1 + qq;

    double *G_long = (double *)R_alloc((size_t)r_long * (size_t)r_long, sizeof(double));
    double *rhs_long_base = (double *)R_alloc((size_t)r_long, sizeof(double));
    fhl_jm_build_long_gram(&xz1, y1, G_long, rhs_long_base);

    double *v_h0 = (double *)R_alloc((size_t)qq, sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double rho = 1.0, rho2 = 1.0, phi = 1.0, alpha = 0.1;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *H_jm = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *hinv = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *rhs_jm = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *beta_vv = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *old_beta_vv = (double *)R_alloc((size_t)r_tot, sizeof(double));
    memset(old_beta_vv, 0, (size_t)r_tot * sizeof(double));
    if (p2 > 0) memcpy(old_beta_vv + p1, surv2.beta_h, (size_t)p2 * sizeof(double));
    if (p3 > 0) memcpy(old_beta_vv + p1 + p2, surv3.beta_h, (size_t)p3 * sizeof(double));
    memcpy(old_beta_vv + pp, v_h0, (size_t)qq * sizeof(double));
    double *mu1 = (double *)R_alloc((size_t)n1, sizeof(double));

    double one = 1.0, zero = 0.0;
    int inc = 1;

    while (convergence > 0.0001 && iteration <= maxiter) {
        iteration++;

        memset(rhs_jm, 0, (size_t)r_tot * sizeof(double));
        double inv_phi = 1.0 / phi;
        for (int r = 0; r < r_long; r++) {
            int r_jm = (r < p1) ? r : (pp + (r - p1));
            rhs_jm[r_jm] = rhs_long_base[r] * inv_phi;
        }

        fhl_jm_surv_step_prepare(&surv2, v_h0, NULL, rho, rhs_jm + p1, rhs_jm + pp);
        fhl_jm_surv_step_prepare(&surv3, v_h0, NULL, rho2, rhs_jm + p1 + p2, rhs_jm + pp);

        memset(H_jm, 0, (size_t)r_tot * (size_t)r_tot * sizeof(double));
        for (int c = 0; c < r_long; c++) {
            int c_jm = (c < p1) ? c : (pp + (c - p1));
            for (int r = 0; r < r_long; r++) {
                int r_jm = (r < p1) ? r : (pp + (r - p1));
                H_jm[r_jm + (size_t)r_tot * (size_t)c_jm] = G_long[r + (size_t)r_long * (size_t)c] * inv_phi;
            }
        }
        fhl_jm_accum_surv_to_H(&surv2, rho, H_jm, r_tot, p1, pp);
        fhl_jm_accum_surv_to_H(&surv3, rho2, H_jm, r_tot, p1 + p2, pp);
        double inv_alpha = 1.0 / alpha;
        for (int c = 0; c < qq; c++) {
            H_jm[(pp + c) + (size_t)r_tot * (size_t)(pp + c)] += inv_alpha;
        }

        (void)fhl_jm_require_spd_logdet(H_jm, r_tot, "jmfit3");
        if (!fhl_c_lognormal_inverse(H_jm, pp, qq, NULL, hinv, NULL)) {
            Rf_error("Singular matrix in jmfit3 (H_jm)");
        }

        F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, hinv, &r_tot, rhs_jm, &inc, &zero, beta_vv, &inc FCONE);
        if (p2 > 0) memcpy(surv2.beta_h, beta_vv + p1, (size_t)p2 * sizeof(double));
        if (p3 > 0) memcpy(surv3.beta_h, beta_vv + p1 + p2, (size_t)p3 * sizeof(double));
        memcpy(v_h0, beta_vv + pp, (size_t)qq * sizeof(double));

        double old_phi = phi;
        double old_alpha = alpha;

        /* Update phi */
        fhl_c_z_mv(&xz1, v_h0, mu1);
        if (p1 > 0) {
            F77_CALL(dgemv)("N", &n1, &p1, &one, xz1.X, &n1, beta_vv, &inc, &one, mu1, &inc FCONE);
        }
        double tr_long = 0.0;
        for (int c = 0; c < r_long; c++) {
            int c_jm = (c < p1) ? c : (pp + (c - p1));
            for (int r = 0; r < r_long; r++) {
                int r_jm = (r < p1) ? r : (pp + (r - p1));
                tr_long += hinv[r_jm + (size_t)r_tot * (size_t)c_jm] * G_long[r + (size_t)r_long * (size_t)c];
            }
        }
        double gamma0 = -(1.0 / phi) * tr_long;
        double rss1 = 0.0;
        for (int r = 0; r < n1; r++) {
            double diff = y1[r] - mu1[r];
            rss1 += diff * diff;
        }
        double denom_phi = (double)n1 + gamma0;
        fhl_jm_check_denom(denom_phi, "phi_h", "jmfit3");
        phi = rss1 / denom_phi;
        fhl_jm_check_pos(phi, "phi_h", "jmfit3");

        /* Update alpha */
        double tr_iz = 0.0, ss_v = 0.0;
        for (int j = 0; j < qq; j++) {
            tr_iz += hinv[(pp + j) + (size_t)r_tot * (size_t)(pp + j)];
            ss_v += v_h0[j] * v_h0[j];
        }
        double gamma1 = tr_iz / alpha;
        double denom_alpha = (double)qq - gamma1;
        fhl_jm_check_denom(denom_alpha, "alpha_h", "jmfit3");
        alpha = ss_v / denom_alpha;
        fhl_jm_check_pos(alpha, "alpha_h", "jmfit3");

        /* Update rho and rho2 */
        double old_rho = rho;
        rho = fhl_jm_surv_rho_update(&surv2, v_h0, hinv, r_tot, p1, pp, rho);

        double old_rho2 = rho2;
        rho2 = fhl_jm_surv_rho_update(&surv3, v_h0, hinv, r_tot, p1 + p2, pp, rho2);

        double max_bv = 0.0;
        for (int j = 0; j < r_tot; j++) {
            double d = fabs(beta_vv[j] - old_beta_vv[j]);
            if (d > max_bv) max_bv = d;
        }
        memcpy(old_beta_vv, beta_vv, (size_t)r_tot * sizeof(double));
        double conv_disp = fabs(old_phi - phi) + fabs(old_alpha - alpha) +
                           fabs(old_rho - rho) + fabs(old_rho2 - rho2);
        convergence = (max_bv > conv_disp) ? max_bv : conv_disp;
        i_count++;
    }

    /* Return list(beta_h, se_beh, v_h0, phi, alpha, rho, rho2, i_count, convergence) */
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 9));
    SEXP beta_h_s = PROTECT(Rf_allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(Rf_allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_vv[j];
        REAL(se_beh_s)[j] = sqrt(hinv[j + (size_t)r_tot * (size_t)j]);
    }
    SEXP v_out_s = PROTECT(Rf_allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal(phi));
    SET_VECTOR_ELT(out, 4, Rf_ScalarReal(alpha));
    SET_VECTOR_ELT(out, 5, Rf_ScalarReal(rho));
    SET_VECTOR_ELT(out, 6, Rf_ScalarReal(rho2));
    SET_VECTOR_ELT(out, 7, Rf_ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 8, Rf_ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}

SEXP fhl_fit_jm_frailty(SEXP del1_s, SEXP X1_s, SEXP Z1_s,
                        SEXP di1_s, SEXP Mi1_s, SEXP idx2_1_s, SEXP beta_h01_s,
                        SEXP del2_s, SEXP X2_s, SEXP Z2_s,
                        SEXP di2_s, SEXP Mi2_s, SEXP idx2_2_s, SEXP beta_h02_s,
                        SEXP v_h0_s, SEXP maxiter_s) {
    int maxiter = Rf_asInteger(maxiter_s);
    if (maxiter == NA_INTEGER || maxiter < 1) {
        Rf_error("Maxiter must be >= 1 in jmfit_FM");
    }

    fhl_jm_surv_ctx_t surv1, surv2;
    fhl_jm_init_surv_block(&surv1, del1_s, X1_s, Z1_s, di1_s, Mi1_s, idx2_1_s, beta_h01_s, -1, "jmfit_FM(surv1)");
    int qq = surv1.qq;
    fhl_jm_init_surv_block(&surv2, del2_s, X2_s, Z2_s, di2_s, Mi2_s, idx2_2_s, beta_h02_s, qq, "jmfit_FM(surv2)");
    if (surv1.n != surv2.n) {
        Rf_error("Observation counts (%d vs %d) must match in jmfit_FM", surv1.n, surv2.n);
    }
    if (TYPEOF(v_h0_s) != REALSXP || Rf_length(v_h0_s) != qq) {
        Rf_error("v_h0 length mismatch in jmfit_FM (expected %d)", qq);
    }

    int n = surv1.n;
    int p1 = surv1.p;
    int p2 = surv2.p;
    int pp = p1 + p2;
    int r_tot = pp + qq;

    double *v_h0 = (double *)R_alloc((size_t)qq, sizeof(double));
    memcpy(v_h0, REAL(v_h0_s), (size_t)qq * sizeof(double));

    double rho = 0.0, alpha = 1.0;
    double convergence = 1.0;
    int iteration = 1, i_count = 0;

    double *H_jm = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *hinv = (double *)R_alloc((size_t)r_tot * (size_t)r_tot, sizeof(double));
    double *rhs_jm = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *beta_vv = (double *)R_alloc((size_t)r_tot, sizeof(double));
    double *old_beta_h = (double *)R_alloc((size_t)(pp > 0 ? pp : 1), sizeof(double));
    double *old_v_h0 = (double *)R_alloc((size_t)qq, sizeof(double));

    double one = 1.0, zero = 0.0;
    int inc = 1;
    double last_gamma1 = 0.0, last_ss_v = 0.0, last_logdet_H = 0.0;

    while (convergence > 0.000001 && iteration <= maxiter) {
        iteration++;
        memcpy(old_v_h0, v_h0, (size_t)qq * sizeof(double));

        /* Evaluate each survival submodel on its own aligned random-effect design Z */
        memset(rhs_jm, 0, (size_t)r_tot * sizeof(double));
        fhl_jm_surv_step_prepare(&surv1, v_h0, NULL, 1.0, rhs_jm, rhs_jm + pp);
        fhl_jm_surv_step_prepare(&surv2, v_h0, NULL, rho, rhs_jm + p1, rhs_jm + pp);

        memset(H_jm, 0, (size_t)r_tot * (size_t)r_tot * sizeof(double));
        fhl_jm_accum_surv_to_H(&surv1, 1.0, H_jm, r_tot, 0, pp);
        fhl_jm_accum_surv_to_H(&surv2, rho, H_jm, r_tot, p1, pp);
        double inv_alpha = 1.0 / alpha;
        for (int c = 0; c < qq; c++) {
            H_jm[(pp + c) + (size_t)r_tot * (size_t)(pp + c)] += inv_alpha;
        }

        last_logdet_H = fhl_jm_require_spd_logdet(H_jm, r_tot, "jmfit_FM");
        if (!fhl_c_lognormal_inverse(H_jm, pp, qq, NULL, hinv, NULL)) {
            Rf_error("Singular matrix in jmfit_FM (H_jm)");
        }

        if (p1 > 0) memcpy(old_beta_h, surv1.beta_h, (size_t)p1 * sizeof(double));
        if (p2 > 0) memcpy(old_beta_h + p1, surv2.beta_h, (size_t)p2 * sizeof(double));

        F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, hinv, &r_tot, rhs_jm, &inc, &zero, beta_vv, &inc FCONE);
        if (p1 > 0) memcpy(surv1.beta_h, beta_vv, (size_t)p1 * sizeof(double));
        if (p2 > 0) memcpy(surv2.beta_h, beta_vv + p1, (size_t)p2 * sizeof(double));
        memcpy(v_h0, beta_vv + pp, (size_t)qq * sizeof(double));

        double old_alpha = alpha;
        double tr_iz = 0.0, ss_v = 0.0;
        for (int j = 0; j < qq; j++) {
            tr_iz += hinv[(pp + j) + (size_t)r_tot * (size_t)(pp + j)];
            ss_v += v_h0[j] * v_h0[j];
        }
        double gamma1 = tr_iz / alpha;
        double denom_alpha = (double)qq - gamma1;
        fhl_jm_check_denom(denom_alpha, "alpha_h", "jmfit_FM");
        alpha = ss_v / denom_alpha;
        fhl_jm_check_pos(alpha, "alpha_h", "jmfit_FM");
        last_gamma1 = gamma1;
        last_ss_v = ss_v;

        double old_rho = rho;
        rho = fhl_jm_surv_rho_update(&surv2, v_h0, hinv, r_tot, p1, pp, rho);

        double diff_beta = 0.0;
        double max_bv = 0.0;
        for (int j = 0; j < pp; j++) {
            double d = fabs(old_beta_h[j] - beta_vv[j]);
            diff_beta += d;
            if (d > max_bv) max_bv = d;
        }
        for (int j = 0; j < qq; j++) {
            double d = fabs(v_h0[j] - old_v_h0[j]);
            if (d > max_bv) max_bv = d;
        }
        double conv_disp = fabs(old_alpha - alpha) + fabs(old_rho - rho) + diff_beta;
        convergence = (max_bv > conv_disp) ? max_bv : conv_disp;
        i_count++;
    }

    /* Post-processing: h0, hp, p.bvh, cAIC, rAIC */
    double hlike01 = 0.0, hlike02 = 0.0;
    for (int r = 0; r < n; r++) {
        hlike01 += surv1.del[r] * surv1.muh[r];
        hlike02 += surv2.del[r] * surv2.muh[r];
    }
    for (int k = 0; k < surv1.idx2; k++) {
        hlike01 -= surv1.di[k] * fhl_c_r_log(surv1.denom[k]);
    }
    for (int k = 0; k < surv2.idx2; k++) {
        hlike02 -= surv2.di[k] * fhl_c_r_log(surv2.denom[k]);
    }
    double h0 = hlike01 + hlike02;
    const double pi_v23 = 3.14159265359;
    double log_2pi = fhl_c_r_log(2.0 * pi_v23);
    double hlike2 = -0.5 * (double)qq * log_2pi - 0.5 * (double)qq * fhl_c_r_log(alpha);
    double hlike3 = -last_ss_v / (2.0 * alpha);
    double hp = h0 + hlike2 + hlike3;

    double log_det_hinv = -last_logdet_H;
    double p_bvh = hp + 0.5 * log_det_hinv + 0.5 * (double)r_tot * log_2pi;

    /* dfc = tr(hinv %*% H0) = r_tot - gamma1 since H0 = H_jm - diag(0, 1/old_alpha) */
    double dfc = (double)r_tot - last_gamma1;
    double cAIC = -2.0 * h0 + 2.0 * dfc;
    double rAIC = -2.0 * p_bvh + 4.0;

    /* Return named result list (beta_h, se_beh, v_h0, phi, alpha, rho, iter, likelihoods, AIC, convergence) */
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 12));
    SEXP beta_h_s = PROTECT(Rf_allocVector(REALSXP, pp));
    SEXP se_beh_s = PROTECT(Rf_allocVector(REALSXP, pp));
    for (int j = 0; j < pp; j++) {
        REAL(beta_h_s)[j] = beta_vv[j];
        REAL(se_beh_s)[j] = sqrt(hinv[j + (size_t)r_tot * (size_t)j]);
    }
    SEXP v_out_s = PROTECT(Rf_allocVector(REALSXP, qq));
    memcpy(REAL(v_out_s), v_h0, (size_t)qq * sizeof(double));

    SET_VECTOR_ELT(out, 0, beta_h_s);
    SET_VECTOR_ELT(out, 1, se_beh_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, Rf_ScalarReal(alpha));
    SET_VECTOR_ELT(out, 4, Rf_ScalarReal(rho));
    SET_VECTOR_ELT(out, 5, Rf_ScalarInteger(i_count));
    SET_VECTOR_ELT(out, 6, Rf_ScalarReal(h0));
    SET_VECTOR_ELT(out, 7, Rf_ScalarReal(hp));
    SET_VECTOR_ELT(out, 8, Rf_ScalarReal(p_bvh));
    SET_VECTOR_ELT(out, 9, Rf_ScalarReal(cAIC));
    SET_VECTOR_ELT(out, 10, Rf_ScalarReal(rAIC));
    SET_VECTOR_ELT(out, 11, Rf_ScalarReal(convergence));

    UNPROTECT(4);
    return out;
}
