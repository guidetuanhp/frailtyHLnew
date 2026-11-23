#include "fhl_linalg.h"
#include "fhl_native_api.h"
#include <stdarg.h>
#include <stdio.h>
#include <float.h>

#undef exp
#undef log
#define exp(x) fhl_c_r_exp(x)
#define log(x) fhl_c_r_log(x)

#ifndef FHL_INLINE
#define FHL_INLINE inline
#endif

/* ==========================================================================
 * Penalized Cox Frailty Variable Selection Engine (1-RE/Multi-RE & Correlated 2-RE)
 * ========================================================================== */

#define FHL_VS_ERR_CAP 256

static int fhl_vs_set_err(char *err_buf, size_t err_cap, const char *fmt, ...) {
    if (err_buf != NULL && err_cap > 0) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err_buf, err_cap, fmt, ap);
        va_end(ap);
        err_buf[err_cap - 1] = '\0';
    }
    return 0;
}

static int fhl_vs_all_finite(const double *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!R_FINITE(a[i])) return 0;
    }
    return 1;
}

static int fhl_vs_penalty_weights(int pen_type, int n, int p, const double *B,
                                  double tun1, double tun2,
                                  double *hft, double *wl_diag,
                                  char *err_buf, size_t err_cap) {
    if (pen_type == 0) { /* "lasso" */
        for (int i = 0; i < p; i++) hft[i] = tun1;
    } else if (pen_type == 1) { /* "scad" */
        double lam = tun1;
        double a = 3.7;
        for (int i = 0; i < p; i++) {
            double absB = fabs(B[i]);
            hft[i] = 0.0;
            if (0.0 <= absB && lam > absB) {
                hft[i] = lam;
            }
            if (lam <= absB && a * lam > absB) {
                hft[i] = (a * lam - absB) / (a - 1.0);
            }
        }
    } else if (pen_type == 2) { /* "hl" */
        double tau = tun1;
        double sig = tun2;
        double tm2_sq = (2.0 - tau) * (2.0 - tau);
        for (int i = 0; i < p; i++) {
            double absB = fabs(B[i]);
            double rad = sqrt(8.0 * tau * (B[i] * B[i]) / sig + tm2_sq);
            double u_i = (rad + 2.0 - tau) / 4.0;
            double denom = sqrt(8.0 * tau * (absB * absB) / sig + tm2_sq);
            double delu_i;
            if (denom > 0.0 && R_FINITE(denom)) {
                delu_i = (2.0 * tau * absB / sig) / denom;
            } else if (tau == 2.0 && sig > 0.0 && R_FINITE(sig)) {
                delu_i = 1.0 / sqrt(sig);
            } else {
                return fhl_vs_set_err(err_buf, err_cap, "HL penalty derivative undefined");
            }
            hft[i] = (((tau - 2.0) / (2.0 * tau) * delu_i / (u_i + 0.00000001)) + 2.0 * delu_i / tau) / (double)n;
        }
    }
    for (int i = 0; i < p; i++) {
        wl_diag[i] = hft[i] / (fabs(B[i]) + 0.00000001);
    }
    return 1;
}

/* Solve Newton step H * step = dft using strict SPD Cholesky + Schur complement on G_zz_u */
static int fhl_vs_schur_solve_step(int n, int p, int q, int r_tot,
                                   const double *G_mat, const double *wl_diag,
                                   const double *u_diag, const double *dft,
                                   double *step_out, const char *err_tag,
                                   char *err_buf, size_t err_cap) {
    if (!fhl_vs_all_finite(G_mat, (size_t)r_tot * r_tot) ||
        !fhl_vs_all_finite(wl_diag, (size_t)p) ||
        !fhl_vs_all_finite(u_diag, (size_t)q) ||
        !fhl_vs_all_finite(dft, (size_t)r_tot)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
    }
    int info = 0;
    double *chol_u = (double *)R_alloc((size_t)q * q, sizeof(double));
    for (int c = 0; c < q; c++) {
        memcpy(chol_u + (size_t)q * c,
               G_mat + p + (size_t)r_tot * (p + c),
               (size_t)q * sizeof(double));
        chol_u[c + (size_t)q * c] += u_diag[c];
        if (chol_u[c + (size_t)q * c] <= 0.0) {
            return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
        }
    }
    F77_CALL(dpotrf)("U", &q, chol_u, &q, &info FCONE);
    if (info != 0) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
    }

    if (p == 0) {
        memcpy(step_out, dft, (size_t)q * sizeof(double));
        int nrhs = 1;
        F77_CALL(dpotrs)("U", &q, &nrhs, chol_u, &q, step_out, &q, &info FCONE);
        if (info != 0 || !fhl_vs_all_finite(step_out, (size_t)q)) {
            return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
        }
        return 1;
    }

    int nrhs = p + 1;
    double *YR = (double *)R_alloc((size_t)q * nrhs, sizeof(double));
    for (int c = 0; c < p; c++) {
        memcpy(YR + (size_t)q * c,
               G_mat + p + (size_t)r_tot * c,
               (size_t)q * sizeof(double));
    }
    memcpy(YR + (size_t)q * p, dft + p, (size_t)q * sizeof(double));
    F77_CALL(dpotrs)("U", &q, &nrhs, chol_u, &q, YR, &q, &info FCONE);
    if (info != 0) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
    }

    double *S_rhs = (double *)R_alloc((size_t)p * (p + 1), sizeof(double));
    for (int c = 0; c < p; c++) {
        memcpy(S_rhs + (size_t)p * c, G_mat + (size_t)r_tot * c, (size_t)p * sizeof(double));
        S_rhs[c + (size_t)p * c] += (double)n * wl_diag[c];
    }
    memcpy(S_rhs + (size_t)p * p, dft, (size_t)p * sizeof(double));
    double neg1 = -1.0, one = 1.0;
    F77_CALL(dgemm)("N", "N", &p, &nrhs, &q, &neg1,
                    G_mat + (size_t)r_tot * p, &r_tot,
                    YR, &q, &one, S_rhs, &p FCONE FCONE);
    if (!fhl_c_solve_spd(S_rhs, S_rhs + (size_t)p * p, step_out, p, 1)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
    }
    memcpy(step_out + p, YR + (size_t)q * p, (size_t)q * sizeof(double));
    int inc = 1;
    F77_CALL(dgemv)("N", &q, &p, &neg1, YR, &q, step_out, &inc, &one, step_out + p, &inc FCONE);
    if (!fhl_vs_all_finite(step_out, (size_t)r_tot)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in %s", err_tag);
    }
    return 1;
}

/* Compute both H22 = (G_zz + U)^{-1} and Hinv = H^{-1} using strict SPD Cholesky + Schur complement.
 * Exactly one of u_diag (diagonal U) or iS0 (Kronecker U = iS0 %x% I_{q1}) must be non-NULL. */
static int fhl_vs_schur_invert_both(int n, int p, int q, int r_tot,
                                    const double *G_mat, const double *wl_diag,
                                    const double *u_diag, const double *iS0, int nrand, int q1,
                                    double *H22_out, double *Hinv_out, const char *err_tag,
                                    char *err_buf, size_t err_cap) {
    if (!fhl_vs_all_finite(G_mat, (size_t)r_tot * r_tot) ||
        !fhl_vs_all_finite(wl_diag, (size_t)p)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix H in %s", err_tag);
    }
    for (int c = 0; c < q; c++) {
        memcpy(H22_out + (size_t)q * c,
               G_mat + p + (size_t)r_tot * (p + c),
               (size_t)q * sizeof(double));
    }
    if (u_diag != NULL) {
        if (!fhl_vs_all_finite(u_diag, (size_t)q)) {
            return fhl_vs_set_err(err_buf, err_cap, "Singular matrix H in %s", err_tag);
        }
        for (int j = 0; j < q; j++) H22_out[j + (size_t)q * j] += u_diag[j];
    } else {
        for (int c_blk = 0; c_blk < nrand; c_blk++) {
            for (int r_blk = 0; r_blk < nrand; r_blk++) {
                double val = iS0[r_blk + nrand * c_blk];
                for (int k = 0; k < q1; k++) {
                    int r = r_blk * q1 + k;
                    int c = c_blk * q1 + k;
                    H22_out[r + (size_t)q * c] += val;
                }
            }
        }
    }
    for (int j = 0; j < q; j++) {
        if (H22_out[j + (size_t)q * j] <= 0.0) {
            return fhl_vs_set_err(err_buf, err_cap, "Singular matrix G_zz_u in %s", err_tag);
        }
    }

    int info = 0;
    F77_CALL(dpotrf)("U", &q, H22_out, &q, &info FCONE);
    if (info != 0) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix G_zz_u in %s", err_tag);
    }

    double *Y = (p > 0) ? (double *)R_alloc((size_t)q * p, sizeof(double)) : NULL;
    if (p > 0) {
        for (int c = 0; c < p; c++) {
            memcpy(Y + (size_t)q * c,
                   G_mat + p + (size_t)r_tot * c,
                   (size_t)q * sizeof(double));
        }
        int info_s = 0;
        F77_CALL(dpotrs)("U", &q, &p, H22_out, &q, Y, &q, &info_s FCONE);
        if (info_s != 0) {
            return fhl_vs_set_err(err_buf, err_cap, "Singular matrix G_zz_u in %s", err_tag);
        }
    }
    int info_i = 0;
    F77_CALL(dpotri)("U", &q, H22_out, &q, &info_i FCONE);
    if (info_i != 0) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix G_zz_u in %s", err_tag);
    }
    for (int c = 0; c < q; c++) {
        for (int r = c + 1; r < q; r++) {
            H22_out[r + (size_t)q * c] = H22_out[c + (size_t)q * r];
        }
    }
    if (!fhl_vs_all_finite(H22_out, (size_t)q * q)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix G_zz_u in %s", err_tag);
    }
    if (p == 0) {
        memcpy(Hinv_out, H22_out, (size_t)q * q * sizeof(double));
        return 1;
    }

    double *S = (double *)R_alloc((size_t)p * p, sizeof(double));
    double *S_inv = (double *)R_alloc((size_t)p * p, sizeof(double));
    for (int c = 0; c < p; c++) {
        memcpy(S + (size_t)p * c, G_mat + (size_t)r_tot * c, (size_t)p * sizeof(double));
        S[c + (size_t)p * c] += (double)n * wl_diag[c];
    }
    double neg1 = -1.0, one = 1.0, zero = 0.0;
    F77_CALL(dgemm)("N", "N", &p, &p, &q, &neg1,
                    G_mat + (size_t)r_tot * p, &r_tot,
                    Y, &q, &one, S, &p FCONE FCONE);
    if (!fhl_c_solve_inv(S, S_inv, p)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix H in %s", err_tag);
    }
    for (int c = 0; c < p; c++) {
        memcpy(Hinv_out + (size_t)r_tot * c, S_inv + (size_t)p * c, (size_t)p * sizeof(double));
    }
    double *top_right = (double *)R_alloc((size_t)p * q, sizeof(double));
    F77_CALL(dgemm)("N", "T", &p, &q, &p, &neg1,
                    S_inv, &p, Y, &q, &zero, top_right, &p FCONE FCONE);
    for (int c = 0; c < q; c++) {
        memcpy(Hinv_out + (size_t)r_tot * (p + c), top_right + (size_t)p * c, (size_t)p * sizeof(double));
        for (int r = 0; r < p; r++) {
            Hinv_out[(p + c) + (size_t)r_tot * r] = top_right[r + (size_t)p * c];
        }
        memcpy(Hinv_out + p + (size_t)r_tot * (p + c), H22_out + (size_t)q * c, (size_t)q * sizeof(double));
    }
    F77_CALL(dgemm)("N", "N", &q, &q, &p, &neg1,
                    Y, &q, top_right, &p, &one,
                    Hinv_out + p + (size_t)r_tot * p, &r_tot FCONE FCONE);
    if (!fhl_vs_all_finite(Hinv_out, (size_t)r_tot * r_tot)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix H in %s", err_tag);
    }
    return 1;
}

/* Invert small (nrand x nrand) symmetric positive definite block covariance S0
 * via strict Cholesky (dpotrf + dpotri) and optionally compute log(det(S0)). */
static int fhl_vs_kron_s0_inv_logdet(int nrand, int n_alpha, const double *alphas,
                                     double *iS0_out, double *logdet_S0_out) {
    double *S0 = (double *)R_alloc((size_t)nrand * nrand, sizeof(double));
    double alpha_new = (n_alpha > nrand) ? alphas[n_alpha - 1] : 0.0;
    for (int c = 0; c < nrand; c++) {
        for (int r = 0; r < nrand; r++) {
            S0[r + nrand * c] = (r == c) ? alphas[r] : alpha_new;
        }
    }
    if (!fhl_vs_all_finite(S0, (size_t)nrand * nrand)) return 0;
    for (int r = 0; r < nrand; r++) {
        if (S0[r + nrand * r] <= 0.0) return 0;
    }
    int info = 0;
    F77_CALL(dpotrf)("U", &nrand, S0, &nrand, &info FCONE);
    if (info != 0) return 0;
    double ld = 0.0;
    for (int j = 0; j < nrand; j++) {
        double d = S0[j + nrand * j];
        if (!R_FINITE(d) || d <= 0.0) return 0;
        ld += 2.0 * log(d);
    }
    if (!R_FINITE(ld)) return 0;
    if (logdet_S0_out != NULL) {
        *logdet_S0_out = ld;
    }
    F77_CALL(dpotri)("U", &nrand, S0, &nrand, &info FCONE);
    if (info != 0) return 0;
    for (int c = 0; c < nrand; c++) {
        for (int r = 0; r <= c; r++) {
            double val = S0[r + nrand * c];
            if (!R_FINITE(val)) return 0;
            iS0_out[r + nrand * c] = val;
            iS0_out[c + nrand * r] = val;
        }
    }
    return 1;
}

/* y = (M0 %x% I_{q1}) * x, O(nrand^2 * q1) */
static void fhl_vs_kron_mv(int nrand, int q1, const double *M0, const double *x, double *y) {
    for (int r_blk = 0; r_blk < nrand; r_blk++) {
        double *yr = y + (size_t)r_blk * q1;
        memset(yr, 0, (size_t)q1 * sizeof(double));
        for (int c_blk = 0; c_blk < nrand; c_blk++) {
            double w = M0[r_blk + nrand * c_blk];
            const double *xc = x + (size_t)c_blk * q1;
            for (int k = 0; k < q1; k++) yr[k] += w * xc[k];
        }
    }
}

/* Single penalized h-likelihood iteration for diagonal random-effect covariance */
static int fhl_c_lognorm1_core(int n, int p, int q, int nrand,
                               const int *qs, const int *rand_idx,
                               const fhl_c_xz_t *xz,
                               const fhl_c_risk_t *risk, const double *di, const double *del,
                               int pen_type, double tun1, double tun2,
                               int varfixed, int keep_alpha_when_fixed,
                               const double *B_in, const double *v_in, const double *alphas_in,
                               double *B_out, double *v_out, double *alphas_out,
                               double *G_mat, double *wl_diag, double *clam0, double *hft,
                               char *err_buf, size_t err_cap) {
    int r_tot = p + q;
    int m = risk->m;
    double one = 1.0, zero = 0.0;
    const double *X = xz->X;

    double *alpha_rep = (double *)R_alloc(q, sizeof(double));
    double *u_diag = (double *)R_alloc(q, sizeof(double));
    for (int j = 0; j < q; j++) {
        alpha_rep[j] = alphas_in[rand_idx[j]];
        u_diag[j] = 1.0 / alpha_rep[j];
    }

    double *eta = (double *)R_alloc(n, sizeof(double));
    double *zv = (double *)R_alloc(n, sizeof(double));
    double *expeta = (double *)R_alloc(n, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, X, &n, B_in, & (int){1}, &zero, eta, & (int){1} FCONE);
    } else {
        memset(eta, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v_in, zv);
    for (int r = 0; r < n; r++) {
        eta[r] += zv[r];
        expeta[r] = exp(eta[r]);
    }

    double *denom = (double *)R_alloc(m, sizeof(double));
    double *cla0 = (double *)R_alloc(m, sizeof(double));
    double *as_vec = (double *)R_alloc(m, sizeof(double));
    fhl_c_risk_backward(risk, expeta, denom, NULL);
    for (int k = 0; k < m; k++) {
        cla0[k] = di[k] / denom[k];
        as_vec[k] = (cla0[k] * cla0[k]) / di[k];
    }
    fhl_c_risk_forward(risk, cla0, clam0, NULL);

    if (!fhl_vs_penalty_weights(pen_type, n, p, B_in, tun1, tun2, hft, wl_diag, err_buf, err_cap)) {
        return 0;
    }

    double *res_vec = (double *)R_alloc(n, sizeof(double));
    for (int r = 0; r < n; r++) res_vec[r] = del[r] - expeta[r] * clam0[r];

    double *dft = (double *)R_alloc(r_tot, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("T", &n, &p, &one, X, &n, res_vec, & (int){1}, &zero, dft, & (int){1} FCONE);
        for (int j = 0; j < p; j++) dft[j] -= (double)n * wl_diag[j] * B_in[j];
    }
    fhl_c_z_tmv(xz, res_vec, dft + p);
    for (int j = 0; j < q; j++) dft[p + j] -= v_in[j] * u_diag[j];

    fhl_c_xz_gram0(xz, 1, risk, expeta, clam0, as_vec, G_mat);

    if (varfixed) {
        double *step = (double *)R_alloc(r_tot, sizeof(double));
        if (!fhl_vs_schur_solve_step(n, p, q, r_tot, G_mat, wl_diag, u_diag, dft, step, "lognorm1 (H)", err_buf, err_cap)) {
            return 0;
        }
        for (int j = 0; j < p; j++) B_out[j] = B_in[j] + step[j];
        for (int j = 0; j < q; j++) v_out[j] = v_in[j] + step[p + j];
        for (int i = 0; i < nrand; i++) {
            alphas_out[i] = keep_alpha_when_fixed ? alphas_in[i] : 0.0;
        }
        return 1;
    }

    double *H22 = (double *)R_alloc((size_t)q * q, sizeof(double));
    double *Hinv = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    if (!fhl_vs_schur_invert_both(n, p, q, r_tot, G_mat, wl_diag, u_diag, NULL, 0, 0,
                                  H22, Hinv, "lognorm1", err_buf, err_cap)) {
        return 0;
    }

    for (int j = 0; j < p; j++) {
        double step = 0.0;
        for (int c = 0; c < r_tot; c++) step += Hinv[j + (size_t)r_tot * c] * dft[c];
        B_out[j] = B_in[j] + step;
    }
    for (int j = 0; j < q; j++) {
        double step = 0.0;
        for (int c = 0; c < r_tot; c++) step += Hinv[(p + j) + (size_t)r_tot * c] * dft[c];
        v_out[j] = v_in[j] + step;
    }

    double *iB_diag = (double *)R_alloc(q, sizeof(double));
    double *c_vh = (double *)R_alloc(q, sizeof(double));
    double *dv = (double *)R_alloc(q, sizeof(double));
    double *z_dv = (double *)R_alloc(n, sizeof(double));
    double *dexpeta = (double *)R_alloc(n, sizeof(double));
    double *mt_dexp = (double *)R_alloc(m, sizeof(double));
    double *dcla0 = (double *)R_alloc(m, sizeof(double));
    double *dBi_vec = (double *)R_alloc(n, sizeof(double));
    double *dAs_vec = (double *)R_alloc(m, sizeof(double));
    double *Hd = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));

    for (int i = 0; i < nrand; i++) {
        for (int j = 0; j < q; j++) {
            iB_diag[j] = (rand_idx[j] == i) ? (u_diag[j] * u_diag[j]) : 0.0;
            c_vh[j] = iB_diag[j] * v_out[j];
        }
        F77_CALL(dgemv)("N", &q, &q, &one, H22, &q, c_vh, & (int){1}, &zero, dv, & (int){1} FCONE);
        fhl_c_z_mv(xz, dv, z_dv);
        for (int r = 0; r < n; r++) dexpeta[r] = expeta[r] * z_dv[r];

        fhl_c_risk_backward(risk, dexpeta, mt_dexp, NULL);
        for (int k = 0; k < m; k++) {
            dcla0[k] = -(di[k] / (denom[k] * denom[k])) * mt_dexp[k];
            dAs_vec[k] = 2.0 * cla0[k] * dcla0[k] / di[k];
        }
        fhl_c_risk_forward(risk, dcla0, dBi_vec, NULL);

        fhl_c_xz_gram1(xz, 1, risk, expeta, clam0, as_vec, dexpeta, dBi_vec, dAs_vec, Hd);
        for (int j = 0; j < q; j++) {
            Hd[(p + j) + (size_t)r_tot * (p + j)] -= iB_diag[j];
        }

        double tr_Hinv_Hd = 0.0;
        for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
            tr_Hinv_Hd += Hinv[idx] * Hd[idx];
        }
        double sgamma = -alphas_in[i] * tr_Hinv_Hd;
        double denom_a = (double)qs[i] - sgamma;
        if (!R_FINITE(denom_a) || denom_a <= 0.0) {
            return fhl_vs_set_err(err_buf, err_cap, "Non-positive REML degrees-of-freedom denominator (q - sgamma <= 0) in lognorm1");
        }
        double ss_vi = 0.0;
        for (int j = 0; j < q; j++) {
            if (rand_idx[j] == i) ss_vi += v_out[j] * v_out[j];
        }
        double alpha_upd = ss_vi / denom_a;
        if (!R_FINITE(alpha_upd) || alpha_upd <= 0.0) {
            return fhl_vs_set_err(err_buf, err_cap, "Non-positive or non-finite dispersion alpha update in lognorm1");
        }
        alphas_out[i] = alpha_upd;
    }
    return 1;
}

/* Single penalized h-likelihood iteration for correlated 2-RE Kronecker covariance */
static int fhl_c_lognorm2_core(int n, int p, int q, int nrand, int n_alpha,
                               const int *qs, const int *rand_idx,
                               const fhl_c_xz_t *xz,
                               const fhl_c_risk_t *risk, const double *di, const double *del,
                               int pen_type, double tun1, double tun2,
                               const double *B_in, const double *v_in, const double *alphas_in,
                               double *B_out, double *v_out, double *alphas_out,
                               double *G_mat, double *wl_diag, double *clam0, double *hft,
                               char *err_buf, size_t err_cap) {
    (void)rand_idx;
    int r_tot = p + q;
    int m = risk->m;
    int q1 = qs[0];
    for (int i = 1; i < nrand; i++) {
        if (qs[i] != q1) {
            return fhl_vs_set_err(err_buf, err_cap, "lognorm2 requires equal block sizes across random effects");
        }
    }
    double one = 1.0, zero = 0.0;
    const double *X = xz->X;
    int nr2 = nrand * nrand;

    /* Kronecker covariance inverse iSig = iS0 %x% I_{q1}, O(nrand^3) */
    double *iS0 = (double *)R_alloc((size_t)nr2, sizeof(double));
    if (!fhl_vs_kron_s0_inv_logdet(nrand, n_alpha, alphas_in, iS0, NULL)) {
        return fhl_vs_set_err(err_buf, err_cap, "S0 not positive definite in lognorm2");
    }

    double *eta = (double *)R_alloc(n, sizeof(double));
    double *zv = (double *)R_alloc(n, sizeof(double));
    double *expeta = (double *)R_alloc(n, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, X, &n, B_in, & (int){1}, &zero, eta, & (int){1} FCONE);
    } else {
        memset(eta, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v_in, zv);
    for (int r = 0; r < n; r++) {
        eta[r] += zv[r];
        expeta[r] = exp(eta[r]);
    }

    double *denom = (double *)R_alloc(m, sizeof(double));
    double *cla0 = (double *)R_alloc(m, sizeof(double));
    double *as_vec = (double *)R_alloc(m, sizeof(double));
    fhl_c_risk_backward(risk, expeta, denom, NULL);
    for (int k = 0; k < m; k++) {
        cla0[k] = di[k] / denom[k];
        as_vec[k] = (cla0[k] * cla0[k]) / di[k];
    }
    fhl_c_risk_forward(risk, cla0, clam0, NULL);

    if (!fhl_vs_penalty_weights(pen_type, n, p, B_in, tun1, tun2, hft, wl_diag, err_buf, err_cap)) {
        return 0;
    }

    double *res_vec = (double *)R_alloc(n, sizeof(double));
    for (int r = 0; r < n; r++) res_vec[r] = del[r] - expeta[r] * clam0[r];

    double *dft = (double *)R_alloc(r_tot, sizeof(double));
    double *iSig_v = (double *)R_alloc(q, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("T", &n, &p, &one, X, &n, res_vec, & (int){1}, &zero, dft, & (int){1} FCONE);
        for (int j = 0; j < p; j++) dft[j] -= (double)n * wl_diag[j] * B_in[j];
    }
    fhl_c_z_tmv(xz, res_vec, dft + p);
    fhl_vs_kron_mv(nrand, q1, iS0, v_in, iSig_v);
    for (int j = 0; j < q; j++) dft[p + j] -= iSig_v[j];

    fhl_c_xz_gram0(xz, 1, risk, expeta, clam0, as_vec, G_mat);

    double *Hinv = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *H22 = (double *)R_alloc((size_t)q * q, sizeof(double));
    if (!fhl_vs_schur_invert_both(n, p, q, r_tot, G_mat, wl_diag, NULL, iS0, nrand, q1,
                                  H22, Hinv, "lognorm2", err_buf, err_cap)) {
        return 0;
    }

    for (int j = 0; j < p; j++) {
        double step = 0.0;
        for (int c = 0; c < r_tot; c++) step += Hinv[j + (size_t)r_tot * c] * dft[c];
        B_out[j] = B_in[j] + step;
    }
    for (int j = 0; j < q; j++) {
        double step = 0.0;
        for (int c = 0; c < r_tot; c++) step += Hinv[(p + j) + (size_t)r_tot * c] * dft[c];
        v_out[j] = v_in[j] + step;
    }

    /* Small nrand x nrand Kronecker factors: A0[i] = iS0 * dS0[i], iB0[i] = A0[i] * iS0 */
    double *A0_all = (double *)R_alloc((size_t)n_alpha * nr2, sizeof(double));
    double *iB0_all = (double *)R_alloc((size_t)n_alpha * nr2, sizeof(double));
    double *dS0 = (double *)R_alloc((size_t)nr2, sizeof(double));
    for (int i = 0; i < n_alpha; i++) {
        double *A0 = A0_all + (size_t)i * nr2;
        double *iB0 = iB0_all + (size_t)i * nr2;
        memset(dS0, 0, (size_t)nr2 * sizeof(double));
        if (i < nrand) {
            dS0[i + nrand * i] = 1.0;
        } else {
            for (int c = 0; c < nrand; c++) {
                for (int r = 0; r < nrand; r++) {
                    if (r != c) dS0[r + nrand * c] = 1.0;
                }
            }
        }
        F77_CALL(dgemm)("N", "N", &nrand, &nrand, &nrand, &one, iS0, &nrand, dS0, &nrand, &zero, A0, &nrand FCONE FCONE);
        F77_CALL(dgemm)("N", "N", &nrand, &nrand, &nrand, &one, A0, &nrand, iS0, &nrand, &zero, iB0, &nrand FCONE FCONE);
    }

    /* Fiber-level contraction for tr(B_j B_i) where B_i = H222 * (iB0[i] %x% I_{q1}).
     * Contracting hk %*% iB0 BEFORE squaring avoids O(1e7) precision cancellation when S0 is near-singular,
     * uses O(1) extra memory (no q x q matrices), and unit-stride column walks in O(q^2) flops. */
    const double *H222 = Hinv + p + (size_t)r_tot * p;
    int na2 = (n_alpha * (n_alpha + 1)) / 2;
    double *tr_BB_tri = (double *)R_alloc((size_t)na2, sizeof(double));
    memset(tr_BB_tri, 0, (size_t)na2 * sizeof(double));

    double *hk = (double *)R_alloc((size_t)nr2, sizeof(double));
    double *B1_all = (double *)R_alloc((size_t)n_alpha * nr2, sizeof(double));
    double *B2_all = (double *)R_alloc((size_t)n_alpha * nr2, sizeof(double));
    const double **hcols = (const double **)R_alloc((size_t)nrand, sizeof(const double *));

    for (int l = 0; l < q1; l++) {
        for (int b = 0; b < nrand; b++) {
            hcols[b] = H222 + (size_t)r_tot * ((size_t)b * q1 + l);
        }
        for (int k = 0; k < q1; k++) {
            for (int b = 0; b < nrand; b++) {
                const double *cb = hcols[b] + k;
                for (int a = 0; a < nrand; a++) {
                    hk[a + nrand * b] = cb[(size_t)a * q1];
                }
            }
            for (int m_idx = 0; m_idx < n_alpha; m_idx++) {
                const double *iB0m = iB0_all + (size_t)m_idx * nr2;
                double *B1m = B1_all + (size_t)m_idx * nr2;
                double *B2m = B2_all + (size_t)m_idx * nr2;
                for (int c = 0; c < nrand; c++) {
                    for (int a = 0; a < nrand; a++) {
                        double s1 = 0.0, s2 = 0.0;
                        for (int b = 0; b < nrand; b++) {
                            s1 += hk[a + nrand * b] * iB0m[b + nrand * c];
                            s2 += hk[b + nrand * c] * iB0m[b + nrand * a];
                        }
                        B1m[a + nrand * c] = s1;
                        B2m[c + nrand * a] = s2;
                    }
                }
            }
            int pair_idx = 0;
            for (int i = 0; i < n_alpha; i++) {
                const double *B2i = B2_all + (size_t)i * nr2;
                for (int j = 0; j <= i; j++) {
                    const double *B1j = B1_all + (size_t)j * nr2;
                    double dot_val = 0.0;
                    for (int c = 0; c < nrand; c++) {
                        for (int a = 0; a < nrand; a++) {
                            dot_val += B1j[a + nrand * c] * B2i[c + nrand * a];
                        }
                    }
                    tr_BB_tri[pair_idx++] += dot_val;
                }
            }
        }
    }

    double *dk = (double *)R_alloc(n_alpha, sizeof(double));
    double *ddk = (double *)R_alloc((size_t)n_alpha * n_alpha, sizeof(double));
    double *c_vh = (double *)R_alloc(q, sizeof(double));
    double *dv = (double *)R_alloc(q, sizeof(double));
    double *z_dv = (double *)R_alloc(n, sizeof(double));
    double *dexpeta = (double *)R_alloc(n, sizeof(double));
    double *mt_dexp = (double *)R_alloc(m, sizeof(double));
    double *dcla0 = (double *)R_alloc(m, sizeof(double));
    double *dBi_vec = (double *)R_alloc(n, sizeof(double));
    double *dAs_vec = (double *)R_alloc(m, sizeof(double));
    double *Hd = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *M0 = (double *)R_alloc((size_t)nr2, sizeof(double));

    for (int i = 0; i < n_alpha; i++) {
        const double *A0i = A0_all + (size_t)i * nr2;
        const double *iB0i = iB0_all + (size_t)i * nr2;

        fhl_vs_kron_mv(nrand, q1, iB0i, v_out, c_vh);
        F77_CALL(dgemv)("N", &q, &q, &one, H22, &q, c_vh, & (int){1}, &zero, dv, & (int){1} FCONE);
        fhl_c_z_mv(xz, dv, z_dv);
        for (int r = 0; r < n; r++) dexpeta[r] = expeta[r] * z_dv[r];

        fhl_c_risk_backward(risk, dexpeta, mt_dexp, NULL);
        for (int k = 0; k < m; k++) {
            dcla0[k] = -(di[k] / (denom[k] * denom[k])) * mt_dexp[k];
            dAs_vec[k] = 2.0 * cla0[k] * dcla0[k] / di[k];
        }
        fhl_c_risk_forward(risk, dcla0, dBi_vec, NULL);

        fhl_c_xz_gram1(xz, 1, risk, expeta, clam0, as_vec, dexpeta, dBi_vec, dAs_vec, Hd);
        for (int c_blk = 0; c_blk < nrand; c_blk++) {
            for (int r_blk = 0; r_blk < nrand; r_blk++) {
                double val = iB0i[r_blk + nrand * c_blk];
                for (int k = 0; k < q1; k++) {
                    int r = r_blk * q1 + k;
                    int c = c_blk * q1 + k;
                    Hd[(p + r) + (size_t)r_tot * (p + c)] -= val;
                }
            }
        }

        double tr_A0i = 0.0;
        for (int r = 0; r < nrand; r++) tr_A0i += A0i[r + nrand * r];
        double tr1 = (double)q1 * tr_A0i;
        double tr2 = 0.0;
        for (int j = 0; j < q; j++) tr2 += v_out[j] * c_vh[j];
        double tr3 = 0.0;
        for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
            tr3 += Hinv[idx] * Hd[idx];
        }
        dk[i] = -0.5 * tr1 + 0.5 * tr2 - 0.5 * tr3;

        int row_base = (i * (i + 1)) / 2;
        for (int j = 0; j <= i; j++) {
            const double *A0j = A0_all + (size_t)j * nr2;
            const double *iB0j = iB0_all + (size_t)j * nr2;
            F77_CALL(dgemm)("N", "N", &nrand, &nrand, &nrand, &one, A0i, &nrand, iB0j, &nrand, &zero, M0, &nrand FCONE FCONE);

            double tr_AiAj = 0.0;
            for (int b = 0; b < nrand; b++) {
                for (int a = 0; a < nrand; a++) {
                    tr_AiAj += A0i[a + nrand * b] * A0j[b + nrand * a];
                }
            }
            tr_AiAj *= (double)q1;

            /* Contract per cluster k before summing across k to preserve cancellation precision */
            double tr_H222_ij = 0.0, v_quad = 0.0;
            for (int k = 0; k < q1; k++) {
                double hk_m = 0.0, vk_m = 0.0;
                for (int b = 0; b < nrand; b++) {
                    double vb = v_out[(size_t)b * q1 + k];
                    const double *hcol_b = H222 + (size_t)r_tot * ((size_t)b * q1 + k) + k;
                    for (int a = 0; a < nrand; a++) {
                        double va = v_out[(size_t)a * q1 + k];
                        hk_m += hcol_b[(size_t)a * q1] * M0[b + nrand * a];
                        vk_m += va * ((j == i) ? M0[a + nrand * b] : 0.5 * (M0[a + nrand * b] + M0[b + nrand * a])) * vb;
                    }
                }
                tr_H222_ij += hk_m;
                v_quad += vk_m;
            }

            double tr_BjBi = tr_BB_tri[row_base + j];
            double val_ij = -0.5 * tr_AiAj + v_quad - 0.5 * tr_BjBi + tr_H222_ij;
            ddk[i + (size_t)n_alpha * j] = val_ij;
            ddk[j + (size_t)n_alpha * i] = val_ij;
        }
    }

    double *ddk_inv = (double *)R_alloc((size_t)n_alpha * n_alpha, sizeof(double));
    if (!fhl_vs_all_finite(ddk, (size_t)n_alpha * n_alpha) ||
        !fhl_c_solve_gesv_inv(ddk, ddk_inv, n_alpha)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix in lognorm2 (ddk)");
    }
    for (int i = 0; i < n_alpha; i++) {
        double step = 0.0;
        for (int j = 0; j < n_alpha; j++) step += ddk_inv[i + (size_t)n_alpha * j] * dk[j];
        alphas_out[i] = alphas_in[i] + step;
    }
    double *iS0_check = (double *)R_alloc((size_t)nr2, sizeof(double));
    if (!fhl_vs_kron_s0_inv_logdet(nrand, n_alpha, alphas_out, iS0_check, NULL)) {
        return fhl_vs_set_err(err_buf, err_cap, "S0 not positive definite in lognorm2");
    }
    return 1;
}

/* Post-fit BIC/GCV criterion and sandwich standard-error evaluation on sparse [X, Z]. */
static int fhl_c_lognorm_post_core(int n, int p, int q, int nrand, int n_alpha,
                                   const int *qs, const int *rand_idx,
                                   const fhl_c_xz_t *xz, const fhl_c_risk_t *risk,
                                   const double *di, const double *del,
                                   const double *beta, const double *v, const double *alphas,
                                   const double *mat, const double *wl_diag, const double *hft,
                                   double err, int varfixed, int corr,
                                   double *bic_out, double *se_out, int *valid_out,
                                   char *err_buf, size_t err_cap) {
    if (valid_out) *valid_out = 0;
    int r_tot = p + q;
    double *mat11 = (double *)R_alloc((size_t)p * p, sizeof(double));
    double *mat12 = (double *)R_alloc((size_t)p * q, sizeof(double));
    double *H22 = (double *)R_alloc((size_t)q * q, sizeof(double));
    for (int c = 0; c < p; c++) {
        memcpy(mat11 + (size_t)p * c, mat + (size_t)r_tot * c, (size_t)p * sizeof(double));
    }
    for (int c = 0; c < q; c++) {
        memcpy(mat12 + (size_t)p * c, mat + (size_t)r_tot * (p + c), (size_t)p * sizeof(double));
        memcpy(H22 + (size_t)q * c, mat + (size_t)r_tot * (p + c) + p, (size_t)q * sizeof(double));
    }

    double logdet_Sig = 0.0;
    double v_iSig_v = 0.0;

    if (!corr) {
        for (int j = 0; j < q; j++) {
            double aj = alphas[rand_idx[j]];
            double ujj = 1.0 / aj;
            H22[j + (size_t)q * j] += ujj;
            logdet_Sig += log(aj);
            v_iSig_v += v[j] * v[j] * ujj;
        }
    } else {
        int q1 = qs[0];
        for (int i = 1; i < nrand; i++) {
            if (qs[i] != q1 || q1 * nrand != q) {
                return fhl_vs_set_err(err_buf, err_cap, "lognorm2 requires equal block sizes across random effects");
            }
        }
        int nr2 = nrand * nrand;
        double *iS0 = (double *)R_alloc((size_t)nr2, sizeof(double));
        double logdet_S0 = 0.0;
        if (!fhl_vs_kron_s0_inv_logdet(nrand, n_alpha, alphas, iS0, &logdet_S0)) {
            *bic_out = 100000.0;
            memset(se_out, 0, (size_t)p * sizeof(double));
            return 1;
        }
        logdet_Sig = (double)q1 * logdet_S0;
        for (int c_blk = 0; c_blk < nrand; c_blk++) {
            for (int r_blk = 0; r_blk < nrand; r_blk++) {
                double val = iS0[r_blk + nrand * c_blk];
                for (int k = 0; k < q1; k++) {
                    int r = r_blk * q1 + k;
                    int c = c_blk * q1 + k;
                    H22[r + (size_t)q * c] += val;
                }
            }
        }
        double *iSig_v = (double *)R_alloc(q, sizeof(double));
        fhl_vs_kron_mv(nrand, q1, iS0, v, iSig_v);
        for (int j = 0; j < q; j++) v_iSig_v += v[j] * iSig_v[j];
    }

    if (!fhl_vs_all_finite(H22, (size_t)q * q)) {
        *bic_out = 100000.0;
        memset(se_out, 0, (size_t)p * sizeof(double));
        if (valid_out) *valid_out = 0;
        return 1;
    }
    double *H22_chol = (double *)R_alloc((size_t)q * q, sizeof(double));
    memcpy(H22_chol, H22, (size_t)q * q * sizeof(double));
    int info = 0;
    F77_CALL(dpotrf)("U", &q, H22_chol, &q, &info FCONE);
    double logdet_H22 = 0.0;
    int det_lt_thresh = 0;
    int chol_failed = (info != 0);

    if (!chol_failed) {
        for (int j = 0; j < q; j++) logdet_H22 += 2.0 * log(H22_chol[j + (size_t)q * j]);
        if (!R_FINITE(logdet_H22) || logdet_H22 < log(0.0000001)) det_lt_thresh = 1;
    }

    if (chol_failed || !R_FINITE(err) || err >= 5.0 || det_lt_thresh || !fhl_vs_all_finite(beta, (size_t)p)) {
        *bic_out = 100000.0;
        memset(se_out, 0, (size_t)p * sizeof(double));
        if (valid_out) *valid_out = 0;
        return 1;
    }

    double one = 1.0, zero = 0.0;
    double *S = (double *)R_alloc((size_t)q * p, sizeof(double));
    for (int r = 0; r < p; r++) {
        for (int c = 0; c < q; c++) S[c + (size_t)q * r] = mat12[r + (size_t)p * c];
    }
    F77_CALL(dpotrs)("U", &q, &p, H22_chol, &q, S, &q, &info FCONE);
    if (info != 0) {
        return fhl_vs_set_err(err_buf, err_cap, "dpotrs failed in fhl_c_lognorm_post");
    }

    double *prod_pp = (double *)R_alloc((size_t)p * p, sizeof(double));
    double *mmat = (double *)R_alloc((size_t)p * p, sizeof(double));
    F77_CALL(dgemm)("N", "N", &p, &p, &q, &one, mat12, &p, S, &q, &zero, prod_pp, &p FCONE FCONE);
    for (size_t idx = 0; idx < (size_t)p * p; idx++) mmat[idx] = mat11[idx] - prod_pp[idx];

    double *pos_mmat_WL = (double *)R_alloc((size_t)p * p, sizeof(double));
    memcpy(pos_mmat_WL, mmat, (size_t)p * p * sizeof(double));
    for (int j = 0; j < p; j++) pos_mmat_WL[j + (size_t)p * j] += (double)n * wl_diag[j];

    double *neg_minv = (double *)R_alloc((size_t)p * p, sizeof(double));
    if (!fhl_c_solve_inv(pos_mmat_WL, neg_minv, p)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix (mmat + n*WL) in fhl_c_lognorm_post");
    }
    double *tmp_pp = (double *)R_alloc((size_t)p * p, sizeof(double));
    double *iHs = (double *)R_alloc((size_t)p * p, sizeof(double));
    F77_CALL(dgemm)("N", "N", &p, &p, &p, &one, neg_minv, &p, mmat, &p, &zero, tmp_pp, &p FCONE FCONE);
    F77_CALL(dgemm)("N", "N", &p, &p, &p, &one, tmp_pp, &p, neg_minv, &p, &zero, iHs, &p FCONE FCONE);
    int se_finite = 1;
    for (int j = 0; j < p; j++) {
        double v_ii = iHs[j + (size_t)p * j];
        se_out[j] = (R_FINITE(v_ii) && v_ii >= 0.0) ? sqrt(v_ii) : NA_REAL;
        if (!R_FINITE(se_out[j])) se_finite = 0;
    }

    double *eta = (double *)R_alloc(n, sizeof(double));
    double *zv = (double *)R_alloc(n, sizeof(double));
    double *expeta = (double *)R_alloc(n, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, xz->X, &n, beta, & (int){1}, &zero, eta, & (int){1} FCONE);
    } else {
        memset(eta, 0, (size_t)n * sizeof(double));
    }
    fhl_c_z_mv(xz, v, zv);
    for (int r = 0; r < n; r++) {
        eta[r] += zv[r];
        expeta[r] = exp(eta[r]);
    }

    int m = risk->m;
    double *term0 = (double *)R_alloc(m, sizeof(double));
    fhl_c_risk_backward(risk, expeta, term0, NULL);

    double sum_del_eta = 0.0, sum_di_log = 0.0;
    for (int r = 0; r < n; r++) sum_del_eta += del[r] * eta[r];
    for (int k = 0; k < m; k++) sum_di_log += di[k] * log(term0[k]);
    double hlike1 = sum_del_eta - sum_di_log;

    double pi_val = 3.14159265359;
    double hlike2 = -0.5 * (double)q * log(2.0 * pi_val) - 0.5 * logdet_Sig;
    double hlike3 = -0.5 * v_iSig_v;
    double hlike = hlike1 + hlike2 + hlike3;

    double adj2 = -0.5 * (logdet_H22 - (double)q * log(2.0 * pi_val));

    double pvh = (!corr && varfixed) ? hlike1 : (hlike + adj2);

    double *A_elam = (double *)R_alloc((size_t)p * p, sizeof(double));
    double *iH = (double *)R_alloc((size_t)p * p, sizeof(double));
    memcpy(A_elam, mmat, (size_t)p * p * sizeof(double));
    for (int j = 0; j < p; j++) {
        double babs = fabs(beta[j]);
        if (babs < 1e-12) babs = 1e-8;
        A_elam[j + (size_t)p * j] += (double)n * (hft[j] / babs);
    }
    if (!fhl_c_solve_spd(A_elam, mmat, iH, p, p)) {
        return fhl_vs_set_err(err_buf, err_cap, "Singular matrix (mmat + n*WL) in fhl_c_lognorm_post");
    }
    double elam = 0.0;
    for (int j = 0; j < p; j++) elam += iH[j + (size_t)p * j];

    *bic_out = -2.0 * pvh + log((double)n) * elam;
    if (valid_out) *valid_out = (R_FINITE(*bic_out) && se_finite) ? 1 : 0;
    return 1;
}

/* Assemble 8-element named return list for the selected tuning parameter */
static SEXP fhl_vs_build_result_list(int p, int q, int n_alpha_out, int alpha_is_matrix,
                                     int pen_type, double sel_tun1, double sel_tun2,
                                     const double *best_B, const double *best_se,
                                     const double *best_v, const double *best_alpha,
                                     double best_bic, int best_rep, int best_conv) {
    SEXP beta_out_s = PROTECT(Rf_allocVector(REALSXP, p));
    SEXP se_out_s = PROTECT(Rf_allocVector(REALSXP, p));
    for (int j = 0; j < p; j++) {
        REAL(beta_out_s)[j] = Rf_fround(best_B[j], 5.0);
        REAL(se_out_s)[j] = Rf_fround(best_se[j], 5.0);
    }

    SEXP v_out_s = PROTECT(Rf_allocVector(REALSXP, q));
    for (int j = 0; j < q; j++) REAL(v_out_s)[j] = Rf_fround(best_v[j], 5.0);

    SEXP alpha_out_s = alpha_is_matrix ? PROTECT(Rf_allocMatrix(REALSXP, n_alpha_out, 1))
                                       : PROTECT(Rf_allocVector(REALSXP, n_alpha_out));
    for (int i = 0; i < n_alpha_out; i++) REAL(alpha_out_s)[i] = Rf_fround(best_alpha[i], 5.0);

    SEXP tun_out_s;
    if (pen_type == 2) {
        tun_out_s = PROTECT(Rf_allocVector(REALSXP, 2));
        REAL(tun_out_s)[0] = sel_tun1;
        REAL(tun_out_s)[1] = sel_tun2;
    } else {
        tun_out_s = PROTECT(Rf_ScalarReal(sel_tun1));
    }

    SEXP bic_out_s = PROTECT(Rf_ScalarReal(Rf_fround(best_bic, 5.0)));
    SEXP rep_out_s = PROTECT(Rf_ScalarInteger(best_rep));
    SEXP conv_out_s = PROTECT(Rf_ScalarInteger(best_conv));

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 8));
    SET_VECTOR_ELT(out, 0, beta_out_s);
    SET_VECTOR_ELT(out, 1, se_out_s);
    SET_VECTOR_ELT(out, 2, v_out_s);
    SET_VECTOR_ELT(out, 3, alpha_out_s);
    SET_VECTOR_ELT(out, 4, tun_out_s);
    SET_VECTOR_ELT(out, 5, bic_out_s);
    SET_VECTOR_ELT(out, 6, rep_out_s);
    SET_VECTOR_ELT(out, 7, conv_out_s);

    SEXP nms = PROTECT(Rf_allocVector(STRSXP, 8));
    SET_STRING_ELT(nms, 0, Rf_mkChar("beta"));
    SET_STRING_ELT(nms, 1, Rf_mkChar("se"));
    SET_STRING_ELT(nms, 2, Rf_mkChar("v"));
    SET_STRING_ELT(nms, 3, Rf_mkChar("alpha"));
    SET_STRING_ELT(nms, 4, Rf_mkChar("tuning_parameter"));
    SET_STRING_ELT(nms, 5, Rf_mkChar("BIC"));
    SET_STRING_ELT(nms, 6, Rf_mkChar("repetition"));
    SET_STRING_ELT(nms, 7, Rf_mkChar("convergence"));
    Rf_setAttrib(out, R_NamesSymbol, nms);

    UNPROTECT(10);
    return out;
}

static char *fhl_vs_format_point_failure(int pen_type, double t1, double t2, const char *reason) {
    char tmp[384];
    if (pen_type == 2) {
        snprintf(tmp, sizeof(tmp), "(tun1=%.6g, tun2=%.6g) -> %s", t1, t2, reason);
    } else {
        snprintf(tmp, sizeof(tmp), "(tun1=%.6g) -> %s", t1, reason);
    }
    tmp[sizeof(tmp) - 1] = '\0';
    size_t len = strlen(tmp) + 1;
    char *dst = (char *)R_alloc(len, sizeof(char));
    memcpy(dst, tmp, len);
    return dst;
}

static void fhl_vs_report_grid_failures(int n_grid, int n_fail, char **fail_msgs) {
    if (n_fail <= 0) return;
    size_t cap = 4096;
    char *joined = (char *)R_alloc(cap, sizeof(char));
    size_t pos = 0;
    joined[0] = '\0';
    for (int i = 0; i < n_fail; i++) {
        const char *sep = (i > 0) ? "; " : "";
        size_t slen = strlen(sep);
        size_t mlen = strlen(fail_msgs[i]);
        if (pos + slen + mlen + 16 >= cap) {
            int rem = n_fail - i;
            snprintf(joined + pos, cap - pos, "; ... (%d more)", rem);
            joined[cap - 1] = '\0';
            break;
        }
        memcpy(joined + pos, sep, slen);
        pos += slen;
        memcpy(joined + pos, fail_msgs[i], mlen);
        pos += mlen;
        joined[pos] = '\0';
    }
    if (n_fail == n_grid) {
        Rf_error("frailty.vs failed at all %d tuning points: %s", n_grid, joined);
    }
    Rf_warning("frailty.vs failed at %d/%d tuning point(s) (BIC = Inf): %s", n_fail, n_grid, joined);
}

/* Complete C grid search for variable selection (1 independent RE block, BIC) */
SEXP fhl_fit_vs_1re(SEXP X_s, SEXP B_s, SEXP Z_list_s, SEXP v_list_s,
                    SEXP pen_type_s, SEXP alpha_s, SEXP tunset1_s, SEXP tunset2_s,
                    SEXP di_s, SEXP Mi_s, SEXP del_s, SEXP varfixed_s) {
    if (!Rf_isMatrix(X_s) || TYPEOF(X_s) != REALSXP) {
        Rf_error("X must be a numeric matrix in fhl_fit_vs_1re");
    }
    if (TYPEOF(B_s) != REALSXP || TYPEOF(alpha_s) != REALSXP ||
        TYPEOF(tunset1_s) != REALSXP || TYPEOF(di_s) != REALSXP || TYPEOF(del_s) != REALSXP ||
        TYPEOF(Z_list_s) != VECSXP || TYPEOF(v_list_s) != VECSXP) {
        Rf_error("Invalid SEXP argument types in fhl_fit_vs_1re");
    }
    int n = Rf_nrows(X_s), p = Rf_ncols(X_s);
    int nrand = LENGTH(Z_list_s);
    int pen_type = Rf_asInteger(pen_type_s);
    int varfixed = Rf_asLogical(varfixed_s);
    int n_grid = LENGTH(tunset1_s);
    if (LENGTH(B_s) != p || LENGTH(v_list_s) != nrand || LENGTH(alpha_s) != nrand || n_grid < 1) {
        Rf_error("non-conformable arguments in fhl_fit_vs_1re");
    }
    if (pen_type == 2 && (TYPEOF(tunset2_s) != REALSXP || LENGTH(tunset2_s) != n_grid)) {
        Rf_error("non-conformable tunset2 in fhl_fit_vs_1re");
    }

    int *qs = (int *)R_alloc(nrand, sizeof(int));
    int *qcum = (int *)R_alloc(nrand + 1, sizeof(int));
    fhl_c_xz_t xz;
    fhl_c_xz_init(&xz, X_s, Z_list_s, n, qs, qcum);
    int q = xz.Q;
    int r_tot = p + q;

    double *v_init = (double *)R_alloc(q, sizeof(double));
    double *v_curr = (double *)R_alloc(q, sizeof(double));
    double *v_next = (double *)R_alloc(q, sizeof(double));
    double *best_v = (double *)R_alloc(q, sizeof(double));
    int *rand_idx = (int *)R_alloc(q, sizeof(int));
    int q_off = 0;
    for (int i = 0; i < nrand; i++) {
        int qi = qs[i];
        SEXP v_i = VECTOR_ELT(v_list_s, i);
        if (TYPEOF(v_i) != REALSXP || LENGTH(v_i) != qi) {
            Rf_error("non-conformable arguments in fhl_fit_vs_1re (v)");
        }
        memcpy(v_init + q_off, REAL(v_i), (size_t)qi * sizeof(double));
        for (int j = 0; j < qi; j++) rand_idx[q_off + j] = i;
        q_off += qi;
    }

    fhl_c_risk_t risk;
    fhl_c_unpack_risk(Mi_s, &risk);
    const double *di = REAL(di_s);
    const double *del = REAL(del_s);
    const double *tunset1 = REAL(tunset1_s);
    const double *tunset2 = (pen_type == 2) ? REAL(tunset2_s) : NULL;

    double *B_init = (double *)R_alloc(p, sizeof(double));
    double *B_curr = (double *)R_alloc(p, sizeof(double));
    double *B_next = (double *)R_alloc(p, sizeof(double));
    double *best_B = (double *)R_alloc(p, sizeof(double));
    double *best_se = (double *)R_alloc(p, sizeof(double));
    double *se_curr = (double *)R_alloc(p, sizeof(double));
    memcpy(B_init, REAL(B_s), (size_t)p * sizeof(double));

    double *alpha_init = (double *)R_alloc(nrand, sizeof(double));
    double *alpha_curr = (double *)R_alloc(nrand, sizeof(double));
    double *alpha_next = (double *)R_alloc(nrand, sizeof(double));
    double *best_alpha = (double *)R_alloc(nrand, sizeof(double));
    memcpy(alpha_init, REAL(alpha_s), (size_t)nrand * sizeof(double));

    double *G_mat = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *wl_diag = (double *)R_alloc(p, sizeof(double));
    double *clam0 = (double *)R_alloc(n, sizeof(double));
    double *hft = (double *)R_alloc(p, sizeof(double));
    char **fail_msgs = (char **)R_alloc((size_t)n_grid, sizeof(char *));
    int n_fail = 0, first_ok_g = -1;

    int has_best = 0, best_g = 0, best_rep = 1, best_conv = 0, last_rep = 1, last_conv = 0;
    double best_bic = 0.0, best_bic_r = 0.0;

    for (int g = 0; g < n_grid; g++) {
        R_CheckUserInterrupt();
        memcpy(B_curr, B_init, (size_t)p * sizeof(double));
        memcpy(v_curr, v_init, (size_t)q * sizeof(double));
        memcpy(alpha_curr, alpha_init, (size_t)nrand * sizeof(double));

        double t1 = tunset1[g];
        double t2 = (tunset2 != NULL) ? tunset2[g] : 0.0;
        int rep = 1, convergeind = 0, pt_ok = 1;
        double div_err = 3.0;
        char err_buf[FHL_VS_ERR_CAP];
        err_buf[0] = '\0';

        while (rep < 2000 && div_err > 0.0000001 && div_err < 5.0) {
            if (rep % 50 == 0) R_CheckUserInterrupt();
            const void *vmax = vmaxget();
            pt_ok = fhl_c_lognorm1_core(n, p, q, nrand, qs, rand_idx,
                                        &xz, &risk, di, del,
                                        pen_type, t1, t2, varfixed, 1,
                                        B_curr, v_curr, alpha_curr,
                                        B_next, v_next, alpha_next,
                                        G_mat, wl_diag, clam0, hft,
                                        err_buf, sizeof(err_buf));
            vmaxset(vmax);
            if (!pt_ok) break;

            if (varfixed) {
                div_err = 0.0;
                for (int j = 0; j < p; j++) {
                    double d = fabs(B_curr[j] - B_next[j]);
                    if (d > div_err) div_err = d;
                }
            } else {
                div_err = 0.0;
                for (int i = 0; i < nrand; i++) div_err += fabs(alpha_curr[i] - alpha_next[i]);
            }
            convergeind = (div_err <= 0.0000001) ? 1 : -1;

            memcpy(B_curr, B_next, (size_t)p * sizeof(double));
            memcpy(v_curr, v_next, (size_t)q * sizeof(double));
            memcpy(alpha_curr, alpha_next, (size_t)nrand * sizeof(double));
            rep++;
        }
        if (pt_ok && (!R_FINITE(div_err) || div_err >= 5.0)) {
            pt_ok = fhl_vs_set_err(err_buf, sizeof(err_buf), "frailty.vs inner iteration diverged (|Delta alpha| >= 5)");
        }

        double bic_g = 0.0;
        int valid_g = 0;
        if (pt_ok) {
            const void *vmax = vmaxget();
            pt_ok = fhl_c_lognorm_post_core(n, p, q, nrand, nrand, qs, rand_idx,
                                            &xz, &risk, di, del,
                                            B_curr, v_curr, alpha_curr,
                                            G_mat, wl_diag, hft,
                                            div_err, varfixed, 0,
                                            &bic_g, se_curr, &valid_g,
                                            err_buf, sizeof(err_buf));
            vmaxset(vmax);
        }

        if (!pt_ok) {
            if (n_grid == 1) Rf_error("%s", err_buf);
            fail_msgs[n_fail++] = fhl_vs_format_point_failure(pen_type, t1, t2, err_buf);
            continue;
        }
        if (first_ok_g < 0) first_ok_g = g;

        last_rep = rep;
        last_conv = convergeind;
        double bic_r = Rf_fround(bic_g, 5.0);
        if (!ISNAN(bic_r) && ((valid_g && !has_best) || (valid_g && has_best == 1 && bic_r < best_bic_r) || (!valid_g && !has_best) || (valid_g && has_best == -1))) {
            has_best = valid_g ? 1 : -1;
            best_g = g;
            best_bic = bic_g;
            best_bic_r = bic_r;
            best_rep = rep;
            best_conv = valid_g ? convergeind : 0;
            memcpy(best_B, B_curr, (size_t)p * sizeof(double));
            memcpy(best_se, se_curr, (size_t)p * sizeof(double));
            memcpy(best_v, v_curr, (size_t)q * sizeof(double));
            memcpy(best_alpha, alpha_curr, (size_t)nrand * sizeof(double));
        }
    }

    fhl_vs_report_grid_failures(n_grid, n_fail, fail_msgs);

    if (!has_best) {
        best_g = (first_ok_g >= 0) ? first_ok_g : 0;
        best_bic = NA_REAL;
        best_rep = last_rep;
        best_conv = last_conv;
        memcpy(best_B, B_curr, (size_t)p * sizeof(double));
        memcpy(best_se, se_curr, (size_t)p * sizeof(double));
        memcpy(best_v, v_curr, (size_t)q * sizeof(double));
        memcpy(best_alpha, alpha_curr, (size_t)nrand * sizeof(double));
    }

    return fhl_vs_build_result_list(p, q, nrand, 0, pen_type,
                                     tunset1[best_g],
                                     (tunset2 != NULL) ? tunset2[best_g] : 0.0,
                                     best_B, best_se, best_v, best_alpha,
                                     best_bic, best_rep, best_conv);
}

/* Complete C grid search for variable selection (2 correlated REs, GCV/BIC) */
SEXP fhl_fit_vs_2re(SEXP X_s, SEXP B_s, SEXP Z_list_s, SEXP v_list_s,
                    SEXP pen_type_s, SEXP alpha_s, SEXP tunset1_s, SEXP tunset2_s,
                    SEXP di_s, SEXP Mi_s, SEXP del_s) {
    if (!Rf_isMatrix(X_s) || TYPEOF(X_s) != REALSXP) {
        Rf_error("X must be a numeric matrix in fhl_fit_vs_2re");
    }
    if (TYPEOF(B_s) != REALSXP || TYPEOF(alpha_s) != REALSXP ||
        TYPEOF(tunset1_s) != REALSXP || TYPEOF(di_s) != REALSXP || TYPEOF(del_s) != REALSXP ||
        TYPEOF(Z_list_s) != VECSXP || TYPEOF(v_list_s) != VECSXP) {
        Rf_error("Invalid SEXP argument types in fhl_fit_vs_2re");
    }
    int n = Rf_nrows(X_s), p = Rf_ncols(X_s);
    int nrand = LENGTH(Z_list_s);
    int n_alpha = LENGTH(alpha_s);
    int pen_type = Rf_asInteger(pen_type_s);
    int n_grid = LENGTH(tunset1_s);
    if (nrand != 2 || n_alpha != 3) {
        Rf_error("fhl_fit_vs_2re requires nrand == 2 and n_alpha == 3");
    }
    if (LENGTH(B_s) != p || LENGTH(v_list_s) != nrand || n_grid < 1) {
        Rf_error("non-conformable arguments in fhl_fit_vs_2re");
    }
    if (pen_type == 2 && (TYPEOF(tunset2_s) != REALSXP || LENGTH(tunset2_s) != n_grid)) {
        Rf_error("non-conformable tunset2 in fhl_fit_vs_2re");
    }

    int *qs = (int *)R_alloc(nrand, sizeof(int));
    int *qcum = (int *)R_alloc(nrand + 1, sizeof(int));
    fhl_c_xz_t xz;
    fhl_c_xz_init(&xz, X_s, Z_list_s, n, qs, qcum);
    if (qs[1] != qs[0]) {
        Rf_error("fhl_fit_vs_2re requires equal random-effect dimensions qs[0] == qs[1]");
    }
    int q = xz.Q;
    int r_tot = p + q;

    double *v_init = (double *)R_alloc(q, sizeof(double));
    double *v_curr = (double *)R_alloc(q, sizeof(double));
    double *v_next = (double *)R_alloc(q, sizeof(double));
    double *best_v = (double *)R_alloc(q, sizeof(double));
    int *rand_idx = (int *)R_alloc(q, sizeof(int));
    int q_off = 0;
    for (int i = 0; i < nrand; i++) {
        int qi = qs[i];
        SEXP v_i = VECTOR_ELT(v_list_s, i);
        if (TYPEOF(v_i) != REALSXP || LENGTH(v_i) != qi) {
            Rf_error("non-conformable arguments in fhl_fit_vs_2re (v)");
        }
        memcpy(v_init + q_off, REAL(v_i), (size_t)qi * sizeof(double));
        for (int j = 0; j < qi; j++) rand_idx[q_off + j] = i;
        q_off += qi;
    }

    fhl_c_risk_t risk;
    fhl_c_unpack_risk(Mi_s, &risk);
    const double *di = REAL(di_s);
    const double *del = REAL(del_s);
    const double *tunset1 = REAL(tunset1_s);
    const double *tunset2 = (pen_type == 2) ? REAL(tunset2_s) : NULL;

    double *B_init = (double *)R_alloc(p, sizeof(double));
    double *B_curr = (double *)R_alloc(p, sizeof(double));
    double *B_next = (double *)R_alloc(p, sizeof(double));
    double *best_B = (double *)R_alloc(p, sizeof(double));
    double *best_se = (double *)R_alloc(p, sizeof(double));
    double *se_curr = (double *)R_alloc(p, sizeof(double));
    memcpy(B_init, REAL(B_s), (size_t)p * sizeof(double));

    double *alpha_init = (double *)R_alloc(n_alpha, sizeof(double));
    double *alpha_curr = (double *)R_alloc(n_alpha, sizeof(double));
    double *alpha_next = (double *)R_alloc(n_alpha, sizeof(double));
    double *best_alpha = (double *)R_alloc(n_alpha, sizeof(double));
    memcpy(alpha_init, REAL(alpha_s), (size_t)n_alpha * sizeof(double));

    double *G_mat = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
    double *wl_diag = (double *)R_alloc(p, sizeof(double));
    double *clam0 = (double *)R_alloc(n, sizeof(double));
    double *hft = (double *)R_alloc(p, sizeof(double));
    char **fail_msgs = (char **)R_alloc((size_t)n_grid, sizeof(char *));
    int n_fail = 0, first_ok_g = -1;

    int has_best = 0, best_g = 0, best_rep = 1, best_conv = 0, last_rep = 1, last_conv = 0;
    double best_bic = 0.0, best_bic_r = 0.0;

    for (int g = 0; g < n_grid; g++) {
        R_CheckUserInterrupt();
        memcpy(B_curr, B_init, (size_t)p * sizeof(double));
        memcpy(v_curr, v_init, (size_t)q * sizeof(double));
        memcpy(alpha_curr, alpha_init, (size_t)n_alpha * sizeof(double));

        double t1 = tunset1[g];
        double t2 = (tunset2 != NULL) ? tunset2[g] : 0.0;
        int rep = 1, convergeind = 0, pt_ok = 1;
        double div_err = 3.0;
        char err_buf[FHL_VS_ERR_CAP];
        err_buf[0] = '\0';

        while (rep < 300 && div_err > 0.0000001 && div_err < 5.0) {
            if (rep % 50 == 0) R_CheckUserInterrupt();
            const void *vmax = vmaxget();
            pt_ok = fhl_c_lognorm2_core(n, p, q, nrand, n_alpha, qs, rand_idx,
                                        &xz, &risk, di, del,
                                        pen_type, t1, t2,
                                        B_curr, v_curr, alpha_curr,
                                        B_next, v_next, alpha_next,
                                        G_mat, wl_diag, clam0, hft,
                                        err_buf, sizeof(err_buf));
            vmaxset(vmax);
            if (!pt_ok) break;

            div_err = 0.0;
            for (int i = 0; i < n_alpha; i++) div_err += fabs(alpha_curr[i] - alpha_next[i]);
            convergeind = (div_err <= 0.0000001) ? 1 : -1;

            memcpy(B_curr, B_next, (size_t)p * sizeof(double));
            memcpy(v_curr, v_next, (size_t)q * sizeof(double));
            memcpy(alpha_curr, alpha_next, (size_t)n_alpha * sizeof(double));
            rep++;
        }
        if (pt_ok && (!R_FINITE(div_err) || div_err >= 5.0)) {
            pt_ok = fhl_vs_set_err(err_buf, sizeof(err_buf), "frailty.vs inner iteration diverged (|Delta alpha| >= 5)");
        }

        double bic_g = 0.0;
        int valid_g = 0;
        if (pt_ok) {
            const void *vmax = vmaxget();
            pt_ok = fhl_c_lognorm_post_core(n, p, q, nrand, n_alpha, qs, rand_idx,
                                            &xz, &risk, di, del,
                                            B_curr, v_curr, alpha_curr,
                                            G_mat, wl_diag, hft,
                                            div_err, 0, 1,
                                            &bic_g, se_curr, &valid_g,
                                            err_buf, sizeof(err_buf));
            vmaxset(vmax);
        }

        if (!pt_ok) {
            if (n_grid == 1) Rf_error("%s", err_buf);
            fail_msgs[n_fail++] = fhl_vs_format_point_failure(pen_type, t1, t2, err_buf);
            continue;
        }
        if (first_ok_g < 0) first_ok_g = g;

        last_rep = rep;
        last_conv = convergeind;
        double bic_r = Rf_fround(bic_g, 5.0);
        if (!ISNAN(bic_r) && ((valid_g && !has_best) || (valid_g && has_best == 1 && bic_r < best_bic_r) || (!valid_g && !has_best) || (valid_g && has_best == -1))) {
            has_best = valid_g ? 1 : -1;
            best_g = g;
            best_bic = bic_g;
            best_bic_r = bic_r;
            best_rep = rep;
            best_conv = valid_g ? convergeind : 0;
            memcpy(best_B, B_curr, (size_t)p * sizeof(double));
            memcpy(best_se, se_curr, (size_t)p * sizeof(double));
            memcpy(best_v, v_curr, (size_t)q * sizeof(double));
            memcpy(best_alpha, alpha_curr, (size_t)n_alpha * sizeof(double));
        }
    }

    fhl_vs_report_grid_failures(n_grid, n_fail, fail_msgs);

    if (!has_best) {
        best_g = (first_ok_g >= 0) ? first_ok_g : 0;
        best_bic = NA_REAL;
        best_rep = last_rep;
        best_conv = last_conv;
        memcpy(best_B, B_curr, (size_t)p * sizeof(double));
        memcpy(best_se, se_curr, (size_t)p * sizeof(double));
        memcpy(best_v, v_curr, (size_t)q * sizeof(double));
        memcpy(best_alpha, alpha_curr, (size_t)n_alpha * sizeof(double));
    }

    return fhl_vs_build_result_list(p, q, n_alpha, 1, pen_type,
                                     tunset1[best_g],
                                     (tunset2 != NULL) ? tunset2[best_g] : 0.0,
                                     best_B, best_se, best_v, best_alpha,
                                     best_bic, best_rep, best_conv);
}
