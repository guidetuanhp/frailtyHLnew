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
 * Competing Risks H-Likelihood Engine (hlike.frailty: cause-specific & Fine-Gray)
 * ========================================================================== */

/* Compute Kaplan-Meier censoring survival G(t) for sorted time[0..N-1] and
   index[0..N-1] (censoring event is index == 0).
   Evaluates G_obs[i] = G(time[i]) and G_y[j] = G(y[j]) (for sorted y[0..ny-1]). */
static void fhl_hlike_compute_km_censoring(
    const double *time, const int *index, int N,
    const double *y, int ny,
    double *G_obs, double *G_y
) {
    int any_cens = 0;
    for (int i = 0; i < N; i++) {
        if (index[i] == 0) {
            any_cens = 1;
            break;
        }
    }
    if (!any_cens) {
        for (int i = 0; i < N; i++) G_obs[i] = 1.0;
        for (int j = 0; j < ny; j++) G_y[j] = 1.0;
        return;
    }

    double S_km = 1.0;
    int i = 0;
    while (i < N) {
        double t_val = time[i];
        int i_end = i + 1;
        while (i_end < N && time[i_end] == t_val) i_end++;
        int n_risk = N - i;
        int d_c = 0;
        for (int r = i; r < i_end; r++) {
            if (index[r] == 0) d_c++;
        }
        if (d_c > 0) {
            S_km *= (1.0 - (double)d_c / (double)n_risk);
        }
        for (int r = i; r < i_end; r++) {
            G_obs[r] = S_km;
        }
        i = i_end;
    }

    int ti = 0;
    for (int j = 0; j < ny; j++) {
        double yj = y[j];
        while (ti < N && time[ti] <= yj) ti++;
        G_y[j] = (ti == 0) ? 1.0 : G_obs[ti - 1];
    }
}

/* Unified O(N + ny) backward risk-set sum: out = t(M) %*% b (length ny)
   M[i, j] = 1 if j < hi[i] else (u_ipcw[i] * g_y[j]). */
static void fhl_hlike_risk_backward(
    const int *hi, int N, int ny, int subhazard,
    const double *u_ipcw, const double *g_y,
    const double *b, double *out, double *work_2ny2
) {
    memset(work_2ny2, 0, (size_t)2 * (size_t)(ny + 1) * sizeof(double));
    double *end_b = work_2ny2;
    double *end_ub = work_2ny2 + (ny + 1);

    for (int i = 0; i < N; i++) {
        int h = hi[i];
        end_b[h] += b[i];
        if (subhazard) end_ub[h] += u_ipcw[i] * b[i];
    }

    long double suf = 0.0L;
    for (int r = ny; r >= 1; r--) {
        suf += end_b[r];
        out[r - 1] = (double)suf;
    }

    if (subhazard) {
        long double pref = 0.0L;
        for (int j = 0; j < ny; j++) {
            pref += end_ub[j];
            out[j] += g_y[j] * (double)pref;
        }
    }
}

/* Unified O(N + ny) forward risk-set sum: out = M %*% lam (length N) */
static void fhl_hlike_risk_forward(
    const int *hi, int N, int ny, int subhazard,
    const double *u_ipcw, const double *g_y,
    const double *lam, double *out, double *work_2ny2
) {
    double *pref_lam = work_2ny2;
    double *suf_glam = work_2ny2 + (ny + 1);

    pref_lam[0] = 0.0;
    long double sp = 0.0L;
    for (int j = 0; j < ny; j++) {
        sp += lam[j];
        pref_lam[j + 1] = (double)sp;
    }

    if (subhazard) {
        suf_glam[ny] = 0.0;
        long double ss = 0.0L;
        for (int j = ny - 1; j >= 0; j--) {
            ss += g_y[j] * lam[j];
            suf_glam[j] = (double)ss;
        }
        for (int i = 0; i < N; i++) {
            int h = hi[i];
            out[i] = pref_lam[h] + u_ipcw[i] * suf_glam[h];
        }
    } else {
        for (int i = 0; i < N; i++) {
            out[i] = pref_lam[hi[i]];
        }
    }
}

/* Compute cumulative risk weights for Gram sweep */
static void fhl_hlike_cum_a(
    const double *a, int ny, int subhazard, const double *g_y,
    double *C0, double *C1, double *T2
) {
    C0[0] = 0.0;
    long double s0 = 0.0L;
    for (int j = 0; j < ny; j++) {
        s0 += a[j];
        C0[j + 1] = (double)s0;
    }
    if (subhazard) {
        C1[0] = 0.0;
        long double s1 = 0.0L;
        for (int j = 0; j < ny; j++) {
            s1 += a[j] * g_y[j];
            C1[j + 1] = (double)s1;
        }
        T2[ny] = 0.0;
        long double s2 = 0.0L;
        for (int j = ny - 1; j >= 0; j--) {
            s2 += a[j] * g_y[j] * g_y[j];
            T2[j] = (double)s2;
        }
    }
}

/* Exact O(N * (p + n)) Gram matrix sweep on sparse B = [X, Z_group] without
   allocating N x n matrix Z or N x ny matrix atRisk.
   Supports both cause-specific (subhazard=0) and Fine-Gray IPCW (subhazard=1),
   and levels lev = 0 (W), 1 (dW), 2 (d2W cross). */
static void fhl_hlike_gram_sweep(
    const double *X, const int *group, int N, int p, int n,
    const int *hi, int ny, int subhazard, const double *u_ipcw, const double *g_y,
    int lev,
    const double *w, const double *b, const double *a,
    const double *dw1, const double *db1, const double *da1,
    const double *dw2, const double *db2, const double *da2,
    const double *ddw12, const double *ddb12, const double *dda12,
    double *G_out
) {
    const void *vmax = vmaxget();
    int k = p + n;

    double *C0_a = (double *)R_alloc((size_t)(ny + 1), sizeof(double));
    double *C1_a = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
    double *T2_a = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
    fhl_hlike_cum_a(a, ny, subhazard, g_y, C0_a, C1_a, T2_a);

    double *C0_da1 = NULL, *C1_da1 = NULL, *T2_da1 = NULL;
    double *C0_da2 = NULL, *C1_da2 = NULL, *T2_da2 = NULL;
    double *C0_dda = NULL, *C1_dda = NULL, *T2_dda = NULL;

    if (lev >= 1) {
        C0_da1 = (double *)R_alloc((size_t)(ny + 1), sizeof(double));
        C1_da1 = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        T2_da1 = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        fhl_hlike_cum_a(da1, ny, subhazard, g_y, C0_da1, C1_da1, T2_da1);
    }
    if (lev == 2) {
        C0_da2 = (double *)R_alloc((size_t)(ny + 1), sizeof(double));
        C1_da2 = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        T2_da2 = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        fhl_hlike_cum_a(da2, ny, subhazard, g_y, C0_da2, C1_da2, T2_da2);

        C0_dda = (double *)R_alloc((size_t)(ny + 1), sizeof(double));
        C1_dda = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        T2_dda = subhazard ? (double *)R_alloc((size_t)(ny + 1), sizeof(double)) : NULL;
        fhl_hlike_cum_a(dda12, ny, subhazard, g_y, C0_dda, C1_dda, T2_dda);
    }

    double *PA_w   = (double *)R_alloc(k, sizeof(double)); memset(PA_w, 0, (size_t)k * sizeof(double));
    double *PA_dw1 = (lev >= 1) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    double *PA_dw2 = (lev == 2) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    double *PA_ddw = (lev == 2) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    if (PA_dw1) memset(PA_dw1, 0, (size_t)k * sizeof(double));
    if (PA_dw2) memset(PA_dw2, 0, (size_t)k * sizeof(double));
    if (PA_ddw) memset(PA_ddw, 0, (size_t)k * sizeof(double));

    double *PU_w   = subhazard ? (double *)R_alloc(k, sizeof(double)) : NULL;
    double *PU_dw1 = (subhazard && lev >= 1) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    double *PU_dw2 = (subhazard && lev == 2) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    double *PU_ddw = (subhazard && lev == 2) ? (double *)R_alloc(k, sizeof(double)) : NULL;
    if (PU_w)   memset(PU_w,   0, (size_t)k * sizeof(double));
    if (PU_dw1) memset(PU_dw1, 0, (size_t)k * sizeof(double));
    if (PU_dw2) memset(PU_dw2, 0, (size_t)k * sizeof(double));
    if (PU_ddw) memset(PU_ddw, 0, (size_t)k * sizeof(double));

    double *row_cross = (double *)R_alloc(k, sizeof(double));
    memset(G_out, 0, (size_t)k * (size_t)k * sizeof(double));

    for (int i = 0; i < N; i++) {
        int r = hi[i];
        int g = group[i];
        double ui = subhazard ? u_ipcw[i] : 0.0;
        double wi = w[i];

        double h_i = 0.0, diagcoef_i = 0.0;
        if (lev == 0) {
            h_i = wi * b[i];
            double k0 = C0_a[r] + (subhazard ? ui * ui * T2_a[r] : 0.0);
            diagcoef_i = wi * wi * k0;
            if (!subhazard) {
                for (int t = 0; t < k; t++) row_cross[t] = wi * PA_w[t];
            } else {
                double ba = C1_a[r] + ui * T2_a[r];
                for (int t = 0; t < k; t++) row_cross[t] = wi * (PA_w[t] + ba * PU_w[t]);
            }
        } else if (lev == 1) {
            double dwi = dw1[i];
            h_i = dwi * b[i] + wi * db1[i];
            double k0 = C0_a[r]   + (subhazard ? ui * ui * T2_a[r]   : 0.0);
            double k1 = C0_da1[r] + (subhazard ? ui * ui * T2_da1[r] : 0.0);
            diagcoef_i = 2.0 * wi * dwi * k0 + wi * wi * k1;
            if (!subhazard) {
                for (int t = 0; t < k; t++) row_cross[t] = wi * PA_w[t] + dwi * PA_dw1[t];
            } else {
                double ba = C1_a[r] + ui * T2_a[r];
                double bda1 = C1_da1[r] + ui * T2_da1[r];
                for (int t = 0; t < k; t++) {
                    row_cross[t] = wi * (PA_w[t] + ba * PU_dw1[t] + bda1 * PU_w[t]) +
                                   dwi * (PA_dw1[t] + ba * PU_w[t]);
                }
            }
        } else {
            double dw1i = dw1[i], dw2i = dw2[i], ddwi = ddw12[i];
            h_i = ddwi * b[i] + dw1i * db2[i] + dw2i * db1[i] + wi * ddb12[i];
            double k0  = C0_a[r]   + (subhazard ? ui * ui * T2_a[r]   : 0.0);
            double k1  = C0_da1[r] + (subhazard ? ui * ui * T2_da1[r] : 0.0);
            double k2  = C0_da2[r] + (subhazard ? ui * ui * T2_da2[r] : 0.0);
            double k12 = C0_dda[r] + (subhazard ? ui * ui * T2_dda[r] : 0.0);
            diagcoef_i = (2.0 * wi * ddwi + 2.0 * dw1i * dw2i) * k0 +
                         2.0 * wi * dw2i * k1 + 2.0 * wi * dw1i * k2 + wi * wi * k12;
            if (!subhazard) {
                for (int t = 0; t < k; t++) {
                    row_cross[t] = wi * PA_w[t] + dw1i * PA_dw1[t] + dw2i * PA_dw2[t] + ddwi * PA_ddw[t];
                }
            } else {
                double ba   = C1_a[r]   + ui * T2_a[r];
                double bda1 = C1_da1[r] + ui * T2_da1[r];
                double bda2 = C1_da2[r] + ui * T2_da2[r];
                double bdda = C1_dda[r] + ui * T2_dda[r];
                for (int t = 0; t < k; t++) {
                    row_cross[t] = wi * (PA_w[t] + ba * PU_ddw[t] + bda1 * PU_dw2[t] + bda2 * PU_dw1[t] + bdda * PU_w[t]) +
                                   dw1i * (PA_dw1[t] + ba * PU_dw2[t] + bda2 * PU_w[t]) +
                                   dw2i * (PA_dw2[t] + ba * PU_dw1[t] + bda1 * PU_w[t]) +
                                   ddwi * (PA_ddw[t] + ba * PU_w[t]);
                }
            }
        }

        double d_half = 0.5 * (h_i - diagcoef_i);
        for (int j = 0; j < p; j++) {
            double bj = X[i + (size_t)N * j];
            if (bj == 0.0) continue;
            double *col_j = G_out + (size_t)j * k;
            for (int t = 0; t < k; t++) col_j[t] -= bj * row_cross[t];
            double dbj = d_half * bj;
            for (int l = 0; l < p; l++) col_j[l] += dbj * X[i + (size_t)N * l];
            col_j[p + g] += dbj;
        }
        {
            double *col_g = G_out + (size_t)(p + g) * k;
            for (int t = 0; t < k; t++) col_g[t] -= row_cross[t];
            for (int l = 0; l < p; l++) col_g[l] += d_half * X[i + (size_t)N * l];
            col_g[p + g] += d_half;
        }

        double al_a = subhazard ? (C0_a[r] - ui * C1_a[r]) : C0_a[r];
        if (lev == 0) {
            double caw = al_a * wi;
            double uw = subhazard ? (ui * wi) : 0.0;
            for (int j = 0; j < p; j++) {
                double bj = X[i + (size_t)N * j];
                if (bj != 0.0) {
                    PA_w[j] += caw * bj;
                    if (subhazard) PU_w[j] += uw * bj;
                }
            }
            PA_w[p + g] += caw;
            if (subhazard) PU_w[p + g] += uw;
        } else if (lev == 1) {
            double dwi = dw1[i];
            double al_da1 = subhazard ? (C0_da1[r] - ui * C1_da1[r]) : C0_da1[r];
            double caw = al_a * dwi + al_da1 * wi;
            double cadw1 = al_a * wi;
            double uw = subhazard ? (ui * wi) : 0.0;
            double udw1 = subhazard ? (ui * dwi) : 0.0;
            for (int j = 0; j < p; j++) {
                double bj = X[i + (size_t)N * j];
                if (bj != 0.0) {
                    PA_w[j]   += caw * bj;
                    PA_dw1[j] += cadw1 * bj;
                    if (subhazard) {
                        PU_w[j]   += uw * bj;
                        PU_dw1[j] += udw1 * bj;
                    }
                }
            }
            PA_w[p + g]   += caw;
            PA_dw1[p + g] += cadw1;
            if (subhazard) {
                PU_w[p + g]   += uw;
                PU_dw1[p + g] += udw1;
            }
        } else {
            double dw1i = dw1[i], dw2i = dw2[i], ddwi = ddw12[i];
            double al_da1 = subhazard ? (C0_da1[r] - ui * C1_da1[r]) : C0_da1[r];
            double al_da2 = subhazard ? (C0_da2[r] - ui * C1_da2[r]) : C0_da2[r];
            double al_dda = subhazard ? (C0_dda[r] - ui * C1_dda[r]) : C0_dda[r];
            double caw   = al_a * ddwi + al_da1 * dw2i + al_da2 * dw1i + al_dda * wi;
            double cadw1 = al_a * dw2i + al_da2 * wi;
            double cadw2 = al_a * dw1i + al_da1 * wi;
            double caddw = al_a * wi;
            double uw   = subhazard ? (ui * wi)   : 0.0;
            double udw1 = subhazard ? (ui * dw1i) : 0.0;
            double udw2 = subhazard ? (ui * dw2i) : 0.0;
            double uddw = subhazard ? (ui * ddwi) : 0.0;
            for (int j = 0; j < p; j++) {
                double bj = X[i + (size_t)N * j];
                if (bj != 0.0) {
                    PA_w[j]   += caw * bj;
                    PA_dw1[j] += cadw1 * bj;
                    PA_dw2[j] += cadw2 * bj;
                    PA_ddw[j] += caddw * bj;
                    if (subhazard) {
                        PU_w[j]   += uw * bj;
                        PU_dw1[j] += udw1 * bj;
                        PU_dw2[j] += udw2 * bj;
                        PU_ddw[j] += uddw * bj;
                    }
                }
            }
            PA_w[p + g]   += caw;
            PA_dw1[p + g] += cadw1;
            PA_dw2[p + g] += cadw2;
            PA_ddw[p + g] += caddw;
            if (subhazard) {
                PU_w[p + g]   += uw;
                PU_dw1[p + g] += udw1;
                PU_dw2[p + g] += udw2;
                PU_ddw[p + g] += uddw;
            }
        }
    }

    for (int j = 0; j < k; j++) {
        G_out[j + (size_t)k * j] *= 2.0;
        for (int t = j + 1; t < k; t++) {
            double val = G_out[t + (size_t)k * j] + G_out[j + (size_t)k * t];
            G_out[t + (size_t)k * j] = val;
            G_out[j + (size_t)k * t] = val;
        }
    }
    vmaxset(vmax);
}

/* 1D Brent root-finding solver for second-order Laplace mode (order == 2):
   dh.dv(v, j) = delta_plus - exp(v) * mu_plus - v * inv_theta */
static FHL_INLINE double fhl_hlike_dh_dv(double v, double delta_plus, double mu_plus, double inv_theta) {
    return delta_plus - exp(v) * mu_plus - v * inv_theta;
}

static double fhl_c_uniroot_brent_dhdv(
    double ax, double bx, double tol, int maxiter,
    double delta_plus, double mu_plus, double inv_theta, int *ok_out
) {
    double a = ax, b = bx, c = ax;
    double fa = fhl_hlike_dh_dv(a, delta_plus, mu_plus, inv_theta);
    double fb = fhl_hlike_dh_dv(b, delta_plus, mu_plus, inv_theta);
    double fc = fa;
    double eps = DBL_EPSILON;
    int iter = 0;

    if (fa == 0.0) { *ok_out = 1; return a; }
    if (fb == 0.0) { *ok_out = 1; return b; }
    if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)) {
        *ok_out = 0;
        return 0.0;
    }

    while (iter < maxiter) {
        double prev_step = b - a;
        if (fabs(fc) < fabs(fb)) {
            a = b;  b = c;  c = a;
            fa = fb; fb = fc; fc = fa;
        }
        double tol1 = 2.0 * eps * fabs(b) + 0.5 * tol;
        double new_step = 0.5 * (c - b);
        if (fabs(new_step) <= tol1 || fb == 0.0) {
            *ok_out = 1;
            return b;
        }
        if (fabs(prev_step) >= tol1 && fabs(fa) > fabs(fb)) {
            double p, q, t1, cb = c - b;
            double s = fb / fa;
            if (a == c) {
                p = cb * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                t1 = fb / fc;
                p = s * (cb * q * (q - t1) - (b - a) * (t1 - 1.0));
                q = (q - 1.0) * (t1 - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            else p = -p;
            if (p < (0.75 * cb * q - 0.5 * fabs(tol1 * q)) &&
                p < fabs(0.5 * prev_step * q)) {
                new_step = p / q;
            }
        }
        if (fabs(new_step) < tol1) {
            new_step = (new_step > 0.0) ? tol1 : -tol1;
        }
        a = b;
        fa = fb;
        b += new_step;
        fb = fhl_hlike_dh_dv(b, delta_plus, mu_plus, inv_theta);
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a;
            fc = fa;
        }
        iter++;
    }
    *ok_out = 1;
    return b;
}

/* Compute log(det(M)) for strictly SPD dim x dim matrix M via Cholesky dpotrf */
static double fhl_hlike_logdet(const double *M, int dim) {
    if (dim <= 0) return 0.0;
    size_t nn_len = (size_t)dim * (size_t)dim;
    for (size_t i = 0; i < nn_len; ++i) {
        if (!R_FINITE(M[i])) Rf_error("Non-finite matrix passed to fhl_hlike_logdet");
    }
    if (dim == 1) {
        if (M[0] <= 0.0) {
            Rf_error("Non-positive-definite matrix passed to fhl_hlike_logdet");
        }
        return log(M[0]);
    }

    for (int j = 0; j < dim; ++j) {
        for (int i = j + 1; i < dim; ++i) {
            double mij = M[i + (size_t)dim * j];
            double mji = M[j + (size_t)dim * i];
            double scale = 1.0;
            if (fabs(mij) > scale) scale = fabs(mij);
            if (fabs(mji) > scale) scale = fabs(mji);
            if (fabs(mij - mji) > 1e-10 * scale) {
                Rf_error("Asymmetric matrix passed to fhl_hlike_logdet");
            }
        }
    }

    size_t mat_bytes = 0;
    if (!fhl_safe_mul3_size((size_t)dim, (size_t)dim, sizeof(double), &mat_bytes)) {
        Rf_error("Allocation size overflow in fhl_hlike_logdet");
    }

    double *work = (double *)malloc(mat_bytes);
    if (!work) Rf_error("Out of memory in fhl_hlike_logdet");
    memcpy(work, M, mat_bytes);

    int info = 0;
    F77_CALL(dpotrf)("U", &dim, work, &dim, &info FCONE);
    if (info != 0) {
        free(work);
        Rf_error("Non-positive-definite matrix passed to fhl_hlike_logdet");
    }
    long double sum_log = 0.0L;
    for (int i = 0; i < dim; i++) {
        sum_log += log(work[i + (size_t)dim * i]);
    }
    free(work);
    return (double)(2.0L * sum_log);
}

/* Construct dSigma(q) (m_s x m_s) for q = 0 .. s-1 (0-based):
   cov_type: 0 = "none", 1 = "independent", 2 = "exchangeable", 3 = "unstructured" */
static void fhl_hlike_dSigma(int cov_type, int m_s, int s, int q, double *dSig) {
    (void)s;
    memset(dSig, 0, (size_t)m_s * m_s * sizeof(double));
    if (cov_type == 0) {
        dSig[0] = 1.0;
    } else if (cov_type == 1) {
        if (q >= 0 && q < m_s) dSig[q + (size_t)m_s * q] = 1.0;
    } else if (cov_type == 2) {
        if (q < m_s) {
            dSig[q + (size_t)m_s * q] = 1.0;
        } else {
            for (int c = 0; c < m_s; c++) {
                for (int r = 0; r < m_s; r++) {
                    if (r != c) dSig[r + (size_t)m_s * c] = 1.0;
                }
            }
        }
    } else if (cov_type == 3) {
        int idx = 0;
        for (int c = 0; c < m_s; c++) {
            for (int r = c; r < m_s; r++) {
                if (idx == q) {
                    dSig[r + (size_t)m_s * c] = 1.0;
                    dSig[c + (size_t)m_s * r] = 1.0;
                    return;
                }
                idx++;
            }
        }
    }
}

/* Small m_s x m_s matrix multiplication: C = A * B */
static void fhl_small_mm(const double *A, const double *B, double *C, int m_s) {
    for (int c = 0; c < m_s; c++) {
        for (int r = 0; r < m_s; r++) {
            double s = 0.0;
            for (int k = 0; k < m_s; k++) {
                s += A[r + (size_t)m_s * k] * B[k + (size_t)m_s * c];
            }
            C[r + (size_t)m_s * c] = s;
        }
    }
}

/* Accumulate single event k Gram matrix Gk ((p+n) x (p+n)) into H0 (dim_H x dim_H)
   and optionally its ZZ block into dH_ZZ_k (n x n) or full H_ZZ. */
static void fhl_hlike_accum_Gk(
    const double *Gk, int k, int p, int n, int cov_type,
    int dim_H, int mp, double *H0, double *ZZ_block_out
) {
    int r_g = p + n;
    int x_off = k * p;
    int z_off = (cov_type == 0) ? 0 : (k * n);

    for (int c = 0; c < p; c++) {
        for (int r = 0; r < p; r++) {
            H0[(x_off + r) + (size_t)dim_H * (x_off + c)] += Gk[r + (size_t)r_g * c];
        }
    }
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < p; r++) {
            H0[(x_off + r) + (size_t)dim_H * (mp + z_off + c)] += Gk[r + (size_t)r_g * (p + c)];
        }
    }
    for (int c = 0; c < p; c++) {
        for (int r = 0; r < n; r++) {
            H0[(mp + z_off + r) + (size_t)dim_H * (x_off + c)] += Gk[(p + r) + (size_t)r_g * c];
        }
    }
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < n; r++) {
            double val = Gk[(p + r) + (size_t)r_g * (p + c)];
            H0[(mp + z_off + r) + (size_t)dim_H * (mp + z_off + c)] += val;
            if (ZZ_block_out) {
                if (cov_type == 0) ZZ_block_out[r + (size_t)n * c] += val;
                else ZZ_block_out[r + (size_t)n * c] = val;
            }
        }
    }
}

/* Trace inner product sum(Hinv_[k] * Gk) without assembling a dim_H x dim_H matrix */
static double fhl_hlike_trace_Hinv_Gk(
    const double *H_inv, int dim_H, int mp,
    const double *Gk, int k, int p, int n, int cov_type
) {
    int r_g = p + n;
    int x_off = k * p;
    int z_off = (cov_type == 0) ? 0 : (k * n);
    double sum = 0.0;

    for (int c = 0; c < p; c++) {
        const double *hcol = H_inv + (size_t)(x_off + c) * dim_H;
        const double *gcol = Gk + (size_t)c * r_g;
        for (int r = 0; r < p; r++) sum += hcol[x_off + r] * gcol[r];
        for (int r = 0; r < n; r++) sum += hcol[mp + z_off + r] * gcol[p + r];
    }
    for (int c = 0; c < n; c++) {
        const double *hcol = H_inv + (size_t)(mp + z_off + c) * dim_H;
        const double *gcol = Gk + (size_t)(p + c) * r_g;
        for (int r = 0; r < p; r++) sum += hcol[x_off + r] * gcol[r];
        for (int r = 0; r < n; r++) sum += hcol[mp + z_off + r] * gcol[p + r];
    }
    return sum;
}

/* Add scale * kronecker(A, diag(n)) (where A is m_s x m_s) into the bottom-right
   (dim_Z x dim_Z) block of M (dim_H x dim_H, starting at offset mp) */
static void fhl_hlike_add_kron_I_scaled(double *M, int dim_H, int mp, const double *A, int m_s, int n, double scale) {
    for (int c2 = 0; c2 < m_s; c2++) {
        for (int c1 = 0; c1 < m_s; c1++) {
            double a_val = scale * A[c1 + (size_t)m_s * c2];
            if (a_val == 0.0) continue;
            for (int j = 0; j < n; j++) {
                int r_idx = mp + c1 * n + j;
                int c_idx = mp + c2 * n + j;
                M[r_idx + (size_t)dim_H * c_idx] += a_val;
            }
        }
    }
}

/* Trace tr(H_inv_ZZ %*% kronecker(A, diag(n))) in O(n * m_s^2) */
static double fhl_hlike_trace_Hinv_kron(const double *H_inv, int dim_H, int mp, const double *A, int m_s, int n) {
    double tr = 0.0;
    for (int c2 = 0; c2 < m_s; c2++) {
        for (int c1 = 0; c1 < m_s; c1++) {
            double a_val = A[c1 + (size_t)m_s * c2];
            if (a_val == 0.0) continue;
            double diag_sum = 0.0;
            for (int j = 0; j < n; j++) {
                diag_sum += H_inv[(mp + c1 * n + j) + (size_t)dim_H * (mp + c2 * n + j)];
            }
            tr += a_val * diag_sum;
        }
    }
    return tr;
}

/* Multiply out = kronecker(A, diag(n)) * vec, where A is m_s x m_s and vec is length m_s * n */
static void fhl_hlike_kron_mv(const double *A, int m_s, int n, const double *vec, double *out, double scale) {
    for (int c1 = 0; c1 < m_s; c1++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int c2 = 0; c2 < m_s; c2++) {
                sum += (scale * A[c1 + (size_t)m_s * c2]) * vec[c2 * n + j];
            }
            out[c1 * n + j] = sum;
        }
    }
}

/* Multiply out = dH_ZZ * vec, where dH_ZZ is stored as:
   - 1 (n x n) matrix if cov_type == 0
   - m (n x n) block-diagonal matrices if cov_type != 0 */
static void fhl_hlike_dHZZ_mv(const double *dH_ZZ_blocks, int m, int n, int cov_type, const double *vec, double *out) {
    double one = 1.0, zero = 0.0;
    int inc = 1;
    if (cov_type == 0) {
        F77_CALL(dsymv)("U", &n, &one, dH_ZZ_blocks, &n, vec, &inc, &zero, out, &inc FCONE);
    } else {
        for (int k = 0; k < m; k++) {
            const double *blk = dH_ZZ_blocks + (size_t)k * n * n;
            F77_CALL(dsymv)("U", &n, &one, blk, &n, vec + (size_t)k * n, &inc, &zero, out + (size_t)k * n, &inc FCONE);
        }
    }
}

/* Complete Newton-Raphson fitting loop for hlike.frailty (order = 0, 1, or 2;
   cause-specific or subHazard = TRUE; sparse group_idx & compressed risk sets) */
SEXP fhl_fit_competing_risks(SEXP X_s, SEXP group_s, SEXP n_s, SEXP time_s, SEXP index_s,
                             SEXP y_list, SEXP d_list, SEXP beta_init_s, SEXP theta_init_s,
                             SEXP v_init_s, SEXP order_s, SEXP cov_type_s, SEXP subhazard_s,
                             SEXP max_iter_s, SEXP tol_s) {
    if (!Rf_isMatrix(X_s) || !Rf_isReal(X_s)) Rf_error("X must be a numeric matrix in fhl_fit_competing_risks");
    if (!Rf_isInteger(group_s)) Rf_error("group must be an integer vector in fhl_fit_competing_risks");
    if (!Rf_isReal(time_s) || !Rf_isInteger(index_s)) Rf_error("Invalid time/index types in fhl_fit_competing_risks");
    if (TYPEOF(y_list) != VECSXP || TYPEOF(d_list) != VECSXP) Rf_error("y_list and d_list must be lists");

    int N = Rf_nrows(X_s);
    int p = Rf_ncols(X_s);
    int n = Rf_asInteger(n_s);
    int m = LENGTH(y_list);
    int order = Rf_asInteger(order_s);
    int cov_type = Rf_asInteger(cov_type_s);
    int subhazard = Rf_asInteger(subhazard_s) ? 1 : 0;
    int max_iter = Rf_asInteger(max_iter_s);
    double tol = Rf_asReal(tol_s);

    if (N <= 0 || n <= 0 || m <= 0 || max_iter <= 0) {
        Rf_error("Invalid dimensions in fhl_fit_competing_risks");
    }
    if (LENGTH(group_s) != N || LENGTH(time_s) != N || LENGTH(index_s) != N || LENGTH(d_list) != m) {
        Rf_error("Length mismatch in fhl_fit_competing_risks");
    }
    if (order < 0 || order > 2 || cov_type < 0 || cov_type > 3) {
        Rf_error("Invalid order or cov_type in fhl_fit_competing_risks");
    }

    int s = 1;
    if (cov_type == 0) s = 1;
    else if (cov_type == 1) s = m;
    else if (cov_type == 2) s = m + 1;
    else if (cov_type == 3) s = (m * (m + 1)) / 2;
    if (order == 2 && s != 1) {
        Rf_error("order == 2 requires s == 1 (frailty.cov = 'none')");
    }
    int m_s = (m < s) ? m : s;
    int npairs = (s * (s + 1)) / 2;

    int mp = m * p;
    int dim_Z = m_s * n;
    int dim_H = mp + dim_Z;
    int r_g = p + n;

    if (Rf_nrows(beta_init_s) != p || Rf_ncols(beta_init_s) != m ||
        Rf_nrows(theta_init_s) != m_s || Rf_ncols(theta_init_s) != m_s ||
        Rf_nrows(v_init_s) != n || Rf_ncols(v_init_s) != m_s) {
        Rf_error("Initial value matrix dimensions do not match in fhl_fit_competing_risks");
    }

    const double *X = REAL(X_s);
    const double *time_p = REAL(time_s);
    const int *index_p = INTEGER(index_s);
    const int *group_1b = INTEGER(group_s);
    int *group = (int *)R_alloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        int gi = group_1b[i];
        if (gi == NA_INTEGER || gi < 1 || gi > n) {
            Rf_error("Cluster index out of bounds [1, %d] at observation %d", n, i + 1);
        }
        group[i] = gi - 1;
    }

    /* Unpack event-specific y, d, delta, and precompute hi[k][0..N-1] in O(N + ny) */
    const double **ys = (const double **)R_alloc(m, sizeof(const double *));
    const double **ds = (const double **)R_alloc(m, sizeof(const double *));
    double **deltas = (double **)R_alloc(m, sizeof(double *));
    int **his = (int **)R_alloc(m, sizeof(int *));
    int *num_ys = (int *)R_alloc(m, sizeof(int));
    int max_ny = 1;
    for (int k = 0; k < m; k++) {
        SEXP yk_s = VECTOR_ELT(y_list, k);
        SEXP dk_s = VECTOR_ELT(d_list, k);
        int ny = LENGTH(yk_s);
        if (ny <= 0 || LENGTH(dk_s) != ny) {
            Rf_error("Invalid y/d vectors for event %d in fhl_fit_competing_risks", k + 1);
        }
        num_ys[k] = ny;
        if (ny > max_ny) max_ny = ny;
        ys[k] = fhl_c_real_ptr(yk_s);
        ds[k] = fhl_c_real_ptr(dk_s);
        deltas[k] = (double *)R_alloc(N, sizeof(double));
        his[k] = (int *)R_alloc(N, sizeof(int));

        const double *yk = ys[k];
        int ptr = 0;
        for (int i = 0; i < N; i++) {
            deltas[k][i] = (index_p[i] == k + 1) ? 1.0 : 0.0;
            while (ptr < ny && yk[ptr] <= time_p[i]) ptr++;
            his[k][i] = ptr;
        }
    }

    /* For subHazard == TRUE (m == 1), compute KM censoring G_obs and G_y once */
    double *u_ipcw = NULL;
    double *g_y = NULL;
    if (subhazard) {
        double *G_obs = (double *)R_alloc(N, sizeof(double));
        g_y = (double *)R_alloc(num_ys[0], sizeof(double));
        u_ipcw = (double *)R_alloc(N, sizeof(double));
        fhl_hlike_compute_km_censoring(time_p, index_p, N, ys[0], num_ys[0], G_obs, g_y);
        for (int i = 0; i < N; i++) {
            u_ipcw[i] = (index_p[i] != 0 && index_p[i] != 1 && G_obs[i] > 0.0) ? (1.0 / G_obs[i]) : 0.0;
        }
    }

    /* For order == 2, precompute delta.i.plus (summed across m event types per cluster) */
    double *delta_sum_cl = NULL;
    double *mu_sum_cl = NULL;
    double *v_tilda = NULL;
    double *exp_Xbeta = NULL;
    if (order == 2) {
        delta_sum_cl = (double *)R_alloc(n, sizeof(double));
        mu_sum_cl = (double *)R_alloc(n, sizeof(double));
        v_tilda = (double *)R_alloc(n, sizeof(double));
        exp_Xbeta = (double *)R_alloc(N, sizeof(double));
        memset(delta_sum_cl, 0, (size_t)n * sizeof(double));
        for (int k = 0; k < m; k++) {
            for (int i = 0; i < N; i++) delta_sum_cl[group[i]] += deltas[k][i];
        }
    }

    /* q.index pairs (0-based q1, q2) */
    int *q1_idx = (int *)R_alloc(npairs, sizeof(int));
    int *q2_idx = (int *)R_alloc(npairs, sizeof(int));
    {
        int idx = 0;
        for (int q1 = 0; q1 < s; q1++) {
            for (int q2 = q1; q2 < s; q2++) {
                q1_idx[idx] = q1;
                q2_idx[idx] = q2;
                idx++;
            }
        }
    }

    /* Precompute dSigma(q) for q = 0 .. s-1 */
    double **dSig_list = (double **)R_alloc(s, sizeof(double *));
    for (int q = 0; q < s; q++) {
        dSig_list[q] = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
        fhl_hlike_dSigma(cov_type, m_s, s, q, dSig_list[q]);
    }

    /* Parameter buffers for current iteration i and next iteration i+1 */
    double *beta_curr = (double *)R_alloc((size_t)p * m, sizeof(double));
    double *beta_next = (double *)R_alloc((size_t)p * m, sizeof(double));
    double *theta_curr = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    double *theta_next = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    double *v_curr = (double *)R_alloc((size_t)n * m_s, sizeof(double));
    double *v_next = (double *)R_alloc((size_t)n * m_s, sizeof(double));

    memcpy(beta_curr, REAL(beta_init_s), (size_t)p * m * sizeof(double));
    memcpy(theta_curr, REAL(theta_init_s), (size_t)m_s * m_s * sizeof(double));
    memcpy(v_curr, REAL(v_init_s), (size_t)n * m_s * sizeof(double));

    /* Per-event working arrays */
    double **exp_eta = (double **)R_alloc(m, sizeof(double *));
    double **denom_k = (double **)R_alloc(m, sizeof(double *));
    double **lambda_0 = (double **)R_alloc(m, sizeof(double *));
    double **Lambda_0 = (double **)R_alloc(m, sizeof(double *));
    double **as_k = (double **)R_alloc(m, sizeof(double *));
    for (int k = 0; k < m; k++) {
        exp_eta[k] = (double *)R_alloc(N, sizeof(double));
        denom_k[k] = (double *)R_alloc(num_ys[k], sizeof(double));
        lambda_0[k] = (double *)R_alloc(num_ys[k], sizeof(double));
        Lambda_0[k] = (double *)R_alloc(N, sizeof(double));
        as_k[k] = (double *)R_alloc(num_ys[k], sizeof(double));
    }

    /* First/second-derivative buffers (allocated leanly based on order & cov_type) */
    double **dv_dtheta = (double **)R_alloc(s, sizeof(double *));
    for (int q = 0; q < s; q++) dv_dtheta[q] = (double *)R_alloc(dim_Z, sizeof(double));
    double **dv2_dtheta2 = (double **)R_alloc(npairs, sizeof(double *));
    for (int q = 0; q < npairs; q++) dv2_dtheta2[q] = (double *)R_alloc(dim_Z, sizeof(double));

    double ***dexp_vec = NULL, ***dlam_vec = NULL, ***dLam_vec = NULL, ***das_vec = NULL;
    double **dH_ZZ_blocks = NULL;
    double *dH_work = NULL;
    double **Hinv_dH_store = NULL;
    double *H_inv_sub = NULL, *dH_sub = NULL;
    int dHZZ_blk_count = (cov_type == 0) ? 1 : m;

    if (order > 0) {
        dexp_vec = (double ***)R_alloc(s, sizeof(double **));
        dlam_vec = (double ***)R_alloc(s, sizeof(double **));
        dLam_vec = (double ***)R_alloc(s, sizeof(double **));
        das_vec  = (double ***)R_alloc(s, sizeof(double **));
        dH_ZZ_blocks = (double **)R_alloc(s, sizeof(double *));
        for (int q = 0; q < s; q++) {
            dexp_vec[q] = (double **)R_alloc(m, sizeof(double *));
            dlam_vec[q] = (double **)R_alloc(m, sizeof(double *));
            dLam_vec[q] = (double **)R_alloc(m, sizeof(double *));
            das_vec[q]  = (double **)R_alloc(m, sizeof(double *));
            for (int k = 0; k < m; k++) {
                dexp_vec[q][k] = (double *)R_alloc(N, sizeof(double));
                dlam_vec[q][k] = (double *)R_alloc(num_ys[k], sizeof(double));
                dLam_vec[q][k] = (double *)R_alloc(N, sizeof(double));
                das_vec[q][k]  = (double *)R_alloc(num_ys[k], sizeof(double));
            }
            dH_ZZ_blocks[q] = (double *)R_alloc((size_t)dHZZ_blk_count * n * n, sizeof(double));
        }
        dH_work = (double *)R_alloc((size_t)dim_H * dim_H, sizeof(double));
        Hinv_dH_store = (double **)R_alloc(s, sizeof(double *));
        if (cov_type == 1 && m > 1) {
            for (int q = 0; q < s; q++) {
                Hinv_dH_store[q] = (double *)R_alloc((size_t)dim_H * r_g, sizeof(double));
            }
            H_inv_sub = (double *)R_alloc((size_t)dim_H * r_g, sizeof(double));
            dH_sub    = (double *)R_alloc((size_t)r_g * r_g, sizeof(double));
        } else {
            for (int q = 0; q < s; q++) {
                Hinv_dH_store[q] = (double *)R_alloc((size_t)dim_H * dim_H, sizeof(double));
            }
        }
    }

    double *G_work = (double *)R_alloc((size_t)r_g * r_g, sizeof(double));
    double *SigmaInv = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    double **S_dS_S = (double **)R_alloc(s, sizeof(double *));
    double **P1_neg = (double **)R_alloc(s, sizeof(double *));
    for (int q = 0; q < s; q++) {
        S_dS_S[q] = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
        P1_neg[q] = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    }
    double **P2_pos = (double **)R_alloc(npairs, sizeof(double *));
    for (int q = 0; q < npairs; q++) {
        P2_pos[q] = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    }

    double *H = (double *)R_alloc((size_t)dim_H * dim_H, sizeof(double));
    double *H_inv = (double *)R_alloc((size_t)dim_H * dim_H, sizeof(double));
    double *HZZ_Q_inv = (double *)R_alloc((size_t)dim_Z * dim_Z, sizeof(double));
    double *G_vec = (double *)R_alloc(dim_H, sizeof(double));
    double *step_bv = (double *)R_alloc(dim_H, sizeof(double));
    double *G_theta = (double *)R_alloc(s, sizeof(double));
    double *H_theta = (double *)R_alloc((size_t)s * s, sizeof(double));
    double *theta_step = (double *)R_alloc(s, sizeof(double));

    double *risk_work = (double *)R_alloc((size_t)2 * (size_t)(max_ny + 1), sizeof(double));
    double *tmp_N1 = (double *)R_alloc(N, sizeof(double));
    double *tmp_N2 = (double *)R_alloc(N, sizeof(double));
    double *tmp_N3 = (double *)R_alloc(N, sizeof(double));
    double *d2exp_k = (double *)R_alloc(N, sizeof(double));
    double *d2Lam_k = (double *)R_alloc(N, sizeof(double));
    double *tmp_Z1 = (double *)R_alloc(dim_Z, sizeof(double));
    double *tmp_Z2 = (double *)R_alloc(dim_Z, sizeof(double));
    double *tmp_ms1 = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    double *tmp_ms2 = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
    double *S_block_tr = (order == 0) ? (double *)R_alloc((size_t)m_s * m_s * m_s * m_s, sizeof(double)) : NULL;

    int converge = 0;
    int final_iter = 1;
    double one = 1.0, zero = 0.0;

    for (int iter = 1; iter <= max_iter; iter++) {
        const void *vmax = vmaxget();
        final_iter = iter;

        /* 1. SigmaInv = solve(theta[,,i]) with strict Cholesky SPD verification */
        if (!fhl_c_solve_inv(theta_curr, SigmaInv, m_s)) {
            Rf_error("Covariance matrix Sigma(theta) is not positive definite in hlike.frailty");
        }

        for (int q = 0; q < s; q++) {
            fhl_small_mm(SigmaInv, dSig_list[q], tmp_ms1, m_s);
            fhl_small_mm(tmp_ms1, SigmaInv, S_dS_S[q], m_s);
            for (int t = 0; t < m_s * m_s; t++) P1_neg[q][t] = -S_dS_S[q][t];
        }
        for (int q = 0; q < npairs; q++) {
            int q1 = q1_idx[q], q2 = q2_idx[q];
            fhl_small_mm(S_dS_S[q2], dSig_list[q1], tmp_ms1, m_s);
            fhl_small_mm(tmp_ms1, SigmaInv, P2_pos[q], m_s);
            fhl_small_mm(SigmaInv, dSig_list[q1], tmp_ms1, m_s);
            fhl_small_mm(tmp_ms1, S_dS_S[q2], tmp_ms2, m_s);
            for (int t = 0; t < m_s * m_s; t++) P2_pos[q][t] += tmp_ms2[t];
        }

        /* 2. Per-event linear predictor, baseline hazard, gradient, and H0 assembly at (beta_curr, v_curr) */
        for (int pass = 0; pass < 2; pass++) {
            const double *b_eval = (pass == 0) ? beta_curr : beta_next;
            const double *v_eval = (pass == 0) ? v_curr : v_next;
            if (pass == 0) memset(G_vec, 0, (size_t)dim_H * sizeof(double));
            memset(H, 0, (size_t)dim_H * dim_H * sizeof(double));
            if (pass == 1 && order == 2) memset(mu_sum_cl, 0, (size_t)n * sizeof(double));

            for (int k = 0; k < m; k++) {
                int v_col = (k < s - 1) ? k : (s - 1);
                const double *beta_k = b_eval + (size_t)k * p;
                const double *v_k = v_eval + (size_t)v_col * n;

                if (p > 0) {
                    F77_CALL(dgemv)("N", &N, &p, &one, X, &N, beta_k, & (int){1}, &zero, tmp_N1, & (int){1} FCONE);
                } else {
                    memset(tmp_N1, 0, (size_t)N * sizeof(double));
                }
                if (pass == 1 && order == 2) {
                    for (int r = 0; r < N; r++) exp_Xbeta[r] = exp(tmp_N1[r]);
                }
                for (int r = 0; r < N; r++) {
                    exp_eta[k][r] = exp(tmp_N1[r] + v_k[group[r]]);
                }

                fhl_hlike_risk_backward(his[k], N, num_ys[k], subhazard, u_ipcw, g_y,
                                        exp_eta[k], denom_k[k], risk_work);
                int ny = num_ys[k];
                for (int j = 0; j < ny; j++) {
                    lambda_0[k][j] = ds[k][j] / denom_k[k][j];
                    as_k[k][j] = (lambda_0[k][j] * lambda_0[k][j]) / ds[k][j];
                }
                fhl_hlike_risk_forward(his[k], N, ny, subhazard, u_ipcw, g_y,
                                       lambda_0[k], Lambda_0[k], risk_work);
                if (pass == 1 && order == 2) {
                    for (int r = 0; r < N; r++) {
                        mu_sum_cl[group[r]] += Lambda_0[k][r] * exp_Xbeta[r];
                    }
                }

                fhl_hlike_gram_sweep(X, group, N, p, n, his[k], ny, subhazard, u_ipcw, g_y,
                                     0, exp_eta[k], Lambda_0[k], as_k[k],
                                     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, G_work);
                fhl_hlike_accum_Gk(G_work, k, p, n, cov_type, dim_H, mp, H, NULL);

                if (pass == 0) {
                    for (int r = 0; r < N; r++) {
                        tmp_N2[r] = deltas[k][r] - exp_eta[k][r] * Lambda_0[k][r];
                    }
                    if (p > 0) {
                        F77_CALL(dgemv)("T", &N, &p, &one, X, &N, tmp_N2, & (int){1}, &zero, G_vec + k * p, & (int){1} FCONE);
                    }
                    int z_off = (cov_type == 0) ? 0 : (k * n);
                    for (int r = 0; r < N; r++) {
                        G_vec[mp + z_off + group[r]] += tmp_N2[r];
                    }
                }
            }

            if (pass == 0) {
                /* Add kronecker(-SigmaInv, diag(n)) %*% v_curr to G_vec[mp .. dim_H-1] */
                fhl_hlike_kron_mv(SigmaInv, m_s, n, v_curr, tmp_Z1, -1.0);
                for (int j = 0; j < dim_Z; j++) G_vec[mp + j] += tmp_Z1[j];
            }

            /* 3. Add Q to H (so H = H0 + Q), then single Cholesky + Schur inversion */
            fhl_hlike_add_kron_I_scaled(H, dim_H, mp, SigmaInv, m_s, n, 1.0);
            if (mp > 0) {
                if (!fhl_c_lognormal_inverse(H, mp, dim_Z, NULL, H_inv, HZZ_Q_inv)) {
                    Rf_error("Singular matrix in hlike.frailty (H)");
                }
            } else {
                if (!fhl_c_solve_inv(H, H_inv, dim_H)) {
                    Rf_error("Hessian matrix H is not positive definite in hlike.frailty");
                }
                memcpy(HZZ_Q_inv, H_inv, (size_t)dim_Z * dim_Z * sizeof(double));
            }

            if (pass == 0) {
                /* Update beta[,,i+1] and v[,,i+1]: est = c(beta, v) + H.inv %*% G */
                F77_CALL(dsymv)("U", &dim_H, &one, H_inv, &dim_H, G_vec, & (int){1}, &zero, step_bv, & (int){1} FCONE);
                for (int r = 0; r < mp; r++) beta_next[r] = beta_curr[r] + step_bv[r];
                for (int r = 0; r < dim_Z; r++) v_next[r] = v_curr[r] + step_bv[mp + r];
            }
        }

        /* 4. First derivatives w.r.t. theta */
        if (order == 0) {
            /* Precompute m_s^4 block Frobenius inner products of Hinv_ZZ once */
            for (int a = 0; a < m_s; a++) {
                for (int c = 0; c < m_s; c++) {
                    for (int d = 0; d < m_s; d++) {
                        for (int b = 0; b < m_s; b++) {
                            double acc = 0.0;
                            for (int col_j = 0; col_j < n; col_j++) {
                                const double *col1 = H_inv + (size_t)(mp + c * n + col_j) * dim_H + (mp + a * n);
                                const double *col2 = H_inv + (size_t)(mp + b * n + col_j) * dim_H + (mp + d * n);
                                for (int row_i = 0; row_i < n; row_i++) {
                                    acc += col1[row_i] * col2[row_i];
                                }
                            }
                            S_block_tr[((a * m_s + c) * m_s + d) * m_s + b] = acc;
                        }
                    }
                }
            }
        }

        for (int q = 0; q < s; q++) {
            fhl_hlike_kron_mv(P1_neg[q], m_s, n, v_next, tmp_Z1, 1.0);
            double neg_one = -1.0;
            F77_CALL(dsymv)("U", &dim_Z, &neg_one, HZZ_Q_inv, &dim_Z, tmp_Z1, & (int){1}, &zero, dv_dtheta[q], & (int){1} FCONE);

            double tr_Hinv_dH = 0.0;
            if (order > 0) {
                memset(dH_work, 0, (size_t)dim_H * dim_H * sizeof(double));
                memset(dH_ZZ_blocks[q], 0, (size_t)dHZZ_blk_count * n * n * sizeof(double));

                for (int k = 0; k < m; k++) {
                    int v_col = (k < s - 1) ? k : (s - 1);
                    const double *dv_qk = dv_dtheta[q] + (size_t)v_col * n;
                    for (int r = 0; r < N; r++) {
                        dexp_vec[q][k][r] = exp_eta[k][r] * dv_qk[group[r]];
                    }
                    int ny = num_ys[k];
                    double *rt_dexp = (double *)R_alloc(ny, sizeof(double));
                    fhl_hlike_risk_backward(his[k], N, ny, subhazard, u_ipcw, g_y,
                                            dexp_vec[q][k], rt_dexp, risk_work);
                    for (int j = 0; j < ny; j++) {
                        dlam_vec[q][k][j] = -(ds[k][j] / (denom_k[k][j] * denom_k[k][j])) * rt_dexp[j];
                        das_vec[q][k][j] = 2.0 * lambda_0[k][j] * dlam_vec[q][k][j] / ds[k][j];
                    }
                    fhl_hlike_risk_forward(his[k], N, ny, subhazard, u_ipcw, g_y,
                                           dlam_vec[q][k], dLam_vec[q][k], risk_work);

                    fhl_hlike_gram_sweep(X, group, N, p, n, his[k], ny, subhazard, u_ipcw, g_y,
                                         1, exp_eta[k], Lambda_0[k], as_k[k],
                                         dexp_vec[q][k], dLam_vec[q][k], das_vec[q][k],
                                         NULL, NULL, NULL, NULL, NULL, NULL, G_work);
                    double *zz_dst = (cov_type == 0) ? dH_ZZ_blocks[q] : (dH_ZZ_blocks[q] + (size_t)k * n * n);
                    fhl_hlike_accum_Gk(G_work, k, p, n, cov_type, dim_H, mp, dH_work, zz_dst);
                    tr_Hinv_dH += fhl_hlike_trace_Hinv_Gk(H_inv, dim_H, mp, G_work, k, p, n, cov_type);
                }
                fhl_hlike_add_kron_I_scaled(dH_work, dim_H, mp, P1_neg[q], m_s, n, 1.0);
                tr_Hinv_dH += fhl_hlike_trace_Hinv_kron(H_inv, dim_H, mp, P1_neg[q], m_s, n);

                if (cov_type == 1 && m > 1) {
                    /* Active-block compression: dH_work is non-zero only on block q (size r_g = p + n) */
                    for (int c = 0; c < p; c++) {
                        int gcol = q * p + c;
                        memcpy(H_inv_sub + (size_t)c * dim_H, H_inv + (size_t)gcol * dim_H, (size_t)dim_H * sizeof(double));
                        for (int r = 0; r < p; r++) dH_sub[r + (size_t)r_g * c] = dH_work[(q * p + r) + (size_t)dim_H * gcol];
                        for (int r = 0; r < n; r++) dH_sub[p + r + (size_t)r_g * c] = dH_work[(mp + q * n + r) + (size_t)dim_H * gcol];
                    }
                    for (int c = 0; c < n; c++) {
                        int gcol = mp + q * n + c;
                        memcpy(H_inv_sub + (size_t)(p + c) * dim_H, H_inv + (size_t)gcol * dim_H, (size_t)dim_H * sizeof(double));
                        for (int r = 0; r < p; r++) dH_sub[r + (size_t)r_g * (p + c)] = dH_work[(q * p + r) + (size_t)dim_H * gcol];
                        for (int r = 0; r < n; r++) dH_sub[p + r + (size_t)r_g * (p + c)] = dH_work[(mp + q * n + r) + (size_t)dim_H * gcol];
                    }
                    F77_CALL(dgemm)("N", "N", &dim_H, &r_g, &r_g, &one, H_inv_sub, &dim_H,
                                    dH_sub, &r_g, &zero, Hinv_dH_store[q], &dim_H FCONE FCONE);
                } else {
                    if (dim_H >= 64) {
                        int nth = fhl_blas_detect_threads(dim_H);
                        fhl_portable_symm_left(dim_H, H_inv, dH_work, Hinv_dH_store[q], nth);
                    } else {
                        F77_CALL(dsymm)("L", "U", &dim_H, &dim_H, &one, H_inv, &dim_H,
                                        dH_work, &dim_H, &zero, Hinv_dH_store[q], &dim_H FCONE FCONE);
                    }
                }
            } else {
                tr_Hinv_dH = fhl_hlike_trace_Hinv_kron(H_inv, dim_H, mp, P1_neg[q], m_s, n);
            }

            /* G.theta[q] */
            fhl_small_mm(SigmaInv, dSig_list[q], tmp_ms1, m_s);
            double tr_SdS = 0.0;
            for (int c = 0; c < m_s; c++) tr_SdS += tmp_ms1[c + (size_t)m_s * c];

            double sum_v = 0.0;
            for (int j = 0; j < n; j++) {
                double quad = 0.0;
                for (int c1 = 0; c1 < m_s; c1++) {
                    double vc1 = v_next[j + (size_t)n * c1];
                    for (int c2 = 0; c2 < m_s; c2++) {
                        quad += vc1 * S_dS_S[q][c1 + (size_t)m_s * c2] * v_next[j + (size_t)n * c2];
                    }
                }
                sum_v += -0.5 * tr_SdS + 0.5 * quad;
            }
            G_theta[q] = sum_v - 0.5 * tr_Hinv_dH;
        }

        /* 5. Second derivatives w.r.t. theta */
        for (int q = 0; q < npairs; q++) {
            int q1 = q1_idx[q], q2 = q2_idx[q];
            if (order > 0) {
                fhl_hlike_dHZZ_mv(dH_ZZ_blocks[q1], m, n, cov_type, dv_dtheta[q2], tmp_Z1);
            } else {
                memset(tmp_Z1, 0, (size_t)dim_Z * sizeof(double));
            }
            fhl_hlike_kron_mv(P1_neg[q1], m_s, n, dv_dtheta[q2], tmp_Z2, 1.0);
            for (int j = 0; j < dim_Z; j++) tmp_Z1[j] += tmp_Z2[j];
            fhl_hlike_kron_mv(P1_neg[q2], m_s, n, dv_dtheta[q1], tmp_Z2, 1.0);
            for (int j = 0; j < dim_Z; j++) tmp_Z1[j] += tmp_Z2[j];
            fhl_hlike_kron_mv(P2_pos[q], m_s, n, v_next, tmp_Z2, 1.0);
            for (int j = 0; j < dim_Z; j++) tmp_Z1[j] += tmp_Z2[j];

            double neg_one = -1.0;
            F77_CALL(dsymv)("U", &dim_Z, &neg_one, HZZ_Q_inv, &dim_Z, tmp_Z1, & (int){1}, &zero, dv2_dtheta2[q], & (int){1} FCONE);

            double tr_HdH = 0.0, tr_Hd2H = 0.0;
            if (order > 0) {
                for (int k = 0; k < m; k++) {
                    int v_col = (k < s - 1) ? k : (s - 1);
                    const double *dv1_k = dv_dtheta[q1] + (size_t)v_col * n;
                    const double *dv2_k = dv_dtheta[q2] + (size_t)v_col * n;
                    const double *ddv_k = dv2_dtheta2[q] + (size_t)v_col * n;

                    for (int r = 0; r < N; r++) {
                        int gr = group[r];
                        double z1 = dv1_k[gr], z2 = dv2_k[gr], zdd = ddv_k[gr];
                        double ee = exp_eta[k][r];
                        d2exp_k[r] = (z1 * z2 + zdd) * ee;
                        tmp_N2[r] = dexp_vec[q2][k][r] * z1;
                        tmp_N3[r] = ee * zdd;
                        tmp_N1[r] = ee * z1;
                    }
                    int ny = num_ys[k];
                    double *c1 = (double *)R_alloc(ny, sizeof(double));
                    double *c2 = (double *)R_alloc(ny, sizeof(double));
                    double *c3 = (double *)R_alloc(ny, sizeof(double));
                    double *d2lam_qk = (double *)R_alloc(ny, sizeof(double));
                    double *d2as_qk = (double *)R_alloc(ny, sizeof(double));

                    fhl_hlike_risk_backward(his[k], N, ny, subhazard, u_ipcw, g_y, tmp_N2, c1, risk_work);
                    fhl_hlike_risk_backward(his[k], N, ny, subhazard, u_ipcw, g_y, tmp_N3, c2, risk_work);
                    fhl_hlike_risk_backward(his[k], N, ny, subhazard, u_ipcw, g_y, tmp_N1, c3, risk_work);
                    for (int j = 0; j < ny; j++) {
                        d2lam_qk[j] = -(as_k[k][j] * c1[j] + as_k[k][j] * c2[j] + das_vec[q2][k][j] * c3[j]);
                        d2as_qk[j] = (2.0 * dlam_vec[q1][k][j] * dlam_vec[q2][k][j] +
                                      2.0 * lambda_0[k][j] * d2lam_qk[j]) / ds[k][j];
                    }
                    fhl_hlike_risk_forward(his[k], N, ny, subhazard, u_ipcw, g_y, d2lam_qk, d2Lam_k, risk_work);

                    fhl_hlike_gram_sweep(X, group, N, p, n, his[k], ny, subhazard, u_ipcw, g_y,
                                         2, exp_eta[k], Lambda_0[k], as_k[k],
                                         dexp_vec[q1][k], dLam_vec[q1][k], das_vec[q1][k],
                                         dexp_vec[q2][k], dLam_vec[q2][k], das_vec[q2][k],
                                         d2exp_k, d2Lam_k, d2as_qk, G_work);
                    tr_Hd2H += fhl_hlike_trace_Hinv_Gk(H_inv, dim_H, mp, G_work, k, p, n, cov_type);
                }
                tr_Hd2H += fhl_hlike_trace_Hinv_kron(H_inv, dim_H, mp, P2_pos[q], m_s, n);

                if (cov_type == 1 && m > 1) {
                    const double *P1 = Hinv_dH_store[q1];
                    const double *P2 = Hinv_dH_store[q2];
                    for (int j2 = 0; j2 < r_g; j2++) {
                        int grow2 = (j2 < p) ? (q2 * p + j2) : (mp + q2 * n + (j2 - p));
                        for (int j1 = 0; j1 < r_g; j1++) {
                            int grow1 = (j1 < p) ? (q1 * p + j1) : (mp + q1 * n + (j1 - p));
                            tr_HdH += P1[grow2 + (size_t)dim_H * j1] * P2[grow1 + (size_t)dim_H * j2];
                        }
                    }
                } else {
                    const double *P1 = Hinv_dH_store[q1];
                    const double *P2 = Hinv_dH_store[q2];
                    double *acc_cols = (double *)R_alloc(dim_H > 0 ? dim_H : 1, sizeof(double));
                    int nth_acc = (dim_H >= 128) ? fhl_blas_detect_threads(dim_H) : 1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nth_acc) schedule(static) if (nth_acc > 1)
#endif
                    for (int c = 0; c < dim_H; c++) {
                        double cs = 0.0;
                        const double *p1_col = P1 + (size_t)c * dim_H;
                        for (int r = 0; r < dim_H; r++) {
                            cs += p1_col[r] * P2[c + (size_t)dim_H * r];
                        }
                        acc_cols[c] = cs;
                    }
                    double acc = 0.0;
                    for (int c = 0; c < dim_H; c++) acc += acc_cols[c];
                    tr_HdH = acc;
                }
            } else {
                tr_Hd2H = fhl_hlike_trace_Hinv_kron(H_inv, dim_H, mp, P2_pos[q], m_s, n);
                const double *A = P1_neg[q1];
                const double *B = P1_neg[q2];
                for (int a = 0; a < m_s; a++) {
                    for (int b = 0; b < m_s; b++) {
                        for (int c = 0; c < m_s; c++) {
                            double acb = A[c + (size_t)m_s * b];
                            if (acb == 0.0) continue;
                            for (int d = 0; d < m_s; d++) {
                                double bda = B[d + (size_t)m_s * a];
                                if (bda != 0.0) {
                                    tr_HdH += acb * bda * S_block_tr[((a * m_s + c) * m_s + d) * m_s + b];
                                }
                            }
                        }
                    }
                }
            }

            /* M_trace = SigmaInv %*% dSigma(q2) %*% SigmaInv %*% dSigma(q1) */
            fhl_small_mm(S_dS_S[q2], dSig_list[q1], tmp_ms1, m_s);
            double tr_part = 0.0;
            for (int c = 0; c < m_s; c++) tr_part += tmp_ms1[c + (size_t)m_s * c];

            double *M_v1 = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
            double *M_v2 = (double *)R_alloc((size_t)m_s * m_s, sizeof(double));
            fhl_small_mm(SigmaInv, dSig_list[q2], tmp_ms1, m_s);
            fhl_small_mm(tmp_ms1, S_dS_S[q1], M_v1, m_s);
            fhl_small_mm(SigmaInv, dSig_list[q1], tmp_ms1, m_s);
            fhl_small_mm(tmp_ms1, S_dS_S[q2], M_v2, m_s);

            double sum_v_h = 0.0;
            for (int j = 0; j < n; j++) {
                double q_v1 = 0.0, q_v2 = 0.0;
                for (int c1 = 0; c1 < m_s; c1++) {
                    double vc1 = v_next[j + (size_t)n * c1];
                    for (int c2 = 0; c2 < m_s; c2++) {
                        double vc2 = v_next[j + (size_t)n * c2];
                        q_v1 += vc1 * M_v1[c1 + (size_t)m_s * c2] * vc2;
                        q_v2 += vc1 * M_v2[c1 + (size_t)m_s * c2] * vc2;
                    }
                }
                sum_v_h += -0.5 * tr_part + 0.5 * q_v1 + 0.5 * q_v2;
            }

            fhl_hlike_kron_mv(P1_neg[q1], m_s, n, dv_dtheta[q2], tmp_Z1, 1.0);
            double v_Qp_dv = 0.0;
            for (int j = 0; j < dim_Z; j++) v_Qp_dv += v_next[j] * tmp_Z1[j];

            double h_val = sum_v_h + v_Qp_dv + 0.5 * (-tr_HdH + tr_Hd2H);
            H_theta[q2 + (size_t)s * q1] = h_val;
            H_theta[q1 + (size_t)s * q2] = h_val;
        }

        /* 6. Update theta[,,i+1] (require observed information H_theta = -d2p/dtheta2 to be strictly PD) */
        if (order == 0 || order == 1) {
            double *H_theta_inv = (double *)R_alloc((size_t)s * s, sizeof(double));
            if (!fhl_c_solve_inv(H_theta, H_theta_inv, s)) {
                Rf_error("dispersion Newton step not ascent (Hessian not negative definite)");
            }
            for (int r = 0; r < s; r++) {
                double st = 0.0;
                for (int c = 0; c < s; c++) st += H_theta_inv[r + (size_t)s * c] * G_theta[c];
                theta_step[r] = st;
            }

            memset(theta_next, 0, (size_t)m_s * m_s * sizeof(double));
            if (cov_type == 0) {
                theta_next[0] = theta_curr[0] + theta_step[0];
            } else if (cov_type == 1) {
                for (int c = 0; c < m_s; c++) {
                    theta_next[c + (size_t)m_s * c] = theta_curr[c + (size_t)m_s * c] + theta_step[c];
                }
            } else if (cov_type == 2) {
                double off_curr = (m_s > 1) ? theta_curr[1] : 0.0;
                double off_next = off_curr + theta_step[s - 1];
                for (int c = 0; c < m_s; c++) {
                    for (int r = 0; r < m_s; r++) {
                        if (r == c) theta_next[r + (size_t)m_s * c] = theta_curr[r + (size_t)m_s * c] + theta_step[r];
                        else theta_next[r + (size_t)m_s * c] = off_next;
                    }
                }
            } else if (cov_type == 3) {
                int idx = 0;
                for (int c = 0; c < m_s; c++) {
                    for (int r = c; r < m_s; r++) {
                        double val = theta_curr[r + (size_t)m_s * c] + theta_step[idx];
                        theta_next[r + (size_t)m_s * c] = val;
                        theta_next[c + (size_t)m_s * r] = val;
                        idx++;
                    }
                }
            }
            if (!fhl_c_solve_inv(theta_next, tmp_ms1, m_s)) {
                Rf_error("Updated covariance matrix Sigma(theta) is not positive definite in hlike.frailty");
            }
        } else {
            /* order == 2: Laplace 2nd order correction using 1D Brent root finder */
            double th = theta_curr[0];
            double inv_th = 1.0 / th;
            double th_m2 = inv_th * inv_th;
            double th_m3 = th_m2 * inv_th;
            double th_m4 = th_m3 * inv_th;
            int *ok_flags = (int *)R_alloc(n > 0 ? n : 1, sizeof(int));
            int nth_br = (n >= 64) ? fhl_blas_detect_threads(n) : 1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(nth_br) schedule(static) if (nth_br > 1)
#endif
            for (int j = 0; j < n; j++) {
                int ok_j = 0;
                v_tilda[j] = fhl_c_uniroot_brent_dhdv(
                    -50.0, 50.0, 1e-3, 1000,
                    delta_sum_cl[j], mu_sum_cl[j], inv_th, &ok_j
                );
                ok_flags[j] = ok_j;
            }
            int fail_cnt = 0;
            for (int j = 0; j < n; j++) {
                if (!ok_flags[j]) fail_cnt++;
            }
            if (fail_cnt > 0) {
                Rf_error("uniroot failed in hlike.frailty (order=2): endpoint values not of opposite sign");
            }

            double trace_dS = 0.0, trace_d2S = 0.0;
            for (int j = 0; j < n; j++) {
                double Aj = exp(v_tilda[j]) * mu_sum_cl[j];
                double Aj2 = Aj * Aj;
                double denom = Aj + inv_th;
                double d_m3 = 1.0 / (denom * denom * denom);
                double d_m4 = d_m3 / denom;
                double d_m5 = d_m4 / denom;
                trace_dS += 6.0 * Aj * th_m2 * d_m3 - 15.0 * Aj2 * th_m2 * d_m4;
                trace_d2S += 12.0 * Aj * th_m3 * d_m3 - 18.0 * Aj * th_m4 * d_m4 -
                             30.0 * Aj2 * th_m3 * d_m4 + 60.0 * Aj2 * th_m4 * d_m5;
            }
            double G_theta_ord2 = G_theta[0] - trace_dS / 24.0;
            double H_theta_ord2 = H_theta[0] - trace_d2S / 24.0;
            if (!R_FINITE(G_theta_ord2) || !R_FINITE(H_theta_ord2) || H_theta_ord2 <= 0.0) {
                Rf_error("dispersion Newton step not ascent (Hessian not negative definite)");
            }
            theta_next[0] = th + G_theta_ord2 / H_theta_ord2;
            if (!R_FINITE(theta_next[0]) || theta_next[0] <= 0.0) {
                Rf_error("Updated covariance matrix Sigma(theta) is not positive definite in hlike.frailty");
            }
        }

        /* 7. Check convergence */
        double max_diff = 0.0;
        for (int j = 0; j < mp; j++) {
            double d = fabs(beta_next[j] - beta_curr[j]);
            if (d > max_diff) max_diff = d;
        }
        for (int j = 0; j < m_s * m_s; j++) {
            double d = fabs(theta_next[j] - theta_curr[j]);
            if (d > max_diff) max_diff = d;
        }
        if (max_diff < tol) {
            memcpy(beta_curr, beta_next, (size_t)mp * sizeof(double));
            memcpy(theta_curr, theta_next, (size_t)m_s * m_s * sizeof(double));
            memcpy(v_curr, v_next, (size_t)dim_Z * sizeof(double));
            converge = 1;
            vmaxset(vmax);
            break;
        }

        memcpy(beta_curr, beta_next, (size_t)mp * sizeof(double));
        memcpy(v_curr, v_next, (size_t)dim_Z * sizeof(double));
        if (iter < max_iter) {
            memcpy(theta_curr, theta_next, (size_t)m_s * m_s * sizeof(double));
        }
        vmaxset(vmax);
    }

    /* 8. Post-processing in C: synchronize H / H_inv / SigmaInv at final converged state */
    SEXP H0_s = PROTECT(Rf_allocMatrix(REALSXP, dim_H, dim_H));
    double *H0_p = REAL(H0_s);
    memcpy(H0_p, H, (size_t)dim_H * dim_H * sizeof(double));
    fhl_hlike_add_kron_I_scaled(H0_p, dim_H, mp, SigmaInv, m_s, n, -1.0);

    if (converge) {
        if (!fhl_c_solve_inv(theta_curr, SigmaInv, m_s)) {
            Rf_error("Covariance matrix Sigma(theta) is not positive definite in hlike.frailty");
        }
        memcpy(H, H0_p, (size_t)dim_H * dim_H * sizeof(double));
        fhl_hlike_add_kron_I_scaled(H, dim_H, mp, SigmaInv, m_s, n, 1.0);
        if (mp > 0) {
            if (!fhl_c_lognormal_inverse(H, mp, dim_Z, NULL, H_inv, HZZ_Q_inv)) {
                Rf_error("Singular matrix in hlike.frailty (H)");
            }
        } else {
            if (!fhl_c_solve_inv(H, H_inv, dim_H)) {
                Rf_error("Hessian matrix H is not positive definite in hlike.frailty");
            }
            memcpy(HZZ_Q_inv, H_inv, (size_t)dim_Z * dim_Z * sizeof(double));
        }
    }

    SEXP H_s = PROTECT(Rf_allocMatrix(REALSXP, dim_H, dim_H));
    memcpy(REAL(H_s), H, (size_t)dim_H * dim_H * sizeof(double));

    double h1 = 0.0;
    for (int k = 0; k < m; k++) {
        for (int r = 0; r < N; r++) {
            h1 += deltas[k][r] * log(exp_eta[k][r]);
        }
        int ny = num_ys[k];
        for (int j = 0; j < ny; j++) {
            h1 -= ds[k][j] * log(denom_k[k][j]);
        }
    }

    double logdet_theta = fhl_hlike_logdet(theta_curr, m_s);
    double logdet_Q_2pi = (double)n * (-logdet_theta - (double)m_s * log(2.0 * M_PI));
    fhl_hlike_kron_mv(SigmaInv, m_s, n, v_curr, tmp_Z1, 1.0);
    double vQv = 0.0;
    for (int j = 0; j < dim_Z; j++) vQv += v_curr[j] * tmp_Z1[j];

    double hp = h1 + 0.5 * logdet_Q_2pi - 0.5 * vQv;
    double logdet_H = fhl_hlike_logdet(H, dim_H);
    double Ahp = hp - 0.5 * logdet_H + 0.5 * (double)dim_H * log(2.0 * M_PI);
    double rAIC = -2.0 * Ahp + 2.0 * (double)s;

    double tr_Hinv_H0 = 0.0;
    for (size_t idx = 0; idx < (size_t)dim_H * dim_H; idx++) {
        tr_Hinv_H0 += H_inv[idx] * H0_p[idx];
    }
    double cAIC = -2.0 * h1 + 2.0 * tr_Hinv_H0;

    /* Return list(beta_i, theta_i, v_i, H, H0, beta_SE, lambda_0,
                   h1, hp, Ahp, rAIC, cAIC, converge, iterations) */
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 14));

    SEXP beta_i_s = PROTECT(Rf_allocMatrix(REALSXP, p, m));
    memcpy(REAL(beta_i_s), beta_curr, (size_t)mp * sizeof(double));
    SET_VECTOR_ELT(out, 0, beta_i_s);

    SEXP theta_i_s = PROTECT(Rf_allocMatrix(REALSXP, m_s, m_s));
    memcpy(REAL(theta_i_s), theta_curr, (size_t)m_s * m_s * sizeof(double));
    SET_VECTOR_ELT(out, 1, theta_i_s);

    SEXP v_i_s = PROTECT(Rf_allocMatrix(REALSXP, n, m_s));
    memcpy(REAL(v_i_s), v_curr, (size_t)dim_Z * sizeof(double));
    SET_VECTOR_ELT(out, 2, v_i_s);

    SET_VECTOR_ELT(out, 3, H_s);
    SET_VECTOR_ELT(out, 4, H0_s);

    SEXP beta_se_s = PROTECT(Rf_allocVector(REALSXP, mp));
    double *se_p = REAL(beta_se_s);
    for (int j = 0; j < mp; j++) {
        se_p[j] = sqrt(H_inv[j + (size_t)dim_H * j]);
    }
    SET_VECTOR_ELT(out, 5, beta_se_s);

    SEXP lam0_list_s = PROTECT(Rf_allocVector(VECSXP, m));
    for (int k = 0; k < m; k++) {
        SEXP lk = PROTECT(Rf_allocVector(REALSXP, num_ys[k]));
        memcpy(REAL(lk), lambda_0[k], (size_t)num_ys[k] * sizeof(double));
        SET_VECTOR_ELT(lam0_list_s, k, lk);
        UNPROTECT(1);
    }
    SET_VECTOR_ELT(out, 6, lam0_list_s);

    SET_VECTOR_ELT(out, 7, Rf_ScalarReal(h1));
    SET_VECTOR_ELT(out, 8, Rf_ScalarReal(hp));
    SET_VECTOR_ELT(out, 9, Rf_ScalarReal(Ahp));
    SET_VECTOR_ELT(out, 10, Rf_ScalarReal(rAIC));
    SET_VECTOR_ELT(out, 11, Rf_ScalarReal(cAIC));
    SET_VECTOR_ELT(out, 12, Rf_ScalarInteger(converge));
    SET_VECTOR_ELT(out, 13, Rf_ScalarInteger(final_iter));

    UNPROTECT(8);
    return out;
}
