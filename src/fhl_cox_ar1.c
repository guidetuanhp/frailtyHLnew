/* ==========================================================================
 * Cox AR(1) Frailty Model Engine (fhl_fit_cox_ar1)
 *
 *   - fhl_c_ar1_step : Single penalized h-likelihood & dispersion update step
 *   - fhl_c_ar1_se   : Post-fit dispersion standard errors, likelihoods & AIC
 *   - fhl_fit_cox_ar1: Complete outer solver using O(n) tridiagonal precision
 *                      IArho, sparse [X, I_n] operator, and Schur complement
 *
 * Formulation notes:
 *   - Uses Schall-type fixed-point updates for alpha_h and cubic score/Hessian
 *     Newton updates for autocorrelation rho.
 *   - Evaluates AR(1) precision U = IArho / alpha across mord = 1 (H_zz in
 *     Laplace beta score), dord = 0, 1, 2, and post-fit H22 = H0_zz + IArho / alpha.
 *   - Analytical block-tridiagonal precision IArho per subject of size q_i:
 *     * q_i == 1: [1 - rho^2]
 *     * q_i >= 2: diag = [1, 1+rho^2, ..., 1+rho^2, 1], off-diag = -rho
 *     * Closed-form log|IArho| = q_active * log|1 - rho^2| in O(1).
 * ========================================================================== */

#include "fhl_linalg.h"
#include "fhl_native_api.h"
#include <Rmath.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#undef exp
#undef log
#define exp(x) fhl_c_r_exp(x)
#define log(x) fhl_c_r_log(x)

/* Compact O(n) general cluster-permutation-safe representation of AR(1) precision IArho */
typedef struct {
    int n;
    int qq;
    int q_active;
    int n_edges;
    double rho;
    int *role;    /* length n: 0 = singleton (m=1), 1 = endpoint (m>=2), 2 = interior (m>=3) */
    int *edge_u;  /* length n_edges (or 1) */
    int *edge_v;  /* length n_edges (or 1) */
    double *diag; /* length n */
} fhl_c_ar1_prec_t;

typedef struct {
    int cluster;
    int seq;
    int row;
} fhl_ar1_obs_t;

static int fhl_ar1_obs_cmp(const void *pa, const void *pb) {
    const fhl_ar1_obs_t *a = (const fhl_ar1_obs_t *)pa;
    const fhl_ar1_obs_t *b = (const fhl_ar1_obs_t *)pb;
    if (a->cluster != b->cluster) return (a->cluster < b->cluster) ? -1 : 1;
    if (a->seq != b->seq)         return (a->seq < b->seq) ? -1 : 1;
    return (a->row < b->row) ? -1 : (a->row > b->row ? 1 : 0);
}

/* Update diagonal values of IArho for a given rho in O(n) */
static void fhl_c_ar1_prec_update_rho(fhl_c_ar1_prec_t *prec, double rho) {
    const double rho2 = rho * rho;
    const double d_single = 1.0 - rho2;
    const double d_end = 1.0;
    const double d_mid = 1.0 + rho2;
    for (int r = 0; r < prec->n; ++r) {
        int rl = prec->role[r];
        prec->diag[r] = (rl == 0) ? d_single : ((rl == 1) ? d_end : d_mid);
    }
    prec->rho = rho;
}

/* Extract row-to-cluster mapping and within-cluster longitudinal order without assuming contiguous clusters */
static void fhl_c_ar1_init_topology(
    fhl_c_ar1_prec_t *prec,
    SEXP z_input, int n
) {
    SEXP z1_s = (TYPEOF(z_input) == VECSXP && !Rf_isS4(z_input))
        ? VECTOR_ELT(z_input, 0) : z_input;

    int *cluster_id = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int *seq_arr    = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    for (int r = 0; r < n; ++r) {
        cluster_id[r] = -1;
        seq_arr[r] = r;
    }

    SEXP s_seq = Rf_getAttrib(z1_s, Rf_install("fhl_ar1_seq"));
    if (!Rf_isNull(s_seq)) {
        if (TYPEOF(s_seq) != INTSXP || Rf_length(s_seq) != n) {
            Rf_error("Invalid AR(1) sequence attribute length");
        }
        const int *sp = INTEGER(s_seq);
        for (int r = 0; r < n; ++r) {
            if (sp[r] == NA_INTEGER) Rf_error("Missing value in AR(1) ordering sequence");
            seq_arr[r] = sp[r];
        }
    }

    int qq = 0;
    SEXP s_grp = Rf_getAttrib(z1_s, Rf_install("fhl_group"));
    if (TYPEOF(z1_s) == INTSXP && !Rf_isMatrix(z1_s)) {
        qq = Rf_length(z1_s);
        const int *src = INTEGER(z1_s);
        int off = 0;
        for (int c = 0; c < qq; ++c) {
            int qi = src[c];
            if (qi < 0 || off + qi > n) Rf_error("Invalid AR(1) cluster sizes");
            for (int k = 0; k < qi; ++k) cluster_id[off + k] = c;
            off += qi;
        }
        if (off != n) Rf_error("AR(1) cluster sizes do not sum to n");
    } else if (!Rf_isNull(s_grp)) {
        if (TYPEOF(s_grp) != INTSXP || Rf_length(s_grp) != n) {
            Rf_error("Invalid fhl_group in AR(1) design");
        }
        SEXP s_dim = Rf_getAttrib(z1_s, R_DimSymbol);
        qq = (TYPEOF(s_dim) == INTSXP && Rf_length(s_dim) == 2) ? INTEGER(s_dim)[1] : 0;
        const int *grp = INTEGER(s_grp);
        for (int r = 0; r < n; ++r) {
            int g = grp[r];
            if (g == NA_INTEGER || g < 1) Rf_error("Invalid cluster index in AR(1) design");
            if (g > qq) qq = g;
            cluster_id[r] = g - 1;
        }
    } else if (Rf_isS4(z1_s) && Rf_inherits(z1_s, "dgCMatrix")) {
        SEXP s_dim = R_do_slot(z1_s, Rf_install("Dim"));
        SEXP s_p   = R_do_slot(z1_s, Rf_install("p"));
        SEXP s_i   = R_do_slot(z1_s, Rf_install("i"));
        SEXP s_x   = R_do_slot(z1_s, Rf_install("x"));
        if (INTEGER(s_dim)[0] != n) Rf_error("AR(1) Z matrix row count does not match n");
        qq = INTEGER(s_dim)[1];
        const int *pp = INTEGER(s_p);
        const int *pi = INTEGER(s_i);
        const double *px = REAL(s_x);
        for (int c = 0; c < qq; ++c) {
            for (int t = pp[c]; t < pp[c + 1]; ++t) {
                double xv = px[t];
                if (xv == 0.0) continue;
                if (fabs(xv - 1.0) > 1e-12) {
                    Rf_error("AR(1) frailty currently supports only random-intercept terms '(1 | cluster)'");
                }
                int row = pi[t];
                if (row < 0 || row >= n || cluster_id[row] != -1) {
                    Rf_error("AR(1) frailty requires each observation to belong to exactly one cluster");
                }
                cluster_id[row] = c;
            }
        }
    } else {
        if (Rf_nrows(z1_s) != n) Rf_error("AR(1) Z matrix row count does not match n");
        qq = Rf_ncols(z1_s);
        const double *z1_p = REAL(z1_s);
        for (int c = 0; c < qq; ++c) {
            for (int r = 0; r < n; ++r) {
                double xv = z1_p[r + (size_t)n * c];
                if (xv == 0.0) continue;
                if (fabs(xv - 1.0) > 1e-12) {
                    Rf_error("AR(1) frailty currently supports only random-intercept terms '(1 | cluster)'");
                }
                if (cluster_id[r] != -1) {
                    Rf_error("AR(1) frailty requires each observation to belong to exactly one cluster");
                }
                cluster_id[r] = c;
            }
        }
    }

    for (int r = 0; r < n; ++r) {
        if (cluster_id[r] < 0 || cluster_id[r] >= qq) {
            Rf_error("AR(1) frailty requires every observation to belong to a valid cluster");
        }
    }

    fhl_ar1_obs_t *obs = (fhl_ar1_obs_t *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(fhl_ar1_obs_t));
    for (int r = 0; r < n; ++r) {
        obs[r].cluster = cluster_id[r];
        obs[r].seq     = seq_arr[r];
        obs[r].row     = r;
    }
    if (n > 1) {
        qsort(obs, (size_t)n, sizeof(fhl_ar1_obs_t), fhl_ar1_obs_cmp);
    }

    int *role   = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int *edge_u = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int *edge_v = (int *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(int));
    double *diag_buf = (double *)R_alloc((size_t)(n > 0 ? n : 1), sizeof(double));

    int q_act = 0;
    int e_cnt = 0;
    int pos = 0;
    while (pos < n) {
        int c_id = obs[pos].cluster;
        int end = pos + 1;
        while (end < n && obs[end].cluster == c_id) ++end;
        int m_i = end - pos;
        q_act++;
        if (m_i == 1) {
            role[obs[pos].row] = 0;
        } else {
            for (int k = 0; k < m_i; ++k) {
                role[obs[pos + k].row] = (k == 0 || k == m_i - 1) ? 1 : 2;
            }
            for (int k = 0; k < m_i - 1; ++k) {
                edge_u[e_cnt] = obs[pos + k].row;
                edge_v[e_cnt] = obs[pos + k + 1].row;
                e_cnt++;
            }
        }
        pos = end;
    }

    if (e_cnt == 0) {
        Rf_error("AR(1) correlation requires repeated observations within at least one cluster");
    }

    prec->n = n;
    prec->qq = qq;
    prec->q_active = q_act;
    prec->n_edges = e_cnt;
    prec->rho = 0.0;
    prec->role = role;
    prec->edge_u = edge_u;
    prec->edge_v = edge_v;
    prec->diag = diag_buf;
}

/* Compute out = IArho %*% v in O(n) using general cluster edge representation */
static void fhl_c_ar1_prec_mv(const fhl_c_ar1_prec_t *prec, const double *v, double *out) {
    const int n = prec->n;
    const double *d = prec->diag;
    const double neg_rho = -prec->rho;
    for (int r = 0; r < n; ++r) {
        out[r] = d[r] * v[r];
    }
    const int ne = prec->n_edges;
    const int *eu = prec->edge_u;
    const int *ev = prec->edge_v;
    for (int e = 0; e < ne; ++e) {
        int u = eu[e], w = ev[e];
        out[u] += neg_rho * v[w];
        out[w] += neg_rho * v[u];
    }
}

/* Add scale * IArho into the [offset:offset+n, offset:offset+n] subblock of mat (ld x ld) */
static void fhl_c_ar1_add_scaled_prec(
    double *mat, int ld, int offset, int n, double scale,
    const fhl_c_ar1_prec_t *prec
) {
    const double off_val = (-prec->rho) * scale;
    for (int r = 0; r < n; ++r) {
        mat[(offset + r) + (size_t)ld * (offset + r)] += prec->diag[r] * scale;
    }
    for (int e = 0; e < prec->n_edges; ++e) {
        int u = prec->edge_u[e], w = prec->edge_v[e];
        mat[(offset + u) + (size_t)ld * (offset + w)] += off_val;
        mat[(offset + w) + (size_t)ld * (offset + u)] += off_val;
    }
}

/* Single penalized h-likelihood & dispersion iteration (sparse [X, I_n] & O(n) tridiagonal IArho) */
static void fhl_c_ar1_step_core(
    int n, int p, int m,
    const double *x, const fhl_c_xz_t *xz,
    const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h0, const double *v_h0, double alpha_h0_in,
    int mord, int dord, int varfixed, int varnonneg,
    const fhl_c_ar1_prec_t *prec, double rho_0,
    double *beta_h, double *v_h, double *alpha_h0_out, double *alpha_h_out,
    double *dft, double *Hinv, double *clam0, double *H,
    double *se_beta_h, double *H0,
    double *rho_out, double *se_rho_out, int *rho_capped_out
) {
    if (rho_capped_out) *rho_capped_out = 0;
    const int r_tot = p + n;
    double one = 1.0, zero = 0.0;
    int inc = 1;

    double alpha_h0 = (alpha_h0_in < 1e-06) ? 1e-06 : alpha_h0_in;
    double alpha_h = alpha_h0;
    *alpha_h0_out = alpha_h0;
    double ial = 1.0 / alpha_h;

    /* muh = x %*% beta_h0 + v_h0 (since z = I_n) */
    double *muh = (double *)R_alloc(n, sizeof(double));
    double *Wi  = (double *)R_alloc(n, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, x, &n, beta_h0, &inc, &zero, muh, &inc FCONE);
    } else {
        memset(muh, 0, (size_t)n * sizeof(double));
    }
    for (int r = 0; r < n; r++) {
        muh[r] += v_h0[r];
        Wi[r] = exp(muh[r]);
    }

    double *denom = (double *)R_alloc(m, sizeof(double));
    double *cla0  = (double *)R_alloc(m, sizeof(double));
    double *As    = (double *)R_alloc(m, sizeof(double));
    fhl_c_risk_backward(risk, Wi, denom, NULL);
    for (int k = 0; k < m; k++) {
        cla0[k] = di[k] / denom[k];
        As[k] = (cla0[k] * cla0[k]) / di[k];
    }
    fhl_c_risk_forward(risk, cla0, clam0, NULL);

    double *res_vec = (double *)R_alloc(n, sizeof(double));
    for (int r = 0; r < n; r++) res_vec[r] = del[r] - Wi[r] * clam0[r];

    double *dft1 = (double *)R_alloc(p > 0 ? p : 1, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("T", &n, &p, &one, x, &n, res_vec, &inc, &zero, dft1, &inc FCONE);
    }

    /* H0 = .G_mat = fhl_c_xz_gram0(xz, 1, risk, Wi, clam0, As) in O(n^2) with nnz(Z)=n */
    fhl_c_xz_gram0(xz, 1, risk, Wi, clam0, As, H0);

    if (mord == 1 && p > 0) {
        /* Laplace beta score uses H_zz = H0_zz + ial * IArho (Ha, Jeong and Lee, 2017, Eq. 4.9 / 5.10) */
        double *zmatz_u = (double *)R_alloc((size_t)n * n, sizeof(double));
        double *dinv    = (double *)R_alloc((size_t)n * n, sizeof(double));
        for (int c = 0; c < n; c++) {
            memcpy(zmatz_u + (size_t)c * n, H0 + (size_t)(p + c) * r_tot + p, (size_t)n * sizeof(double));
        }
        fhl_c_ar1_add_scaled_prec(zmatz_u, n, 0, n, ial, prec);
        if (!fhl_c_solve_inv(zmatz_u, dinv, n)) {
            Rf_error("Singular matrix in AR1frailtyHL (zmatz_u)");
        }

        double *dv_k     = (double *)R_alloc(n, sizeof(double));
        double *dmu0_k   = (double *)R_alloc(n, sizeof(double));
        double *mt_wxz   = (double *)R_alloc(m, sizeof(double));
        double *dcla0b_k = (double *)R_alloc(m, sizeof(double));
        double *dc       = (double *)R_alloc(n, sizeof(double));
        double *temp4_k  = (double *)R_alloc(m, sizeof(double));
        double *G_dw_z   = (double *)R_alloc((size_t)n * n, sizeof(double));
        double neg_one = -1.0;

        for (int k = 0; k < p; k++) {
            const double *g_zk = H0 + (size_t)k * r_tot + p;
            F77_CALL(dgemv)("N", &n, &n, &neg_one, dinv, &n, g_zk, &inc, &zero, dv_k, &inc FCONE);
            const double *x_col = x + (size_t)k * n;
            for (int r = 0; r < n; r++) {
                dmu0_k[r] = Wi[r] * (x_col[r] + dv_k[r]);
            }
            fhl_c_risk_backward(risk, dmu0_k, mt_wxz, NULL);
            for (int j = 0; j < m; j++) {
                dcla0b_k[j] = -As[j] * mt_wxz[j];
                temp4_k[j] = (2.0 * cla0[j] / di[j]) * dcla0b_k[j];
            }
            fhl_c_risk_forward(risk, dcla0b_k, dc, NULL);
            fhl_c_xz_gram1(xz, 0, risk, Wi, clam0, As, dmu0_k, dc, temp4_k, G_dw_z);

            double ad1_k = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : ad1_k) schedule(static) if (n >= 256)
#endif
            for (size_t idx = 0; idx < (size_t)n * n; idx++) {
                ad1_k += dinv[idx] * G_dw_z[idx];
            }
            dft1[k] -= 0.5 * ad1_k;
        }
    }

    /* dft2 = (del - Wi * clam0) - ial * (IArho %*% v_h0) */
    for (int r = 0; r < p; r++) dft[r] = dft1[r];
    {
        double *IArho_v0 = (double *)R_alloc(n, sizeof(double));
        fhl_c_ar1_prec_mv(prec, v_h0, IArho_v0);
        for (int r = 0; r < n; r++) dft[p + r] = res_vec[r] - ial * IArho_v0[r];
    }

    /* Assemble H = H0 with bottom-right n x n block += ial * IArho */
    memcpy(H, H0, (size_t)r_tot * r_tot * sizeof(double));
    fhl_c_ar1_add_scaled_prec(H, r_tot, p, n, ial, prec);

    /* Compute Hinv (and H22 when dord == 1 || 2) with 1 strict SPD Cholesky + Schur complement */
    double *H22 = (dord == 1 || dord == 2) ? (double *)R_alloc((size_t)n * n, sizeof(double)) : NULL;
    int inv_ok = (p > 0)
        ? fhl_c_lognormal_inverse(H, p, n, NULL, Hinv, H22)
        : fhl_c_solve_inv(H, Hinv, r_tot);
    if (!inv_ok) {
        Rf_error("Singular matrix in AR1frailtyHL (H)");
    }
    if (p == 0 && H22 != NULL) {
        memcpy(H22, Hinv, (size_t)n * (size_t)n * sizeof(double));
    }

    /* be_h = be_h0 + Hinv %*% dft */
    double *step_all = (double *)R_alloc(r_tot, sizeof(double));
    F77_CALL(dgemv)("N", &r_tot, &r_tot, &one, Hinv, &r_tot, dft, &inc, &zero, step_all, &inc FCONE);
    for (int r = 0; r < p; r++) {
        beta_h[r] = beta_h0[r] + step_all[r];
        se_beta_h[r] = sqrt(Hinv[r + (size_t)r_tot * r]);
    }
    for (int r = 0; r < n; r++) {
        v_h[r] = v_h0[r] + step_all[p + r];
    }

    double nonneg_adj = varnonneg ? 2.0 : 0.0;
    if (dord == 0) {
        /* Fixed-(beta, v) score: d p_bv / d alpha |_{v fixed} = 0 => alpha = (v^T IArho v) / (n - gamma0 - nonneg_adj) */
        double *IArho_vh = (double *)R_alloc(n, sizeof(double));
        double tr_hinv_IArho = 0.0, v_IArho_v = 0.0;
        fhl_c_ar1_prec_mv(prec, v_h, IArho_vh);
        for (int j = 0; j < n; j++) {
            v_IArho_v += v_h[j] * IArho_vh[j];
            tr_hinv_IArho += prec->diag[j] * Hinv[(p + j) + (size_t)r_tot * (p + j)];
        }
        for (int e = 0; e < prec->n_edges; ++e) {
            int u = prec->edge_u[e], w = prec->edge_v[e];
            tr_hinv_IArho -= prec->rho * (Hinv[(p + u) + (size_t)r_tot * (p + w)] +
                                          Hinv[(p + w) + (size_t)r_tot * (p + u)]);
        }
        double gamma0 = tr_hinv_IArho / alpha_h;
        if (!varfixed) {
            double denom_a = (double)n - gamma0 - nonneg_adj;
            if (!R_FINITE(v_IArho_v) || !R_FINITE(denom_a) || denom_a == 0.0) {
                Rf_error("Singular dispersion update in AR1frailtyHL");
            }
            alpha_h = v_IArho_v / denom_a;
        }
    } else if (dord == 1 || dord == 2) {
        double ial2 = ial * ial;
        double *IArho_vh = (double *)R_alloc(n, sizeof(double));
        double *c_vh1    = (double *)R_alloc(n, sizeof(double));
        double *dv1      = (double *)R_alloc(n, sizeof(double));
        fhl_c_ar1_prec_mv(prec, v_h, IArho_vh);
        for (int r = 0; r < n; r++) c_vh1[r] = ial2 * IArho_vh[r];
        F77_CALL(dgemv)("N", &n, &n, &one, H22, &n, c_vh1, &inc, &zero, dv1, &inc FCONE);

        double *dWi1 = (double *)R_alloc(n, sizeof(double));
        for (int r = 0; r < n; r++) dWi1[r] = Wi[r] * dv1[r];

        double *mt_dWi1 = (double *)R_alloc(m, sizeof(double));
        double *dcla01  = (double *)R_alloc(m, sizeof(double));
        double *dBi1    = (double *)R_alloc(n, sizeof(double));
        double *dAs1    = (double *)R_alloc(m, sizeof(double));
        fhl_c_risk_backward(risk, dWi1, mt_dWi1, NULL);
        for (int k = 0; k < m; k++) {
            dcla01[k] = -(di[k] / (denom[k] * denom[k])) * mt_dWi1[k];
            dAs1[k] = 2.0 * cla0[k] * dcla01[k] / di[k];
        }
        fhl_c_risk_forward(risk, dcla01, dBi1, NULL);

        double *Hd1 = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
        fhl_c_xz_gram1(xz, 1, risk, Wi, clam0, As, dWi1, dBi1, dAs1, Hd1);
        fhl_c_ar1_add_scaled_prec(Hd1, r_tot, p, n, -ial2, prec);

        double tr_Hinv_Hd1 = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tr_Hinv_Hd1) schedule(static) if (r_tot >= 128)
#endif
        for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
            tr_Hinv_Hd1 += Hinv[idx] * Hd1[idx];
        }
        double gamma1 = -alpha_h * tr_Hinv_Hd1;

        double gamma21 = 0.0;
        if (dord == 2) {
            double *x_beta = (double *)R_alloc(n, sizeof(double));
            if (p > 0) {
                F77_CALL(dgemv)("N", &n, &p, &one, x, &n, beta_h, &inc, &zero, x_beta, &inc FCONE);
            } else {
                memset(x_beta, 0, (size_t)n * sizeof(double));
            }
            double sum_term1 = 0.0;
            for (int r = 0; r < n; r++) {
                double zmuu = exp(x_beta[r]) * clam0[r];
                double u_h_r = exp(v_h[r]);
                double ude1 = u_h_r * dv1[r];
                double zu = zmuu * u_h_r;
                double aa1 = zu + ial;
                double bb1 = (zmuu * ude1) - ial2;
                double aa1_2 = aa1 * aa1;
                double aa1_3 = aa1_2 * aa1;
                double aa1_4 = aa1_2 * aa1_2;
                double term11 = ((aa1 * zmuu * ude1) - (2.0 * zu * bb1)) / aa1_3;
                double term21 = ((2.0 * aa1 * zmuu * zu * ude1) - (3.0 * (zu * zu) * bb1)) / aa1_4;
                sum_term1 += (3.0 * term11) - (5.0 * term21);
            }
            gamma21 = -(alpha_h / 12.0) * sum_term1;
        }

        double v_IArho_v = 0.0;
        for (int r = 0; r < n; r++) v_IArho_v += v_h[r] * IArho_vh[r];
        if (!varfixed) {
            double denom_a = (double)n - gamma1 - gamma21 - nonneg_adj;
            if (!R_FINITE(v_IArho_v) || !R_FINITE(denom_a) || denom_a == 0.0) {
                Rf_error("Singular dispersion update in AR1frailtyHL");
            }
            alpha_h = v_IArho_v / denom_a;
        }
    }
    *alpha_h_out = alpha_h;

    /* Exact O(n) AR(1) rho trace and cubic score/Hessian update (cluster-permutation invariant) */
    double L1 = 0.0, L2 = 0.0, L3 = 0.0;
    const int q_eff = prec->q_active;
    for (int idx = 0; idx < n; ++idx) {
        int h_idx = p + idx;
        double diag_val = Hinv[h_idx + (size_t)r_tot * h_idx] + v_h[idx] * v_h[idx];
        L1 += diag_val;
        int rl = prec->role[idx];
        if (rl == 0) {
            L3 += 2.0 * diag_val;
        } else if (rl == 1) {
            L3 += diag_val;
        }
    }
    for (int e = 0; e < prec->n_edges; ++e) {
        int idx_u = prec->edge_u[e];
        int idx_w = prec->edge_v[e];
        int h_u = p + idx_u;
        int h_w = p + idx_w;
        double sub_val = Hinv[h_w + (size_t)r_tot * h_u] + v_h[idx_w] * v_h[idx_u];
        double sup_val = Hinv[h_u + (size_t)r_tot * h_w] + v_h[idx_u] * v_h[idx_w];
        L2 += 0.5 * (sub_val + sup_val);
    }

    double c1 = (double)(n - q_eff) * (L1 - L3);
    double c2 = (double)(2 * q_eff - n) * L2;
    double c3 = (double)n * L3 - (double)(n + q_eff) * L1;
    double c4 = (double)n * L2;
    double rho2 = rho_0 * rho_0;
    double df_rho = c1 * (rho2 * rho_0) + c2 * rho2 + c3 * rho_0 + c4;
    double d2f_rho = 3.0 * c1 * rho2 + 2.0 * c2 * rho_0 + c3;
    if (!R_FINITE(df_rho) || !R_FINITE(d2f_rho) || d2f_rho == 0.0) {
        Rf_error("Singular Hessian for rho in AR1frailtyHL");
    }
    double rho_next = rho_0 - df_rho / d2f_rho;
    if (!R_FINITE(rho_next)) {
        Rf_error("Non-finite rho update in AR1frailtyHL");
    }
    if (fabs(rho_next) >= 1.0) {
        rho_next = 0.5 * (rho_0 + (rho_next > 0.0 ? 0.99 : -0.99));
        if (rho_capped_out) *rho_capped_out = 1;
    }
    *rho_out = rho_next;
    *se_rho_out = sqrt(fabs(-1.0 / d2f_rho));
}

/* Post-fit dispersion standard errors, marginal likelihoods, and AIC criteria */
static SEXP fhl_c_ar1_se_core(
    int n, int p,
    const double *x, const double *del, const fhl_c_risk_t *risk, const double *di,
    const double *beta_h1, const double *v_h1,
    double alpha_h, double alpha_h1,
    const double *Hinv, const double *clam0, const double *H0,
    const fhl_c_ar1_prec_t *prec
) {
    int r_tot = p + n;
    int m = risk->m;
    double one = 1.0, zero = 0.0;
    int inc = 1;
    double pi_val = 3.14159265359;

    double *eta = (double *)R_alloc(n, sizeof(double));
    double *expeta = (double *)R_alloc(n, sizeof(double));
    double *x_beta = (double *)R_alloc(n, sizeof(double));
    if (p > 0) {
        F77_CALL(dgemv)("N", &n, &p, &one, x, &n, beta_h1, &inc, &zero, x_beta, &inc FCONE);
    } else {
        memset(x_beta, 0, (size_t)n * sizeof(double));
    }
    for (int r = 0; r < n; r++) {
        eta[r] = x_beta[r] + v_h1[r];
        expeta[r] = exp(eta[r]);
    }

    double *term0 = (double *)R_alloc(m, sizeof(double));
    fhl_c_risk_backward(risk, expeta, term0, NULL);

    double hlike1 = 0.0;
    for (int r = 0; r < n; r++) hlike1 += del[r] * eta[r];
    for (int k = 0; k < m; k++) hlike1 -= di[k] * log(term0[k]);

    (void)alpha_h;
    double hlike2 = 0.0, hlike3 = 0.0;
    double logdet_IArho = 0.0;
    if (alpha_h1 > 0.0) {
        logdet_IArho = (double)prec->q_active * log(fabs(1.0 - prec->rho * prec->rho));
        hlike2 = -((double)n / 2.0) * log(2.0 * pi_val) - ((double)n / 2.0) * log(alpha_h1) + 0.5 * logdet_IArho;

        double *IArho_v = (double *)R_alloc(n, sizeof(double));
        fhl_c_ar1_prec_mv(prec, v_h1, IArho_v);
        double quad = 0.0;
        for (int r = 0; r < n; r++) quad += v_h1[r] * IArho_v[r];
        hlike3 = -quad / (2.0 * alpha_h1);
    }
    double hliken = hlike1 + hlike2 + hlike3;

    double hpn1 = 0.0, hpn2 = 0.0, second = 0.0, df1 = 0.0;
    if (alpha_h1 <= 0.0) {
        /* Exact Cox limit (alpha -> 0+): drop frailty terms and evaluate H0_bb */
        double logdet_bb = 0.0;
        if (p > 0) {
            double *H0_bb = (double *)R_alloc((size_t)p * p, sizeof(double));
            double *H0_bb_inv = (double *)R_alloc((size_t)p * p, sizeof(double));
            for (int c = 0; c < p; c++) {
                for (int r = 0; r < p; r++) {
                    H0_bb[r + (size_t)p * c] = H0[r + (size_t)r_tot * c];
                }
            }
            if (!fhl_c_solve_inv(H0_bb, H0_bb_inv, p)) {
                Rf_error("Singular matrix in AR1frailtyHL (Cox limit H0_bb)");
            }
            for (size_t idx = 0; idx < (size_t)p * p; idx++) {
                H0_bb_inv[idx] *= 2.0 * pi_val;
            }
            logdet_bb = fhl_c_logdet_spd(H0_bb_inv, p, 1, 1e-05, 0, 0.0);
        }
        hpn2 = hlike1;
        hpn1 = hlike1 + 0.5 * logdet_bb;
        second = 0.0;
        df1 = (double)p;
    } else {
        double ialph1 = 1.0 / alpha_h1;
        double sum_temp4 = 0.0;
        for (int r = 0; r < n; r++) {
            double zmu = exp(x_beta[r]) * clam0[r];
            double b31 = zmu * exp(v_h1[r]);
            double a21 = b31 + ialph1;
            double a21_2 = a21 * a21;
            double S11 = 3.0 * (b31 / a21_2);
            double S21 = 5.0 * ((b31 * b31) / (a21_2 * a21));
            sum_temp4 += (S11 - S21);
        }
        second = -sum_temp4 / 24.0;

        if (alpha_h1 <= 1e-03) {
            /* Scaled Schur-Sylvester factorization for small alpha without singular-value clipping */
            double *scaled_H22 = (double *)R_alloc((size_t)n * n, sizeof(double));
            for (int c = 0; c < n; c++) {
                for (int r = 0; r < n; r++) {
                    scaled_H22[r + (size_t)n * c] = alpha_h1 * H0[(p + r) + (size_t)r_tot * (p + c)];
                }
            }
            fhl_c_ar1_add_scaled_prec(scaled_H22, n, 0, n, 1.0, prec);
            double logdet_scaled_H22 = fhl_c_logdet_spd(scaled_H22, n, 0, 0.0, 0, 0.0);

            double logdet_2pi_Hinv_bb = 0.0;
            if (p > 0) {
                double *mat_2pi_Hinv_bb = (double *)R_alloc((size_t)p * p, sizeof(double));
                for (int c = 0; c < p; c++) {
                    for (int r = 0; r < p; r++) {
                        mat_2pi_Hinv_bb[r + (size_t)p * c] = 2.0 * pi_val * Hinv[r + (size_t)r_tot * c];
                    }
                }
                logdet_2pi_Hinv_bb = fhl_c_logdet_spd(mat_2pi_Hinv_bb, p, 1, 1e-05, 0, 0.0);
            }

            hpn2 = hlike1 + hlike3 + 0.5 * logdet_IArho - 0.5 * logdet_scaled_H22;
            hpn1 = hpn2 + 0.5 * logdet_2pi_Hinv_bb;
        } else {
            double *mat_2pi_Hinv = (double *)R_alloc((size_t)r_tot * r_tot, sizeof(double));
            for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
                mat_2pi_Hinv[idx] = 2.0 * pi_val * Hinv[idx];
            }
            double logdet1 = fhl_c_logdet_spd(mat_2pi_Hinv, r_tot, 1, 1e-05, 0, 0.0);
            hpn1 = hliken + 0.5 * logdet1;

            /* Exact H22 = H0[p + 1:n, p + 1:n] + IArho %*% iD */
            double *H22_scaled = (double *)R_alloc((size_t)n * n, sizeof(double));
            double inv_2pi = 1.0 / (2.0 * pi_val);
            for (int c = 0; c < n; c++) {
                for (int r = 0; r < n; r++) {
                    H22_scaled[r + (size_t)n * c] = H0[(p + r) + (size_t)r_tot * (p + c)] * inv_2pi;
                }
            }
            fhl_c_ar1_add_scaled_prec(H22_scaled, n, 0, n, ialph1 * inv_2pi, prec);
            double logdet_H22 = fhl_c_logdet_spd(H22_scaled, n, 0, 0.0, 1, 1e+05);
            hpn2 = hliken - 0.5 * logdet_H22;
        }

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : df1) schedule(static) if (r_tot >= 128)
#endif
        for (size_t idx = 0; idx < (size_t)r_tot * r_tot; idx++) {
            df1 += Hinv[idx] * H0[idx];
        }
    }
    double hpn3 = hpn1 + second;

    SEXP out = PROTECT(allocVector(VECSXP, 7));
    SET_VECTOR_ELT(out, 0, ScalarReal(0.0));
    SEXP s_hliken = PROTECT(allocMatrix(REALSXP, 1, 1)); REAL(s_hliken)[0] = hliken;
    SEXP s_hpn1 = PROTECT(allocMatrix(REALSXP, 1, 1));   REAL(s_hpn1)[0] = hpn1;
    SEXP s_hpn2 = PROTECT(allocMatrix(REALSXP, 1, 1));   REAL(s_hpn2)[0] = hpn2;
    SEXP s_hpn3 = PROTECT(allocMatrix(REALSXP, 1, 1));   REAL(s_hpn3)[0] = hpn3;
    SEXP s_hlike1 = PROTECT(allocMatrix(REALSXP, 1, 1)); REAL(s_hlike1)[0] = hlike1;
    SET_VECTOR_ELT(out, 1, s_hliken);
    SET_VECTOR_ELT(out, 2, s_hpn1);
    SET_VECTOR_ELT(out, 3, s_hpn2);
    SET_VECTOR_ELT(out, 4, s_hpn3);
    SET_VECTOR_ELT(out, 5, s_hlike1);
    SET_VECTOR_ELT(out, 6, ScalarReal(df1));

    UNPROTECT(6);
    return out;
}

/* Complete outer loop for AR(1) frailty model + post-fit SE */
SEXP fhl_fit_cox_ar1(
    SEXP x_s, SEXP z_list_s, SEXP y_s, SEXP del_s, SEXP Mi_s,
    SEXP idx2_s, SEXP t2_s, SEXP di_s,
    SEXP beta_init_s, SEXP v_init_s, SEXP alpha_init_s,
    SEXP mord_s, SEXP dord_s, SEXP varfixed_s, SEXP varnonneg_s,
    SEXP max_iter_s, SEXP convergence_s
) {
    if (TYPEOF(x_s) != REALSXP || !Rf_isMatrix(x_s) || Rf_isNull(z_list_s) ||
        TYPEOF(del_s) != REALSXP || TYPEOF(di_s) != REALSXP ||
        TYPEOF(beta_init_s) != REALSXP || TYPEOF(v_init_s) != REALSXP ||
        TYPEOF(alpha_init_s) != REALSXP) {
        Rf_error("Invalid arguments passed to fhl_fit_cox_ar1");
    }
    int n = nrows(x_s), p = ncols(x_s);
    int r_tot = p + n;
    int mord = asInteger(mord_s), dord = asInteger(dord_s);
    int varfixed = asLogical(varfixed_s), varnonneg = asLogical(varnonneg_s);
    int max_iter = asInteger(max_iter_s);
    double tol = asReal(convergence_s);

    fhl_c_ar1_prec_t prec;
    fhl_c_ar1_init_topology(&prec, z_list_s, n);

    fhl_c_risk_t risk;
    fhl_c_unpack_risk(Mi_s, &risk);

    fhl_c_xz_t xz;
    fhl_c_xz_init_identity(&xz, REAL(x_s), n, p);

    double *beta_curr = (double *)R_alloc(p > 0 ? p : 1, sizeof(double));
    double *v_curr    = (double *)R_alloc(n > 0 ? n : 1, sizeof(double));
    if (p > 0) memcpy(beta_curr, REAL(beta_init_s), (size_t)p * sizeof(double));
    if (n > 0) memcpy(v_curr, REAL(v_init_s), (size_t)n * sizeof(double));
    double alpha_curr = REAL(alpha_init_s)[0];
    double rho = 0.5;

    SEXP beta_h0_s = PROTECT(allocMatrix(REALSXP, p, 1));
    SEXP v_h0_s = PROTECT(allocMatrix(REALSXP, n, 1));
    SEXP beta_h_s = PROTECT(allocMatrix(REALSXP, p, 1));
    SEXP v_h_s = PROTECT(allocMatrix(REALSXP, n, 1));
    SEXP alpha_h0_out_s = PROTECT(allocVector(REALSXP, 1));
    SEXP alpha_h_out_s = PROTECT(allocVector(REALSXP, 1));
    SEXP dft_s = PROTECT(allocMatrix(REALSXP, r_tot, 1));
    SEXP Hinv_s = PROTECT(allocMatrix(REALSXP, r_tot, r_tot));
    SEXP clam0_s = PROTECT(allocMatrix(REALSXP, n, 1));
    SEXP H_s = PROTECT(allocMatrix(REALSXP, r_tot, r_tot));
    SEXP se_beta_s = PROTECT(allocMatrix(REALSXP, p, 1));
    SEXP H0_s = PROTECT(allocMatrix(REALSXP, r_tot, r_tot));

    double err = 1.0;
    int print_i = 1;
    double print_err = 1.0;
    double rho_0_last = 0.5, rho_h_last = 0.5, se_rho_last = 0.0;
    fhl_c_ar1_prec_update_rho(&prec, rho);

    for (int ii = 1; ii <= max_iter; ii++) {
        if (err >= tol) {
            const void *vmax = vmaxget();
            fhl_c_ar1_prec_update_rho(&prec, rho);
            if (p > 0) memcpy(REAL(beta_h0_s), beta_curr, (size_t)p * sizeof(double));
            if (n > 0) memcpy(REAL(v_h0_s), v_curr, (size_t)n * sizeof(double));
            rho_0_last = rho;
            int rho_capped = 0;

            fhl_c_ar1_step_core(
                n, p, risk.m,
                REAL(x_s), &xz, REAL(del_s), &risk, REAL(di_s),
                REAL(beta_h0_s), REAL(v_h0_s), alpha_curr,
                mord, dord, varfixed, varnonneg,
                &prec, rho_0_last,
                REAL(beta_h_s), REAL(v_h_s), REAL(alpha_h0_out_s), REAL(alpha_h_out_s),
                REAL(dft_s), REAL(Hinv_s), REAL(clam0_s), REAL(H_s),
                REAL(se_beta_s), REAL(H0_s),
                &rho_h_last, &se_rho_last, &rho_capped
            );
            vmaxset(vmax);

            double alpha_h0_val = REAL(alpha_h0_out_s)[0];
            double alpha_h1_val = REAL(alpha_h_out_s)[0];
            double diff_b = 0.0, diff_v = 0.0;
            for (int j = 0; j < p; j++) diff_b += fabs(REAL(beta_h_s)[j] - REAL(beta_h0_s)[j]);
            for (int j = 0; j < n; j++) diff_v += fabs(REAL(v_h_s)[j] - REAL(v_h0_s)[j]);
            err = fabs(alpha_h0_val - alpha_h1_val) + diff_v + diff_b + fabs(rho_h_last - rho_0_last);
            if (rho_capped && err < tol) err = tol;

            if (p > 0) memcpy(beta_curr, REAL(beta_h_s), (size_t)p * sizeof(double));
            if (n > 0) memcpy(v_curr, REAL(v_h_s), (size_t)n * sizeof(double));
            alpha_curr = alpha_h1_val;
            rho = rho_h_last;
            if (!R_FINITE(err)) {
                print_err = NA_REAL;
                print_i = ii;
                break;
            }
            print_err = err;
            print_i = ii;
        }
    }
    fhl_c_ar1_prec_update_rho(&prec, rho_h_last);

    SEXP res2 = PROTECT(allocVector(VECSXP, 25));
    SET_VECTOR_ELT(res2, 0, x_s);
    SET_VECTOR_ELT(res2, 1, R_NilValue);
    SET_VECTOR_ELT(res2, 2, y_s);
    SET_VECTOR_ELT(res2, 3, del_s);
    SET_VECTOR_ELT(res2, 4, Mi_s);
    SET_VECTOR_ELT(res2, 5, idx2_s);
    SET_VECTOR_ELT(res2, 6, t2_s);
    SET_VECTOR_ELT(res2, 7, di_s);
    SET_VECTOR_ELT(res2, 8, beta_h0_s);
    SET_VECTOR_ELT(res2, 9, v_h0_s);
    SET_VECTOR_ELT(res2, 10, beta_h_s);
    SET_VECTOR_ELT(res2, 11, v_h_s);
    SET_VECTOR_ELT(res2, 12, alpha_h0_out_s);
    SET_VECTOR_ELT(res2, 13, alpha_h_out_s);
    SET_VECTOR_ELT(res2, 14, dft_s);
    SET_VECTOR_ELT(res2, 15, Hinv_s);
    SET_VECTOR_ELT(res2, 16, clam0_s);
    SET_VECTOR_ELT(res2, 17, H_s);
    SET_VECTOR_ELT(res2, 18, R_NilValue);
    SET_VECTOR_ELT(res2, 19, se_beta_s);
    SET_VECTOR_ELT(res2, 20, R_NilValue);
    SET_VECTOR_ELT(res2, 21, H0_s);
    SET_VECTOR_ELT(res2, 22, ScalarReal(rho_0_last));
    SET_VECTOR_ELT(res2, 23, ScalarReal(rho_h_last));
    SET_VECTOR_ELT(res2, 24, ScalarReal(se_rho_last));

    SEXP res3 = PROTECT(fhl_c_ar1_se_core(
        n, p,
        REAL(x_s), REAL(del_s), &risk, REAL(di_s),
        REAL(beta_h_s), REAL(v_h_s),
        REAL(alpha_h0_out_s)[0], REAL(alpha_h_out_s)[0],
        REAL(Hinv_s), REAL(clam0_s), REAL(H0_s),
        &prec
    ));

    SEXP out = PROTECT(allocVector(VECSXP, 5));
    SET_VECTOR_ELT(out, 0, res2);
    SET_VECTOR_ELT(out, 1, res3);
    SET_VECTOR_ELT(out, 2, ScalarInteger(print_i));
    SET_VECTOR_ELT(out, 3, ScalarReal(print_err));
    SET_VECTOR_ELT(out, 4, R_NilValue);

    UNPROTECT(15);
    return out;
}
