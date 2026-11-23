/* Portable C + OpenMP + SIMD dense kernels (Cholesky, triangular inverse, symm).
   Included twice by src/fhl_blas.c: once as a generic build and once with an
   AVX2/FMA target attribute selected at run time on x86-64 CPUs that support it.
   Requires macros FHL_K(name) and FHL_KATTR. */

#ifdef FHL_USE_AVX2_INTRINSICS
#include <immintrin.h>

/* Leaf AVX2/FMA inner-panel helpers contain no `#pragma omp parallel` regions so
   Clang does not outline non-AVX2 `.omp_outlined.` functions containing `__m256d`. */
static FHL_KATTR void FHL_K(trsm_UT_col8_leaf)(int kb, int jmax, int j0, const double *U, int ldu,
                                               double *panel, int ldp, double *Upack) {
    if (jmax == 8) {
        for (int jj = 0; jj < 8; jj++) {
            const double *pj = panel + (size_t)ldp * (j0 + jj);
            for (int i = 0; i < kb; i++)
                Upack[i * 8 + jj] = pj[i];
        }
        for (int i = 0; i < kb; i++) {
            const double *ui = U + (size_t)ldu * i;
            __m256d s0 = _mm256_load_pd(Upack + i * 8);
            __m256d s1 = _mm256_load_pd(Upack + i * 8 + 4);
            for (int l = 0; l < i; l++) {
                __m256d u = _mm256_set1_pd(ui[l]);
                s0 = _mm256_fnmadd_pd(u, _mm256_load_pd(Upack + l * 8), s0);
                s1 = _mm256_fnmadd_pd(u, _mm256_load_pd(Upack + l * 8 + 4), s1);
            }
            __m256d inv_d = _mm256_set1_pd(1.0 / ui[i]);
            _mm256_store_pd(Upack + i * 8, _mm256_mul_pd(s0, inv_d));
            _mm256_store_pd(Upack + i * 8 + 4, _mm256_mul_pd(s1, inv_d));
        }
        for (int jj = 0; jj < 8; jj++) {
            double *pj = panel + (size_t)ldp * (j0 + jj);
            for (int i = 0; i < kb; i++)
                pj[i] = Upack[i * 8 + jj];
        }
    } else {
        for (int jj = 0; jj < jmax; jj++) {
            double *pj = panel + (size_t)ldp * (j0 + jj);
            for (int i = 0; i < kb; i++) {
                const double *ui = U + (size_t)ldu * i;
                double s = 0;
                for (int l = 0; l < i; l++)
                    s += ui[l] * pj[l];
                pj[i] = (pj[i] - s) / ui[i];
            }
        }
    }
}

static void FHL_K(simd_trsm_UT_left)(int kb, int rem, const double *U, int ldu,
                                     double *panel, int ldp, int nth) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nth) if (rem >= 64)
#else
    (void)nth;
#endif
    {
        double Upack[96 * 8] __attribute__((aligned(64)));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int j0 = 0; j0 < rem; j0 += 8) {
            int jmax = (rem - j0 < 8) ? (rem - j0) : 8;
            FHL_K(trsm_UT_col8_leaf)(kb, jmax, j0, U, ldu, panel, ldp, Upack);
        }
    }
}

static FHL_KATTR void FHL_K(syrk_upper_blk8_leaf)(int kb, int rem, int i0, int imax,
                                                  const double *panel, int ldp, double *C, int ldc,
                                                  double *Apack) {
    if (imax == 8) {
        const double *p0 = panel + (size_t)ldp * (i0 + 0),
                     *p1 = panel + (size_t)ldp * (i0 + 1);
        const double *p2 = panel + (size_t)ldp * (i0 + 2),
                     *p3 = panel + (size_t)ldp * (i0 + 3);
        const double *p4 = panel + (size_t)ldp * (i0 + 4),
                     *p5 = panel + (size_t)ldp * (i0 + 5);
        const double *p6 = panel + (size_t)ldp * (i0 + 6),
                     *p7 = panel + (size_t)ldp * (i0 + 7);
        for (int l = 0; l < kb; l++) {
            double *ap = Apack + l * 8;
            ap[0] = p0[l];
            ap[1] = p1[l];
            ap[2] = p2[l];
            ap[3] = p3[l];
            ap[4] = p4[l];
            ap[5] = p5[l];
            ap[6] = p6[l];
            ap[7] = p7[l];
        }
        for (int j = i0; j < i0 + 8; j++) {
            const double *pj = panel + (size_t)ldp * j;
            double *cj = C + (size_t)ldc * j;
            for (int ii = i0; ii <= j; ii++) {
                const double *pi = panel + (size_t)ldp * ii;
                double s = 0;
                for (int l = 0; l < kb; l++)
                    s += pi[l] * pj[l];
                cj[ii] -= s;
            }
        }
        int j0 = i0 + 8;
        for (; j0 + 3 < rem; j0 += 4) {
            const double *bj0 = panel + (size_t)ldp * (j0 + 0),
                         *bj1 = panel + (size_t)ldp * (j0 + 1);
            const double *bj2 = panel + (size_t)ldp * (j0 + 2),
                         *bj3 = panel + (size_t)ldp * (j0 + 3);
            __m256d c00 = _mm256_loadu_pd(C + i0 + (size_t)ldc * (j0 + 0)),
                    c10 = _mm256_loadu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 0));
            __m256d c01 = _mm256_loadu_pd(C + i0 + (size_t)ldc * (j0 + 1)),
                    c11 = _mm256_loadu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 1));
            __m256d c02 = _mm256_loadu_pd(C + i0 + (size_t)ldc * (j0 + 2)),
                    c12 = _mm256_loadu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 2));
            __m256d c03 = _mm256_loadu_pd(C + i0 + (size_t)ldc * (j0 + 3)),
                    c13 = _mm256_loadu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 3));
            for (int l = 0; l < kb; l++) {
                const double *ap = Apack + l * 8;
                __m256d av0 = _mm256_load_pd(ap), av1 = _mm256_load_pd(ap + 4);
                __m256d b0 = _mm256_set1_pd(bj0[l]), b1 = _mm256_set1_pd(bj1[l]);
                __m256d b2 = _mm256_set1_pd(bj2[l]), b3 = _mm256_set1_pd(bj3[l]);
                c00 = _mm256_fnmadd_pd(av0, b0, c00);
                c10 = _mm256_fnmadd_pd(av1, b0, c10);
                c01 = _mm256_fnmadd_pd(av0, b1, c01);
                c11 = _mm256_fnmadd_pd(av1, b1, c11);
                c02 = _mm256_fnmadd_pd(av0, b2, c02);
                c12 = _mm256_fnmadd_pd(av1, b2, c12);
                c03 = _mm256_fnmadd_pd(av0, b3, c03);
                c13 = _mm256_fnmadd_pd(av1, b3, c13);
            }
            _mm256_storeu_pd(C + i0 + (size_t)ldc * (j0 + 0), c00);
            _mm256_storeu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 0), c10);
            _mm256_storeu_pd(C + i0 + (size_t)ldc * (j0 + 1), c01);
            _mm256_storeu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 1), c11);
            _mm256_storeu_pd(C + i0 + (size_t)ldc * (j0 + 2), c02);
            _mm256_storeu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 2), c12);
            _mm256_storeu_pd(C + i0 + (size_t)ldc * (j0 + 3), c03);
            _mm256_storeu_pd(C + i0 + 4 + (size_t)ldc * (j0 + 3), c13);
        }
        for (; j0 < rem; j0++) {
            const double *bj = panel + (size_t)ldp * j0;
            __m256d c0 = _mm256_loadu_pd(C + i0 + (size_t)ldc * j0),
                    c1 = _mm256_loadu_pd(C + i0 + 4 + (size_t)ldc * j0);
            for (int l = 0; l < kb; l++) {
                const double *ap = Apack + l * 8;
                __m256d b0 = _mm256_set1_pd(bj[l]);
                c0 = _mm256_fnmadd_pd(_mm256_load_pd(ap), b0, c0);
                c1 = _mm256_fnmadd_pd(_mm256_load_pd(ap + 4), b0, c1);
            }
            _mm256_storeu_pd(C + i0 + (size_t)ldc * j0, c0);
            _mm256_storeu_pd(C + i0 + 4 + (size_t)ldc * j0, c1);
        }
    } else {
        for (int j = i0; j < rem; j++) {
            const double *pj = panel + (size_t)ldp * j;
            double *cj = C + (size_t)ldc * j;
            int iend = (j < i0 + imax - 1) ? j : (i0 + imax - 1);
            for (int ii = i0; ii <= iend; ii++) {
                const double *pi = panel + (size_t)ldp * ii;
                double s = 0;
                for (int l = 0; l < kb; l++)
                    s += pi[l] * pj[l];
                cj[ii] -= s;
            }
        }
    }
}

static void FHL_K(simd_syrk_upper_panel)(int kb, int rem, const double *panel, int ldp,
                                         double *C, int ldc, int nth) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nth) if (rem >= 64)
#else
    (void)nth;
#endif
    {
        double Apack[96 * 8] __attribute__((aligned(64)));
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int i0 = 0; i0 < rem; i0 += 8) {
            int imax = (rem - i0 < 8) ? (rem - i0) : 8;
            FHL_K(syrk_upper_blk8_leaf)(kb, rem, i0, imax, panel, ldp, C, ldc, Apack);
        }
    }
}

static FHL_KATTR int FHL_K(portable_potrf_upper)(int r, double *U, int nth) {
    int nb = (r >= 400) ? 96 : 48;
    for (int k = 0; k < r; k += nb) {
        int kb = (r - k < nb) ? (r - k) : nb;
        for (int j = 0; j < kb; j++) {
            double *uj = U + k + (size_t)r * (k + j);
            for (int i = 0; i < j; i++) {
                const double *ui = U + k + (size_t)r * (k + i);
                double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                for (int l = 0; l < i; l++)
                    s += ui[l] * uj[l];
                uj[i] = (uj[i] - s) / ui[i];
            }
            double ds = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : ds)
#endif
            for (int l = 0; l < j; l++)
                ds += uj[l] * uj[l];
            double d = uj[j] - ds;
            if (!(d > 0))
                return k + j + 1;
            uj[j] = sqrt(d);
        }
        int rem = r - (k + kb);
        if (rem > 0) {
            double *panel = U + k + (size_t)r * (k + kb);
            FHL_K(simd_trsm_UT_left)(kb, rem, U + k + (size_t)r * k, r, panel, r, nth);
            FHL_K(simd_syrk_upper_panel)(kb, rem, panel, r, U + (k + kb) + (size_t)r * (k + kb), r,
                                         nth);
        }
    }
    return 0;
}

static FHL_KATTR void FHL_K(potri_trtri_panel_leaf)(int r, int p_id, const double *U,
                                                    double *Rpack, const double *inv_diag) {
    int i0 = p_id * 8;
    int imax = (r - i0 < 8) ? (r - i0) : 8;
    double *rp = Rpack + (size_t)p_id * r * 8;
    for (int j = i0; j < i0 + imax; j++) {
        double *dst = rp + (size_t)j * 8;
        for (int ii = 0; ii < 8; ii++)
            dst[ii] = 0.0;
    }
    for (int ii = 0; ii < imax; ii++) {
        int row = i0 + ii;
        rp[(size_t)row * 8 + ii] = inv_diag[row];
        for (int j = row + 1; j < i0 + imax; j++) {
            const double *uj = U + (size_t)r * j;
            double s = 0;
            for (int l = row; l < j; l++)
                s += rp[(size_t)l * 8 + ii] * uj[l];
            rp[(size_t)j * 8 + ii] = -s * inv_diag[j];
        }
    }
    for (int j = i0 + imax; j < r; j++) {
        const double *uj = U + (size_t)r * j;
        __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
        for (int l = i0; l < j; l++) {
            const double *rl = rp + (size_t)l * 8;
            __m256d u = _mm256_set1_pd(uj[l]);
            s0 = _mm256_fmadd_pd(_mm256_loadu_pd(rl), u, s0);
            s1 = _mm256_fmadd_pd(_mm256_loadu_pd(rl + 4), u, s1);
        }
        __m256d id = _mm256_set1_pd(-inv_diag[j]);
        _mm256_storeu_pd(rp + (size_t)j * 8, _mm256_mul_pd(s0, id));
        _mm256_storeu_pd(rp + (size_t)j * 8 + 4, _mm256_mul_pd(s1, id));
    }
}

static FHL_KATTR void FHL_K(potri_lauum_panel_leaf)(int r, int n_panels, int p_i,
                                                    const double *Rpack, double *inv) {
    int i0 = p_i * 8;
    int imax = (r - i0 < 8) ? (r - i0) : 8;
    const double *rpi = Rpack + (size_t)p_i * r * 8;
    for (int p_j = p_i; p_j < n_panels; p_j++) {
        int j0 = p_j * 8;
        int jmax = (r - j0 < 8) ? (r - j0) : 8;
        const double *rpj = Rpack + (size_t)p_j * r * 8;
        if (imax == 8 && jmax == 8) {
            for (int half = 0; half < 8; half += 4) {
                int jb = j0 + half;
                __m256d c00 = _mm256_setzero_pd(), c10 = _mm256_setzero_pd();
                __m256d c01 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
                __m256d c02 = _mm256_setzero_pd(), c12 = _mm256_setzero_pd();
                __m256d c03 = _mm256_setzero_pd(), c13 = _mm256_setzero_pd();
                for (int l = jb; l < r; l++) {
                    const double *ai = rpi + (size_t)l * 8;
                    const double *bj = rpj + (size_t)l * 8 + half;
                    __m256d av0 = _mm256_loadu_pd(ai), av1 = _mm256_loadu_pd(ai + 4);
                    __m256d b0 = _mm256_set1_pd(bj[0]), b1 = _mm256_set1_pd(bj[1]);
                    __m256d b2 = _mm256_set1_pd(bj[2]), b3 = _mm256_set1_pd(bj[3]);
                    c00 = _mm256_fmadd_pd(av0, b0, c00);
                    c10 = _mm256_fmadd_pd(av1, b0, c10);
                    c01 = _mm256_fmadd_pd(av0, b1, c01);
                    c11 = _mm256_fmadd_pd(av1, b1, c11);
                    c02 = _mm256_fmadd_pd(av0, b2, c02);
                    c12 = _mm256_fmadd_pd(av1, b2, c12);
                    c03 = _mm256_fmadd_pd(av0, b3, c03);
                    c13 = _mm256_fmadd_pd(av1, b3, c13);
                }
                _mm256_storeu_pd(inv + i0 + (size_t)r * (jb + 0), c00);
                _mm256_storeu_pd(inv + i0 + 4 + (size_t)r * (jb + 0), c10);
                _mm256_storeu_pd(inv + i0 + (size_t)r * (jb + 1), c01);
                _mm256_storeu_pd(inv + i0 + 4 + (size_t)r * (jb + 1), c11);
                _mm256_storeu_pd(inv + i0 + (size_t)r * (jb + 2), c02);
                _mm256_storeu_pd(inv + i0 + 4 + (size_t)r * (jb + 2), c12);
                _mm256_storeu_pd(inv + i0 + (size_t)r * (jb + 3), c03);
                _mm256_storeu_pd(inv + i0 + 4 + (size_t)r * (jb + 3), c13);
            }
        } else {
            for (int jj = 0; jj < jmax; jj++) {
                int j = j0 + jj;
                double *cj = inv + (size_t)r * j;
                for (int ii = 0; ii < imax; ii++) {
                    int i = i0 + ii;
                    if (i <= j) {
                        double s = 0;
                        for (int l = j; l < r; l++)
                            s += rpi[(size_t)l * 8 + ii] * rpj[(size_t)l * 8 + jj];
                        cj[i] = s;
                    }
                }
            }
        }
    }
}

static void FHL_K(portable_potri_from_U)(int r, const double *U, double *inv,
                                         double *Rpack, double *inv_diag, int nth) {
    for (int j = 0; j < r; j++)
        inv_diag[j] = 1.0 / U[j + (size_t)r * j];
    int n_panels = (r + 7) / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#else
    (void)nth;
#endif
    for (int p_id = 0; p_id < n_panels; p_id++) {
        FHL_K(potri_trtri_panel_leaf)(r, p_id, U, Rpack, inv_diag);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#endif
    for (int p_i = 0; p_i < n_panels; p_i++) {
        FHL_K(potri_lauum_panel_leaf)(r, n_panels, p_i, Rpack, inv);
    }
}

static FHL_KATTR void FHL_K(symm_left_mc_leaf)(int r, int im, int kc, int mc,
                                               const double *A, const double *B, double *C,
                                               double *Apack) {
    int m_len = (r - im < mc) ? (r - im) : mc;
    for (int l0 = 0; l0 < r; l0 += kc) {
        int lmax = (r - l0 < kc) ? (r - l0) : kc;
        int p_idx = 0;
        for (int ii = 0; ii + 7 < m_len; ii += 8, p_idx++) {
            int i0 = im + ii;
            const double *a0 = A + l0 + (size_t)r * (i0 + 0),
                         *a1 = A + l0 + (size_t)r * (i0 + 1);
            const double *a2 = A + l0 + (size_t)r * (i0 + 2),
                         *a3 = A + l0 + (size_t)r * (i0 + 3);
            const double *a4 = A + l0 + (size_t)r * (i0 + 4),
                         *a5 = A + l0 + (size_t)r * (i0 + 5);
            const double *a6 = A + l0 + (size_t)r * (i0 + 6),
                         *a7 = A + l0 + (size_t)r * (i0 + 7);
            double *ap_base = Apack + (size_t)p_idx * (8 * lmax);
            for (int l = 0; l < lmax; l++) {
                double *p = ap_base + (size_t)l * 8;
                p[0] = a0[l];
                p[1] = a1[l];
                p[2] = a2[l];
                p[3] = a3[l];
                p[4] = a4[l];
                p[5] = a5[l];
                p[6] = a6[l];
                p[7] = a7[l];
            }
        }
        int j0 = 0;
        for (; j0 + 3 < r; j0 += 4) {
            const double *bj0 = B + l0 + (size_t)r * (j0 + 0),
                         *bj1 = B + l0 + (size_t)r * (j0 + 1);
            const double *bj2 = B + l0 + (size_t)r * (j0 + 2),
                         *bj3 = B + l0 + (size_t)r * (j0 + 3);
            int ii = 0, pid = 0;
            for (; ii + 7 < m_len; ii += 8, pid++) {
                int i0 = im + ii;
                const double *ap_base = Apack + (size_t)pid * (8 * lmax);
                __m256d c00, c10, c01, c11, c02, c12, c03, c13;
                if (l0 == 0) {
                    c00 = _mm256_setzero_pd();
                    c10 = _mm256_setzero_pd();
                    c01 = _mm256_setzero_pd();
                    c11 = _mm256_setzero_pd();
                    c02 = _mm256_setzero_pd();
                    c12 = _mm256_setzero_pd();
                    c03 = _mm256_setzero_pd();
                    c13 = _mm256_setzero_pd();
                } else {
                    c00 = _mm256_loadu_pd(C + i0 + (size_t)r * (j0 + 0));
                    c10 = _mm256_loadu_pd(C + i0 + 4 + (size_t)r * (j0 + 0));
                    c01 = _mm256_loadu_pd(C + i0 + (size_t)r * (j0 + 1));
                    c11 = _mm256_loadu_pd(C + i0 + 4 + (size_t)r * (j0 + 1));
                    c02 = _mm256_loadu_pd(C + i0 + (size_t)r * (j0 + 2));
                    c12 = _mm256_loadu_pd(C + i0 + 4 + (size_t)r * (j0 + 2));
                    c03 = _mm256_loadu_pd(C + i0 + (size_t)r * (j0 + 3));
                    c13 = _mm256_loadu_pd(C + i0 + 4 + (size_t)r * (j0 + 3));
                }
                for (int l = 0; l < lmax; l++) {
                    const double *ap = ap_base + (size_t)l * 8;
                    __m256d av0 = _mm256_load_pd(ap), av1 = _mm256_load_pd(ap + 4);
                    __m256d b0 = _mm256_set1_pd(bj0[l]), b1 = _mm256_set1_pd(bj1[l]);
                    __m256d b2 = _mm256_set1_pd(bj2[l]), b3 = _mm256_set1_pd(bj3[l]);
                    c00 = _mm256_fmadd_pd(av0, b0, c00);
                    c10 = _mm256_fmadd_pd(av1, b0, c10);
                    c01 = _mm256_fmadd_pd(av0, b1, c01);
                    c11 = _mm256_fmadd_pd(av1, b1, c11);
                    c02 = _mm256_fmadd_pd(av0, b2, c02);
                    c12 = _mm256_fmadd_pd(av1, b2, c12);
                    c03 = _mm256_fmadd_pd(av0, b3, c03);
                    c13 = _mm256_fmadd_pd(av1, b3, c13);
                }
                _mm256_storeu_pd(C + i0 + (size_t)r * (j0 + 0), c00);
                _mm256_storeu_pd(C + i0 + 4 + (size_t)r * (j0 + 0), c10);
                _mm256_storeu_pd(C + i0 + (size_t)r * (j0 + 1), c01);
                _mm256_storeu_pd(C + i0 + 4 + (size_t)r * (j0 + 1), c11);
                _mm256_storeu_pd(C + i0 + (size_t)r * (j0 + 2), c02);
                _mm256_storeu_pd(C + i0 + 4 + (size_t)r * (j0 + 2), c12);
                _mm256_storeu_pd(C + i0 + (size_t)r * (j0 + 3), c03);
                _mm256_storeu_pd(C + i0 + 4 + (size_t)r * (j0 + 3), c13);
            }
            for (; ii < m_len; ii++) {
                int i = im + ii;
                const double *ai = A + l0 + (size_t)r * i;
                double s0 = (l0 == 0) ? 0 : C[i + (size_t)r * (j0 + 0)],
                       s1 = (l0 == 0) ? 0 : C[i + (size_t)r * (j0 + 1)];
                double s2 = (l0 == 0) ? 0 : C[i + (size_t)r * (j0 + 2)],
                       s3 = (l0 == 0) ? 0 : C[i + (size_t)r * (j0 + 3)];
                for (int l = 0; l < lmax; l++) {
                    double a = ai[l];
                    s0 += a * bj0[l];
                    s1 += a * bj1[l];
                    s2 += a * bj2[l];
                    s3 += a * bj3[l];
                }
                C[i + (size_t)r * (j0 + 0)] = s0;
                C[i + (size_t)r * (j0 + 1)] = s1;
                C[i + (size_t)r * (j0 + 2)] = s2;
                C[i + (size_t)r * (j0 + 3)] = s3;
            }
        }
        for (; j0 < r; j0++) {
            const double *bj = B + l0 + (size_t)r * j0;
            for (int ii = 0; ii < m_len; ii++) {
                int i = im + ii;
                const double *ai = A + l0 + (size_t)r * i;
                double s = (l0 == 0) ? 0 : C[i + (size_t)r * j0];
                for (int l = 0; l < lmax; l++)
                    s += ai[l] * bj[l];
                C[i + (size_t)r * j0] = s;
            }
        }
    }
}

static void FHL_K(portable_symm_left)(int r, const double *A, const double *B, double *C,
                                      int nth) {
    int kc = 256, mc = 32;
#ifdef _OPENMP
#pragma omp parallel num_threads(nth)
#else
    (void)nth;
#endif
    {
        double Apack[32 * 256] __attribute__((aligned(64)));
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int im = 0; im < r; im += mc) {
            FHL_K(symm_left_mc_leaf)(r, im, kc, mc, A, B, C, Apack);
        }
    }
}

#else

static FHL_KATTR void FHL_K(simd_trsm_UT_left)(int kb, int rem, const double *U, int ldu,
                                               double *panel, int ldp, int nth) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nth) if (rem >= 64)
#else
    (void)nth;
#endif
    for (int j0 = 0; j0 < rem; j0 += 4) {
        int jmax = (rem - j0 < 4) ? (rem - j0) : 4;
        if (jmax == 4) {
            double *p0 = panel + (size_t)ldp * (j0 + 0), *p1 = panel + (size_t)ldp * (j0 + 1);
            double *p2 = panel + (size_t)ldp * (j0 + 2), *p3 = panel + (size_t)ldp * (j0 + 3);
            for (int i = 0; i < kb; i++) {
                const double *ui = U + (size_t)ldu * i;
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s0, s1, s2, s3)
#endif
                for (int l = 0; l < i; l++) {
                    double u = ui[l];
                    s0 += u * p0[l];
                    s1 += u * p1[l];
                    s2 += u * p2[l];
                    s3 += u * p3[l];
                }
                double inv_d = 1.0 / ui[i];
                p0[i] = (p0[i] - s0) * inv_d;
                p1[i] = (p1[i] - s1) * inv_d;
                p2[i] = (p2[i] - s2) * inv_d;
                p3[i] = (p3[i] - s3) * inv_d;
            }
        } else {
            for (int jj = 0; jj < jmax; jj++) {
                double *pj = panel + (size_t)ldp * (j0 + jj);
                for (int i = 0; i < kb; i++) {
                    const double *ui = U + (size_t)ldu * i;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = 0; l < i; l++)
                        s += ui[l] * pj[l];
                    pj[i] = (pj[i] - s) / ui[i];
                }
            }
        }
    }
}

static FHL_KATTR void FHL_K(simd_syrk_upper_panel)(int kb, int rem, const double *panel, int ldp,
                                                   double *C, int ldc, int nth) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth) if (rem >= 64)
#else
    (void)nth;
#endif
    for (int j0 = 0; j0 < rem; j0 += 4) {
        int jmax = (rem - j0 < 4) ? (rem - j0) : 4;
        if (jmax == 4) {
            const double *pj0 = panel + (size_t)ldp * (j0 + 0),
                         *pj1 = panel + (size_t)ldp * (j0 + 1);
            const double *pj2 = panel + (size_t)ldp * (j0 + 2),
                         *pj3 = panel + (size_t)ldp * (j0 + 3);
            double *cj0 = C + (size_t)ldc * (j0 + 0), *cj1 = C + (size_t)ldc * (j0 + 1);
            double *cj2 = C + (size_t)ldc * (j0 + 2), *cj3 = C + (size_t)ldc * (j0 + 3);
            int i = 0;
            for (; i + 1 <= j0; i += 2) {
                const double *pi0 = panel + (size_t)ldp * (i + 0),
                             *pi1 = panel + (size_t)ldp * (i + 1);
                double s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s00, s01, s02, s03, s10, s11, s12, s13)
#endif
                for (int l = 0; l < kb; l++) {
                    double a0 = pi0[l], a1 = pi1[l];
                    double b0 = pj0[l], b1 = pj1[l], b2 = pj2[l], b3 = pj3[l];
                    s00 += a0 * b0;
                    s01 += a0 * b1;
                    s02 += a0 * b2;
                    s03 += a0 * b3;
                    s10 += a1 * b0;
                    s11 += a1 * b1;
                    s12 += a1 * b2;
                    s13 += a1 * b3;
                }
                cj0[i + 0] -= s00;
                cj1[i + 0] -= s01;
                cj2[i + 0] -= s02;
                cj3[i + 0] -= s03;
                cj0[i + 1] -= s10;
                cj1[i + 1] -= s11;
                cj2[i + 1] -= s12;
                cj3[i + 1] -= s13;
            }
            for (int jj = 0; jj < 4; jj++) {
                int j = j0 + jj;
                const double *pj = panel + (size_t)ldp * j;
                double *cj = C + (size_t)ldc * j;
                for (int ii = i; ii <= j; ii++) {
                    const double *pi = panel + (size_t)ldp * ii;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = 0; l < kb; l++)
                        s += pi[l] * pj[l];
                    cj[ii] -= s;
                }
            }
        } else {
            for (int jj = 0; jj < jmax; jj++) {
                int j = j0 + jj;
                const double *pj = panel + (size_t)ldp * j;
                double *cj = C + (size_t)ldc * j;
                for (int ii = 0; ii <= j; ii++) {
                    const double *pi = panel + (size_t)ldp * ii;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = 0; l < kb; l++)
                        s += pi[l] * pj[l];
                    cj[ii] -= s;
                }
            }
        }
    }
}

static FHL_KATTR int FHL_K(portable_potrf_upper)(int r, double *U, int nth) {
    int nb = (r >= 400) ? 96 : 48;
    for (int k = 0; k < r; k += nb) {
        int kb = (r - k < nb) ? (r - k) : nb;
        for (int j = 0; j < kb; j++) {
            double *uj = U + k + (size_t)r * (k + j);
            for (int i = 0; i < j; i++) {
                const double *ui = U + k + (size_t)r * (k + i);
                double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                for (int l = 0; l < i; l++)
                    s += ui[l] * uj[l];
                uj[i] = (uj[i] - s) / ui[i];
            }
            double ds = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : ds)
#endif
            for (int l = 0; l < j; l++)
                ds += uj[l] * uj[l];
            double d = uj[j] - ds;
            if (!(d > 0))
                return k + j + 1;
            uj[j] = sqrt(d);
        }
        int rem = r - (k + kb);
        if (rem > 0) {
            double *panel = U + k + (size_t)r * (k + kb);
            FHL_K(simd_trsm_UT_left)(kb, rem, U + k + (size_t)r * k, r, panel, r, nth);
            FHL_K(simd_syrk_upper_panel)(kb, rem, panel, r, U + (k + kb) + (size_t)r * (k + kb), r,
                                         nth);
        }
    }
    return 0;
}

static FHL_KATTR void FHL_K(portable_potri_from_U)(int r, const double *U, double *inv,
                                                   double *Rinv_T, double *inv_diag, int nth) {
    for (int j = 0; j < r; j++)
        inv_diag[j] = 1.0 / U[j + (size_t)r * j];
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#else
    (void)nth;
#endif
    for (int i0 = 0; i0 < r; i0 += 4) {
        int imax = (r - i0 < 4) ? (r - i0) : 4;
        if (imax == 4) {
            double *r0 = Rinv_T + (size_t)r * (i0 + 0), *r1 = Rinv_T + (size_t)r * (i0 + 1);
            double *r2 = Rinv_T + (size_t)r * (i0 + 2), *r3 = Rinv_T + (size_t)r * (i0 + 3);
            for (int jj = 0; jj < 4; jj++) {
                double *rj = Rinv_T + (size_t)r * (i0 + jj);
                int row = i0 + jj;
                rj[row] = inv_diag[row];
                for (int j = row + 1; j < i0 + 4; j++) {
                    const double *uj = U + (size_t)r * j;
                    double s = 0;
                    for (int l = row; l < j; l++)
                        s += rj[l] * uj[l];
                    rj[j] = -s * inv_diag[j];
                }
            }
            for (int j = i0 + 4; j < r; j++) {
                const double *uj = U + (size_t)r * j;
                double s0 = r0[i0] * uj[i0] + r0[i0 + 1] * uj[i0 + 1] + r0[i0 + 2] * uj[i0 + 2];
                double s1 = r1[i0 + 1] * uj[i0 + 1] + r1[i0 + 2] * uj[i0 + 2];
                double s2 = r2[i0 + 2] * uj[i0 + 2];
                double s3 = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s0, s1, s2, s3)
#endif
                for (int l = i0 + 3; l < j; l++) {
                    double u = uj[l];
                    s0 += r0[l] * u;
                    s1 += r1[l] * u;
                    s2 += r2[l] * u;
                    s3 += r3[l] * u;
                }
                double id = -inv_diag[j];
                r0[j] = s0 * id;
                r1[j] = s1 * id;
                r2[j] = s2 * id;
                r3[j] = s3 * id;
            }
        } else {
            for (int ii = 0; ii < imax; ii++) {
                int i = i0 + ii;
                double *ri = Rinv_T + (size_t)r * i;
                ri[i] = inv_diag[i];
                for (int j = i + 1; j < r; j++) {
                    const double *uj = U + (size_t)r * j;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = i; l < j; l++)
                        s += ri[l] * uj[l];
                    ri[j] = -s * inv_diag[j];
                }
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#endif
    for (int j0 = 0; j0 < r; j0 += 4) {
        int jmax = (r - j0 < 4) ? (r - j0) : 4;
        if (jmax == 4) {
            const double *rj0 = Rinv_T + (size_t)r * (j0 + 0), *rj1 = Rinv_T + (size_t)r * (j0 + 1);
            const double *rj2 = Rinv_T + (size_t)r * (j0 + 2), *rj3 = Rinv_T + (size_t)r * (j0 + 3);
            double *cj0 = inv + (size_t)r * (j0 + 0), *cj1 = inv + (size_t)r * (j0 + 1);
            double *cj2 = inv + (size_t)r * (j0 + 2), *cj3 = inv + (size_t)r * (j0 + 3);
            int i = 0;
            for (; i + 1 <= j0; i += 2) {
                const double *ri0 = Rinv_T + (size_t)r * (i + 0),
                             *ri1 = Rinv_T + (size_t)r * (i + 1);
                double s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
                int l_start = j0 + 3;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s00, s01, s02, s03, s10, s11, s12, s13)
#endif
                for (int l = l_start; l < r; l++) {
                    double a0 = ri0[l], a1 = ri1[l];
                    double b0 = rj0[l], b1 = rj1[l], b2 = rj2[l], b3 = rj3[l];
                    s00 += a0 * b0;
                    s01 += a0 * b1;
                    s02 += a0 * b2;
                    s03 += a0 * b3;
                    s10 += a1 * b0;
                    s11 += a1 * b1;
                    s12 += a1 * b2;
                    s13 += a1 * b3;
                }
                for (int l = j0; l < l_start; l++) {
                    double a0 = ri0[l], a1 = ri1[l];
                    if (l >= j0 + 0) {
                        double b0 = rj0[l];
                        s00 += a0 * b0;
                        s10 += a1 * b0;
                    }
                    if (l >= j0 + 1) {
                        double b1 = rj1[l];
                        s01 += a0 * b1;
                        s11 += a1 * b1;
                    }
                    if (l >= j0 + 2) {
                        double b2 = rj2[l];
                        s02 += a0 * b2;
                        s12 += a1 * b2;
                    }
                }
                cj0[i + 0] = s00;
                cj1[i + 0] = s01;
                cj2[i + 0] = s02;
                cj3[i + 0] = s03;
                cj0[i + 1] = s10;
                cj1[i + 1] = s11;
                cj2[i + 1] = s12;
                cj3[i + 1] = s13;
            }
            for (int jj = 0; jj < 4; jj++) {
                int j = j0 + jj;
                const double *rj = Rinv_T + (size_t)r * j;
                double *cj = inv + (size_t)r * j;
                for (int ii = i; ii <= j; ii++) {
                    const double *ri = Rinv_T + (size_t)r * ii;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = j; l < r; l++)
                        s += ri[l] * rj[l];
                    cj[ii] = s;
                }
            }
        } else {
            for (int jj = 0; jj < jmax; jj++) {
                int j = j0 + jj;
                const double *rj = Rinv_T + (size_t)r * j;
                double *cj = inv + (size_t)r * j;
                for (int ii = 0; ii <= j; ii++) {
                    const double *ri = Rinv_T + (size_t)r * ii;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = j; l < r; l++)
                        s += ri[l] * rj[l];
                    cj[ii] = s;
                }
            }
        }
    }
}

static FHL_KATTR void FHL_K(portable_symm_left)(int r, const double *A, const double *B, double *C,
                                                int nth) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
#else
    (void)nth;
#endif
    for (int j0 = 0; j0 < r; j0 += 4) {
        int jmax = (r - j0 < 4) ? (r - j0) : 4;
        if (jmax == 4) {
            const double *bj0 = B + (size_t)r * (j0 + 0), *bj1 = B + (size_t)r * (j0 + 1);
            const double *bj2 = B + (size_t)r * (j0 + 2), *bj3 = B + (size_t)r * (j0 + 3);
            double *cj0 = C + (size_t)r * (j0 + 0), *cj1 = C + (size_t)r * (j0 + 1);
            double *cj2 = C + (size_t)r * (j0 + 2), *cj3 = C + (size_t)r * (j0 + 3);
            int ii = 0;
            for (; ii + 1 < r; ii += 2) {
                const double *ai0 = A + (size_t)r * (ii + 0), *ai1 = A + (size_t)r * (ii + 1);
                double s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s00, s01, s02, s03, s10, s11, s12, s13)
#endif
                for (int l = 0; l < r; l++) {
                    double a0 = ai0[l], a1 = ai1[l];
                    double b0 = bj0[l], b1 = bj1[l], b2 = bj2[l], b3 = bj3[l];
                    s00 += a0 * b0;
                    s01 += a0 * b1;
                    s02 += a0 * b2;
                    s03 += a0 * b3;
                    s10 += a1 * b0;
                    s11 += a1 * b1;
                    s12 += a1 * b2;
                    s13 += a1 * b3;
                }
                cj0[ii + 0] = s00;
                cj1[ii + 0] = s01;
                cj2[ii + 0] = s02;
                cj3[ii + 0] = s03;
                cj0[ii + 1] = s10;
                cj1[ii + 1] = s11;
                cj2[ii + 1] = s12;
                cj3[ii + 1] = s13;
            }
            for (; ii < r; ii++) {
                const double *ai = A + (size_t)r * ii;
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s0, s1, s2, s3)
#endif
                for (int l = 0; l < r; l++) {
                    double a = ai[l];
                    s0 += a * bj0[l];
                    s1 += a * bj1[l];
                    s2 += a * bj2[l];
                    s3 += a * bj3[l];
                }
                cj0[ii] = s0;
                cj1[ii] = s1;
                cj2[ii] = s2;
                cj3[ii] = s3;
            }
        } else {
            for (int jj = 0; jj < jmax; jj++) {
                int j = j0 + jj;
                const double *bj = B + (size_t)r * j;
                double *cj = C + (size_t)r * j;
                for (int ii = 0; ii < r; ii++) {
                    const double *ai = A + (size_t)r * ii;
                    double s = 0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : s)
#endif
                    for (int l = 0; l < r; l++)
                        s += ai[l] * bj[l];
                    cj[ii] = s;
                }
            }
        }
    }
}

#endif
