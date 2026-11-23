/* Portable OpenMP + generic/AVX2 blocked Cholesky (dpotrf), triangular inversion
 * (dpotri), and symmetric left multiplication (dsymm) kernels shared by
 * fhl_linalg.c and fhl_competing_risks.c.
 */
#include "fhl_core.h"
#include <stdint.h>

static int detect_physical_cores(void) {
#ifdef _OPENMP
    int limit = 0;
    const char *env_lim = getenv("OMP_THREAD_LIMIT");
    if (env_lim != NULL && env_lim[0] != '\0') {
        limit = atoi(env_lim);
    }
    const char *env_omp = getenv("OMP_NUM_THREADS");
    if (env_omp != NULL && env_omp[0] != '\0') {
        int m = omp_get_max_threads();
        if (limit >= 1 && m > limit)
            m = limit;
        return (m >= 1) ? m : 1;
    }
    int n = omp_get_num_procs();
    int def = (n >= 2) ? 2 : 1;
    int max_t = omp_get_max_threads();
    if (max_t >= 1 && def > max_t)
        def = max_t;
    if (limit >= 1 && def > limit)
        def = limit;
    return (def >= 1) ? def : 1;
#else
    return 1;
#endif
}

/* ============================================================================
   Portable C + OpenMP + SIMD Linear Algebra Kernels for R's Built-in BLAS Mode
   Parallelizes Cholesky factorization, triangular inversion, and symmetric
   matrix multiplication across OpenMP threads with 4x4 register blocking.
   ============================================================================ */

#define FHL_K(n) n##_generic
#define FHL_KATTR
#include "fhl_kernels.h"
#undef FHL_K
#undef FHL_KATTR

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__amd64__)) && !defined(FHL_NO_AVX2)
#define FHL_HAVE_AVX2_CLONE 1
#define FHL_USE_AVX2_INTRINSICS 1
#define FHL_K(n) n##_avx2
#define FHL_KATTR __attribute__((target("avx2,fma")))
#include "fhl_kernels.h"
#undef FHL_K
#undef FHL_KATTR
#undef FHL_USE_AVX2_INTRINSICS
#endif

/* Run-time dispatch: AVX2/FMA clone when the CPU supports it, generic otherwise. */
potrf_kernel_fn fhl_portable_potrf_upper = portable_potrf_upper_generic;
potri_kernel_fn fhl_portable_potri_from_U = portable_potri_from_U_generic;
symm_kernel_fn fhl_portable_symm_left = portable_symm_left_generic;
int fhl_cpu_has_avx2 = 0;

void fhl_blas_init_runtime(void) {
#ifdef FHL_HAVE_AVX2_CLONE
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        fhl_portable_potrf_upper = portable_potrf_upper_avx2;
        fhl_portable_potri_from_U = portable_potri_from_U_avx2;
        fhl_portable_symm_left = portable_symm_left_avx2;
        fhl_cpu_has_avx2 = 1;
    }
#endif
}

int fhl_blas_detect_threads(int n) {
    if (n < 64) return 1;
    return detect_physical_cores();
}

/* Single-pass strict SPD symmetric matrix inverse via Cholesky factorization */
int fhl_blas_inverse_c(int n, const double *A, double *out) {
    if (n < 1 || !A || !out) return 0;
    if (n == 1) {
        if (!R_FINITE(A[0]) || A[0] <= 0.0) return 0;
        out[0] = 1.0 / A[0];
        return R_FINITE(out[0]) ? 1 : 0;
    }
    for (int j = 0; j < n; j++) {
        double ajj = A[j + (size_t)n * j];
        if (!R_FINITE(ajj) || ajj <= 0.0) return 0;
        for (int i = j + 1; i < n; i++) {
            double aij = A[i + (size_t)n * j];
            double aji = A[j + (size_t)n * i];
            if (!R_FINITE(aij) || !R_FINITE(aji)) return 0;
            double scale = 1.0;
            if (fabs(aij) > scale) scale = fabs(aij);
            if (fabs(aji) > scale) scale = fabs(aji);
            if (fabs(aij - aji) > 1e-10 * scale) return 0;
        }
    }
    if (n >= 64) {
        int nth = (n >= 160) ? detect_physical_cores() : 1;
        size_t sn = (size_t)n;
        if (sn > SIZE_MAX - 7) return 0;
        size_t np8 = (sn + 7) / 8 * 8;
        if (sn > SIZE_MAX / sizeof(double) ||
            sn > (SIZE_MAX / sizeof(double)) / sn ||
            np8 > (SIZE_MAX / sizeof(double)) / sn) {
            return 0;
        }
        double *U = (double *)malloc(sn * sn * sizeof(double));
        double *Rinv_T = (double *)malloc(np8 * sn * sizeof(double));
        double *inv_diag = (double *)malloc(sn * sizeof(double));
        if (!U || !Rinv_T || !inv_diag) {
            free(U); free(Rinv_T); free(inv_diag);
            return 0;
        }
        memcpy(U, A, sn * sn * sizeof(double));
        if (fhl_portable_potrf_upper(n, U, nth) != 0) {
            free(U); free(Rinv_T); free(inv_diag);
            return 0;
        }
        for (int j = 0; j < n; j++) {
            double ujj = U[j + sn * (size_t)j];
            if (!R_FINITE(ujj) || ujj <= 0.0) {
                free(U); free(Rinv_T); free(inv_diag);
                return 0;
            }
        }
        fhl_portable_potri_from_U(n, U, out, Rinv_T, inv_diag, nth);
        free(U); free(Rinv_T); free(inv_diag);
    } else {
        if (out != A) {
            memcpy(out, A, (size_t)n * (size_t)n * sizeof(double));
        }
        int info = 0;
        const char *uplo = "U";
        F77_CALL(dpotrf)(uplo, &n, out, &n, &info FCONE);
        if (info != 0) return 0;
        for (int j = 0; j < n; j++) {
            double ujj = out[j + (size_t)n * (size_t)j];
            if (!R_FINITE(ujj) || ujj <= 0.0) return 0;
        }
        F77_CALL(dpotri)(uplo, &n, out, &n, &info FCONE);
        if (info != 0) return 0;
    }
    for (int j = 0; j < n; j++) {
        if (!R_FINITE(out[j + (size_t)n * j])) return 0;
        for (int i = j + 1; i < n; i++) {
            double v = out[j + (size_t)n * i];
            if (!R_FINITE(v)) return 0;
            out[i + (size_t)n * j] = v;
        }
    }
    return 1;
}
