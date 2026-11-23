#ifndef FHL_CORE_H
#define FHL_CORE_H

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Include <omp.h> BEFORE any R headers so R macros (e.g. match -> Rf_match)
 * can never corrupt LLVM/Clang OpenMP 5.0 context-selector declarations. */
#ifdef _OPENMP
#if defined(__clang__) || defined(__GNUC__)
#pragma push_macro("match")
#undef match
#include <omp.h>
#pragma pop_macro("match")
#else
#include <omp.h>
#endif
#else
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_procs(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif
#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#ifndef R_NO_REMAP_RMATH
#define R_NO_REMAP_RMATH
#endif
#ifndef STRICT_R_HEADERS
#define STRICT_R_HEADERS
#endif

#include <Rconfig.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#ifndef FCONE
#define FCONE
#endif
#include <R_ext/Utils.h>

static inline size_t fhl_checked_mul_size(size_t a, size_t b, const char *ctx) {
    if (a != 0 && b > SIZE_MAX / a)
        Rf_error("size_t multiplication overflow in %s", ctx);
    return a * b;
}

static inline size_t fhl_checked_mul3_size(size_t a, size_t b, size_t c, const char *ctx) {
    return fhl_checked_mul_size(fhl_checked_mul_size(a, b, ctx), c, ctx);
}

/* Core OpenMP / SIMD Cholesky & Symmetric Multiply Dispatch (src/fhl_blas.c) */
typedef int (*potrf_kernel_fn)(int, double *, int);
typedef void (*potri_kernel_fn)(int, const double *, double *, double *, double *, int);
typedef void (*symm_kernel_fn)(int, const double *, const double *, double *, int);

extern potrf_kernel_fn fhl_portable_potrf_upper;
extern potri_kernel_fn fhl_portable_potri_from_U;
extern symm_kernel_fn fhl_portable_symm_left;
extern int fhl_cpu_has_avx2;

void fhl_blas_init_runtime(void);
int fhl_blas_detect_threads(int n);

/* Strict SPD matrix inverse (AVX2/OpenMP blocked Cholesky for n >= 64, LAPACK dpotrf+dpotri for n < 64) */
int fhl_blas_inverse_c(int n, const double *A, double *out);

#endif /* FHL_CORE_H */
