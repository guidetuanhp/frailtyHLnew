/* Native .Call routine table and DLL registration for frailtyHLnew. */
#include <R_ext/Rdynload.h>
#include "fhl_core.h"
#include "fhl_native_api.h"

static const R_CallMethodDef fhl_calls[] = {
    /* Cox Frailty Models */
    {"fhl_fit_cox_gamma",       (DL_FUNC)&fhl_fit_cox_gamma,       17},
    {"fhl_fit_cox_normal",      (DL_FUNC)&fhl_fit_cox_normal,      17},
    {"fhl_fit_cox_ar1",         (DL_FUNC)&fhl_fit_cox_ar1,         17},

    /* Competing Risks */
    {"fhl_fit_competing_risks", (DL_FUNC)&fhl_fit_competing_risks, 15},

    /* Mixed-Effects AFT Models */
    {"fhl_fit_aft_1re",         (DL_FUNC)&fhl_fit_aft_1re,          8},
    {"fhl_fit_aft_2re",         (DL_FUNC)&fhl_fit_aft_2re,         10},

    /* Joint Models */
    {"fhl_fit_jm_cox1",         (DL_FUNC)&fhl_fit_jm_cox1,         12},
    {"fhl_fit_jm_cox2",         (DL_FUNC)&fhl_fit_jm_cox2,         19},
    {"fhl_fit_jm_frailty",      (DL_FUNC)&fhl_fit_jm_frailty,      16},

    /* Variable Selection */
    {"fhl_fit_vs_1re",          (DL_FUNC)&fhl_fit_vs_1re,          12},
    {"fhl_fit_vs_2re",          (DL_FUNC)&fhl_fit_vs_2re,          11},

    {NULL, NULL, 0}
};

/**
 * Initialize BLAS/OpenMP runtime, register the .Call ABI, and disable dynamic symbol lookup.
 * @param dll DLL descriptor supplied by R during package loading.
 */
void R_init_frailtyHLnew(DllInfo *dll) {
    fhl_blas_init_runtime();
    R_registerRoutines(dll, NULL, fhl_calls, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
