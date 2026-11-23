# frailtyHLnew architecture and mathematical specification

This document summarizes the R-to-C architecture and mathematical formulations
implemented in `frailtyHLnew`. Each estimator validates inputs and builds model
frames once in R before invoking a dedicated C `.Call` entrypoint.

## Statistical quantities

For observation `i`, the survival data are `(T_i, delta_i, x_i, g_i)` (or
`(L_i, T_i, delta_i, x_i, g_i)` for left-truncated data), with `delta_i = 1` for
an observed event and `delta_i = 0` for right censoring.

Let `eta_i = x_i' beta + z_i' v`, `w_i = exp(eta_i)`, and let `t_k` denote
distinct observed nonzero event times with `d_k` events at `t_k`. Define:

    R_k = sum_{i: L_i < t_k <= T_i} w_i,
    Delta Lambda_0(t_k) = d_k / R_k,
    ell_profile = sum_i delta_i eta_i - sum_k d_k log(R_k).

For Gamma frailty (`u_j = exp(v_j)`), log-normal frailty (`v_j ~ N(0, alpha)`),
AR(1) frailty (`v | rho ~ N(0, alpha * A_rho)`), competing risks
(`hlike.frailty`), multilevel AFT (`mlmfit` with `HL(0,1)` and `HL(1,1)`), joint
models (`jmfit`), and penalized Cox variable selection (`frailty.vs`), R
prepares the sparse random-effect design and compressed risk-set descriptors and
dispatches to the corresponding native C solver.

## Linear algebra specification

* **Strict SPD Cholesky solvers:** Symmetric `(beta, v)` h-likelihood Hessians
  `H`, random-effect blocks `H_zz`, Schur complements `S`, penalized `(beta, v)`
  systems, and unclipped `fhl_c_logdet_spd` evaluations enforce strict positive
  definiteness via single-pass Cholesky (`fhl_kernels.h` blocked AVX2+FMA/OpenMP
  for `n >= 64`, LAPACK `dpotrf`/`dpotri`/`dpotrs` for `n < 64`).
* **Dedicated non-SPD solvers:** General LU (`dgesv`) or bounded SVD (`dgesvd`)
  routines are invoked only where the target operator is mathematically
  non-symmetric or indefinite: bounded SVD when explicit singular-value bounds
  are requested in `AR1`, `fhl_c_solve_gesv` for the asymmetric fixed-effect /
  dispersion system in Gamma `dord = 0`, and `fhl_c_solve_gesv_inv` for the
  negative-definite `3x3` dispersion Hessian in correlated 2-RE variable
  selection (guarded by a Cholesky SPD check on `S0`), post-fit `u`-scale `Hu`
  in Gamma frailty, and multi-RE post-fit dispersion observed information at
  zero-variance boundaries.

## Native modules (`src/`, 11 registered `.Call` entrypoints)

| Module | Responsibility & Exported `.Call` Routines |
|---|---|
| `fhl_init.c` / `fhl_native_api.h` | Initializes BLAS/OpenMP runtime (`fhl_blas_init_runtime()`) and registers the 11 `.Call` entrypoints with `R_useDynamicSymbols(dll, FALSE)` |
| `fhl_core.h` / `fhl_blas.c` / `fhl_kernels.h` | Platform headers, runtime SIMD kernel selection, OpenMP thread helpers, and single-pass blocked Cholesky/inverse/symm kernels |
| `fhl_linalg.h` / `fhl_linalg.c` | Compressed `O(n+m)` risk-set operators (`fhl_c_risk_forward`, `fhl_c_risk_backward`), strip-parallel sparse `[X, Z]` CSR/CSC Gram & direct trace kernels (`fhl_c_xz_gram0`/`1`/`2`, `fhl_c_xz_gram1_trace_batch`), strict SPD Cholesky & Schur-complement solvers (`fhl_c_solve_spd`, `fhl_c_solve_inv`, `fhl_c_lognormal_inverse`, `fhl_c_schur_solve_step`), and UCRT `exp`/`log` |
| `fhl_cox_gamma.c` | Gamma Cox frailty (1-RE, Multi-RE, LTRC, `p >= 0`): `fhl_fit_cox_gamma` |
| `fhl_cox_normal.c` | Log-normal Cox frailty (1-RE, Multi-RE, LTRC, `p >= 0`): `fhl_fit_cox_normal` |
| `fhl_cox_ar1.c` | AR(1) Cox frailty with `O(n)` tridiagonal precision: `fhl_fit_cox_ar1` |
| `fhl_competing_risks.c` | Competing risks (`CmpRsk` / `hlike.frailty`): `fhl_fit_competing_risks` |
| `fhl_aft.c` | Multilevel AFT mixed models (`mlmfit`, `HL(0,1)` & `HL(1,1)`): `fhl_fit_aft_1re`, `fhl_fit_aft_2re` |
| `fhl_joint_model.c` | Joint longitudinal-survival & bivariate frailty (`jmfit`): `fhl_fit_jm_cox1`, `fhl_fit_jm_cox2`, `fhl_fit_jm_frailty` |
| `fhl_var_select.c` | Penalized Cox variable selection (`frailty.vs`, native grid search): `fhl_fit_vs_1re`, `fhl_fit_vs_2re` |

## R modules (`R/`, 11 files)

* `R/frailtyHL.R`: Public `frailtyHL()`, `summary.frailtyHL()`, `fhl_validate_frailtyhl_args()`, `fhl_prepare_cox_data()`, `fhl_format_cox_result()`, `.onUnload`.
* `R/frailtyHL_frames.R`: `FrailtyFrames()`, `fhl_build_risk_set()`, `fhl_make_data()`, `design_frailtyHL()`, `HGLMFrames()`, `HGLMFactorList()`, `design_hglm()`, and mixed-model random-effect formula parsers.
* `R/cox_gamma.R`, `R/cox_normal.R`, `R/cox_ar1.R`: Native runners for Gamma, Normal, and AR(1) Cox frailty models.
* `R/competing_risks.R`: `CmpRsk()`, `hlike.frailty()`, `print.hlike()`, `summary.hlike()`.
* `R/aft_mixed.R`: `mlmfit()`, `mlmfit1()`, `mlmfit2()`, `fhl_mlm_prepare()`, and output formatter.
* `R/joint_model.R`: `jointmodeling()`, `jmfit()`, `jmfit2()`, `jmfit3()`, `jmfit_FM()`.
* `R/var_select.R`, `R/var_select_1re.R`, `R/var_select_2re.R`: `frailty.vs()`, `frailty1.vs()`, `frailty2.vs()`, validators, and summary printers.
