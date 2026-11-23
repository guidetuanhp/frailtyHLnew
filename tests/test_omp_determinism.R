# Strip-parallel Gram kernel & OpenMP thread-count bitwise determinism tests
library(frailtyHLnew)
library(survival)

data(kidney, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")

# 1. Nested & LTRC (general risk-set) HL(1,1) / HL(1,2) / Multi-RE convergence and finiteness
invisible(capture.output({
  fg_k <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), data = kidney,
                    RandDist = "Gamma", mord = 1, dord = 2)
  fn_k <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), data = kidney,
                    RandDist = "Normal", mord = 1, dord = 2)
  fg_ltrc <- frailtyHL(Surv(tstart, tstop, status) ~ treat + (1 | center) + (1 | id), data = cgd,
                       RandDist = "Gamma", mord = 1, dord = 1, Maxiter = 100)
  fn_ltrc <- frailtyHL(Surv(tstart, tstop, status) ~ treat + (1 | center) + (1 | id), data = cgd,
                       RandDist = "Normal", mord = 1, dord = 1, Maxiter = 100)
}))
stopifnot(
  all(is.finite(fg_k$FixCoef)), all(is.finite(fg_k$RandCoef)),
  all(is.finite(fn_k$FixCoef)), all(is.finite(fn_k$RandCoef)),
  all(is.finite(fg_ltrc$FixCoef)), all(is.finite(fg_ltrc$RandCoef)),
  all(is.finite(fn_ltrc$FixCoef)), all(is.finite(fn_ltrc$RandCoef))
)

# 2. Bitwise OMP_NUM_THREADS=1 vs OMP_NUM_THREADS=2 determinism across k >= 128 strip-parallel kernels
rscript <- file.path(R.home("bin"), "Rscript")
lib_vec <- paste0("c(", paste(shQuote(.libPaths()), collapse = ","), ")")
run_omp_sub <- function(nth) {
  scr <- tempfile("fhl_omp_det_", fileext = ".R")
  rds <- tempfile("fhl_omp_det_", fileext = ".rds")
  on.exit(unlink(c(scr, rds), force = TRUE), add = TRUE)
  writeLines(c(
    sprintf("Sys.setenv(OMP_NUM_THREADS = '%d')", nth),
    sprintf(".libPaths(%s)", lib_vec),
    "library(survival); library(frailtyHLnew)",
    "set.seed(2024L); q <- 140L; N <- q * 5L",
    "df <- data.frame(y = rexp(N, 0.25), d = rbinom(N, 1, 0.75), x1 = rnorm(N), x2 = rnorm(N), id = factor(rep(seq_len(q), each = 5L)))",
    "invisible(capture.output({",
    "  fg <- frailtyHL(Surv(y, d) ~ x1 + x2 + (1 | id), data = df, RandDist = 'Gamma', mord = 1, dord = 2)",
    "  fn <- frailtyHL(Surv(y, d) ~ x1 + x2 + (1 | id), data = df, RandDist = 'Normal', mord = 1, dord = 2)",
    "}))",
    sprintf("saveRDS(list(g = c(fg$FixCoef, fg$RandCoef, fg$aic, fg$v_h), gi = fg$iter, n = c(fn$FixCoef, fn$RandCoef, fn$aic, fn$v_h), ni = fn$iter), %s)",
            shQuote(normalizePath(rds, winslash = "/", mustWork = FALSE)))
  ), scr)
  st <- system2(rscript, shQuote(scr), stdout = TRUE, stderr = TRUE)
  if (!is.null(attr(st, "status")) && attr(st, "status") != 0L) stop("OMP subprocess failed")
  readRDS(rds)
}
r1 <- run_omp_sub(1L)
r2 <- run_omp_sub(2L)
stopifnot(
  identical(r1$gi, r2$gi),
  identical(r1$ni, r2$ni),
  max(abs(r1$g - r2$g)) == 0.0,
  max(abs(r1$n - r2$n)) == 0.0
)

cat("OPENMP BITWISE DETERMINISM TESTS PASS\n")
