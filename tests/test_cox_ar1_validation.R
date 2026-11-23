# Cox Gamma small-variance KKT, AR(1) row/cluster invariance, and input validation tests
suppressPackageStartupMessages({
  library(frailtyHLnew)
  library(survival)
  library(cmprsk)
})

data(kidney, package = "frailtyHLnew")
data(rats, package = "frailtyHLnew")
data(ren, package = "frailtyHLnew")
data(renal, package = "frailtyHLnew")
data(test, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")

# 1. Cox Gamma small & large varinit invariance + KKT boundary handling
for (vi in c(0, 1e-8, 1e-6, 1e-5, 1e-4, 0.163, 50)) {
  capture.output(fit <- frailtyHL(
    Surv(time, status) ~ sex + age + (1 | id), kidney,
    RandDist = "Gamma", mord = 0, dord = 1, varinit = vi, Maxiter = 200
  ))
  stopifnot(
    identical(unname(fit$convergence), "converged"),
    abs(as.numeric(fit$RandCoef[1L, 1L]) - 0.472842) < 1e-4,
    abs(fit$likelihood[1L, "-2*p_b,v(hp)"] - 370.8078) < 1e-3,
    abs(fit$FixCoef[1L, 1L] - (-1.62439)) < 1e-4
  )
}

# Continuity of -2*p_b,v(hp) across alpha = 1e-5
capture.output(f_99 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
  RandDist = "Gamma", mord = 1, dord = 1, varfixed = TRUE, varinit = 9.99e-6))
capture.output(f_10 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
  RandDist = "Gamma", mord = 1, dord = 1, varfixed = TRUE, varinit = 1e-5))
stopifnot(
  abs(f_99$likelihood[1L, "-2*p_b,v(hp)"] - 377.44) < 0.05,
  abs(f_10$likelihood[1L, "-2*p_b,v(hp)"] - 377.44) < 0.05,
  abs(f_99$likelihood[1L, "-2*p_v(hp)"] - f_10$likelihood[1L, "-2*p_v(hp)"]) < 0.05
)

# Boundary KKT convergence on zero-variance synthetic Gamma data
set.seed(1)
df_z <- data.frame(
  time = rexp(80, rate = exp(0.3 * (x1 <- rnorm(80)))),
  status = rbinom(80, 1, 0.7), x1 = x1, id = factor(rep(1:20, each = 4))
)
capture.output(fz0 <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Gamma", varinit = 0))
capture.output(fz1 <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Gamma", varinit = 0.1))
stopifnot(
  identical(unname(fz0$convergence), "converged"),
  identical(unname(fz1$convergence), "converged"),
  as.numeric(fz0$RandCoef[1L, 1L]) == 0,
  as.numeric(fz1$RandCoef[1L, 1L]) == 0,
  is.finite(as.numeric(fz0$RandCoef[1L, 2L]))
)

# 2. Cox AR(1) row-permutation, cluster-relabeling, interleaving & LTRC invariance
capture.output(f_ar0 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 1, Maxiter = 500))
rats_u <- unique(ren$rat)
idx_rev <- unlist(lapply(rev(rats_u), function(r) which(ren$rat == r)))
ren_rel <- ren[idx_rev, ]
ren_rel$rat <- factor(ren_rel$rat, levels = rev(rats_u))
capture.output(f_ar_rev <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren_rel, RandDist = "AR1", mord = 0, dord = 1, Maxiter = 500))

by_r <- split(seq_len(nrow(ren)), factor(ren$rat, levels = rats_u))
idx_int <- na.omit(unlist(lapply(seq_len(max(lengths(by_r))), function(k) vapply(by_r, `[`, integer(1L), k))))
capture.output(f_ar_int <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren[idx_int, ], RandDist = "AR1", mord = 0, dord = 1, Maxiter = 500))

stopifnot(
  identical(unname(f_ar0$convergence), "converged"),
  abs(f_ar0$rho - 0.811372) < 1e-4,
  abs(f_ar_rev$rho - f_ar0$rho) < 1e-10,
  abs(f_ar_int$rho - f_ar0$rho) < 1e-10,
  abs(f_ar_rev$FixCoef[1L, 1L] - f_ar0$FixCoef[1L, 1L]) < 1e-10,
  abs(f_ar_int$FixCoef[1L, 1L] - f_ar0$FixCoef[1L, 1L]) < 1e-10,
  abs(as.numeric(f_ar_int$RandCoef[1L, 1L]) - as.numeric(f_ar0$RandCoef[1L, 1L])) < 1e-10
)

# LTRC AR(1) invariance
set.seed(101)
rows_l <- lapply(1:20, function(i) {
  mi <- sample(2:4, 1); xi <- rnorm(1); g <- rexp(mi, rate = exp(0.2 * xi))
  ts <- c(0, cumsum(g[-mi])) + runif(1, 0, 4)
  data.frame(id = factor(i), t1 = ts, t2 = ts + g, status = rbinom(mi, 1, 0.75), x = xi + rnorm(mi, 0, 0.1))
})
df_ltrc <- do.call(rbind, rows_l)
capture.output(fl1 <- frailtyHL(Surv(t1, t2, status) ~ x + (1 | id), df_ltrc, RandDist = "AR1"))
capture.output(fl2 <- frailtyHL(Surv(t1, t2, status) ~ x + (1 | id), df_ltrc[order(df_ltrc$t1), ], RandDist = "AR1"))
stopifnot(abs(fl1$rho - fl2$rho) < 1e-12, abs(fl1$FixCoef[1L, 1L] - fl2$FixCoef[1L, 1L]) < 1e-12)

# 3. Non-convergence warning when terminated early at Maxiter
w_d0 <- FALSE
capture.output(withCallingHandlers(
  f_d0 <- frailtyHL(Surv(time, status) ~ rx + (1 | litter), rats, RandDist = "Gamma", mord = 0, dord = 0, Maxiter = 5),
  warning = function(w) { w_d0 <<- TRUE; invokeRestart("muffleWarning") }
))
stopifnot(w_d0, identical(unname(f_d0$convergence), "did not converge"))

# 4. Multi-RE Gamma Hinv_u finiteness and dimensions
capture.output(f_mre <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id), cgd,
  RandDist = "Gamma", varfixed = TRUE, varinit = c(0.1, 0.5), Maxiter = 5))
stopifnot(all(is.finite(f_mre$Hinv_u)), identical(dim(f_mre$Hinv_u), dim(f_mre$Hinv)))

# 5. Input validation, convergence reporting & S3 dispatch
assert_err <- function(expr, pattern = "") {
  ok <- tryCatch({ expr; FALSE }, error = function(e) grepl(pattern, conditionMessage(e), ignore.case = TRUE))
  stopifnot(ok)
}

# AFT q = 1 rejection
renal1 <- transform(subset(renal, first == 1), id = 1L)
jm_q1a <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(sur_time, status) ~ sex + age + (1 | id), RandDist = "gaussian")
jm_q1b <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(sur_time, status) ~ age + (sex | id), RandDist = "gaussian")
assert_err(mlmfit(jm_q1a, renal1, Maxiter = 5), "at least 2 levels")
assert_err(mlmfit(jm_q1b, renal1, Maxiter = 5), "at least 2 levels")

# AR(1) singletons only
df_sing <- kidney[match(unique(kidney$id), kidney$id), ]
assert_err(frailtyHL(Surv(time, status) ~ sex + (1 | id), df_sing, RandDist = "AR1"), "repeated observations")

# Unsupported arguments rejected cleanly (frailtyHL weights warns without error)
w_wt <- FALSE
capture.output(withCallingHandlers(
  frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney, weights = rep(2, nrow(kidney)), varfixed = TRUE, varinit = 0.2),
  warning = function(w) { if (grepl("weights", conditionMessage(w))) w_wt <<- TRUE; invokeRestart("muffleWarning") }
))
stopifnot(w_wt)
assert_err(frailtyHL(Surv(time, status) ~ sex + age + offset(rep(0.1, nrow(kidney))) + (1 | id), kidney), "offset")
assert_err(mlmfit(jm_q1a, subset(renal, first == 1), weights = rep(1, 5)), "weights")
assert_err(mlmfit(jm_q1a, subset(renal, first == 1), subset = sex == 1), "subset")
jm_off <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(sur_time, status) ~ sex + (1 | id), RandDist = "gaussian", Offset = rep(0, 10))
assert_err(mlmfit(jm_off, subset(renal, first == 1)), "Offset")

# Domain & non-finite value checks
df_neg <- kidney; df_neg$time[1L] <- -1
assert_err(frailtyHL(Surv(time, status) ~ sex + (1 | id), df_neg), "Negative survival")
df_ltrc_bad <- kidney; df_ltrc_bad$t0 <- df_ltrc_bad$time
assert_err(suppressWarnings(frailtyHL(Surv(t0, time, status) ~ sex + (1 | id), df_ltrc_bad, na.action = na.pass)), "Entry times must be finite")
assert_err(frailtyHLnew:::fhl_validate_surv_data(kidney$time, kidney$status, matrix(kidney$sex, ncol = 1), L0 = kidney$time), "strictly less than exit")
df_nan <- kidney; df_nan$age[1L] <- NaN
assert_err(frailtyHL(Surv(time, status) ~ sex + age + (1 | id), df_nan, na.action = na.pass), "Non-finite")
df_aft0 <- subset(renal, first == 1); df_aft0$sur_time[1L] <- 0
assert_err(mlmfit(jm_q1a, df_aft0), "strictly positive")

# frailty.vs zero-beta division safety + LTRC rejection + S3 methods
assert_err(frailty.vs(Surv(tstart, tstop, status) ~ treat + (1 | id), model = "lognorm", penalty = "lasso", data = cgd, tun1 = 0.05), "right-censored")
capture.output(f_lasso0 <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
  model = "lognorm", penalty = "lasso", data = kidney, B = c(0, -1.6, 0, 0, 0), tun1 = 0.25))
stopifnot(all(is.finite(f_lasso0$beta)), all(is.finite(f_lasso0$se)), is.finite(f_lasso0$BIC))

capture.output(f_cr <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), test,
  inits = list(beta = matrix(0, 2, 2), theta = matrix(0.5, 1, 1), v = matrix(0, length(unique(test$id)), 1)),
  order = 1, frailty.cov = "none", MAX.ITER = 200))
stopifnot(inherits(summary(f_cr), "hlike") || is.list(f_cr))
capture.output({
  print(f_cr)
  summary(f_cr)
})
capture.output(sm_cox <- summary(f_ar0))
stopifnot(inherits(f_ar0, "frailtyHL"), inherits(sm_cox, "summary.frailtyHL"), inherits(summary.frailtyHL(f_ar0), "summary.frailtyHL"))

cat("COX AR1 AND INPUT VALIDATION TESTS PASS\n")
