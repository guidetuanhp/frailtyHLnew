# Strict SPD verification, post-fit state synchronization, and grid-isolation tests
library(frailtyHLnew)
library(survival)

assert_err <- function(expr, pattern = NULL) {
  ok <- tryCatch({
    suppressWarnings(invisible(capture.output(force(expr))))
    FALSE
  }, error = function(e) {
    if (!is.null(pattern) && !grepl(pattern, conditionMessage(e))) {
      stop(sprintf("Expected error matching '%s', got: %s", pattern, conditionMessage(e)))
    }
    TRUE
  })
  if (!isTRUE(ok)) stop("Expected error was not raised")
}

data(kidney, package = "frailtyHLnew")
data(ren, package = "frailtyHLnew")
data(test, package = "frailtyHLnew")
data(bladder, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")

# 1. Cox Gamma small-alpha convergence & Cholesky/QR stability
invisible(capture.output(
  f_cgd_gam <- frailtyHL(Surv(tstop - tstart, status) ~ treat + sex + (1 | center),
                         cgd, RandDist = "Gamma", mord = 1, dord = 1,
                         varinit = 1e-4, Maxiter = 100)
))
stopifnot(all(is.finite(f_cgd_gam$FixCoef)), is.finite(f_cgd_gam$RandCoef[1, 1]), f_cgd_gam$RandCoef[1, 1] > 0)

# 2. AR(1) likelihood continuity across alpha = 1e-5 boundary
invisible(capture.output({
  f_ar_hi <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren,
                       RandDist = "AR1", varfixed = TRUE, varinit = 1.01e-5, Maxiter = 200)
  f_ar_lo <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren,
                       RandDist = "AR1", varfixed = TRUE, varinit = 0.99e-5, Maxiter = 200)
}))
stopifnot(abs(unname(f_ar_hi$likelihood[1, "-2*p_b,v(hp)"]) -
                unname(f_ar_lo$likelihood[1, "-2*p_b,v(hp)"])) < 0.05)

# 3. Cox Normal/Gamma Maxiter = 1 post-fit likelihood synchronization
invisible(capture.output({
  fn1 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                   RandDist = "Normal", mord = 0, dord = 1, varfixed = FALSE, varinit = 0.8, Maxiter = 1)
  fg1 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                   RandDist = "Gamma", mord = 0, dord = 1, varfixed = FALSE, varinit = 0.8, Maxiter = 1)
}))
a1_n <- unname(fn1$RandCoef[1, 1])
ora_n <- unname(fn1$likelihood[1, "-2h0"]) + length(fn1$v_h) * log(2 * pi * a1_n) + sum(fn1$v_h^2) / a1_n
a1_g <- unname(fg1$RandCoef[1, 1]); gam <- 1 / a1_g
ora_g <- unname(fg1$likelihood[1, "-2h0"]) - 2 * sum(gam * (fg1$v_h - fg1$u_h) - gam * log(a1_g) - lgamma(gam))
stopifnot(abs(unname(fn1$likelihood[1, "-2*hp"]) - ora_n) < 1e-11,
          abs(unname(fg1$likelihood[1, "-2*hp"]) - ora_g) < 1e-11)

# 4. Competing risks synchronized output + strict SPD & ascent stops
q_t <- length(unique(test$id))
b0_t <- c(sapply(1:2, function(k) survival::coxph(Surv(time, status == k) ~ x1 + x2, data = test)$coef))
invisible(capture.output(
  fit_cr5 <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), test,
                           inits = list(beta = b0_t, theta = matrix(c(1, 0.5, 0.5, 1), 2, 2), v = matrix(0, q_t, 2)),
                           order = 1, frailty.cov = "unstructured", MAX.ITER = 5, TOL = 1e-5)
))
stopifnot(all(is.finite(fit_cr5$beta$Estimate)), all(is.finite(fit_cr5$theta$Estimate)), is.finite(fit_cr5$cAIC))
q_b <- length(unique(bladder$center))
th_bk_b <- matrix(c(0.07356526, 0.07632592, 0.07632592, 0.09516952), 2, 2)
b_bk_b  <- matrix(c(-0.6674387, -0.1484962, 0.1199023, 0.6865932), 2, 2)
assert_err(
  hlike.frailty(CmpRsk(surtime, status) ~ CHEMO + AGE + cluster(center), bladder,
                list(beta = b_bk_b, theta = th_bk_b, v = matrix(0, q_b, 2)),
                order = 1, frailty.cov = "unstructured", MAX.ITER = 30, TOL = 1e-6),
  "not positive definite"
)
b0_b <- c(sapply(1:2, function(k) survival::coxph(Surv(surtime, status == k) ~ CHEMO + AGE, data = bladder)$coef))
assert_err(
  hlike.frailty(CmpRsk(surtime, status) ~ CHEMO + AGE + cluster(center), bladder,
                inits = list(beta = b0_b, theta = matrix(c(1, 0.5, 0.5, 1), 2, 2), v = matrix(0, q_b, 2)),
                order = 1, frailty.cov = "unstructured", MAX.ITER = 500, TOL = 1e-8),
  "dispersion Newton step not ascent"
)

# 5. AFT zero-fixed-effect guard, integer interaction factor, 2-RE covariance & degenerate boundary stop
assert_err(mlmfit(jointmodeling("mean", "AFT", "log", Surv(time, status) ~ -1 + (1 | id), "gaussian"), kidney, Maxiter = 5))
df_col <- data.frame(
  time = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
  status = c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0),
  x = c(-1, 0.5, 1.2, -0.3, 0.8, -0.7, 1.1, 0.2, -0.9, 0.4),
  g1 = c(1L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L),
  g2 = c(4L, 1L, 2L, 3L, 4L, 1L, 2L, 3L, 4L, 5L)
)
invisible(capture.output(
  fit_col <- mlmfit(jointmodeling("mean", "AFT", "log", Surv(time, status) ~ x + (1 | g1:g2), "gaussian"), df_col, Maxiter = 5)
))
stopifnot(all(is.finite(fit_col$F.Est)), all(is.finite(fit_col$D.Est)))
df_aft_2re <- data.frame(
  time = kidney$time, status = kidney$status,
  age = kidney$age, sex = kidney$sex,
  id = factor(kidney$id),
  g2 = factor(((seq_len(nrow(kidney)) - 1L) %% 4L) + 1L)
)
invisible(capture.output(
  fit_aft2 <- mlmfit(jointmodeling("mean", "AFT", "log", Surv(time, status) ~ sex + age + (1 | id) + (1 | g2), "gaussian"),
                     df_aft_2re, Maxiter = 20)
))
stopifnot(all(is.finite(fit_aft2$D.Est)), all(fit_aft2$D.Est[, 1L] > 0))
df_deg_aft <- data.frame(
  time = c(1, 1, 1, 1, 1, 1), status = c(1, 0, 0, 0, 0, 0),
  x = c(1, 1, 1, 1, 1, 1), id = factor(c(1, 1, 2, 2, 3, 3))
)
assert_err(mlmfit(jointmodeling("mean", "AFT", "log", Surv(time, status) ~ x + (1 | id), "gaussian"), df_deg_aft, Maxiter = 10))

# 6. Joint modeling cluster alignment, validation, Maxiter >= 1 & strict PD
assert_err(
  jointmodeling(Model = "mean", RespDist = "normal", Link = "identity",
                LinPred = y ~ -1 + (1 | id), RandDist = "gaussian")
)
df_jm_long <- data.frame(
  id = factor(rep(1:8, each = 2)),
  x = rep(c(-1, 1, -0.5, 0.5, -1.2, 0.8, 0.3, -0.3), each = 2),
  y = rep(0, 16)
)
df_jm_surv <- data.frame(
  id = factor(1:8),
  time = c(2, 4, 6, 8, 3, 5, 7, 9),
  status = rep(1, 8),
  x = c(-1, 1, -0.5, 0.5, -1.2, 0.8, 0.3, -0.3)
)
jm_m <- jointmodeling("mean", "gaussian", "identity", y ~ x + (1 | id), "gaussian")
jm_s <- jointmodeling("mean", "FM", "log", Surv(time, status) ~ x + (1 | id), "gaussian")
assert_err(jmfit(list(jm_m, jm_s), list(df_jm_long, df_jm_surv), Maxiter = 0))
assert_err(
  jmfit(list(jm_m, jm_s), list(df_jm_long, df_jm_surv), Maxiter = 10),
  "not positive definite|non-positive|degenerate|diverged"
)

# 7. Explicit no-intercept (-1) formula handling & initial-value guards
invisible(capture.output({
  fhl_std <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney, RandDist = "Normal", Maxiter = 50)
  fhl_m1  <- frailtyHL(Surv(time, status) ~ -1 + sex + age + (1 | id), kidney, RandDist = "Normal", Maxiter = 50)
  vs_std  <- frailty.vs(Surv(time, status) ~ sex + age + (1 | id), model = "lognorm", penalty = "lasso", data = kidney, tun1 = 0.03)
  vs_m1   <- frailty.vs(Surv(time, status) ~ -1 + sex + age + (1 | id), model = "lognorm", penalty = "lasso", data = kidney, tun1 = 0.03)
}))
stopifnot(
  max(abs(fhl_std$FixCoef[, 1L] - fhl_m1$FixCoef[, 1L])) <= 1e-8,
  max(abs(vs_std$beta - vs_m1$beta)) <= 1e-8
)
assert_err(frailty.vs(Surv(time, status) ~ sex + age + (1 | id), model = "lognorm", penalty = "lasso",
                      data = kidney, B = c(NA, 0), tun1 = 0.03))
assert_err(frailty.vs(Surv(time, status) ~ sex + age + (1 | id), model = "lognorm", penalty = "lasso",
                      data = kidney, v = numeric(length(unique(kidney$id))), tun1 = 0.03))

# 8. Correlated 2-RE grouping alignment, SPD S0 & alpha routing
set.seed(2028)
n2 <- 100L; q2 <- 20L
x1_2 <- rnorm(n2); grp2 <- factor(((seq_len(n2) - 1L) %% q2) + 1L)
unrel2 <- factor(((seq_len(n2) * 7L) %% q2) + 1L, labels = paste0("U", seq_len(q2)))
df_2re <- data.frame(time = rexp(n2, 0.2), status = rep(1L, n2), x1 = x1_2, grp = grp2, unrelated_g2 = unrel2)
assert_err(
  frailty.vs(Surv(time, status) ~ x1 + (1 | grp) + (1 | unrelated_g2),
             model = "lognorm", penalty = "HL", data = df_2re,
             B = 0.0, alpha = c(0.1, 0.1, 0.05), tun1 = 0.2, tun2 = 0.05),
  "same grouping"
)
assert_err(
  frailty.vs(Surv(time, status) ~ x1 + (1 | grp) + (x1 | grp),
             model = "lognorm", penalty = "HL", data = df_2re,
             B = 0.0, alpha = c(0.1, 0.1, 0.105), tun1 = 0.2, tun2 = 0.05),
  "positive definite"
)
assert_err(
  frailty.vs(Surv(time, status) ~ x1 + (1 | grp),
             model = "lognorm", penalty = "HL", data = df_2re,
             B = 0.0, alpha = c(0.1, 0.02), tun1 = 0.2, tun2 = 0.05)
)

# 9. HL tau = 2 limit at B = 0 & multi-point grid failure isolation vs all-fail stop
kidney2 <- kidney
kidney2$age <- (kidney2$age - mean(kidney2$age)) / sd(kidney2$age)
kidney2$GN <- as.numeric(kidney2$disease == "GN")
kidney2$AN <- as.numeric(kidney2$disease == "AN")
kidney2$PKD <- as.numeric(kidney2$disease == "PKD")
invisible(capture.output(
  fit_tau2 <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
                         model = "lognorm", penalty = "HL", data = kidney2,
                         B = rep(0, 5), tun1 = 2.0, tun2 = 0.1)
))
stopifnot(is.finite(fit_tau2$BIC), all(is.finite(fit_tau2$beta)), all(is.finite(fit_tau2$se)))

k_col <- kidney2; k_col$x_zero <- 0.0
grid_warns <- character(0)
withCallingHandlers(
  invisible(capture.output(
    fit_part <- frailty.vs(Surv(time, status) ~ age + x_zero + sex + (1 | id),
                           model = "lognorm", penalty = "lasso", data = k_col,
                           B = c(0, 0, -1), tun1 = c(0, 0.05))
  )),
  warning = function(w) { grid_warns <<- c(grid_warns, conditionMessage(w)); invokeRestart("muffleWarning") }
)
stopifnot(abs(fit_part$tuning_parameter - 0.05) < 1e-12, length(grid_warns) >= 1L)
df_q2 <- data.frame(time = 1:8, status = c(1, 0, 1, 0, 0, 1, 0, 1),
                    x1 = c(-2, -1, 0, 1, -1, 0, 1, 2), id = factor(rep(1:2, each = 4)))
assert_err(
  frailty.vs(Surv(time, status) ~ x1 + (1 | id), model = "lognorm", penalty = "lasso",
             data = df_q2, B = 0.1, v = list(c(0, 0)), alpha = 50.0, tun1 = c(0.01, 0.02))
)

cat("SPD, SYNCHRONIZATION, AND GRID ISOLATION TESTS PASS\n")
