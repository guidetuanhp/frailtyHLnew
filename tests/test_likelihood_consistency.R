# Likelihood continuity, multi-RE invariance, competing risks, and joint model consistency tests
library(survival)
library(cmprsk)
library(frailtyHLnew)

data(kidney, package = "frailtyHLnew")
data(rats, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")
data(ren, package = "frailtyHLnew")
data(renal, package = "frailtyHLnew")
data(test, package = "frailtyHLnew")
data(bladder0, package = "frailtyHLnew")

# ==============================================================================
# 1. Gamma & Normal varnonneg = TRUE, boundary KKT, and small-alpha continuity
# ==============================================================================
invisible(capture.output({
  g01_nn <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                      RandDist = "Gamma", mord = 0, dord = 1,
                      varnonneg = TRUE, convergence = 1e-9, Maxiter = 200)
  g01_nn_tiny <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                           RandDist = "Gamma", mord = 0, dord = 1,
                           varnonneg = TRUE, varinit = 1e-6, convergence = 1e-8)
  n01_nn <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                      RandDist = "Normal", mord = 0, dord = 1,
                      varnonneg = TRUE, convergence = 1e-9, Maxiter = 200)
  g12_k <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                     RandDist = "Gamma", mord = 1, dord = 2,
                     varnonneg = FALSE, convergence = 1e-9, Maxiter = 200)
}))

stopifnot(
  identical(unname(g01_nn$convergence), "converged"),
  abs(g01_nn$RandCoef[1L, 1L] - 0.61693448) < 1e-5,
  abs(g01_nn$RandCoef[1L, 2L] - 0.27623973) < 1e-5,
  abs(1.0 / g01_nn$RandCoef[1L, 2L]^2 - 13.104719) < 1e-4,
  abs(g01_nn_tiny$RandCoef[1L, 1L] - g01_nn$RandCoef[1L, 1L]) < 1e-6,
  identical(unname(n01_nn$convergence), "converged"),
  abs(n01_nn$RandCoef[1L, 1L] - 0.79005695) < 1e-5,
  abs(n01_nn$RandCoef[1L, 2L] - 0.43450854) < 1e-5,
  abs(1.0 / n01_nn$RandCoef[1L, 2L]^2 - 5.296675) < 1e-4,
  identical(unname(g12_k$convergence), "converged"),
  abs(g12_k$RandCoef[1L, 1L] - 0.56945092) < 1e-5,
  abs(g12_k$RandCoef[1L, 2L] - 0.26727186) < 1e-5
)

# Boundary behavior (zero true frailty variance dataset df_z)
set.seed(1)
df_z <- data.frame(
  time = rexp(80, rate = exp(0.3 * (x1_z <- rnorm(80)))),
  status = rbinom(80, 1, 0.7), x1 = x1_z, id = factor(rep(1:20, each = 4))
)
invisible(capture.output({
  fz_g0  <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Gamma",  varinit = 0)
  fz_g1  <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Gamma",  varinit = 0.1)
  fz_gnn <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Gamma",  varinit = 0, varnonneg = TRUE)
  fz_n0  <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Normal", varinit = 0)
  fz_n1  <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Normal", varinit = 0.1)
  fz_nnn <- frailtyHL(Surv(time, status) ~ x1 + (1 | id), df_z, RandDist = "Normal", varinit = 0, varnonneg = TRUE)
}))
stopifnot(
  identical(unname(fz_g0$convergence), "converged"),  fz_g0$RandCoef[1L, 1L] == 0,  is.finite(fz_g0$RandCoef[1L, 2L]),  fz_g0$RandCoef[1L, 2L] > 0,
  identical(unname(fz_g1$convergence), "converged"),  fz_g1$RandCoef[1L, 1L] == 0,  is.finite(fz_g1$RandCoef[1L, 2L]),
  identical(unname(fz_gnn$convergence), "converged"), fz_gnn$RandCoef[1L, 1L] > 0,  is.finite(fz_gnn$RandCoef[1L, 2L]),
  identical(unname(fz_n0$convergence), "converged"),  fz_n0$RandCoef[1L, 1L] == 0,  is.finite(fz_n0$RandCoef[1L, 2L]),  fz_n0$RandCoef[1L, 2L] > 0,
  identical(unname(fz_n1$convergence), "converged"),  fz_n1$RandCoef[1L, 1L] == 0,  is.finite(fz_n1$RandCoef[1L, 2L]),
  identical(unname(fz_nnn$convergence), "converged"), fz_nnn$RandCoef[1L, 1L] > 0,  is.finite(fz_nnn$RandCoef[1L, 2L])
)

# Small-alpha likelihood continuity across 1.59e-6, 1e-5, and 1e-3 + Cox alpha=0 limit
cox_ref <- coxph(Surv(time, status) ~ sex + age, data = kidney, ties = "breslow")
alpha_grid <- c(0, 1e-8, 1e-6, 1.5e-6, 1.6e-6, 9.99e-6, 1e-5, 1.01e-5, 9.99e-4, 1e-3, 1.001e-3)
for (dist in c("Gamma", "Normal")) {
  pbv_vals <- numeric(length(alpha_grid))
  for (i in seq_along(alpha_grid)) {
    invisible(capture.output(
      fi <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
                      RandDist = dist, mord = 1, dord = 2,
                      varfixed = TRUE, varinit = alpha_grid[i],
                      Maxiter = 200, convergence = 1e-10)
    ))
    stopifnot(identical(unname(fi$convergence), "converged"))
    pbv_vals[i] <- fi$likelihood[1L, "-2*p_b,v(hp)"]
    if (alpha_grid[i] == 0) {
      stopifnot(
        identical(fi$Model, "Cox model"),
        max(abs(fi$FixCoef[, 1L] - coef(cox_ref))) < 1e-6
      )
    }
  }
  stopifnot(
    abs(pbv_vals[5L] - pbv_vals[4L]) < 1e-4,
    abs(pbv_vals[7L] - pbv_vals[6L]) < 1e-5,
    abs(pbv_vals[8L] - pbv_vals[7L]) < 1e-5,
    abs(pbv_vals[10L] - pbv_vals[9L]) < 1e-3,
    abs(pbv_vals[11L] - pbv_vals[10L]) < 1e-3
  )
}

# ==============================================================================
# 2. Competing Risks Covariance Structures & Indefinite Matrix Rejection
# ==============================================================================
n_test <- length(unique(test$id))
inits_cr04 <- list(beta = matrix(0, 2, 2), theta = matrix(0.4, 1, 1), v = matrix(0, n_test, 1))
inits_cr2  <- list(beta = matrix(0, 2, 2), theta = matrix(c(0.5, 0.1, 0.1, 0.5), 2, 2), v = matrix(0, n_test, 2))
inits_cr_ind <- list(beta = matrix(0, 2, 2), theta = diag(0.5, 2), v = matrix(0, n_test, 2))
inits_cr_sh  <- list(beta = matrix(0, 2, 1), theta = matrix(0.5, 1, 1), v = matrix(0, n_test, 1))

invisible(capture.output({
  cr_t04  <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr04, order = 1, frailty.cov = "none", MAX.ITER = 100, TOL = 1e-8)
  cr_o0   <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr04, order = 0, frailty.cov = "none", MAX.ITER = 100, TOL = 1e-8)
  cr_o2   <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr04, order = 2, frailty.cov = "none", MAX.ITER = 100, TOL = 1e-8)
  cr_sh   <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr_sh, order = 1, frailty.cov = "none", subHazard = TRUE, MAX.ITER = 100, TOL = 1e-8)
  cr_ind  <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr_ind, order = 1, frailty.cov = "independent", MAX.ITER = 100, TOL = 1e-8)
  cr_ex   <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr2, order = 1, frailty.cov = "exchangeable", MAX.ITER = 100, TOL = 1e-8)
  cr_un   <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id), data = test,
                           inits = inits_cr2, order = 1, frailty.cov = "unstructured", MAX.ITER = 100, TOL = 1e-8)
}))
stopifnot(
  cr_t04$converge == 1L, cr_t04$iterations <= 12L, abs(cr_t04$theta$Estimate[1L] - 0.39999192) < 1e-5,
  cr_o0$converge == 1L, cr_o2$converge == 1L, cr_sh$converge == 1L,
  cr_ind$converge == 1L, cr_ex$converge == 1L, cr_un$converge == 1L,
  max(abs(sort(cr_ex$theta$Estimate) - sort(cr_un$theta$Estimate))) < 1e-5
)

# Indefinite covariance case (bladder0, order=1, th0=0.3): must stop cleanly on non-SPD matrix
b0_pos <- subset(bladder0, Surtime > 0)
n_b0 <- length(unique(b0_pos$Center))
m_b0 <- max(b0_pos$Status)
inits_b0 <- list(beta = matrix(0, 2, m_b0), theta = matrix(0.3, 1, 1), v = matrix(0, n_b0, 1))
err_b0 <- tryCatch({
  suppressWarnings(capture.output(
    hlike.frailty(CmpRsk(Surtime, Status) ~ Chemo + Tustat + cluster(Center), data = b0_pos,
                  inits = inits_b0, order = 1, frailty.cov = "none", MAX.ITER = 30, TOL = 1e-6)
  ))
  "NO_ERROR"
}, error = function(e) conditionMessage(e))
stopifnot(grepl("not positive definite", err_b0, fixed = TRUE))

# ==============================================================================
# 3. Gamma Multi-RE + Permutation Invariance + 3-RE Left-Truncated Data
# ==============================================================================
invisible(capture.output({
  cgd_01 <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                      cgd, RandDist = "Gamma", mord = 0, dord = 1, Maxiter = 200)
  cgd_11 <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                      cgd, RandDist = "Gamma", mord = 1, dord = 1, Maxiter = 200)
}))
stopifnot(
  identical(unname(cgd_01$convergence), "converged"), cgd_01$iter <= 15L,
  max(abs(cgd_01$RandCoef[, 1L] - c(0.125644, 0.237198))) < 1e-5,
  max(abs(cgd_01$RandCoef[, 2L] - c(0.148566, 0.222895))) < 1e-5,
  identical(unname(cgd_11$convergence), "converged"), cgd_11$iter <= 15L,
  max(abs(cgd_11$RandCoef[, 1L] - c(0.126308, 0.237385))) < 1e-5,
  max(abs(cgd_11$RandCoef[, 2L] - c(0.148985, 0.223073))) < 1e-5,
  max(abs(cgd_11$Hinv_u - t(cgd_11$Hinv_u))) < 1e-10
)

# Row/cluster permutation invariance on cgd Multi-RE
set.seed(2025)
cgd_perm <- cgd[sample(nrow(cgd)), ]
cgd_perm$center <- factor(cgd_perm$center, levels = sample(unique(cgd$center)))
cgd_perm$id <- factor(cgd_perm$id, levels = sample(unique(cgd$id)))
invisible(capture.output(
  cgd_11_p <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                        cgd_perm, RandDist = "Gamma", mord = 1, dord = 1, Maxiter = 200)
))
stopifnot(
  max(abs(cgd_11$FixCoef - cgd_11_p$FixCoef)) < 1e-7,
  max(abs(cgd_11$RandCoef - cgd_11_p$RandCoef)) < 1e-7
)

# 3-RE Left-truncated Gamma Multi-RE
set.seed(303)
q3_1 <- 8L; q3_2 <- 20L; q3_3 <- 40L; n3 <- 160L
g3_1 <- factor(((seq_len(n3) - 1L) %% q3_1) + 1L)
g3_2 <- factor(((seq_len(n3) - 1L) %% q3_2) + 1L)
g3_3 <- factor(((seq_len(n3) - 1L) %% q3_3) + 1L)
u3_1 <- log(rgamma(q3_1, shape = 1 / 0.25, scale = 0.25))
u3_2 <- log(rgamma(q3_2, shape = 1 / 0.35, scale = 0.35))
u3_3 <- log(rgamma(q3_3, shape = 1 / 0.45, scale = 0.45))
x3_a <- rnorm(n3); x3_b <- rbinom(n3, 1L, 0.5)
eta3 <- 0.4 * x3_a - 0.5 * x3_b + u3_1[as.integer(g3_1)] + u3_2[as.integer(g3_2)] + u3_3[as.integer(g3_3)]
t0_3 <- runif(n3, 0, 0.4)
ev3 <- t0_3 + rexp(n3, rate = 0.5 * exp(eta3))
cen3 <- t0_3 + runif(n3, 0.5, 4.0)
df_3re <- data.frame(L = t0_3, R = pmin(ev3, cen3), st = as.integer(ev3 <= cen3),
                     x1 = x3_a, x2 = x3_b, c1 = g3_1, c2 = g3_2, c3 = g3_3)
invisible(capture.output(
  fit_3re <- frailtyHL(Surv(L, R, st) ~ x1 + x2 + (1 | c1) + (1 | c2) + (1 | c3),
                       df_3re, RandDist = "Gamma", mord = 0, dord = 1, Maxiter = 200)
))
stopifnot(
  identical(unname(fit_3re$convergence), "converged"),
  all(is.finite(fit_3re$RandCoef)), all(fit_3re$RandCoef[, 2L] > 0)
)

# cgd HL(1,2): non-convergent configuration reports non-convergence honestly
w_cgd12 <- FALSE
capture.output(withCallingHandlers(
  cgd_12 <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                      cgd, RandDist = "Gamma", mord = 1, dord = 2, Maxiter = 50),
  warning = function(w) { w_cgd12 <<- TRUE; invokeRestart("muffleWarning") }
))
stopifnot(w_cgd12, identical(unname(cgd_12$convergence), "did not converge"))

# ==============================================================================
# 4. Gamma dord = 0 & AR(1) mord = 1 / dord = 0
# ==============================================================================
invisible(capture.output({
  kid_p0_00 <- frailtyHL(Surv(time, status) ~ (1 | id), kidney, RandDist = "Gamma", mord = 0, dord = 0, Maxiter = 200)
  rat_p0_00 <- frailtyHL(Surv(time, status) ~ (1 | litter), rats, RandDist = "Gamma", mord = 0, dord = 0, Maxiter = 200)
  rat_rx_00 <- frailtyHL(Surv(time, status) ~ rx + (1 | litter), rats, RandDist = "Gamma", mord = 0, dord = 0, Maxiter = 200)
  rat_rx_10 <- frailtyHL(Surv(time, status) ~ rx + (1 | litter), rats, RandDist = "Gamma", mord = 1, dord = 0, Maxiter = 200)
  cgd_00    <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                         cgd, RandDist = "Gamma", mord = 0, dord = 0, Maxiter = 200)
  cgd_10    <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
                         cgd, RandDist = "Gamma", mord = 1, dord = 0, Maxiter = 200)
}))
stopifnot(
  identical(unname(kid_p0_00$convergence), "converged"), kid_p0_00$iter <= 15L,
  abs(kid_p0_00$RandCoef[1L, 1L] - 0.1005374) < 1e-5, abs(kid_p0_00$RandCoef[1L, 2L] - 0.246477) < 1e-5,
  identical(unname(rat_p0_00$convergence), "converged"), rat_p0_00$iter <= 20L,
  abs(rat_p0_00$RandCoef[1L, 1L] - 0.0169926) < 1e-5, abs(rat_p0_00$RandCoef[1L, 2L] - 0.192871) < 1e-5,
  identical(unname(rat_rx_00$convergence), "converged"), rat_rx_00$iter <= 20L,
  abs(rat_rx_00$RandCoef[1L, 1L] - 0.0124678) < 1e-5, abs(rat_rx_00$RandCoef[1L, 2L] - 0.193155) < 1e-5,
  identical(unname(rat_rx_10$convergence), "converged"), rat_rx_10$iter <= 20L,
  abs(rat_rx_10$RandCoef[1L, 1L] - 0.0124681) < 1e-5, abs(rat_rx_10$RandCoef[1L, 2L] - 0.193155) < 1e-5,
  identical(unname(cgd_00$convergence), "converged"),
  identical(unname(cgd_10$convergence), "converged")
)

# AR(1): mord=0/1 & dord=0/1 + cluster/row permutation
invisible(capture.output({
  ar_01 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 1, Maxiter = 500)
  ar_11 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 1, dord = 1, Maxiter = 500)
  ar_00 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 0, Maxiter = 1000)
  ar_10 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 1, dord = 0, Maxiter = 1000)
}))
stopifnot(
  identical(unname(ar_01$convergence), "converged"),
  abs(ar_01$FixCoef[1L, 1L] - (-0.92696045)) < 1e-5,
  abs(ar_01$RandCoef[1L, 1L] - 0.16233475) < 1e-5,
  abs(ar_01$rho - 0.81137188) < 1e-5,
  identical(unname(ar_11$convergence), "converged"),
  abs(ar_11$FixCoef[1L, 1L] - (-0.957761)) < 1e-5,
  abs(ar_11$RandCoef[1L, 1L] - 0.167238) < 1e-5,
  abs(ar_11$rho - 0.809001) < 1e-5,
  identical(unname(ar_00$convergence), "converged"), ar_00$RandCoef[1L, 1L] > 0.05,
  identical(unname(ar_10$convergence), "converged"), ar_10$RandCoef[1L, 1L] > 0.05
)

set.seed(12345)
cls_ar <- sample(unique(ren$rat))
rows_by_cls <- lapply(cls_ar, function(id) which(ren$rat == id))
perm_ar <- integer(0); rem_ar <- rows_by_cls
while (length(rem_ar) > 0L) {
  pk <- sample(length(rem_ar), 1L)
  perm_ar <- c(perm_ar, rem_ar[[pk]][1L])
  if (length(rem_ar[[pk]]) == 1L) rem_ar <- rem_ar[-pk] else rem_ar[[pk]] <- rem_ar[[pk]][-1L]
}
ren_perm <- ren[perm_ar, ]
ren_perm$rat <- factor(ren_perm$rat, levels = sample(levels(factor(ren$rat))))
invisible(capture.output(
  ar_11_p <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren_perm, RandDist = "AR1", mord = 1, dord = 1, Maxiter = 500)
))
stopifnot(
  max(abs(ar_11$FixCoef - ar_11_p$FixCoef)) < 1e-9,
  max(abs(ar_11$RandCoef - ar_11_p$RandCoef)) < 1e-9,
  abs(ar_11$rho - ar_11_p$rho) < 1e-9
)

# ==============================================================================
# 5. Joint Models Renal Benchmark Consistency (Ha, Jeong and Lee, 2017, Sec. 9.2)
# ==============================================================================
rc_filt <- subset(renal, month <= sur_time)
rs_filt <- subset(rc_filt, first == 1)
jm_l <- jointmodeling(Model = "mean", RespDist = "gaussian", Link = "identity",
                      LinPred = icr ~ month + sex + age + (1 | id), RandDist = "gaussian")
jm_s <- jointmodeling(Model = "mean", RespDist = "FM", Link = "log",
                      LinPred = Surv(sur_time, status) ~ sex + age + (1 | id), RandDist = "gaussian")
invisible(capture.output(
  fit_r92 <- jmfit(list(jm_l, jm_s), list(rc_filt, rs_filt), Maxiter = 200)
))
stopifnot(
  fit_r92$iter == 97L,
  abs(fit_r92$D.Est[1L] - 0.013225) < 1e-4,
  abs(fit_r92$D.Est[2L] - 0.027001) < 1e-4,
  abs(fit_r92$D.Est[3L] - (-15.25595)) < 1e-4
)

cat("LIKELIHOOD CONTINUITY AND MODEL CONSISTENCY TESTS PASS\n")
