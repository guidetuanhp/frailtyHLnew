# End-to-end model-family, export surface, and boundary tests for frailtyHLnew
library(survival)
library(cmprsk)
library(frailtyHLnew)

# 1. Public export surface check
expected_exports <- c("frailtyHL", "jointmodeling", "hlike.frailty", "jmfit", "frailty.vs", "mlmfit", "CmpRsk", "summary.frailtyHL")
stopifnot(identical(sort(getNamespaceExports("frailtyHLnew")), sort(expected_exports)))

# 2. All (RandDist, mord, dord) Cox models + schema + summary method
set.seed(42)
n_sim <- 40L
q_sim <- 8L
grp_sim <- rep(seq_len(q_sim), each = n_sim / q_sim)
x1_sim <- rnorm(n_sim)
x2_sim <- rbinom(n_sim, 1, 0.5)
t_sim <- rexp(n_sim, rate = exp(0.3 * x1_sim - 0.2 * x2_sim))
ev_sim <- sample(c(0, 1), n_sim, replace = TRUE, prob = c(0.25, 0.75))
df_sim <- data.frame(time = t_sim, status = ev_sim, x1 = x1_sim, x2 = x2_sim, id = grp_sim)

for (dist in c("Gamma", "Normal")) {
  for (m_ord in c(0L, 1L)) {
    for (d_ord in c(0L, 1L, 2L)) {
      invisible(capture.output(
        fit <- frailtyHL(
          Surv(time, status) ~ x1 + x2 + (1 | id),
          data = df_sim, RandDist = dist, mord = m_ord, dord = d_ord,
          varinit = 0.2, Maxiter = 250, convergence = 1e-6
        )
      ))
      stopifnot(
        is.null(fit$backend),
        identical(fit$Method, sprintf("HL(%d,%d)", m_ord, d_ord)),
        is.matrix(fit$FixCoef), all(is.finite(fit$FixCoef)),
        is.matrix(fit$likelihood), is.matrix(fit$aic), is.matrix(fit$Hinv),
        identical(dim(fit$Hinv), c(2L + q_sim, 2L + q_sim))
      )
      if (dist == "Gamma") {
        stopifnot(!is.null(fit$u_h), is.matrix(fit$Hinv_u), identical(dim(fit$Hinv_u), c(2L + q_sim, 2L + q_sim)))
      }
      sm <- summary(fit)
      stopifnot(inherits(fit, "frailtyHL"), inherits(sm, "summary.frailtyHL"))
    }
  }
}

# 3. Single-RE + Multi-RE + (1 | factor(id)) + weights/subset/na.action + boundary cases
data(kidney, package = "frailtyHLnew")
data(rats, package = "frailtyHLnew")
data(renal, package = "frailtyHLnew")
data(ren, package = "frailtyHLnew")
data(bladder0, package = "frailtyHLnew")
data(bladder, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")
data(test, package = "frailtyHLnew")

kidney_na <- kidney
kidney_na$sex[c(3, 19)] <- NA
kidney_na$wt <- rep(c(1, 1.2), length.out = nrow(kidney_na))
kidney_m <- kidney
kidney_m$grp2 <- factor((kidney_m$id %% 4L) + 1L)

invisible(capture.output({
  fit_fac <- frailtyHL(Surv(time, status) ~ sex + age + (1 | factor(id)), kidney, RandDist = "Gamma")
  fit_wsn <- frailtyHL(Surv(time, status) ~ sex + age + (1 | factor(id)), kidney_na,
    weights = wt, subset = age > 20, na.action = na.omit, RandDist = "Normal"
  )
  fit_mre_n <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id) + (1 | grp2), kidney_m,
    RandDist = "Normal", mord = 1, dord = 2, Maxiter = 10
  )
  fit_mre_g <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id) + (1 | grp2), kidney_m,
    RandDist = "Gamma", mord = 1, dord = 2, Maxiter = 10
  )
  fit_cox0 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
    RandDist = "Normal", varfixed = TRUE, varinit = 0
  )
  fit_p0_n <- frailtyHL(Surv(time, status) ~ (1 | id), kidney, RandDist = "Normal", mord = 0, dord = 1)
  fit_p0_g <- frailtyHL(Surv(time, status) ~ (1 | id), kidney, RandDist = "Gamma", mord = 0, dord = 1)
  fit_max2 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
    RandDist = "Gamma", mord = 1, dord = 2, Maxiter = 2
  )
}))
stopifnot(
  all(is.finite(fit_fac$FixCoef)), all(is.finite(fit_wsn$FixCoef)),
  all(is.finite(fit_mre_n$FixCoef)), all(is.finite(fit_mre_g$FixCoef)),
  identical(fit_cox0$Model, "Cox model"), fit_cox0$RandCoef[1L, 1L] == 0,
  identical(nrow(fit_p0_n$FixCoef), 0L), identical(fit_p0_n$p, 0L), all(is.finite(fit_p0_n$RandCoef)),
  identical(nrow(fit_p0_g$FixCoef), 0L), identical(fit_p0_g$p, 0L), all(is.finite(fit_p0_g$RandCoef)),
  identical(unname(fit_max2$convergence), "did not converge"), identical(as.integer(fit_max2$iter), 2L)
)

# Zero-time censored observations in bladder0 and bladder match filtered datasets
bladder0_pos <- subset(bladder0, Surtime > 0)
bladder_cs1 <- transform(bladder, ev1 = as.numeric(status == 1))
bladder_cs1_pos <- subset(bladder_cs1, surtime > 0)
invisible(capture.output({
  fit_b0 <- frailtyHL(Surv(Surtime, Status) ~ Chemo + Tustat + (1 | Center), bladder0, RandDist = "Gamma")
  fit_b0_pos <- frailtyHL(Surv(Surtime, Status) ~ Chemo + Tustat + (1 | Center), bladder0_pos, RandDist = "Gamma")
  fit_bl <- frailtyHL(Surv(surtime, ev1) ~ CHEMO + AGE + SEX + TLOCC + CIS + (1 | center), bladder_cs1, RandDist = "Normal")
  fit_bl_pos <- frailtyHL(Surv(surtime, ev1) ~ CHEMO + AGE + SEX + TLOCC + CIS + (1 | center), bladder_cs1_pos, RandDist = "Normal")
}))
stopifnot(
  max(abs(fit_b0$FixCoef - fit_b0_pos$FixCoef)) < 1e-12,
  max(abs(fit_bl$FixCoef - fit_bl_pos$FixCoef)) < 1e-12
)

# 4. AR(1) models (dord = 0, 1, 2 and left-truncated)
df_lt <- data.frame(
  L0 = pmax(0, kidney$time * 0.2), time = kidney$time,
  status = kidney$status, sex = kidney$sex, age = kidney$age, id = kidney$id
)
invisible(capture.output({
  fit_a00 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 0, Maxiter = 5)
  fit_a01 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 1, Maxiter = 5)
  fit_a02 <- frailtyHL(Surv(time, del) ~ gp + (1 | rat), ren, RandDist = "AR1", mord = 0, dord = 2, Maxiter = 5)
  fit_a_lt <- frailtyHL(Surv(L0, time, status) ~ sex + age + (1 | id), df_lt, RandDist = "AR1", mord = 0, dord = 1, Maxiter = 5)
}))
stopifnot(
  all(is.finite(fit_a00$FixCoef)), is.finite(fit_a00$rho),
  all(is.finite(fit_a01$FixCoef)), is.finite(fit_a01$rho),
  all(is.finite(fit_a02$FixCoef)), is.finite(fit_a02$rho),
  all(is.finite(fit_a_lt$FixCoef)), is.finite(fit_a_lt$rho)
)

# 5. Competing risks (hlike.frailty) including order = 2 covariance validation
n_t <- length(unique(test$id))
inits_t <- list(beta = matrix(0, 2, 2), theta = matrix(0.5, 1, 1), v = matrix(0, n_t, 1))
inits_t2 <- list(beta = matrix(0, 2, 2), theta = diag(0.5, 2), v = matrix(0, n_t, 2))
inits_t_sh <- list(beta = matrix(0, 2, 1), theta = matrix(0.5, 1, 1), v = matrix(0, n_t, 1))
invisible(capture.output({
  fit_cr1 <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id),
    data = test,
    inits = inits_t, order = 1, frailty.cov = "none", MAX.ITER = 10
  )
  fit_cr2 <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id),
    data = test,
    inits = inits_t, order = 2, frailty.cov = "none", MAX.ITER = 5
  )
  fit_cr_sh <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id),
    data = test,
    inits = inits_t_sh, order = 1, frailty.cov = "none", subHazard = TRUE, MAX.ITER = 10
  )
  fit_cr_ind <- hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id),
    data = test,
    inits = inits_t2, order = 1, frailty.cov = "independent", MAX.ITER = 5
  )
}))
stopifnot(
  all(is.finite(fit_cr1$beta$Estimate)), all(is.finite(fit_cr2$beta$Estimate)),
  all(is.finite(fit_cr_sh$beta$Estimate)), all(is.finite(fit_cr_ind$beta$Estimate))
)
err_cr2 <- tryCatch(
  hlike.frailty(CmpRsk(time, status) ~ x1 + x2 + cluster(id),
    data = test,
    inits = inits_t2, order = 2, frailty.cov = "independent"
  ),
  error = function(e) conditionMessage(e)
)
stopifnot(identical(err_cr2, "order = 2 is only supported when frailty.cov = 'none'"))

# 6. Mixed-effects AFT (mlmfit 1-RE, 2-RE, HL(0,1) & HL(1,1), LTRC rejection)
cgd_rc <- cgd[cgd$tstart == 0 | !duplicated(cgd$id), ]
cgd_rc$stime <- pmax(cgd_rc$tstop - cgd_rc$tstart, 1e-4)
jm_m1 <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(stime, status) ~ treat + sex + (1 | id), RandDist = "gaussian")
jm_m2 <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(stime, status) ~ treat + sex + (1 | center) + (1 | id), RandDist = "gaussian")
jm_k1 <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(time, status) ~ sex + age + (1 | id), RandDist = "gaussian")
invisible(capture.output({
  fit_m1 <- mlmfit(jm_m1, cgd_rc, Maxiter = 20)
  fit_m2 <- mlmfit(jm_m2, cgd_rc, Maxiter = 20)
  fit_m1_hl11 <- mlmfit(jm_k1, kidney, Maxiter = 200, mord = 1)
}))
stopifnot(all(is.finite(fit_m1$F.Est)), all(is.finite(fit_m2$F.Est)), all(is.finite(fit_m1_hl11$F.Est)))

jm_lt <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(L0, time, status) ~ sex + age + (1 | id), RandDist = "gaussian")
stopifnot(inherits(try(mlmfit(jm_lt, df_lt), silent = TRUE), "try-error"))

# 7. Joint modeling (jmfit2, jmfit3, jmfit_FM)
renal_s <- subset(renal, first == 1)
jm2_l <- jointmodeling(Model = "mean", RespDist = "gaussian", Link = "identity", LinPred = icr ~ month + sex + age + (1 | id), RandDist = "gaussian")
jm2_s <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(sur_time, status) ~ sex + age + (1 | id), RandDist = "gaussian")
invisible(capture.output(fit_jm2 <- jmfit(list(jm2_l, jm2_s), list(renal, renal_s), Maxiter = 10)))
stopifnot(all(is.finite(fit_jm2$F.Est)), all(is.finite(fit_jm2$D.Est)))

set.seed(1042)
q_j <- 20L
id_jl <- rep(seq_len(q_j), each = 3L)
v_j <- rnorm(q_j, 0, 0.35)
df_jl <- data.frame(id = id_jl, t = rep(1:3, q_j), x1 = rnorm(3L * q_j), y = 1 + 0.3 * rep(1:3, q_j) - 0.2 * rnorm(3L * q_j) + v_j[id_jl] + rnorm(3L * q_j, 0, 0.3))
df_js <- data.frame(
  id = seq_len(q_j), x2 = rnorm(q_j), x3 = rbinom(q_j, 1L, 0.5),
  time1 = pmax(rexp(q_j, 0.35), 1e-3), st1 = rep(c(1L, 1L, 0L, 1L), 5L),
  time2 = pmax(rexp(q_j, 0.30), 1e-3), st2 = rep(c(1L, 0L, 1L, 1L), 5L)
)
jm3_l <- jointmodeling(Model = "mean", RespDist = "gaussian", Link = "identity", LinPred = y ~ t + x1 + (1 | id), RandDist = "gaussian")
jm3_s1 <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(time1, st1) ~ x2 + x3 + (1 | id), RandDist = "gaussian")
jm3_s2 <- jointmodeling(Model = "mean", RespDist = "AFT", Link = "log", LinPred = Surv(time2, st2) ~ x2 + x3 + (1 | id), RandDist = "gaussian")
fm1_j <- jointmodeling(Model = "mean", RespDist = "FM", Link = "log", LinPred = Surv(time1, st1) ~ x2 + x3 + (1 | id), RandDist = "gaussian")
fm2_j <- jointmodeling(Model = "mean", RespDist = "FM", Link = "log", LinPred = Surv(time2, st2) ~ x2 + x3 + (1 | id), RandDist = "gaussian")
invisible(capture.output({
  fit_jm3 <- jmfit(list(jm3_l, jm3_s1, jm3_s2), list(df_jl, df_js), Maxiter = 10)
  fit_fm <- jmfit(fm1_j, df_js, jm2 = fm2_j, data2 = df_js, Maxiter = 10)
}))
stopifnot(all(is.finite(fit_jm3$F.Est)), all(is.finite(fit_fm$F.Est)))

# 8. Variable selection (frailty.vs 1-RE & correlated 2-RE)
vs_1_div_stopped <- tryCatch({
  invisible(capture.output(
    frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
      model = "lognorm", penalty = "LASSO", data = kidney,
      B = c(0.09, -1.66, 0.17, 0.39, -1.16), tun1 = c(0.02, 0.032)
    )
  ))
  FALSE
}, error = function(e) TRUE)
stopifnot(isTRUE(vs_1_div_stopped))
invisible(capture.output({
  vs_1 <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
    model = "lognorm", penalty = "LASSO", data = kidney,
    B = NULL, tun1 = c(0.02, 0.032)
  )
  vs_2 <- frailty.vs(Surv(time, status) ~ sex + age + (1 | id),
    model = "lognorm", penalty = "SCAD", data = kidney,
    B = NULL, tun1 = 0.05
  )
}))
stopifnot(
  all(is.finite(vs_1$beta)), is.finite(vs_1$BIC), vs_1$BIC < 99999, vs_1$convergence == 1L,
  all(is.finite(vs_2$beta))
)

set.seed(2028)
n_vs <- 150L
q1_vs <- 25L
x1_vs <- rnorm(n_vs)
x2_vs <- rnorm(n_vs)
x3_vs <- rbinom(n_vs, 1, 0.5)
id_vs <- factor(((seq_len(n_vs) - 1L) %% q1_vs) + 1L)
z1_vs <- rnorm(q1_vs)
z2_vs <- rnorm(q1_vs)
u1_vs <- sqrt(0.4) * z1_vs
u2_vs <- (0.1 / sqrt(0.4)) * z1_vs + sqrt(0.3 - 0.01 / 0.4) * z2_vs
eta_vs <- 0.6 * x1_vs - 0.3 * x2_vs + u1_vs[as.integer(id_vs)] + x1_vs * u2_vs[as.integer(id_vs)]
tt_vs <- rexp(n_vs) / exp(eta_vs)
cc_vs <- rexp(n_vs, 0.3)
df_sim_ln2 <- data.frame(
  time = pmin(tt_vs, cc_vs) + 0.01, status = as.integer(tt_vs <= cc_vs),
  x1 = x1_vs, x2 = x2_vs, x3 = x3_vs, id = id_vs
)
invisible(capture.output(
  vs_ln2 <- frailty.vs(Surv(time, status) ~ x1 + x2 + x3 + (1 | id) + (x1 | id),
    model = "lognorm", penalty = "HL", data = df_sim_ln2,
    B = c(0.5, -0.2, 0.0), alpha = c(0.3, 0.25, 0.05),
    tun1 = c(0.2, 0.5), tun2 = c(0.05, 0.1)
  )
))
stopifnot(all(is.finite(vs_ln2$beta)), is.finite(vs_ln2$BIC))

cat("FRAILTYHLNEW MODEL-FAMILY TESTS PASS\n")
