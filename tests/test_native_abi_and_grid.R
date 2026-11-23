# Native .Call registration, AFT design dispatch, and grid-search consistency tests
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

# 1. Verify the 11 registered .Call entrypoints
dll_info <- getDLLRegisteredRoutines("frailtyHLnew")
call_names <- sort(names(dll_info[[".Call"]]))
expected_calls <- sort(c(
  "fhl_fit_cox_gamma",
  "fhl_fit_cox_normal",
  "fhl_fit_cox_ar1",
  "fhl_fit_competing_risks",
  "fhl_fit_aft_1re",
  "fhl_fit_aft_2re",
  "fhl_fit_jm_cox1",
  "fhl_fit_jm_cox2",
  "fhl_fit_jm_frailty",
  "fhl_fit_vs_1re",
  "fhl_fit_vs_2re"
))
stopifnot(
  length(call_names) == 11L,
  identical(call_names, expected_calls)
)

# 2. Compressed risk-set descriptor rejects raw dense matrix Mi cleanly
data(kidney, package = "frailtyHLnew")
data(cgd, package = "frailtyHLnew")
n_k <- nrow(kidney)
assert_err(
  .Call("fhl_fit_cox_normal",
        matrix(kidney$sex, n_k, 1L), list(diag(n_k)),
        matrix(kidney$time, n_k, 1L), matrix(as.double(kidney$status), n_k, 1L),
        matrix(1.0, n_k, 4L), 1L, 4L, matrix(1.0, 4L, 1L),
        matrix(0.0, 1L, 1L), matrix(0.0, n_k, 1L), 0.1,
        0L, 1L, FALSE, FALSE, 1L, 1e-6, PACKAGE = "frailtyHLnew"),
  "fhl_build_risk_set"
)

# 3. Unified mlmfit preparation + internal C dispatch (cluster index vs general Z)
k_aft <- kidney
k_aft$ones <- 1.0
k_aft$age_sc <- (k_aft$age - mean(k_aft$age)) / sd(k_aft$age)
invisible(capture.output({
  fit_aft_idx <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(time, status) ~ sex + age_sc + (1 | id), "gaussian"),
    k_aft, Maxiter = 200
  )
  fit_aft_gen <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(time, status) ~ sex + age_sc + (ones | id), "gaussian"),
    k_aft, Maxiter = 200
  )
  fit_aft_slope <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(time, status) ~ sex + age_sc + (age_sc | id), "gaussian"),
    k_aft, Maxiter = 200
  )
}))
assert_err(
  mlmfit(jointmodeling("mean", "AFT", "log", Surv(time, status) ~ sex + (1 + age_sc | id), "gaussian"),
         k_aft, Maxiter = 50),
  "Correlated random effects"
)
stopifnot(
  max(abs(fit_aft_idx$F.Est - fit_aft_gen$F.Est)) <= 1e-12,
  max(abs(fit_aft_idx$D.Est - fit_aft_gen$D.Est)) <= 1e-12,
  all(is.finite(fit_aft_slope$F.Est)),
  all(is.finite(fit_aft_slope$D.Est)),
  all(fit_aft_slope$D.Est[, 1L] > 0)
)

cgd_aft <- cgd
cgd_aft$ones <- 1.0
cgd_aft$age_sc <- (cgd_aft$age - mean(cgd_aft$age)) / sd(cgd_aft$age)
invisible(capture.output({
  fit_aft2_idx <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id), "gaussian"),
    cgd_aft, Maxiter = 200
  )
  fit_aft2_gen <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(tstop - tstart, status) ~ treat + (1 | center) + (ones | id), "gaussian"),
    cgd_aft, Maxiter = 200
  )
  fit_aft2_slope <- mlmfit(
    jointmodeling("mean", "AFT", "log", Surv(tstop - tstart, status) ~ treat + age_sc + (1 | center) + (age_sc | id), "gaussian"),
    cgd_aft, Maxiter = 200
  )
}))
stopifnot(
  max(abs(fit_aft2_idx$F.Est - fit_aft2_gen$F.Est)) <= 1e-11,
  max(abs(fit_aft2_idx$D.Est - fit_aft2_gen$D.Est)) <= 1e-11,
  all(is.finite(fit_aft2_slope$F.Est)),
  all(is.finite(fit_aft2_slope$D.Est)),
  all(fit_aft2_slope$D.Est[, 1L] > 0)
)

# 4. Native frailty.vs grid search (1-RE & 2-RE): multi-point grid matches single-point optimum
k_vs <- kidney
k_vs$age <- (k_vs$age - mean(k_vs$age)) / sd(k_vs$age)
k_vs$GN <- as.numeric(k_vs$disease == "GN")
k_vs$AN <- as.numeric(k_vs$disease == "AN")
k_vs$PKD <- as.numeric(k_vs$disease == "PKD")

invisible(capture.output({
  vs_grid <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
                        model = "lognorm", penalty = "lasso", data = k_vs,
                        B = c(0.074, -1.659, 0.173, 0.387, -1.161),
                        tun1 = c(0.01, 0.029, 0.05))
  vs_pt <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
                      model = "lognorm", penalty = "lasso", data = k_vs,
                      B = c(0.074, -1.659, 0.173, 0.387, -1.161),
                      tun1 = vs_grid$tuning_parameter)
}))
stopifnot(
  identical(vs_grid$beta, vs_pt$beta),
  identical(vs_grid$alpha, vs_pt$alpha),
  identical(vs_grid$BIC, vs_pt$BIC)
)

# Partial grid failure warning vs single-point immediate stop on 2-RE grid
cgd_vs <- cgd
cgd_vs$age <- (cgd_vs$age - mean(cgd_vs$age)) / sd(cgd_vs$age)
cgd_vs$x_zero <- 0.0
w_2re <- character(0)
withCallingHandlers(
  invisible(capture.output(
    vs2_part <- frailty.vs(Surv(tstop - tstart, status) ~ treat + x_zero + age + (1 | center) + (1 | id),
                           model = "lognorm", penalty = "lasso", data = cgd_vs,
                           B = c(-1, 0, 0), tun1 = c(0, 0.05))
  )),
  warning = function(w) { w_2re <<- c(w_2re, conditionMessage(w)); invokeRestart("muffleWarning") }
)
stopifnot(
  abs(vs2_part$tuning_parameter - 0.05) < 1e-12,
  any(grepl("BIC = Inf", w_2re, fixed = TRUE))
)
assert_err(
  frailty.vs(Surv(tstop - tstart, status) ~ treat + x_zero + age + (1 | center) + (1 | id),
             model = "lognorm", penalty = "lasso", data = cgd_vs,
             B = c(-1, 0, 0), tun1 = 0),
  "not positive definite|Singular"
)

# 5. Lexical scoping in frailtyHL formula evaluation
run_scoped_cox <- function() {
  local_df <- kidney[1:40, ]
  frailtyHL(Surv(time, status) ~ sex + age + (1 | id), data = local_df,
            RandDist = "Normal", mord = 0, dord = 1, Maxiter = 30)
}
invisible(capture.output(fit_scoped <- run_scoped_cox()))
stopifnot(all(is.finite(fit_scoped$FixCoef)))

cat("NATIVE ABI, AFT DISPATCH, AND GRID SEARCH TESTS PASS\n")
