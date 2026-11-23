# Native boundary, input validation, and namespace lifecycle tests for frailtyHLnew
library(frailtyHLnew)

assert_rf_error <- function(expr, label) {
  caught <- tryCatch({
    force(expr)
    FALSE
  }, error = function(e) TRUE)
  if (!isTRUE(caught)) {
    stop(sprintf("Expected Rf_error was not raised for: %s", label))
  }
}

# 1. Competing-risks native boundary checks (fhl_fit_competing_risks)
assert_rf_error(.Call("fhl_fit_competing_risks",
  matrix(1, 2, 1), c(1L, 3L), 2L, c(1.0, 2.0), c(1L, 2L),
  list(c(1.0), c(2.0)), list(c(1.0), c(1.0)),
  matrix(0, 1, 2), matrix(0.5, 1, 1), matrix(0, 2, 1),
  1L, 0L, 0L, 1L, 1e-6, PACKAGE = "frailtyHLnew"), "HlikeFit_group_oob")
assert_rf_error(.Call("fhl_fit_competing_risks",
  matrix(1, 2, 1), c(1L, 2L), 2L, c(1.0, 2.0), c(1L, 2L),
  list(c(1.0), c(2.0)), list(c(1.0), c(1.0)),
  matrix(0, 1, 2), diag(0.5, 2), matrix(0, 2, 2),
  2L, 1L, 0L, 1L, 1e-6, PACKAGE = "frailtyHLnew"), "HlikeFit_order2_cov")
assert_rf_error(.Call("fhl_fit_competing_risks",
  matrix(1, 2, 1), c(1L, 2L), 2L, c(1.0, 2.0), c(1L, 2L),
  list(c(1.0), c(2.0)), list(c(1.0), c(1.0)),
  matrix(0, 99, 2), matrix(0.5, 1, 1), matrix(0, 2, 1),
  1L, 0L, 0L, 1L, 1e-6, PACKAGE = "frailtyHLnew"), "HlikeFit_beta_dim_mismatch")

# 2. Joint modeling (fhl_fit_jm_cox1, fhl_fit_jm_cox2, fhl_fit_jm_frailty) native boundary checks
Z_ok <- Matrix::sparseMatrix(i = 1:4, j = c(1L, 1L, 2L, 2L), x = 1.0, dims = c(4L, 2L), repr = "C")
Z_badq <- Matrix::sparseMatrix(i = 1:4, j = c(1L, 2L, 3L, 3L), x = 1.0, dims = c(4L, 3L), repr = "C")
Mi_ok <- frailtyHLnew:::fhl_build_risk_set(c(1.0, 2.0, 1.5, 2.5), rep(0.0, 4L), c(1.0, 2.0))
Mi_bad2 <- frailtyHLnew:::fhl_build_risk_set(c(1.0, 2.0), rep(0.0, 2L), c(1.0, 2.0))
X_ok <- matrix(1.0, 4, 1)
y_ok <- c(1.0, 2.0, 1.5, 2.5)
del_ok <- c(1.0, 1.0, 0.0, 0.0)
di_ok <- c(1.0, 1.0)
assert_rf_error(.Call("fhl_fit_jm_cox1", y_ok, c(1.0, 2.0), Z_ok, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JM2_X1_not_matrix")
assert_rf_error(.Call("fhl_fit_jm_cox1", c(1.0, 2.0), X_ok, Z_ok, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JM2_y1_len_mismatch")
assert_rf_error(.Call("fhl_fit_jm_cox1", y_ok, X_ok, Z_ok, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), -1L, PACKAGE = "frailtyHLnew"), "JM2_neg_maxiter")
assert_rf_error(.Call("fhl_fit_jm_cox1", y_ok, X_ok, Z_ok, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0), 1L, PACKAGE = "frailtyHLnew"), "JM2_v_len_mismatch")
assert_rf_error(.Call("fhl_fit_jm_cox2", y_ok, X_ok, Z_ok, del_ok, X_ok, Z_badq, di_ok, Mi_ok, 2L, 0.0, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JM3_q_mismatch")
assert_rf_error(.Call("fhl_fit_jm_cox2", y_ok, X_ok, Z_ok, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(1.0, 0.0), X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JM3_del3_len_mismatch")
assert_rf_error(.Call("fhl_fit_jm_frailty", del_ok, X_ok, Z_ok, di_ok, Mi_ok, 0L, 0.0, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JMFM_idx2_nonpos")
assert_rf_error(.Call("fhl_fit_jm_frailty", del_ok, X_ok, Z_ok, di_ok, Mi_bad2, 2L, 0.0, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JMFM_risk_dim_mismatch")
assert_rf_error(.Call("fhl_fit_jm_frailty", del_ok, X_ok, Z_ok, di_ok, matrix(1.0, 4, 2), 2L, 0.0, del_ok, X_ok, Z_ok, di_ok, Mi_ok, 2L, 0.0, c(0.0, 0.0), 1L, PACKAGE = "frailtyHLnew"), "JMFM_dense_Mi_rejected")

# 3. Multilevel AFT & Variable Selection native boundary checks
assert_rf_error(.Call("fhl_fit_aft_1re", y_ok, X_ok, c(1L, 1L, 99L, 2L), 2L, del_ok, 0.0, numeric(2), 5L, PACKAGE = "frailtyHLnew"), "AFT1_cluster_oob")
assert_rf_error(.Call("fhl_fit_aft_2re", y_ok, X_ok, c(1L, 1L, 2L, 2L), 2L, c(1L, 2L, 3L, 2L), 2L, del_ok, 0.0, numeric(4), 5L, PACKAGE = "frailtyHLnew"), "AFT2_group2_oob")
assert_rf_error(.Call("fhl_fit_aft_2re", y_ok, matrix(0, 4, 0), c(1L, 1L, 2L, 2L), 2L, c(1L, 2L, 1L, 2L), 2L, del_ok, numeric(0), numeric(4), 5L, PACKAGE = "frailtyHLnew"), "AFT2_zero_p")
assert_rf_error(.Call("fhl_fit_vs_1re", X_ok, numeric(2), list(Z_ok), list(numeric(2)), 0L, 0.1, c(0.01), NULL, Mi_ok, di_ok, del_ok, FALSE, PACKAGE = "frailtyHLnew"), "VS1_beta_len_mismatch")
assert_rf_error(.Call("fhl_fit_vs_2re", X_ok, 0.0, list(Z_ok), list(numeric(2)), 2L, c(0.1, 0.1), c(0.2, 0.3), c(0.1), di_ok, Mi_ok, del_ok, PACKAGE = "frailtyHLnew"), "VS2_tunset2_mismatch")

# 4. Repeated C-native fits + GC + unloadNamespace / reload lifecycle stress test
data(kidney, package = "frailtyHLnew")
for (k in seq_len(10L)) {
  invisible(capture.output({
    frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney, RandDist = "Gamma", mord = 1, dord = 2, Maxiter = 10)
    frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney, RandDist = "Normal", mord = 1, dord = 2, Maxiter = 10)
  }))
}
rm(k)
invisible(gc())

unloadNamespace("frailtyHLnew")
invisible(gc())
library(frailtyHLnew)
invisible(capture.output(
  fit_reload <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney, RandDist = "Gamma", mord = 0, dord = 1, Maxiter = 15)
))
stopifnot(all(is.finite(fit_reload$FixCoef)), all(is.finite(fit_reload$RandCoef)))

cat("MEMORY SAFETY, BOUNDARY, AND NAMESPACE LIFECYCLE TESTS PASS\n")
