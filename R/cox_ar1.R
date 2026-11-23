# ==============================================================================
# Cox AR(1) Frailty Runner (C tridiagonal precision kernel)
# ==============================================================================

fhl_run_cox_ar1 <- function(prep, mord, dord, varfixed, varnonneg, max_iter, convergence) {
  x_mat <- as.matrix(prep$x)
  storage.mode(x_mat) <- "double"
  n_obs <- nrow(x_mat)
  z1 <- prep$z[[1L]]
  if (nrow(z1) != n_obs) {
    stop("AR(1) design matrix row count does not match data rows")
  }
  if (methods::is(z1, "dgCMatrix")) {
    if (length(z1@x) != n_obs || any(abs(z1@x - 1.0) > 1e-12) ||
        any(tabulate(z1@i + 1L, nbins = n_obs) != 1L)) {
      stop("AR(1) frailty currently supports only random-intercept terms '(1 | cluster)' with one cluster per observation")
    }
    qrep <- Matrix::colSums(z1)
  } else {
    z1_mat <- as.matrix(z1)
    if (any(rowSums(z1_mat != 0) != 1L) || any(rowSums(abs(z1_mat - 1.0) <= 1e-12) != 1L)) {
      stop("AR(1) frailty currently supports only random-intercept terms '(1 | cluster)' with one cluster per observation")
    }
    qrep <- colSums(z1_mat)
  }
  if (!any(qrep >= 2L)) {
    stop("AR(1) correlation requires repeated observations within at least one cluster")
  }
  if (!is.null(prep$ar1_seq)) {
    attr(z1, "fhl_ar1_seq") <- as.integer(prep$ar1_seq)
  }
  z_list <- list(z1)
  v_init <- matrix(0, nrow = n_obs, ncol = 1L)
  .Call(
    "fhl_fit_cox_ar1",
    x_mat, z_list, prep$y, as.double(prep$del), prep$Mi,
    as.integer(prep$idx2), prep$t2, as.double(prep$di),
    as.double(prep$beta_h), as.double(v_init), as.double(prep$alpha_h),
    as.integer(mord), as.integer(dord),
    as.logical(varfixed), as.logical(varnonneg),
    as.integer(max_iter), as.double(convergence),
    PACKAGE = "frailtyHLnew"
  )
}
