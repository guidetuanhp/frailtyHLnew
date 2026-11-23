# ==============================================================================
# Cox Gamma Frailty Runner (single-RE & multi-RE unified C kernel)
# ==============================================================================

fhl_run_cox_gamma <- function(prep, mord, dord, varfixed, varnonneg, max_iter, convergence) {
  .Call(
    "fhl_fit_cox_gamma",
    prep$x, prep$z, prep$y, prep$del, prep$Mi,
    as.integer(prep$idx2), prep$t2, prep$di,
    prep$beta_h, prep$v_h, as.double(prep$alpha_h),
    as.integer(mord), as.integer(dord),
    isTRUE(varfixed), isTRUE(varnonneg),
    as.integer(max_iter), as.double(convergence),
    PACKAGE = "frailtyHLnew"
  )
}
