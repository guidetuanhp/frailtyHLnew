# ==============================================================================
# Variable Selection: Uncorrelated / Diagonal Lognormal Frailty (1-RE / Multi-RE)
# ==============================================================================

frailty1.vs <- function(
  formula, model, penalty, data, B = "null", v = "null", alpha = "null",
  tun1 = "null", tun2 = "null", varfixed = FALSE, varinit = 0.1,
  eval_env = NULL
) {
  if (identical(B, "null")) B <- NULL
  if (identical(v, "null")) v <- NULL
  if (identical(alpha, "null")) alpha <- NULL
  if (identical(tun1, "null")) tun1 <- NULL
  if (identical(tun2, "null")) tun2 <- NULL
  has_data <- !missing(data)
  prep_env <- if (has_data) environment() else if (!is.null(eval_env)) eval_env else parent.frame()
  penalty <- fhl_validate_frailty_vs_args(formula, model, penalty, varfixed, varinit, tun1, tun2)
  prep <- fhl_vs_prepare_data(mc = match.call(), formula = formula, caller_env = prep_env, build_risk = TRUE)

  if (is.null(alpha)) {
    alpha <- if (isTRUE(varfixed)) rep(max(varinit, 1e-06), prep$nrand) else rep(0.1, prep$nrand)
  }
  if (!is.numeric(alpha) || length(alpha) != prep$nrand || any(!is.finite(alpha)) || any(alpha <= 0)) {
    stop("'alpha' must contain positive finite numeric values of length nrand")
  }
  inits <- fhl_vs_prepare_inits(prep, B, v, formula, has_data, if (has_data) data else NULL, prep_env)

  result <- fhl_run_vs_1re(
    X = prep$x, B = inits$B, Z = prep$z, v = inits$v,
    penalty = penalty, alpha = alpha, tun1 = tun1, tun2 = tun2,
    di = prep$di, Mi = prep$Mi, del = prep$del, varfixed = varfixed
  )
  fhl_vs_print_result(result, model, penalty, formula, prep$namesX, prep$res1)
}

fhl_run_vs_1re <- function(X, B, Z, v, penalty, alpha, tun1, tun2, di, Mi, del, varfixed) {
  penalty <- fhl_normalize_penalty(penalty)
  pen_type <- switch(penalty, lasso = 0L, scad = 1L, hl = 2L)
  X_mat <- as.matrix(X)
  storage.mode(X_mat) <- "double"
  Z_arg <- fhl_vs_as_z_arg(Z)
  v_dbl <- lapply(v, as.double)
  B_dbl <- as.double(B)
  alpha_dbl <- as.double(alpha)
  di_dbl <- as.double(di)
  del_dbl <- as.double(del)
  vf_lgl <- as.logical(varfixed)
  grid <- fhl_vs_grid_vectors(pen_type, tun1, tun2)
  .Call(
    "fhl_fit_vs_1re",
    X_mat, B_dbl, Z_arg, v_dbl,
    as.integer(pen_type), alpha_dbl,
    grid$tunset1, grid$tunset2,
    di_dbl, Mi, del_dbl,
    vf_lgl,
    PACKAGE = "frailtyHLnew"
  )
}
