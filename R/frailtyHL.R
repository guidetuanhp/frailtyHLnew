# ==============================================================================
# Public Cox Frailty API, Data Preparation, Formatter & Lifecycle
# ==============================================================================

frailtyHL <- function(
  formula, data, weights, subset, na.action, RandDist = "Normal",
  mord = 0, dord = 1, Maxiter = 200, convergence = 10^-6, varfixed = FALSE,
  varinit = c(0.16300000000000001), varnonneg = FALSE
) {
  fhl_validate_frailtyhl_args(
    formula, RandDist, mord, dord, Maxiter, convergence,
    varfixed, varinit, varnonneg
  )
  caller_env <- parent.frame()
  mc <- match.call()
  prep <- fhl_prepare_cox_data(
    mc = mc,
    formula = formula,
    caller_env = caller_env,
    varinit = varinit
  )
  fit_c <- switch(RandDist,
    Gamma  = fhl_run_cox_gamma(prep, mord, dord, varfixed, varnonneg, Maxiter, convergence),
    Normal = fhl_run_cox_normal(prep, mord, dord, varfixed, varnonneg, Maxiter, convergence),
    AR1    = fhl_run_cox_ar1(prep, mord, dord, varfixed, varnonneg, Maxiter, convergence)
  )
  fhl_format_cox_result(
    fit_c = fit_c,
    prep = prep,
    formula = formula,
    RandDist = RandDist,
    mord = mord,
    dord = dord,
    varfixed = varfixed,
    varinit = varinit,
    convergence = convergence
  )
}

summary.frailtyHL <- function(object, ...) {
  ans <- list(0)
  names(ans)[1] <- "Model"
  if (object$Model == "Cox model") {
    ans$Model <- "Results from the Cox model"
  }
  if (object$Model == "gamma frailty model") {
    ans$Model <- "Results from the gamma frailty model"
  }
  if (object$Model == "log-normal frailty model") {
    ans$Model <- "Results from the log-normal frailty model"
  }
  if (object$Model == "log-normal frailty model with AR(1)") {
    ans$Model <- "Results from the log-normal frailty model with AR(1)"
  }
  ans$formula <- object$formula
  if (!is.null(object$Method)) {
    ans$method <- paste0("Method : ", object$Method)
  }
  ans$FixCoef <- object$FixCoef
  ans$RandCoef <- object$RandCoef
  ans$likelihood <- object$likelihood
  ans$iter <- object$iter
  ans$convergence <- object$convergence
  ans$aic <- object$aic
  class(ans) <- "summary.frailtyHL"
  ans
}

fhl_validate_frailtyhl_args <- function(
  formula, RandDist, mord, dord, Maxiter, convergence, varfixed, varinit, varnonneg
) {
  if (missing(formula) || !inherits(formula, "formula") || length(formula) < 3L) {
    stop("'formula' must be a two-sided formula")
  }
  bars <- expandSlash(findbars(formula[[3]]))
  if (!length(bars)) {
    stop("No random effects terms specified in formula")
  }
  for (b in bars) {
    cc <- findplus(b)
    if (cc == 1) {
      stop("Correlated random effects are not currently allowed in the HGLM routines")
    }
    if (cc == -1) {
      stop("You do not need to specify '-1' for no intercept it is done be default")
    }
  }
  if (!is.character(RandDist) || length(RandDist) != 1L || is.na(RandDist) ||
    !(RandDist %in% c("Normal", "Gamma", "AR1"))) {
    stop("'RandDist' must be one of \"Normal\", \"Gamma\", or \"AR1\"")
  }
  if (RandDist == "AR1" && length(bars) != 1L) {
    stop("AR(1) frailty currently requires a single random-effect term")
  }
  if (!is.numeric(mord) || length(mord) != 1L || is.na(mord) || !is.finite(mord) ||
    !(mord %in% c(0, 1))) {
    stop("'mord' must be 0 or 1")
  }
  if (!is.numeric(dord) || length(dord) != 1L || is.na(dord) || !is.finite(dord) ||
    !(dord %in% c(0, 1, 2))) {
    stop("'dord' must be 0, 1, or 2")
  }
  if (!is.logical(varfixed) || length(varfixed) != 1L || is.na(varfixed)) {
    stop("'varfixed' must be TRUE or FALSE")
  }
  if (!is.logical(varnonneg) || length(varnonneg) != 1L || is.na(varnonneg)) {
    stop("'varnonneg' must be TRUE or FALSE")
  }
  if (!is.numeric(varinit) || length(varinit) < 1L || any(!is.finite(varinit)) ||
    any(varinit < 0)) {
    stop("'varinit' must contain finite non-negative numeric values")
  }
  if (!is.numeric(Maxiter) || length(Maxiter) != 1L || is.na(Maxiter) || !is.finite(Maxiter) ||
    Maxiter < 1) {
    stop("'Maxiter' must be >= 1")
  }
  if (!is.numeric(convergence) || length(convergence) != 1L || is.na(convergence) ||
    !is.finite(convergence) || convergence <= 0) {
    stop("'convergence' must be a positive finite number")
  }
  invisible(TRUE)
}

fhl_prepare_cox_data <- function(mc, formula, caller_env, varinit) {
  indx <- match(c("formula", "data", "weights", "subset", "na.action"),
    names(mc),
    nomatch = 0L
  )
  if (indx[1L] == 0L) {
    stop("A formula argument is required")
  }
  fr <- FrailtyFrames(mc, formula, contrasts = NULL, eval_env = caller_env)
  Y <- fr$Y
  if (length(fr$off) > 0L) {
    stop("'offset()' terms are not supported in frailtyHL")
  }
  if (length(fr$wts) > 0L) {
    warning("'weights' argument is ignored by frailtyHL (weighted Cox frailty likelihood is not implemented)")
  }
  has_icpt <- length(fr$fixef) > 0L && identical(names(fr$fixef)[1L], "(Intercept)")
  namesX <- if (has_icpt) names(fr$fixef)[-1L] else names(fr$fixef)
  FL <- HGLMFactorList(formula, fr, 0L, 0L, sparse = TRUE)
  namesRE <- FL$namesRE
  nc_y <- ncol(Y)
  if (is.null(nc_y) || !(nc_y %in% c(2L, 3L))) {
    stop("'formula' response must be a valid Surv(time, status) or Surv(entry, exit, status) object")
  }
  leftT <- if (nc_y == 3L) 1L else 0L
  n_y <- nrow(Y)
  y <- matrix(Y[, if (leftT == 0L) 1L else 2L], n_y, 1L)
  L0 <- if (leftT == 0L) matrix(0, n_y, 1L) else matrix(Y[, 1L], n_y, 1L)
  x <- if (has_icpt) fr$X[, -1L, drop = FALSE] else fr$X
  if (!has_icpt && ncol(x) > 0L && qr(x)$rank < ncol(x)) {
    stop("Fixed-effect design is rank deficient without intercept; the Cox baseline hazard absorbs the intercept in frailtyHL")
  }
  z <- FL$Design
  n <- nrow(x)
  p <- ncol(x)
  nrand <- length(z)
  q <- vapply(z, function(zi) dim(zi)[2L], integer(1L))
  if (any(q < 2L)) {
    stop("Random-effect grouping factor must have at least 2 levels")
  }
  del <- matrix(0, n, 1L)
  del[, 1L] <- censor <- Y[, if (leftT == 0L) 2L else 3L]
  fhl_validate_surv_data(y, censor, x, L0, fn_name = "frailtyHL")
  ar1_seq <- seq_len(n)
  res1 <- fhl_make_data(y, x, del, z, L0)
  o_sort <- attr(res1, "sort_order")
  if (!is.null(o_sort)) {
    ar1_seq <- ar1_seq[o_sort]
  }
  y <- res1[[1L]]
  x <- res1[[2L]]
  del <- res1[[3L]]
  z <- res1[[4L]]
  Mi <- res1[[5L]]
  idx2 <- res1[[6L]]
  t2 <- res1[[7L]]
  di <- res1[[8L]]
  beta_h <- matrix(0, p, 1L)
  qcum <- cumsum(c(0L, q))
  v_h <- matrix(0, qcum[nrand + 1L], 1L)
  alpha_h <- rep(0.1, nrand)
  k_init <- min(length(varinit), nrand)
  if (k_init > 0L) alpha_h[seq_len(k_init)] <- varinit[seq_len(k_init)]
  list(
    n = n, p = p, q = q, nrand = nrand,
    x = x, z = z, y = y, del = del, Mi = Mi, idx2 = idx2, t2 = t2, di = di,
    beta_h = beta_h, v_h = v_h, alpha_h = alpha_h, ar1_seq = ar1_seq,
    namesX = namesX, namesRE = namesRE, censor = censor
  )
}

fhl_print_model_header <- function(varfixed, varinit, RandDist, censor, n, formula, mord, dord) {
  result <- list(0)
  names(result)[1] <- "Model"
  sum_init <- sum(abs(varinit))
  if (varfixed == TRUE && sum_init < 1.0000000000000001e-05) {
    print("Results from the Cox model")
    result$Model <- "Cox model"
  } else {
    if (RandDist == "Gamma") {
      print("Results from the gamma frailty model")
      result$Model <- "gamma frailty model"
    }
    if (RandDist == "Normal") {
      print("Results from the log-normal frailty model")
      result$Model <- "log-normal frailty model"
    }
    if (RandDist == "AR1") {
      print("Results from the log-normal frailty model with AR(1)")
      result$Model <- "log-normal frailty model with AR(1)"
    }
  }
  nevent <- sum(censor)
  print("Number of data : ")
  print(n)
  print("Number of event : ")
  print(nevent)
  print("Model for conditional hazard : ")
  result$formula <- formula
  print(formula)
  if (mord == 0 && dord == 0) {
    print("Method : HL(0,0)")
    result$Method <- "HL(0,0)"
  }
  if (mord == 0 && dord == 1) {
    print("Method : HL(0,1)")
    result$Method <- "HL(0,1)"
  }
  if (mord == 0 && dord == 2) {
    print("Method : HL(0,2)")
    result$Method <- "HL(0,2)"
  }
  if (mord == 1 && dord == 0) {
    print("Method : HL(1,0)")
    result$Method <- "HL(1,0)"
  }
  if (mord == 1 && dord == 1) {
    print("Method : HL(1,1)")
    result$Method <- "HL(1,1)"
  }
  if (mord == 1 && dord == 2) {
    print("Method : HL(1,2)")
    result$Method <- "HL(1,2)"
  }
  print("Estimates from the mean model")
  result
}

fhl_format_cox_result <- function(
  fit_c, prep, formula, RandDist, mord, dord, varfixed, varinit, convergence
) {
  res2 <- fit_c[[1L]]
  res3 <- fit_c[[2L]]
  print_i <- fit_c[[3L]]
  print_err <- fit_c[[4L]]
  beta_h <- res2[[11L]]
  v_h <- res2[[12L]]
  alpha_h <- res2[[14L]]
  Hinv <- res2[[16L]]
  se_beta <- res2[[20L]]
  u_h <- if (RandDist == "Gamma") res2[[23L]] else NULL
  Hinv_u <- if (RandDist == "Gamma") res2[[24L]] else NULL
  rho_h <- if (RandDist == "AR1") res2[[24L]] else 0.5

  converged_flag <- is.finite(print_err) && (print_err < convergence)
  names(print_i) <- "iteration : "
  print(print_i)
  names(print_err) <- "convergence : "
  print(print_err)
  if (converged_flag) {
    print("converged")
  } else {
    print("did not converge")
    warning(sprintf(
      "frailtyHL did not converge within %d iterations (final error = %s)",
      as.integer(print_i), format(print_err, digits = 4)
    ))
  }

  result <- fhl_print_model_header(
    varfixed, varinit, RandDist, prep$censor, prep$n, formula, mord, dord
  )
  z_beta <- beta_h / se_beta
  pval <- 2 * pnorm(abs(z_beta), lower.tail = FALSE)
  beta_coeff <- cbind(beta_h, se_beta, z_beta, pval)
  colnames(beta_coeff) <- c("Estimate", "Std. Error", "t-value", "p-value")
  rownames(beta_coeff) <- prep$namesX
  result$FixCoef <- beta_coeff
  print(beta_coeff, 4)

  print("Estimates from the dispersion model")
  se_alpha_h <- res3[[1L]]
  hlike <- -2 * res3[[2L]]
  p1 <- -2 * res3[[3L]]
  p2 <- -2 * res3[[4L]]
  p3 <- -2 * res3[[5L]]
  p0 <- -2 * res3[[6L]]
  p4 <- p3 - (p1 - p2)
  df1 <- res3[[7L]]

  for (i in seq_len(prep$nrand)) {
    if (alpha_h[i] < 1.0000000000000001e-05) {
      alpha_h[i] <- 0
    }
  }
  if (RandDist == "AR1") {
    lam_coeff <- cbind(alpha_h)
    colnames(lam_coeff) <- "Estimate"
  } else {
    lam_coeff <- cbind(alpha_h, se_alpha_h)
    colnames(lam_coeff) <- c("Estimate", "Std. Error")
  }
  rownames(lam_coeff) <- prep$namesRE
  print(lam_coeff, 4)
  result$RandCoef <- lam_coeff
  result$rho <- if (RandDist == "AR1") rho_h else 0
  if (RandDist == "AR1") {
    rho_coeff <- cbind(rho_h)
    print("Estimates for rho in the AR(1) model")
    print(rho_coeff, 4)
  }

  if (mord == 0 && dord <= 1) {
    like_value <- cbind(p0, hlike, p1)
    colnames(like_value) <- c("-2h0", "-2*hp", "-2*p_b,v(hp)")
  } else if (mord == 0 && dord == 2) {
    like_value <- cbind(p0, hlike, p1, p3)
    colnames(like_value) <- c("-2h0", "-2*hp", "-2*p_b,v(hp)", "-2*s_b,v(hp)")
  } else if (mord == 1 && dord <= 1) {
    like_value <- cbind(p0, hlike, p2, p1)
    colnames(like_value) <- c("-2h0", "-2*hp", "-2*p_v(hp)", "-2*p_b,v(hp)")
  } else {
    like_value <- cbind(p0, hlike, p2, p4, p1, p3)
    colnames(like_value) <- c(
      "-2h0", "-2*hp", "-2*p_v(hp)", "-2*s_v(hp)", "-2*p_b,v(hp)", "-2*s_b,v(hp)"
    )
  }
  result$likelihood <- like_value
  result$iter <- print_i
  result$convergence <- if (converged_flag) "converged" else "did not converge"
  names(result$convergence) <- "convergence : "
  print(like_value, 5)

  caic <- p0 + 2 * df1
  n_lam <- nrow(lam_coeff)
  if (varfixed == TRUE) {
    n_lam <- 0
    maic <- hlike + 2 * nrow(beta_coeff) + 2 * n_lam
  } else {
    maic <- p2 + 2 * nrow(beta_coeff) + 2 * n_lam
  }
  raic <- p1 + 2 * n_lam
  if (RandDist == "AR1") {
    disp_coeff <- cbind(alpha_h, rho_h)
    n_disp <- ncol(disp_coeff)
    maic <- p2 + 2 * nrow(beta_coeff) + 2 * n_disp
    raic <- p1 + 2 * n_disp
  }
  if (RandDist == "Gamma" && mord == 1 && dord == 2) {
    maic <- p4 + 2 * nrow(beta_coeff) + 2 * n_lam
    raic <- p3 + 2 * n_lam
  }
  aic <- cbind(caic, maic, raic)
  colnames(aic) <- c("cAIC", "pAIC", "rAIC")
  print(aic, 5)

  result$aic <- aic
  result$v_h <- v_h
  result$Hinv <- Hinv
  result$p <- prep$p
  result$q <- prep$q
  if (RandDist == "Gamma") {
    result$u_h <- u_h
    result$Hinv_u <- Hinv_u
  }
  class(result) <- "frailtyHL"
  result
}

# ------------------------------------------------------------------------------
# Package Unload Hook
# ------------------------------------------------------------------------------
.onUnload <- function(libpath) {
  library.dynam.unload("frailtyHLnew", libpath)
}
