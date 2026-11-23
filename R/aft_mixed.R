# ==============================================================================
# Multilevel AFT Frailty Models (mlmfit Dispatcher, mlmfit1, mlmfit2)
# ==============================================================================
mlmfit <- function(jm1, data, weights, subset, na.action, Maxiter = 200, mord = 0, dord = 1) {
  Call <- match.call()
  indx <- match(c("data", "weights", "subset", "na.action"),
    names(Call),
    nomatch = 0L
  )
  if (missing(jm1) || !is.list(jm1) || length(jm1) < 3L || !inherits(jm1[[3L]], "formula")) {
    stop("'jm1' must be a jointmodeling() specification containing a model formula")
  }
  if (length(jm1) >= 5L && !is.null(jm1[[5L]])) {
    stop("'Offset' in jointmodeling() is not supported in mlmfit")
  }
  if (indx[1L] == 0L || missing(data))
    stop("A 'data' argument is required in mlmfit")
  if (!missing(weights) && !is.null(Call$weights))
    stop("'weights' argument is not supported in mlmfit")
  if (!missing(subset) && !is.null(Call$subset))
    stop("'subset' argument is not supported in mlmfit")
  if (!missing(na.action) && !is.null(Call$na.action))
    stop("'na.action' argument is not supported in mlmfit")
  if (!is.numeric(Maxiter) || length(Maxiter) != 1L || is.na(Maxiter) || !is.finite(Maxiter) ||
      Maxiter < 1 || Maxiter > .Machine$integer.max || Maxiter != floor(Maxiter)) {
    stop("'Maxiter' must be a finite integer >= 1 and <= .Machine$integer.max")
  }
  if (!is.numeric(mord) || length(mord) != 1L || is.na(mord) || !(mord %in% c(0, 1))) {
    stop("Only mord = 0 (HL(0,1)) and mord = 1 (HL(1,1)) are supported in mlmfit")
  }
  if (!is.numeric(dord) || length(dord) != 1L || is.na(dord) || dord != 1) {
    stop("Only dord = 1 is supported in mlmfit")
  }
  mord <- as.integer(mord)
  dord <- as.integer(dord)
  if (mord == 1L) {
    if (!is.null(jm1[[1L]]) && jm1[[1L]] != "mean")
      stop("HL(1,1) AFT requires jointmodeling Model = 'mean'")
    if (!is.null(jm1[[2L]]) && jm1[[2L]] != "log")
      stop("HL(1,1) Log-Normal AFT requires Link = 'log'")
    if (length(jm1) >= 4L && !is.null(jm1[[4L]]) && !(jm1[[4L]] %in% c("gaussian", "Normal", "normal")))
      stop("HL(1,1) Log-Normal AFT supports only gaussian random effects")
  }
  Maxiter <- as.integer(Maxiter)
  bars <- expandSlash(findbars(jm1[[3L]][[3L]]))
  kkk <- length(bars)
  data_surv <- data
  prep <- fhl_mlm_prepare(jm1[[3L]], data_surv, bars, allow_all_censored = (mord == 1L))
  if (kkk == 1L) {
    return(mlmfit1(jm1 = jm1, data_surv = data_surv, Maxiter = Maxiter, bars = bars, prep = prep, mord = mord, dord = dord))
  }
  if (kkk == 2L) {
    return(mlmfit2(jm1 = jm1, data_surv = data_surv, Maxiter = Maxiter, bars = bars, prep = prep, mord = mord, dord = dord))
  }
  stop("mlmfit supports 1 or 2 random effects")
}

# Validate that AFT response is right-censored Surv(time, status) only
fhl_validate_mlm_response <- function(Y) {
  nc_y <- ncol(Y)
  stype <- attr(Y, "type")
  if (isTRUE(nc_y == 3L) || identical(stype, "counting")) {
    stop("Left-truncated/counting-process Surv(entry, exit, status) data are not supported in mlmfit (right censoring only)")
  }
  if (is.null(nc_y) || nc_y != 2L || (!is.null(stype) && !identical(stype, "right"))) {
    stop("mlmfit only supports right-censored Surv(time, status) responses")
  }
  invisible(TRUE)
}

# Reject NaN or Inf values across model-frame columns before na.action omits NaNs (NA still uses na.action)
fhl_validate_mf_nonfinite <- function(frame_form, data, special = c("strata", "cluster")) {
  mf_pass <- model.frame(terms(frame_form, special), data = data, na.action = stats::na.pass)
  for (nm in names(mf_pass)) {
    col <- mf_pass[[nm]]
    if (is.matrix(col)) {
      if (is.numeric(col) && (any(is.nan(col)) || any(is.infinite(col)))) {
        stop("Non-finite values (NaN/Inf) are not allowed in mlmfit inputs")
      }
    } else if (is.numeric(col)) {
      if (any(is.nan(col)) || any(is.infinite(col))) {
        stop("Non-finite values (NaN/Inf) are not allowed in mlmfit inputs")
      }
    }
  }
  invisible(TRUE)
}

# Unified single-pass model-frame & random-effect preparation for 1-RE / 2-RE AFT models
fhl_mlm_prepare <- function(formula, data, bars, allow_all_censored = FALSE) {
  special <- c("strata", "cluster")
  frame_form <- subbars(formula)
  fixed_form <- nobars(formula)
  if (!inherits(fixed_form, "formula")) {
    fixed_form <- stats::as.formula(substitute(foo ~ 1, list(foo = fixed_form)), env = environment(formula))
  }
  environment(fixed_form) <- environment(frame_form) <- environment(formula)
  fhl_validate_mf_nonfinite(frame_form, data, special = special)
  mf <- model.frame(terms(frame_form, special), data = data, drop.unused.levels = TRUE)
  if (length(model.offset(mf)) > 0L) {
    stop("'offset()' terms are not supported in mlmfit")
  }
  if (length(model.weights(mf)) > 0L) {
    stop("'weights' are not supported in mlmfit")
  }
  Y <- model.response(mf, "any")
  fhl_validate_mlm_response(Y)
  nrand <- length(bars)
  if (!(nrand %in% c(1L, 2L))) {
    stop("mlmfit supports 1 or 2 random effects")
  }
  y_time <- as.double(Y[, 1L])
  delta_vec <- as.double(Y[, 2L])
  mt_fix <- terms(fixed_form, special, data = mf)
  sp_fix <- attr(mt_fix, "specials")
  if (!is.null(sp_fix$cluster) || !is.null(sp_fix$strata)) {
    stop("'cluster()' and 'strata()' terms are not supported in the fixed part of mlmfit")
  }
  x_mat <- if (!is.empty.model(mt_fix)) model.matrix(mt_fix, mf) else matrix(0.0, nrow(mf), 0L)
  if (ncol(x_mat) < 1L) {
    stop("mlmfit requires at least one fixed-effect predictor or intercept (pp >= 1)")
  }
  dimnames(x_mat) <- NULL
  storage.mode(x_mat) <- "double"
  if (allow_all_censored) {
    if (length(y_time) < 1L || any(!is.finite(y_time)) || any(y_time <= 0))
      stop("Survival times must be strictly positive and finite in mlmfit")
    if (any(!is.finite(delta_vec)) || any(!(delta_vec %in% c(0, 1))))
      stop("Event indicator must be binary (0 or 1) in mlmfit")
    if (any(!is.finite(x_mat)))
      stop("Non-finite values are not allowed in fixed-effect design matrix in mlmfit")
  } else {
    fhl_validate_surv_data(y_time, delta_vec, x_mat, fn_name = "mlmfit", require_pos_time = TRUE)
  }

  is_intercept_only <- TRUE
  for (b in bars) {
    if (!is.numeric(b[[2L]]) || length(b[[2L]]) != 1L || b[[2L]] != 1) {
      is_intercept_only <- FALSE
      break
    }
  }

  if (is_intercept_only) {
    group_list <- vector("list", nrand)
    qq_vec <- integer(nrand)
    form_env <- environment(formula)
    for (k in seq_len(nrand)) {
      fac <- fhl_eval_group_factor(bars[[k]][[3L]], mf, form_env)
      group_list[[k]] <- as.integer(fac)
      qq_vec[k] <- nlevels(fac)
    }
    if (any(qq_vec < 2L)) {
      stop("Random-effect grouping factor must have at least 2 levels")
    }
    return(list(
      y = log(as.double(y_time)),
      x_mat = x_mat,
      delta = delta_vec,
      intercept_only = TRUE,
      re_list = group_list,
      qq_vec = qq_vec
    ))
  }

  FL <- HGLMFactorList(formula, list(mf = mf), 0L, 0L)
  z_list <- vector("list", nrand)
  qq_vec <- integer(nrand)
  for (k in seq_len(nrand)) {
    zi_mat <- as.matrix(FL$Design[[k]])
    if (nrow(zi_mat) != nrow(mf)) {
      stop("Random-effect design matrix row count does not match number of observations")
    }
    if (any(!is.finite(zi_mat))) {
      stop("Non-finite values (NA/NaN/Inf) are not allowed in random-effect design matrix in mlmfit")
    }
    storage.mode(zi_mat) <- "double"
    z_list[[k]] <- zi_mat
    qq_vec[k] <- ncol(zi_mat)
  }
  if (any(qq_vec < 2L)) {
    stop("Random-effect grouping factor must have at least 2 levels")
  }
  list(
    y = log(as.double(y_time)),
    x_mat = x_mat,
    delta = delta_vec,
    intercept_only = FALSE,
    re_list = z_list,
    qq_vec = qq_vec
  )
}

fhl_format_aft_result <- function(beta_h, se_beh, v_h, d_est, d_names, i, convergence) {
  t_value <- beta_h / se_beh
  p_value <- 2 * (1 - pnorm(abs(t_value)))
  converged_flag <- is.finite(convergence) && (convergence <= 1e-6)

  print("iterations : ")
  print(i)
  print("convergence : ")
  print(convergence)
  if (converged_flag) {
    print("converged")
  } else {
    print("did not converge")
    warning(sprintf("mlmfit did not converge within %d iterations (final error = %s)",
                    as.integer(i), format(convergence, digits = 4)))
  }

  F.Est <- cbind(beta_h, se_beh, t_value, p_value)
  D.Est <- d_est
  V.Est <- as.vector(v_h)
  ress1 <- list(
    F.Est = F.Est,
    D.Est = D.Est,
    V.Est = V.Est,
    iter = i,
    convergence = convergence,
    converged = converged_flag
  )

  print("Estimates for fixed effects")
  print_mat1 <- ress1$F.Est
  colnames(print_mat1) <- c("Estimate", "Std. Error", "t_value", "p_value")
  print(round(print_mat1, digits = 5))
  print("Estimates for dispersion parameters")
  print_mat2 <- ress1$D.Est
  colnames(print_mat2) <- d_names
  print(round(print_mat2, digits = 5))
  ress1
}

fhl_mlmfit_hl11_dispatch <- function(prep, nrand, Maxiter) {
  pp <- ncol(prep$x_mat)
  qq <- sum(prep$qq_vec)
  d_names <- if (nrand == 1L) c("alpha_h", "phi_h") else c("alpha1_h", "alpha2_h", "phi_h")

  make_fail <- function(reason) {
    warning(reason)
    se_na <- rep(NA_real_, pp)
    F.Est <- cbind(beta_h = se_na, se_beh = se_na, t_value = se_na, p_value = se_na)
    D.Est <- if (nrand == 1L) cbind(alpha_h = NA_real_, phi_h = NA_real_) else cbind(alpha1_h = NA_real_, alpha2_h = NA_real_, phi_h = NA_real_)
    list(
      F.Est = F.Est, D.Est = D.Est, V.Est = rep(NA_real_, qq),
      Method = "HL(1,1)", mord = 1L, dord = 1L,
      iter = 0L, convergence = NA_real_, converged = FALSE, status = "NOT_CONVERGED"
    )
  }

  if (sum(prep$delta) == 0) {
    return(make_fail("All observations are right-censored (sum(delta) == 0); location is unidentifiable."))
  }
  if (qr(prep$x_mat)$rank < pp) {
    return(make_fail("Fixed-effect design matrix X is rank-deficient."))
  }
  if (sum(prep$delta) <= pp) {
    warning(sprintf("Very few uncensored events (%d <= p = %d); estimates may hit parameter boundary or fail.",
                    as.integer(sum(prep$delta)), pp))
  }

  if (nrand == 1L) {
    qq1 <- prep$qq_vec[1L]
    c_fit <- .Call(
      "fhl_fit_aft_1re",
      prep$y, prep$x_mat, prep$re_list[[1L]], as.integer(qq1), prep$delta,
      numeric(pp), numeric(qq1), c(as.integer(Maxiter), 1L),
      PACKAGE = "frailtyHLnew"
    )
  } else {
    qq1 <- prep$qq_vec[1L]
    qq2 <- prep$qq_vec[2L]
    c_fit <- .Call(
      "fhl_fit_aft_2re",
      prep$y, prep$x_mat,
      prep$re_list[[1L]], as.integer(qq1),
      prep$re_list[[2L]], as.integer(qq2),
      prep$delta, numeric(pp), numeric(qq1 + qq2), c(as.integer(Maxiter), 1L),
      PACKAGE = "frailtyHLnew"
    )
  }

  beta_h <- as.numeric(c_fit[[1L]])
  se_beh <- as.numeric(c_fit[[2L]])
  v_h <- as.vector(c_fit[[3L]])
  alpha_vec <- as.numeric(c_fit[[4L]])
  phi_h <- as.numeric(c_fit[[5L]])
  iter_done <- as.integer(c_fit[[6L]])
  conv_val <- as.numeric(c_fit[[7L]])
  conv_flag <- isTRUE(c_fit[[8L]])
  status_str <- as.character(c_fit[[9L]])
  reason_str <- as.character(c_fit[[10L]])

  if (!conv_flag) {
    warning(sprintf("HL(1,1) did not converge: %s (iter = %d, convergence = %s)",
                    reason_str, iter_done, format(conv_val, digits = 4)))
    t_value <- rep(NA_real_, pp)
    p_value <- rep(NA_real_, pp)
  } else {
    t_value <- beta_h / se_beh
    p_value <- 2.0 * (1.0 - pnorm(abs(t_value)))
  }

  F.Est <- cbind(beta_h = beta_h, se_beh = se_beh, t_value = t_value, p_value = p_value)
  D.Est <- if (nrand == 1L) {
    cbind(alpha_h = alpha_vec[1L], phi_h = phi_h)
  } else {
    cbind(alpha1_h = alpha_vec[1L], alpha2_h = alpha_vec[2L], phi_h = phi_h)
  }

  print("Method : HL(1,1)")
  print("iterations : ")
  print(iter_done)
  print("convergence : ")
  print(conv_val)
  print(paste("status :", status_str))
  print("Estimates for fixed effects")
  pm1 <- F.Est
  colnames(pm1) <- c("Estimate", "Std. Error", "t_value", "p_value")
  print(round(pm1, digits = 5))
  print("Estimates for dispersion parameters")
  pm2 <- D.Est
  colnames(pm2) <- d_names
  print(round(pm2, digits = 5))

  list(
    F.Est = F.Est, D.Est = D.Est, V.Est = v_h,
    Method = "HL(1,1)", mord = 1L, dord = 1L,
    iter = iter_done, convergence = conv_val,
    converged = conv_flag, status = status_str
  )
}

# ------------------------------------------------------------------------------
# Single-RE AFT Frailty Solver (mlmfit1)
# ------------------------------------------------------------------------------
mlmfit1 <- function(jm1, data_surv, Maxiter = 200, bars = NULL, prep = NULL, mord = 0, dord = 1) {
  if (is.null(bars)) bars <- expandSlash(findbars(jm1[[3L]][[3L]]))
  if (is.null(prep)) prep <- fhl_mlm_prepare(jm1[[3L]], data_surv, bars, allow_all_censored = (mord == 1L))
  if (as.integer(mord) == 1L) {
    return(fhl_mlmfit_hl11_dispatch(prep, nrand = 1L, Maxiter = as.integer(Maxiter)))
  }
  pp <- ncol(prep$x_mat)
  qq <- prep$qq_vec[1L]
  c_fit <- .Call(
    "fhl_fit_aft_1re",
    prep$y, prep$x_mat, prep$re_list[[1L]], as.integer(qq), prep$delta,
    numeric(pp), numeric(qq), as.integer(Maxiter),
    PACKAGE = "frailtyHLnew"
  )
  fhl_format_aft_result(
    beta_h = c_fit[[1L]],
    se_beh = c_fit[[2L]],
    v_h = c_fit[[3L]],
    d_est = cbind(alpha_h = c_fit[[4L]], phi_h = c_fit[[5L]]),
    d_names = c("alpha_h", "phi_h"),
    i = c_fit[[6L]],
    convergence = c_fit[[7L]]
  )
}

# ------------------------------------------------------------------------------
# Two-RE AFT Frailty Solver (mlmfit2)
# ------------------------------------------------------------------------------
mlmfit2 <- function(jm1, data_surv, Maxiter = 300, bars = NULL, prep = NULL, mord = 0, dord = 1) {
  if (is.null(bars)) bars <- expandSlash(findbars(jm1[[3L]][[3L]]))
  if (is.null(prep)) prep <- fhl_mlm_prepare(jm1[[3L]], data_surv, bars, allow_all_censored = (mord == 1L))
  if (as.integer(mord) == 1L) {
    return(fhl_mlmfit_hl11_dispatch(prep, nrand = 2L, Maxiter = as.integer(Maxiter)))
  }
  pp <- ncol(prep$x_mat)
  qq1 <- prep$qq_vec[1L]
  qq2 <- prep$qq_vec[2L]
  c_fit <- .Call(
    "fhl_fit_aft_2re",
    prep$y, prep$x_mat,
    prep$re_list[[1L]], as.integer(qq1),
    prep$re_list[[2L]], as.integer(qq2),
    prep$delta, numeric(pp), numeric(qq1 + qq2), as.integer(Maxiter),
    PACKAGE = "frailtyHLnew"
  )
  fhl_format_aft_result(
    beta_h = c_fit[[1L]],
    se_beh = c_fit[[2L]],
    v_h = c_fit[[3L]],
    d_est = cbind(alpha1_h = c_fit[[4L]], alpha2_h = c_fit[[5L]], phi_h = c_fit[[6L]]),
    d_names = c("alpha1_h", "alpha2_h", "phi_h"),
    i = c_fit[[7L]],
    convergence = c_fit[[8L]]
  )
}
