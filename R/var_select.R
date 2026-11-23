# ==============================================================================
# Variable Selection for Cox Frailty Models (frailty.vs, shared helpers)
# ==============================================================================

fhl_normalize_penalty <- function(penalty) {
  if (missing(penalty) || !is.character(penalty) || length(penalty) != 1L || is.na(penalty)) {
    stop("'penalty' must be one of \"LASSO\", \"SCAD\", or \"HL\"")
  }
  pen <- tolower(penalty)
  if (!(pen %in% c("lasso", "scad", "hl"))) {
    stop("'penalty' must be one of \"LASSO\", \"SCAD\", or \"HL\"")
  }
  pen
}

fhl_validate_frailty_vs_args <- function(
  formula, model, penalty, varfixed, varinit, tun1, tun2
) {
  if (missing(formula) || !inherits(formula, "formula") || length(formula) < 3L) {
    stop("'formula' must be a two-sided formula")
  }
  if (!length(expandSlash(findbars(formula[[3L]])))) {
    stop("No random effects terms specified in formula")
  }
  if (missing(model) || !is.character(model) || length(model) != 1L || is.na(model) ||
      !identical(model, "lognorm")) {
    stop("'model' must be \"lognorm\"")
  }
  pen <- fhl_normalize_penalty(penalty)
  if (!is.logical(varfixed) || length(varfixed) != 1L || is.na(varfixed)) {
    stop("'varfixed' must be TRUE or FALSE")
  }
  if (!is.numeric(varinit) || length(varinit) < 1L || any(!is.finite(varinit)) ||
      any(varinit < 0)) {
    stop("'varinit' must contain finite non-negative numeric values")
  }
  if (is.null(tun1) || !is.numeric(tun1) || length(tun1) < 1L || any(!is.finite(tun1))) {
    stop("Tuning parameter 'tun1' must be a finite numeric vector")
  }
  if (pen %in% c("lasso", "scad")) {
    if (any(tun1 < 0)) {
      stop("Tuning parameter 'tun1' must be non-negative")
    }
    if (!is.null(tun2) && (!is.numeric(tun2) || length(tun2) < 1L || any(!is.finite(tun2)))) {
      stop("Tuning parameter 'tun2' must be NULL or finite numeric")
    }
  } else {
    if (any(tun1 <= 0)) {
      stop("Shape parameter 'tun1' must be positive when penalty = 'HL'")
    }
    if (is.null(tun2)) {
      stop("Tuning parameter 'tun2' must not be NULL when penalty = 'HL'")
    }
    if (!is.numeric(tun2) || length(tun2) < 1L || any(!is.finite(tun2)) || any(tun2 <= 0)) {
      stop("Tuning parameter 'tun2' must be a positive finite numeric vector when penalty = 'HL'")
    }
  }
  pen
}

fhl_vs_prepare_data <- function(mc, formula, caller_env, build_risk = TRUE) {
  indx <- match(c("formula", "data"), names(mc), nomatch = 0L)
  if (indx[1L] == 0L)
    stop("A formula argument is required")
  temp <- mc[c(1L, indx)]
  temp[[1L]] <- as.name("model.frame")
  temp$formula <- terms(subbars(formula), c("strata", "cluster"))
  m <- eval(temp, caller_env)
  mt_sp <- attr(m, "terms")
  if (!is.null(attr(mt_sp, "specials")$strata) || !is.null(attr(mt_sp, "specials")$cluster))
    stop("'strata()' and 'cluster()' terms are not supported in frailty.vs")
  Y <- model.extract(m, "response")
  if (!inherits(Y, "Surv") || !identical(attr(Y, "type"), "right"))
    stop("frailty.vs only supports right-censored Surv(time, status) responses")
  fr <- FrailtyFrames(mc, formula, contrasts = NULL, eval_env = caller_env)
  if (length(fr$off) > 0L)
    stop("'offset()' terms are not supported in frailty.vs")
  if (length(fr$wts) > 0L)
    stop("'weights' argument is not supported in frailty.vs")
  nc_y <- ncol(Y)
  if (is.null(nc_y) || nc_y != 2L || identical(attr(Y, "type"), "counting"))
    stop("frailty.vs only supports right-censored Surv(time, status) responses")
  has_icpt <- length(fr$fixef) > 0L && identical(names(fr$fixef)[1L], "(Intercept)")
  namesX <- if (has_icpt) names(fr$fixef)[-1L] else names(fr$fixef)
  FL <- HGLMFactorList(formula, fr, 0L, 0L, sparse = TRUE)
  y <- matrix(Y[, 1L], nrow(Y), 1L)
  x <- if (has_icpt) fr$X[, -1L, drop = FALSE] else fr$X
  z <- FL$Design
  n <- nrow(x)
  p <- ncol(x)
  if (p < 1L)
    stop("frailty.vs requires at least one fixed covariate in 'formula'")
  if (!has_icpt && qr(x)$rank < p)
    stop("Fixed-effect design is rank deficient without intercept; the Cox baseline hazard absorbs the intercept in frailty.vs")
  nrand <- length(z)
  q <- vapply(z, ncol, integer(1L))
  if (any(q < 2L))
    stop("Random-effect grouping factor must have at least 2 levels")
  del <- matrix(0, n, 1L)
  del[, 1L] <- Y[, 2L]
  fhl_validate_surv_data(y, del, x, fn_name = "frailty.vs")
  if (!isTRUE(build_risk)) {
    return(list(n = n, p = p, q = q, nrand = nrand, z_names = lapply(z, colnames)))
  }
  res1 <- fhl_make_data(y, x, del, z)
  list(
    y = as.numeric(res1[[1L]]),
    x = as.matrix(res1[[2L]]),
    del = as.numeric(res1[[3L]]),
    z = res1[[4L]],
    Mi = res1[[5L]],
    idx2 = res1[[6L]],
    t2 = as.numeric(res1[[7L]]),
    di = as.numeric(res1[[8L]]),
    p = p, q = q, nrand = nrand,
    namesX = namesX,
    res1 = res1
  )
}

fhl_vs_as_z_arg <- function(Z) {
  lapply(Z, function(z_i) {
    if (inherits(z_i, "dgCMatrix") || !is.null(attr(z_i, "fhl_group"))) {
      return(z_i)
    }
    m <- as.matrix(z_i)
    storage.mode(m) <- "double"
    m
  })
}

fhl_vs_grid_vectors <- function(pen_type, tun1, tun2) {
  if (pen_type == 2L) {
    list(tunset1 = as.double(rep(tun1, each = length(tun2))),
         tunset2 = as.double(rep(tun2, length(tun1))))
  } else {
    list(tunset1 = as.double(tun1), tunset2 = NULL)
  }
}

fhl_vs_print_result <- function(result, model, penalty, formula, namesX, res1) {
  print("Result of variable selection in frailty model")
  print("==Fitted model==")
  print(paste("model :", model))
  print(paste("penalty :", penalty))
  print("formula :")
  print(formula)
  cn <- if (result$convergence == 1) "converge" else "did not converge"
  print(cn)
  if (!isTRUE(result$convergence == 1) || !is.finite(result$BIC) || result$BIC >= 99999.9) {
    warning(sprintf(
      "frailty.vs did not produce a converged/well-conditioned solution (convergence = %s, BIC = %s)",
      as.character(result$convergence), format(result$BIC)
    ))
  }
  print("==Fixed coefficients==")
  print_mat <- cbind(result$beta, result$se)
  rownames(print_mat) <- namesX
  colnames(print_mat) <- c("Estimate", "Std. Error")
  print(print_mat)
  print("==Dispersion parameter==")
  print(result$alpha)
  print("==Tuning parameter==")
  print(result$tuning_parameter)
  print("==BIC==")
  print(as.numeric(result$BIC))
  result$res1 <- res1
  result
}

fhl_vs_prepare_inits <- function(prep, B, v, formula, has_data, data, caller_env) {
  if (identical(B, "null")) B <- NULL
  if (identical(v, "null")) v <- NULL
  if (is.null(v)) {
    v <- lapply(prep$q, numeric)
  } else {
    if (!is.list(v) || length(v) != prep$nrand ||
        any(vapply(seq_len(prep$nrand), function(i) {
          vi <- v[[i]]
          !is.numeric(vi) || length(vi) != prep$q[i] || any(!is.finite(vi))
        }, logical(1L)))) {
      stop("'v' must be NULL or a list of nrand finite numeric vectors with lengths matching the random-effect levels")
    }
    v <- lapply(v, as.double)
  }
  if (is.null(B)) {
    B_init <- tryCatch({
      suppressWarnings(invisible(utils::capture.output(
        fit0 <- if (!isTRUE(has_data)) {
          eval(substitute(frailtyHL(f), list(f = formula)), caller_env)
        } else {
          frailtyHL(formula, data = data)
        }
      )))
      as.numeric(fit0$FixCoef[, 1L])
    }, error = function(e) rep(0, prep$p))
    if (length(B_init) != prep$p || any(!is.finite(B_init))) {
      B_init <- rep(0, prep$p)
    }
    B <- B_init
  } else if (!is.numeric(B) || length(B) != prep$p || any(!is.finite(B))) {
    stop(sprintf("'B' must be a finite numeric vector of length %d", prep$p))
  }
  list(B = as.double(B), v = v)
}

fhl_vs_check_2re_corr <- function(alpha, z_names) {
  if (length(z_names) != 2L || is.null(z_names[[1L]]) || is.null(z_names[[2L]]) ||
      !identical(z_names[[1L]], z_names[[2L]])) {
    stop("correlated 2-RE requires the same grouping factor/levels")
  }
  if (!is.numeric(alpha) || length(alpha) != 3L || any(!is.finite(alpha))) {
    stop("'alpha' must be a finite numeric vector of length 3 for correlated 2-RE")
  }
  if (alpha[1L] <= 0 || alpha[2L] <= 0 || abs(alpha[3L]) >= sqrt(alpha[1L]) * sqrt(alpha[2L])) {
    stop("initial 2x2 random-effect covariance S0 must be symmetric positive definite")
  }
}

frailty.vs <- function(
  formula, model, penalty, data, B = NULL, v = NULL, alpha = NULL,
  tun1 = NULL, tun2 = NULL, varfixed = FALSE, varinit = 0.1
) {
  user_env <- parent.frame()
  has_data <- !missing(data)
  penalty <- fhl_validate_frailty_vs_args(formula, model, penalty, varfixed, varinit, tun1, tun2)
  info <- fhl_vs_prepare_data(mc = match.call(), formula = formula, caller_env = user_env, build_risk = FALSE)
  if (!is.null(alpha)) {
    if (!is.numeric(alpha) || any(!is.finite(alpha))) {
      stop("'alpha' must contain finite numeric values")
    }
    if (info$nrand == 1L && length(alpha) != 1L) {
      stop("'alpha' must have length 1 for a 1-random-effect model")
    }
    if (info$nrand == 2L) {
      if (!(length(alpha) %in% c(2L, 3L))) {
        stop("'alpha' must have length 2 (independent) or 3 (correlated) for a 2-random-effect model")
      }
      if (isTRUE(varfixed) && length(alpha) == 3L) {
        stop("'varfixed = TRUE' is not supported for correlated 2-RE ('alpha' of length 3)")
      }
    }
    if (info$nrand >= 3L && length(alpha) != info$nrand) {
      stop(sprintf("'alpha' must have length %d for a %d-random-effect model", info$nrand, info$nrand))
    }
  }
  if (isTRUE(varfixed) || length(alpha) <= info$nrand) {
    if (has_data) {
      return(frailty1.vs(
        formula = formula, model = model, penalty = penalty, data = data,
        B = B, v = v, alpha = alpha, tun1 = tun1, tun2 = tun2,
        varfixed = varfixed, varinit = varinit
      ))
    }
    return(frailty1.vs(
      formula = formula, model = model, penalty = penalty,
      B = B, v = v, alpha = alpha, tun1 = tun1, tun2 = tun2,
      varfixed = varfixed, varinit = varinit, eval_env = user_env
    ))
  }
  fhl_vs_check_2re_corr(alpha, info$z_names)
  if (has_data) {
    return(frailty2.vs(
      formula = formula, model = model, penalty = penalty, data = data,
      B = B, v = v, alpha = alpha, tun1 = tun1, tun2 = tun2
    ))
  }
  frailty2.vs(
    formula = formula, model = model, penalty = penalty,
    B = B, v = v, alpha = alpha, tun1 = tun1, tun2 = tun2,
    eval_env = user_env
  )
}
