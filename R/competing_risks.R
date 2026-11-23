# ==============================================================================
# Competing Risks Hierarchical-Likelihood Models (CmpRsk, hlike.frailty)
# ==============================================================================
CmpRsk <- function(time, index) {
  if (!is.numeric(time) || !is.numeric(index) || length(time) < 1L || length(time) != length(index)) {
    stop("Invalid time or index")
  }
  if (any(!is.finite(time)) || any(time < 0)) stop("Invalid time")
  if (any(!is.finite(index)) || any(index < 0) || !any(index > 0) || any(index != floor(index))) {
    stop("Invalid index")
  }
  u_idx <- sort(unique(index))
  if (length(u_idx) > 1L && any(diff(u_idx) != 1)) stop("Invalid index")
  temp <- cbind(time, index)
  class(temp) <- "CmpRsk"
  temp
}

hlike.frailty <- function(
  formula, data, inits, order = 1, frailty.cov = "none",
  subHazard = FALSE, alpha = .05, MAX.ITER = 100, TOL = 1E-6
) {
  if (missing(formula) || !inherits(formula, "formula") || length(formula) < 3L) {
    stop("Invalid formula")
  }
  if (!is.numeric(MAX.ITER) || length(MAX.ITER) != 1L || is.na(MAX.ITER) || !is.finite(MAX.ITER) || MAX.ITER < 1) {
    stop("'MAX.ITER' must be >= 1")
  }
  if (!is.numeric(TOL) || length(TOL) != 1L || is.na(TOL) || !is.finite(TOL) || TOL <= 0) {
    stop("'TOL' must be positive finite")
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || is.na(alpha) || !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("'alpha' must be in (0, 1)")
  }
  mf <- model.frame(formula, data)
  if (length(model.offset(mf)) > 0L) stop("'offset()' terms are not supported in hlike.frailty")
  if (length(model.weights(mf)) > 0L) stop("'weights' argument is not supported in hlike.frailty")
  if (!is(model.response(mf), "CmpRsk") | is.na(pmatch("cluster", names(mf))))
    stop("Invalid formula")
  resp <- pmatch("CmpRsk", names(mf))
  clust <- pmatch("cluster", names(mf))
  pred <- names(mf)[setdiff(seq_len(ncol(mf)), c(resp, clust))]
  if (length(pred) < 1L) stop("hlike.frailty requires at least one fixed predictor")
  temp.data <- data.frame(unclass(mf[, resp]), mf[, pred, drop = FALSE], mf[, clust])
  names(temp.data) <- c("time", "index", pred, "cluster")
  sort.data <- temp.data[order(temp.data$time), ]
  N <- nrow(sort.data)
  if (N < 1L) stop("No observations after model framing")
  n <- length(unique(sort.data$cluster))
  m <- max(sort.data$index)
  p <- length(pred)
  if (m < 1) stop("At least one competing event (index >= 1) is required")

  X <- as.matrix(sort.data[, pred, drop = FALSE])
  colnames(X) <- pred
  storage.mode(X) <- "double"
  if (any(!is.finite(sort.data$time)) || any(sort.data$time < 0) ||
      any(sort.data$time == 0 & sort.data$index > 0) ||
      any(!is.finite(sort.data$index)) || any(!is.finite(X))) {
    stop("Invalid or non-finite values in hlike.frailty inputs")
  }
  cl_fac <- droplevels(factor(sort.data$cluster))
  if (nlevels(cl_fac) < 2L)
    stop("Random-effect grouping factor must have at least 2 levels")
  if (!is.numeric(order) || length(order) != 1L || is.na(order) || !(order %in% c(0, 1, 2)))
    stop("'order' must be 0, 1, or 2")
  group_idx <- as.integer(cl_fac)

  if (m == 1 && frailty.cov != "none") stop("Invalid frailty.cov argument")
  if (subHazard && frailty.cov != "none") stop("Invalid frailty.cov argument")
  s <- switch(frailty.cov,
    none = 1L,
    independent = m,
    exchangeable = m + 1L,
    unstructured = sum(1:m),
    stop("Invalid frailty.cov argument")
  )
  if (order == 2 && frailty.cov != "none") {
    stop("order = 2 is only supported when frailty.cov = 'none'")
  }
  if (subHazard) {
    m <- 1L
  }
  if (!all(c("beta", "theta", "v") %in% names(inits))) stop("Invalid initial values")
  if (any(!is.finite(as.double(inits$beta))) || any(!is.finite(as.double(inits$theta))) ||
      any(!is.finite(as.double(inits$v)))) {
    stop("Initial values in 'inits' must be finite")
  }
  if (length(inits$beta) != p * m || min(m, s) != ncol(as.matrix(inits$v)) ||
      nrow(as.matrix(inits$theta)) != min(m, s) || ncol(as.matrix(inits$theta)) != min(m, s) ||
      nrow(as.matrix(inits$v)) != n) {
    stop("Invalid initial values")
  }

  y <- vector("list", m)
  d <- vector("list", m)
  for (k in seq_len(m)) {
    t_k <- sort.data$time[sort.data$index == k]
    y_k <- unique(t_k)
    y[[k]] <- as.double(y_k)
    d[[k]] <- as.double(tabulate(match(t_k, y_k), nbins = length(y_k)))
  }
  cov_type_int <- switch(frailty.cov,
    none = 0L,
    independent = 1L,
    exchangeable = 2L,
    unstructured = 3L
  )
  m_s <- min(m, s)
  beta_init_mat <- matrix(as.double(inits$beta), nrow = p, ncol = m)
  theta_init_mat <- matrix(as.double(inits$theta), nrow = m_s, ncol = m_s)
  v_init_mat <- matrix(as.double(inits$v), nrow = n, ncol = m_s)

  c_fit <- .Call(
    "fhl_fit_competing_risks",
    X, group_idx, as.integer(n),
    as.double(sort.data$time), as.integer(sort.data$index),
    y, d,
    beta_init_mat, theta_init_mat, v_init_mat,
    as.integer(order), as.integer(cov_type_int),
    as.integer(isTRUE(subHazard)),
    as.integer(MAX.ITER), as.double(TOL),
    PACKAGE = "frailtyHLnew"
  )
  beta_est <- c_fit[[1L]]
  theta_est <- c_fit[[2L]]
  v_i <- c_fit[[3L]]
  H <- c_fit[[4L]]
  H0 <- c_fit[[5L]]
  beta.SE <- c_fit[[6L]]
  lambda.0 <- c_fit[[7L]]
  h.1 <- c_fit[[8L]]
  h.p <- c_fit[[9L]]
  A.h.p <- c_fit[[10L]]
  rAIC <- c_fit[[11L]]
  cAIC <- c_fit[[12L]]
  converge <- c_fit[[13L]]
  i <- c_fit[[14L]]

  if (m_s == 1L) {
    v.out <- matrix(v_i[group_idx, 1L], nrow = N, ncol = 1L)
  } else {
    v.out <- v_i[group_idx, , drop = FALSE]
    colnames(v.out) <- as.character(seq_len(m_s))
  }
  rownames(v.out) <- sort.data$cluster

  lower.beta <- c(beta_est) - qnorm(1 - alpha / 2) * beta.SE
  upper.beta <- c(beta_est) + qnorm(1 - alpha / 2) * beta.SE
  beta.CI <- data.frame(
    rep(seq_len(m), each = p), rep(colnames(X), times = m),
    c(beta_est), beta.SE, lower.beta, upper.beta
  )
  colnames(beta.CI) <- c(
    "Type", "Effect", "Estimate", "SE",
    paste0(alpha / 2 * 100, "%"), paste0((1 - alpha / 2) * 100, "%")
  )
  Var.Comp <- paste(
    "Sigma.",
    unlist(sapply(seq_len(m_s), function(q) q:m_s)),
    rep(seq_len(m_s), times = m_s:1),
    sep = ""
  )
  theta.Est <- data.frame(
    Var.Comp,
    Estimate = theta_est[lower.tri(theta_est, diag = TRUE)]
  )

  if (isTRUE(subHazard)) {
    loglik.noFrailty <- crr(sort.data$time, sort.data$index, X)$loglik
  } else {
    loglik.noFrailty <- sum(sapply(seq_len(m), function(k) {
      coxph(Surv(time, index == k) ~ X, data = sort.data)$loglik[2L]
    }))
  }
  if (frailty.cov == "none") {
    psi.hat <- -2 * (loglik.noFrailty - A.h.p)
    theta.pvalue <- ifelse(psi.hat <= 0, 1, 0.5 * pchisq(psi.hat, 1, lower.tail = FALSE))
  } else {
    theta.pvalue <- NULL
  }

  out <- list(
    beta = beta.CI, theta = theta.Est, v = v.out, theta.pvalue = theta.pvalue,
    lambda.0 = lapply(seq_len(m), function(k) {
      matrix(lambda.0[[k]], dimnames = list(Time = y[[k]], Type = k))
    }),
    time = y, h1 = h.1, hp = h.p, Ahp = A.h.p,
    loglik.noFrailty = loglik.noFrailty, rAIC = rAIC, cAIC = cAIC, H = H, H0 = H0,
    converge = converge, iterations = i, call = match.call()
  )
  class(out) <- "hlike"
  print(out$beta); cat("\n"); print(out$theta)
  cat("\n", ifelse(out$converge == 1, "Successfully Converged", "Failed to Converge"), "\n")
  if (out$converge != 1) {
    warning(sprintf("hlike.frailty did not converge within MAX.ITER = %d (converge = %d)",
                    as.integer(MAX.ITER), as.integer(out$converge)))
  }
  out
}

print.hlike <- function(x, ...) {
  print(x$beta); cat("\n"); print(x$theta)
  cat("\n", ifelse(x$converge == 1, "Successfully Converged", "Failed to Converge"), "\n")
  invisible(x)
}

summary.hlike <- function(object, ...) {
  print(object$beta); cat("\n"); print(object$theta)
  cat("\n", ifelse(object$converge == 1, "Successfully Converged", "Failed to Converge"), "\n")
  invisible(object)
}
