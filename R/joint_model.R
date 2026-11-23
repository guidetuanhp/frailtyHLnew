# ==============================================================================
# Joint Modeling Engine (jointmodeling, jmfit, jmfit2, jmfit3, jmfit_FM)
# ==============================================================================
jointmodeling <- function(
  Model = "mean", RespDist = "gaussian", Link = NULL, LinPred = "constant", RandDist = NULL,
  Offset = NULL
) {
  if (!is.character(Model) || length(Model) != 1L || is.na(Model) || Model != "mean") {
    stop("Only Model = \"mean\" is supported in jointmodeling()")
  }
  if (!is.character(RespDist) || length(RespDist) != 1L || is.na(RespDist) ||
      !(RespDist %in% c("gaussian", "AFT", "FM"))) {
    stop("Unsupported 'RespDist' in jointmodeling(); expected \"gaussian\", \"AFT\", or \"FM\"")
  }
  if (is.null(Link)) {
    Link <- if (RespDist == "gaussian") "identity" else "log"
  }
  if (!is.character(Link) || length(Link) != 1L || is.na(Link) || !(Link %in% c("identity", "log"))) {
    stop("Unsupported 'Link' in jointmodeling(); expected \"identity\" or \"log\"")
  }
  if (RespDist == "gaussian" && Link != "identity") {
    stop("For RespDist = \"gaussian\", Link must be \"identity\"")
  }
  if (RespDist %in% c("AFT", "FM") && Link != "log") {
    stop("For RespDist = \"AFT\" or \"FM\", Link must be \"log\"")
  }
  if (!is.null(RandDist) && (!is.character(RandDist) || length(RandDist) != 1L || is.na(RandDist) ||
      RandDist != "gaussian")) {
    stop("Unsupported 'RandDist' in jointmodeling(); expected NULL or \"gaussian\"")
  }
  out <- list(Model, Link, LinPred, RandDist, Offset)
  attr(out, "RespDist") <- RespDist
  out
}

fhl_validate_jm_maxiter <- function(Maxiter) {
  if (!is.numeric(Maxiter) || length(Maxiter) != 1L || is.na(Maxiter) ||
      !is.finite(Maxiter) || Maxiter < 1 || Maxiter > .Machine$integer.max ||
      Maxiter != floor(Maxiter)) {
    stop("'Maxiter' must be a finite integer between 1 and .Machine$integer.max")
  }
  as.integer(Maxiter)
}

fhl_validate_jm_component <- function(comp, name = "jm", allowed_resp = NULL) {
  if (!is.list(comp) || length(comp) < 3L || !inherits(comp[[3L]], "formula")) {
    stop(sprintf("Invalid '%s' jointmodeling specification", name))
  }
  if (!identical(comp[[1L]], "mean")) {
    stop(sprintf("Only Model = \"mean\" is supported for '%s'", name))
  }
  if (!is.null(comp[[2L]]) && !(comp[[2L]] %in% c("identity", "log"))) {
    stop(sprintf("Unsupported 'Link' in '%s'", name))
  }
  if (length(comp) >= 4L && !is.null(comp[[4L]]) && !identical(comp[[4L]], "gaussian")) {
    stop(sprintf("Unsupported 'RandDist' in '%s'; expected NULL or \"gaussian\"", name))
  }
  if (length(comp) >= 5L && !is.null(comp[[5L]])) {
    stop("'Offset' in jointmodeling() is not supported in jmfit")
  }
  rd <- attr(comp, "RespDist", exact = TRUE)
  if (!is.null(rd) && !is.null(allowed_resp) && !(rd %in% allowed_resp)) {
    stop(sprintf("Unsupported RespDist '%s' for '%s' in this joint model", rd, name))
  }
  invisible(TRUE)
}

fhl_align_cluster_Z <- function(Z_ref, Z_other, label1 = "model 1", label2 = "model 2") {
  c1 <- colnames(Z_ref)
  c2 <- colnames(Z_other)
  if (is.null(c1) || is.null(c2)) {
    stop(sprintf("Missing cluster column names between %s and %s", label1, label2))
  }
  if (anyDuplicated(c1) || anyDuplicated(c2)) {
    stop(sprintf("Duplicated random-effect cluster column names between %s and %s (random slopes are not supported in jmfit)", label1, label2))
  }
  if (length(c1) != length(c2) || !setequal(c1, c2)) {
    missing_in_2 <- setdiff(c1, c2)
    missing_in_1 <- setdiff(c2, c1)
    stop(sprintf(
      "Random-effect cluster IDs do not match between %s and %s (missing in %s: %d; missing in %s: %d)",
      label1, label2, label2, length(missing_in_2), label1, length(missing_in_1)
    ))
  }
  if (identical(c1, c2)) {
    return(Z_other)
  }
  Z_other[, match(c1, c2), drop = FALSE]
}

# ------------------------------------------------------------------------------
# Joint Modeling Dispatcher (jmfit)
# ------------------------------------------------------------------------------
jmfit <- function(jm, data, jm2 = NULL, data2 = NULL, Maxiter = 200) {
  Maxiter <- fhl_validate_jm_maxiter(Maxiter)
  if (is.null(jm2)) {
    kkk <- length(jm)
    if (kkk == 2L) return(jmfit2(jm[[1L]], jm[[2L]], data[[1L]], data[[2L]], Maxiter))
    if (kkk == 3L) return(jmfit3(jm[[1L]], jm[[2L]], jm[[3L]], data[[1L]], data[[2L]], Maxiter))
    stop("Unsupported number of joint-modeling components in 'jm'")
  }
  if (is.data.frame(data) && is.list(jm2) && length(jm2) >= 3L && inherits(jm2[[3L]], "formula")) {
    return(jmfit_FM(jm, jm2, data, data2, Maxiter))
  }
  jmfit_FM(jm, data, jm2, data2, Maxiter)
}

# ------------------------------------------------------------------------------
# Gaussian + Cox Frailty Joint Model Estimator (jmfit2)
# ------------------------------------------------------------------------------
jmfit2 <- function(jm1, jm2, data_conti, data_surv, Maxiter = 200) {
  Maxiter <- fhl_validate_jm_maxiter(Maxiter)
  fhl_validate_jm_component(jm1, "jm1", allowed_resp = "gaussian")
  fhl_validate_jm_component(jm2, "jm2", allowed_resp = c("FM", "AFT"))
  res1 <- design_hglm(jm1[[3L]], DataMain = data_conti, sparse = TRUE)
  res2 <- design_frailtyHL(jm2[[3L]], data = data_surv, caller_env = environment())

  if (is.null(res1[[3L]]) || length(res1[[7L]]) != 1L || length(res2[[4L]]) != 1L || ncol(res1[[3L]]) != ncol(res2[[4L]][[1L]])) {
    stop("jmfit2 requires matching single random-effect cluster structure between longitudinal and survival models")
  }
  Z2_aligned <- fhl_align_cluster_Z(res1[[3L]], res2[[4L]][[1L]], "jm1", "jm2")

  X1_mat <- as.matrix(res1[[2L]]); storage.mode(X1_mat) <- "double"
  X2_mat <- as.matrix(res2[[2L]]); storage.mode(X2_mat) <- "double"
  if (ncol(X1_mat) + ncol(X2_mat) < 1L) {
    stop("jmfit2 requires at least one fixed-effect covariate (p >= 1)")
  }

  c_fit <- .Call(
    "fhl_fit_jm_cox1",
    as.double(res1[[1L]]), X1_mat, res1[[3L]],
    as.double(res2[[3L]]), X2_mat, Z2_aligned,
    as.double(res2[[8L]]), res2[[5L]], as.integer(res2[[6L]]),
    as.double(res2[[9L]]), as.double(res2[[11L]]), as.integer(Maxiter),
    PACKAGE = "frailtyHLnew"
  )
  beta_h <- c_fit[[1L]]
  se_beh <- c_fit[[2L]]
  v_h0 <- c_fit[[3L]]
  phi_h <- c_fit[[4L]]
  alpha_h <- c_fit[[5L]]
  rho_h <- c_fit[[6L]]
  i <- c_fit[[7L]]
  convergence <- c_fit[[8L]]
  converged_flag <- is.finite(convergence) && (convergence <= 1e-4)
  t_value <- beta_h / se_beh
  p_value <- 2 * (1 - pnorm(abs(t_value)))

  print("iterations : ")
  print(i)
  print("convergence : ")
  print(convergence)
  if (converged_flag) {
    print("converged")
  } else {
    print("did not converge")
    warning(sprintf("jmfit did not converge within %d iterations (final error = %s)",
                    as.integer(i), format(convergence, digits = 4)))
  }

  F.Est <- cbind(beta_h, se_beh, t_value, p_value)
  D.Est <- cbind(phi_h, alpha_h, rho_h)
  V.Est <- as.vector(v_h0)
  res <- list(
    F.Est = F.Est, D.Est = D.Est, V.Est = V.Est,
    iter = i, convergence = convergence, converged = converged_flag
  )

  print("Estimates for fixed effects")
  print_mat1 <- res$F.Est
  colnames(print_mat1) <- c("Estimate", "Std. Error", "t_value", "p_value")
  rownames(print_mat1) <- c(res1[[8L]], res2[[15L]])
  print(round(print_mat1, digits = 5))

  print("Estimates for dispersion parameters")
  print_mat2 <- res$D.Est
  colnames(print_mat2) <- c("phi_h", "alpha_h", "gamma_h")
  print(round(print_mat2, digits = 5))
  res
}

# ------------------------------------------------------------------------------
# Gaussian + Bivariate Cox Frailty Joint Model Estimator (jmfit3)
# ------------------------------------------------------------------------------
jmfit3 <- function(jm1, jm2, jm3, data_conti, data_surv, Maxiter = 200) {
  Maxiter <- fhl_validate_jm_maxiter(Maxiter)
  fhl_validate_jm_component(jm1, "jm1", allowed_resp = "gaussian")
  fhl_validate_jm_component(jm2, "jm2", allowed_resp = c("FM", "AFT"))
  fhl_validate_jm_component(jm3, "jm3", allowed_resp = c("FM", "AFT"))
  res1 <- design_hglm(jm1[[3L]], DataMain = data_conti, sparse = TRUE)
  res2 <- design_frailtyHL(jm2[[3L]], data = data_surv, caller_env = environment())
  res3 <- design_frailtyHL(jm3[[3L]], data = data_surv, caller_env = environment())

  if (is.null(res1[[3L]]) || length(res1[[7L]]) != 1L || length(res2[[4L]]) != 1L || length(res3[[4L]]) != 1L ||
      ncol(res1[[3L]]) != ncol(res2[[4L]][[1L]]) || ncol(res1[[3L]]) != ncol(res3[[4L]][[1L]])) {
    stop("jmfit3 requires matching single random-effect cluster structure across all three submodels")
  }
  Z2_aligned <- fhl_align_cluster_Z(res1[[3L]], res2[[4L]][[1L]], "jm1", "jm2")
  Z3_aligned <- fhl_align_cluster_Z(res1[[3L]], res3[[4L]][[1L]], "jm1", "jm3")

  X1_mat <- as.matrix(res1[[2L]]); storage.mode(X1_mat) <- "double"
  X2_mat <- as.matrix(res2[[2L]]); storage.mode(X2_mat) <- "double"
  X3_mat <- as.matrix(res3[[2L]]); storage.mode(X3_mat) <- "double"
  if (ncol(X1_mat) + ncol(X2_mat) + ncol(X3_mat) < 1L) {
    stop("jmfit3 requires at least one fixed-effect covariate (p >= 1)")
  }

  c_fit <- .Call(
    "fhl_fit_jm_cox2",
    as.double(res1[[1L]]), X1_mat, res1[[3L]],
    as.double(res2[[3L]]), X2_mat, Z2_aligned,
    as.double(res2[[8L]]), res2[[5L]], as.integer(res2[[6L]]), as.double(res2[[9L]]),
    as.double(res3[[3L]]), X3_mat, Z3_aligned,
    as.double(res3[[8L]]), res3[[5L]], as.integer(res3[[6L]]), as.double(res3[[9L]]),
    as.double(res2[[11L]]), as.integer(Maxiter),
    PACKAGE = "frailtyHLnew"
  )
  beta_h <- c_fit[[1L]]
  se_beh <- c_fit[[2L]]
  v_h0 <- c_fit[[3L]]
  phi_h <- c_fit[[4L]]
  alpha_h <- c_fit[[5L]]
  rho_h <- c_fit[[6L]]
  rho2_h <- c_fit[[7L]]
  i <- c_fit[[8L]]
  convergence <- c_fit[[9L]]
  converged_flag <- is.finite(convergence) && (convergence <= 1e-4)
  t_value <- beta_h / se_beh
  p_value <- 2 * (1 - pnorm(abs(t_value)))

  print("iterations : ")
  print(i)
  print("convergence : ")
  print(convergence)
  if (converged_flag) {
    print("converged")
  } else {
    print("did not converge")
    warning(sprintf("jmfit did not converge within %d iterations (final error = %s)",
                    as.integer(i), format(convergence, digits = 4)))
  }

  rho1_h <- rho_h
  F.Est <- cbind(beta_h, se_beh, t_value, p_value)
  D.Est <- cbind(phi_h, alpha_h, rho1_h, rho2_h)
  V.Est <- as.vector(v_h0)
  res <- list(
    F.Est = F.Est, D.Est = D.Est, V.Est = V.Est,
    iter = i, convergence = convergence, converged = converged_flag
  )

  print("Estimates for fixed effects")
  print_mat3 <- res$F.Est
  colnames(print_mat3) <- c("Estimate", "Std. Error", "t_value", "p_value")
  rownames(print_mat3) <- c(res1[[8L]], res2[[15L]], res3[[15L]])
  print(round(print_mat3, digits = 5))

  print("Estimates for dispersion parameters")
  print_mat4 <- res$D.Est
  colnames(print_mat4) <- c("phi_h", "alpha_h", "gamma1_h", "gamma2_h")
  print(round(print_mat4, digits = 5))
  res
}

# ------------------------------------------------------------------------------
# Bivariate Frailty Model Joint Estimator (jmfit_FM)
# ------------------------------------------------------------------------------
jmfit_FM <- function(jm1, jm2, data_conti, data_surv, Maxiter = 200) {
  Maxiter <- fhl_validate_jm_maxiter(Maxiter)
  fhl_validate_jm_component(jm1, "jm1", allowed_resp = c("FM", "AFT"))
  fhl_validate_jm_component(jm2, "jm2", allowed_resp = c("FM", "AFT"))
  res1 <- design_frailtyHL(jm1[[3L]], data = data_conti, caller_env = environment())
  res2 <- design_frailtyHL(jm2[[3L]], data = data_surv, caller_env = environment())

  if (length(res1[[4L]]) != 1L || length(res2[[4L]]) != 1L ||
      nrow(res1[[2L]]) != nrow(res2[[2L]]) ||
      ncol(res1[[4L]][[1L]]) != ncol(res2[[4L]][[1L]])) {
    stop("jmfit_FM requires matching sample sizes and single random-effect cluster dimensions")
  }
  Z2_aligned <- fhl_align_cluster_Z(res1[[4L]][[1L]], res2[[4L]][[1L]], "jm1", "jm2")

  X1_mat <- as.matrix(res1[[2L]]); storage.mode(X1_mat) <- "double"
  X2_mat <- as.matrix(res2[[2L]]); storage.mode(X2_mat) <- "double"
  if (ncol(X1_mat) + ncol(X2_mat) < 1L) {
    stop("jmfit_FM requires at least one fixed-effect covariate (p >= 1)")
  }

  c_fit <- .Call(
    "fhl_fit_jm_frailty",
    as.double(res1[[3L]]), X1_mat, res1[[4L]][[1L]],
    as.double(res1[[8L]]), res1[[5L]], as.integer(res1[[6L]]), as.double(res1[[9L]]),
    as.double(res2[[3L]]), X2_mat, Z2_aligned,
    as.double(res2[[8L]]), res2[[5L]], as.integer(res2[[6L]]), as.double(res2[[9L]]),
    as.double(res1[[11L]]), as.integer(Maxiter),
    PACKAGE = "frailtyHLnew"
  )
  beta_h <- c_fit[[1L]]
  se_beh <- c_fit[[2L]]
  v_h0 <- c_fit[[3L]]
  alpha_h <- c_fit[[4L]]
  rho_h <- c_fit[[5L]]
  i <- c_fit[[6L]]
  h0 <- c_fit[[7L]]
  hp <- c_fit[[8L]]
  p.bvh <- c_fit[[9L]]
  cAIC <- c_fit[[10L]]
  rAIC <- c_fit[[11L]]
  convergence <- c_fit[[12L]]
  converged_flag <- is.finite(convergence) && (convergence <= 1e-6)

  t_value <- beta_h / se_beh
  p_value <- 2 * (1 - pnorm(abs(t_value)))

  print("iterations : ")
  print(i)
  print("convergence : ")
  print(convergence)
  if (converged_flag) {
    print("converged")
  } else {
    print("did not converge")
    warning(sprintf("jmfit_FM did not converge within %d iterations (final error = %s)",
                    as.integer(i), format(convergence, digits = 4)))
  }

  F.Est <- cbind(beta_h, se_beh, t_value, p_value)
  D.Est <- cbind(alpha_h, rho_h)
  V.Est <- as.vector(v_h0)

  print(F.Est)
  print(D.Est)

  h.like <- cbind(h0, hp, p.bvh)
  print(h.like)

  AIC <- cbind(cAIC, rAIC)
  print(AIC)

  list(
    F.Est = F.Est, D.Est = D.Est, V.Est = V.Est,
    iter = i, convergence = convergence, converged = converged_flag
  )
}
