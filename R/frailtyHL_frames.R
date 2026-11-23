# ==============================================================================
# Model Frames, Design Matrices & Risk-Set Construction (Frailty / HGLM)
# ==============================================================================
FrailtyFrames <-
  function(
    mc, formula, contrasts, vnms = character(0),
    y, eval_env = NULL
  ) {
    if (missing(contrasts) || is.function(contrasts))
      contrasts <- NULL
    mf <- mc
    m <- match(
      c("data", "weights", "subset", "na.action", "offset"), names(mf),
      0
    )
    mf <- mf[c(1, m)]
    frame.form <- subbars(formula)
    if (length(vnms) > 0)
      frame.form[[3]] <- substitute(foo + bar, list(foo = parse(text = paste(vnms,
        collapse = " + "
      ))[[1]], bar = frame.form[[3]]))
    fixed.form <- nobars(formula)
    if (!inherits(fixed.form, "formula"))
      fixed.form <- stats::as.formula(substitute(foo ~ 1, list(foo = fixed.form)), env = environment(formula))
    environment(fixed.form) <- environment(frame.form) <- environment(formula)
    mf$formula <- frame.form
    mf$drop.unused.levels <- TRUE
    mf[[1]] <- as.name("model.frame")
    if (m[1] > 0L)
      names(mf)[2] <- "data"
    fe <- mf
    if (is.null(eval_env)) {
      eval_env <- if (sys.nframe() >= 2L) parent.frame(2L) else parent.frame()
    }
    mf <- tryCatch(eval(mf, eval_env), error = function(e) eval(mf, environment(formula)))
    fe$formula <- fixed.form
    fe <- tryCatch(eval(fe, eval_env), error = function(e) eval(fe, environment(formula)))
    Y <- model.response(mf, "any")
    if (length(dim(Y)) == 1) {
      nm <- rownames(Y)
      dim(Y) <- NULL
      if (!is.null(nm))
        names(Y) <- nm
    }
    mt <- attr(fe, "terms")
    X <- if (!is.empty.model(mt))
      model.matrix(mt, mf, contrasts)
    else matrix(, NROW(Y), 0)
    storage.mode(X) <- "double"
    fixef <- numeric(ncol(X))
    names(fixef) <- colnames(X)
    dimnames(X) <- NULL
    wts <- model.weights(mf)
    if (is.null(wts))
      wts <- numeric(0)
    off <- model.offset(mf)
    if (is.null(off))
      off <- numeric(0)
    if (any(wts <= 0))
      stop(gettextf("negative weights or weights of zero are not allowed"))
    if (length(off) && length(off) != NROW(Y))
      stop(gettextf(
        "number of offsets is %d should equal %d (number of observations)",
        length(off), NROW(Y)
      ))
    attr(mf, "terms") <- mt
    list(
      Y = Y, X = X, wts = as.double(wts), off = as.double(off),
      mf = mf, fixef = fixef
    )
  }

# ------------------------------------------------------------------------------
# Compressed O(n + m) Risk-Set & Data Construction
# ------------------------------------------------------------------------------
fhl_build_risk_set <-
  function(y, L, event) {
    ord <- as.integer(order(event))
    inv <- as.integer(order(ord))
    grid <- event[ord]
    lo <- as.integer(findInterval(L, grid))
    hi <- as.integer(findInterval(y, grid))
    n <- length(y)
    m <- length(event)
    stopifnot(all(lo <= hi), m > 0L)
    nested <- all(L == 0)
    obs <- if (nested) as.integer(order(hi, decreasing = TRUE)) else NULL
    list(
      nr = n, nc = m,
      lo = lo, hi = hi, ord = ord, inv = inv,
      nested = nested, obs = obs
    )
  }

fhl_validate_surv_data <-
  function(y, del, x, L0 = NULL, fn_name = "frailtyHL", require_pos_time = FALSE) {
    n <- nrow(x)
    if (is.null(n) || n < 1L)
      stop(sprintf("%s requires at least 1 observation", fn_name))
    y_v <- as.numeric(y)
    del_v <- as.numeric(del)
    if (length(y_v) != n || length(del_v) != n)
      stop(sprintf("Response and design matrix row counts do not match in %s", fn_name))
    if (any(!is.finite(y_v)) || any(!is.finite(del_v)) || any(!is.finite(x)))
      stop(sprintf("Non-finite values (NA/NaN/Inf) are not allowed in %s inputs", fn_name))
    if (any(!(del_v %in% c(0, 1))))
      stop(sprintf("Censoring indicator must contain only 0 or 1 in %s", fn_name))
    if (!any(del_v == 1))
      stop(sprintf("At least one uncensored event (status = 1) is required in %s", fn_name))
    if (isTRUE(require_pos_time)) {
      if (any(y_v <= 0))
        stop(sprintf("Survival times must be strictly positive (> 0) in %s", fn_name))
    } else {
      if (any(y_v < 0))
        stop(sprintf("Negative survival times are not allowed in %s", fn_name))
      if (any(y_v == 0 & del_v == 1))
        stop(sprintf("Uncensored events with survival time == 0 are not supported in %s", fn_name))
    }
    if (!is.null(L0)) {
      L0_v <- as.numeric(L0)
      if (length(L0_v) != n || any(!is.finite(L0_v)) || any(L0_v < 0))
        stop(sprintf("Entry times must be finite and non-negative in %s", fn_name))
      if (any(L0_v >= y_v & !(L0_v == 0 & y_v == 0)))
        stop(sprintf("Entry time must be strictly less than exit time for left-truncated data in %s", fn_name))
    }
    invisible(TRUE)
  }

fhl_make_data <-
  function(y, x, del, z, L0 = NULL) {
    if (is.null(L0))
      L0 <- y - y
    o <- NULL
    if (is.unsorted(L0[, 1], na.rm = TRUE) || anyNA(L0[, 1])) {
      o <- order(L0[, 1], na.last = NA)
      y <- y[o, , drop = FALSE]
      x <- x[o, , drop = FALSE]
      del <- del[o, , drop = FALSE]
      L0 <- L0[o, , drop = FALSE]
      z <- lapply(z, function(a) a[o, , drop = FALSE])
    }
    ev <- y[del[, 1] == 1, 1]
    event <- unique(ev[ev != 0])
    if (!length(event))
      stop("No nonzero event times: upstream domain unsupported")
    di <- matrix(tabulate(match(ev, event), nbins = length(event)), ncol = 1)
    t2 <- matrix(c(event, rep(0, length(ev) - length(event))), ncol = 1)
    Mi <- fhl_build_risk_set(y[, 1], L0[, 1], event)
    res <- list(y, x, del, z, Mi, length(event), t2, di)
    attr(res, "sort_order") <- o
    res
  }

design_frailtyHL <-
  function(formula, data, weights, subset, na.action, caller_env = NULL) {
    Call <- match.call()
    mc <- match.call()
    if (!missing(subset) && !is.null(subset))
      stop("'subset' is not supported in design_frailtyHL / joint models")
    indx <- match(c("formula", "data", "weights", "subset", "na.action"),
      names(Call),
      nomatch = 0
    )
    if (indx[1] == 0)
      stop("A formula argument is required")
    if (!inherits(formula, "formula") || length(formula) < 3L)
      stop("'formula' must be a two-sided formula")
    if (!length(expandSlash(findbars(formula[[3]]))))
      stop("No random effects terms specified in formula")
    if (is.null(caller_env)) caller_env <- parent.frame()
    temp <- Call[c(1, indx)]
    temp[[1]] <- as.name("model.frame")
    special <- c("strata", "cluster")
    temp$formula <- terms(subbars(formula), special)
    m <- eval(temp, caller_env)
    mt_sp <- attr(m, "terms")
    if (!is.null(attr(mt_sp, "specials")$strata) || !is.null(attr(mt_sp, "specials")$cluster))
      stop("'strata()' and 'cluster()' terms are not supported in jmfit")
    Y <- model.extract(m, "response")
    fr <- FrailtyFrames(mc, formula, contrasts = NULL, eval_env = caller_env)
    if (length(fr$off) > 0L)
      stop("'offset()' terms are not supported in design_frailtyHL")
    if (length(fr$wts) > 0L)
      stop("'weights' are not supported in design_frailtyHL")
    has_icpt <- length(fr$fixef) > 0L && identical(names(fr$fixef)[1L], "(Intercept)")
    namesX <- if (has_icpt) names(fr$fixef)[-1L] else names(fr$fixef)
    FL <- HGLMFactorList(formula, fr, 0L, 0L, sparse = TRUE)
    if (!inherits(Y, "Surv") || !(attr(Y, "type") %in% c("right", "counting")))
      stop("'formula' response must be a right-censored Surv(time, status) or counting Surv(entry, exit, status) object")
    nc_y <- ncol(Y)
    if (is.null(nc_y) || !(nc_y %in% c(2L, 3L)))
      stop("'formula' response must be a Surv(time, status) or Surv(entry, exit, status) object")
    leftT <- if (nc_y == 3L) 1L else 0L
    n_y <- nrow(Y)
    y <- matrix(Y[, if (leftT == 0L) 1L else 2L], n_y, 1)
    L0 <- if (leftT == 0L) matrix(0, n_y, 1) else matrix(Y[, 1], n_y, 1)
    x <- if (has_icpt) fr$X[, -1L, drop = FALSE] else fr$X
    z <- FL$Design
    n <- nrow(x)
    p <- ncol(x)
    nrand <- length(z)
    q <- vapply(z, function(zi) dim(zi)[2L], integer(1))
    del <- matrix(0, n, 1)
    del[, 1] <- Y[, if (leftT == 0L) 2L else 3L]
    fhl_validate_surv_data(y, del, x, L0, fn_name = "jmfit")
    res1 <- fhl_make_data(y, x, del, z, L0)
    y <- res1[[1]]
    x <- res1[[2]]
    del <- res1[[3]]
    z <- res1[[4]]
    Mi <- res1[[5]]
    idx2 <- res1[[6]]
    t2 <- res1[[7]]
    di <- res1[[8]]
    beta_h <- matrix(0, p, 1)
    qcum <- cumsum(c(0, q))
    v_h <- matrix(0, qcum[nrand + 1], 1)
    list(y, x, del, z, Mi, idx2, t2, di, beta_h, qcum, v_h, p, q, nrand, namesX)
  }

# ------------------------------------------------------------------------------
# HGLM Model Frames, Random-Effect Factor Lists & Design Matrix Builders
# ------------------------------------------------------------------------------
HGLMFrames <-
  function(mc, formula, contrasts, vnms = character(0)) {
    mf <- mc
    if ("subset" %in% names(mf) && !is.null(mf[["subset"]])) {
      sub_expr <- mf[["subset"]]
      p_env <- if (sys.nframe() >= 2L) parent.frame(2L) else parent.frame()
      sub_is_null <- is.symbol(sub_expr) &&
        exists(as.character(sub_expr), envir = p_env, inherits = TRUE) &&
        is.null(get(as.character(sub_expr), envir = p_env, inherits = TRUE))
      if (!sub_is_null)
        stop("'subset' is not supported in HGLMFrames / joint models")
    }
    m <- match(
      c("DataMain", "weights", "na.action", "offset"),
      names(mf), 0
    )
    mf <- mf[c(1, m)]
    frame.form <- subbars(formula)
    if (length(vnms) > 0)
      frame.form[[3]] <- substitute(foo + bar, list(foo = parse(text = paste(vnms,
        collapse = " + "
      ))[[1]], bar = frame.form[[3]]))
    fixed.form <- nobars(formula)
    if (!inherits(fixed.form, "formula"))
      fixed.form <- stats::as.formula(substitute(foo ~ 1, list(foo = fixed.form)), env = environment(formula))
    environment(fixed.form) <- environment(frame.form) <- environment(formula)
    mf$formula <- frame.form
    mf$drop.unused.levels <- TRUE
    mf[[1]] <- as.name("model.frame")
    if (m[1L] > 0L)
      names(mf)[2] <- "data"
    fe <- mf
    mf <- eval(mf, parent.frame(2))
    fe$formula <- fixed.form
    fe <- eval(fe, parent.frame(2))
    Y <- model.response(mf, "any")
    if (length(dim(Y)) == 1) {
      nm <- rownames(Y)
      dim(Y) <- NULL
      if (!is.null(nm))
        names(Y) <- nm
    }
    mt <- attr(fe, "terms")
    X <- if (!is.empty.model(mt))
      model.matrix(mt, mf, contrasts)
    else matrix(, NROW(Y), 0)
    storage.mode(X) <- "double"
    fixef <- numeric(ncol(X))
    names(fixef) <- colnames(X)
    dimnames(X) <- NULL
    wts <- model.weights(mf)
    if (is.null(wts))
      wts <- numeric(0)
    off <- model.offset(mf)
    if (is.null(off))
      off <- numeric(0)
    if (any(wts <= 0))
      stop(gettextf("negative weights or weights of zero are not allowed"))
    if (length(off) && length(off) != NROW(Y))
      stop(gettextf(
        "number of offsets is %d should equal %d (number of observations)",
        length(off), NROW(Y)
      ))
    attr(mf, "terms") <- mt
    list(
      Y = Y, X = X, wts = as.double(wts), off = as.double(off),
      mf = mf, fixef = fixef
    )
  }

# Evaluate random-effect grouping expression safely, treating ':' as factor interaction
fhl_eval_group_factor <- function(expr, mf, env) {
  if (is.call(expr) && identical(expr[[1L]], as.name(":"))) {
    f1 <- fhl_eval_group_factor(expr[[2L]], mf, env)
    f2 <- fhl_eval_group_factor(expr[[3L]], mf, env)
    return(droplevels(interaction(f1, f2, drop = TRUE, sep = ":", lex.order = TRUE)))
  }
  col_nm <- paste(deparse(expr), collapse = "")
  val <- if (col_nm %in% names(mf)) {
    mf[[col_nm]]
  } else {
    eval(expr, envir = mf, enclos = env)
  }
  if (length(val) != nrow(mf)) {
    stop(sprintf(
      "Random-effect grouping '%s' length (%d) does not match number of observations (%d)",
      col_nm, length(val), nrow(mf)
    ))
  }
  if (anyNA(val)) {
    stop(sprintf("Missing values are not allowed in random-effect grouping '%s'", col_nm))
  }
  droplevels(as.factor(val))
}

HGLMFactorList <-
  function(formula, fr, rmInt, drop, sparse = FALSE) {
    mf <- fr$mf
    raw_bars <- findbars(formula[[3]])
    if (!is.null(raw_bars)) {
      if (!is.list(raw_bars)) raw_bars <- list(raw_bars)
      for (rb in raw_bars) {
        if ("/" %in% all.names(rb[[3L]]))
          stop("Nested random-effect syntax '(... | g1/g2)' is not supported; use explicit '(... | g1:g2)' or separate terms")
      }
    }
    bars <- expandSlash(findbars(formula[[3]]))
    for (i in 1:length(bars)) {
      checkcorr <- findplus(bars[[i]])
      if (checkcorr == 1)
        stop("Correlated random effects are not currently allowed in the HGLM routines")
      if (checkcorr == -1)
        stop("You do not need to specify '-1' for no intercept it is done be default")
    }
    if (!length(bars))
      stop("No random effects terms specified in formula")
    names(bars) <- unlist(lapply(bars, function(x) paste(deparse(x[[3]]), collapse = "")))
    fl <- lapply(bars, function(x) {
      if (is.numeric(x[[2L]]) && all(as.numeric(x[[2L]]) == 0))
        stop("Zero random-effect design '(0 | group)' is not supported")
      ff <- fhl_eval_group_factor(x[[3]], mf, environment(formula))
      if (nlevels(ff) < 2L)
        stop("Random-effect grouping factor must have at least 2 levels")
      if (isTRUE(sparse) && identical(x[[2]], 1) && !rmInt && !drop && !anyNA(ff)) {
        g <- as.integer(ff)
        n_obs <- length(ff)
        q_lvl <- length(levels(ff))
        z_sp <- Matrix::sparseMatrix(
          i = seq_len(n_obs),
          j = g,
          x = 1.0,
          dims = c(n_obs, q_lvl),
          dimnames = list(names(ff), levels(ff))
        )
        return(list(f = ff, Design = z_sp))
      }
      im <- as(ff, "sparseMatrix")
      if (!isTRUE(validObject(im, test = TRUE)))
        stop(
          "invalid conditioning factor in random effect: ",
          format(x[[3]])
        )
      if (is.name(x[[2]])) {
        tempexp <- paste("~", as.character(x[[2]]), "-1")
        tempexp <- as.formula(tempexp)[[2]]
      } else tempexp <- x[[2]]
      mm <- model.matrix(
        eval(substitute(~expr, list(expr = tempexp))),
        mf
      )
      if (rmInt) {
        if (is.na(icol <- match("(Intercept)", colnames(mm))))
          if (ncol(mm) < 2)
            stop("lhs of a random-effects term cannot be an intercept only")
        mm <- mm[, -icol, drop = FALSE]
      }
      if (ncol(mm) < 1L || nrow(mm) != nrow(mf))
        stop("Random-effect term has invalid or empty design matrix")
      ans <- list(f = ff, A = do.call(rbind, lapply(
        seq_len(ncol(mm)),
        function(j) im
      )), Zt = do.call(rbind, lapply(
        seq_len(ncol(mm)),
        function(j) {
          im@x <- mm[, j]
          im
        }
      )), ST = matrix(0, ncol(mm), ncol(mm), dimnames = list(
        colnames(mm),
        colnames(mm)
      )))
      if (drop) {
        ans$A@x <- rep(0, length(ans$A@x))
        ans$Zt <- drop0(ans$Zt)
      }
      ans
    })
    Design <- list(0)
    Subject <- list(0)
    for (i in 1:length(fl)) {
      Subject[[i]] <- as.factor(fl[[i]]$f)
      if (!is.null(fl[[i]]$Design)) {
        Design[[i]] <- fl[[i]]$Design
      } else {
        tempmat <- fl[[i]]$Zt
        if (isTRUE(sparse)) {
          tempmat <- Matrix::t(tempmat)
          if (!methods::is(tempmat, "dgCMatrix")) {
            tempmat <- methods::as(methods::as(tempmat, "generalMatrix"), "CsparseMatrix")
          }
        } else {
          tempmat <- as.matrix(t(tempmat))
        }
        Design[[i]] <- tempmat
      }
    }
    list(Design = Design, Subject = Subject, namesRE = names(bars))
  }

design_hglm <-
  function(formula, DataMain, BinomialDen = NULL, weights, subset, na.action, sparse = FALSE) {
    if (!missing(subset) && !is.null(subset))
      stop("'subset' is not supported in design_hglm / joint models")
    mc <- match.call()
    formulaMean <- formula
    fr <- HGLMFrames(mc, formulaMean, contrasts = NULL)
    if (length(fr$off) > 0L)
      stop("'offset()' terms are not supported in joint models")
    if (length(fr$wts) > 0L)
      stop("'weights' are not supported in joint models")
    if (length(fr$Y) < 1L || any(!is.finite(fr$Y)) || any(!is.finite(fr$X)))
      stop("Non-finite values (NA/NaN/Inf) are not allowed in longitudinal joint-model inputs")
    namesX <- names(fr$fixef)
    namesY <- names(fr$mf)[1]
    y <- matrix(fr$Y, length(fr$Y), 1)
    if (is.null(BinomialDen)) BinomialDen <- (y + 1) / (y + 1)
    x <- fr$X
    n <- nrow(x)
    p <- ncol(x)
    random_mean <- findbars(formulaMean)
    if (!is.null(random_mean)) {
      FL <- HGLMFactorList(formulaMean, fr, 0L, 0L, sparse = isTRUE(sparse))
      namesRE <- FL$namesRE
      z <- FL$Design
      nrand <- length(z)
      q <- rep(0, nrand)
      for (i in 1:nrand) {
        q[i] <- dim(z[[i]])[2]
      }
      if (nrand == 1L) {
        zz <- z[[1L]]
      } else {
        zz <- do.call(cbind, z)
      }
      z <- zz
    } else {
      z <- NULL
      nrand <- 1
      q <- rep(0, nrand)
      for (i in 1:nrand) q[i] <- 0
    }
    qcum <- cumsum(c(0, q))
    v_h <- matrix(0, qcum[nrand + 1], 1)
    res <- list(y, x, z, v_h, n, p, q, namesX)
  }

# ------------------------------------------------------------------------------
# Mixed-Model Random-Effect Formula Parsing Utilities
# ------------------------------------------------------------------------------
findbars <-
  function(term) {
    if (is.name(term) || !is.language(term))
      return(NULL)
    if (term[[1]] == as.name("("))
      return(findbars(term[[2]]))
    if (!is.call(term))
      stop("term must be of class call")
    if (term[[1]] == as.name("|"))
      return(term)
    if (length(term) == 2)
      return(findbars(term[[2]]))
    c(findbars(term[[2]]), findbars(term[[3]]))
  }

nobars <-
  function(term) {
    if (!("|" %in% all.names(term)))
      return(term)
    if (is.call(term) && term[[1]] == as.name("|"))
      return(NULL)
    if (length(term) == 2) {
      nb <- nobars(term[[2]])
      if (is.null(nb))
        return(NULL)
      term[[2]] <- nb
      return(term)
    }
    nb2 <- nobars(term[[2]])
    nb3 <- nobars(term[[3]])
    if (is.null(nb2))
      return(nb3)
    if (is.null(nb3))
      return(nb2)
    term[[2]] <- nb2
    term[[3]] <- nb3
    term
  }

subbars <-
  function(term) {
    if (is.name(term) || !is.language(term))
      return(term)
    if (length(term) == 2) {
      term[[2]] <- subbars(term[[2]])
      return(term)
    }
    stopifnot(length(term) >= 3)
    if (is.call(term) && term[[1]] == as.name("|"))
      term[[1]] <- as.name("+")
    for (j in 2:length(term)) term[[j]] <- subbars(term[[j]])
    term
  }

slashTerms <-
  function(x) {
    if (!("/" %in% all.names(x)))
      return(x)
    if (x[[1]] != as.name("/"))
      stop("unparseable formula for grouping factor")
    list(slashTerms(x[[2]]), slashTerms(x[[3]]))
  }

expandSlash <-
  function(bb) {
    if (!is.list(bb))
      return(expandSlash(list(bb)))
    unlist(lapply(bb, function(x) {
      if (length(x) > 2 && is.list(trms <- slashTerms(x[[3]])))
        return(lapply(unlist(makeInteraction(trms)), function(trm) substitute(foo |
          bar, list(foo = x[[2]], bar = trm))))
      x
    }))
  }

makeInteraction <- function(x)
### from a list of length 2 return recursive interaction terms
{
  if (length(x) < 2) return(x)
  trm1 <- makeInteraction(x[[1]])
  trm11 <- if (is.list(trm1)) trm1[[1]] else trm1
  list(substitute(foo:bar, list(foo = x[[2]], bar = trm11)), trm1)
}

findplus <-
  function(term) {
    if (is.numeric(term))
      return(0)
    if (!is.language(term))
      stop("Unsupported random-effect specification before '|'")
    if (length(term) == 1)
      return(0)
    if (term[[1]] == as.name("|"))
      return(findplus(term[[2]]))
    if (!is.call(term))
      stop("term must be of class call")
    if (term[[1]] == as.name("+"))
      return(1)
    if (term[[1]] == as.name("-"))
      return(-1)
    stop("Unsupported random-effect LHS expression before '|'; expected '1', variable name, '+' or '-'")
  }
