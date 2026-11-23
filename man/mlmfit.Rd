\name{mlmfit}
\alias{mlmfit}
\title{Accelerated Failure Time (AFT) Models with Random Effects}
\description{
\code{mlmfit} fits right-censored log-normal accelerated failure time (AFT)
linear mixed models with one or two random-effect components via hierarchical
likelihood.
}
\usage{
mlmfit(jm1, data, weights, subset, na.action, Maxiter = 200, mord = 0, dord = 1)
}
\arguments{
  \item{jm1}{A \code{jointmodeling} object specifying the AFT random-effect model (\code{RespDist = "AFT"}).}
  \item{data}{Data frame containing the variables referenced in \code{jm1}.}
  \item{weights}{Case weights (unsupported; must be omitted).}
  \item{subset}{Subset expression (unsupported; must be omitted).}
  \item{na.action}{Missing-data filter function (unsupported; must be omitted).}
  \item{Maxiter}{Maximum number of outer iterations (default \code{200}).}
  \item{mord}{Order of Laplace approximation for fixed effects: \code{0} for \code{HL(0,1)} (default)
or \code{1} for \code{HL(1,1)}.}
  \item{dord}{Order of Laplace approximation for dispersion parameters (currently \code{1}).}
}
\value{
A named list containing:
\item{F.Est}{Matrix of fixed-effect estimates, standard errors, t-statistics, and p-values.}
\item{D.Est}{Matrix of residual and random-effect variance component estimates and standard errors.}
\item{V.Est}{Estimated random-effect predictions.}
\item{iter}{Number of outer iterations executed.}
\item{convergence}{Final convergence criterion value.}
\item{converged}{Logical indicator of convergence within \code{Maxiter}.}
}
