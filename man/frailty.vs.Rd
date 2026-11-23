\name{frailty.vs}
\alias{frailty.vs}
\title{Penalized Variable Selection for Frailty Models}
\description{
\code{frailty.vs} performs penalized variable selection (LASSO, SCAD, and HL)
for fixed effects in log-normal Cox frailty models via hierarchical likelihood.
}
\usage{
frailty.vs(formula, model, penalty, data, B = NULL, v = NULL,
alpha = NULL, tun1 = NULL, tun2 = NULL, varfixed = FALSE, varinit = 0.1)
}
\arguments{
  \item{formula}{A formula object, with the response on the left of a \code{~} operator,
and the terms for the fixed and random effects on the right, e.g.,
\code{Surv(time, status) ~ x + (1 | id)}.}
  \item{model}{Character string specifying the frailty distribution (\code{"lognorm"}).}
  \item{penalty}{Penalty function (\code{"LASSO"}, \code{"SCAD"}, or \code{"HL"}, case-insensitive).}
  \item{data}{Data frame containing the variables in \code{formula}.}
  \item{B}{Optional numeric vector of initial fixed-effect coefficients.}
  \item{v}{Optional list of initial random-effect vectors (zeros by default).}
  \item{alpha}{Optional numeric vector of initial frailty variance components.}
  \item{tun1}{Tuning parameter \eqn{\gamma} for LASSO and SCAD, or shape parameter \eqn{\omega > 2} for HL.}
  \item{tun2}{Regularization parameter \eqn{\gamma > 0} for HL (not used for LASSO or SCAD).}
  \item{varfixed}{Logical; if \code{TRUE} (\code{FALSE}), frailty variance parameters are fixed (estimated).}
  \item{varinit}{Starting value for frailty variance when \code{alpha} is \code{NULL} (default \code{0.1}).}
}
\details{
For LASSO and SCAD, \code{tun1} specifies the regularization parameter \eqn{\gamma}
and \code{tun2} may be omitted (\code{NULL}). For the HL penalty, \code{tun1}
specifies the shape parameter \eqn{\omega} (e.g., \code{c(2.1, 3, 10, 30, 50)})
and \code{tun2} specifies the regularization parameter \eqn{\gamma > 0}.
}
\value{
A named list containing:
\item{beta}{Estimated fixed-effect regression coefficients at the selected tuning parameter.}
\item{se}{Sandwich standard errors of the estimated fixed-effect coefficients.}
\item{v}{Estimated random-effect (log-frailty) predictions.}
\item{alpha}{Estimated (or fixed) frailty variance-covariance parameters.}
\item{tuning_parameter}{Selected tuning parameter value(s) minimizing the selection criterion.}
\item{BIC}{Minimum BIC (or GCV for correlated two-component models) achieved on the tuning grid.}
\item{repetition}{Iteration count for the selected tuning parameter.}
\item{convergence}{Integer convergence indicator (\code{1} for converged).}
\item{res1}{Matrix summarizing tuning grid candidates and corresponding criterion values.}
}
\examples{
data(kidney, package = "frailtyHLnew")
kidney_std <- kidney
kidney_std$age <- (kidney_std$age - mean(kidney_std$age)) / sd(kidney_std$age)
fit_lasso <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
  model = "lognorm", penalty = "lasso", data = kidney_std,
  B = c(0.09, -1.66, 0.17, 0.39, -1.16), tun1 = 0.032)
fit_hl <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
  model = "lognorm", penalty = "HL", data = kidney_std,
  B = c(0.09, -1.66, 0.17, 0.39, -1.16), tun1 = 2.1, tun2 = 0.224)
}
