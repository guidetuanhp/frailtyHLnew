\name{frailty.vs}
\alias{frailty.vs}
\title{Penalized Variable Selection for Frailty Models}
\description{frailty.vs is variable-selection procedures 
(LASSO, SCAD and HL) of fixed effects in frailty models.}
\usage{
frailty.vs(formula, model, penalty, data, B = NULL, v = NULL, 
alpha = NULL, tun1 = NULL, tun2 = NULL, varfixed = FALSE, varinit = 0.1)
}
\arguments{
  \item{formula}{A formula object, with the response on the left of a ~ operator, and the terms for
the fixed and random effects on the right. e.g. formula=Surv(time,status)~x+(1|id),
time : survival time, status : censoring indicator having 1 (0) for uncensored
(censored) observation, x : fixed covariate, id : random effect.}
  \item{model}{Log-normal frailty models ("lognorm")}
  \item{penalty}{Penalty functions ("LASSO", "SCAD", or "HL", case-insensitive)}
  \item{data}{Dataframe used}
  \item{B}{Initial values of fixed effects}
  \item{v}{Initial values of random effects. Zeros are default}
  \item{alpha}{Initial value of variance of random effects.}
  \item{tun1}{Tuning parameter \eqn{\gamma} for LASSO and SCAD, and shape parameter \eqn{\omega > 2} for HL.}
  \item{tun2}{Tuning parameter \eqn{\gamma} for HL (not used for LASSO or SCAD).}
  \item{varfixed}{Logical value: if TRUE (FALSE), the value of one or more of the variance terms for the frailties is fixed (estimated).}
  \item{varinit}{Starting values for frailties, the default is 0.1.} 
}
\details{
For LASSO and SCAD, \code{tun1} specifies the regularization parameter \eqn{\gamma}
and \code{tun2} may be omitted (\code{NULL}). For the HL penalty, \code{tun1}
specifies the shape parameter \eqn{\omega} (\eqn{w > 2}, e.g., \code{c(2.1, 3, 10, 30, 50)})
and \code{tun2} specifies the regularization parameter \eqn{\gamma > 0}.
}
\examples{
\donttest{
data(kidney, package = "frailtyHLnew")
fit_lasso <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
  model = "lognorm", penalty = "lasso", data = kidney,
  B = c(0.09, -1.66, 0.17, 0.39, -1.16), tun1 = 0.032)
fit_hl <- frailty.vs(Surv(time, status) ~ age + sex + GN + AN + PKD + (1 | id),
  model = "lognorm", penalty = "HL", data = kidney,
  B = c(0.09, -1.66, 0.17, 0.39, -1.16), tun1 = 2.1, tun2 = 0.224)
}
}

