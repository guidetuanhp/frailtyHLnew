\name{hlike.frailty}
\alias{hlike.frailty}
\title{Competing Risk Frailty Models using H-Likelihood}
\description{
Performs hierarchical-likelihood estimation for univariate frailty models,
cause-specific competing-risk frailty models, and Fine-Gray subhazard frailty
models under univariate or multivariate normal random-effect distributions.
}
\usage{
hlike.frailty(formula, data, inits, order = 1, frailty.cov = "none", subHazard = FALSE,
alpha = 0.05, MAX.ITER = 100, TOL = 1e-06)
}
\arguments{
  \item{formula}{Formula whose left-hand side is a \code{CmpRsk} object and right-hand
side specifies numeric main-effect predictors plus a \code{cluster()} term.}
  \item{data}{Data frame containing the variables used in \code{formula}.}
  \item{inits}{Named list of initial values with components \code{beta}, \code{v}, and \code{theta}.}
  \item{order}{Order of Laplace approximation: \code{0}, \code{1} (first-order), or \code{2}
(second-order; supported when \code{frailty.cov = "none"}).}
  \item{frailty.cov}{Covariance structure for the frailty distribution: \code{"none"}
(univariate normal), \code{"independent"}, \code{"exchangeable"}, or \code{"unstructured"}.}
  \item{subHazard}{Logical; if \code{TRUE}, fits the Fine-Gray subdistribution hazard frailty model.}
  \item{alpha}{Significance level for \eqn{100(1 - \alpha)\%} confidence intervals.}
  \item{MAX.ITER}{Maximum number of iterations.}
  \item{TOL}{Convergence tolerance.}
}
\value{
An object of class \code{"hlike"} containing:
\item{beta}{Data frame of fixed-effect estimates, standard errors, confidence limits, z-statistics, and p-values.}
\item{theta}{Data frame of estimated frailty variance-covariance components and standard errors.}
\item{v}{Matrix of predicted cluster-level random effects.}
\item{theta.pvalue}{Likelihood-ratio test p-value against the no-frailty model.}
\item{lambda.0}{List of estimated baseline hazard increments per event type.}
\item{time}{List of distinct observed event times per event type.}
\item{h1, hp, Ahp}{Hierarchical and adjusted profile log-likelihood values.}
\item{loglik.noFrailty}{Log-likelihood of the corresponding model without frailty.}
\item{rAIC, cAIC}{Restricted and conditional AIC model-selection criteria.}
\item{H, H0}{Full and unpenalized Hessian matrices at the final state.}
\item{converge}{Integer convergence indicator (\code{1} if converged, \code{0} otherwise).}
\item{iterations}{Number of outer iterations executed.}
\item{call}{Matched function call.}
}
