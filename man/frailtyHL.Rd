\name{frailtyHL}
\alias{frailtyHL}
\alias{summary.frailtyHL}
\title{Fitting Frailty Models using H-likelihood Approach}
\description{
\code{frailtyHL} fits Cox proportional-hazards frailty models using h-likelihood
estimation procedures for normal (log-normal), gamma, and AR(1) frailty
distributions, including shared and multilevel (nested) random-effect structures
(Ha, Lee and Song, 2001; Ha and Lee, 2003, 2005; Lee, Nelder and Pawitan, 2017).
}
\usage{
frailtyHL(formula, data, weights, subset, na.action, RandDist = "Normal",
mord = 0, dord = 1, Maxiter = 200, convergence = 10^-6, varfixed = FALSE,
varinit = c(0.163), varnonneg = FALSE)

\method{summary}{frailtyHL}(object, ...)
}
\arguments{
  \item{formula}{A formula object, with the \code{Surv} response on the left of a \code{~}
operator, and the fixed and random-effect terms on the right, e.g.,
\code{Surv(time, status) ~ x + (1 | id)}.}
  \item{data}{Data frame containing the variables named in \code{formula}.}
  \item{weights}{Vector of case weights (ignored with a warning by the Cox frailty engine).}
  \item{subset}{Expression indicating which subset of rows of \code{data} should be used.}
  \item{na.action}{A missing-data filter function.}
  \item{RandDist}{Distribution for random effects (\code{"Normal"}, \code{"Gamma"}, or \code{"AR1"}).}
  \item{mord}{Order of Laplace approximation for mean parameters (\code{0} or \code{1}; default \code{0}).}
  \item{dord}{Order of Laplace approximation for dispersion components (\code{0}, \code{1}, or \code{2};
default \code{1}). Note that \code{dord = 0} uses a PQL-type dispersion estimating equation
(see Ha, Jeong and Lee, 2017, Table 4.1 for standard HL orders).}
  \item{Maxiter}{Maximum number of outer iterations (default \code{200}).}
  \item{convergence}{Convergence tolerance (default \code{1e-6}).}
  \item{varfixed}{Logical; if \code{TRUE} (\code{FALSE}), frailty variance components are fixed (estimated).}
  \item{varinit}{Starting value(s) for frailty variance components (default \code{c(0.163)}).}
  \item{varnonneg}{Logical; if \code{TRUE} (\code{FALSE}), reports zero (\code{NaN}) SE at zero variance estimates.}
  \item{object}{A fitted object of class \code{"frailtyHL"}.}
  \item{...}{Further arguments passed to or from other methods (ignored).}
}
\details{
The \code{frailtyHLnew} package produces estimates of fixed effects and frailty
parameters as well as their standard errors, marginal likelihood approximations,
and AIC criteria (cAIC, mAIC, and rAIC) for shared, multilevel, and AR(1) frailty models.
}
\value{
\code{frailtyHL} returns an object of class \code{"frailtyHL"} containing:
\item{Model}{Character description of the fitted model family.}
\item{formula}{Model formula used in the fit.}
\item{Method}{Hierarchical-likelihood approximation order label (e.g., \code{"HL(0,1)"}).}
\item{FixCoef}{Matrix of fixed-effect estimates, standard errors, t-statistics, and p-values.}
\item{RandCoef}{Matrix of frailty variance estimates and standard errors.}
\item{rho}{Estimated autocorrelation parameter (for \code{RandDist = "AR1"}).}
\item{likelihood}{Matrix of h-likelihood and adjusted profile likelihood values.}
\item{iter}{Number of iterations performed.}
\item{convergence}{Character convergence status (\code{"converged"} or \code{"did not converge"}).}
\item{aic}{Matrix of conditional, marginal, and restricted AIC values (\code{cAIC}, \code{mAIC}, \code{rAIC}).}
\item{v_h}{Predicted log-frailty random effects.}
\item{Hinv}{Inverse h-likelihood Hessian matrix at convergence.}
\item{p}{Number of fixed-effect parameters.}
\item{q}{Vector of random-effect dimensions.}
\item{u_h}{Predicted frailty values \eqn{\exp(v_h)} (for \code{RandDist = "Gamma"}).}
\item{Hinv_u}{Frailty-scale inverse Hessian matrix (for \code{RandDist = "Gamma"}).}

\code{summary.frailtyHL} returns an object of class \code{"summary.frailtyHL"} summarizing
the model formula, approximation method, fixed and random coefficient tables, likelihoods, and AICs.
}
\references{
Ha, I. D. and Lee, Y. (2003). Estimating frailty models via Poisson Hierarchical generalized linear
models. Journal of Computational and Graphical Statistics, 12, 663-681.

Ha, I. D. and Lee, Y. (2005). Comparison of hierarchical likelihood versus orthodox best linear
unbiased predictor approaches for frailty models. Biometrika, 92, 717-723.

Ha, I. D., Lee, Y. and Song, J. K. (2001). Hierarchical likelihood approach for frailty models.
Biometrika, 88, 233-243.

Lee, Y., Nelder, J. A. and Pawitan, Y. (2017). Generalised linear models with random effects:
unified analysis via h-likelihood. 2nd Edition. Chapman and Hall: London.}
\examples{
data(kidney, package = "frailtyHLnew")
kidney_ln01 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
  RandDist = "Normal", mord = 0, dord = 1)

data(rats, package = "frailtyHLnew")
rat_cox <- frailtyHL(Surv(time, status) ~ rx + (1 | litter), rats,
  varfixed = TRUE, varinit = c(0))

\donttest{
kidney_ln11 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
  RandDist = "Normal", mord = 1, dord = 1)
kidney_g12 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney,
  RandDist = "Gamma", mord = 1, dord = 2)
data(cgd, package = "frailtyHLnew")
cgd_ln11 <- frailtyHL(Surv(tstop - tstart, status) ~ treat + (1 | center) + (1 | id),
  cgd, RandDist = "Normal", mord = 1, dord = 1, convergence = 1e-4, varinit = c(0.03, 1.0))
}
}
