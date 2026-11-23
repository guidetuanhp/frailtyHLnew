\name{hlike.frailty}
\alias{hlike.frailty}
\title{Competing Risk Frailty Models using H-Likelihood}
\description{Perform hierarchical likelihood estimation of the univariate frailty model, cause-specific frailty
model and subhazard frailty model. Assuming either a univariate normal or multivariate
normal distribution for the random effects V, where different covariance structures can be
assumed for the multivariate normal distribution.
}
\usage{
hlike.frailty(formula, data, inits, order = 1, frailty.cov = "none", subHazard = FALSE, 
alpha = 0.05, MAX.ITER = 100, TOL = 1e-06)
}
\arguments{
  \item{formula}{left-hand side is a CmpRsk object (see details), right-hand
side is predictors (currently limited to numeric main effects), must
include a cluster term that identifies the cluster variable.
}
  \item{data}{dataframe containing the variables used in the formula}
  \item{inits}{list of initial values, three named components: beta, v and theta}
  \item{order}{numeric, order of the Laplace approximation, 0=no order, 1=first-order,
2=second-order; second-order only applies to models with a univariate
normal distribution (\code{frailty.cov = "none"})}
  \item{frailty.cov}{character string "none", "independent", "exchangeable", or
"unstructured" specifying the covariance structure for a multivariate
normal distribution; "none" indicates univariate normal distribution}
  \item{subHazard}{logical, if TRUE fits the subhazard frailty model}
  \item{alpha}{numeric, 100(1-alpha) percent confidence intervals}
  \item{MAX.ITER}{numeric, maximum number of iterations}
  \item{TOL}{numeric, tolerance limit}
}

