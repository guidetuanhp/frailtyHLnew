\name{jmfit}
\alias{jmfit}
\title{Joint Modelling of Longitudinal and Time-to-Event Data}
\description{
\code{jmfit} fits joint models of longitudinal repeated measurements and
time-to-event survival data (including bivariate or competing-event frailty
models) via hierarchical likelihood.
}
\usage{
jmfit(jm, data, jm2 = NULL, data2 = NULL, Maxiter = 200)
}
\arguments{
  \item{jm}{A \code{jointmodeling} object (for bivariate frailty models) or a list of
two or three \code{jointmodeling} objects specifying the longitudinal and survival submodels.}
  \item{data}{A data frame (when \code{jm2} is supplied) or a list of data frames containing
the longitudinal and survival datasets used in \code{jm}.}
  \item{jm2}{Optional second \code{jointmodeling} object when fitting a bivariate survival frailty model.}
  \item{data2}{Optional data frame corresponding to \code{jm2}.}
  \item{Maxiter}{Maximum number of iterations (default \code{200}).}
}
\value{
A named list containing:
\item{F.Est}{Matrix of fixed-effect parameter estimates, standard errors, t-values, and p-values across submodels.}
\item{D.Est}{Matrix (or vector) of dispersion, frailty variance, and association parameter estimates.}
\item{V.Est}{Estimated shared subject-level random effects.}
\item{iter}{Number of iterations performed.}
\item{convergence}{Final parameter-change convergence norm.}
\item{converged}{Logical flag indicating whether convergence was achieved within \code{Maxiter}.}
}
