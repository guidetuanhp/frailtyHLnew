\name{frailtyHLnew-package}
\alias{frailtyHLnew-package}
\docType{package}
\title{H-likelihood Approach for Frailty Models}
\description{The frailtyHLnew package fits frailty models which are Cox's proportional hazards models incorporating
random effects. The package implements h-likelihood estimation procedures for lognormal, gamma, and AR(1)
frailty distributions using Laplace approximations when numerical integration is intractable
(Ha, Lee and Song, 2001; Ha and Lee, 2003, 2005; Lee, Nelder and Pawitan, 2017; Ha, Jeong and Lee, 2017).
It also supports competing-risk frailty models, multilevel AFT random-effect models, joint modelling of
longitudinal linear mixed models and survival frailty models, and penalized variable-selection procedures
(LASSO, SCAD and HL).}
\details{
\tabular{ll}{
Package: \tab frailtyHLnew\cr
Type: \tab Package\cr
Version: \tab 2.4.0\cr
Date: \tab 2026-11-23\cr
License: \tab GPL (>= 2)\cr
LazyLoad: \tab yes\cr
}
}
\author{Il Do Ha, Maengseok Noh, Jiwoong Kim, Youngjo Lee

Maintainer: Maengseok Noh <msnoh@pknu.ac.kr>
}
\references{
Ha, I. D. and Lee, Y. (2003). Estimating frailty models via Poisson Hierarchical generalized linear
models. Journal of Computational and Graphical Statistics, 12, 663-681.

Ha, I. D. and Lee, Y. (2005). Comparison of hierarchical likelihood versus orthodox best linear
unbiased predictor approaches for frailty models. Biometrika, 92, 717-723.

Ha, I. D., Lee, Y. and Song, J. K. (2001). Hierarchical likelihood approach for frailty models.
Biometrika, 88, 233-243.

Ha, I. D., Jeong, J. and Lee, Y. (2017). Statistical modelling of survival data with random effects.
Springer.

Lee, Y., Nelder, J. A. and Pawitan, Y. (2017). Generalised linear models with random effects:
unified analysis via h-likelihood. 2nd Edition. Chapman and Hall: London.}
\examples{
data(kidney, package = "frailtyHLnew")
kidney_g12 <- frailtyHL(Surv(time, status) ~ sex + age + (1 | id), kidney)
}
