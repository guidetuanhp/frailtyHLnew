\name{jointmodeling}
\alias{jointmodeling}
\title{Defining Fixed and Random Submodels for Joint and AFT Models}
\description{
\code{jointmodeling} constructs a submodel specification object defining the
response distribution, link function, and linear predictor (fixed and random
terms) for use in \code{mlmfit} and \code{jmfit}.
}
\usage{
jointmodeling(Model = "mean", RespDist = "gaussian", Link = NULL, LinPred = "constant",
RandDist = NULL, Offset = NULL)
}
\arguments{
  \item{Model}{Character string specifying the submodel component (\code{"mean"}, default).}
  \item{RespDist}{Distribution of the response variable: \code{"gaussian"} (linear mixed model),
\code{"AFT"} (accelerated failure time model), or \code{"FM"} (Cox frailty model).}
  \item{Link}{Link function for the linear predictor (\code{"identity"} for \code{"gaussian"},
\code{"log"} for \code{"AFT"} or \code{"FM"}).}
  \item{LinPred}{Formula specifying the response, fixed effects, and random-effect bars.}
  \item{RandDist}{Distribution of the random effects (\code{"gaussian"}).}
  \item{Offset}{Optional numeric offset vector.}
}
\value{
A five-element list storing the validated submodel specification (\code{Model},
\code{Link}, \code{LinPred}, \code{RandDist}, and \code{Offset}) with attribute
\code{"RespDist"} set to \code{RespDist}.
}
