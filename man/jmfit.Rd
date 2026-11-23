\name{jmfit}
\alias{jmfit}
\title{Joint Modelling of Longitudinal and Time-to-Event Data}
\description{
jmfit is used to fit joint modelling of longitudinal and time-to-event 
data by using h-likelihood. The response of interest would involve repeated measurements
over time on the same subject as well as time to an event of interest
with or without competing risks.
}
\usage{
jmfit(jm, data, jm2 = NULL, data2 = NULL, Maxiter = 200)
}
\arguments{
  \item{jm}{list of \code{jointmodeling} objects which specify the first reponses of interest.}
  \item{data}{list of dataframes containing the variables used in the jm.}
  \item{jm2}{list of \code{jointmodeling} object which specifies the second reponses.}
  \item{data2}{dataframes containing the variables used in the jm2.}
  \item{Maxiter}{numeric, maximum number of iterations; default=200.}
}

