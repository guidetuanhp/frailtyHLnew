\name{CmpRsk}
\alias{CmpRsk}
\title{Model Formula of Competing Risk}
\description{
A \code{CmpRsk} object is used as the response variable in a competing-risk model formula.
It is created using \code{CmpRsk(time, index)}, where \code{time} is the event time and
\code{index} is an event-type indicator.
}
\usage{
CmpRsk(time, index)
}
\arguments{
  \item{time}{Numeric vector of event or censoring times.}
  \item{index}{Integer vector of event indicators; values must be sequential
non-negative integers where \code{0} denotes right censoring and positive
integers denote distinct competing event types.}
}
\value{
An object of class \code{"CmpRsk"}, implemented as a two-column numeric matrix
with columns \code{time} (observed event or censoring times) and \code{index}
(competing event indicators).
}
