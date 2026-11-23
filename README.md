# frailtyHLnew

Hierarchical-likelihood (`h`-likelihood) estimation for survival frailty models
with a unified C/OpenMP numerical backend:

- **Cox Frailty Models (`frailtyHL`)**:
  - Gamma frailty (single & multiple random effects, right-censored & left-truncated): `fhl_fit_cox_gamma`
  - Log-normal frailty (single & multiple random effects, right-censored & left-truncated): `fhl_fit_cox_normal`
  - AR(1) frailty (tridiagonal precision): `fhl_fit_cox_ar1`
- **Competing Risks (`CmpRsk`, `hlike.frailty`)**: `fhl_fit_competing_risks`
- **Multilevel AFT Mixed Models (`mlmfit`, `HL(0,1)` & `HL(1,1)`)**: `fhl_fit_aft_1re`, `fhl_fit_aft_2re`
- **Joint Longitudinal-Survival & Bivariate Frailty Models (`jointmodeling`, `jmfit`)**: `fhl_fit_jm_cox1`, `fhl_fit_jm_cox2`, `fhl_fit_jm_frailty`
- **Penalized Variable Selection (`frailty.vs` — LASSO, SCAD, HL)**: `fhl_fit_vs_1re`, `fhl_fit_vs_2re`

See [`inst/architecture.md`](inst/architecture.md) for mathematical formulations,
linear-algebra invariants, and native module organization.

## Installation

Install pre-built binaries (Windows / macOS) or source (Linux) directly from R-universe without requiring local compiler setup on Windows/macOS:

```r
install.packages(
  "frailtyHLnew",
  repos = c("https://guidetuanhp.r-universe.dev", "https://cloud.r-project.org")
)
```

Or compile from GitHub source:

```r
# install.packages("remotes")
remotes::install_github("guidetuanhp/frailtyHLnew")
```

## Quick Start

```r
library(frailtyHLnew)
data(kidney, package = "frailtyHLnew")

fit <- frailtyHL(
  Surv(time, status) ~ sex + age + (1 | id),
  data = kidney,
  RandDist = "Normal",
  mord = 0,
  dord = 1
)
summary(fit)
```
