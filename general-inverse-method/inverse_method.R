# Title: INVERSE DISTRIBUTION METHOD
# Author: SGARBI, GIORGIO

# Notation:
# f: function to which target distribution is proportional
library(ggplot2)

# cdf of distribution d
# image: [0,1]
cdf <- function(x, d, lower, upper) {
  if (x < lower) {
    return (0)
  }
  if (x > upper){
    return (1)
  }
  integral = integrate(d, lower = lower, upper = x,  subdivisions = 100L)
  value = integral$value
  return (value)
}

# inverse of f
# choose interval c(a, b) such as ( f(a) - a ) * (f(b) - b) < 0
# (Intermediate Value Theorem)
inv <- function(x, f, interval = c(-100, 100)){
  unir = uniroot(
      (function(y) f(y) - x),
      interval = interval,
      tol = 10^(-200))
  root = unir$root
  return (root)
}

#sample from distribution d
sampler <- function(f, N, lower, upper){
  samples = c()
  
  # normalize f
  integral = integrate(f,  lower, upper, subdivisions = 100L)
  norm = integral$value
  d <- function(x) f(x)/norm
  
  # INVERSE METHOD SAMPLING
  # sample from uniform
  u = runif(N)
  
  # inverse of CDF of d, i.e, F^-1(x)
  inv_cdf = function(a) inv(a, function(x) cdf(x, d, lower, upper))
  
  # inverse of CDF of d at u, i.e, F^-1(u)
  s = sapply(X = u, FUN = inv_cdf) 
  
  samples = c(samples, s)
  return (samples)
}

# plot samples and target distribution
plot_ <- function(f, N = 10000, lower, upper) {
  # normalize f
  integral = integrate(f, lower, upper)
  norm = integral$value
  d <- function(x) f(x)/norm
  
  samples = sampler (f, N, lower, upper)
  
  #plot
  base = ggplot(data = data.frame(samples), aes(x = samples))
  gg = base +
    geom_histogram(binwidth = 0.05, aes(y = ..density..), color = "lightblue") +
    stat_function(fun = d, color = "orange", lwd = 1.2)
  gg
}

# Examples:
standard_norm <- function(x) dnorm(x)

# Laplace not normalized
laplace <- function(x) exp(-abs(x))

# Piecewise Exponential (application in Adaptive Rejection Sampling)
f <- function(x) {
  lower = -1
  upper = 1
  is0 = sapply( x, function(i) isTRUE(i < lower || i > upper))
  result = ifelse(is0, 0, exp(2*x))
  return (result)
}

# Gamma(2.3, 4.1)
gamma <- function(x) {
  lower = 0
  upper = Inf
  is0 = sapply( x, function(i) isTRUE(i < lower || i > upper))
  result = ifelse(is0, 0, x^(2.3-1)*exp(-4.1*x))
  return (result)
}

#plot examples
plot_(standard_norm, 5000, -10, 10)
plot_(laplace, 5000, -10, 10)
plot_(f, 5000, -1, 1)
plot_(gamma, 5000, 0, 20)






