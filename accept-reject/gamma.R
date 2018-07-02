library(ggplot2)

get_M <- function (alpha, a, b) {
  m = b**(-a) * (  (alpha-a)/( (1-b) * exp(1) )  )**(alpha-a) 
  return (m)
}

target_dist <- function (x, alpha, beta) {
  # target distribution: Gamma(alpha, beta = 1), alpha not integer
  return ( dgamma (x, shape = alpha, rate = beta) )
}

proposed_dist <- function (x, a, b) {
  return ( dgamma(x, shape = a, rate = b) )
}

test <- function(sample, alpha, beta, a, b) {
  # Sample "u" from Unif(0,1)
  u = runif(1)
  ratio = target_dist(sample, alpha, beta) / 
    ( get_M(alpha, a, b) * proposed_dist(sample, a, b) )
  return (isTRUE(u < ratio))
}

generate_samples <- function(alpha = 2.63, n = 10000, beta = 1, floor = FALSE) {
  samples = c()
  # definition
  if (floor)
  {a = floor(alpha) + 10
  }
  else {
    a = ceiling(alpha)
  }

  # b to find best proposal (M minimum with f =< M*g)
  b = a/alpha
  
  its = 0
  while ((length(samples) < n) && (its < 10000)){
    its = its + 1
    # sample from Gamma(a, b)
    sample = rgamma(1, shape = a, rate = b)
    # if sample passes the test, add it to samples
    if ( isTRUE( test(sample, alpha, beta, a, b) ) ){
      samples = c(samples, sample)
    }
  }
  print (n/its)
  return (data.frame(samples))
}


# Testing
# Beta needs to be 1
N = 10000
ALPHA = 2.63
BETA = 1


gamma_samples = generate_samples(alpha = ALPHA, beta = BETA, n = N)
gg = ggplot(data = gamma_samples, aes(x = samples))+
  geom_histogram(binwidth = 0.1, aes(y = ..density..), color = "lightblue") +
  stat_function(fun = dgamma, args = list (shape = ALPHA, rate = BETA), color = "darkblue") +
  stat_function(fun = dgamma, args = list (shape = 3, rate = BETA), color = "red")

gg

