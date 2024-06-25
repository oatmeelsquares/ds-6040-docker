data {
  int<lower=0> N;           // number of data points
  array[N] int<lower=0> y;        // observed counts
}

parameters {
  real<lower=0> theta;      // rate parameter for Poisson
}

model {
  // Log-normal prior 
  theta ~ lognormal(122, 9999);
  
  // Poisson likelihood
  y ~ poisson(theta);
}