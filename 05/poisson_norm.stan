data {
  int<lower=0> N;           // number of data points
  array[N] int<lower=0> y;        // observed counts
}

parameters {
  real theta;      // unconstrained parameter
}

transformed parameters {
    real<lower=0> exp_theta; // squashed parameter
    exp_theta = exp(theta);
}

model {
  // Normal prior 
  theta ~ normal(122, 9999);
  
  // Poisson likelihood
  y ~ poisson(exp_theta);
}
