data {
  int<lower=0> N;         // Number of observations
  array[N] int<lower=0, upper=1> y;  // Binary outcomes (0s and 1s)
  real mu_prior;           // Prior mean for theta
  real<lower=0> sigma_prior;        // Prior standard deviation for theta
}

parameters {
  real theta;   // Unconstrained parameter
}

transformed parameters {
    real<lower=0, upper=1> eta; // squashed parameter
    eta = inv_logit(theta);
}

model {
  // Normal prior
  theta ~ normal(mu_prior, sigma_prior);  // Prior distribution for theta

  // Binomial likelihood
  y ~ binomial(N, eta);  // Binomial likelihood
}

