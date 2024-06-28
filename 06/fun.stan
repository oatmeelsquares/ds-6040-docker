data {
  int<lower=1> n;             // Number of data points
  int<lower=2> d;             // Dimension of the data
  matrix[d,n] y;             // Data matrix
//  int eta;      // hyperparameter for lsk (eta > 1 for low correlations, < 1 for high correlations)
  vector[d] prior_mu;
}

parameters {
  vector[d] mu;
  cholesky_factor_corr[d] L_Omega;  // correlation matrix factor
  vector<lower=0>[d] L_std;   // standard deviations of each stock
}

transformed parameters {
  matrix[d,d] L_Sigma;           // correlation matrix cholesky factor
  L_Sigma = diag_post_multiply(diag_pre_multiply(L_std, L_Omega), L_std);   // multiply ch-factor of corr matrix by stdevs to get ch-factor of cov matrix
}

model {
  L_Omega ~ lkj_corr_cholesky(1);
  L_std ~ normal(0, 2.5); 
  mu ~ multi_normal_cholesky(prior_mu, L_Sigma);         // Prior on the mean

  // Likelihood
  for (i in 1:n) {
    y[,i] ~ multi_normal_cholesky(mu, L_Sigma); // Likelihood of the data based on prior mean and cov matrix
  }
}

generated quantities {
  // simulate from prior pd
  vector[d] prior_pd;
  prior_pd = multi_normal_cholesky_rng(prior_mu, L_Sigma);


  // simulate from post pd
  vector[d] y_tilde;
  y_tilde = multi_normal_cholesky_rng(mu, L_Sigma);
}
