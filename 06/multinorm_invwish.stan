data {
  int<lower=1> n;             // Number of data points
  int<lower=2> d;             // Dimension of the data
  matrix[d, n] y;             // Data matrix
  int prior_wish;  // inverse-wishart hyperparameter
  vector[d] prior_mu;  //
  vector<lower=0>[d] diag_cov;  // Diagonal elements of the prior covariance matrix
}

parameters {
  vector[d] mu;                // mean
  cov_matrix[d] Sigma;          // covariance matrix
}
 

model {
  Sigma ~ inv_wishart(prior_wish, diag_matrix(diag_cov)); // Prior on the covariance matrix
  mu ~ multi_normal(prior_mu, Sigma);         // Prior on the mean

  // Likelihood
  for (i in 1:n) {
    y[,i] ~ multi_normal(mu, Sigma); // Likelihood of the data
  }

}

generated quantities {
  cov_matrix[d] prior_Sigma;
  prior_Sigma = inv_wishart_rng(prior_wish, diag_matrix(diag_cov));

  vector[d] prior_pd;
  prior_pd = multi_normal_rng(prior_mu, prior_Sigma);

  vector[d] y_tilde;
  y_tilde = multi_normal_rng(mu, Sigma);
}