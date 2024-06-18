import scipy
import numpy as np
import pandas as pd

a = 10
b = 11
n = 42
s = 15
a2 = a + n / 2
b2 = b + n * s / 2
log_norm_const =  scipy.special.gammaln(a2) - a2 * np.log(b2) 
dependent_on_theta = False

prior_a = 8
prior_b = .5
prior = scipy.stats.invgamma(prior_a, scale = prior_b)

prior_param_samples = prior.rvs(10000)

prior_predic_samps = np.empty(1000000)
i = 0
for x in [scipy.stats.norm(0, theta).rvs(100) for theta in prior_param_samples]:
    for y in x:
        prior_predic_samps[i] = y
        i += 1


a = 10
b = 11
n = 42
xbar = 15
post_mean = xbar * (n / (1/b + n)) + a * (1/b / (1/b + n))
post_var = 1 / (1/b + n)
log_norm_const2 = -(np.log(1) - np.log(post_var)/2 - np.log(2 * np.pi)/2)

stock_data = pd.read_csv('SPY-STK.csv', usecols = ['time', 'bid_price_close'])

one_day_returns = stock_data.bid_price_close.pct_change()*100

a = prior_a
b = prior_b
n = one_day_returns.shape[0]
s = np.mean(one_day_returns ** 2)

posterior_a = a + n / 2
posterior_b = b + n * s / 2
posterior = scipy.stats.invgamma(posterior_a, scale = posterior_b)

theta_samples = posterior.rvs(10000)
post_pred_samps = np.full(10000, [scipy.stats.norm(0, theta).rvs(1)[0] for theta in theta_samples])

