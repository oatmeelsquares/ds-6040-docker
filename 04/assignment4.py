
import numpy as np
import pandas as pd

prior_hyperparams = np.full(38, 1)
prior_hyperparams

# Define the possible numbers on the roulette wheel
numbers = np.arange(0, 38)  
colors = ['green'] + ['red', 'black'] * 18  + ['green']

# simulate some data
num_days = 100
num_spins_per_day = 100
num_rows = num_days*num_spins_per_day
my_data = pd.DataFrame({'number':np.random.choice(numbers, num_rows)})
my_data['color'] = my_data.number.apply( lambda num : colors[num])
my_data['day'] = np.repeat(np.arange(1,(num_spins_per_day+1)),num_days)
my_data



my_data.color = 1
y_data = np.full((100, 38), my_data.pivot_table(columns = 'number', index = 'day', aggfunc = 'sum', fill_value = 0))

post_hyperparams = np.array([alpha + ysum for alpha, ysum in zip(prior_hyperparams, y_data.sum(axis = 0))]).transpose()


best_bet = np.argmax(post_hyperparams) + 1


adj_prices = pd.read_csv('stocks.csv', index_col = 'Date')
rets = adj_prices.pct_change().iloc[1:,]*100
#from matplotlib.pyplot import scatter
#scatter(rets.SPY, rets.QQQ)



from scipy.stats import invwishart
from scipy.stats import multivariate_normal

def sim_data(nu0, Lambda0, mu0, kappa0, num_sims):
    pi_Sigma = invwishart(df = nu0, scale = Lambda0)
    sigma_samples = pi_Sigma.rvs(num_sims)
    mu_samples = np.array([multivariate_normal(mean = mu0,
                                      cov = Sigma/kappa0
                                     ).rvs(1)
                  for Sigma in sigma_samples])
    theta_samples = zip(mu_samples, sigma_samples)
    fake_y = np.array([multivariate_normal(mu, sigma).rvs(1)
              for mu, sigma in theta_samples])
    #scatter(fake_y[:,0], fake_y[:, 1])
    return fake_y

# Pick a prior by assigning specific values to `nu0`, `Lambda0`, `mu0` and `kappa0`. Use your `sim_data()` function to choose wisely.
nu0 = 10
Lambda0 = np.array([[10, 9], [9, 10]])
mu0 = np.array([0, 0])
kappa0 = 1
sim_data(nu0, Lambda0, mu0, kappa0, 1000)

n = len(rets)
nu_n = nu0 + n
kappa_n = kappa0 + n

ybar = rets.mean().values
mu_n = mu0 * kappa0 / (kappa0 + n) + ybar * n / (kappa0 + n)
S = (rets.transpose() @ rets).values

ybar_minus_mu0 = (ybar - mu0).reshape(2,1)
Lambda_n = (Lambda0 + S + ybar_minus_mu0 @ybar_minus_mu0.transpose() * kappa0 * n / (kappa0 + n))



# uncomment after you have a working implementation of sim_data()!
post_pred_sims = sim_data(nu_n, Lambda_n, mu_n, kappa_n, 1239)


def get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma, s = 1):
    k = len(mu_n)
    post_mean = mu_n
    post_var = Lambda_n / (nu_n - k - 1)*(1 + 1/kappa_n)
    V_inv = np.linalg.inv(post_var)
    ones = np.repeat(1, k)
    q1 = ones.transpose() @ V_inv @ post_mean
    q2 = ones.transpose() @ V_inv @ ones
    le_fraction = (q1 - gamma*s)/q2
    return V_inv @ (post_mean - le_fraction * ones) / gamma
    
best_weights = get_weights(nu_n, Lambda_n, mu_n, kappa_n, gamma=10, s = 1)
best_weights

