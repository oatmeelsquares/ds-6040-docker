import scipy
import math

n = 400
y = 250
# how do we get the arguments for sc.beta?
a = y + 1
b = n - y + 1
log_norm_const = math.log(scipy.special.beta(a, b))

# "either"?
dependent_on_theta = False

# do not edit this cell!
import numpy as np
import pandas as pd

# Define the possible numbers on the roulette wheel
numbers = np.arange(0, 38)  # 0 to 36 for numbers, 37 for double zero
# Define the colors of the numbers
colors = ['green'] + ['red', 'black'] * 18  + ['green']

num_rows = 100
my_data = pd.DataFrame({'number':np.random.choice(numbers, num_rows)})
my_data['color'] = my_data.number.apply( lambda num : colors[num])
my_data.head()

# why 18/38??
# E(X) for Beta(a, b) is a / (a + b)
prior_hyperparam1 = 18
prior_hyperparam2 = 20

n = 100
y = my_data.query("color == 'red'").shape[0]
posterior_hyperparam1 = y + prior_hyperparam1
posterior_hyperparam2 = n - y + prior_hyperparam2
pos1 = posterior_hyperparam1
pos2 = posterior_hyperparam2

post = scipy.stats.beta(pos1, pos2)
my_interval = post.interval(.95)

sim_params = post.rvs(size = 1000)
post_pred_samples = [scipy.stats.binom(n, theta).rvs(1)[0] for theta in sim_params]
