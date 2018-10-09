import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './HydroBayesOpt')
from bayes_regression import BayesianRegression

def run_bayesian_regression_experiment():
	# Two models
	function_linear = lambda x: np.array([[1, xx] for xx in x])
	function_polynomial = lambda x: np.array([[xx**4, 4.*xx**3, 8.*xx, 1] for xx in x])

	n_features = 4
	function = function_polynomial

	# Create priors
	prior_V = np.eye(n_features, dtype=float)
	prior_mu = np.zeros(n_features)

	# Create dataset
	N = [3, 5, 15, 50]
	bounds = [-4., 2.]
	var_error = 5.

	# Generate random data set
	x0 = np.random.uniform(bounds[0], bounds[1], N[-1])
	y0 = np.sum(function(x0), axis=1)
	y0 += np.random.normal(0, var_error, N[-1])

	# Create regression object
	regression = BayesianRegression(var_error, prior_mu, prior_V, function=function_polynomial)

	# Add data to regression model in stages and sample model from parameter posterior
	fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	n_previous = 0
	for ax, n in zip(axes.ravel(), N):
		regression.add_observations(x0[n_previous:n], y0[n_previous:n])
		n_previous = n
		regression.plot_sample_models_1d(ax, 5)
		ax.set_title('N={}'.format(n))

	# Create new regression object
	regression = BayesianRegression(var_error, prior_mu, prior_V, function=function_polynomial)

	# Add data to regression model in stages and plot model mean and error variance
	plt.figure(1)
	fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	n_previous = 0
	for ax, n in zip(axes.ravel(), N):
		regression.add_observations(x0[n_previous:n], y0[n_previous:n])
		n_previous = n
		regression.plot_predictive_model(ax)
		ax.set_title('N={}'.format(n))

	# plot parameter distirbution after all data is added
	plt.figure(2)
	regression.plot_parameter_distributions()

	plt.show()


run_bayesian_regression_experiment()
# run_gaussian_processes_experiment()