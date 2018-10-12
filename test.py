import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.insert(0, './HydroBayesOpt')
from bayes_regression import BayesianRegression
from gp import GaussianProcesses
from matplotlib.colors import rgb2hex
np.random.seed(3)

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

def run_gaussian_processes_experiment():
	boundaries = [-5., 5.01]
	gp = GaussianProcesses(0., boundaries, kernel='rbf', kernel_params=1.)
	n_samples = 5
	dx = 2

	X = np.array(np.arange(boundaries[0], boundaries[1], np.ptp(boundaries) / n_samples)) + 0.5 + np.random.uniform(-0.2, 0.2, n_samples)
	y = np.array(np.random.uniform(-1, 1, n_samples))

	gp.add_data(X, y)
	samples, x_samples = gp.sample_functions(3, (boundaries[1] - boundaries[0]) / dx)

	plt.figure(1)

	# plt.figure(2)
	cmap = cm.get_cmap('Set2')
	x_star = np.arange(boundaries[0], boundaries[1], dx)
	y_star, sigma_star = gp.predict(x_star)
	# sigma_star = 2. * sigma_star ** .5
	plt.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor='lightgrey', label='$\mu \pm \sigma$')
	plt.plot(x_star, y_star, c='r', label='$\mu$')
	plt.plot(x_star, y_star + sigma_star, c='darkgrey', alpha=0.4)
	plt.plot(x_star, y_star - sigma_star, c='darkgrey', alpha=0.4)
	plt.scatter(X, y, c=rgb2hex(cmap(2)), s=20)
	plt.ylim([-1.5, 1.5])
	plt.xlim(boundaries)
	plt.legend()
	plt.xlabel('x')

	plt.figure(2)
	plt.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor='lightgrey', label='$\mu \pm \sigma$')
	plt.plot(x_star, y_star, c='darkgrey')
	plt.plot(x_star, y_star + sigma_star, c='darkgrey', alpha=0.4)
	plt.plot(x_star, y_star - sigma_star, c='darkgrey', alpha=0.4)
	colors = [cmap(i) for i in range(3)]
	for s, c in zip(samples, colors):
		plt.plot(x_samples, s, c=c)#, alpha=0.2)
	plt.ylim([-1.5, 1.5])
	plt.xlim(boundaries)
	plt.legend()
	plt.xlabel('y')

	plt.show()

# run_bayesian_regression_experiment()
run_gaussian_processes_experiment()