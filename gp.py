import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from matplotlib.colors import rgb2hex


class BayesianRegression:
	X = -1
	phi_X = -1
	y = -1
	prior_mu = -1
	prior_V = -1
	n_features = -1
	inv_prior_V = -1
	inv_prior_V_dot_prior_mu = -1
	var_error = -1
	function = -1
	cmap = cm.get_cmap('Set2')

	def __init__(self, var_error, prior_mu, prior_V, function=lambda x: np.array([[1, xx] for xx in x])):
		self.var_error = var_error
		self.prior_mu = prior_mu
		self.prior_V = prior_V
		self.n_features = len(prior_mu)
		self.inv_prior_V = np.linalg.pinv(prior_V)
		self.inv_prior_V_dot_prior_mu =  np.dot(self.inv_prior_V, self.prior_mu)
		self.function = function

	def add_observations(self, x, y):
		x, y = np.array(x), np.array(y)
		if isinstance(self.X, int):
			self.X = x
			self.phi_X = np.array(self.function(x))
			self.y = np.array(y)
		else:
			self.X = np.hstack((self.X, x))
			self.phi_X = np.vstack((self.phi_X, self.function(x)))
			self.y = np.hstack((self.y, y))

		# Parameter Gaussian posterior mean and covariance matrix
		XTX = np.dot(self.phi_X.T, self.phi_X)
		self.V_N = self.var_error**2 * np.linalg.pinv(self.var_error**2 * self.inv_prior_V + XTX)
		self.mu_N = np.dot(self.V_N, self.inv_prior_V_dot_prior_mu) + 1. / self.var_error**2 * np.dot(self.V_N, np.dot(self.phi_X.T, self.y))

		return self.V_N, self.mu_N

	def sample_models_parameter_posterior(self, n_samples):
		# Sample from bivariate standard normal
		std_normal_samples = np.random.multivariate_normal(np.zeros(self.n_features), np.eye(self.n_features, dtype=float), (n_samples, self.n_features))[:, :, 0]

		# Standard deviation matrix from covariance matrix
		sqrt_V_N = np.linalg.cholesky(self.V_N)

		# Apply covariance structure to samples from standard normal
		theta_samples = self.mu_N + np.dot(std_normal_samples, sqrt_V_N)

		return theta_samples

	def predictive_posterior(self, x_star):
		n = len(x_star)
		x_star_mod = self.function(x_star)

		mu_star = np.dot(self.mu_N, x_star_mod.T)
		sigma_star = np.ones(len(x_star))

		for i in range(len(x_star)):
			sigma_star_no_noise = np.dot(x_star_mod[i], np.dot(self.V_N, x_star_mod[i].T))
			sigma_star[i] = self.var_error + sigma_star_no_noise

		return mu_star, sigma_star

	def plot_sample_models_1d(self, ax, n_samples, c_model=cmap(0), c_data=cmap(1), alpha=0.4):
		x_star = np.arange(min(self.X) - np.ptp(self.X) * 0.1, max(self.X) + np.ptp(self.X) * 0.1, 1e-2 * np.ptp(self.X))
		theta_samples = self.sample_models_parameter_posterior(n_samples)
		for s in theta_samples:
			ax.plot(x_star, np.dot(self.function(x_star), s), c=c_model, alpha=alpha)

		ax.scatter(self.X, self.y, c=rgb2hex(c_data), s=8)

	def plot_predictive_model(self, ax, c_model=cmap(0), c_data=cmap(1), alpha=0.6):
		x_star = np.arange(min(self.X) - np.ptp(self.X) * 0.1, max(self.X) + np.ptp(self.X) * 0.1, 1e-2 * np.ptp(self.X))
		phi_x_star = self.function(x_star)

		y_star, sigma_star = self.predictive_posterior(x_star)

		ax.plot(x_star, y_star, c=c_model)
		ax.plot(x_star, y_star + sigma_star, c=c_model, alpha=0.4)
		ax.plot(x_star, y_star - sigma_star, c=c_model, alpha=0.4)
		ax.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor=c_model, alpha=0.2)

		ax.scatter(self.X, self.y, c=rgb2hex(c_data), s=8)

	def plot_parameter_distributions(self, cmap=cm.get_cmap('magma'), savefig=''):
		fig, axes = plt.subplots(self.n_features, self.n_features)
		for i in range(self.n_features):
			for j in range(self.n_features):
				if i != j:
					xlim = [self.mu_N[i] + 15. * self.V_N[i, i], self.mu_N[i] - 15. * self.V_N[i, i]]
					ylim = [self.mu_N[j] + 15. * self.V_N[j, j], self.mu_N[j] - 15. * self.V_N[j, j]]
					beta1, beta2 = np.mgrid[xlim[0]:xlim[1]:(xlim[1] - xlim[0]) / 100, ylim[0]:ylim[1]:(ylim[1] - ylim[0]) / 100]
					pos = np.empty(beta1.shape + (2,))
					pos[:, :, 0] = beta1; pos[:, :, 1] = beta2

					# Create multivariate normal from model's mean vector and covariance matrix.
					rows_cols = np.array([i, j])
					rv = multivariate_normal(self.mu_N[[i, j]], self.V_N[rows_cols[:, None], rows_cols])
					z = rv.pdf(pos)
					axes[i, j].contourf(beta1, beta2, z, 50, cmap=cmap)
					axes[i, j].get_xaxis().set_visible(False)
					axes[i, j].get_yaxis().set_visible(False)

		if len(savefig) > 0:
			plt.savefig(savefig)
		else:
			plt.show()

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
	var_error = 2.5

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

	plt.show()

	# plot parameter distirbution after all data is added
	regression.plot_parameter_distributions()


run_bayesian_regression_experiment()
# run_gaussian_processes_experiment()

