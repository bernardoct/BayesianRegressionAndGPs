import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def polynomial(x0):
	x0_prime = np.array([x0**4, 4.*x0**3, 8.*x0, np.ones(len(x0))]).T
	y = x0**4 - 4.*x0**3 + 8.*x0 + 1
	return x0_prime, y

def linear(x0):
	x0_prime = np.array([x0, np.ones(len(x0))]).T
	y = x0 + 1
	return x0_prime, y

def parameter_posterior(sigma, prior_mu, prior_V, x0, y0):
	inv_prior_V = np.linalg.pinv(prior_V)

	# Parameter Gaussian posterior mean and covariance matrix
	XTX = np.dot(x0.T, x0)
	V_N = sigma**2 * np.linalg.pinv(sigma**2 * inv_prior_V + XTX)
	mu_N = np.dot(V_N, np.dot(inv_prior_V, prior_mu)) + 1. / sigma**2 * np.dot(V_N, np.dot(x0.T, y0))

	return mu_N, V_N

def fit_and_sample_models(sigma, prior_mu, prior_V, x0, y0, n_samples):
	n_features = len(x0[0])

	# Parameter Gaussian posterior mean and covariance matrix
	mu_N, V_N = parameter_posterior(sigma, prior_mu, prior_V, x0, y0)

	# Sample from bivariate standard normal
	std_normal_samples = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features, dtype=float), (n_samples, n_features))[:, :, 0]

	# Standard deviation matrix from covariance matrix
	sqrt_V_N = np.linalg.cholesky(V_N)

	# Apply covariance structure to samples from standard normal
	theta_samples = mu_N + np.dot(std_normal_samples, sqrt_V_N)

	return theta_samples

def predictive_posterior(x_star, mu_N, V_N, sigma, function):
	n = len(x_star)
	x_star_mod, _ = function(x_star)

	mu_star = np.dot(mu_N, x_star_mod.T)
	sigma_star = np.ones(len(x_star))

	for i in range(len(x_star)):
		sigma_star_no_noise = np.dot(x_star_mod[i], np.dot(V_N, x_star_mod[i].T))
		sigma_star[i] = sigma + sigma_star_no_noise

	return mu_star, sigma_star

def fit_and_plot_sample_models(x_star, function, sigma, prior_mu, prior_V, x0_prime, x0, y0, N, n_model_samples, c_model, c_dots, p_size, y_range, bounds):
	# Generate new polynomial features (x1, x2, ...)
	x_star_prime, _ = function(x_star)

	# Plot sample models
	plt.figure(1)
	fig1, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	for ax, n in zip(axes.ravel(), N):
		# Fit and sample parameters for univariate linear models.
		theta_samples = fit_and_sample_models(sigma, prior_mu, prior_V, x0_prime[:n], y0[:n], n_model_samples)

		# Estimate y for each sampled model for each sample x (sample with added function features)
		Y = np.dot(theta_samples, x_star_prime.T)

		# Plot models
		for y in Y:
			ax.plot(x_star, y, c=c_model, alpha=0.3)

		# Plot dataset
		ax.scatter(x0[:n], y0[:n], c=c_dots[:3], s=p_size)

		ax.set_ylim(y_range)
		ax.set_xlim(bounds)
		ax.set_title('N = {}'.format(n))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

def fit_and_plot_posterior_predictive(x_star, function, sigma, prior_mu, prior_V, x0_prime, x0, y0, c_model, c_dots, p_size, y_range, bounds):
	# Plot posterior predictive
	plt.figure(2)
	fig1, axes = plt.subplots(2, 2, sharex=True, sharey=True)
	for ax, n in zip(axes.ravel(), N):
		mu_N, V_N = parameter_posterior(sigma, prior_mu, prior_V, x0_prime[:n], y0[:n])

		y_star, sigma_star = predictive_posterior(x_star, mu_N, V_N, sigma, function)

		ax.plot(x_star, y_star, c=c_model)
		ax.plot(x_star, y_star + sigma_star, c=c_model, alpha=0.2)
		ax.plot(x_star, y_star - sigma_star, c=c_model, alpha=0.2)
		ax.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor=c_model, alpha=0.1)

		ax.scatter(x0[:n], y0[:n], c=c_dots[:3], s=p_size)

		ax.set_ylim(y_range)
		ax.set_xlim(bounds)
		ax.set_title('N = {}'.format(n))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)


# Create priors
n_features = 2
function = linear
prior_V = np.eye(n_features, dtype=float)
prior_mu = np.zeros(n_features)

# Create dataset
N = [2, 4, 10, 50]
bounds = [-1.3, 3.3]
sigma = 0.5

x0 = np.random.uniform(bounds[0], bounds[1], N[-1])
x0_prime, y0 = function(x0)
y0 += np.random.normal(0, sigma, N[-1])

# Sample 100 points at regulat intervals
x_star = np.arange(bounds[0], bounds[1], 0.01 * (bounds[1] - bounds[0]))

# Set plot parameters

cmap = cm.get_cmap('Set2')
c_model = cmap(0)
c_dots = cmap(1)

p_size = 8.
y_range = [-2, 6]
n_model_samples = 5

fit_and_plot_sample_models(x_star, function, sigma, prior_mu, prior_V, x0_prime, x0, y0, N, n_model_samples, c_model, c_dots, p_size, y_range, bounds)
fit_and_plot_posterior_predictive(x_star, function, sigma, prior_mu, prior_V, x0_prime, x0, y0, c_model, c_dots, p_size, y_range, bounds)

plt.show()