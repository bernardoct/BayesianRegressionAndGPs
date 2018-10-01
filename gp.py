import numpy as np
import matplotlib.pyplot as plt


def parameter_posterior(sigma, prior_mu, prior_V, x0, y0):
	inv_prior_V = np.linalg.pinv(prior_V)

	# Parameter Gaussian posterior mean and covariance matrix
	V_N = sigma**2 * np.linalg.pinv(sigma**2 * inv_prior_V + np.dot(x0.T, x0))
	mu_N = np.dot(V_N, np.dot(inv_prior_V, prior_mu)) + 1. / sigma**2 * np.dot(V_N, np.dot(x0.T, y0))

	return mu_N, V_N

def fit_and_sample_models(sigma, prior_mu, prior_V, x0, y0):
	# Parameter Gaussian posterior mean and covariance matrix
	mu_N, V_N = parameter_posterior(sigma, prior_mu, prior_V, x0, y0)

	# Sample from bivariate standard normal
	std_normal_samples = np.random.multivariate_normal([0., 0.], np.eye(2, dtype=float), (5, 2))[:, :, 0]

	# Standard deviation matrix from covariance matrix
	sqrt_V_N = np.linalg.cholesky(V_N)

	# Apply covariance structure to samples from standard normal
	theta_samples = mu_N + np.dot(std_normal_samples, sqrt_V_N)

	return theta_samples

# Create priors
prior_V = np.eye(2, dtype=float)
prior_mu = np.array([0., 0.])

# Create dataset
N = 5
sigma = 10.
x0 = np.vstack((np.ones(N), np.random.uniform(-10, 10, N))).T
y0 = 1. - 2. * x0[:, 1] + np.random.normal(0., sigma, N)
# Fit and sample parameters for univariate linear models.
theta_samples = fit_and_sample_models(sigma, prior_mu, prior_V, x0, y0)

# Estimate y for each sampled model for samples x = {-10, 10}
xsample = np.array([[1., -10.], [1., 10.]])
Y = np.dot(theta_samples, xsample.T)

# Plot models
for y in Y:
	plt.plot(xsample[:, 1], y, c='r', alpha=0.3)

# Plot dataset
plt.scatter(x0[:, 1], y0)
plt.show()