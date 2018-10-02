import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parameter_posterior(sigma, prior_mu, prior_V, x0, y0):
	inv_prior_V = np.linalg.pinv(prior_V)

	# Parameter Gaussian posterior mean and covariance matrix
	V_N = sigma**2 * np.linalg.pinv(sigma**2 * inv_prior_V + np.dot(x0.T, x0))
	mu_N = np.dot(V_N, np.dot(inv_prior_V, prior_mu)) + 1. / sigma**2 * np.dot(V_N, np.dot(x0.T, y0))

	return mu_N, V_N

def fit_and_sample_models(sigma, prior_mu, prior_V, x0, y0, n_samples):
	# Parameter Gaussian posterior mean and covariance matrix
	mu_N, V_N = parameter_posterior(sigma, prior_mu, prior_V, x0, y0)

	# Sample from bivariate standard normal
	std_normal_samples = np.random.multivariate_normal([0., 0.], np.eye(2, dtype=float), (n_samples, 2))[:, :, 0]

	# Standard deviation matrix from covariance matrix
	sqrt_V_N = np.linalg.cholesky(V_N)

	# Apply covariance structure to samples from standard normal
	theta_samples = mu_N + np.dot(std_normal_samples, sqrt_V_N)

	return theta_samples

def predictive_posterior(x_star, mu_N, V_N, sigma):
	n = len(x_star)
	x_star_mod = np.vstack((np.ones(n), x_star)).T

	mu_star = np.dot(mu_N, x_star_mod.T)
	sigma_star = np.ones(len(x_star))

	for i in range(len(x_star)):
		sigma_star_no_noise = np.dot(x_star_mod[i], np.dot(V_N, x_star_mod[i].T))
		sigma_star[i] = sigma + sigma_star_no_noise

	return mu_star, sigma_star

cmap = cm.get_cmap('Set2')
c_model = cmap(0)
c_dots = cmap(1)
p_size = 8.

# Create priors
prior_V = np.eye(2, dtype=float)
prior_mu = np.array([0., 0.])

# Create dataset
N = [2, 4, 10, 60]
n_samples = 5
sigma = 7.
x0 = np.vstack((np.ones(N[-1]), np.random.uniform(-10, 10, N[-1]))).T
y0 = 1. - 2. * x0[:, 1] + np.random.normal(0., sigma, N[-1])

y_range = [-40, 40]

plt.figure(1)
fig1, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, n in zip(axes.ravel(), N):
	# Fit and sample parameters for univariate linear models.
	theta_samples = fit_and_sample_models(sigma, prior_mu, prior_V, x0[:n], y0[:n], n_samples)

	# Estimate y for each sampled model for samples x = {-10, 10}
	xsample = np.array([[1., -10.], [1., 10.]])
	Y = np.dot(theta_samples, xsample.T)

	# Plot models
	for y in Y:
		ax.plot(xsample[:, 1], y, c=c_model, alpha=0.3)

	# Plot dataset
	ax.scatter(x0[:n, 1], y0[:n], c=c_dots, s=p_size)
	print c_dots

	ax.set_ylim(y_range)
	ax.set_title('N = {}'.format(n))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)


# Posterior predictive
x_star = np.arange(-10, 10, 0.1)

plt.figure(2)
fig1, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, n in zip(axes.ravel(), N):
	mu_N, V_N = parameter_posterior(sigma, prior_mu, prior_V, x0[:n], y0[:n])

	y_star, sigma_star = predictive_posterior(x_star, mu_N, V_N, sigma)

	ax.plot(x_star, y_star, c=c_model)
	ax.plot(x_star, y_star + sigma_star, c=c_model, alpha=0.2)
	ax.plot(x_star, y_star - sigma_star, c=c_model, alpha=0.2)
	ax.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor=c_model, alpha=0.1)

	ax.scatter(x0[:n, 1], y0[:n], c=c_dots, s=p_size)

	ax.set_ylim(y_range)
	ax.set_title('N = {}'.format(n))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()