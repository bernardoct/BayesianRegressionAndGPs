import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib.colors import rgb2hex


class GaussianProcesses:

	var_error = -1
	kernel = 'rbf'
	kernel_params = 1.
	mu = -1
	K = -1
	X = -1
	y = -1
	boundaries = []

	def __init__(self, var_error, boundaries, kernel='rbf', kernel_params=1.):
		self.var_error = var_error
		self.kernel_params=kernel_params
		if kernel == 'rbf':
			self.kernel = self._rbf_kernel
		elif kernel == 'linear':
			self.kernel = self._linear_kernel
		elif kernel == 'polynomial':
			self.kernel = self._polynomial
		else:
			'{} is not a valid kernel.'.format(kernel)

		self.boundaries = boundaries

	def _rbf_kernel(self, X, Z, kernel_params=1.):
		X = np.atleast_2d(X)
		Z = np.atleast_2d(Z)

		k_xz = np.exp(-0.5 * cdist(X.T / kernel_params, Z.T / kernel_params, metric='sqeuclidean'))

		return k_xz

	def _linear_kernel(self, X, Z, kernel_params=None):
		if len(X.shape) == 1:
			X = np.array([X])
		if len(Z.shape) == 1:
			Z = np.array([Z])

		k_xz = np.dot(X.T, Z)
		
		return k_xz

	def _polynomial(self, X, Z, kernel_params=1):
		if len(X.shape) == 1:
			X = np.array([X])
		if len(Z.shape) == 1:
			Z = np.array([Z])
		
		k_xz = (1. + np.dot(X.T, Z)) ** self.kernel_params

		return k_xz


	def add_data(self, X, y):
		if isinstance(self.X, int):
			self.X = X
			self.y = y
		else:
			self.X = np.hstack((self.X, X))
			self.y = np.hstack((self.y, y))

		self.K_xx = self.kernel(self.X, self.X, self.kernel_params) + self.var_error * np.eye(len(self.y))
		self.inv_K_xx = np.linalg.pinv(self.K_xx)

	def _get_predictive_distribution_params(self, X_star):
		if isinstance(self.X, int):
			K = self.kernel(X_star, X_star) 
			mu = np.zeros(len(X_star))
		else:
			K_sx = self.kernel(X_star, self.X)
			K_sx_inv_K_xx = np.dot(K_sx, self.inv_K_xx)
			
			mu = np.dot(K_sx_inv_K_xx, self.y)
			K = self.kernel(X_star, X_star) - np.dot(K_sx_inv_K_xx, K_sx.T)

		return mu, K

	def sample_functions(self, n_samples, resolution=100):
		x_star = np.arange(self.boundaries[0], self.boundaries[1], np.array(self.boundaries).ptp() / resolution)
		mu, K = self._get_predictive_distribution_params(x_star)
		
		return np.random.multivariate_normal(mu, K, n_samples), x_star

	def predict(self, X_star):
		mu, K = self._get_predictive_distribution_params(X_star)
		return mu, np.diagonal(K)

	def get_boundaries(self):
		return boundaries

boundaries = [-5., 5.]
gp = GaussianProcesses(0.001, boundaries, kernel='rbf', kernel_params=1.)

X = np.array(np.arange(boundaries[0], boundaries[1], np.ptp(boundaries) / 7)) + 0.5 + np.random.uniform(-0.2, 0.2, 7)
y = np.array(np.random.uniform(-1, 1, 7))

gp.add_data(X, y)
samples, x_samples = gp.sample_functions(15)

plt.figure(1)

# plt.figure(2)
cmap = cm.get_cmap('Set2')
x_star = np.arange(boundaries[0], boundaries[1], 0.05)
y_star, sigma_star = gp.predict(x_star)
sigma_star = 2. * sigma_star ** .5
plt.plot(x_star, y_star, c=cmap(0))
plt.plot(x_star, y_star + sigma_star, c=cmap(0), alpha=0.4)
plt.plot(x_star, y_star - sigma_star, c=cmap(0), alpha=0.4)
plt.fill_between(x_star, y_star + sigma_star, y_star - sigma_star, facecolor=cmap(0), alpha=0.2)
plt.scatter(X, y, c=cmap(1), s=20)

plt.figure(2)
for s in samples:
	plt.plot(x_samples, s, c=cmap(1), alpha=0.2)

plt.show()