from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math 
import helper as hlp

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
K = 30
num_iterations = 300

is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    points_expanded = tf.expand_dims(X, 0)
    centroids_expanded = tf.expand_dims(MU, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    distances = tf.transpose(distances)
    return distances

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    dist = distanceFunc(data, mu)
    sigma = tf.reshape(sigma, [1, K])
    dist = tf.cast(dist, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    norm_curve = tf.pow(1/tf.sqrt(2*math.pi*tf.square(sigma)), dim) * tf.exp((-dist)/(2*tf.square(sigma)))
    log_PDF = tf.log(norm_curve)
    # Outputs:
    # log Gaussian PDF N X K
    return log_PDF


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    log_pi = tf.reshape(log_pi, [1, K])
    PDF = tf.exp(log_PDF)
    pi = tf.exp(log_pi)
    # Outputs
    # log_post: N X K
    return log_pi + log_PDF - hlp.reduce_logsumexp(PDF*pi, 1, True)

    
X = tf.constant(data, dtype=tf.float32)    
MU = tf.Variable(tf.random_normal(shape = (K, dim)), name = 'mean', dtype=tf.float32)
PHI = tf.Variable(tf.random_normal(shape = (K, 1)), name = 'phi', dtype=tf.float32)
PSI = tf.Variable(tf.random_normal(shape = (K, 1)), name = 'psi', dtype=tf.float32)
dist = distanceFunc(X, MU)
PHI = tf.reshape(PHI, [1, K])
sigma_squared = tf.exp(PHI)
norm_curve = tf.pow(1/tf.sqrt(2*math.pi*sigma_squared), dim) * tf.exp((-dist)/(2*sigma_squared))
PI = tf.exp(hlp.logsoftmax(PSI))
PI = tf.reshape(PI, [1, K])

assignments = tf.argmax(tf.exp(log_posterior(tf.log(norm_curve), tf.log(PI))), 1)

loss = - tf.reduce_sum(hlp.reduce_logsumexp(PI*norm_curve))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.022, beta1 = 0.9, beta2 = 0.99, epsilon=1e-5) 
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_loss = []
    for iteration in range(num_iterations):
        _, loss_iteration, mu, sigmaSquared, pi, phi, psi, clusters = sess.run([train, loss, MU, sigma_squared, PI, PHI, PSI, assignments])
        total_loss.append(loss_iteration)
        
                
plt.plot(total_loss)
plt.ylabel('Loss')
plt.xlabel('Number of updates')
plt.show()

print('Best model parameters: MU = {}, sigma_squared = {}, pi = {}, phi = {}, psi = {}'.format(mu, sigmaSquared, pi, phi, psi))

plt.scatter(data[clusters == 0, 0], data[clusters == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
plt.scatter(data[clusters == 1, 0], data[clusters == 1, 1], s = 20, c = 'green', label = 'Cluster 2')
plt.scatter(data[clusters == 2, 0], data[clusters == 2, 1], s = 20, c = 'blue', label = 'Cluster 3')
plt.scatter(data[clusters == 3, 0], data[clusters == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')
plt.scatter(data[clusters == 4, 0], data[clusters == 4, 1], s = 20, c = 'magenta', label = 'Cluster 5')
plt.scatter(mu[:, 0], mu[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.show()


################## Computing loss function for validation set ##################
from numpy import linalg as LA

def logsumexp(input_matrix, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a matrix in log domain."""

  max_input_matrix1 = input_matrix.max(reduction_indices, keepdims=keep_dims)
  max_input_matrix2 = max_input_matrix1
  if not keep_dims:
    max_input_matrix2 = np.expand_dims(max_input_matrix2, reduction_indices)
  return np.log(
      np.sum(
          np.exp(input_matrix - max_input_matrix2),
          reduction_indices,
          keepdims=keep_dims)) + max_input_matrix1
                
                
dist_list = []
for row_m in mu:
    for row_d in val_data:
        dist_list.append((LA.norm(row_d-row_m))**2)
distance = np.array(dist_list)
distance = distance.reshape(K, valid_batch)
distance = distance.transpose()

val_norm_curve = np.power(1/np.sqrt(2*math.pi*sigmaSquared), dim) * np.exp((-distance)/(2*sigmaSquared))
val_loss = - np.sum(logsumexp(pi*val_norm_curve))

print('Training Loss converged to: {}'.format(total_loss[num_iterations-1]))
print("Validation loss: {}".format(val_loss))
################################################################################