from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from numpy import linalg as LA
# Loading data
#data = np.load('data2D.npy')
data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

K = 30
loss_func = 0
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


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    points_expanded = tf.expand_dims(X, 0)
    centroids_expanded = tf.expand_dims(MU, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    return tf.transpose(distances)

X = tf.constant(data, dtype=tf.float32)    
MU = tf.Variable(tf.random_normal(shape = (K, dim)), name = 'clusters', dtype=tf.float32)
squared_dist = distanceFunc(X, MU)

for c in range(K):
    assignments = tf.argmin(squared_dist, 1)
    where = tf.where(tf.equal(assignments, c))
    where = tf.transpose(where)
    points = tf.gather(squared_dist, where)
    size = tf.size(points)/K
    a = tf.reshape(points, [tf.cast(size, tf.int32),K])
    b = a[:,c]
    loss_func = loss_func + tf.reduce_sum(b)
    
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1, beta1 = 0.9, beta2 = 0.99, epsilon=1e-5) 
train = optimizer.minimize(loss_func)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_loss = []
    for iteration in range(num_iterations):
        _, loss, clusters, mu = sess.run([train, loss_func, assignments, MU])
        total_loss.append(loss)

num_cluster1 = 0 
num_cluster2 = 0 
num_cluster3 = 0 
num_cluster4 = 0 
num_cluster5 = 0

for cluster in clusters:
    if cluster == 0:
        num_cluster1 += 1
    elif cluster == 1:
        num_cluster2 += 1
    elif cluster == 2:
        num_cluster3 += 1
    elif cluster == 3:
        num_cluster4 += 1
    elif cluster == 4:
        num_cluster5 += 1
        
percent_cl1 =  num_cluster1 / num_pts * 100  
percent_cl2 =  num_cluster2 / num_pts * 100 
percent_cl3 =  num_cluster3 / num_pts * 100 
percent_cl4 =  num_cluster4 / num_pts * 100 
percent_cl5 =  num_cluster5 / num_pts * 100     
print('Percent belonging to cluster 1: {}, cluster 2: {}, cluster 3: {}, cluster 4: {}, cluster 5: {}'.format(percent_cl1, percent_cl2, percent_cl3, percent_cl4, percent_cl5))

################## Computing loss function for validation set ##################
dist_list = []
for row_m in mu:
    for row_d in val_data:
        dist_list.append((LA.norm(row_d-row_m))**2)
dist = np.array(dist_list)
dist = dist.reshape(K, valid_batch)
dist = dist.transpose()

valid_loss = 0
for c in range(K):
    assignments = np.argmin(dist, 1)
    where = np.where(np.equal(assignments, c))
    points = dist[where[0]]
    cluster_dist = points[:,c]
    valid_loss = valid_loss + cluster_dist.sum()

print("Validation loss: {}".format(valid_loss))

################################################################################
plt.plot(total_loss)
plt.ylabel('Loss')
plt.xlabel('Number of updates')
plt.show()

plt.scatter(data[clusters == 0, 0], data[clusters == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data[clusters == 1, 0], data[clusters == 1, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(data[clusters == 2, 0], data[clusters == 2, 1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(data[clusters == 3, 0], data[clusters == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(data[clusters == 4, 0], data[clusters == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(mu[:, 0], mu[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.show()