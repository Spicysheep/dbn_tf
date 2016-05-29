import tensorflow as tf
import numpy as np
import input_data
from PIL import Image
from util import tile_raster_images

# This is a helpful article for understanding what all of the
# rbm pieces are: http://deeplearning4j.org/restrictedboltzmannmachine.html

def sample_prob(probs): # I"m not sure what the sampling is for
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

alpha = 1.0 # I'm not sure what alpha is for
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Only the inputs, train_X, are used here since this is 
# unsupervised learning. No targets are used. The test 
# set is alos not used
train_X = mnist.train.images

X = tf.placeholder("float", [None, 784])

neuron_count = 10

rbm_w = tf.placeholder("float", [784, neuron_count])
rbm_visible_bias = tf.placeholder("float", [784]) # v stands for visible?
rbm_hb = tf.placeholder("float", [neuron_count]) # h stands for hidden?

h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb)) # I think this is the first hidden layer, going from visible/input to hidden
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_visible_bias)) # This could be the hidden layer going to the visibl layer
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb) # Why is this third activation necessary?

w_positive_grad = tf.matmul(tf.transpose(X), h0) # How is this a gradient? Shouldn't subtraction occur somewhere?
w_negative_grad = tf.matmul(tf.transpose(v1), h1) # How is this a gradient? Shouldn't subtraction occur somewhere?

update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_visible_bias + alpha * tf.reduce_mean(X - v1, 0) # This seems more like a gradient
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)

h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_visible_bias))
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# 'n' means new and 'o' means old?
n_w = np.zeros([784, neuron_count], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([neuron_count], np.float32)
o_w = np.zeros([784, neuron_count], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([neuron_count], np.float32)

print(sess.run(err_sum, feed_dict={X: train_X, rbm_w: o_w, rbm_visible_bias: o_vb, rbm_hb: o_hb}))
for start, end in zip(range(0, len(train_X), batchsize), range(batchsize, len(train_X), batchsize)):
    batch = train_X[start:end]
    n_w = sess.run(update_w, feed_dict={X: batch, rbm_w: o_w, rbm_visible_bias: o_vb, rbm_hb: o_hb})
    n_vb = sess.run(update_vb, feed_dict={X: batch, rbm_w: o_w, rbm_visible_bias: o_vb, rbm_hb: o_hb})
    n_hb = sess.run(update_hb, feed_dict={X: batch, rbm_w: o_w, rbm_visible_bias: o_vb, rbm_hb: o_hb})
    o_w = n_w
    o_vb = n_vb
    o_hb = n_hb
    if start % 10000 == 0:
        print(sess.run(err_sum, feed_dict={X: train_X, rbm_w: n_w, rbm_visible_bias: n_vb, rbm_hb: n_hb}))
        # If you provide too many tiles you get blank ones at the end
        pictures_tall = 5
        pictures_wide = 2
        image = Image.fromarray(
            tile_raster_images(
                X=n_w.T,
                img_shape=(28, 28),
                tile_shape=(pictures_tall, pictures_wide),
                tile_spacing=(1, 1)
            )
        )
        image.save("rbm_%d.png" % (start / 10000))


