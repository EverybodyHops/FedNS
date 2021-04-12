import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

SEED = 17

class model_values:
    def __init__(self):
        self.v_list = []

class model:
    def __init__(self, sess, learning_rate=1e-4):
        self.sess = sess

        # create network
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])
        self.y_conv = self.create_network(self.x)
        self.cross_entropy = -tf.reduce_sum(self.y_* tf.log(tf.clip_by_value(self.y_conv, 1e-10, 1.0)))
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        self.result_entropy = -tf.reduce_mean(self.y_conv * tf.log(tf.clip_by_value(self.y_conv, 1e-10, 1.0)))
        self.confidence = tf.reduce_mean(tf.reduce_max(self.y_conv, axis=1))

        # create support tensor
        self.all_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
        self.set_values_plh = []
        self.set_values_op = []
        for i, v in enumerate(self.all_variable):
            print(v)
            self.set_values_plh.append(tf.placeholder("float", shape=v.shape))
            self.set_values_op.append(tf.assign(v, self.set_values_plh[i]))

        # init
        self.sess.run(tf.initialize_all_variables())

    def create_network(self, x):
        def weight_variable(name, shape):
            return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(seed=SEED, stddev=0.1))

        def bias_variable(name, shape):
            return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("network"):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            W_conv1 = weight_variable('w_conv1', [5, 5, 1, 32])
            b_conv1 = bias_variable('b_conv1', [32])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable('w_conv2', [5, 5, 32, 64])
            b_conv2 = bias_variable('b_conv2', [64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

            W_fc1 = weight_variable('w_fc1', [7 * 7 * 64, 512])
            b_fc1 = bias_variable('b_fc1', [512])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            W_fc2 = weight_variable('w_fc2', [512, 10])
            b_fc2 = bias_variable('b_fc2', [10])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

            return y_conv

    def get_model_values(self):
        ret = model_values()
        for v in self.all_variable:
            ret.v_list.append(v.eval())
        return ret

    def set_model_values(self, model_v):
        self.sess.run(self.set_values_op, feed_dict={
            plh: v for plh, v in zip(self.set_values_plh, model_v.v_list)
        })

    def train(self, x_batch, y_batch):
        x_batch, y_batch = tflearn.data_utils.shuffle(x_batch, y_batch)
        self.sess.run(self.train_step, feed_dict={
            self.x: x_batch,
            self.y_: y_batch
        })

    def vaild(self, vaild_x, vaild_y):
        return self.accuracy.eval(feed_dict={
            self.x: vaild_x,
            self.y_: vaild_y
        })
    
    def get_result_entropy(self, x_batch):
        return self.result_entropy.eval(feed_dict={
            self.x: x_batch
        })

    def get_confidence(self, x_batch):
        return self.confidence.eval(feed_dict={
            self.x: x_batch
        })

if __name__ == "__main__":
    session = tf.InteractiveSession()
    m = model(sess=session)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(m.vaild(mnist.validation.images, mnist.validation.labels))

    for i in range(500):
        batch = mnist.train.next_batch(50)
        m.train(batch[0], batch[1])
        if i % 10 == 0:
            print(i)
            print("acc: ", m.vaild(mnist.validation.images, mnist.validation.labels))
            print("r_entropy:", m.get_result_entropy(mnist.train.next_batch(50)[0]))
        if i == 100:
            mv = m.get_model_values()

    m.set_model_values(mv)
    print(m.vaild(mnist.validation.images, mnist.validation.labels))
