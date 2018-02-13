import tensorflow as tf
import numpy as np

# === network parameters ===
training_iterations = 50000
learn_rate = 1e-4
batch_size = 1000


# === input and output ===

ideal_m = 5.14
ideal_b = 2.111

def new_data():
    x = np.random.uniform(-10, 10, size=(batch_size, 1))
    y = ideal_m * x + ideal_b
    # add noise
    y = y + np.random.normal(0, 0.4, size=(batch_size, 1))
    return x, y



# === model ===

# input
x = tf.placeholder(tf.float32, shape=[None, 1])
# correct answer
y_label = tf.placeholder(tf.float32, shape=[None, 1])

# slope and y intercept
# these values are the ones being "learned"
m = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)

# linear equation
# y is the output
y = tf.add(tf.matmul(x, m), b)

# error calculation
error = tf.reduce_mean(tf.square(y_label - y))

# training function
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(error)

# run the moder and learn
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(training_iterations):
    _x, _y = new_data()
    sess.run(train, feed_dict={x: _x, y_label: _y})

    # print step
    if i % 1000 == 0:
        print('m: %f b: %f error:%f' %
            sess.run((m, b, error), feed_dict={x: _x, y_label: _y}))
