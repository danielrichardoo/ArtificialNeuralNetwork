import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 2 * Fully connection 

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape))

def init_bias(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(x, w):
    # x --> [batch, image height, image width, color channel]
    # y --> [filter_height, filter_width, in_channel, out]
    # Padding same : hitung sisa di pinggir
    # Padding valid : sisa di pinggir di buang
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')

def max_pooling2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_layer(x, shape):
    weight = init_weight(shape)
    bias = init_bias([shape[3]])
    y = conv2d(x,weight) + bias # sama kayak wx + bias
    return tf.nn.relu(y)

def fully_connected(x, output_size):
    # 28 x 28 -> 1 x 784
    # WEIGHT = jumlah baris(jumlah fitur / input, jumlah hidden)
    input_size = int(x.get_shape()[1])
    print(input_size)
    weight = init_weight([input_size, output_size])
    bias = init_bias([output_size])
    return tf.matmul(x,weight) + bias

# Placeholder
image_width = 28
image_height = 28
x = tf.placeholder(tf.float32, [None, image_width * image_height])
y_true = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x,[-1,28,28,1])
convo1_layer = convolutional_layer(x_image, [5,5,1,8])
convo1_pooling = max_pooling2by2(convo1_layer)

convo2_layer = convolutional_layer(convo1_pooling, [10,10,8,16])
convo2_pooling = max_pooling2by2(convo2_layer)

x_flat = tf.reshape(convo2_pooling, [-1, 7 * 7 * 16])
f_connection_1 = fully_connected(x_flat, 10)
f_connection_2 = fully_connected(f_connection_1, 10)

# loss function & optimizer
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=f_connection_2))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

num_of_epoch = 1000

# Run the training
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_of_epoch):
        x_batch, y_batch = mnist_data.train.next_batch(50)
        sess.run(train,feed_dict={x:x_batch , y_true:y_batch })

    if i % 200 == 0:
        matches = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_prediction, 1))
        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        print('ITERATTION: ', i , " Accuracy: ",end="")
        print(sess.run(acc,feed_dict={x: mnist_data.test.images, y_true: mnist_data.test.labels}))


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 1 * Fully connection 

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape))

def init_bias(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2d(x, w):
    # x --> [batch, image height, image width, color channel]
    # y --> [filter_height, filter_width, in_channel, out]
    # Padding same : hitung sisa di pinggir
    # Padding valid : sisa di pinggir di buang
    return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')

def max_pooling2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_layer(x, shape):
    weight = init_weight(shape)
    bias = init_bias([shape[3]])
    y = conv2d(x,weight) + bias # sama kayak wx + bias
    return tf.nn.relu(y)

def fully_connected(x, output_size):
    # 28 x 28 -> 1 x 784
    # WEIGHT = jumlah baris(jumlah fitur / input, jumlah hidden)
    input_size = int(x.get_shape()[1])
    print(input_size)
    weight = init_weight([input_size, output_size])
    bias = init_bias([output_size])
    return tf.matmul(x,weight) + bias

# Placeholder
image_width = 28
image_height = 28
x = tf.placeholder(tf.float32, [None, image_width * image_height])
y_true = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x,[-1,28,28,1])
convo1_layer = convolutional_layer(x_image, [5,5,1,8])
convo1_pooling = max_pooling2by2(convo1_layer)

convo2_layer = convolutional_layer(convo1_pooling, [10,10,8,16])
convo2_pooling = max_pooling2by2(convo2_layer)

x_flat = tf.reshape(convo2_pooling, [-1, 7 * 7 * 16])
f_connection_1 = fully_connected(x_flat, 10)

# loss function & optimizer
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=f_connection_1))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

num_of_epoch = 1000

# Run the training
with tf.Session() as sess:
    sess.run(init)

    for i in range(num_of_epoch):
        x_batch, y_batch = mnist_data.train.next_batch(50)
        sess.run(train,feed_dict={x:x_batch , y_true:y_batch })

    if i % 200 == 0:
        matches = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_prediction, 1))
        acc = tf.reduce_mean(tf.cast(matches,tf.float32))
        print('ITERATTION: ', i , " Accuracy: ",end="")
        print(sess.run(acc,feed_dict={x: mnist_data.test.images, y_true: mnist_data.test.labels}))
