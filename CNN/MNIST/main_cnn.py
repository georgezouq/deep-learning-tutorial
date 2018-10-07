# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
# Build a Multiplayer Convolutional Network

Getting 92% accuracy on MNIST is bad. It's almost embarrassingly bad. In this section, we'll fix that,
jumping from a very simple model to something moderately sophisticated: a small convolutional neural
network. This will get us to around 99.2% accuracy -- not state of the art, but respectable.

RESHAPE -> CONV1 -> POOL1 -> CONV2 -> POOL2 -> FC1 -> DROUPOUT -> FC2 -> LOSS

'''

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


'''
## Weight Initialization

To create this model, we're going to need to create a lot of weights and biases. One should generally
initialize weights with a small amount of noise for symmetry breaking, and prevent 0 gradients. Since
we're using `ReLU` neurons, it is also good practice to initialize them with a slightly positive
initial bias to avoid `dead neurons`. Instead of doing this repeatedly while we build the model, let's
create two handly function to do it for us.
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''
## Convolution and Pooling

TensorFlow also gives us a lot of flexibility in convolution and pooling operations. How do we handle
the boundaries? What is our stride size?  In this example, we're always going to choose the vanilla
version. our convolutions uses a stride of one and are zero padded so that the output is the same size
as the input. Our pooling is plain old max pooling over 2x2 block. To keep our code cleaner, let's
also abstract those operations into functions.
'''

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

'''
## First Convolutional Layer

We can now implement our first layer. It will consist of convolution, followed by max pooling. The
convolution will compute 32 features for each 5x5 patch.  Its weight tensor will have a shape of
`[5, 5, 1, 32]`. The first two dimensions are the patch size, the next is the number of input
channels, and the last is the number of output channels.

We will also have a bias vector with a component for each output channel.
'''

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

'''
To apply the layer, we first reshape `x` to a 4d tensor, with the second and third dimensions
corresponding to image with and height, and the final dimension corresponding to the number of
color channels.
'''

x_image = tf.reshape(x, [-1, 28, 28, 1])

'''
We then convolve `x_image` with the weight tensor, add the bias, apply the ReLU function, and
finally max pool. The `max_pool_2x2` method will reduce the image size to 14x14.
'''

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
## Second Convolutional Layer

In order to build a deep network, we stack several layers of this type. The second layer will
have 64 features for each 5x5 patch.
'''

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
## Densely Connected Layer

Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons
to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch
of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
'''

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
## Dropout

To reduce overfitting, we will apply dropout before the readout layer. We create a `placeholder`
for the probability that a neuron's output is kept during dropout. This allows us to turn dropout
on during training, and turn it off during testing. TensorFlow's `tf.nn.dropout` op  automatically
handles scaling neural outputs in addition to masking them, so dropout just works without any
additional scaling
'''

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
## Readout Layer

Finally, we add a layer, just like for the one layer softmax regression above.
'''

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''
## Train and Evaluate the Model

How well does this model do? To train and evaluate it we will use code that is nearl identical to
that for the simple one layer SoftMax network above.

The differences are that:

- We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer
- We will include the additional parameter `keep_prob` in `feed_dict` to control the dropout rate.
- We will add logging to every 100th  iteration in the training process.

We will also use `tf.Session` ratheer then `tf.InteractiveSession`. This better separates the process
of creating the graph (model specification) and the process of evaluating the graph (model fitting).
It generally makes for cleaner code. The `tf.Session` is created within a `with block` so that it is
automatically destroyed once the block is exited.

Feel free to run this code. Be aware that it does 20, 000 training iterations and may take a while (
possibly up to half an hour), depending on your processor.

'''

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1.0
            })

            print('step %d training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 0.5
        })
    print('text accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    }))
