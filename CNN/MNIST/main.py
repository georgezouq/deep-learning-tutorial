# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data

'''
- The MNIST data split into three parts:
    55,000 Training data   (mnist.train)
    10,000 Test data       (mnist.test)
    5,000  Validation data (mnist.validation)

  For every data point is an image of handwritten digit and a corresponding label.
  Each image is 28 pixels by 28 pixels
  We can flatten this array into a vector of 28 x 28 = 784 number.

  The result iss that mnist.train.images is a tensor(an n-dimensional array) with a
  shape of [55000, 784]. The first dimension is an index into the list of images and
  the second dimension is the index for each pixel in each image. Each entry in the
  tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular
  image

  Each image in MNIST has a corresponding label, a number between 0 and 9 representing
  the digit drawn in the image.

  For the purposes of this tutorial,  we're going to want our label a "one-hot vectors"
  Consequently, `mnist.train.labels` is a [55000, 10] vector
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


'''
- Softmax Regressions

    We want look at an image and give the probabilities for it being each digit.

    This is a classic case where a softmax regression is a natural, simple model. If you
    want to an object being one of several different things, softmax is the thing to do.

    Softmax gives us a list of values between 0 and 1 that add up to 1.

    Softmax has two steps:
        First add up the evidence of our input being in certain classes, and then we
        convert the evidence into probabilities.

        To tally up the evidence that a given image is in a particular class,

    We often think of softmax: exponentiating its inputs and then normalizing them.

    Exponentiating means that one more unit of evidence increases the given to any
    hypothesis multiplicatively. And conversely, having one less unit of evidence
    means that a hypothesis. No hypothesis ever has zero or negative weight. Softmax
    then normalizes these weights, so that they add up to one, forming a valid probability
    distribution.
'''

'''
- Implementing the Regression


'''

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

'''
    X isn't a specific value, It's a placeholder, a value that we'll input then we ask TensorFlow
    to run computation. We want to be able to input any number of MNIST images, each flattened into
    a 784-dimensional vector. We represent this as a 2-D tensor of floating-point numbers, with a
    shape [None, 784] (Here None means that a dimension can be of any length)

    Blew we will define weights and biases for out model. A Variable a modifiable tensor that lives
    in tensor that lives in TensorFlow's graph of interacting operations. It can be used and even
    modified by the computation.
'''

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''
    Implement out model:
'''

y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
- Training

    In order to train out model, we need to define what it means for the model to be good. Well,
    actually, in machine learning we typically define what it means for a model to be bad. We
    call this the cost, or the loss, and it represents how far off our model is from our desired
    outcome. We try to minimize the error, and the smaller the error margin. the better our model
    is.

    One very common, very nice function to etermine the loss of a model is called "cross-entropy".
    Cross-entropy arises fropm thinking about information compressing code in information theory
    but it winds up being an important idea in lots of areas, from gambling to machine learning.

                            Hy'(y) = −∑y′log⁡(y)

    Where y is our predicted probability distribution, and y' is the true distribution
'''

y_ = tf.placeholder(tf.float32, [None, 10])

'''
 Implement the cross-entropy function, −∑y′log⁡(y):
'''

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

'''
    `tf.log` compute loarithm of each element of `y`. Next we multiply each element of `y_`
    with the corresponding element of `tf.log(y)`. Then `tf.reduce_sum` adds elements in the
    second dimension of y, due to the `reduction_indices = [1]` parameter. Finally, `tf/reduce_mean`
    computes the mean over all the examples in the batch

   On the source we can simple use `tf.nn.softmax_cross_entropy_with_logits`
'''

'''
    Automatically use the back propagation algorithm to efficiently determine how your variables
    affect the loss you ask it to minimize.
'''

train_step = tf. train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
    We minimie cross_entropy using the gradient descent algorithm with a learning rate of 0.5. Gradient
    descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the
    direction that reduces the cost.

    What TensorFlow actually does here, behind the scenes, is to add new operations to your graph
    which implement back propagation and gradient descent. Then it gives you back a sigle operation
    which when run does a step of gradient descent training, slightly tweaking your variables to
    reduce the loss.
'''

'''
    Launch the model in an InteractiveSession:
'''
sess = tf.InteractiveSession()

'''
    Create an operation to initialize the variables we created:
'''

tf.global_variables_initializer().run()

# Train step 1000 times

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''
    On each step, we get a "batch" of one hundred random data points from our training set. We run
    `train_step` feeding in the batches data to replace the `placeholder`s.

    Using small  batches of random data is called stochastic training -- in this case , stochastic
    gradient descent.Ideally, we'd like to use all our data for every step of training because that
    would give us a better sense of what we should be doing, but that's expensive. So, instead, we
    use a different subset every time. Doing this  is cheap and has much of the same benefit.
'''

'''
- Evaluating Our Model

    First, Figure out where we predicted the correct label. `tf.argmax` is an extremely useful
    function which gives you the index of the highest entry in a tensor along some axis. For
    example, `tf,argmax(y, 1)` is the label our model think s is most likely for each input,
    while `tf.argmax(y_, 1)` is the correct label. We can use `tf.equal` to check if our
    prediction matches the truth.
'''

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

'''
    That gives us a list of booleans. To determine what fraction are correct, we cast to floating
    point numbers and then take the mean. For example [True, False, True, True] would becode [1,
    0, 1, 1] which would become 0.75
'''

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Finally, we ask for our accuracy on our test data:

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
