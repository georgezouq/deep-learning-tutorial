# coding=utf-8

'''
Task:

    - Create a softmax regression function that is a model for recognizing MNIST digits
    based on looking at every pixel in the image

    - Use TensorFlow to train the model to recognize digits by having it `look` at
    thousands of examples (and run our first TensorFlow session to do so)

    - Check the model's accuracy with our test data

    - Build, train, and test a multilayer convolutional neural network to improve the result

'''

# Setup

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
## Start TensorFlow InteractiveSession

TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this
backend is called a session. The common usage for TensorFlow programs is to first create a graph
and then launch it in a session.

Here we instead use the convenient `InteractiveSession` class, which makes TensorFlow more flexible
about how you structure your code. It allows you to interleave operations which build a `computation
graph` with ones that run the graph. This is particularly convenient when working in interactive
contexts like IPython. If you are not using an `InteractiveSession`, then you should build the entire
computation graph before starting a session and launching the graph.

'''

import tensorflow as tf
sess = tf.InteractiveSession()

'''
## Computation Graph

To do efficient numerical computing in python, we typically use libraries like `NumPy` that do
expensive operations such as matrix multiplication outside Python, using highly efficient code
implemented in another language. Unfortunately, there can still be a lot of overhead from switching
back to Python every operation. This overhead is especially had if you want to run computations on
GPUs or in a distributed manner, where there can be a high cost to transferring data.

TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid
this overhead. Instead of running a single expensive operation independently from Python, TensorFlow
lets us describe a graph of interacting operations that run entirely outside Python. This approach is
similar to that used in Teano or Torch.

The role of the Python code is therefore to build this external computation graph, and to dictate
which parts of the computation graph should be run.
'''

'''
## Build a Softmax Regression Model

### Placeholders
'''

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

'''
### Variables

We now define the weights `W` and biases `b` for our model. We could imagine treating these like
additional inputs, but TensorFlow has an even better way to handle them: `Variable`. `Variable`
is a value that lives in TensorFlow's computation graph. it can be used and even modified by the
computation. In machine learning applications, one generally has the model parameters by `Variable`s.
'''

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

'''
We pass the initial value for each parameter in the call to `tf.Variable`. In this case, we
initialize both `W` and `b` as tensors full of zeros. `W` is a 784 x 10 matrix and `b` is
10-dimensional vector (because we have 10 classes).

Before `Variable`s can be used within a session, they must be initialized using that session.
This step takes the initial values (in this case tensors full of zeros) that have already been
specified, and assigns then to each `Varibale`. This can be done of all `Varibales` at once:
'''

sess.run(tf.global_variables_initializer())

'''
### Predicted Class and Loss Function

We can now implement our regression model. It only takes one line! We muliply the vectorized input
images `x` by the weight matrix `W`, add the bias `b`.

'''

y = tf.matmul(x, W) + b

'''
We can specify a loss function just as easily. Loss indicates how bad model's prediction was on a
example; we try to minimize that while training across all the examples. Here, our loss function
is the cross-entropy between the target and the softmax activation function applied to the model's
prediction. As in the beginners tutorial, we use the stable formulation.
'''

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

'''
Note that `tf.nn.softmax_cross_entropy_with_logits` internally applies the softmax on the model's
unnormalized model prediction and sums across all classses, and `tf.reduce_mean` takes the average
over these sums.
'''

'''
## Train the Model

Now that we have defined our model and training loss function, it is straightforward to train using
TensorFlow. Because TensorFlow knows the entire computation graph, it can use automatic differentiation
to find the gradients of the loss with respect to each of the variables. TensorFlow has a variety of
`built-in optimization algorithms`. For this example, we will use  steepest gradient descent, with
step length of 0.5, to descend the cross entropy.
'''

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
What TensorFlow actually did in that single line was to add new operations to the computation graph
These operations included ones to compute gradients, compute parameter update steps, and apply update
steps to the parameters.

The returned operation `traom_step`, when run, will apply the gradient descent updates to the parameters
Training the model can therefore be accomplished by repeatedly running `train_step`.
'''

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

'''
We load 100 training examples in each training interation. We then run the `train_step` operation,
using `feed_dict` to replace the `placeholder` tensors `x` and `y_` with the training examples.
Note that you can replace any tensor in your computation graph using `feed_dict` -- it's not
restricted to just `placehoolder`s.
'''

'''
## Evaluate the Model

How well did our model do?

First we'll figure out where we predicted the correct label. `tf.argmax` is an extremely useful which
gives you the index of the highest entry in a tensor along some axis. For example, `tf.argmax(y, 1)`
is the label our model thinks is most likely for each input, while `tf.argmax(y_, 1)` is the true
label. We can use `tf.equal` to check if our prediction matches the truth.
'''

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

'''
That gives us a list of booleans. To determine what fraction are correct, we cast to floating point
number and then take the mean. For example, `[True, False, True, True]` would become `[1, 0, 1, 1]`
which would become `0.75`
'''

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Finally, we can evaluate our accuracy on the test data. This should be about 92% correct
'''

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


