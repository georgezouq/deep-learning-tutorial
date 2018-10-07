# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

'''
Download
    The `input_data.read_data_sets()` function will ensure that the correct data has been downloaded
    to your local training folder and then unpack that data to return a dictionary of `DataSet`
    instances
'''

data_sets = input_data.read_data_set()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
Inputs and Placeholders

    The `placeholder_inputs()` function creates two `tf.placeholder` ops that define the shape of the
    inputs, including the `batch_size`, to the rest of the graph and into which the actual training
    examples will be fed.
'''

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

'''
    Further down, in the training loop, the full image and label datasets are sliced fit the
    `batch_size` for each step, matched with these placeholder ops, and then passed into the
    `sess.run()` function using the `feed_dict` parameter.
'''


'''
## Build the Graph

    After placeholders for the data, the graph is built from the `mnist.py` file according to a
    3-stage pattern:

    1. `inference()` - Builds the graph as far as required for running the network forward to make
    predictions.

    2. `loss()` - Adds to the inference graph the ops required to generate loss.
        `training()`

    3. `training()` - Adds to the loss graph the ops required to compute and apply gradients

### Inference

    The `inference()` function builds the graph as far as needed to return the tensor that would
    contain the output predictions.

    It takes the images placeholder as input and builds on top of it a pair of fully connected
    layers with `ReLU` activation followed by a ten node linear layer specifying the output logits.

    Each layer is created beneath a unique `tf.name_scope` that acts as a prefix to the items
    created within that scope.

'''

with tf.name_scope('hidden 1'):

    '''
        Within the defined scope, the weights and biases to be used by each of these layers are
        generated into `tf.Variable` instances, with their desired shapes:
    '''

    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)))
    )

    biases = tf.Variable(
        tf.zeros([hidden1_units]),
        name='biases'
    )

    '''
        When these are created under the `hidden1` scope, the unique name given to the weights
        variable would be `hidden1/weights`.

        Each variable is given initializer ops as part of their construction.

        In this most common case, the weights are initialized with the `tf.truncated_normal` and
        given their shape of a 2-D tensor

        - `IMAGE_PIXELS` representing the number of units in the layer from which the weights
        connect
        - `hidden1_units` representing the number of units in the layer to which the weights connect.

        For the first layer, named `hidden1`, the dimensions are `[IMAGE_PIXELS, hidden1_units]`
        because the weights are connecting the image inputs to the hidden1 layer. The
        `tf.truncated_normal` initializer generates a random distribution with a given mean and
        standard deviation.

        Then teh biases are initialized with `tf.zeros` to ensure they start with all zero values,
        and their shape is simple the number units in the layer to which they connect.

        The graph's three primary ops - two `tf.nn.relu` ops wrapping `tf.matmul` for the hidden
        layers
    '''


