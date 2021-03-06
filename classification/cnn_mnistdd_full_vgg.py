import tensorflow as tf
import numpy as np

he_normal = tf.contrib.layers.variance_scaling_initializer()
#he_normal = tf.truncated_normal_initializer(stddev=0.1)

def Conv(inputs, kernel_size, strides, num_filters, weight_decay, name):
    '''
    Helper function to create a Conv2D layer
    '''
    with tf.variable_scope("conv2D_%s" % name):
        filter_shape = [kernel_size, kernel_size, inputs.get_shape()[3], num_filters]
        w = tf.get_variable(name='W', shape=filter_shape, 
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            initializer=he_normal)
        b = tf.get_variable(name='b', shape=[num_filters],
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        out = tf.nn.relu(conv)
    return out

def fc(inputs, dropout_keep_prob, num_outputs, name):
    with tf.variable_scope(name):
        w = tf.get_variable('W', [inputs.get_shape()[1], num_outputs], initializer=he_normal)
        b = tf.get_variable('b', [num_outputs], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
        out = tf.nn.relu(out)
        out = tf.nn.dropout(out, dropout_keep_prob, name='drop')
    return out

class CNN():
    def __init__(self, num_classes=10, num_digits=2, weight_decay=5e-4):
        # input tensors
        self.input_x = tf.placeholder(tf.float32, [None, 64*64], name="input_x")
        self.input_x_reshaped = tf.reshape(self.input_x, [-1, 64, 64, 1])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes*num_digits], name="input_y")
        self.input_y_reshaped = tf.reshape(self.input_y, [-1, num_digits, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # CNN Architecture
        # Block 1
        self.conv1_1 = Conv(inputs=self.input_x_reshaped, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv1_1")
        self.conv1_2 = Conv(inputs=self.conv1_1, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv1_2")
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

        # Block 2
        self.conv2_1 = Conv(inputs=self.pool1, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv2_1")
        self.conv2_2 = Conv(inputs=self.conv2_1, kernel_size=3, strides=1, num_filters=128, weight_decay=weight_decay, name="conv2_2")
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
       
        # Block 3
        self.conv3_1 = Conv(inputs=self.pool2, kernel_size=3, strides=1, num_filters=256, weight_decay=weight_decay, name="conv3_1")
        self.conv3_2 = Conv(inputs=self.conv3_1, kernel_size=3, strides=1, num_filters=256, weight_decay=weight_decay, name="conv3_2")
        self.conv3_3 = Conv(inputs=self.conv3_2, kernel_size=3, strides=1, num_filters=256, weight_decay=weight_decay, name="conv3_3")
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

        # Block 4
        self.conv4_1 = Conv(inputs=self.pool3, kernel_size=3, strides=1, num_filters=512, weight_decay=weight_decay, name="conv4_1")
        self.conv4_2 = Conv(inputs=self.conv4_1, kernel_size=3, strides=1, num_filters=512, weight_decay=weight_decay, name="conv4_2")
        self.conv4_3 = Conv(inputs=self.conv4_2, kernel_size=3, strides=1, num_filters=512, weight_decay=weight_decay, name="conv4_3")
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

        shape = int(np.prod(self.pool4.get_shape()[1:]))
        self.flatten = tf.reshape(self.pool4, (-1, shape))

        self.fc1 = fc(inputs=self.flatten, dropout_keep_prob=self.dropout_keep_prob, num_outputs=4096, name='fc1')
        self.fc2 = fc(inputs=self.fc1, dropout_keep_prob=self.dropout_keep_prob, num_outputs=4096, name='fc2')


        with tf.variable_scope('fc'):
            w = tf.get_variable('w', [self.fc2.get_shape()[1], num_classes*num_digits], initializer=he_normal)
            b = tf.get_variable('b', [num_classes*num_digits], initializer=tf.constant_initializer(0.0))
            out = tf.matmul(self.fc2, w) + b
            self.fc = tf.reshape(out, [-1, num_digits, num_classes])

        # Cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc, 2, name="Predictions")
            self.ground_truth = tf.argmax(self.input_y_reshaped, 2, name="Ground_Truth")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc, labels=self.input_y_reshaped)
            regularization_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.reduce_mean(losses) + regularization_losses

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.ground_truth)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")