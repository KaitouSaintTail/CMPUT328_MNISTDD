import tensorflow as tf
import numpy as np

bias_initializer = tf.constant_initializer(0.1)
kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)

# ADDED
# Bounding box intersection over union calculation
def intersection_over_union(predictions, ground_truth):
    iou_counter = 0
    for pred, gt in zip(predictions, ground_truth):
        # box 1
        x1a = max(gt[1], pred[1])
        y1a = max(gt[0], pred[0])
        x1b = min(gt[1], pred[1]) + 28
        y1b = min(gt[0], pred[0]) + 28

        # box 2
        x2a = max(gt[3], pred[3])
        y2a = max(gt[2], pred[2])
        x2b = min(gt[3], pred[3]) + 28
        y2b = min(gt[2], pred[2]) + 28

        # compute area of intersection
        intersection1 = (x1b - x1a + 1) * (y1b - y1a + 1)
        intersection2 = (x2b - x2a + 1) * (y2b - y2a + 1)

        # compute the area of both the prediction and ground truth rectangles
        # always the same: 29 * 29
        # gtArea1 = ((gt[0] + 28) - gt[0] + 1) * ((gt[1] + 28) - gt[1] + 1)
        # predArea1 = ((pred[0] + 28) - gt[0] + 1) * ((pred[1] + 28) - pred[1] + 1)
        #
        # gtArea2 = ((gt[2] + 28) - gt[2] + 1) * ((gt[3] + 28) - gt[3] + 1)
        # predArea2 = ((pred[2] + 28) - gt[2] + 1) * ((pred[3] + 28) - pred[3] + 1)
        gtArea1 = predArea1 = gtArea2 = predArea2 = 29 * 29

        # compute the union
        union1 = gtArea1 + predArea1 - intersection1
        union2 = gtArea2 + predArea2 - intersection2

        # compute the intersection over union
        iou1 = intersection1 / float(union1)
        iou2 = intersection2 / float(union2)

        iou = np.mean([iou1, iou2])
        if iou >= 0.5:
            iou_counter += 1

    return iou_counter / float(len(ground_truth))

def Conv(inputs, kernel_size, strides, num_filters, weight_decay, name):
    '''
    Helper function to create a Conv2D layer
    '''
    with tf.variable_scope("conv2D_%s" % name):
        filter_shape = [kernel_size, kernel_size, inputs.get_shape()[3], num_filters]
        w = tf.get_variable(name='W', shape=filter_shape, 
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            initializer=kernel_initializer)
        b = tf.get_variable(name='b', shape=[num_filters],
            initializer=bias_initializer)
        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, b)
        out = tf.nn.relu(conv)
    return out

def fc(inputs, dropout_keep_prob, num_outputs, name):
    with tf.variable_scope(name):
        w = tf.get_variable('W', [inputs.get_shape()[1], num_outputs], initializer=kernel_initializer)
        b = tf.get_variable('b', [num_outputs], initializer=bias_initializer)
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
        self.conv1 = Conv(inputs=self.input_x_reshaped, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv1")
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

        self.conv2 = Conv(inputs=self.pool1, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv2")
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

        self.conv3 = Conv(inputs=self.pool2, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv3")
        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool3")

        self.conv4 = Conv(inputs=self.pool3, kernel_size=3, strides=1, num_filters=64, weight_decay=weight_decay, name="conv4")
        self.pool4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool4")

        shape = int(np.prod(self.pool4.get_shape()[1:]))
        self.flatten = tf.reshape(self.pool4, (-1, shape))

        self.fc1 = fc(inputs=self.flatten, dropout_keep_prob=self.dropout_keep_prob, num_outputs=1024, name='fc1')

        with tf.variable_scope('fc'):
            w = tf.get_variable('w', [self.fc1.get_shape()[1], num_classes*num_digits], initializer=kernel_initializer)
            b = tf.get_variable('b', [num_classes*num_digits], initializer=bias_initializer)
            out = tf.matmul(self.fc1, w) + b
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