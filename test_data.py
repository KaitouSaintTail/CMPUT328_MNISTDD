import data_helper
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_mnistdd import CNN

dataset_path = "dataset/"

x_train, y_train, x_test, y_test = data_helper.load_dataset(dataset_path, zca_whitening=False)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

plt.title("Sample Images")
plt.subplot(131),plt.imshow(x_train[10].reshape((64, 64)), cmap='gray'),plt.title("y_train[10]")
plt.subplot(132),plt.imshow(x_train[500].reshape((64, 64)), cmap='gray'),plt.title("y_train[500]")
plt.subplot(133),plt.imshow(x_test[40].reshape((64, 64)), cmap='gray'),plt.title("y_test[40]")
print(y_train[10])
print(y_train[500])
print(y_test[40], y_true[40])
plt.show()

cnn = CNN()
sess = tf.Session()
# Initialize Graph
sess.run(tf.global_variables_initializer())

res, ground_truth = sess.run([cnn.ground_truth, cnn.predictions], feed_dict={cnn.input_x: [x_train[10]], cnn.input_y: [y_train[10]], cnn.dropout_keep_prob: 1.0})
print(res)
print(ground_truth)