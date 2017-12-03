import tensorflow as tf
import numpy as np
import datetime
import data_helper
from cnn_mnistdd_vgg_half import CNN
import os

# Preprocessing Settings
tf.flags.DEFINE_boolean("zca_whitening", False, "Enable usage of ZCA Whitening (default: False)")
tf.flags.DEFINE_string("database_path", "dataset/", "Path for the dataset to be used.")
tf.flags.DEFINE_string("checkpoint_path", "checkpoints/", "Path for the dataset to be used.")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Loading database here
print("Loading database...")
x_train, y_train, x_test, y_test = data_helper.load_dataset(FLAGS.database_path, zca_whitening=FLAGS.zca_whitening)
print("Shape:",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("Success!")

sess = tf.Session()
cnn = CNN()

# Initialize Graph
global_step = tf.Variable(0, name="global_step", trainable=False)
sess.run(tf.global_variables_initializer())

# Restore model
checkpoint_dir = os.path.join(os.path.curdir, FLAGS.checkpoint_path)
saver = tf.train.Saver(max_to_keep=5)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
print("Model restored!")

def dev_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y: y_batch, 
                 cnn.dropout_keep_prob: 1.0}
    loss, preds, labels = sess.run([cnn.loss, cnn.predictions, cnn.ground_truth], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return preds, labels, loss

# Start Evaluation
print("\nEvaluation:")
# Initialize parameters
test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)
test_all_true = 0
test_one_true = 0
sum_loss = 0
i = 0
current_step = tf.train.global_step(sess, global_step)
num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1

# Testing loop
for test_batch in test_batches:
    x_test_batch, y_test_batch = zip(*test_batch)
    preds, labels, test_loss = dev_step(x_test_batch, y_test_batch)
    sum_loss += test_loss
    i += 1
    # Handle predictions here
    res = np.count_nonzero((np.array(preds)-np.array(labels))==0, axis=1)
    all_true_per_batch = np.count_nonzero(res==2)
    one_true_per_batch = all_true_per_batch + np.count_nonzero(res==1)
    test_all_true += all_true_per_batch
    test_one_true += one_true_per_batch

time_str = datetime.datetime.now().isoformat()
all_true_acc = test_all_true/len(y_test)
one_true_acc = test_one_true/len(y_test)
print("{}: Evaluation Summary, Epoch {}, Loss {:g}, All True Acc {:g}, One True Acc {:g}".format(time_str, int(current_step//num_batches_per_epoch)+1, sum_loss/i, all_true_acc, one_true_acc))
