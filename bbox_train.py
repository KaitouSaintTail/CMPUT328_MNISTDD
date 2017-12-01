import tensorflow as tf
import numpy as np
import datetime
import data_helper
from cnn_mnistdd_shallow import CNN
import os

# Parameters settings
# Data loading params
tf.flags.DEFINE_string("database_path", "dataset/", "Path for the dataset to be used.")

# Preprocessing Settings
tf.flags.DEFINE_boolean("zca_whitening", True, "Enable usage of ZCA Whitening (default: False)")

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("weight_decay", 5e-4, "Weight decay rate for L2 regularization (default: 5e-4)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-3)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 120, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_boolean("enable_moving_average", False, "Enable usage of Exponential Moving Average (default: False)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value))
print("")

# Loading database here
print("Loading database...")
x_train, y_train, x_valid, y_valid = data_helper.load_dataset(FLAGS.database_path, zca_whitening=FLAGS.zca_whitening)
bbox_train, bbox_valid = data_helper.load_bbox(FLAGS.database_path)
num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1
print("Shape:",x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, bbox_train.shape, bbox_valid.shape)
print("Success!")

sess = tf.Session()
cnn = CNN()

# Optimizer and LR Decay
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
grads_and_vars = optimizer.compute_gradients(cnn.loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
# Initialize Graph
sess.run(tf.global_variables_initializer())

# Output directory for Tensorflow models and summaries
out_dir = os.path.curdir
print("Writing to {}\n".format(out_dir))

# Tensorboard
def add_gradient_summaries(grads_and_vars):
    grad_summaries = []
    for grad, var in grads_and_vars:
        if grad is not None:
            grad_hist_summary = tf.summary.histogram(var.op.name + "/gradient", grad)
            grad_summaries.append(grad_hist_summary)
    return grad_summaries
hist_summaries = []
for var in tf.trainable_variables():
    hist_hist_summary = tf.summary.histogram(var.op.name + "/histogram", var)
    hist_summaries.append(hist_hist_summary)
hist_summaries_merged = tf.summary.merge(hist_summaries)
grad_summaries = add_gradient_summaries(grads_and_vars)
grad_summaries_merged = tf.summary.merge(grad_summaries)

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", cnn.loss)
# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, hist_summaries_merged, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Saver
# Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model_bbox_pred")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# Train Step and Dev Step
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y: y_batch, 
                 #cnn.is_training: True,
                 cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
    _, step, loss, summaries = sess.run([train_op, global_step, cnn.loss, train_summary_op], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Step {}, Epoch {}, Loss {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss))
    if step%100==0:
        train_summary_writer.add_summary(summaries, global_step=step)

def dev_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y: y_batch, 
                 #cnn.is_training: False,
                 cnn.dropout_keep_prob: 1.0}
    loss, preds, labels = sess.run([cnn.loss, cnn.predictions, cnn.ground_truth], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return preds, labels, loss

# Generate batches
train_batches = data_helper.batch_iter(list(zip(x_train, bbox_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for train_batch in train_batches:
    x_batch, y_batch = zip(*train_batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    # Testing loop
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        test_batches = data_helper.batch_iter(list(zip(x_valid, bbox_valid)), FLAGS.batch_size, 1, shuffle=False)
        sum_loss = 0
        i = 0

        for test_batch in test_batches:
            x_valid_batch, y_valid_batch = zip(*test_batch)
            preds, labels, test_loss = dev_step(x_valid_batch, y_valid_batch)
            sum_loss += test_loss
            i += 1

        time_str = datetime.datetime.now().isoformat()
        print("{}: Evaluation Summary, Epoch {}, Loss {:g}".format(
              time_str, int(current_step//num_batches_per_epoch)+1, sum_loss/i))
        #if test_score > max_score:
        #    max_score = test_score
        #    max_score_step = current_step
        #    if max_score>14650:
        #        path = saver_3.save(sess, checkpoint_prefix_3, global_step=current_step)
        #        print("Saved current model checkpoint with maxscore to {}".format(path))
