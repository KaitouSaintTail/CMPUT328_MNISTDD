import tensorflow as tf
import numpy as np
import datetime
import data_helper
from cnn_mnistdd_lenet import CNN
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
x_train, y_train, x_test, y_test = data_helper.load_dataset(FLAGS.database_path, zca_whitening=FLAGS.zca_whitening)
num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1
print("Shape:",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("Success!")

sess = tf.Session()
cnn = CNN()

# Optimizer and LR Decay
#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
#lr_decay_fn = lambda lr, global_step : tf.train.exponential_decay(lr, global_step, FLAGS.num_epochs*num_batches_per_epoch, 0.95, staircase=True)
#train_op = tf.contrib.layers.optimize_loss(loss=cnn.loss, global_step=global_step, clip_gradients=4.0,
#    learning_rate=FLAGS.learning_rate, optimizer=lambda lr: optimizer, learning_rate_decay_fn=lr_decay_fn)
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
acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary, hist_summaries_merged, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

acc_list = []
loss_list = []

# Saver
# Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.join(out_dir, "checkpoints")
checkpoint_prefix_1 = os.path.join(checkpoint_dir, "model_all")
checkpoint_prefix_2 = os.path.join(checkpoint_dir, "model_one")
checkpoint_prefix_3 = os.path.join(checkpoint_dir, "model_score")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver_1 = tf.train.Saver(tf.global_variables(), max_to_keep=1)
saver_2 = tf.train.Saver(tf.global_variables(), max_to_keep=1)
saver_3 = tf.train.Saver(tf.global_variables(), max_to_keep=1)

# Train Step and Dev Step
def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y: y_batch, 
                 #cnn.is_training: True,
                 cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
    _, step, loss, accuracy, summaries = sess.run([train_op, global_step, cnn.loss, cnn.accuracy, train_summary_op], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Step {}, Epoch {}, Loss {:g}, Acc {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, accuracy))
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

max_all_acc = 0.0
max_one_acc = 0.0
max_all_step = 0
max_one_step = 0
max_score = 0
max_score_step = 0
# Generate batches
train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for train_batch in train_batches:
    x_batch, y_batch = zip(*train_batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    # Testing loop
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)
        test_all_true = 0
        test_one_true = 0
        test_score = 0
        sum_loss = 0
        i = 0

        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            preds, labels, test_loss = dev_step(x_test_batch, y_test_batch)
            sum_loss += test_loss
            i += 1
            # Handle predictions here
            preds = np.array(preds)
            labels = np.array(labels)
            res = np.count_nonzero((np.array(preds)-np.array(labels))==0, axis=1)
            all_true_per_batch = np.count_nonzero(res==2)
            one_true_per_batch = all_true_per_batch + np.count_nonzero(res==1)
            test_all_true += all_true_per_batch
            test_one_true += one_true_per_batch
        test_score += test_all_true*3 + test_one_true-test_all_true

        time_str = datetime.datetime.now().isoformat()
        all_true_acc = test_all_true/len(y_test)
        one_true_acc = test_one_true/len(y_test)
        acc_list.append(all_true_acc)
        loss_list.append(sum_loss/i)
        print("{}: Evaluation Summary, Epoch {}, Loss {:g}, All True Acc {:g}, One True Acc {:g}, Score {:g}".format(
              time_str, int(current_step//num_batches_per_epoch)+1, sum_loss/i, all_true_acc, one_true_acc, test_score))
        if all_true_acc > max_all_acc:
            max_all_acc = all_true_acc
            max_all_step = current_step
            if all_true_acc>0.96:
                path = saver_1.save(sess, checkpoint_prefix_1, global_step=current_step)
                print("Saved current model checkpoint with max all accuracy to {}".format(path))
        print("{}: Current Max All Acc {:g} in Iteration {}".format(time_str, max_all_acc, max_all_step))
        if one_true_acc > max_one_acc:
            max_one_acc = one_true_acc
            max_one_step = current_step
            if one_true_acc>0.995:
                path = saver_2.save(sess, checkpoint_prefix_2, global_step=current_step)
                print("Saved current model checkpoint with max one accuracy to {}".format(path))
        print("{}: Current Max One Acc {:g} in Iteration {}".format(time_str, max_one_acc, max_one_step))

        if test_score > max_score:
            max_score = test_score
            max_score_step = current_step
            if max_score>14650:
                path = saver_3.save(sess, checkpoint_prefix_3, global_step=current_step)
                print("Saved current model checkpoint with maxscore to {}".format(path))
        print("{}: Current Max Score {:g} in Iteration {}\n".format(time_str, max_score, max_score_step))

np.save("acc_list.npy", np.array(acc_list))
np.save("loss_list.npy", np.array(loss_list))
