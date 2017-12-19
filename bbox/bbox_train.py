import tensorflow as tf
import numpy as np
import datetime
import data_helper
#from cnn_bbox_shallow import CNN
from cnn_bbox_vgg_half import CNN
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
acc_list = []
loss_list = []

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", cnn.loss)
# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, hist_summaries_merged, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

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
def train_step(x_batch, y_batch_bbox, y_batch_labels):
    """
    A single training step
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y_labels: y_batch_labels, 
                 cnn.input_y_bbox: y_batch_bbox, 
                 #cnn.is_training: True,
                 cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
    _, step, loss, summaries, acc = sess.run([train_op, global_step, cnn.loss, train_summary_op, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: Step {}, Epoch {}, Loss {:g}, Classify acc {:g}".format(time_str, step, int(step//num_batches_per_epoch)+1, loss, acc))
    if step%100==0:
        train_summary_writer.add_summary(summaries, global_step=step)

def dev_step(x_batch, y_batch_bbox, y_batch_labels):
    """
    Evaluates model on a dev set
    """
    feed_dict = {cnn.input_x: x_batch, 
                 cnn.input_y_labels: y_batch_labels, 
                 cnn.input_y_bbox: y_batch_bbox, 
                 #cnn.is_training: False,
                 cnn.dropout_keep_prob: 1.0}
    loss, preds_bbox, labels_bbox, preds, labels = sess.run([cnn.loss, cnn.predictions_bbox, cnn.ground_truth_bbox, cnn.predictions_classification, cnn.ground_truth_classification], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return preds_bbox, labels_bbox, preds, labels, loss

def compute_intersection_over_union(predictions, ground_truth, threshold=0.5):
    num_all_pass = 0
    num_one_pass = 0
    for i in range(len(predictions)):
        ious = []
        bbox_pred = predictions[i]
        bbox_true = ground_truth[i]
        for j in range(2):
            xA = max(bbox_pred[j,0], bbox_true[j,0])
            yA = max(bbox_pred[j,1], bbox_true[j,1])
            xB = min(bbox_pred[j,0]+28, bbox_true[j,0]+28)
            yB = min(bbox_pred[j,1]+28, bbox_true[j,1]+28)

            #xB = min(bbox_pred[j,2], bbox_true[j,2])
            #yB = min(bbox_pred[j,3], bbox_true[j,3])
            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)
            #bbox_area_1 = (bbox_pred[j, 2] - bbox_pred[j, 0] + 1)*(bbox_pred[j, 3] - bbox_pred[j, 1] + 1)
            bbox_area_1 = 29*29
            bbox_area_2 = 29*29
            # compute the intersection over union
            iou = interArea / float(bbox_area_1 + bbox_area_2 - interArea)
            ious.append(iou)
        if ious[0] > threshold or ious[1] > threshold:
            num_one_pass += 1
            if ious[0] > threshold and ious[1] > threshold:
                num_all_pass += 1
    return num_all_pass/len(predictions), num_one_pass/len(predictions)

max_all_acc = 0.0
max_one_acc = 0.0
max_all_step = 0
max_one_step = 0
max_score = 0
max_score_step = 0
# Generate batches
train_batches = data_helper.batch_iter(list(zip(x_train, bbox_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for train_batch in train_batches:
    x_batch, y_batch_bbox, y_batch_labels = zip(*train_batch)
    train_step(x_batch, y_batch_bbox, y_batch_labels)
    current_step = tf.train.global_step(sess, global_step)
    # Testing loop
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        test_batches = data_helper.batch_iter(list(zip(x_valid, bbox_valid, y_valid)), FLAGS.batch_size, 1, shuffle=False)
        sum_loss = 0
        i = 0

        valid_bbox_preds = []
        valid_preds = []
        valid_gt = []
        for test_batch in test_batches:
            x_valid_batch, y_valid_batch_bbox, y_valid_batch_labels = zip(*test_batch)
            preds_bbox, labels_bbox, preds, labels, test_loss = dev_step(x_valid_batch, y_valid_batch_bbox, y_valid_batch_labels)
            valid_bbox_preds.append(preds_bbox)
            valid_preds.append(preds)
            valid_gt.append(labels)
            sum_loss += test_loss
            i += 1
        valid_bbox_preds = np.concatenate(valid_bbox_preds)
        valid_labels_preds = np.concatenate(valid_preds)
        valid_gts = np.concatenate(valid_gt)
        acc_all_bbox, acc_one_bbox = compute_intersection_over_union(valid_bbox_preds, bbox_valid)

        res = np.count_nonzero((np.array(valid_labels_preds)-np.array(valid_gts))==0, axis=1)
        all_true = np.count_nonzero(res==2)
        one_true = all_true + np.count_nonzero(res==1)
        test_score = all_true*3 + one_true-all_true
        all_true_acc = all_true/len(valid_gts)
        one_true_acc = one_true/len(valid_gts)

        time_str = datetime.datetime.now().isoformat()
        print("{}: Evaluation Summary, Epoch {}, Loss {:g}, acc bbox all {:g}, acc bbox one {:g}, All True Acc {:g}, One True Acc {:g}, Score {:g}".format(
              time_str, int(current_step//num_batches_per_epoch)+1, sum_loss/i, acc_all_bbox, acc_one_bbox, all_true_acc, one_true_acc, test_score))

        if all_true_acc > max_all_acc:
            max_all_acc = all_true_acc
            max_all_step = current_step
            max_acc_bbox_all = (acc_all_bbox, acc_one_bbox)
            if all_true_acc>0.96:
                path = saver_1.save(sess, checkpoint_prefix_1, global_step=current_step)
                print("Saved current model checkpoint with max all accuracy to {}".format(path))
        print("{}: Current Max All Acc {:g} in Iteration {}".format(time_str, max_all_acc, max_all_step))
        print("{}: Max All Bbox All Acc {:g} Bbox one Acc {:g}".format(time_str, max_acc_bbox_all[0], max_acc_bbox_all[1]))
        if one_true_acc > max_one_acc:
            max_one_acc = one_true_acc
            max_one_step = current_step
            max_acc_bbox_one = (acc_all_bbox, acc_one_bbox)
            if one_true_acc>0.995:
                path = saver_2.save(sess, checkpoint_prefix_2, global_step=current_step)
                print("Saved current model checkpoint with max one accuracy to {}".format(path))
        print("{}: Current Max One Acc {:g} in Iteration {}".format(time_str, max_one_acc, max_one_step))
        print("{}: Max One Bbox All Acc {:g} Bbox one Acc {:g}".format(time_str, max_acc_bbox_one[0], max_acc_bbox_one[1]))

        if test_score > max_score:
            max_score = test_score
            max_score_step = current_step
            max_acc_bbox_score = (acc_all_bbox, acc_one_bbox)
            if max_score>14650:
                path = saver_3.save(sess, checkpoint_prefix_3, global_step=current_step)
                print("Saved current model checkpoint with maxscore to {}".format(path))
        print("{}: Current Max Score {:g} in Iteration {}\n".format(time_str, max_score, max_score_step))
        print("{}: Max Score Bbox All Acc {:g} Bbox one Acc {:g}".format(time_str, max_acc_bbox_score[0], max_acc_bbox_score[1]))