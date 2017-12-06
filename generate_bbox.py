import tensorflow as tf
import numpy as np
import datetime
import data_helper
from cnn_bbox_shallow import CNN
import os

# Preprocessing Settings
tf.flags.DEFINE_string("database_path", "dataset/", "Path for the dataset to be used.")
tf.flags.DEFINE_boolean("zca_whitening", True, "Enable usage of ZCA Whitening (default: False)")
tf.flags.DEFINE_string("checkpoint_path", "checkpoints/", "Path for the dataset to be used.")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Loading database here
print("Loading database...")
x_train, y_train, x_test, y_test = data_helper.load_dataset(FLAGS.database_path, zca_whitening=FLAGS.zca_whitening)
num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1
print("Shape:",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("Success!")

sess = tf.Session()
cnn = CNN()

# Initialize Graph
global_step = tf.Variable(0, name="global_step", trainable=False)
sess.run(tf.global_variables_initializer())

# Restore model
checkpoint_dir = os.path.join(os.path.curdir, FLAGS.checkpoint_path)
saver = tf.train.Saver(max_to_keep=1)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
print("Model restored!")

def test_step(x_batch):
    """
    Evaluates model on a test set
    """
    feed_dict = {cnn.input_x: x_batch, 
                 #cnn.input_y: y_batch, 
                 cnn.dropout_keep_prob: 1.0}
    preds = sess.run([cnn.predictions], feed_dict)
    return preds[0]

def generate_bbox():
    # Start Evaluation
    print("Testing...")
    # Initialize parameters
    test_batches = data_helper.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)
    current_step = tf.train.global_step(sess, global_step)
    all_preds = []
    # Testing loop
    for test_batch in test_batches:
        preds = test_step(test_batch)
        all_preds.append(preds)
    # Handle predictions here
    all_preds = np.concatenate(all_preds)
    return all_preds

if __name__ == "__main__":
    y_preds = generate_bbox()
    print("Success!")
    print("Shape of generated bbox:", y_preds.shape)
    np.save("bbox_valid_generated.npy", y_preds.astype(int))