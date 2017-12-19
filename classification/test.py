import tensorflow as tf
import numpy as np
import datetime
import data_helper
from cnn_mnistdd_vgg_half import CNN
import os
from scipy.misc import imsave

# Preprocessing Settings
tf.flags.DEFINE_boolean("zca_whitening", True, "Enable usage of ZCA Whitening (default: False)")
tf.flags.DEFINE_string("checkpoint_path", "checkpoints/", "Path for the dataset to be used.")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Loading database here
print("Loading database...")
x_test = data_helper.load_test_dataset(zca_whitening=FLAGS.zca_whitening)
print("Shape:", x_test.shape)
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

def test():
    # Start Evaluation
    print("Testing...")
    # Initialize parameters
    test_batches = data_helper.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)
    test_all_true = 0
    test_one_true = 0
    current_step = tf.train.global_step(sess, global_step)
    all_preds = []
    # Testing loop
    for test_batch in test_batches:
        preds = test_step(test_batch)
        all_preds.append(preds)
    # Handle predictions here
    all_preds = np.concatenate(all_preds)
    return all_preds

def calculate_acc(y_preds):
    y_test = data_helper.load_test_labels()
    num_all_true = 0
    num_one_true = 0
    for i in range(len(y_test)):
        test_pred = y_preds[i]
        test_true = y_test[i]
        if test_pred[0] == test_true[0]:
            num_one_true+=1
        elif test_pred[0] == test_true[1]:
            num_one_true+=1
        elif test_pred[1] == test_true[0]:
            num_one_true+=1
        elif test_pred[1] == test_true[1]:
            num_one_true+=1

    res = np.count_nonzero((np.array(y_preds)-np.array(y_test))==0, axis=1)
    all_true = np.count_nonzero(res==2)
    #one_true = all_true + np.count_nonzero(res==1)
    all_true_acc = all_true/len(y_test)
    one_true_acc = num_one_true/len(y_test)
    score = all_true*3 + num_one_true - all_true
    time_str = datetime.datetime.now().isoformat()
    print("{}: Test Summary,  All True Acc {:g}, One True Acc {:g}, Score {:g}".format(time_str, all_true_acc, one_true_acc, score))
    return all_true_acc, one_true_acc, score

def calculate_wrong_index(y_preds):
    y_test = data_helper.load_test_labels()
    res = np.count_nonzero((np.array(y_preds)-np.array(y_test))==0, axis=1)
    y_one_wrong_index = []
    y_both_wrong_index = []
    for i in range(len(res)):
        if (res[i] == 1):
            y_one_wrong_index.append(i)
        elif (res[i] == 0):
            y_both_wrong_index.append(i)
        else:
            continue
    return y_one_wrong_index, y_both_wrong_index

def save_wrong_images(y_preds, y_one_wrong_index, y_both_wrong_index):
    y_test = data_helper.load_test_labels()
    for i in y_one_wrong_index:
        img = x_test[i].reshape((64, 64))
        label = y_test[i]
        wrong_pred = y_preds[i]
        imsave("one_wrong/"+str(i)+".png", img)
        text_file = open("one_wrong/"+str(i)+".txt", "w")
        text_file.write("labels: " + str(label) + "\npredictions: " + str(wrong_pred))
        text_file.close()
    for i in y_both_wrong_index:
        img = x_test[i].reshape((64, 64))
        label = y_test[i]
        wrong_pred = y_preds[i]
        imsave("both_wrong/"+str(i)+".png", img)
        text_file = open("both_wrong/"+str(i)+".txt", "w")
        text_file.write("labels: " + str(label) + "\npredictions: " + str(wrong_pred))
        text_file.close()
    
if __name__ == "__main__":
    y_preds = test()
    print("shape:", y_preds.shape)
    all_true_acc, one_true_acc, score = calculate_acc(y_preds)
    y_one_wrong_index, y_both_wrong_index = calculate_wrong_index(y_preds)
    save_wrong_images(y_preds, y_one_wrong_index, y_both_wrong_index)