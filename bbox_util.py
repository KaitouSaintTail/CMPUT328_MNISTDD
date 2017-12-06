import numpy as np
import os
import matplotlib.pyplot as plt
import data_helper

num_classes = 10
num_digits = 2

def load_bbox(dataset_path):
    bbox_train = np.load(os.path.join(dataset_path, "train_bboxes.npy"))
    bbox_valid = np.load(os.path.join(dataset_path, "valid_bboxes.npy"))
    #bbox_train = bbox_train[:,:,:-2]
    #bbox_valid = bbox_valid[:,:,:-2]
    return bbox_train, bbox_valid

def crop_digits(data, labels, bboxes):
    digit_1_data = []
    digit_2_data = []
    digit_1_labels = []
    digit_2_labels = []
    for i in range(len(bboxes)):
        digits = data[i].reshape((64, 64))
        digit_1 = digits[bboxes[i,0,0]:bboxes[i,0,2], bboxes[i,0,1]:bboxes[i,0,3]]
        digit_2 = digits[bboxes[i,1,0]:bboxes[i,1,2], bboxes[i,1,1]:bboxes[i,1,3]]
        digit_1_data.append(digit_1)
        digit_2_data.append(digit_2)
        digit_1_labels.append(labels[i,:10])
        digit_2_labels.append(labels[i,10:])

    return np.array(digit_1_data), np.array(digit_1_labels), np.array(digit_2_data), np.array(digit_2_labels)


# Loading database here
print("Loading database...")
x_train, y_train, x_valid, y_valid = data_helper.load_dataset("dataset/", zca_whitening=False)
bbox_train, bbox_valid = load_bbox("dataset/")
x_train_1, y_train_1, x_train_2, y_train_2 = crop_digits(x_train, y_train, bbox_train)
print(x_train_1.shape)
plt.title("Sample Images")
plt.subplot(131),plt.imshow(x_train[10].reshape((64, 64)), cmap='gray'),plt.title("digits")
plt.subplot(132),plt.imshow(x_train_1[10], cmap='gray'),plt.title("digit_1")
plt.subplot(133),plt.imshow(x_train_2[10], cmap='gray'),plt.title("digit_2")
plt.show()