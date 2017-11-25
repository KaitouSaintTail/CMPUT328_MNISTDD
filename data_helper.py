import numpy as np
import os

num_classes = 10
num_digits = 2

def load_dataset(dataset_path, zca_whitening=True):
    """Load numpy format dataset stored in a specific path."""
    x_train = np.load(os.path.join(dataset_path, "train_X.npy")).astype('float32')/255.
    y_train = np.load(os.path.join(dataset_path, "train_Y.npy"))
    x_valid = np.load(os.path.join(dataset_path, "valid_X.npy")).astype('float32')/255.
    y_valid = np.array(np.load(os.path.join(dataset_path, "valid_Y.npy")))
# ADDED
    bbox_train = np.load(os.path.join(dataset_path, "train_bboxes.npy"))
    bbox_valid = np.load(os.path.join(dataset_path, "valid_bboxes.npy"))

    y_train = np.array([to_one_hot_encodings(n) for n in y_train])
    y_valid = np.array([to_one_hot_encodings(n) for n in y_valid])
    #zca_principal_components(x_train)

# ADDED
    # resize bounding boxes into n x num_digits x 2 since boxes are fixed size
    bbox_train = bbox_train[:,:,:-2]
    bbox_valid = bbox_valid[:,:,:-2]

    # resize bounding boxes into n x 4
    bbox_train = np.array([n.reshape(4, 1, order='F') for n in bbox_train])
    bbox_train = np.reshape(bbox_train, (-1, 4))
    bbox_valid = np.array([n.reshape(4, 1, order='F') for n in bbox_valid])
    bbox_valid = np.reshape(bbox_train, (-1, 4))

    if zca_whitening:
        # data whitening
        principal_components = np.load("x_train_zca.npy")
        print("Shape of ZCA Matrix:", principal_components.shape)
        white_x_train = np.dot(x_train, principal_components)
        x_train = np.reshape(white_x_train, x_train.shape)
        white_x_valid = np.dot(x_valid, principal_components)
        x_valid = np.reshape(white_x_valid, x_valid.shape)

# ADDED
    return x_train, y_train, x_valid, y_valid, bbox_train, bbox_valid

def zca_principal_components(x_train, zca_epsilon=0.1):
    """Calculate ZCA Matrix over training set and save 
    ZCA matrix as a numpy format file."""
    # ZCA Matrix should only be calculated over Training Set
    # and it needs to applied to all Train, Dev and test Sets.
    print(x_train.shape)
    sigma = np.dot(x_train.T, x_train) / x_train.shape[0] 
    u, s, _ = np.linalg.svd(sigma) 
    principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + zca_epsilon))), u.T)
    np.save("x_train_zca.npy", principal_components)

def to_one_hot_encodings(label):
    """Convert a label to one-hot encoding."""
    vector = np.zeros(num_classes*num_digits)
    for i in range(len(label)):
        idx = i * num_classes + label[i]
        vector[idx] = 1
    return vector

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
