import pandas as pd
import tensorflow as tf
import numpy as np


class GenericDataset(object):
    """Wrapper class for datasets"""

    def __init__(self):
        """Initialize instance"""


class MNISTDataset(object):
    """Container class for MNIST datasets imported from CSV"""

    def __init__(
            self,
            data_points=None,
            images=None,
            labels=None,
            classes=None):
        """Construct dataset object"""
        if images is not None:
            assert len(images) == len(labels)

        # Variables for mini-batch parsing
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):
        """Get next mini-batch"""
        start = self.index_in_epoch

        # Shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            permutation = np.arange(self.data_points)
            np.random.shuffle(permutation)
            self.images = self.images[permutation]
            self.labels = self.labels[permutation]
            self.classes = self.classes[permutation]

        # Go to the next epoch
        if start + batch_size > self.data_points:

            # Get the remainer of the examples in this epoch
            remaining_data_points = self.data_points - start
            images_remaining = self.images[start:self.data_points]
            labels_remaining = self.labels[start:self.data_points]

            # Shuffle the data
            if shuffle:
                permutation = np.arange(self.data_points)
                np.random.shuffle(permutation)
                self.images = self.images[permutation]
                self.labels = self.labels[permutation]
                self.classes = self.classes[permutation]

            # Start next epoch, use remaining images in the new epoch
            start = 0
            self.index_in_epoch = batch_size - remaining_data_points
            end = self.index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]

            inputs = np.concatenate((images_remaining, images_new_part), axis=0)
            outputs = np.concatenate((labels_remaining, labels_new_part), axis=0)

            # Finish epoch
            self.epochs_completed += 1

            return inputs, outputs

        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch

            inputs = self.images[start:end]
            outputs = self.labels[start:end]

            return inputs, outputs

    def test_data(self, dataset, name, one_hot=False):
        """Optional function

        Make sure the created data can be accepted into TF.
        """
        session = tf.Session()
        print("---")
        print("{0} dataset:".format(name))

        if one_hot:
            print("Labels: \t\t{0}".format(dataset.classes[0:5]))
        else:
            print("Labels: \t\t{0}".format(dataset.labels[0:5]))

        inputs = tf.placeholder(dtype='float32', shape=[5])
        ones = tf.ones(shape=inputs.shape, dtype='float32')
        plus_one = tf.add(inputs, ones)

        session = tf.Session()

        if one_hot:
            plus_one = session.run(plus_one, feed_dict={inputs: dataset.classes})
        else:
            plus_one = session.run(plus_one, feed_dict={inputs: dataset.labels})

        print("Labels plus one: \t{0}".format(plus_one))

        session.close()


def get_labeled_images(raw_csv_data, one_hot=False):
    """
    Transform CSV data to required input format.

    Separates the raw input data (array of flattned picture bytes)
    from the labels
    """
    # Use a wrapper class to strore all the datasets
    dataframe = MNISTDataset()

    # Get total number of columns from the CSV
    columns = raw_csv_data.shape[1]
    data_points = raw_csv_data.shape[0]
    dataframe.data_points = data_points

    # Images are stored in the CSV as one-dimentional vectors (784 numbers from 28x28 pixels)
    # We get 2 numpy.ndarrays of images(size 784) and labels(size 1)
    images = raw_csv_data[list(range(1, columns))].values.tolist()
    labels = raw_csv_data[0].values.tolist()

    images = np.array(images)
    labels = np.array(labels)
    # print("nparray:\n", images[0:2])
    # print("nparray shape:\n", images[0:2].shape)

    dataframe.images = images

    # Transform the digit numbers into 'one-hot' vectors
    # Note: names for the datasets are consistant with the
    # TensorFlow imported data from the official tutorials
    if one_hot:
        dataframe.classes = labels

        i = 1  # Counter for images
        hot_labels = np.empty([data_points, 10])
        for lbl in labels:
            hot_label = np.zeros(10, dtype='float32')
            hot_label[int(lbl)] = np.float32(1)

            # Display how many images were pre-processed
            hot_labels[i - 1] = hot_label
            print("Processed images: {0} / {1}".format(i, data_points))
            i += 1

        dataframe.labels = hot_labels

    else:
        dataframe.labels = labels

    return dataframe


def read_data_sets(path_to_folder, one_hot=False):
    """Get a TF.Dataset from CSV file path"""
    # Load the data using pandas
    test_data_raw = pd.read_csv(
        filepath_or_buffer="{0}mnist_test.csv".format(path_to_folder),
        sep=",",
        dtype='float32',
        header=None,
        index_col=False)

    train_data_raw = pd.read_csv(
        filepath_or_buffer="{0}mnist_train.csv".format(path_to_folder),
        sep=",",
        dtype='float32',
        header=None,
        index_col=False)

    # print(test_data_raw.head(), "\n", train_data_raw.describe(include='all'))

    # Get a processed dataset object
    test_dataset = get_labeled_images(test_data_raw, one_hot)
    train_dataset = get_labeled_images(train_data_raw, one_hot)

    # Create a new (empty) dataset object to wrap train/test datasets
    whole_dataset = GenericDataset()

    # Bundle all the datasets to a single dataset
    whole_dataset.test = test_dataset
    whole_dataset.train = train_dataset

    return whole_dataset
