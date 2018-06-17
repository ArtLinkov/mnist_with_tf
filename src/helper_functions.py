import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import math


# Import data from the local tf tutorials
# ########################################
def import_tutorial_data(show_stats=False):
	from tensorflow.examples.tutorials.mnist import input_data
	data = input_data.read_data_sets("./tf_tutorial_data/", one_hot=True)

	# add a new feature to the dataset called classes
	data.test.classes = np.array([label.argmax() for label in data.test.labels])

	# Display data stats
	if show_stats is True:
		print("----------------------------------")
		print("Size of:")
		print("- Training-set:\t\t{}".format(len(data.train.labels)))
		print("- Test-set:\t\t{}".format(len(data.test.labels)))
		print("- Validation-set:\t{}".format(len(data.validation.labels)))
		print("  ---------")

		# First 5 expected outputs and labels
		print("First 5 true outputs: \n{}".format(data.test.labels[0:5]))
		print("First 5 true labels: {}".format(data.test.classes[0:5]))
		print("----------------------------------")

	return data


# Import data from the local CSV files
# ########################################
def import_from_csv(show_stats=False):
	import csv_data
	data = csv_data.read_data_sets("./csv_data/", one_hot=True)

	# Display data stats
	if show_stats is True:
		print("----------------------------------")
		print("Size of:")
		print("- Training-set:\t\t{}".format(len(data.train.labels)))
		print("- Test-set:\t\t{}".format(len(data.test.labels)))
		print("  ---------")

		# First 5 expected outputs and labels
		print("First 5 true outputs: \n{}".format(data.test.labels[0:5]))
		print("First 5 true labels: \n{}".format(data.test.classes[0:5]))
		print("----------------------------------")

	return data


# Helper functions to create layers
# #######################################
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05), name="weights")


def new_baises(layer_size):
	return tf.Variable(tf.constant(0.05, shape=[layer_size]), name="biases")


def new_conv_layer(
	input, 				# Previous layer
	input_channels, 	# Number of channels/filters in previous layer
	filter_size,	 	# Window size of the filters
	filter_number, 		# Number of filters in the layer
	use_pooling=True): 	# Use 2x2 max-pooling

	# Shape of the filter-weights for the convolution.
	# Note: This format is determined by the TF API:
	# [width, height, input-dimension, output-dimension]
	shape = [filter_size, filter_size, input_channels, filter_number]

	weights = new_weights(shape=shape)
	biases = new_baises(layer_size=filter_number)

	# Create the TensorFlow operation for convolution.
	# Note1: strides are set to 1 in all dimentions.
	# However, the first and last must always be 1, becasue
	# the first is for image-number and last is for input-channels.
	# E.g. strides=[1,2,2,1] would mean that the filter is moved 2px
	# across the x and y axises of the image.
	# Note2: padding is set to 'SAME' which means the input image is padded
	# with zeros so that the size of the output feature map equals the input
	layer = tf.nn.conv2d(
		input=input,
		filter=weights,
		strides=[1, 1, 1, 1],
		padding="SAME")

	# Add the biases
	layer += biases

	# Use max-pooling if use_pulling=True
	if use_pooling:
		layer = tf.nn.max_pool(
			value=layer,
			ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1],
			padding="SAME")

	# Add ReLU activation
	layer = tf.nn.relu(layer, name="conv_layer")

	return layer, weights


# Helper function to create fully-connected layers
def flatten_layer(layer):
	# The shape of the layer is assumed to be:
	# layer_shape == [num_images, img_heigt, img_width, num_channels]
	layer_shape = layer.get_shape()

	# The number of featues is: img_heigt x img_width x num_channels
	# TF has a function to calculate this
	num_features = layer_shape[1:4].num_elements()

	# Reshape the layer to [num_images, num_features].
	# Note: We just set the size of the second dimension to num_features
	# and the size of the first dimension to -1, which means
	# the size in that dimension is calculated so that the total size of the tensor
	# will be unchanged from the reshaping.
	layer_flat = tf.reshape(layer, [-1, num_features], name="layer_flat")

	return layer_flat, num_features


def new_fc_layer(
	input,	 			# Previous layer
	num_inputs,			# Number of inputs from previous layer
	num_outputs,	 	# Number of outputs
	use_relu=True): 	# Use relu?

	# Create weights & biases
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_baises(layer_size=num_outputs)

	# Calculate the feed forward
	layer = tf.add(tf.matmul(input, weights), biases, name="fc_layer")

	# Add activation
	if use_relu:
		layer = tf.nn.relu(layer, name="fc_layer")

	return layer


# Helper functions to show performance
# #####################################
# Helper function to plot images
def plot_9_images(image_shape, images, classes_true, classes_pred=None, name="Image examples"):
	# print(len(images))
	# print(len(classes_true))
	assert len(images) == len(classes_true)

	# Create figure with 3x3 sub-plots
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image
		ax.imshow(images[i].reshape(image_shape), cmap='binary')

		# Show true and predicted classes
		if classes_pred is None:
			xlabel = "True: {0}".format(classes_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(classes_true[i], classes_pred[i])

		ax.set_xlabel(xlabel)

		# Remove tick from plot
		ax.set_xticks([])
		ax.set_yticks([])

	# Show plot
	plt.gcf().canvas.set_window_title(name)
	plt.show()


# Plot 9 images of mis-classified examples
def plot_error_examples(image_shape, data, correct_predictions, predicted_classes):
	# Get a list of Bools for each image correct/incorrect prediction, as well as
	# a list of a list of the predicted class for each image and negate the boolean array.
	incorrect = (correct_predictions == False)

	# Get the images from the test-set that have been incorrectly classified.
	images = data.test.images[incorrect]

	# Get the predicted classes for those images.
	predicted_classes = predicted_classes[incorrect]

	# Get the true classes for those images.
	true_classes = data.test.classes[incorrect]

	# Plot the first 9 images.
	plot_9_images(
		image_shape=image_shape,
		images=images[0:9],
		classes_true=true_classes[0:9],
		classes_pred=predicted_classes[0:9],
		name="Misclassified image examples")


# Use scikit-learn to plot a confusion matrix
def print_confusion_matrix(classes_true, classes_pred, num_classes, plot_image=False):
	# From scikit-learn
	cm = confusion_matrix(y_true=classes_true, y_pred=classes_pred)
	# print("cm min: {0}, cm max: {1}".format(cm.min(), cm.max()))

	# Print the matrix as text
	print(cm)

	# Plot the matrix as image, using matplotlib
	if plot_image is True:
		# Load the matrix into an image
		plt.imshow(cm, interpolation='nearest', cmap=plt.cm.tab20)
		min_value = int(cm.min())
		max_value = int(cm.max())
		# Make some visual adjustments
		# plt.tight_layout()
		plt.colorbar(ticks=range(min_value, max_value, int(max_value / 10)))
		tick_marks = np.arange(num_classes)
		plt.xticks(tick_marks, range(num_classes))
		plt.yticks(tick_marks, range(num_classes))
		plt.xlabel('Predicted')
		plt.ylabel('True')

		# Show plot
		plt.gcf().canvas.set_window_title("confusion matrix")
		plt.show()


def plot_conv_weights(weights, input_channel=0, name="Figure 1"):
	# Normalize min/max so that color intesities can be compared between images
	w_min = np.min(weights)
	w_max = np.max(weights)

	# Number of filters in the conv-layer
	num_filters = weights.shape[3]

	# Number of grids to plot
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots
	fig, axes = plt.subplots(num_grids, num_grids)
	fig.subplots_adjust(hspace=0.5, wspace=0.5)

	# Create sub-plots
	for i, ax in enumerate(axes.flat):
		# Since num_grids can be more than num_filters
		if i < num_filters:

			# Get weights for the i'th filters' weights
			image = weights[:, :, input_channel, i]

			# Set label
			ax.set_xlabel("Filter: {0}".format(i))

			# plot the sub-image
			ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic', interpolation='nearest')

		# Remove ticks
		ax.set_xticks([])
		ax.set_yticks([])

	# Set title and show plot
	plt.gcf().canvas.set_window_title("{0}, filter weights".format(name))
	plt.show()


def plot_conv_layer_output(input_image, conv_output, name="Figure 1"):
	# Number of filters in the conv-layer
	num_filters = conv_output.shape[3]

	# Number of grids to plot
	num_grids = math.ceil(math.sqrt(num_filters + 1))

	# Create figure with a grid of sub-plots
	fig, axes = plt.subplots(num_grids, num_grids)
	fig.subplots_adjust(hspace=0.5, wspace=0.5)

	# Create sub-plots
	for i, ax in enumerate(axes.flat):
		# Since num_grids can be more than num_filters
		if i < num_filters:
			if i == 0:
				ax.set_xlabel("Input image", weight="bold")
				ax.imshow(input_image, cmap='binary', interpolation='nearest')
			else:
				# Get weights for the i'th filters' weights
				image = conv_output[0, :, :, i]

				# Set label
				ax.set_xlabel("Filter: {0}".format(i - 1))

				# plot the sub-image
				ax.imshow(image, cmap='binary', interpolation='nearest')

		# Remove ticks
		ax.set_xticks([])
		ax.set_yticks([])

	# Set title and show plot
	plt.gcf().canvas.set_window_title("{0}, output".format(name))
	plt.show()
