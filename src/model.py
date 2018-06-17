import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import helper_functions as hf  # Helper functions are in a separate file

# Import data from a local CSV file
# #########################
data = hf.import_from_csv(show_stats=True)

# Or, import data from the local tf tutorials (As seen in the official tutorial)
# #########################
# data = hf.import_tutorial_data(show_stats=True)


# Data dimentions
# #########################
# Input
num_channels = 1 						# Number of color channels, in this case, 1 grayscale channel
img_size = 28							# MNIST images are 28px in each direction
img_size_flat = img_size * img_size 	# Our image data is stored in a single vector of this length
img_shape = (img_size, img_size)		# Tuple of image shape (h,w), will be used to reshape arrays
# Output
num_classes = 10						# Output size, per each digit (0-9)

# hf.plot_9_images(image_shape=img_shape, images=data.test.images[0:9],classes_true=data.test.classes[0:9])

# Neural network layers:
# #########################
# Conv layar 1
filter1_size = 5  		# Each filter will have a 5x5 window
filter1_number = 16  	# A total of 16 filters in this layer

# Conv layar 2
filter2_size = 5  		# Each filter will have a 5x5 window
filter2_number = 36  	# A total of 36 filters in this layer

# Fully-connected later
fc1_size = 128 			# Number of neurons in the fc layer

# Palceholders
# #############

# The image data is given as a flat vector
inputs_flat = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat], name="inputs_flat")

# Since the conv-layers expect a 2D input, we need to reshape the inputs
# Note: number of images is inferred automatically by using -1
inputs_2d = tf.reshape(inputs_flat, [-1, img_size, img_size, num_channels])

true_outputs = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="true_outputs")

true_classes = tf.argmax(true_outputs, axis=1, name="true_classes")

# Computational model
# ####################
layer_conv1, weights_conv1 = hf.new_conv_layer(
	input=inputs_2d,
	input_channels=num_channels,
	filter_size=filter1_size,
	filter_number=filter1_number,
	use_pooling=True)

layer_conv2, weights_conv2 = hf.new_conv_layer(
	input=layer_conv1,
	input_channels=filter1_number,
	filter_size=filter2_size,
	filter_number=filter2_number,
	use_pooling=True)

layer_flat, num_features = hf.flatten_layer(layer_conv2)

layer_fc1 = hf.new_fc_layer(
	input=layer_flat,
	num_inputs=num_features,
	num_outputs=fc1_size,
	use_relu=True)

layer_fc2 = hf.new_fc_layer(
	input=layer_fc1,
	num_inputs=fc1_size,
	num_outputs=num_classes,
	use_relu=False)

normalized_outputs = tf.nn.softmax(layer_fc2)
predicted_classes = tf.argmax(normalized_outputs, axis=1)

# Model optimization
# ###################

# TensorFlow has a built-in function for calculating the cross-entropy.
# Note that it uses the values of the logits because it also calculates the softmax internally.
cost_values = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=true_outputs)

# All the cost values need to be combined to a single scalar
# We must also specify how we combine the information and what we do with it
# reduce refers to reducing an array
global_cost = tf.reduce_mean(cost_values)

# Update parameters using preset optimizers
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(global_cost)

# Initialize the session
session = tf.Session()

# Variables can only be accessed after initialization within a session
session.run(tf.global_variables_initializer())

# Track total epochs trained
global_epochs = 0


# Perfom the optimization using mini-batch SGD
def train(epochs, batch_size, print_freq):
	# Ensure we update the global variable rather than a local copy
	global global_epochs

	# Start time count
	start_time = time.time()

	for e in range(global_epochs, (global_epochs + epochs)):
		# Get a batch of training examples
		# inputs_batch holds a batch of images, and true_outputs_batch holds the true labels for the images
		# next_batch is a helper function of the dataset imported from the tutorial lib
		inputs_batch, true_outputs_batch = data.train.next_batch(batch_size)

		# Place the batches into a dict with the proper names to feed it into the placeholder
		# The placeholder for true_classes is not set because it is unused during training
		feed_dict_batch = {
			inputs_flat: inputs_batch,
			true_outputs: true_outputs_batch}

		session.run(optimizer, feed_dict=feed_dict_batch)

		# Print status every "print_freq" epochs
		if e % print_freq == 0:
			classes_pred, classes_true = session.run([predicted_classes, true_classes], feed_dict=feed_dict_batch)
			correct_predictions = tf.equal(classes_true, classes_pred)  # Returns an array of Bool
			correct_predictions = tf.count_nonzero(tf.cast(correct_predictions, tf.int32))

			total = len(inputs_batch)
			correct = session.run(correct_predictions, feed_dict=feed_dict_batch)
			accuracy = (correct / total)

			msg = "Epoch:{0:}\nTraining accuracy:{1:.2%} ({2} / {3})"

			print(msg.format(e + 1, accuracy, correct, total))

	global_epochs += epochs
	end_time = time.time()
	time_dif = end_time - start_time

	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Show performance using helper functions
# ########################################

def show_stats(
	confusion_matrix=False,
	error_examples=False,
	conv_outputs=False,
	conv_weights=False,
	all=False):

	# Evaluate model accuracy on the test data
	# Get the true classifications for the test-set
	classes_true = data.test.classes

	# This dict will be used in the evaluation of the model
	feed_dict_test = {
		inputs_flat: data.test.images,
		true_outputs: data.test.labels}

	# Get the predicted classifications for the test-set
	classes_pred = session.run(predicted_classes, feed_dict=feed_dict_test)
	correct_predictions = tf.equal(classes_true, classes_pred)  # Returns an array of Bool

	# Make sure to cast from Bool to Float before counting correct answers
	correct_predictions_num = tf.count_nonzero(tf.cast(correct_predictions, tf.int32))

	total = len(data.test.images)
	correct_pred, correct_num = session.run([correct_predictions, correct_predictions_num], feed_dict=feed_dict_test)
	accuracy = (correct_num / total)

	print("Accuracy on test-set: {0:.2%} ({1} / {2})".format(accuracy, correct_num, total))

	if all:
		confusion_matrix = True
		error_examples = True
		conv_outputs = True
		conv_weights = True

	if confusion_matrix:
		hf.print_confusion_matrix(
			classes_true=classes_true,
			classes_pred=classes_pred,
			num_classes=num_classes,
			plot_image=True)

	if error_examples:
		hf.plot_error_examples(
			image_shape=img_shape,
			data=data,
			correct_predictions=correct_pred,
			predicted_classes=classes_pred)

	if conv_outputs:
		random_image_index = np.random.randint(0, len(data.test.images))
		image_flat = data.test.images[random_image_index]
		image = image_flat.reshape(img_shape)

		conv1, conv2 = session.run([layer_conv1, layer_conv2], feed_dict={inputs_flat: [image_flat]})
		hf.plot_conv_layer_output(input_image=image, conv_output=conv1, name="Conv1")
		hf.plot_conv_layer_output(input_image=image, conv_output=conv2, name="Conv2")

	if conv_weights:
		w_conv1, w_conv2 = session.run([weights_conv1, weights_conv2])
		hf.plot_conv_weights(w_conv1, input_channel=0, name="Conv1")
		hf.plot_conv_weights(w_conv2, input_channel=0, name="Conv2")
