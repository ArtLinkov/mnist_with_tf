# mnist_with_tf
A simple example of how to use TensorFlow to solve MNIST using a Convnet

 This is an improved tutorial building upon the basic documentation from the TF docs, and includes loading of data from CSV.
 
 As I now write this library, most of the basic tutorials for TF start from a pre-processed dataset for MNIST.
 I believe that most data scientists will agree that half of the work, if not more, is the preparation of the data before it even reaches our fancy models.
 Therefore, the motivation for this library was to demonstrate a simple implementation start-to-finish using MNIST as an example.
 
 Notes:
 -------
 - The raw MNIST data (in CSV format) was downloaded from [here](https://pjreddie.com/projects/mnist-in-csv/)
 - Some of the code was inspired by the excellent tutorials from [here](https://github.com/Hvass-Labs/TensorFlow-Tutorials), which I highly recommend.
 - The parsing and extracting of data in this example was done using pandas and numpy. All the datasets are processed and kept in-memory during the run of the model. This method, while very convenient to use, is not the most efficient one and is suited for small to medium datasets. For datsests larger than what can be held in memory, an iterator is required (more info on TF iterators can be found [here](https://www.tensorflow.org/versions/master/programmers_guide/datasets#creating_an_iterator))
 
 Usage:
 -------
 1) unizp "mnist_test.zip" & "mnist_train.zip" (same folder)
 
 2) run "run_model.py"
