## PERSON OR NOT IMAGE IDENTIFIER IN TENSORFLOW

**DESCRIPTION** 

This algorithm is able to recognize if an image contains a person/people in it or not. It learns from a set of 1,526 images divided into training, validation, and test sets. The images are stored in an HD5 file, with corresponding labels as to whether they contain a person/people or not.

**INSTALLATION**

This program runs on PYTHON 3 so all dependencies will need to be versions that are compatible with that version.

*Dependencies you will need:*

- TensorFlow
- Numpy
- Scipy
- Matplotlib
- Python Imaging Library (PIL)

*Image files*

The images used for this project come from the [INRIA Dataset](http://pascal.inrialpes.fr/data/human/). Unfortunately, the collection of images is too big to be stored in GitHub, so they'll need to be downloaded from [here](https://www.dropbox.com/s/ebnuk1nmibvcs7s/Dataset.zip?dl=0). The folder contains the HD5 file already created to map the images. If this doesn't work for you somehow, you can just download the images and then run the array+and+label+data.py file to create your own HD5 data file (which will work with the models so long as they are stored in the same directory).

*Models*

To "install" the models, simply download the Person_or_not_DL_TF.py and pnp_tf_utils.py files, and place them in the same directory as the Dataset folder containing the HD5 file and image files.

**USAGE**

This 5-layer ReLu/Sigmoid model uses Xavier initialization, L2 regularization, gradient descent, and Adam optimization. The five layers here are 230400, 30, 6, 4, 1, with a learning rate of 0.01 and a lambda of 0.8. There is also a model that includes minibatch gradient descent included in here, but it takes FAR TOO LONG to run and is ultimately very inefficient so the default setting uses batch gradient descent. You can edit line 234 in Person_or_not_DL_TF.py to use the minibatches if you wish, though. Run Person_or_not_DL.py in the command line to train the model and predict on the validation and test sets. For 1500 epochs - as it is set to by default - it took almost three hours on my little Macbook Pro to run, but if you have access to more computing power, it will take less time!

**ANALYSIS**

Based on my testing, using the exact same settings as my [previous version](https://github.com/priyankaincode/Person-Image-Classifier-NN) built from scratch but with the addition of Adam optimization and Xavier initialization results in significantly lower accuracy - **60 percent on training and 58/62 percent on val/test!!** I gave up on trying to fix it further simply because it took a VERY long time to run, but if I was to move this to Google Cloud or have access to greater computing power, I would try to further tune the learning rate (increase) and lambda (increase) to see if I could raise the accuracy, in addition to increasing the number of hidden layers. I would also just do regular optimization instead of Adam, to see if I can replicate my earlier non-TensorFlow results. Truthfully, though, for this size data set, not using TensorFlow is significantly faster on my computer and also shows better results.

**CREDIT**

I wrote pretty much all of this! But most pieces of code were written originally for various assignments in Dr. Andrew Ng's Deep Learning Specialization courses on Coursera (specifically the Hyperparameter Tuning, Regularization, and Optimization course) so thanks to him and his team for helping me learn to do it all in Python! And for creating the HD5 file, I followed [this tutorial](http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html) that was immensely helpful as well. And, of course, I would be nowhere without the Stackoverflow community. None of us would be!
