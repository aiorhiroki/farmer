from keras.datasets import cifar10
from farmer.Classifier import Classifier

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# fit farmer classification
Classifier(epochs=5, optimizer='adm').fit_from_array(x_train, y_train, x_test, y_test)
