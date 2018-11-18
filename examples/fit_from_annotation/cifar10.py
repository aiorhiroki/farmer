from farmer.Classifier import Classifier
from ncc.dataset import prepare_data

# load data
prepare_data('cifar10')

# fit farmer classification
Classifier().fit_from_annotation('cifar10/train_annotation.csv')
