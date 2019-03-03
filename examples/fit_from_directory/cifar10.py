from farmer.Classifier import Classifier
from ncc.dataset import prepare_data

num_images_per_class = 100
module = 'cifar10'

prepare_data(nb_image=num_images_per_class, module=module, annotation_file=False)

Classifier(epochs=10).fit_from_directory(module)
