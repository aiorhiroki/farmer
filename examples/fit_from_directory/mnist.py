from farmer.Classifier import Classifier
from ncc.dataset import prepare_data

num_images_per_class = 100
module = 'mnist'

prepare_data(nb_image=num_images_per_class, module=module, annotation_file=False)

Classifier().fit_from_directory(module)
