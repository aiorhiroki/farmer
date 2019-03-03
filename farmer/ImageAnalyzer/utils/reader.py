from ncc.dataset import prepare_data
from ncc.preprocessing import list_files


def train_test_files():
    prepare_data(nb_image=10, module='cifar10', annotation_file=False)
    train_files, train_labels = list_files('cifar10/train')
    test_files, test_labels = list_files('cifar10/test')
    train_set = [[image, label] for image, label in zip(train_files, train_labels)]
    test_set = [[image, label] for image, label in zip(test_files, test_labels)]

    return train_set, test_set


"""
def train_test_files():
    from glob import glob
    import re
    step = 2
    train_folder = '/mnt/hdd/data/ColonPathology/train/step5/images/*.jpg'.format(step)
    test_folder = '/mnt/hdd/data/ColonPathology/test/images/*.jpg'
    train_files = glob(train_folder)
    test_files = glob(test_folder)

    train_set = annotate_image(train_files, step)
    test_set = annotate_image(test_files, step, training=False)

    return train_set, test_set


def annotate_image(image_files, step, training=True):
    class_names = {'nor': 0, 'ad2': 1, 'ad3': 1, 'can': 2, 'lym': 2}
    nb_cancers = 0
    nb_normals = 0
    if not training:
        trained_names = ['nor', 'can', 'ad2', 'ad3', 'lym']
    elif step == 1:
        trained_names = ['nor', 'can', 'ad2']
    elif step == 2:
        trained_names = ['nor', 'can', 'ad2', 'ad3']
    else:
        trained_names = ['nor', 'can', 'ad2', 'ad3', 'lym']
    annotation_set = []
    for image_file in image_files:
        class_name = re.match('(.*?)_', image_file.split('/')[-1]).group(1)
        if class_name in trained_names:
            class_idx = class_names[class_name]
            if step == 1 and class_idx == 0:
                nb_normals += 1
                if training and nb_normals > 72:
                    continue
            elif step == 1 and class_idx == 2:
                nb_cancers += 1
                if training and nb_cancers > 72:
                    continue
            annotation_set.append([image_file, class_idx])

    return annotation_set
"""
