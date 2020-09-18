from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def show_matrix(
        y_test, y_prediction, class_names, save_dir=None, show_plot=False):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        title='Confusion matrix, without normalization',
        save_dir=save_dir
    )

    # Plot normalized confusion matrix
    plot_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=True,
        title='Normalized confusion matrix',
        save_dir=save_dir
    )

    if show_plot:
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_dir=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_dir:
        if normalize:
            plt.savefig(f'{save_dir}/normalized_confusion.png')
        else:
            plt.savefig(f'{save_dir}/count_confusion.png')
    # plt.tight_layout()  # this may cause error
