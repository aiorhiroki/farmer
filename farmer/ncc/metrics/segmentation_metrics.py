import numpy as np
import itertools
from tqdm import tqdm
from ..utils import ImageUtil
import matplotlib.pyplot as plt


def iou_dice_val(
        nb_classes, height, width, data_set, model, train_colors=None):
    image_util = ImageUtil(nb_classes, (height, width))
    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    print('validation...')
    for image_file, seg_file in tqdm(data_set):
        # Get a training sample and make a prediction using current model
        sample = image_util.read_image(image_file, anti_alias=True)
        target = image_util.read_image(
            seg_file, normalization=False, train_colors=train_colors)
        predicted = np.asarray(
            model.predict_on_batch(np.expand_dims(sample, axis=0)))[0]
        confusion += calc_segmentation_confusion(
            predicted, target, nb_classes)

    iou = calc_iou_from_confusion(confusion)
    dice = calc_dice_from_confusion(confusion)

    return {'iou': iou, 'dice': dice}


def calc_segmentation_confusion(y_pred, y_true, nb_classes):
    # Convert predictions and target from categorical to integer format
    y_pred = np.argmax(y_pred, axis=-1).ravel()
    y_true = np.argmax(y_true, axis=-1).ravel()
    x = y_pred + nb_classes * y_true
    bincount_2d = np.bincount(
        x.astype(np.int32), minlength=nb_classes**2)
    assert bincount_2d.size == nb_classes**2
    confusion = bincount_2d.reshape((nb_classes, nb_classes))

    return confusion


def calc_iou_from_confusion(confusion):
    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, 0) - true_positive
    false_negative = np.sum(confusion, 1) - true_positive
    # Just in case we get a division by 0, set the value to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / \
            (true_positive + false_positive + false_negative)

    iou[np.isnan(iou)] = 0
    return iou


def calc_dice_from_confusion(confusion):
    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, 0) - true_positive
    false_negative = np.sum(confusion, 1) - true_positive
    # Just in case we get a division by 0, set the value to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * true_positive / \
            (2 * true_positive + false_positive + false_negative)

    dice[np.isnan(dice)] = 0
    return dice


def detection_rate_confusions(pred_labels, gt_labels, nb_classes):
    """
    gt_labels: iterable container (Width, Height)
    prediction_labels: iterable container (Width, Height)
    nb_classes: number of classes

    """
    confusion_tabel = np.zeros((nb_classes, 4), dtype=np.uint8)
    for gt_label, pred_label in zip(gt_labels, pred_labels):
        for class_id in range(nb_classes):
            gt_mask = gt_label == class_id
            pred_mask = pred_label == class_id
            if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
                confusion_tabel[class_id, 0] += 1
            elif np.sum(gt_mask) == 0 and np.sum(pred_mask) > 0:
                confusion_tabel[class_id, 1] += 1
            elif np.sum(gt_mask*pred_mask) == 0:
                confusion_tabel[class_id, 2] += 1
            else:
                confusion_tabel[class_id, 3] += 1

    return confusion_tabel


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_file=None):
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
    plt.tight_layout()
    plt.savefig('{}.png'.format(save_file))
