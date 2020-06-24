import numpy as np
import os
import itertools
from tqdm import tqdm
from ..utils import ImageUtil, get_imageset
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
            model.predict(np.expand_dims(sample, axis=0)))[0]
        confusion += calc_segmentation_confusion(
            predicted, target, nb_classes)

    tp = np.diag(confusion)
    fp = np.sum(confusion, 0) - tp
    fn = np.sum(confusion, 1) - tp

    iou = calc_iou_from_confusion(tp, fp, fn)
    dice = calc_dice_from_confusion(tp, fp, fn)
    precision = calc_precision_from_confusion(tp, fp)
    recall = calc_recall_from_confusion(tp, fn)

    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}


def calc_segmentation_confusion(y_pred, y_true, nb_classes):
    # Convert predictions and target from categorical to integer format
    y_pred = np.argmax(y_pred, axis=-1).ravel()
    y_true = y_true.ravel()
    x = y_pred + nb_classes * y_true
    bincount_2d = np.bincount(
        x.astype(np.int32), minlength=nb_classes**2)
    assert bincount_2d.size == nb_classes**2
    confusion = bincount_2d.reshape((nb_classes, nb_classes))

    return confusion


def calc_iou_from_confusion(tp, fp, fn):
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / (tp + fp + fn)

    iou[np.isnan(iou)] = 0
    return [float(i) for i in iou]


def calc_dice_from_confusion(tp, fp, fn):
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * tp / (2 * tp + fp + fn)

    dice[np.isnan(dice)] = 0
    return [float(d) for d in dice]


def calc_precision_from_confusion(tp, fp):
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp)

    precision[np.isnan(precision)] = 0
    return [float(p) for p in precision]


def calc_recall_from_confusion(tp, fn):
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = tp / (tp + fn)

    recall[np.isnan(recall)] = 0
    return [float(r) for r in recall]


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
            elif np.sum(gt_mask * pred_mask) == 0:
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


def generate_segmentation_result(
    nb_classes,
    height,
    width,
    annotations,
    model,
    save_dir,
    train_colors=None,
):
    image_util = ImageUtil(nb_classes, (height, width))
    for sample_image_path in annotations:
        input_image_path, mask_image_path = sample_image_path
        sample_image = image_util.read_image(
            input_image_path, anti_alias=True)
        segmented = image_util.read_image(
            mask_image_path, normalization=False, train_colors=train_colors)

        output = model.predict(np.expand_dims(sample_image, axis=0))[0]
        confusion = calc_segmentation_confusion(output, segmented, nb_classes)
        tp = np.diag(confusion)
        fp = np.sum(confusion, 0) - tp
        fn = np.sum(confusion, 1) - tp
        dice = calc_dice_from_confusion(tp, fp, fn)
        segmented = image_util.cast_to_onehot(segmented)
        result_image = get_imageset(
            sample_image, output, segmented, put_text=f'dice: {dice}')
        save_image_name = os.path.basename(input_image_path)
        result_image.save(f"{save_dir}/{save_image_name}")
