import numpy as np
import itertools
from tqdm import tqdm
from ..utils import ImageUtil
import matplotlib.pyplot as plt

import torch


def iou_dice_val(
        nb_classes, height, width, data_set, model, framework, train_colors=None):
    image_util = ImageUtil(nb_classes, (height, width))
    conf = np.zeros((nb_classes, nb_classes), dtype=np.int32)

    print('validation...')

    if framework == "pytorch": 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

    for image_file, seg_file in tqdm(data_set):
        # Get a training sample and make a prediction using current model
        sample = image_util.read_image(image_file, anti_alias=True)
        target = image_util.read_image(
            seg_file, normalization=False, train_colors=train_colors)

        if framework == "tensorflow":
            predicted = np.asarray(model.predict_on_batch(
                np.expand_dims(sample, axis=0)))[0]

        elif framework == "pytorch":        
            # Convert to channel fisrt
            sample_tensor = torch.tensor(sample).permute(2, 0, 1)
            sample_tensor = sample_tensor.unsqueeze(0)
            sample_tensor = sample_tensor.to(device, dtype=torch.float)
            # Predect on model
            output = model(sample_tensor)
            # delete temporary dimension and rearrange dimensions to fit Tensorflow logic
            output = torch.squeeze(output[0], 0)
            output = output.permute(1, 2, 0)  # Convert to channel last
            predicted = output.to("cpu").detach().numpy()

        # Convert predictions and target from categorical to integer format
        predicted = np.argmax(predicted, axis=-1).ravel()
        target = target.ravel()

        x = predicted + nb_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=nb_classes**2)
        assert bincount_2d.size == nb_classes**2
        conf += bincount_2d.reshape((nb_classes, nb_classes))

    # Compute the IoU and mean IoU from the confusion matrix
    true_positive = np.diag(conf)
    false_positive = np.sum(conf, 0) - true_positive
    false_negative = np.sum(conf, 1) - true_positive

    # Just in case we get a division by 0, set the value to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = true_positive / \
            (true_positive + false_positive + false_negative)
        dice = 2 * true_positive / \
            (2 * true_positive + false_positive + false_negative)

    iou[np.isnan(iou)] = 0
    dice[np.isnan(dice)] = 0

    return {'iou': iou, 'dice': dice}


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
