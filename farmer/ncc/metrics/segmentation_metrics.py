import numpy as np
from pathlib import Path
import itertools
from tqdm import tqdm
from ..utils import get_imageset
import matplotlib.pyplot as plt
import cv2
import json
from ..metrics.surface_dice import metrics as surface_distance

def calc_segmentation_metrics(confusion):
    tp = np.diag(confusion)
    fp = np.sum(confusion, 0) - tp
    fn = np.sum(confusion, 1) - tp
    tn = np.sum(confusion) - (fp + fn + tp)

    iou = calc_iou_from_confusion(tp, fp, fn)
    dice = calc_dice_from_confusion(tp, fp, fn)
    precision = calc_precision_from_confusion(tp, fp)
    recall = calc_recall_from_confusion(tp, fn)
    sepecificity = calc_sepecificity_from_confusion(tn, fp)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'specificity': sepecificity
    }


def iou_dice_val(
        nb_classes,
        dataset,
        model,
        batch_size
):
    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)

    print('\nvalidation...')
    for i, (image, mask) in enumerate(tqdm(dataset)):
        if i == 0:
            images = np.zeros((batch_size,) + image.shape, dtype=image.dtype)
            masks = np.zeros((batch_size,) + mask.shape, dtype=mask.dtype)

        image_index = i % batch_size

        images[image_index] = image
        masks[image_index] = mask

        if i == len(dataset) - 1 or image_index == batch_size - 1:
            output = model.predict(images)
            for j in range(image_index + 1):
                confusion += calc_segmentation_confusion(
                    output[j], masks[j], nb_classes)

            images[:] = 0
            masks[:] = 0

    return calc_segmentation_metrics(confusion)


def calc_segmentation_confusion(y_pred, y_true, nb_classes):
    # Convert predictions and target from categorical to integer format
    # y_pred: onehot, y_true: onehot
    y_pred = np.argmax(y_pred, axis=-1).ravel()
    y_true = np.argmax(y_true, axis=-1).ravel()
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


def calc_sepecificity_from_confusion(tn, fp):
    with np.errstate(divide='ignore', invalid='ignore'):
        sepecificity = tn / (fp + tn)

    sepecificity[np.isnan(sepecificity)] = 0
    return [float(s) for s in sepecificity]


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

def calc_surface_dice(pred_out, gt_label, nb_classes, vertical=1.0, horizontal=1.0, tolerance=0.0):
    """
    surface dice calculation

    Args:
        pred_out (np.array, shape (h,w,nb_classes)): prediction output. 
        gt_mask (np.array, shape (h,w)): ground truth mask. 
        nb_classes (int): the number of classes
        vertical (float, optional): real length (mm) of pixel in the vertical direction. Defaults to 1.0.
        horizontal (float, optional): real length (mm) of pixel in the horizontal direction. Defaults to 1.0.
        tolerance (float, optional): acceptable tolerance (mm) of boundary. Defaults to 0.0.

    Returns:
        surface_dice (float): 
    """
    class_surface_dice = list()
    
    # convert array (value: class_id)
    pred_label = np.uint8(np.argmax(pred_out, axis=2))
    gt_label = np.uint8(np.argmax(gt_label, axis=2))

    for class_id in range(nb_classes):
        gt_mask = gt_label == class_id
        pred_mask = pred_label == class_id
        
        # convert bool np.array mask
        gt_mask = np.asarray(gt_mask, dtype=np.bool)
        pred_mask = np.asarray(pred_mask, dtype=np.bool)
        
        # if both masks are empty, the result is NaN.
        if (np.sum(gt_mask==True) == 0) & (np.sum(pred_mask==True) == 0):
            surface_dice = 0.0
        else:        
            surface_distances = surface_distance.compute_surface_distances(
                gt_mask,
                pred_mask,
                spacing_mm=(vertical, horizontal))
            surface_dice = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=tolerance)
        
        class_surface_dice.append(surface_dice)
    
    return class_surface_dice

def generate_segmentation_result(
    nb_classes,
    dataset,
    model,
    save_dir,
    batch_size
):
    confusion_all = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    image_dice_list = list()
    dice_list = list()
    surface_dice_list = list()
    
    print('\nsave predicted image...')
    for i, (image, mask) in enumerate(tqdm(dataset)):
        if i == 0:
            images = np.zeros((batch_size,) + image.shape, dtype=image.dtype)
            masks = np.zeros((batch_size,) + mask.shape, dtype=mask.dtype)

        batch_index = i // batch_size
        image_index = i % batch_size

        images[image_index] = image
        masks[image_index] = mask

        if i == len(dataset) - 1 or image_index == batch_size - 1:
            output = model.predict(images)
            for j in range(image_index + 1):
                confusion = calc_segmentation_confusion(
                    output[j], masks[j], nb_classes)
                metrics = calc_segmentation_metrics(confusion)
                dice = metrics['dice']
                surface_dice = calc_surface_dice(output[j], masks[j], nb_classes)
                
                result_image = get_imageset(
                    images[j], output[j], masks[j],
                    put_text=f'dice: {np.round(dice, 3)}    surface dice: {np.round(surface_dice, 3)}')
                data_index = batch_index * batch_size + j
                *input_file, _ = dataset.annotations[data_index]
                image_path = Path(input_file[0])
                save_image_dir = Path(save_dir) / image_path.parent.name
                save_image_dir.mkdir(exist_ok=True)
                save_image_path = str(save_image_dir / image_path.name)
                image_dice_list.append([save_image_path, dice])
                dice_list.append(dice)
                surface_dice_list.append([save_image_path, surface_dice])
                
                result_image_out = result_image[:, :, ::-1]   # RGB => BGR
                cv2.imwrite(save_image_path, result_image_out)

                confusion_all += confusion

            images[:] = 0
            masks[:] = 0
    
    with open(f"{save_dir}/dice.json", "w") as fw:
        json.dump(image_dice_list, fw, ensure_ascii=True, indent=4)

    with open(f"{save_dir}/surface_dice.json", "w") as fw:
        json.dump(surface_dice_list, fw, ensure_ascii=True, indent=4)

    dice_class_axis = np.array(dice_list).T.tolist()
    for i in range(len(dice_class_axis)):
        plt.figure()
        plt.hist(dice_class_axis[i])
        plt.savefig(f"{save_dir}/dice_hist_class_{i}.png")

    metrics = calc_segmentation_metrics(confusion_all)
    # append surface_dice to metrics
    mean_surface_dice = np.mean(list(map(lambda x: x[1], surface_dice_list)), axis=0)
    metrics['surface_dice'] = [float(x) for x in mean_surface_dice]

    return metrics
