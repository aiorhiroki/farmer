import numpy as np
import cv2


def confusion_visualize(images, labels, model, class_names):
    _, height, width, _ = images.shape
    frame_percent = 2  # outframe
    frame = int((height + width) / 2 * (frame_percent/100)) + 1
    nb_classes = len(class_names)
    result = np.zeros((height*nb_classes, width*nb_classes, 3))
    predict_prob = model.predict(images)
    if np.max(images) <= 1:
        images *= 255
    predict_cls = np.argmax(predict_prob, axis=1)
    for true_index in range(nb_classes):
        for predict_index in range(nb_classes):
            index_range = np.where((labels == true_index)
                                   & (predict_cls == predict_index))
            prob = predict_prob[index_range]
            one_picture = np.zeros((height, width, 3))
            if true_index == predict_index:
                one_picture[:, :, 1] = 255
            else:
                one_picture[:, :, 2] = 255
            if len(prob) == 0:
                one_picture[frame:-frame, frame:-
                            frame] = np.zeros((height-2*frame, width-2*frame, 3))
            else:
                sort_range = np.argsort(prob[:, predict_index])[::-1]
                one_picture[frame:-frame, frame:-frame] = images[index_range[0]
                                                                 [sort_range][0]][frame:-frame, frame:-frame]
            result[height*true_index:height *
                   (true_index+1), width*predict_index:width*(predict_index+1)] = one_picture
    cv2.imwrite('confusion_visualize.png', result)
