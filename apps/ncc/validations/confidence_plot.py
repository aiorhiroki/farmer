import matplotlib.pyplot as plt
import numpy as np
import cv2


def confidence_plot(prediction, images, labels, class_index, max_row=5):
    if np.mean(images) < 1:
        images = images*255
    font = cv2.FONT_HERSHEY_PLAIN
    _, height, width, channel = images.shape
    if channel == 3:
        display = np.zeros((max_row*(height+20), 10*width, channel))
        display_one_raw = np.zeros((height+20, width, channel))
    elif channel == 1:
        display = np.zeros((max_row*(height+20), 10*width))
        display_one_raw = np.zeros((height+20, width))
        images = images[..., 0]
    prediction_cls = np.argmax(prediction, axis=1)
    class_index_range = (prediction_cls == class_index)
    inferred_image = images[class_index_range]
    inferred_probability = prediction[class_index_range][:, class_index]
    labels_class = labels[class_index_range]
    for i in range(10):
        prediction_bool = (
            0.9 - i*0.1 < inferred_probability) & (inferred_probability < 1.0 - i*0.1)
        confidence_images = inferred_image[prediction_bool]
        confidence_labels = labels_class[prediction_bool]
        for j, confidence_image in enumerate(confidence_images):
            if j == max_row:
                break
            display_one = display_one_raw.copy()
            display_one[-height:] = confidence_image
            if confidence_labels[j] != class_index:
                cv2.putText(display_one, str(
                    confidence_labels[j]), (10, 20), font, 1, (255, 255, 255))
            display[j*(height+20):(j+1)*(height+20), i *
                    width:(i+1)*width] = display_one
    plt.figure()
    plt.imshow(display, cmap='gray')
    plt.yticks([])
    plt.xticks(np.arange(10)*width, (10-np.arange(10))/10)
    plt.xlabel('confidence')
    plt.show()
