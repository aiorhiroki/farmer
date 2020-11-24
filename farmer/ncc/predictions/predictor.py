from abc import ABCMeta, abstractmethod
import colorsys
import numpy as np
import cv2


class Predictor(metaclass=ABCMeta):
    class_axis = 0

    def __init__(self, model):
        self.model = model
        self.height, self.width = self.model.input_shape[1:3]
        self.nb_class = self.model.output_shape[-1] - 1

    def predict_frame(self, frame: np.array) -> np.array:
        frame_height, frame_width = frame.shape[:2]
        in_frame = self.pre_process(frame)
        prediction = self.model.predict(in_frame)[0]
        res = self.post_process(prediction, frame_height, frame_width)
        return res

    def pre_process(self, frame: np.array) -> np.array:
        in_frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        in_frame = in_frame.astype("float32")/255.
        in_frame = np.expand_dims(in_frame, axis=0)
        return in_frame

    @classmethod
    @abstractmethod
    def post_process(
            cls,
            prediction: np.array,
            height: int = None,
            width: int = None) -> np.array:
        pass

    @abstractmethod
    def overlay(self, frame: np.array) -> np.array:
        pass


class Classifier(Predictor):
    class_axis = 0

    def __init__(self, class_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

    @classmethod
    def post_process(cls, prediction, height=None, width=None):
        return np.argmax(prediction, axis=cls.class_axis)

    def overlay(self, frame: np.array) -> np.array:
        prediction_class = self.predict_frame(frame)
        cv2.rectangle(frame, (0, 150), (250, 250), (255, 255, 255), -1)
        cv2.putText(
            frame, self.class_names[prediction_class], (15, 215),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        return frame


class Segmenter(Predictor):
    class_axis = 2

    def __init__(self, model):
        super().__init__(model)
        self.colors = self.generate_colors()
        self.min_area = 100

    @classmethod
    def post_process(cls, prediction, height: int, width: int):
        res = np.argmax(prediction, axis=cls.class_axis)
        res = cv2.resize(res, (width, height), interpolation=cv2.INTER_NEAREST)
        return res

    def overlay(self, frame: np.array, class_indices=None) -> np.array:
        alpha = 0.4
        res = self.predict_frame(frame)
        res[res != 1] = 0  # TODO: multi class indices
        res = self.delete_small_mask(res.astype("uint8"))
        masks = np.identity(self.nb_class + 1)[res]

        dst = frame.astype(np.uint8).copy()
        for i in range(self.nb_class):
            # color = self.colors[i]
            mask = masks[:, :, i + 1]

            """Apply the given mask to the image.
            """
            for c in range(3):
                dst[:, :, c] = np.where(
                    mask == 1,
                    dst[:, :, c] * (1 - alpha) + alpha * 255,
                    dst[:, :, c]
                )
        return dst

    def generate_colors(self, bright=True):
        """
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [
            (i / self.nb_class, 1, brightness) for i in range(self.nb_class)
        ]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    def delete_small_mask(self, mask: np.array) -> np.array:
        gray = mask.copy()
        _, bw = cv2.threshold(
            gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return mask
        for i in range(0, len(contours)):
            area = cv2.contourArea(contours[i])
            if area < self.min_area:
                cv2.drawContours(gray, contours, i, 0, -1)
        return gray
