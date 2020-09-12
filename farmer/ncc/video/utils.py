import cv2
import numpy as np
from timeit import default_timer as timer
from tensorflow.keras.models import load_model


class FPS(object):
    """ Calculate FPS.
        example) fps = FPS()
                 while(cap.isOpended()):
                     # Your processing
                     fps.calculate(draw)
                     cv2.imshow('test', draw)
    """

    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()

    def calculate(self, draw, show=True):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time += exec_time
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        if show:
            cv2.rectangle(draw, (0, 0), (60, 20), (255, 255, 255), -1)
            cv2.putText(draw, self.fps, (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        else:
            print(self.fps)


def delete_small_mask(mask: np.array, threshold: int) -> np.array:
    gray = mask.copy()
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return mask

    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold:
            cv2.drawContours(gray, contours, i, 0, -1)
    return gray


class OutsideExcluder:
    """
    draw cross lines if frame is outside body
    out_ex = OutsideExcluder(MODEl_PATH)
    out, draw = out_ex.out(draw)
    """

    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.input_size = self.model.input_shape[1:3]  # height, width
        self.color = (255, 0, 0)

    def out(self, draw: np.ndarray) -> (bool, np.ndarray):
        height, width = draw.shape[:2]
        in_frame = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        in_frame = cv2.resize(in_frame, self.input_size) / 255.
        # outside recognition
        prediction = self.model.predict(np.expand_dims(in_frame, axis=0))[0]
        outside_class = np.argmax(prediction)
        if outside_class == 0:
            cv2.line(draw, (0, 0), (width, height), self.color, thickness=4)
            cv2.line(draw, (width, 0), (0, height), self.color, thickness=4)
            return True, draw
        return False, draw
