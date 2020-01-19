import cv2
import numpy as np
from ..video import FPS


def classify_movie(model, video_file, class_names=None, input_size=(299, 299), start_frame=None):
    video = cv2.VideoCapture(video_file)
    if start_frame:
        video.set(1, start_frame)
    fps = FPS()

    while True:
        # Capture frame-by-frame
        ret, draw = video.read()

        # Our operations on the frame come here
        frame = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        in_frame = cv2.resize(frame, input_size) / 255.
        prediction = model.predict(np.expand_dims(in_frame, axis=0))[0]
        prediction_class = np.argmax(prediction)
        if class_names:
            prediction_class = class_names[prediction_class]

        cv2.rectangle(draw, (0, 30), (60, 50), (255, 255, 255), -1)
        cv2.putText(draw, str(prediction_class), (3, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        fps.calculate(draw)

        # Display the resulting frame
        cv2.imshow('frame', draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()
