import os
from tqdm import tqdm
import cv2


def predict_on_video(
        predictor,
        video_path,
        save_dir="result_video",
        start=None,
        end=None):

    video = cv2.VideoCapture(video_path)
    save_video_name = os.path.basename(video_path).split(".")[0] + ".avi"
    save_path = f"{save_dir}/{save_video_name}"
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
    if start is not None:
        start_frame = _to_frame(start)
        video.set(1, start_frame)
    if end is None:
        end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        end_frame = _to_frame(end)
    for _ in tqdm(range(end_frame - start_frame)):
        ret, frame = video.read()
        if not ret:
            break
        in_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = predictor.overlay(in_frame)
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        writer.write(out_frame)

    video.release()
    writer.release()


def _to_frame(time):
    time_list = time.split(":")
    if len(time_list) == 3:
        frame = int(time_list[2])*60*60*30
        frame += int(time_list[1])*60*30
        frame += int(time_list[0])*30
    else:
        frame = int(time_list[0])
    return frame
