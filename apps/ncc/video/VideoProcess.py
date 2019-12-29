# coding: utf-8

import os
import sys
import cv2


class VideoProcess:

    # 動画保存の初期設定を行う
    def __init__(self, video_file):
        self.video_file = video_file
        self.video = cv2.VideoCapture(video_file)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.frame_width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    # 目的のフレーム番号から指定した秒数だけ抜き出して保存する
    def extract(self, save_dir, target_frame, duration_second):
        if not 0 <= target_frame < self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError('frame index is invalid')
        self.video.set(1, target_frame)
        video_writer = cv2.VideoWriter(
            save_dir + self.video_file.replace('.mp4', '').split('/')[-1] + '_' + str(target_frame) + '.mp4',
            self.fourcc,
            self.frame_rate,
            (self.frame_width, self.frame_height)
        )
        for _ in range(duration_second * self.frame_rate):
            is_capturing, frame = self.video.read()
            if is_capturing:
                video_writer.write(frame)
            else:
                print('the end of video')

    def to_frames(self, save_dir='./frames/', image_file='%s.jpg', start=0, end=None):
        """ Split video to frames and save as jpg files.
        """
        os.makedirs(save_dir, exist_ok=True)

        self.video.set(1, start)
        if not end:
            end = int(self.video.get(7))
        if start > end:
            raise ValueError('start(args) must be larger than end(args).')

        for _ in range(start, end):
            frame_id = int(self.video.get(1))
            flag, frame = self.video.read()
            if not flag:
                break
            cv2.imwrite(os.path.join(save_dir, image_file % str(frame_id).zfill(6)), frame)
            sys.stdout.write('\r%s%d%s%d' % ('Saving ... ', frame_id, ' / ', end))
            sys.stdout.flush()

        print('\nDone')

    @property
    def properties(self):
        print('----- Video Properties -----')
        print('Path: ', self.video_file)
        print('fps: ', self.frame_rate)
        print('frame_count: ', self.frame_count)
        print('frame_height: ', self.frame_height)
        print('frame_width: ', self.frame_width)

        return self.frame_rate, self.frame_count, self.frame_height, self.frame_width
