import cv2
from timeit import default_timer as timer


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
            cv2.rectangle(draw, (0,0), (60,20), (255,255,255), -1)
            cv2.putText(draw, self.fps, (3,13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        else:
            print(self.fps)