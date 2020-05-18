from os.path import expanduser

import cv2
import os

from gfootball.common.colors import RED

class Writer(object):
    def __init__(self, filename, frame_rate=25.0):
        filename = expanduser(filename)
        assert filename.endswith('.mp4'), filename
        if cv2.os.path.exists(filename):
            assert not os.system('rm %s' % filename)
        # assert not cv2.os.path.exists(filename), filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, frame_rate, (1280, 720))

    def write(self, frame):
        assert frame.shape == (720, 1280, 3), frame.shape
        self.out.write(frame)

    def close(self):
        self.out.release()

def write_text_on_frame(frame, text, color=RED, bottom_left_corner_of_text=(30, 40), thickness=2, font_scale=1.0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, bottom_left_corner_of_text, font, font_scale, color, thickness)
