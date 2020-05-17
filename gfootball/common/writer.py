from os.path import expanduser

import cv2

class Writer(object):
    def __init__(self, filename, frame_rate=25.0):
        filename = expanduser(filename)
        assert filename.endswith('.mp4'), filename
        # assert not cv2.os.path.exists(filename), filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, frame_rate, (1280, 720))

    def write(self, frame):
        assert frame.shape == (720, 1280, 3), frame.shape
        self.out.write(frame)

    def close(self):
        self.out.release()
