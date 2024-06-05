import base64
import cv2
import numpy as np
import uuid
import os


class Video:
    def __init__(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")

        self.video_writer = None
        self.video_filename = None
        self.video_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.video_fps = 15.0
        self.video_size = (640, 480)
        self.frames = list()

    def convert_base64_to_img(self, payload):
        nparr = np.frombuffer(base64.b64decode(payload), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def write_to_buffer(self, img):
        self.frames.append(img)

    def save_to_file(self) -> str:
        self.video_filename = f"./tmp/{uuid.uuid4()}.avi"

        self.video_writer = cv2.VideoWriter(
            self.video_filename,
            self.video_fourcc,
            self.video_fps,
            self.video_size,
        )

        for frame in self.frames:
            self.video_writer.write(frame)

        self.frames = list()

        return self.video_filename

    def release_video_writer(self):
        if self.video_writer is None:
            return

        self.video_writer.release()
        self.video_writer = None
        self.video_filename = None
