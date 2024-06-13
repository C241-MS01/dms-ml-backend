import uuid
import os
import imageio as iio


class Video:
    def __init__(self):
        path = "./tmp"
        if not os.path.exists(path):
            os.makedirs(path)

        self.filename = None
        self.fps = 10
        self.frames = {}

    def write_to_buffer(self, key, img):
        self.frames[key].append(img)

    def save_to_file(self, key) -> str:
        self.filename = f"./tmp/{key}-{uuid.uuid4()}.mp4"

        w = iio.get_writer(uri=self.filename, fps=self.fps, codec="libx264")

        for frame in self.frames[key]:
            w.append_data(frame)

        w.close()

        del self.frames[key]

        return self.filename
