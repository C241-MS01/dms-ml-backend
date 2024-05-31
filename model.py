import time
import base64
import cv2
import torch
import mediapipe as mp
import numpy as np


class Model:
    def __init__(self):
        # --- PyTorch model for object detection ---
        # load the model
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

        # --- Drawing and Create Face Mesh on Face ---
        # drawing on faces
        self.mpDraw = mp.solutions.drawing_utils

        # create face mesh
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()

        # convert nomalized to pixel coordinates
        self.denormalize_coordinates = self.mpDraw._normalized_to_pixel_coordinates

        # drawing specification
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # --- Landmark of eye ---
        # landmark points to left eye
        self.all_left_eye_idxs = list(self.mpFaceMesh.FACEMESH_LEFT_EYE)
        # flatten and remove duplicates
        self.all_left_eye_idxs = set(np.ravel(self.all_left_eye_idxs))

        # landmark points to right eye
        self.all_right_eye_idxs = list(self.mpFaceMesh.FACEMESH_RIGHT_EYE)
        self.all_right_eye_idxs = set(np.ravel(self.all_right_eye_idxs))

        # Combined for plotting Landmark points for both eye
        self.all_idxs = self.all_left_eye_idxs.union(self.all_right_eye_idxs)

        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
        self.chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
        self.all_chosen_idxs = self.chosen_left_eye_idxs + self.chosen_right_eye_idxs

        # --- info before start ---
        # image resize
        self.width = 800
        self.height = 450

        # color code
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        # threshold for detection
        self.ear_thresh = 0.13
        self.time_thresh = 3
        self.ear_below_thresh_time = 0
        self.start_time = 0

    def process_frame(self, string):
        # convert base64 to image
        img = self.convert_base64_to_image(string)
        # resize the image frame
        img = cv2.resize(img, (self.width, self.height))

        # Detect object
        detection_result = self.model(img)
        print(detection_result)

        # Convert the BGR to RGB image
        img.flags.writeable = False
        img_h, img_w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = self.calculate_avg_ear(
                landmarks,
                self.chosen_left_eye_idxs,
                self.chosen_right_eye_idxs,
                img_w,
                img_h,
            )

            print(EAR)
            for faceLms in results.multi_face_landmarks:
                for idx in self.all_chosen_idxs:
                    landmark = faceLms.landmark[idx]
                    x, y, z = landmark.x, landmark.y, landmark.z

                    if EAR > self.ear_thresh:
                        self.ear_below_thresh_time = 0
                        cv2.circle(
                            img,
                            (int(x * img.shape[1]), int(y * img.shape[0])),
                            2,
                            self.GREEN,
                            -1,
                        )

                        cv2.putText(
                            img,
                            text="Steady",
                            org=(15, 35),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )

                    else:
                        if self.ear_below_thresh_time == 0:
                            self.start_time = time.perf_counter()
                        self.ear_below_thresh_time = (
                            time.perf_counter() - self.start_time
                        )
                        cv2.circle(
                            img,
                            (int(x * img.shape[1]), int(y * img.shape[0])),
                            2,
                            self.RED,
                            -1,
                        )

                        cv2.putText(
                            img,
                            text="Drowsy",
                            org=(15, 35),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )

                        if self.ear_below_thresh_time >= self.time_thresh:
                            cv2.putText(
                                img,
                                text="Driver is sleeping!",
                                org=(15, 100),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1,
                                color=(0, 0, 255),
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                            print("Driver is sleeping!")

    def convert_base64_to_image(self, base64_string):
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    # --- Formula Eye Aspect Ratio (EAR) ---

    # calculate EAR
    def calculate_avg_ear(
        self, landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h
    ):
        left_ear, left_lm_coordinates = self.get_ear(
            landmarks, left_eye_idxs, image_w, image_h
        )
        right_ear, right_lm_coordinates = self.get_ear(
            landmarks, right_eye_idxs, image_w, image_h
        )
        Avg_EAR = (left_ear + right_ear) / 2.0

        return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

    # calculate EAR for one eye
    def get_ear(self, landmarks, refer_idxs, frame_width, frame_height):
        try:
            coords_points = []
            for i in refer_idxs:
                lm = landmarks[i]
                coord = self.denormalize_coordinates(
                    lm.x, lm.y, frame_width, frame_height
                )
                coords_points.append(coord)

            # eye landmark (x, y) coordinates
            P2_P6 = self.distance(coords_points[1], coords_points[5])
            P3_P5 = self.distance(coords_points[2], coords_points[4])
            P1_P4 = self.distance(coords_points[0], coords_points[3])

            # compute EAR
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        except:
            ear = 0.0
        coords_points = None

        return ear, coords_points

    # calculate l2-norm between two points
    def distance(self, point_1, point_2):
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist
